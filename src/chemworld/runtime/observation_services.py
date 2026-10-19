"""Observation and scoring services for the transactional runtime."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from chemworld.foundation import Observation, PhysicalConstitution, WorldState, equipment_settings
from chemworld.foundation.state import ProcessLedger, selected_phase_id
from chemworld.physchem.configurable_spectra import SPECTRAL_MODEL_ID, acquire_spectrum
from chemworld.physchem.equilibrium_chemistry import (
    SolubilityProductSpec,
    apply_precipitation_hooks,
    solve_monoprotic_acid_base,
)
from chemworld.physchem.equilibrium_mechanism import (
    EquilibriumMechanismFamily,
    solve_coupled_weak_acid_precipitation,
)
from chemworld.physchem.spectroscopy_adapter_manifest import (
    ValidatedInstrumentRuntimeProvider,
)
from chemworld.runtime.full_process_contract import unrecoverable_crystallization_result
from chemworld.runtime.mechanisms import CompiledMechanism
from chemworld.runtime.phase_ledger_services import ChemWorldPhaseLedgerServices
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.runtime.spectral_settings import spectral_settings
from chemworld.world.observation_contracts import TaskObservationContract
from chemworld.world.observation_kernel import (
    base_observed_mask,
    base_public_values,
    observation_units,
    processed_estimate,
)
from chemworld.world.operations import instrument_name, operation_name
from chemworld.world.scoring import (
    PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
    TaskScoringContract,
    task_score_observation,
)
from chemworld.world.separation_kernel import downstream_truth_values
from chemworld.world.species_roles import PHASE_PRODUCT_AMOUNT_KEY
from chemworld.world.spectral_contract import (
    SPECTRAL_INSTRUMENTS,
    acquisition_cost,
    acquisition_seconds,
)


class ChemWorldObservationKernel:
    """Generate partial, noisy public observations from hidden state."""

    def __init__(
        self,
        constitution: PhysicalConstitution,
        objective: str,
        compiled_mechanism: CompiledMechanism,
        scoring_contract: TaskScoringContract | None = None,
        observation_contract: TaskObservationContract | None = None,
        *,
        observation_noise_multiplier: float = 1.0,
        instrument_provider: ValidatedInstrumentRuntimeProvider | None = None,
    ) -> None:
        self.constitution = constitution
        self.objective = objective
        self.compiled_mechanism = compiled_mechanism
        self.scoring_contract = scoring_contract or TaskScoringContract.from_success_metrics(
            objective=objective,
        )
        self.observation_contract = observation_contract
        if not np.isfinite(observation_noise_multiplier) or observation_noise_multiplier <= 0.0:
            raise ValueError("observation_noise_multiplier must be positive and finite")
        self.observation_noise_multiplier = float(observation_noise_multiplier)
        self.species_view = MechanismSpeciesView(compiled_mechanism)
        self.phase_ledgers = ChemWorldPhaseLedgerServices(self.species_view)
        self.instrument_provider = instrument_provider or ValidatedInstrumentRuntimeProvider()
        self.last_provider_execution: dict[str, Any] = {}

    def observe(
        self,
        state: WorldState,
        action: dict[str, Any],
        rng: np.random.Generator,
    ) -> Observation:
        operation = operation_name(action["operation"])
        if operation != "measure":
            process = state.process or ProcessLedger()
            last = dict(process.last_observation)
            last_mask = dict(process.last_observed_mask)
            values = self._base_public_values(state)
            observed_mask = self._base_observed_mask()
            values.update(last)
            observed_mask.update({str(key): bool(value) for key, value in last_mask.items()})
            values["cost"] = min(1.0, state.ledger.cost)
            values["safety_risk"] = state.ledger.risk
            observed_mask["cost"] = True
            observed_mask["safety_risk"] = True
            values["score"] = self._score(values)
            observed_mask["score"] = True
            return Observation(
                values=values,
                units=self._observation_units(),
                observed_mask=observed_mask,
                processed_estimate=self._processed_estimate(values, observed_mask),
            )

        instrument_id = instrument_name(action.get("instrument", "hplc"))
        instrument = self.constitution.instruments[instrument_id]
        if instrument_id in SPECTRAL_INSTRUMENTS:
            return self._spectral_observation(state, instrument_id, rng)
        truth_values = self._truth_values(state)
        noisy = self._base_public_values(state)
        observed_mask = self._base_observed_mask()
        observable_keys = (
            instrument.observable_keys
            if self.observation_contract is None
            else self.observation_contract.observable_keys_for_instrument(
                instrument_id,
                instrument.observable_keys,
            )
        )
        selected_keys = set(observable_keys)
        empty_crystals = unrecoverable_crystallization_result(
            state, tolerance=self.constitution.tolerance
        )
        undefined_crystal_keys = (
            {"crystal_size", "crystal_purity", "crystal_csd_quality", "crystal_fines_fraction"}
            if empty_crystals
            else set()
        )
        for key in instrument.observable_keys:
            if key in undefined_crystal_keys:
                continue
            std = instrument.noise_std.get(key, 0.0) * self.observation_noise_multiplier
            value = float(np.clip(truth_values[key] + rng.normal(0.0, std), 0.0, 1.0))
            if key in selected_keys:
                noisy[key] = value
                observed_mask[key] = True

        if observed_mask["byproduct_signal"] and observed_mask["degradation_warning"]:
            byproduct_signal = self._observed_value(noisy, "byproduct_signal")
            degradation_warning = self._observed_value(noisy, "degradation_warning")
            noisy["virtual_spectrum_summary"] = float(
                np.clip(
                    0.55 * byproduct_signal + 0.45 * degradation_warning,
                    0.0,
                    1.0,
                )
            )
            observed_mask["virtual_spectrum_summary"] = True
        noisy["cost"] = min(1.0, state.ledger.cost)
        noisy["safety_risk"] = state.ledger.risk
        observed_mask["cost"] = True
        observed_mask["safety_risk"] = True
        noisy["score"] = self._score(noisy)
        observed_mask["score"] = True
        public_species_amounts = self._public_species_amounts(
            state,
            observable_keys=observable_keys,
        )
        raw_signal = self._raw_signal(
            instrument_id, noisy, state, rng, species_amounts_mol=public_species_amounts
        )
        raw_signal["acquisition"] = {
            "clock_semantics": "instantaneous_snapshot",
            "sample_time_s": state.ledger.time_s,
            "result_time_s": state.ledger.time_s,
        }
        if empty_crystals and instrument_id in {"particle_size", "final_assay"}:
            raw_signal["sample_outcome"] = {
                "status": "negative_result",
                "reason": "below_crystal_recovery_threshold",
                "recovery_threshold_mol": self.constitution.tolerance,
                "undefined_observables": sorted(undefined_crystal_keys & selected_keys),
            }
        return Observation(
            values=noisy,
            units=self._observation_units(),
            observed_mask=observed_mask,
            raw_signal=raw_signal,
            processed_estimate=self._processed_estimate(noisy, observed_mask),
            uncertainty={
                f"{key}_std": float(std) * self.observation_noise_multiplier
                for key, std in instrument.noise_std.items()
                if observed_mask.get(key, False)
            },
            instrument_id=instrument_id,
            cost=instrument.cost,
            sample_consumed_L=instrument.sample_volume_L,
        )

    def _spectral_observation(
        self, state: WorldState, instrument_id: str, rng: np.random.Generator
    ) -> Observation:
        configuration = spectral_settings(state, instrument_id)
        packet = acquire_spectrum(
            instrument_id,
            (
                self.species_view.reactant_amount(state) / state.volume_L,
                self.species_view.target_amount(state) / state.volume_L,
                self.species_view.impurity_amount(state) / state.volume_L,
            ),
            configuration,
            rng,
            noise_multiplier=self.observation_noise_multiplier,
        )
        seconds = acquisition_seconds(instrument_id, configuration)
        packet["acquisition"] = {
            "clock_semantics": "instantaneous_snapshot",
            "sample_time_s": state.ledger.time_s,
            "result_time_s": state.ledger.time_s,
            "analysis_effort_s": seconds,
            "effort_ledger": "analysis_time_s",
            "reactor_advance_s": 0.0,
        }
        values, mask = self._base_public_values(state), self._base_observed_mask()
        fit = packet["processed_estimates"]["concentration_mol_L"]
        a, p, b = (
            fit[key] for key in ("reactant_reporter", "target_reporter", "impurity_reporter")
        )
        instrument = self.constitution.instruments[instrument_id]
        allowed = (
            instrument.observable_keys
            if self.observation_contract is None
            else self.observation_contract.observable_keys_for_instrument(
                instrument_id, instrument.observable_keys
            )
        )
        estimates = {
            "selectivity": p / (p + b) if p is not None and b is not None and p + b > 0 else None,
            "purity": p / (a + p + b)
            if all(v is not None for v in (a, p, b)) and a + p + b > 0
            else None,
            "byproduct_signal": b / (a + p + b)
            if all(v is not None for v in (a, p, b)) and a + p + b > 0
            else None,
        }
        for key in allowed:
            values[key] = estimates[key]
            mask[key] = estimates[key] is not None
        values["score"] = self._score(values)
        self.last_provider_execution = {
            "model_id": SPECTRAL_MODEL_ID,
            "maturity": "development",
            "role": "runtime",
            "success": True,
            "failure_reason": None,
            "provenance": ["finite anonymous reporter calibration; acquired-trace NNLS"],
        }
        return Observation(
            values=values,
            units=self._observation_units(),
            observed_mask=mask,
            raw_signal=packet,
            processed_estimate=self._processed_estimate(values, mask),
            uncertainty={},
            instrument_id=instrument_id,
            cost=acquisition_cost(instrument_id, configuration),
            sample_consumed_L=instrument.sample_volume_L,
        )

    def failed_observation(self) -> Observation:
        """Return a non-informative observation for failed action preconditions."""

        units = self._observation_units()
        return Observation(
            values=dict.fromkeys(units, None),
            units=units,
            observed_mask=dict.fromkeys(units, False),
            raw_signal={},
            processed_estimate={},
            uncertainty={},
            instrument_id=None,
            cost=0.0,
            sample_consumed_L=0.0,
        )

    @staticmethod
    def _processed_estimate(
        values: dict[str, float | None],
        observed_mask: dict[str, bool],
    ) -> dict[str, float | None]:
        return processed_estimate(values, observed_mask)

    def _raw_signal(
        self,
        instrument_id: str,
        values: dict[str, float | None],
        state: WorldState,
        rng: np.random.Generator,
        *,
        species_amounts_mol: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if instrument_id == "particle_size":
            self.last_provider_execution = {
                "model_id": "synthetic-particle-summary-v1",
                "maturity": "development",
                "role": "runtime",
                "success": True,
                "failure_reason": None,
                "provenance": ["population-balance scalar summary with Gaussian noise"],
            }
            return {
                "kind": "particle_size_signal",
                "d50_um": (
                    None
                    if values["crystal_size"] is None
                    else 250.0 * float(values["crystal_size"])
                ),
                "fines_number_fraction": values["crystal_fines_fraction"],
                "csd_quality": values["crystal_csd_quality"],
                "fines_threshold_um": 20.0,
                "weighting": "number",
                "metadata": {"synthetic": True, "raw_images_available": False},
            }
        replicate_count = 3 if instrument_id == "final_assay" else 2
        provider_seed = int(rng.integers(0, 2**31 - 1))
        provider_result = self.instrument_provider.evaluate(
            {
                "instrument_id": instrument_id,
                "public_values": values,
                "public_species_amounts_mol": species_amounts_mol,
                "sample_basis_volume_L": state.volume_L,
                "seed": provider_seed,
                "replicate_count": replicate_count,
            }
        )
        contract = self.instrument_provider.model_contract
        self.last_provider_execution = {
            "model_id": contract.model_id,
            "provider_path": contract.provider_path,
            "maturity": contract.maturity.value,
            "role": contract.role.value,
            "success": provider_result.success,
            "failure_reason": provider_result.failure_reason,
            "diagnostics": dict(provider_result.diagnostics),
            "provenance": list(provider_result.provenance),
        }
        if not provider_result.success:
            raise ValueError(provider_result.failure_reason or "instrument runtime provider failed")
        packet = provider_result.outputs.get("packet")
        if not isinstance(packet, dict):
            raise ValueError("instrument runtime provider returned no public packet")
        if species_amounts_mol is not None:
            packet["source"] = "task_public_species_calibration"
            packet["visibility"] = "task_observation_contract"
        return packet

    @staticmethod
    def _observation_units() -> dict[str, str]:
        return observation_units()

    @staticmethod
    def _base_public_values(state: WorldState) -> dict[str, float | None]:
        return base_public_values(cost=state.ledger.cost, safety_risk=state.ledger.risk)

    @staticmethod
    def _base_observed_mask() -> dict[str, bool]:
        return base_observed_mask()

    @staticmethod
    def _observed_value(values: dict[str, float | None], key: str) -> float:
        value = values.get(key)
        return 0.0 if value is None else float(value)

    def _score(self, values: dict[str, float | None]) -> float:
        return task_score_observation(
            contract=self.scoring_contract,
            values=values,
        )

    def _truth_values(self, state: WorldState) -> dict[str, float]:
        truth = self.species_view.truth_values(state)
        downstream = (
            self._partition_v3_truth_values(state)
            if self.scoring_contract.contract_id == PARTITION_S0_EXTRACTION_EFFICIENCY_V3
            else downstream_truth_values(
                state,
                product_amount_mol=self.species_view.target_amount(state),
                impurity_amount_mol=self.species_view.impurity_amount(state),
                initial_product_mol=max(self.species_view.initial_reactant_amount(state), 1.0e-12),
                target_species=self.species_view.target_species_for_state(state),
                impurity_species=self.species_view.impurity_species_for_state(state),
            )
        )
        truth.update(downstream)
        from chemworld.runtime.full_process_contract import (
            active,
            population_active,
            seed_provenance_active,
        )

        if active(state):
            phases = {} if state.phases is None else state.phases.phases
            denominator = max(self.species_view.initial_reactant_amount(state), 1.0e-12)
            if state.metadata.get("full_process_task_id") == "reaction-to-distillation":
                # Same current receiver and original charge as the native purity/recovery assay.
                truth["distillate_purity"] = downstream["purity"]
                truth["distillate_recovery"] = downstream["recovery"]
            if "solid" in phases:
                solid = phases["solid"].species_amounts_mol
                product = sum(solid.get(k, 0.0) for k in self.species_view.target_species)
                impurity = sum(solid.get(k, 0.0) for k in self.species_view.impurity_species)
                metrics = {} if state.process is None else state.process.metrics
                seed = metrics.get(
                    "retained_seed_mol",
                    equipment_settings(state.equipment, "crystallizer").get("seed_target_mol", 0.0),
                )
                if seed_provenance_active(state):
                    # A filtered slurry can be reheated, seeded or cooled again.
                    # Its current solid provenance supersedes the last filter receipt.
                    seed = equipment_settings(state.equipment, "crystallizer").get(
                        "seed_target_mol", 0.0
                    )
                truth["crystal_purity"] = product / max(product + impurity, 1.0e-12)
                truth["crystal_yield"] = float(
                    np.clip((product - min(seed, product)) / denominator, 0.0, 1.0)
                )
                if population_active(state):
                    settings = equipment_settings(state.equipment, "crystallizer")
                    cv = min(max(float(settings.get("csd_cv", 0)), 0), 1)
                    fines = float(settings.get("csd_fines_number_fraction", 0))
                    truth["crystal_size"] = min(float(settings.get("csd_d50_m", 0)) / 250e-6, 1)
                    truth["crystal_fines_fraction"] = fines
                    truth["crystal_csd_quality"] = 0.55 * (1 - cv) + 0.45 * (1 - fines)
        equilibrium = self._equilibrium_truth_values(state)
        # Once the electrochemical runtime has solved its coupled aqueous
        # state, instrument-facing equilibrium signals must come from that
        # solved state.  The generic equilibrium-characterization proxy remains
        # the fallback before electrolysis and for non-electrochemical tasks.
        equilibrium.update(self._electrochemical_equilibrium_truth_values(state))
        truth.update(equilibrium)
        return truth

    def _partition_v3_truth_values(self, state: WorldState) -> dict[str, float]:
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        product_amount = sum(
            float(entry.get(PHASE_PRODUCT_AMOUNT_KEY, 0.0)) for entry in phase_ledger.values()
        )
        impurity_amount = sum(
            float(entry.get("impurity_mol", 0.0)) for entry in phase_ledger.values()
        )
        initial_product_amount = 0.0
        if state.species is not None:
            initial_product_amount = sum(
                max(
                    float(state.species.initial_amounts_mol.get(species_id, 0.0)),
                    0.0,
                )
                for species_id in self.phase_ledgers.product_candidate_species()
            )
        if initial_product_amount <= 0.0:
            initial_product_amount = self.species_view.initial_reactant_amount(state)
        return downstream_truth_values(
            state,
            phase_ledger,
            product_amount_mol=product_amount,
            impurity_amount_mol=impurity_amount,
            initial_product_mol=max(initial_product_amount, 1.0e-12),
        )

    @staticmethod
    def _electrochemical_equilibrium_truth_values(state: WorldState) -> dict[str, float]:
        process_metrics = {} if state.process is None else state.process.metrics
        pH = process_metrics.get("electrolyte_pH")
        if not isinstance(pH, (int, float)) or not np.isfinite(float(pH)):
            return {}

        cell = equipment_settings(state.equipment, "electrochemical_cell")
        precipitating_salt = max(float(cell.get("precipitating_salt_mol", 0.0)), 0.0)
        precipitated = max(
            float(process_metrics.get("electrolyte_precipitated_mol", 0.0)),
            0.0,
        )
        inventory_scale = max(
            float(cell.get("electrolyte_acid_total_mol", 0.0))
            + float(cell.get("supporting_electrolyte_mol", 0.0))
            + precipitating_salt,
            1.0e-12,
        )
        residual = max(
            abs(float(process_metrics.get("electrolyte_charge_balance_error_eq", 0.0))),
            abs(float(process_metrics.get("electrolyte_material_balance_error_mol", 0.0)))
            / inventory_scale,
        )
        diagnostic = cell.get("aqueous_equilibrium_diagnostic", {})
        converged = isinstance(diagnostic, dict) and diagnostic.get("converged") is True
        return {
            "pH_normalized": float(np.clip(float(pH) / 14.0, 0.0, 1.0)),
            "acid_dissociation_fraction": float(
                np.clip(
                    float(
                        process_metrics.get(
                            "electrolyte_acid_dissociation_fraction",
                            0.0,
                        )
                    ),
                    0.0,
                    1.0,
                )
            ),
            "precipitation_signal": float(
                np.clip(
                    precipitated / max(precipitating_salt, 1.0e-12),
                    0.0,
                    1.0,
                )
            ),
            "equilibrium_residual": float(np.clip(residual, 0.0, 1.0)),
            "equilibrium_confidence": float(
                np.clip((1.0 - residual) if converged else 0.0, 0.0, 1.0)
            ),
        }

    def _equilibrium_truth_values(self, state: WorldState) -> dict[str, float]:
        """Return public equilibrium-characterization signals in [0, 1].

        The underlying weak-acid and precipitation calculations use D4
        equilibrium-chemistry kernels, but only normalized instrument-facing
        estimates are returned here. Hidden equilibrium constants and mechanism
        species ids remain outside the public observation.
        """

        volume_L = max(float(state.volume_L), 1.0e-9)
        acid_total = max(self.species_view.initial_reactant_amount(state), 0.0)
        if acid_total <= 0.0:
            return {
                "pH_normalized": 7.0 / 14.0,
                "acid_dissociation_fraction": 0.0,
                "precipitation_signal": 0.0,
                "equilibrium_residual": 1.0,
                "equilibrium_confidence": 0.0,
            }

        if state.metadata.get("equilibrium_entity_panel_version"):
            from chemworld.world.mixtures import working_solvents

            configured_profiles = state.metadata.get("equilibrium_entity_profiles")
            if not isinstance(configured_profiles, dict) or any(
                str(i) not in configured_profiles for i in range(4)
            ):
                raise ValueError("EQ-E active medium has no registered private profile")
            medium = working_solvents(state)
            profile = {
                key: (
                    medium.log_property
                    if key == "activity_coefficient_ratio"
                    else medium.linear_property
                )(tuple(configured_profiles[str(i)][key] for i in range(4)))
                for key in (
                    "pka_shift",
                    "log10_ksp",
                    "cation_fraction",
                    "activity_coefficient_ratio",
                )
            }
            coupled = solve_coupled_weak_acid_precipitation(
                acid_total_mol=acid_total,
                volume_L=volume_L,
                pka=float(state.metadata["equilibrium_entity_base_pka"])
                + float(profile["pka_shift"]),
                log10_ksp=float(profile["log10_ksp"]),
                mechanism_family="direct_free_ion_precipitation",
                cation_fraction=float(profile["cation_fraction"]),
                activity_coefficient_ratio=float(profile["activity_coefficient_ratio"]),
            )
            concentration = acid_total / volume_L
            identity_signal = max(
                coupled.acid_dissociation_fraction,
                coupled.precipitation_signal,
            )
            concentration_quality = 1.0 - min(
                abs(np.log10(max(concentration, 1.0e-12)) - np.log10(0.08)) / 4.0,
                1.0,
            )
            confidence = float(
                np.clip(
                    0.55 * (1.0 - min(coupled.equilibrium_residual, 1.0))
                    + 0.20 * min(identity_signal / 0.08, 1.0)
                    + 0.25 * concentration_quality,
                    0.0,
                    1.0,
                )
            )
            return {
                "pH_normalized": float(np.clip(coupled.pH / 14.0, 0.0, 1.0)),
                "acid_dissociation_fraction": float(
                    np.clip(coupled.acid_dissociation_fraction, 0.0, 1.0)
                ),
                "precipitation_signal": float(np.clip(coupled.precipitation_signal, 0.0, 1.0)),
                "equilibrium_residual": float(np.clip(coupled.equilibrium_residual, 0.0, 1.0)),
                "equilibrium_confidence": confidence,
            }

        if state.metadata.get("equilibrium_mechanism_benchmark_version") == "eq-s-v0.2":
            pka = self._hidden_acid_pka(state)
            mechanism_family = str(
                state.metadata.get(
                    "equilibrium_mechanism_family",
                    "direct_free_ion_precipitation",
                )
            )
            configured_beta = state.metadata.get("equilibrium_aqueous_association_beta_L_per_mol")
            coupled = solve_coupled_weak_acid_precipitation(
                acid_total_mol=acid_total,
                volume_L=volume_L,
                pka=pka,
                log10_ksp=float(state.metadata.get("hidden_equilibrium_log10_ksp", -5.2)),
                mechanism_family=cast(EquilibriumMechanismFamily, mechanism_family),
                cation_fraction=float(
                    state.metadata.get(
                        "equilibrium_precipitating_cation_fraction",
                        0.15,
                    )
                ),
                association_beta_L_per_mol=(
                    None if configured_beta is None else float(configured_beta)
                ),
                activity_coefficient_ratio=float(
                    state.metadata.get("equilibrium_activity_coefficient_ratio", 1.0)
                ),
            )
            concentration = acid_total / volume_L
            identifiability = 1.0 - min(
                abs(coupled.acid_dissociation_fraction - 0.08) / 0.08,
                1.0,
            )
            concentration_quality = 1.0 - min(
                abs(np.log10(max(concentration, 1.0e-12)) - np.log10(0.125)) / 4.0,
                1.0,
            )
            confidence = float(
                np.clip(
                    0.45 * (1.0 - min(coupled.equilibrium_residual, 1.0))
                    + 0.30 * identifiability
                    + 0.25 * concentration_quality,
                    0.0,
                    1.0,
                )
            )
            return {
                "pH_normalized": float(np.clip(coupled.pH / 14.0, 0.0, 1.0)),
                "acid_dissociation_fraction": float(
                    np.clip(coupled.acid_dissociation_fraction, 0.0, 1.0)
                ),
                "precipitation_signal": float(np.clip(coupled.precipitation_signal, 0.0, 1.0)),
                "equilibrium_residual": float(np.clip(coupled.equilibrium_residual, 0.0, 1.0)),
                "equilibrium_confidence": confidence,
            }

        pka = self._hidden_acid_pka(state)
        strong_cation = min(0.020, 0.15 * acid_total)
        strong_anion = min(0.020, 0.08 * acid_total)
        acid = solve_monoprotic_acid_base(
            acid_total_mol=acid_total,
            volume_L=volume_L,
            pka=pka,
            temperature_K=state.temperature_K,
            strong_cation_mol=strong_cation,
            strong_anion_mol=strong_anion,
            activity_coefficient_ratio=float(
                state.metadata.get("equilibrium_activity_coefficient_ratio", 1.0)
            ),
        )
        precipitation = apply_precipitation_hooks(
            {
                "Ag+": strong_cation + 0.10 * acid.species_amounts_mol.get("A-", 0.0),
                "Cl-": strong_anion + 0.08 * acid.species_amounts_mol.get("HA", 0.0),
            },
            (
                SolubilityProductSpec(
                    precipitate_id="AgCl_public",
                    cation_id="Ag+",
                    anion_id="Cl-",
                    ksp=float(state.metadata.get("hidden_equilibrium_ksp", 1.8e-10)),
                ),
            ),
            volume_L=volume_L,
        )
        precipitation_signal = float(
            np.clip(precipitation.total_precipitated_mol / max(acid_total, 1.0e-12), 0.0, 1.0)
        )
        residual = float(
            np.clip(
                abs(acid.charge_balance_error_eq) / max(acid_total / volume_L, 1.0e-12),
                0.0,
                1.0,
            )
        )
        identifiability = 1.0 - min(
            abs(acid.acid_dissociation_fraction - 0.5) / 0.5,
            1.0,
        )
        concentration = acid_total / volume_L
        concentration_quality = 1.0 - min(abs(concentration - 0.25) / 0.25, 1.0)
        confidence = float(
            np.clip(
                0.25 * (1.0 - residual)
                + 0.40 * identifiability
                + 0.20 * concentration_quality
                + 0.15 * (1.0 - min(abs(acid.pH - pka) / 2.0, 1.0)),
                0.0,
                1.0,
            )
        )
        return {
            "pH_normalized": float(np.clip(acid.pH / 14.0, 0.0, 1.0)),
            "acid_dissociation_fraction": float(np.clip(acid.acid_dissociation_fraction, 0.0, 1.0)),
            "precipitation_signal": precipitation_signal,
            "equilibrium_residual": residual,
            "equilibrium_confidence": confidence,
        }

    @staticmethod
    def _hidden_acid_pka(state: WorldState) -> float:
        configured = state.metadata.get("hidden_equilibrium_pka")
        if configured is not None:
            return float(configured)
        scenario_id = str(state.metadata.get("scenario_id", "equilibrium-characterization"))
        jitter = (sum(ord(char) for char in scenario_id) % 13) / 100.0
        temperature_shift = np.clip((state.temperature_K - 298.15) / 600.0, -0.25, 0.25)
        return float(4.55 + jitter - temperature_shift)

    def _public_species_amounts(
        self,
        state: WorldState,
        *,
        observable_keys: tuple[str, ...],
    ) -> dict[str, float] | None:
        if self.observation_contract is None:
            return state.species_amounts
        keys = set(observable_keys)
        downstream_public_amounts = self._selected_phase_public_species_amounts(
            state,
            observable_keys=keys,
        )
        if downstream_public_amounts is not None:
            return downstream_public_amounts
        public_amounts: dict[str, float] = {}
        reaction_keys = {
            "yield",
            "selectivity",
            "conversion",
            "byproduct_signal",
            "degradation_warning",
        }
        target_keys = {
            "yield",
            "selectivity",
            "purity",
            "recovery",
            "phase_ratio",
            "product_in_organic",
            "product_in_aqueous",
            "crystal_yield",
            "crystal_purity",
            "distillate_purity",
            "distillate_recovery",
            "flow_conversion",
            "selective_product_yield",
            "electrochemical_conversion",
            "electrochemical_selectivity",
        }
        impurity_keys = {
            "selectivity",
            "byproduct_signal",
            "purity",
            "impurity_signal",
            "process_mass_balance_error",
            "crystal_purity",
            "distillate_purity",
        }
        if keys.intersection(reaction_keys):
            public_amounts["reactant_public"] = self.species_view.reactant_amount(state)
        if keys.intersection(target_keys):
            public_amounts["target_public"] = self.species_view.target_amount(state)
        if keys.intersection(impurity_keys):
            public_amounts["impurity_public"] = max(
                self.species_view.byproduct_amount(state),
                self.species_view.impurity_amount(state)
                - self.species_view.degradation_amount(state),
                0.0,
            )
        if "degradation_warning" in keys:
            public_amounts["degradation_public"] = self.species_view.degradation_amount(state)
        return {
            species_id: amount for species_id, amount in public_amounts.items() if amount > 0.0
        } or None

    def _selected_phase_public_species_amounts(
        self,
        state: WorldState,
        *,
        observable_keys: set[str],
    ) -> dict[str, float] | None:
        downstream_keys = {
            "purity",
            "recovery",
            "phase_ratio",
            "product_in_organic",
            "product_in_aqueous",
            "impurity_signal",
            "process_mass_balance_error",
            "crystal_yield",
            "crystal_purity",
            "distillate_purity",
            "distillate_recovery",
        }
        if not observable_keys.intersection(downstream_keys):
            return None
        if state.phases is None:
            return None
        selected_id = selected_phase_id(state.phases)
        if selected_id is None:
            return None
        selected = state.phases.phases.get(selected_id)
        if selected is None:
            return None

        phase_amounts = selected.species_amounts_mol
        target_amount = self._phase_amount(phase_amounts, self.species_view.target_species)
        byproduct_amount = self._phase_amount(phase_amounts, self.species_view.byproduct_species)
        degradation_amount = self._phase_amount(
            phase_amounts,
            self.species_view.degradation_species,
        )
        impurity_amount = self._phase_amount(phase_amounts, self.species_view.impurity_species)
        reactant_amount = float(phase_amounts.get(self.species_view.reactant_species(state), 0.0))

        public_amounts: dict[str, float] = {}
        if reactant_amount > 0.0:
            public_amounts["reactant_public"] = reactant_amount
        if target_amount > 0.0:
            public_amounts["target_public"] = target_amount
        if impurity_amount > 0.0 or byproduct_amount > 0.0:
            public_amounts["impurity_public"] = max(
                impurity_amount - degradation_amount,
                byproduct_amount,
                0.0,
            )
        if degradation_amount > 0.0:
            public_amounts["degradation_public"] = degradation_amount
        return public_amounts or None

    @staticmethod
    def _phase_amount(
        phase_amounts: dict[str, float],
        species_ids: tuple[str, ...],
    ) -> float:
        return sum(float(phase_amounts.get(species_id, 0.0)) for species_id in species_ids)


__all__ = ["ChemWorldObservationKernel"]
