"""Crystallization state-update services for the transactional runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    process_with_metrics,
    upsert_equipment_record,
)
from chemworld.foundation.samples import StoredSample
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state import PhaseLedger, PhaseRecord
from chemworld.physchem.crystallization_units import (
    CoolingCrystallizationResult,
    CrystallizationExecutionSpec,
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
)
from chemworld.physchem.crystallization_validation import CrystallizationGridCase
from chemworld.physchem.elements import molecular_weight
from chemworld.runtime.full_process_contract import (
    population_active,
    population_settings,
    seed_provenance_active,
)
from chemworld.runtime.material_routing import _active, _contents
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.actions import SOLVENTS
from chemworld.world.mixtures import crystallization_medium_modifier
from chemworld.world.parameters import ChemWorldParameters
from chemworld.world.thermal_kernel import account_temperature_transition

DEFAULT_CRYSTAL_SEED_DIAMETER_M = 100.0e-6


def _action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def _effective_seed_diameter_m(crystallizer_settings: dict[str, Any]) -> float:
    """Return a physical seed diameter for the next population-balance step.

    An empty crystal population correctly reports ``csd_d50_m == 0``.  That
    population statistic is not a physical seed diameter and must not make a
    subsequent cooling step invalid.  If crystals are later seeded, the
    operation contract uses the reference seed diameter below; when no solid
    population exists, the positive value is accepted by the PBM but carries
    zero seed mass and therefore creates no particles.
    """

    recorded_d50_m = crystallizer_settings.get("csd_d50_m")
    if recorded_d50_m is None:
        return DEFAULT_CRYSTAL_SEED_DIAMETER_M
    try:
        candidate = float(recorded_d50_m)
    except (TypeError, ValueError):
        return DEFAULT_CRYSTAL_SEED_DIAMETER_M
    if candidate > 0.0 and np.isfinite(candidate):
        return candidate
    return DEFAULT_CRYSTAL_SEED_DIAMETER_M


def _solid_phase_amounts(
    state: WorldState,
    *,
    target_species: str,
    impurity_species: str,
) -> tuple[float, float]:
    if state.phases is None or "solid" not in state.phases.phases:
        return 0.0, 0.0
    solid = state.phases.phases["solid"]
    return (
        float(solid.species_amounts_mol.get(target_species, 0.0)),
        float(solid.species_amounts_mol.get(impurity_species, 0.0)),
    )


def _crystallization_phases(
    state: WorldState,
    *,
    target_species: str,
    impurity_species: str,
    product_mol: float,
    impurity_mol: float,
    mother_liquor_volume_L: float | None = None,
    solid_selected: bool = True,
) -> PhaseLedger:
    mother_liquor_volume = (
        state.volume_L if mother_liquor_volume_L is None else mother_liquor_volume_L
    )
    product_mol = float(np.clip(product_mol, 0.0, state.species_amounts.get(target_species, 0.0)))
    impurity_mol = float(
        np.clip(impurity_mol, 0.0, state.species_amounts.get(impurity_species, 0.0))
    )
    solid_amounts = dict.fromkeys(state.species_amounts, 0.0)
    solid_amounts[target_species] = product_mol
    solid_amounts[impurity_species] = impurity_mol
    liquor_amounts = state.species_amounts.copy()
    liquor_amounts[target_species] = max(
        liquor_amounts.get(target_species, 0.0) - product_mol,
        0.0,
    )
    liquor_amounts[impurity_species] = max(
        liquor_amounts.get(impurity_species, 0.0) - impurity_mol,
        0.0,
    )
    return PhaseLedger(
        {
            "mother_liquor": PhaseRecord(
                phase_id="mother_liquor",
                vessel_id=state.vessel_id,
                phase_type="liquid",
                volume_L=mother_liquor_volume,
                species_amounts_mol=liquor_amounts,
                settled=True,
                selected=not solid_selected,
                metadata={},
            ),
            "solid": PhaseRecord(
                phase_id="solid",
                vessel_id=state.vessel_id,
                phase_type="solid",
                volume_L=0.0,
                species_amounts_mol=solid_amounts,
                settled=True,
                selected=solid_selected,
                metadata={},
            ),
        }
    )


class ChemWorldCrystallizationServices:
    """Apply seed, cooling crystallization, and filtration updates."""

    def __init__(
        self,
        world: ChemWorldParameters,
        species_view: MechanismSpeciesView,
    ) -> None:
        self.world = world
        self.species_view = species_view
        self.runtime_provider: Any | None = None
        self.last_provider_execution: dict[str, Any] = {}

    def _target_molecular_weight_kg_mol(self) -> float:
        target = self.species_view.primary_target_species
        species = self.species_view.mechanism.network.species[
            self.species_view.mechanism.network.species_index[target]
        ]
        return molecular_weight(species.composition) / 1000.0

    def seed_crystals(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        seed_mass = float(np.clip(_action_float(action, "seed_mass_g", 0.005), 0.0, 0.050))
        target_species = self.species_view.primary_target_species
        impurity_species = self.species_view.primary_impurity_species
        seed_target_mol = seed_mass / 1000.0 / self._target_molecular_weight_kg_mol()
        previous_settings = equipment_settings(state.equipment, "crystallizer")
        cumulative_seed_mass = float(previous_settings.get("crystal_seed_mass_g", 0.0)) + seed_mass
        cumulative_seed_target = (
            float(previous_settings.get("seed_target_mol", 0.0)) + seed_target_mol
        )
        existing_solid_product, existing_solid_impurity = _solid_phase_amounts(
            state,
            target_species=target_species,
            impurity_species=impurity_species,
        )
        species_amounts = state.species_amounts.copy()
        species_amounts[target_species] = species_amounts.get(target_species, 0.0) + seed_target_mol
        augmented_state = state.replace(species_amounts=species_amounts)
        phases = _crystallization_phases(
            augmented_state,
            target_species=target_species,
            impurity_species=impurity_species,
            product_mol=existing_solid_product + seed_target_mol,
            impurity_mol=existing_solid_impurity,
            solid_selected=False,
        )
        equipment = upsert_equipment_record(
            augmented_state.equipment,
            equipment_id="crystallizer",
            equipment_type="crystallizer",
            attached_vessel_id=augmented_state.vessel_id,
            status="seeded" if seed_mass > 0.0 else "configured",
            settings={
                "crystal_seeded": seed_mass > 0.0,
                "crystal_seed_mass_g": cumulative_seed_mass,
                "seed_target_mol": cumulative_seed_target,
                "seed_charge_count": int(previous_settings.get("seed_charge_count", 0)) + 1,
                "last_seed_time_s": augmented_state.ledger.time_s,
                "seed_model_id": "explicit_target_seed_mass_v1",
            },
        )
        process_metrics = {} if state.process is None else state.process.metrics
        initial_target = max(
            float(
                process_metrics.get(
                    "pre_separation_product_mol",
                    self.species_view.target_amount(state),
                )
            ),
            1.0e-12,
        )
        process = process_with_metrics(
            augmented_state.process,
            pre_separation_product_mol=initial_target,
        )
        ledger = augmented_state.ledger.with_updates(
            cost=augmented_state.ledger.cost + 0.012 + 0.20 * seed_mass
        )
        if population_active(state):
            from math import pi

            cohorts = tuple(tuple(c) for c in previous_settings.get("population_cohorts", ()))
            seed_particle_mass_g = 1200.0 * pi / 6 * DEFAULT_CRYSTAL_SEED_DIAMETER_M**3 * 1000
            if seed_mass > 0:
                cohorts += ((seed_mass / seed_particle_mass_g, DEFAULT_CRYSTAL_SEED_DIAMETER_M),)
            equipment = upsert_equipment_record(
                equipment,
                equipment_id="crystallizer",
                equipment_type="crystallizer",
                attached_vessel_id=state.vessel_id,
                settings=population_settings(cohorts),
            )
        return augmented_state.replace(
            ledger=ledger,
            equipment=equipment,
            phases=phases,
            process=process,
        )

    def cool_crystallize(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        duration = float(np.clip(_action_float(action, "duration_s", 1200.0), 0.0, 14_400.0))
        target_temperature = float(
            np.clip(_action_float(action, "target_temperature_K", 278.15), 250.0, 330.0)
        )
        crystallizer_settings = equipment_settings(state.equipment, "crystallizer")
        target_species = self.species_view.primary_target_species
        impurity_species = self.species_view.primary_impurity_species
        explicit_seed_mass_g = float(crystallizer_settings.get("crystal_seed_mass_g", 0.0))
        explicit_seed_target_mol = float(crystallizer_settings.get("seed_target_mol", 0.0))
        phase_records = {} if state.phases is None else state.phases.phases
        mother_liquor = phase_records.get("mother_liquor")
        existing_solid_product, existing_solid_impurity = _solid_phase_amounts(
            state,
            target_species=target_species,
            impurity_species=impurity_species,
        )
        feed_amounts = (
            state.species_amounts if mother_liquor is None else mother_liquor.species_amounts_mol
        )
        feed_volume_L = state.volume_L if mother_liquor is None else mother_liquor.volume_L
        dissolved_product_mol = max(
            float(feed_amounts.get(target_species, 0.0)),
            0.0,
        )
        impurity_mol = max(float(feed_amounts.get(impurity_species, 0.0)), 0.0)
        target_molecular_weight = self._target_molecular_weight_kg_mol()
        # Every crystal already present in the slurry participates in the next
        # population-balance step.  Passing only the originally charged seed
        # would recreate material and discard the physical crystal population
        # on repeated cooling.
        effective_seed_target_mol = existing_solid_product
        effective_seed_mass_g = effective_seed_target_mol * target_molecular_weight * 1000.0
        initial_concentration = dissolved_product_mol / max(feed_volume_L, 1.0e-12)
        solubility_multiplier = crystallization_medium_modifier(state, self.world, "solubility")
        nucleation_multiplier = crystallization_medium_modifier(state, self.world, "nucleation")
        growth_multiplier = crystallization_medium_modifier(state, self.world, "growth")
        occlusion_multiplier = crystallization_medium_modifier(state, self.world, "occlusion")
        reference_solubility = (
            self.world.crystallization_reference_solubility_mol_L
            * self.world.domain_parameter("crystallization_solubility_multiplier")
            * solubility_multiplier
        )
        dissolution_enthalpy = 20_000.0
        occlusion_law_id = str(
            state.metadata.get(
                "crystallization_impurity_occlusion_law_id",
                "linear_supersaturation_transfer_v1",
            )
        )
        occlusion_capacity_multiplier = float(
            state.metadata.get("crystallization_impurity_occlusion_capacity_multiplier", 1.0)
        )
        if not 1.0 <= occlusion_capacity_multiplier <= 6.0:
            raise ValueError("crystallization occlusion capacity multiplier must be in [1, 6]")
        solubility = SolubilityCurveSpec(
            model_id="runtime_vanthoff_material_solubility_v2",
            reference_solubility_mol_L=reference_solubility,
            reference_temperature_K=298.15,
            dissolution_enthalpy_J_mol=dissolution_enthalpy,
            minimum_temperature_K=250.0,
            maximum_temperature_K=430.0,
            provenance_id="chemworld-world-law-v0.2-solubility-policy",
        )
        kinetics = CrystallizationKineticsSpec(
            model_id="runtime_cooling_population_balance_v1",
            primary_nucleation_coefficient_per_L_s=(
                2.0e7
                * self.world.domain_parameter("crystallization_nucleation_multiplier")
                * nucleation_multiplier
            ),
            primary_nucleation_exponent=2.0,
            growth_coefficient_m_s=(
                2.0e-8
                * self.world.domain_parameter("crystallization_growth_multiplier")
                * growth_multiplier
            ),
            growth_exponent=1.0,
            crystal_density_kg_m3=1200.0,
            target_molecular_weight_kg_mol=target_molecular_weight,
            nucleus_diameter_m=8.0e-6,
            impurity_occlusion_mol_per_mol=(
                0.02 * occlusion_multiplier * occlusion_capacity_multiplier
            ),
            supersaturation_occlusion_factor=0.5,
            impurity_occlusion_law_id=occlusion_law_id,
            impurity_surface_half_saturation_mol_L=0.010,
            fines_threshold_m=20.0e-6,
            provenance_id="chemworld-world-law-v0.2-crystallization-kinetics",
        )
        case = CrystallizationGridCase(
            feed_amounts_mol={
                target_species: dissolved_product_mol,
                impurity_species: impurity_mol,
            },
            target_component=target_species,
            impurity_component=impurity_species,
            solvent_volume_L=max(feed_volume_L, 1.0e-9),
            initial_temperature_K=state.temperature_K,
            final_temperature_K=target_temperature,
            duration_s=max(duration, 1.0e-9),
            solubility_curve=solubility,
            kinetics=kinetics,
            seed_mass_g=effective_seed_mass_g,
            seed_diameter_m=_effective_seed_diameter_m(crystallizer_settings),
            initial_cohorts=tuple(
                tuple(c) for c in crystallizer_settings.get("population_cohorts", ())
            )
            if population_active(state)
            else None,
            retain_population=population_active(state),
        )
        execution_spec = CrystallizationExecutionSpec.closed_loop_runtime()
        provider = self.runtime_provider
        if provider is None:
            from chemworld.physchem.crystallization_adapter_manifest import (
                ValidatedCrystallizationRuntimeProvider,
            )

            provider = ValidatedCrystallizationRuntimeProvider()
        time_steps = max(24, min(120, round(duration / 30.0)))
        provider_result = provider.evaluate(
            {
                "case": case,
                "time_steps": time_steps,
                "execution_spec": execution_spec,
            }
        )
        contract = provider.model_contract
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
            raise ValueError(
                provider_result.failure_reason or "crystallization runtime provider failed"
            )
        result = provider_result.outputs.get("result")
        if not isinstance(result, CoolingCrystallizationResult):
            raise ValueError("crystallization runtime provider returned no validated result")
        from chemworld.physchem.crystallization_adapter_manifest import (
            crystallization_runtime_adapter_manifest,
        )

        provider_manifest_hash = crystallization_runtime_adapter_manifest().manifest_hash
        crystallized = result.crystals_amounts_mol[target_species]
        crystallized_from_solution = result.crystallized_from_solution_mol
        occluded_impurity = existing_solid_impurity + result.crystals_amounts_mol[impurity_species]
        process_metrics = {} if state.process is None else state.process.metrics
        initial_p = max(
            float(
                process_metrics.get(
                    "pre_separation_product_mol",
                    max(
                        self.species_view.target_amount(state) - explicit_seed_target_mol,
                        0.0,
                    ),
                )
            ),
            dissolved_product_mol,
            1.0e-12,
        )
        cumulative_crystallized = (
            float(process_metrics.get("crystallized_from_solution_mol", 0.0))
            + crystallized_from_solution
        )
        cumulative_crystal_total = crystallized + occluded_impurity
        cumulative_crystal_purity = (
            crystallized / cumulative_crystal_total if cumulative_crystal_total > 1.0e-12 else 0.0
        )
        process = process_with_metrics(
            state.process,
            pre_separation_product_mol=initial_p,
            seed_target_mol=explicit_seed_target_mol,
            crystallized_from_solution_mol=cumulative_crystallized,
            crystal_yield=float(np.clip(cumulative_crystallized / initial_p, 0.0, 1.0)),
            seed_excluded_recovery=float(np.clip(cumulative_crystallized / initial_p, 0.0, 1.0)),
            crystal_purity=float(np.clip(cumulative_crystal_purity, 0.0, 1.0)),
            crystal_size=float(
                np.clip(result.crystal_size_distribution.d50_m / 250.0e-6, 0.0, 1.0)
            ),
            crystal_csd_quality=float(
                np.clip(
                    0.55
                    * (
                        1.0
                        - np.clip(
                            result.crystal_size_distribution.coefficient_of_variation,
                            0.0,
                            1.0,
                        )
                    )
                    + 0.45 * (1.0 - result.crystal_size_distribution.fines_number_fraction),
                    0.0,
                    1.0,
                )
            ),
            crystal_fines_fraction=float(
                np.clip(
                    result.crystal_size_distribution.fines_number_fraction,
                    0.0,
                    1.0,
                )
            ),
        )
        cooling_depth = float(np.clip((state.temperature_K - target_temperature) / 55.0, 0.0, 1.0))
        crystallization_heat = -dissolution_enthalpy * crystallized_from_solution
        thermal = account_temperature_transition(
            state=state,
            world=self.world,
            final_temperature_K=target_temperature,
            duration_s=max(duration, 1.0e-9),
            phase_change_heat_J=crystallization_heat,
        )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=(
                state.ledger.cost
                + 0.018
                + duration / 3600.0 * 0.018
                + abs(thermal.jacket_energy_J) / 250_000.0
            ),
            risk=max(0.0, state.ledger.risk - 0.02 * cooling_depth),
            energy_jacket_J=state.ledger.energy_jacket_J + thermal.jacket_energy_J,
            heat_reaction_J=state.ledger.heat_reaction_J + thermal.phase_change_heat_J,
            heat_loss_J=state.ledger.heat_loss_J + thermal.heat_loss_J,
        )
        phases = _crystallization_phases(
            state,
            target_species=target_species,
            impurity_species=impurity_species,
            product_mol=crystallized,
            impurity_mol=occluded_impurity,
        )
        execution_history = list(crystallizer_settings.get("execution_history", ()))
        execution_history.append(
            {
                "execution_index": len(execution_history) + 1,
                "model_id": contract.model_id,
                "provider_path": contract.provider_path,
                "provider_manifest_hash": provider_manifest_hash,
                "initial_temperature_K": state.temperature_K,
                "final_temperature_K": target_temperature,
                "duration_s": duration,
                "explicit_seed_mass_g": explicit_seed_mass_g,
                "effective_seed_mass_g": effective_seed_mass_g,
                "prior_solid_target_mol": existing_solid_product,
                "added_crystallized_from_solution_mol": crystallized_from_solution,
                "cumulative_solid_target_mol": crystallized,
                "target_recovery": result.target_recovery,
                "crystal_purity": cumulative_crystal_purity,
                "csd_d50_m": result.crystal_size_distribution.d50_m,
                "diagnostics": dict(provider_result.diagnostics),
                "provenance": list(provider_result.provenance),
            }
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="crystallizer",
            equipment_type="crystallizer",
            attached_vessel_id=state.vessel_id,
            status="completed",
            settings={
                "crystallization_model_id": result.model_id,
                "provider_path": contract.provider_path,
                "provider_manifest_hash": provider_manifest_hash,
                "provider_maturity": contract.maturity.value,
                "provider_role": contract.role.value,
                "provider_diagnostics": dict(provider_result.diagnostics),
                "provider_provenance": list(provider_result.provenance),
                "execution_history": execution_history,
                "execution_spec": execution_spec.to_dict(),
                "solubility_model_id": solubility.model_id,
                "reference_solubility_mol_L": reference_solubility,
                "reference_solubility_temperature_K": solubility.reference_temperature_K,
                "feed_concentration_mol_L": initial_concentration,
                "kinetics_model_id": kinetics.model_id,
                "impurity_occlusion_law_id": kinetics.impurity_occlusion_law_id,
                "seed_target_mol": explicit_seed_target_mol,
                "effective_seed_target_mol": effective_seed_target_mol,
                "crystallized_from_solution_mol": cumulative_crystallized,
                "thermal_transition": thermal.to_dict(),
                "material_balance_error_mol": result.material_balance_error_mol,
                "component_balance_errors_mol": result.component_balance_errors_mol,
                "particle_target_balance_error_mol": (result.particle_target_balance_error_mol),
                "growth_solver_converged": result.growth_solver_converged,
                "growth_solver_iterations": result.growth_solver_iterations,
                "growth_solver_max_residual_mol": result.growth_solver_max_residual_mol,
                "cooling_rate_K_s": result.cooling_rate_K_s,
                "maximum_supersaturation_ratio": result.maximum_supersaturation_ratio,
                "final_supersaturation_ratio": result.final_supersaturation_ratio,
                "csd_d10_m": result.crystal_size_distribution.d10_m,
                "csd_d50_m": result.crystal_size_distribution.d50_m,
                "csd_d90_m": result.crystal_size_distribution.d90_m,
                "csd_cv": result.crystal_size_distribution.coefficient_of_variation,
                "csd_fines_number_fraction": (
                    result.crystal_size_distribution.fines_number_fraction
                ),
                "csd_total_particle_count": (result.crystal_size_distribution.total_particle_count),
                "csd_number_moment_0": result.crystal_size_distribution.number_moment_0,
                "csd_length_moment_1_m": (result.crystal_size_distribution.length_moment_1_m),
                "csd_area_moment_2_m2": (result.crystal_size_distribution.area_moment_2_m2),
                "csd_volume_moment_3_m3": (result.crystal_size_distribution.volume_moment_3_m3),
                "temperature_history_K": [
                    state.temperature_K,
                    *[step.temperature_K for step in result.step_reports],
                ],
                "supersaturation_history": [
                    step.supersaturation_ratio for step in result.step_reports
                ],
                "nucleation_history_per_L_s": [
                    step.nucleation_rate_per_L_s for step in result.step_reports
                ],
                "growth_history_m_s": [step.growth_rate_m_s for step in result.step_reports],
                "model_warnings": list(result.warnings),
                "provenance": result.provenance,
            },
        )
        if population_active(state):
            dissolved_seed = float(crystallizer_settings.get("dissolved_seed_target_mol", 0.0))
            transferred_seed = min(
                dissolved_seed,
                dissolved_seed * crystallized_from_solution / max(dissolved_product_mol, 1e-12),
            )
            equipment = upsert_equipment_record(
                equipment,
                equipment_id="crystallizer",
                equipment_type="crystallizer",
                attached_vessel_id=state.vessel_id,
                settings={
                    **population_settings(result.population_cohorts),
                    **(
                        {
                            "seed_target_mol": explicit_seed_target_mol + transferred_seed,
                            "dissolved_seed_target_mol": dissolved_seed - transferred_seed,
                        }
                        if seed_provenance_active(state)
                        else {}
                    ),
                },
            )
        metadata = {
            **state.metadata,
            "last_energy_transition": {
                "operation": "cool_crystallize",
                **thermal.to_dict(),
            },
        }
        return state.replace(
            temperature_K=target_temperature,
            ledger=ledger,
            process=process,
            phases=phases,
            equipment=equipment,
            metadata=metadata,
        )

    def resuspend_crystals(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        """Retain the separated filtrate and put wet cake in fresh, isothermal solvent.

        This operation does not force dissolution. Subsequent heating uses the
        solubility curve, particle shrinkage and dissolution-energy model.
        """
        assert state.phases is not None
        liquor = state.phases.phases["mother_liquor"]
        solid = state.phases.phases["solid"]
        cake_liquor = state.phases.phases["cake_liquor"]
        crystal_settings = equipment_settings(state.equipment, "crystallizer")
        dissolved_seed = float(crystal_settings.get("dissolved_seed_target_mol", 0.0))
        liquid_total = liquor.volume_L + cake_liquor.volume_L
        retained_liquid_fraction = cake_liquor.volume_L / max(liquid_total, 1e-12)
        volume = _action_float(action, "volume_L", 0.015)
        solvent = int(_action_float(action, "solvent", 0))
        if not 0.0001 <= volume <= 0.080 or solvent not in range(len(SOLVENTS)):
            raise ValueError("Invalid resuspension volume or solvent")
        sample_id = next(
            (
                f"filtrate-{i:04d}"
                for i in range(1, 17)
                if f"filtrate-{i:04d}" not in state.samples.samples
            ),
            None,
        )
        if sample_id is None:
            raise ValueError("All 16 disposable filtrate containers have been used")
        source = _active(state)
        source_contents = _contents(source)
        sample = StoredSample(
            sample_id=sample_id,
            source_vessel_id=state.vessel_id,
            collection_operation="resuspend_crystals",
            collected_at_s=state.ledger.time_s,
            volume_L=liquor.volume_L,
            species_amounts_mol=liquor.species_amounts_mol,
            temperature_K=state.temperature_K,
            solvent=liquor.solvents.dominant_index,
            quenched=state.quenched,
            seed_target_mol=dissolved_seed * (1.0 - retained_liquid_fraction),
            lineage=source.lineage,
        )
        sample = replace(
            sample,
            contents=replace(
                _contents(sample),
                species=source_contents.species,
                phases=PhaseLedger(
                    {
                        "reactor_liquid": replace(
                            liquor, phase_id="reactor_liquid", vessel_id=sample_id, selected=False
                        )
                    }
                ),
                reference_sources=source_contents.reference_sources,
                reference_shares=source_contents.reference_shares,
            ),
        )
        liquid_volume = volume + cake_liquor.volume_L
        phases = PhaseLedger(
            {
                "solid": replace(solid, volume_L=0.0, selected=False),
                "mother_liquor": replace(
                    liquor,
                    volume_L=liquid_volume,
                    species_amounts_mol=cake_liquor.species_amounts_mol,
                    solvents=cake_liquor.solvents + SolventInventory.pure(solvent, volume),
                    selected=True,
                ),
            }
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="crystal_filter",
            equipment_type="solid_liquid_filter",
            attached_vessel_id=state.vessel_id,
            status="resuspended",
            settings={"crystals_filtered": False},
        )
        equipment = upsert_equipment_record(
            equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id=state.vessel_id,
            settings={"solvent_volume_L": liquid_volume, "catalyst_amount_mol": 0.0},
        )
        equipment = upsert_equipment_record(
            equipment,
            equipment_id="crystallizer",
            equipment_type="crystallizer",
            attached_vessel_id=state.vessel_id,
            settings={"dissolved_seed_target_mol": dissolved_seed * retained_liquid_fraction},
        )
        return state.replace(
            phases=phases,
            species_amounts=phases.total_amounts_mol(),
            volume_L=liquid_volume,
            equipment=equipment,
            samples=replace(
                state.samples.append(sample),
                containers_used=state.samples.containers_used + 1,
                active_lineage=source.lineage,
                active_reference_sources=source_contents.reference_sources,
                active_reference_shares=source_contents.reference_shares,
            ),
            ledger=state.ledger.with_updates(
                cost=state.ledger.cost
                + 0.030
                + volume * 8.0 * float(self.world.solvent_costs[solvent])
            ),
        )

    def filter_crystals(self, state: WorldState) -> WorldState:
        target_species = self.species_view.primary_target_species
        impurity_species = self.species_view.primary_impurity_species
        solid_product, solid_impurity = _solid_phase_amounts(
            state,
            target_species=target_species,
            impurity_species=impurity_species,
        )
        product = solid_product * 0.96
        impurity = solid_impurity * 0.92
        purity = product / max(product + impurity, 1.0e-12)
        process_metrics = {} if state.process is None else state.process.metrics
        crystallizer_settings = equipment_settings(state.equipment, "crystallizer")
        seed_target_mol = float(crystallizer_settings.get("seed_target_mol", 0.0))
        retained_seed_mol = min(seed_target_mol, solid_product) * 0.96
        product_from_solution = max(product - retained_seed_mol, 0.0)
        target_amount = max(self.species_view.target_amount(state) - seed_target_mol, 0.0)
        initial_p = max(
            float(process_metrics.get("pre_separation_product_mol", target_amount)),
            1.0e-12,
        )
        solvent_loss = min(
            1.0,
            float(process_metrics.get("solvent_loss", 0.0)) + 0.04,
        )
        process = process_with_metrics(
            state.process,
            pre_separation_product_mol=initial_p,
            solvent_loss=solvent_loss,
            retained_seed_mol=retained_seed_mol,
            filtered_product_from_solution_mol=product_from_solution,
            crystal_yield=float(np.clip(product_from_solution / initial_p, 0.0, 1.0)),
            seed_excluded_recovery=float(np.clip(product_from_solution / initial_p, 0.0, 1.0)),
            crystal_purity=float(np.clip(purity, 0.0, 1.0)),
            recovery=float(np.clip(product_from_solution / initial_p, 0.0, 1.0)),
            purity=float(np.clip(purity, 0.0, 1.0)),
        )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + 480.0,
            cost=state.ledger.cost + 0.026,
        )
        phases = _crystallization_phases(
            state,
            target_species=target_species,
            impurity_species=impurity_species,
            product_mol=product,
            impurity_mol=impurity,
            mother_liquor_volume_L=state.volume_L * 0.92,
        )
        # The wet cake retains 8% of the well-mixed mother liquor, including
        # its dissolved solutes. Both liquid fractions remain in the ledger.
        liquor = phases.phases["mother_liquor"]
        phases = PhaseLedger(
            {
                **phases.phases,
                "mother_liquor": replace(
                    liquor,
                    species_amounts_mol={
                        s: n * 0.92 for s, n in liquor.species_amounts_mol.items()
                    },
                ),
                "cake_liquor": replace(
                    liquor,
                    phase_id="cake_liquor",
                    volume_L=state.volume_L * 0.08,
                    species_amounts_mol={
                        s: n * 0.08 for s, n in liquor.species_amounts_mol.items()
                    },
                    selected=False,
                ),
            }
        )
        phase_totals = phases.total_amounts_mol()
        filter_balance_error = max(
            (
                abs(state.species_amounts.get(species_id, 0.0) - amount_mol)
                for species_id, amount_mol in phase_totals.items()
            ),
            default=0.0,
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="crystal_filter",
            equipment_type="solid_liquid_filter",
            attached_vessel_id=state.vessel_id,
            status="completed",
            settings={
                "crystals_filtered": True,
                "filtered_product_mol": product,
                "retained_seed_mol": retained_seed_mol,
                "filtered_product_from_solution_mol": product_from_solution,
                "filtered_impurity_mol": impurity,
                "filter_purity": purity,
                "target_solid_retention_fraction": 0.96,
                "impurity_solid_retention_fraction": 0.92,
                "target_returned_to_filtrate_mol": solid_product - product,
                "impurity_returned_to_filtrate_mol": solid_impurity - impurity,
                "component_balance_error_mol": filter_balance_error,
                "provenance": [
                    "ChemWorld solid-liquid filtration retention ledger v1",
                    "coupled to cooling_crystallization_population_balance_v1 solid phase",
                ],
            },
        )
        if population_active(state):
            equipment = upsert_equipment_record(
                equipment,
                equipment_id="crystallizer",
                equipment_type="crystallizer",
                attached_vessel_id=state.vessel_id,
                settings={
                    **population_settings(
                        tuple(
                            (n * 0.96, d)
                            for n, d in crystallizer_settings.get("population_cohorts", ())
                        )
                    ),
                    "seed_target_mol": retained_seed_mol,
                    **(
                        {
                            "dissolved_seed_target_mol": float(
                                crystallizer_settings.get("dissolved_seed_target_mol", 0.0)
                            )
                            + seed_target_mol
                            - retained_seed_mol
                        }
                        if seed_provenance_active(state)
                        else {}
                    ),
                },
            )
        return state.replace(ledger=ledger, phases=phases, process=process, equipment=equipment)


__all__ = ["ChemWorldCrystallizationServices"]
