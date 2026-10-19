"""Phase-ledger and downstream separation services for the transactional runtime."""

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
from chemworld.foundation.solvents import ORGANIC_DISTRIBUTION_RATIOS, SolventInventory
from chemworld.foundation.state import PhaseLedger, PhaseRecord, selected_phase_id
from chemworld.physchem.extraction_units import (
    DistributionCoefficientModelSpec,
)
from chemworld.physchem.phase_equilibrium_units import (
    LLEContactorSpec,
    StabilityAwareExtractionRequest,
    simulate_stability_aware_extraction,
)
from chemworld.runtime.phase_ledger_services import (
    ChemWorldPhaseLedgerServices,
    action_float,
    empty_phase,
)
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.runtime.vnext_downstream import (
    PhaseSlice,
    run_bounded_transfer,
    run_sorbent_drying,
    run_vacuum_concentration,
)
from chemworld.world.actions import SOLVENTS
from chemworld.world.parameters import ChemWorldParameters
from chemworld.world.phase_kernel import (
    INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1,
    partition_split,
)
from chemworld.world.species_roles import PHASE_PRODUCT_AMOUNT_KEY


class ChemWorldPhaseSeparationServices:
    """Maintain phase ledgers and execute extraction-style separation steps."""

    def __init__(
        self,
        world: ChemWorldParameters,
        species_view: MechanismSpeciesView,
        *,
        nominal_pair_contract: str | None = None,
    ) -> None:
        if nominal_pair_contract not in {
            None,
            INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1,
        }:
            raise ValueError(
                f"Unsupported partition nominal-pair contract: {nominal_pair_contract!r}"
            )
        self.world = world
        self.species_view = species_view
        self.nominal_pair_contract = nominal_pair_contract
        self.phase_ledgers = ChemWorldPhaseLedgerServices(species_view)

    def _removed_inventory_summary(
        self,
        state: WorldState,
        *,
        operation: str,
        inventories: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], float, float, float]:
        target_species = set(self.species_view.target_species_for_state(state))
        impurity_species = set(self.species_view.impurity_species_for_state(state))
        removed_volume = 0.0
        removed_product = 0.0
        removed_impurity = 0.0
        normalized: dict[str, dict[str, Any]] = {}
        for inventory_id, values in inventories.items():
            amounts = values.get("species_amounts_mol", {})
            amounts = dict(amounts) if isinstance(amounts, dict) else {}
            product = float(values.get("product_mol", 0.0)) + sum(
                float(amounts.get(species_id, 0.0)) for species_id in target_species
            )
            impurity = float(values.get("impurity_mol", 0.0)) + sum(
                float(amounts.get(species_id, 0.0)) for species_id in impurity_species
            )
            volume = max(float(values.get("volume_L", 0.0)), 0.0)
            removed_volume += volume
            removed_product += product
            removed_impurity += impurity
            normalized[str(inventory_id)] = {
                "volume_L": volume,
                "product_mol": product,
                "impurity_mol": impurity,
                "species_amounts_mol": {
                    str(species_id): float(amount)
                    for species_id, amount in amounts.items()
                    if float(amount) > 0.0
                },
            }
        history = list(state.metadata.get("removed_phase_inventory_history", ()))
        history.append(
            {
                "operation": operation,
                "time_s": state.ledger.time_s,
                "inventories": normalized,
            }
        )
        return (
            {"removed_phase_inventory_history": history},
            removed_volume,
            removed_product,
            removed_impurity,
        )

    @staticmethod
    def _account_removed_inventory(
        state: WorldState,
        *,
        removed_volume_L: float,
        removed_product_mol: float,
        removed_impurity_mol: float,
    ) -> WorldState:
        metrics = {} if state.process is None else state.process.metrics
        process = process_with_metrics(
            state.process,
            removed_phase_product_mol=(
                float(metrics.get("removed_phase_product_mol", 0.0)) + removed_product_mol
            ),
            removed_phase_impurity_mol=(
                float(metrics.get("removed_phase_impurity_mol", 0.0)) + removed_impurity_mol
            ),
        )
        process = replace(
            process,
            waste_L=process.waste_L + removed_volume_L,
        )
        return state.replace(process=process)

    def add_phase(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        volume = float(np.clip(action_float(action, "volume_L", 0.015), 0.0, 0.060))
        phase_name = str(action.get("phase", "aqueous"))
        if phase_name not in {"aqueous", "organic"}:
            phase_name = "organic"
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        phase = phase_ledger.setdefault(phase_name, empty_phase())
        phase["volume_L"] += volume
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.015 + 0.35 * volume)
        return self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            phase_settled=False,
            selected_phase=None,
            ledger=ledger,
            volume_L=state.volume_L + volume,
        )

    def add_extractant(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        volume = float(np.clip(action_float(action, "volume_L", 0.018), 0.0, 0.060))
        extractant = int(np.clip(action_float(action, "extractant", 3.0), 0, len(SOLVENTS) - 1))
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        organic = phase_ledger.setdefault("organic", empty_phase())
        organic["volume_L"] += volume
        extractor_settings = equipment_settings(state.equipment, "liquid_liquid_extractor")
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="liquid_liquid_extractor",
            equipment_type="liquid_liquid_extractor",
            attached_vessel_id=state.vessel_id,
            settings={
                "extractant": extractant,
                "extractant_volume_L": float(extractor_settings.get("extractant_volume_L", 0.0))
                + volume,
            },
        )
        risk = min(
            1.0,
            state.ledger.risk + 0.04 + 0.05 * float(self.world.solvent_risks[extractant]),
        )
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + 0.025 + 0.80 * volume,
            risk=risk,
        )
        return self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            metadata_updates={"extractant": extractant},
            phase_settled=False,
            selected_phase=None,
            ledger=ledger,
            volume_L=state.volume_L + volume,
            equipment=equipment,
        )

    def mix_phases(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        duration = float(np.clip(action_float(action, "duration_s", 180.0), 0.0, 1800.0))
        stirring = float(np.clip(action_float(action, "stirring_speed_rpm", 700.0), 100.0, 1200.0))
        process_metrics = {} if state.process is None else state.process.metrics
        if "pre_separation_product_mol" not in process_metrics:
            state = state.replace(
                process=process_with_metrics(
                    state.process,
                    pre_separation_product_mol=max(
                        self.phase_ledgers.phase_product_amount(state),
                        1.0e-12,
                    ),
                )
            )
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        phase_ledger.setdefault(
            "aqueous",
            {
                "volume_L": 0.0,
                PHASE_PRODUCT_AMOUNT_KEY: 0.0,
                "impurity_mol": 0.0,
                "solvent_loss": 0.0,
            },
        )
        organic = phase_ledger.setdefault("organic", empty_phase())
        aqueous = phase_ledger["aqueous"]
        reactor_liquid = phase_ledger.pop("reactor_liquid", None)
        if reactor_liquid is not None:
            aqueous["volume_L"] += float(reactor_liquid.get("volume_L", 0.0))
        p_total = self.phase_ledgers.phase_product_amount(state)
        impurity_total = self.phase_ledgers.phase_impurity_amount(state)
        extractor_settings = equipment_settings(state.equipment, "liquid_liquid_extractor")
        reactor_settings = equipment_settings(state.equipment, "batch_reactor")
        partition_inputs: dict[str, Any] = {}
        if self.nominal_pair_contract == INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1:
            solvent = int(reactor_settings.get("solvent", 0))
            partition_inputs = {
                "extractant": int(extractor_settings.get("extractant", solvent)),
                "nominal_pair_contract": self.nominal_pair_contract,
            }
        else:
            # Preserve the historical runtime law for every task/contract except
            # the explicitly selected partition v3 benchmark.
            solvent = int(
                extractor_settings.get(
                    "extractant",
                    reactor_settings.get("solvent", 0),
                )
            )
        carrier_pool = SolventInventory()
        if state.phases is not None:
            for phase in state.phases.phases.values():
                carrier_pool = carrier_pool + phase.solvents
        organic_medium, aqueous_medium = carrier_pool.split(
            organic["volume_L"], ORGANIC_DISTRIBUTION_RATIOS
        )
        split = partition_split(
            solvent_inventory=aqueous_medium,
            extractant_inventory=organic_medium,
            product_mol=p_total,
            impurity_mol=impurity_total,
            solvent=solvent,
            temperature_K=state.temperature_K,
            duration_s=duration,
            stirring_speed_rpm=stirring,
            organic_volume_L=organic["volume_L"],
            aqueous_volume_L=aqueous["volume_L"],
            coefficient_multiplier=self.world.domain_parameter("partition_coefficient_multiplier"),
            coefficient_exponent=self.world.domain_parameter("partition_coefficient_exponent"),
            composition_coupling_multiplier=float(
                state.metadata.get("partition_composition_coupling_multiplier", 1.0)
            ),
            phase_volume_multiplier=self.world.domain_parameter(
                "partition_phase_volume_multiplier"
            ),
            **partition_inputs,
        )
        organic[PHASE_PRODUCT_AMOUNT_KEY] = split["organic_product_mol"]
        aqueous[PHASE_PRODUCT_AMOUNT_KEY] = split["aqueous_product_mol"]
        organic["impurity_mol"] = split["organic_impurity_mol"]
        aqueous["impurity_mol"] = split["aqueous_impurity_mol"]
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="phase_mixer",
            equipment_type="phase_mixer",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={"stirring_speed_rpm": stirring},
        )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + 0.01 + duration / 3600.0 * 0.015,
        )
        return self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            metadata_updates={
                "partition_coefficient": split["partition_coefficient"],
                "impurity_partition_coefficient": split["impurity_partition_coefficient"],
                "lle_phase_status": split["lle_phase_status"],
                "lle_minimum_tpd_like": split["lle_minimum_tpd_like"],
                "lle_partition_log_spread": split["lle_partition_log_spread"],
                "extraction_model_id": split["extraction_model_id"],
                "extraction_converged": split["extraction_converged"],
                "extraction_material_balance_error_mol": split[
                    "extraction_material_balance_error_mol"
                ],
                "extraction_entrained_aqueous_volume_L": split[
                    "extraction_entrained_aqueous_volume_L"
                ],
                "extraction_provenance": list(split["extraction_provenance"]),
            },
            phase_settled=False,
            ledger=ledger,
            equipment=equipment,
        )

    def settle_phases(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        duration = float(np.clip(action_float(action, "duration_s", 300.0), 0.0, 3600.0))
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + duration / 3600.0 * 0.006,
        )
        return self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            phase_settled=duration >= 60.0,
            ledger=ledger,
        )

    def separate_phase(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        target = str(action.get("target_phase", "organic"))
        if target not in {"organic", "aqueous"}:
            target = "organic"
        if state.phases is None or target not in state.phases.phases:
            raise ValueError(f"target phase is unavailable: {target}")
        selected = state.phases.phases[target]
        removed = {
            phase_id: {
                "volume_L": phase.volume_L,
                "species_amounts_mol": phase.species_amounts_mol,
            }
            for phase_id, phase in state.phases.phases.items()
            if phase_id != target
        }
        metadata_updates, removed_volume, removed_product, removed_impurity = (
            self._removed_inventory_summary(
                state,
                operation="separate_phase",
                inventories=removed,
            )
        )
        entrained_volume_L = float(state.metadata.get("extraction_entrained_aqueous_volume_L", 0.0))
        contact_volume_L = sum(
            max(float(phase.volume_L), 0.0) for phase in state.phases.phases.values()
        )
        entrainment_fraction = float(
            np.clip(entrained_volume_L / max(contact_volume_L, 1.0e-12), 0.0, 1.0)
        )
        selected_record = PhaseRecord(
            phase_id=selected.phase_id,
            vessel_id=selected.vessel_id,
            phase_type=selected.phase_type,
            volume_L=selected.volume_L,
            species_amounts_mol=selected.species_amounts_mol,
            settled=True,
            selected=True,
            metadata=selected.metadata,
        )
        phases = PhaseLedger({target: selected_record})
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.025)
        process_metrics = {} if state.process is None else state.process.metrics
        process = process_with_metrics(
            state.process,
            solvent_loss=min(
                1.0,
                float(process_metrics.get("solvent_loss", 0.0)) + entrainment_fraction,
            ),
        )
        candidate = state.replace(
            species_amounts=phases.total_amounts_mol(),
            phases=phases,
            metadata=self.phase_ledgers.phase_metadata(
                state,
                {},
                updates=metadata_updates,
            ),
            ledger=ledger,
            process=process,
            volume_L=selected.volume_L,
        )
        selected_ledger = self.phase_ledgers.phase_ledger(candidate)
        candidate = candidate.replace(
            process=process_with_metrics(
                candidate.process,
                **self.phase_ledgers.phase_process_metrics(candidate, selected_ledger),
            )
        )
        return self._account_removed_inventory(
            candidate,
            removed_volume_L=removed_volume,
            removed_product_mol=removed_product,
            removed_impurity_mol=removed_impurity,
        )

    def wash_phase(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        volume = float(np.clip(action_float(action, "wash_volume_L", 0.010), 0.0, 0.040))
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        target = selected_phase_id(state.phases) or "organic"
        phase = phase_ledger.setdefault(
            target,
            {
                "volume_L": state.volume_L,
                PHASE_PRODUCT_AMOUNT_KEY: self.phase_ledgers.phase_product_amount(state),
                "impurity_mol": 0.0,
                "solvent_loss": 0.0,
            },
        )
        feed = {
            "product": max(float(phase[PHASE_PRODUCT_AMOUNT_KEY]), 0.0),
            "impurity": max(float(phase["impurity_mol"]), 0.0),
        }
        if sum(feed.values()) <= 0.0 or target != "organic":
            ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.02 + 0.25 * volume)
            return self.phase_ledgers.with_phase_ledger(
                state,
                phase_ledger,
                selected_phase=target,
                ledger=ledger,
                volume_L=phase["volume_L"],
            )
        distribution_model = DistributionCoefficientModelSpec(
            model_id="runtime_wash_distribution_v1",
            component_ids=("product", "impurity"),
            intrinsic_partition_coefficients={
                "product": max(
                    float(state.metadata.get("partition_coefficient", 1.0)),
                    0.05,
                ),
                "impurity": max(
                    float(state.metadata.get("impurity_partition_coefficient", 0.25)),
                    0.05,
                ),
            },
            provenance_id="chemworld-world-law-vnext-wash-policy",
        )
        result = simulate_stability_aware_extraction(
            StabilityAwareExtractionRequest(
                feed_amounts_mol=feed,
                distribution_model=distribution_model,
                target_component="product",
                contactor=LLEContactorSpec(
                    aqueous_volume_L=max(volume, 1.0e-9),
                    organic_volume_L=max(float(phase["volume_L"]), 1.0e-9),
                    extraction_stages=1,
                    extraction_stage_efficiency=0.95,
                    extraction_entrainment_fraction=0.01,
                    maximum_contact_volume_L=max(volume + float(phase["volume_L"]), 0.10),
                ),
                temperature_K=state.temperature_K,
            )
        )
        retained = result.outlet("extract")
        removed = result.outlet("raffinate")
        stage = result.stage_reports[0]
        phase[PHASE_PRODUCT_AMOUNT_KEY] = retained["product"]
        phase["impurity_mol"] = retained["impurity"]
        phase["volume_L"] += stage.entrained_volume_L
        phase["solvent_loss"] += stage.entrained_volume_L / max(
            volume,
            1.0e-12,
        )
        removed_wash_volume = max(volume - stage.entrained_volume_L, 0.0)
        metadata_updates, removed_volume, removed_product, removed_impurity = (
            self._removed_inventory_summary(
                state,
                operation="wash",
                inventories={
                    "wash_aqueous": {
                        "product_mol": removed["product"],
                        "impurity_mol": removed["impurity"],
                        "volume_L": removed_wash_volume,
                    }
                },
            )
        )
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + 0.02 + 0.25 * volume,
            risk=min(1.0, state.ledger.risk + 0.02 * (1.0 - result.impurity_rejection)),
        )
        next_state = self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            metadata_updates={
                "wash_model_id": result.model_id,
                "wash_distribution_model_id": result.distribution_model_id,
                "wash_material_balance_error_mol": result.material_balance_error_mol,
                "wash_converged": result.all_stages_converged,
                "wash_entrained_aqueous_volume_L": (result.entrained_volume_L),
                "wash_provenance": list(result.provenance),
                **metadata_updates,
            },
            selected_phase=target,
            ledger=ledger,
            volume_L=phase["volume_L"],
        )
        return self._account_removed_inventory(
            next_state,
            removed_volume_L=removed_volume,
            removed_product_mol=removed_product,
            removed_impurity_mol=removed_impurity,
        )

    def dry_phase(self, state: WorldState) -> WorldState:
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        target = selected_phase_id(state.phases) or "organic"
        phase = phase_ledger.setdefault(
            target,
            {
                "volume_L": state.volume_L,
                PHASE_PRODUCT_AMOUNT_KEY: self.phase_ledgers.phase_product_amount(state),
                "impurity_mol": 0.0,
                "solvent_loss": 0.0,
            },
        )
        phase_slice = PhaseSlice(
            product_mol=max(float(phase[PHASE_PRODUCT_AMOUNT_KEY]), 0.0),
            impurity_mol=max(float(phase["impurity_mol"]), 0.0),
            volume_L=max(float(phase["volume_L"]), 0.0),
            solvent_loss=max(float(phase.get("solvent_loss", 0.0)), 0.0),
        )
        metadata_updates: dict[str, Any] = {
            "drying_model_id": "chemworld_sorbent_drying_vnext",
            "drying_skipped_empty_phase": not phase_slice.has_material,
        }
        removed_volume = 0.0
        removed_product = 0.0
        removed_impurity = 0.0
        medium = state.phases.phases[target].solvents if state.phases else SolventInventory()
        water_mol = medium.volumes_L[0] / 0.018
        metadata_updates["drying_skipped_no_water"] = water_mol <= 0
        if phase_slice.has_material and water_mol > 0:
            result = run_sorbent_drying(phase_slice, water_mol=water_mol)
            phase[PHASE_PRODUCT_AMOUNT_KEY] = result.dried_liquid_amounts_mol.get("product", 0.0)
            phase["impurity_mol"] = result.dried_liquid_amounts_mol.get("impurity", 0.0)
            removed_water_L = result.spent_sorbent_inventory_mol.get("residual_water", 0.0) * 0.018
            medium = SolventInventory(
                (max(medium.volumes_L[0] - removed_water_L, 0.0), *medium.volumes_L[1:])
            )
            phase["volume_L"] = medium.volume_L
            phase["solvent_loss"] = result.residual_drying_component_fraction
            removed_metadata, removed_volume, removed_product, removed_impurity = (
                self._removed_inventory_summary(
                    state,
                    operation="dry",
                    inventories={
                        "spent_sorbent": {
                            "product_mol": result.spent_sorbent_inventory_mol.get("product", 0.0),
                            "impurity_mol": result.spent_sorbent_inventory_mol.get("impurity", 0.0),
                            "volume_L": removed_water_L,
                        }
                    },
                )
            )
            metadata_updates.update(
                {
                    "drying_endpoint_met": result.endpoint_met,
                    "drying_carrier_removed_L": removed_water_L,
                    "drying_material_balance_error_mol": result.material_balance_error_mol,
                    "drying_volume_balance_error_L": result.volume_balance_error_L,
                    "drying_product_recovery": result.product_recovery,
                    "drying_warnings": list(result.warnings),
                    "drying_provenance": list(result.provenance),
                    "drying_provider_path": (
                        "chemworld.physchem.drying_adapter_manifest.SorbentDryingProvider"
                    ),
                    **removed_metadata,
                }
            )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + 300.0, cost=state.ledger.cost + 0.018
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="sorbent_dryer",
            equipment_type="finite_capacity_sorbent_dryer",
            attached_vessel_id=state.vessel_id,
            status="dried",
            settings=metadata_updates,
        )
        next_state = self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            selected_phase=target,
            ledger=ledger,
            volume_L=phase["volume_L"],
            equipment=equipment,
        )
        if next_state.phases is not None:
            phases = next_state.phases.phases.copy()
            phases[target] = replace(phases[target], solvents=medium)
            next_state = next_state.replace(phases=PhaseLedger(phases))
        return self._account_removed_inventory(
            next_state,
            removed_volume_L=removed_volume,
            removed_product_mol=removed_product,
            removed_impurity_mol=removed_impurity,
        )

    def concentrate_phase(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        duration = float(np.clip(action_float(action, "duration_s", 300.0), 0.0, 3600.0))
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        target = selected_phase_id(state.phases) or "organic"
        phase = phase_ledger.setdefault(
            target,
            {
                "volume_L": state.volume_L,
                PHASE_PRODUCT_AMOUNT_KEY: self.phase_ledgers.phase_product_amount(state),
                "impurity_mol": 0.0,
                "solvent_loss": 0.0,
            },
        )
        phase_slice = PhaseSlice(
            product_mol=max(float(phase[PHASE_PRODUCT_AMOUNT_KEY]), 0.0),
            impurity_mol=max(float(phase["impurity_mol"]), 0.0),
            volume_L=max(float(phase["volume_L"]), 0.0),
            solvent_loss=max(float(phase.get("solvent_loss", 0.0)), 0.0),
        )
        metadata_updates: dict[str, Any] = {
            "concentration_model_id": "chemworld_vacuum_concentration_vnext",
            "concentration_skipped_empty_phase": not phase_slice.has_material,
        }
        concentration_extent = 0.0
        heat_duty_J = 0.0
        removed_volume = 0.0
        removed_product = 0.0
        removed_impurity = 0.0
        if phase_slice.has_material:
            result = run_vacuum_concentration(
                phase_slice,
                initial_temperature_K=state.temperature_K,
                duration_s=duration,
                solvents=state.phases.phases[target].solvents if state.phases else None,
                carrier_heat_capacity_J_L_K=self.world.rho_cp_J_per_L_K,
            )
            phase[PHASE_PRODUCT_AMOUNT_KEY] = result.liquid_amounts_mol.get("product", 0.0)
            phase["impurity_mol"] = result.liquid_amounts_mol.get("impurity", 0.0)
            phase["volume_L"] = result.final_equivalent_liquid_volume_L
            removed_metadata, removed_volume, removed_product, removed_impurity = (
                self._removed_inventory_summary(
                    state,
                    operation="concentrate",
                    inventories={
                        "concentrate_condensate": {
                            "product_mol": result.condensate_amounts_mol.get("product", 0.0),
                            "impurity_mol": result.condensate_amounts_mol.get("impurity", 0.0),
                            "volume_L": result.condensate_equivalent_liquid_volume_L,
                        },
                        "concentrate_vent": {
                            "product_mol": result.vent_amounts_mol.get("product", 0.0),
                            "impurity_mol": result.vent_amounts_mol.get("impurity", 0.0),
                            "volume_L": result.vent_equivalent_liquid_volume_L,
                        },
                    },
                )
            )
            concentration_extent = 1.0 - result.solvent_remaining_fraction
            phase["solvent_loss"] += concentration_extent
            heat_duty_J = result.heat_duty_J
            metadata_updates.update(
                {
                    "concentration_endpoint_met": result.endpoint_met,
                    "remaining_carriers_L": [
                        result.liquid_amounts_mol.get(f"carrier-{i}", 0.0) * 0.018 for i in range(4)
                    ],
                    "final_temperature_K": result.final_temperature_K,
                    "concentration_termination_reason": result.termination_reason,
                    "concentration_material_balance_error_mol": (result.material_balance_error_mol),
                    "concentration_energy_balance_error_J": result.energy_balance_error_J,
                    "concentration_target_recovery": result.target_recovery,
                    "concentration_warnings": list(result.warnings),
                    "concentration_provenance": list(result.provenance),
                    "concentration_provider_path": (
                        "chemworld.physchem.concentration_adapter_manifest."
                        "VacuumConcentrationProvider"
                    ),
                    **removed_metadata,
                }
            )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + duration / 3600.0 * 0.035,
            risk=min(1.0, state.ledger.risk + 0.015 * concentration_extent),
            energy_jacket_J=state.ledger.energy_jacket_J + heat_duty_J,
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="vacuum_concentrator",
            equipment_type="energy_limited_vacuum_concentrator",
            attached_vessel_id=state.vessel_id,
            status="concentrated",
            settings=metadata_updates,
        )
        next_state = self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            selected_phase=target,
            ledger=ledger,
            volume_L=phase["volume_L"],
            equipment=equipment,
        )
        if "remaining_carriers_L" in metadata_updates and next_state.phases is not None:
            phases = next_state.phases.phases.copy()
            phases[target] = replace(
                phases[target],
                solvents=SolventInventory(tuple(metadata_updates["remaining_carriers_L"])),
            )
            next_state = next_state.replace(
                phases=PhaseLedger(phases), temperature_K=metadata_updates["final_temperature_K"]
            )
        return self._account_removed_inventory(
            next_state,
            removed_volume_L=removed_volume,
            removed_product_mol=removed_product,
            removed_impurity_mol=removed_impurity,
        )

    def transfer_phase(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        fraction = float(np.clip(action_float(action, "transfer_fraction", 0.98), 0.0, 1.0))
        phase_ledger = self.phase_ledgers.phase_ledger(state)
        target = selected_phase_id(state.phases) or "organic"
        phase = phase_ledger.setdefault(
            target,
            {
                "volume_L": state.volume_L,
                PHASE_PRODUCT_AMOUNT_KEY: self.phase_ledgers.phase_product_amount(state),
                "impurity_mol": 0.0,
                "solvent_loss": 0.0,
            },
        )
        phase_slice = PhaseSlice(
            product_mol=max(float(phase[PHASE_PRODUCT_AMOUNT_KEY]), 0.0),
            impurity_mol=max(float(phase["impurity_mol"]), 0.0),
            volume_L=max(float(phase["volume_L"]), 0.0),
            solvent_loss=max(float(phase.get("solvent_loss", 0.0)), 0.0),
        )
        metadata_updates: dict[str, Any] = {
            "transfer_model_id": "chemworld_transfer_holdup_vnext",
            "transfer_skipped_empty_phase": not phase_slice.has_material,
        }
        delivered_fraction = 0.0
        removed_volume = 0.0
        removed_product = 0.0
        removed_impurity = 0.0
        if phase_slice.has_material:
            result = run_bounded_transfer(phase_slice, fraction=fraction)
            phase[PHASE_PRODUCT_AMOUNT_KEY] = result.target_delivered_amounts_mol.get(
                "product", 0.0
            )
            phase["impurity_mol"] = result.target_delivered_amounts_mol.get("impurity", 0.0)
            phase["volume_L"] = result.target_delivered_volume_L
            removed_metadata, removed_volume, removed_product, removed_impurity = (
                self._removed_inventory_summary(
                    state,
                    operation="transfer",
                    inventories={
                        "transfer_source_heel": {
                            "product_mol": result.source_remaining_amounts_mol.get("product", 0.0),
                            "impurity_mol": result.source_remaining_amounts_mol.get(
                                "impurity", 0.0
                            ),
                            "volume_L": result.source_remaining_volume_L,
                        },
                        "transfer_line_holdup": {
                            "product_mol": result.final_line_amounts_mol.get("product", 0.0),
                            "impurity_mol": result.final_line_amounts_mol.get("impurity", 0.0),
                            "volume_L": result.final_line_volume_L,
                        },
                    },
                )
            )
            delivered_fraction = result.overall_source_delivery_fraction
            phase["solvent_loss"] += 1.0 - delivered_fraction
            metadata_updates.update(
                {
                    "transfer_material_balance_error_mol": result.material_balance_error_mol,
                    "transfer_volume_balance_error_L": result.volume_balance_error_L,
                    "transfer_source_heel_volume_L": result.source_remaining_volume_L,
                    "transfer_line_holdup_volume_L": result.final_line_volume_L,
                    "transfer_delivery_fraction": delivered_fraction,
                    "transfer_warnings": list(result.warnings),
                    "transfer_provenance": list(result.provenance),
                    "transfer_provider_path": (
                        "chemworld.physchem.transfer_adapter_manifest.TransferUnitProvider"
                    ),
                    **removed_metadata,
                }
            )
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.01)
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="transfer_line",
            equipment_type="finite_holdup_transfer_line",
            attached_vessel_id=state.vessel_id,
            status="transferred",
            settings=metadata_updates,
        )
        next_state = self.phase_ledgers.with_phase_ledger(
            state,
            phase_ledger,
            selected_phase=target,
            ledger=ledger,
            volume_L=phase["volume_L"],
            equipment=equipment,
        )
        return self._account_removed_inventory(
            next_state,
            removed_volume_L=removed_volume,
            removed_product_mol=removed_product,
            removed_impurity_mol=removed_impurity,
        )


__all__ = ["ChemWorldPhaseSeparationServices"]
