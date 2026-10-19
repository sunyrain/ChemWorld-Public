"""Reaction advancement and thermal-risk services for the transactional runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    process_with_metrics,
    selected_phase_id,
    upsert_equipment_record,
)
from chemworld.foundation.state import PhaseLedger, PhaseRecord
from chemworld.physchem.crystallization_units import SolubilityCurveSpec
from chemworld.physchem.reactor_shared import JacketTemperatureProgram
from chemworld.runtime.full_process_contract import (
    FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    FULL_PROCESS_SEED_CONTRACT,
    FULL_PROCESS_THERMAL_CONTRACT,
    population_active,
    population_settings,
    seed_provenance_active,
    shrink_population,
)
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.parameters import ChemWorldParameters
from chemworld.world.reaction_kernel import integrate_compiled_reaction_ode
from chemworld.world.thermal_kernel import account_temperature_transition, pressure_and_risk


def _action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def _bounded_action_float(
    action: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _action_float(action, key, default)
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be finite and in [{minimum}, {maximum}]")
    return value


class ChemWorldReactionThermalServices:
    """Advance reaction state and apply thermal pressure/risk ledgers."""

    def __init__(
        self,
        world: ChemWorldParameters,
        species_view: MechanismSpeciesView,
    ) -> None:
        self.world = world
        self.species_view = species_view

    def _redissolve_crystals_for_heating(
        self,
        state: WorldState,
        target_temperature_K: float,
    ) -> tuple[WorldState, float]:
        """Equilibrate a slurry toward the hotter solubility boundary."""

        if (
            target_temperature_K <= state.temperature_K
            or state.phases is None
            or "mother_liquor" not in state.phases.phases
            or "solid" not in state.phases.phases
        ):
            return state, 0.0
        phases = state.phases.phases.copy()
        liquor = phases["mother_liquor"]
        solid = phases["solid"]
        target_species = self.species_view.primary_target_species
        impurity_species = self.species_view.primary_impurity_species
        solid_target = max(
            float(solid.species_amounts_mol.get(target_species, 0.0)),
            0.0,
        )
        if solid_target <= 1.0e-12:
            return state, 0.0
        from chemworld.world.mixtures import crystallization_medium_modifier

        solubility_multiplier = crystallization_medium_modifier(state, self.world, "solubility")
        curve = SolubilityCurveSpec(
            model_id="runtime_vanthoff_material_solubility_v2",
            reference_solubility_mol_L=(
                self.world.crystallization_reference_solubility_mol_L
                * self.world.domain_parameter("crystallization_solubility_multiplier")
                * solubility_multiplier
            ),
            reference_temperature_K=298.15,
            dissolution_enthalpy_J_mol=20_000.0,
            minimum_temperature_K=250.0,
            maximum_temperature_K=430.0,
            provenance_id="chemworld-world-law-v0.2-solubility-policy",
        )
        equilibrium_capacity = (
            curve.solubility_mol_per_l(min(target_temperature_K, 430.0)) * liquor.volume_L
        )
        dissolved_target = max(
            float(liquor.species_amounts_mol.get(target_species, 0.0)),
            0.0,
        )
        redissolved_target = min(
            solid_target,
            max(equilibrium_capacity - dissolved_target, 0.0),
        )
        if redissolved_target <= 1.0e-12:
            return state, 0.0
        solid_impurity = max(
            float(solid.species_amounts_mol.get(impurity_species, 0.0)),
            0.0,
        )
        redissolved_impurity = solid_impurity * redissolved_target / solid_target
        liquor_amounts = liquor.species_amounts_mol.copy()
        solid_amounts = solid.species_amounts_mol.copy()
        liquor_amounts[target_species] = dissolved_target + redissolved_target
        solid_amounts[target_species] = solid_target - redissolved_target
        liquor_amounts[impurity_species] = (
            float(liquor_amounts.get(impurity_species, 0.0)) + redissolved_impurity
        )
        solid_amounts[impurity_species] = solid_impurity - redissolved_impurity
        phases["mother_liquor"] = replace(liquor, species_amounts_mol=liquor_amounts)
        phases["solid"] = replace(solid, species_amounts_mol=solid_amounts)
        process_metrics = {} if state.process is None else state.process.metrics
        process = process_with_metrics(
            state.process,
            crystal_redissolved_target_mol=(
                float(process_metrics.get("crystal_redissolved_target_mol", 0.0))
                + redissolved_target
            ),
        )
        crystallizer_settings = equipment_settings(state.equipment, "crystallizer")
        dissolution_history = list(crystallizer_settings.get("dissolution_history", ()))
        dissolution_history.append(
            {
                "initial_temperature_K": state.temperature_K,
                "target_temperature_K": target_temperature_K,
                "redissolved_target_mol": redissolved_target,
                "redissolved_impurity_mol": redissolved_impurity,
            }
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="crystallizer",
            equipment_type="crystallizer",
            attached_vessel_id=state.vessel_id,
            status="partially_redissolved",
            settings={"dissolution_history": dissolution_history},
        )
        if population_active(state):
            remaining = (solid_target - redissolved_target) / solid_target
            cohorts = shrink_population(
                crystallizer_settings.get("population_cohorts", ()), remaining
            )
            equipment = upsert_equipment_record(
                equipment,
                equipment_id="crystallizer",
                equipment_type="crystallizer",
                attached_vessel_id=state.vessel_id,
                settings={
                    **population_settings(cohorts),
                    "seed_target_mol": float(crystallizer_settings.get("seed_target_mol", 0.0))
                    * remaining,
                    **(
                        {
                            "dissolved_seed_target_mol": float(
                                crystallizer_settings.get("dissolved_seed_target_mol", 0.0)
                            )
                            + float(crystallizer_settings.get("seed_target_mol", 0.0))
                            * (1 - remaining)
                        }
                        if seed_provenance_active(state)
                        else {}
                    ),
                },
            )
        return (
            state.replace(
                phases=PhaseLedger(phases),
                process=process,
                equipment=equipment,
            ),
            redissolved_target,
        )

    def _quenched_thermal_hold(
        self,
        state: WorldState,
        *,
        duration_s: float,
        target_temperature_K: float,
        heat: bool,
        phase_change_heat_J: float = 0.0,
    ) -> WorldState:
        """Apply thermal time after quench without advancing reaction chemistry."""

        thermal = account_temperature_transition(
            state=state,
            world=self.world,
            final_temperature_K=target_temperature_K,
            duration_s=duration_s,
            phase_change_heat_J=phase_change_heat_J,
        )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration_s,
            cost=(
                state.ledger.cost
                + 0.01
                + duration_s / 3600.0 * 0.015
                + abs(thermal.jacket_energy_J) / 250_000.0
            ),
            energy_jacket_J=state.ledger.energy_jacket_J + thermal.jacket_energy_J,
            heat_reaction_J=state.ledger.heat_reaction_J + thermal.phase_change_heat_J,
            heat_loss_J=state.ledger.heat_loss_J + thermal.heat_loss_J,
        )
        reactor_settings = equipment_settings(state.equipment, "batch_reactor")
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id=state.vessel_id,
            status="quenched",
            settings={
                "reaction_stopped": True,
                "last_operation": "heat" if heat else "wait",
                "last_operation_semantic": "thermal_hold_after_quench",
                "reaction_chemistry_advanced": False,
                "reaction_advance_index": int(reactor_settings.get("reaction_advance_index", 0)),
            },
        )
        process_metrics = {} if state.process is None else state.process.metrics
        process = process_with_metrics(
            state.process,
            reaction_chemistry_stopped=1.0,
            quenched_hold_cumulative_time_s=(
                float(process_metrics.get("quenched_hold_cumulative_time_s", 0.0)) + duration_s
            ),
        )
        return state.replace(
            temperature_K=target_temperature_K,
            ledger=ledger,
            equipment=equipment,
            process=process,
            metadata={
                **state.metadata,
                "last_energy_transition": {
                    "operation": "heat" if heat else "wait",
                    "reaction_chemistry_advanced": False,
                    **thermal.to_dict(),
                },
            },
        )

    def integrate(
        self,
        state: WorldState,
        action: dict[str, Any],
        *,
        heat: bool,
        jacket_program: JacketTemperatureProgram | None = None,
    ) -> WorldState:
        duration = _bounded_action_float(
            action,
            "duration_s",
            600.0,
            minimum=1.0e-6 if jacket_program is not None else 1.0,
            maximum=14_400.0,
        )
        target_temperature = _bounded_action_float(
            action,
            "target_temperature_K",
            state.temperature_K,
            minimum=250.0,
            maximum=520.0,
        )
        reactor_settings = equipment_settings(state.equipment, "batch_reactor")
        stirring_speed = _bounded_action_float(
            action,
            "stirring_speed_rpm",
            float(reactor_settings.get("stirring_speed_rpm", 600.0)),
            minimum=100.0,
            maximum=1200.0,
        )
        if state.quenched and jacket_program is None:
            working_state, dissolved = (
                self._redissolve_crystals_for_heating(state, target_temperature)
                if heat
                and state.metadata.get("full_process_contract_id")
                in {
                    FULL_PROCESS_THERMAL_CONTRACT,
                    FULL_PROCESS_SEED_CONTRACT,
                    FULL_PROCESS_FREE_RESEARCH_CONTRACT,
                }
                else (state, 0.0)
            )
            held_state = self._quenched_thermal_hold(
                working_state,
                duration_s=duration,
                target_temperature_K=target_temperature,
                heat=heat,
                phase_change_heat_J=(20000.0 * dissolved if seed_provenance_active(state) else 0.0),
            )
            if dissolved > 0 and not seed_provenance_active(state):
                energy = 20000.0 * dissolved
                held_state = held_state.replace(
                    ledger=held_state.ledger.with_updates(
                        energy_jacket_J=held_state.ledger.energy_jacket_J + energy,
                        heat_reaction_J=held_state.ledger.heat_reaction_J + energy,
                    )
                )
            return held_state
        working_state, redissolved_target = (
            self._redissolve_crystals_for_heating(state, target_temperature)
            if heat
            else (state, 0.0)
        )
        active_phase_id: str | None = None
        active_phase: PhaseRecord | None = None
        if working_state.phases is not None:
            if "mother_liquor" in working_state.phases.phases:
                active_phase_id = "mother_liquor"
            else:
                active_phase_id = selected_phase_id(working_state.phases)
            if active_phase_id is not None:
                active_phase = working_state.phases.phases[active_phase_id]
        kernel_state = working_state
        if active_phase is not None and active_phase_id != "reactor_liquid":
            kernel_state = working_state.replace(
                species_amounts=active_phase.species_amounts_mol,
                volume_L=active_phase.volume_L,
                phases=None,
            )
        compiled_mechanism = self.species_view.mechanism
        if compiled_mechanism is None:
            raise RuntimeError("Reaction advancement requires a compiled mechanism")
        try:
            result = integrate_compiled_reaction_ode(
                state=kernel_state,
                world=self.world,
                compiled_mechanism=compiled_mechanism,
                duration_s=duration,
                target_temperature_K=target_temperature,
                heat=heat,
                stirring_speed_rpm=stirring_speed,
                jacket_program=jacket_program,
            )
        except (RuntimeError, ValueError):
            # The kernel transaction manager owns rollback.  Return a candidate
            # that deterministically fails the vessel-temperature constitution;
            # no physical or resource mutation from the failed model is kept.
            maximum_temperature = 470.0
            if (
                working_state.vessels is not None
                and state.vessel_id in working_state.vessels.vessels
            ):
                maximum_temperature = working_state.vessels.vessels[
                    state.vessel_id
                ].max_temperature_K
            return working_state.replace(temperature_K=maximum_temperature + 1.0)
        if result is None:
            raise RuntimeError("positive-duration reaction advance returned no result")
        dissolution_heat_J = 20_000.0 * redissolved_target
        ledger = working_state.ledger.with_updates(
            time_s=working_state.ledger.time_s + result.duration_s,
            cost=working_state.ledger.cost + result.cost_delta,
            energy_jacket_J=(
                working_state.ledger.energy_jacket_J + result.energy_jacket_J + dissolution_heat_J
            ),
            heat_reaction_J=(
                working_state.ledger.heat_reaction_J + result.heat_reaction_J + dissolution_heat_J
            ),
            heat_loss_J=working_state.ledger.heat_loss_J + result.heat_loss_J,
        )
        advance_index = int(reactor_settings.get("reaction_advance_index", 0)) + int(
            not state.quenched
        )
        operation_type = "heat" if heat else "wait"
        equipment = upsert_equipment_record(
            working_state.equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id=state.vessel_id,
            status="advanced",
            settings={
                "stirring_speed_rpm": result.stirring_speed_rpm,
                "reaction_advance_index": advance_index,
                "last_operation": operation_type,
                "last_operation_semantic": "advance",
                "repeat_semantic": "each committed repeat advances additional physical time",
                "runtime_provider_id": result.provider_id,
                "reaction_runtime_model_id": result.model_id,
                "reaction_model_id": result.provenance["reaction_network_model_id"],
                "reactor_model_id": result.provenance["reactor_model_id"],
                "reaction_network_id": result.provenance["network_id"],
                "mechanism_id": result.provenance["mechanism_id"],
                "mechanism_version": result.provenance["mechanism_version"],
                "mechanism_hash": result.provenance["mechanism_hash"],
                "solver_diagnostic": result.solver_diagnostic,
                "reactor_diagnostic": result.reactor_diagnostic,
                "termination_reason": result.termination_reason,
                "material_balance_error_mol": result.material_balance_error_mol,
                "maximum_conservation_drift_mol": result.maximum_conservation_drift_mol,
                "element_inventory_residuals_mol": result.element_inventory_residuals_mol,
                "charge_inventory_residual_mol": result.charge_inventory_residual_mol,
                "energy_balance_residual_J": result.energy_balance_residual_J,
                "trajectory_digest": result.trajectory_digest,
            },
        )
        process = process_with_metrics(
            working_state.process,
            reaction_advance_count=float(advance_index),
            reaction_cumulative_time_s=(
                float(
                    0.0
                    if working_state.process is None
                    else working_state.process.metrics.get("reaction_cumulative_time_s", 0.0)
                )
                + (0.0 if state.quenched else result.duration_s)
            ),
        )
        phases = working_state.phases
        species_amounts = result.species_amounts
        if active_phase is not None and active_phase_id != "reactor_liquid":
            assert active_phase_id is not None
            assert working_state.phases is not None
            phase_records = working_state.phases.phases.copy()
            phase_records[active_phase_id] = replace(
                phase_records[active_phase_id],
                species_amounts_mol=result.species_amounts,
            )
            phases = PhaseLedger(phase_records)
            species_amounts = phases.total_amounts_mol()
        return working_state.replace(
            species_amounts=species_amounts,
            temperature_K=result.temperature_K,
            ledger=ledger,
            equipment=equipment,
            process=process,
            phases=phases,
        )

    def with_risk_and_pressure(self, state: WorldState) -> WorldState:
        flow_settings = equipment_settings(state.equipment, "flow_reactor")
        pressure_override = flow_settings.get("outlet_pressure_Pa")
        controller = equipment_settings(state.equipment, "process_controller")
        if controller:
            from chemworld.foundation.gas import GasBoundary

            pressure_override = GasBoundary.from_dict(controller["gas"]).pressure_pa
        pressure, risk = pressure_and_risk(
            state=state,
            solvent_risks=self.world.solvent_risks,
            pressure_override_Pa=(
                float(pressure_override) if isinstance(pressure_override, int | float) else None
            ),
        )
        return state.replace(pressure_Pa=pressure, ledger=state.ledger.with_updates(risk=risk))


__all__ = ["ChemWorldReactionThermalServices"]
