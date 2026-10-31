"""Reaction advancement and thermal-risk services for the transactional runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    process_with_metrics,
    selected_phase_id,
    upsert_equipment_record,
)
from chemworld.foundation.state import PhaseLedger, PhaseRecord
from chemworld.physchem.reactor_shared import JacketTemperatureProgram
from chemworld.runtime.action_values import bounded_action_float
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.full_process_contract import (
    seed_provenance_active,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.parameters import ChemWorldParameters
from chemworld.world.reaction_kernel import ReactionValidityError, integrate_compiled_reaction_ode
from chemworld.world.thermal_kernel import account_temperature_transition, pressure_and_risk


class ChemWorldReactionThermalServices:
    """Advance reaction state and apply thermal pressure/risk ledgers."""

    def __init__(
        self,
        world: ChemWorldParameters,
        species_view: MechanismSpeciesView,
        *,
        composition=None,
        modules=None,
    ) -> None:
        self.world = world
        self.species_view = species_view
        from chemworld.world.domain_modules import DomainModuleRegistry

        self.modules = modules if modules is not None else DomainModuleRegistry.default()
        from chemworld.runtime.advance_paths import PhysicalAdvanceCoordinator

        self.advancer = PhysicalAdvanceCoordinator(
            self, modules=self.modules, composition=composition
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
        ledger = advance_ledger(
            state.ledger,
            duration_s,
            cost_delta=(
                0.01 + duration_s / 3600.0 * 0.015 + abs(thermal.jacket_energy_J) / 250_000.0
            ),
            energy_deltas={
                "energy_jacket_J": thermal.jacket_energy_J,
                "heat_reaction_J": thermal.phase_change_heat_J,
                "heat_loss_J": thermal.heat_loss_J,
            },
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
    ) -> ModelProviderResult:
        return self.advancer.batch(state, action, heat=heat, jacket_program=jacket_program)

    def _integrate_legacy(
        self,
        state: WorldState,
        action: dict[str, Any],
        *,
        heat: bool,
        jacket_program: JacketTemperatureProgram | None = None,
    ) -> ModelProviderResult:
        duration = bounded_action_float(
            action,
            "duration_s",
            600.0,
            minimum=1.0e-6 if jacket_program is not None else 1.0,
            maximum=14_400.0,
        )
        target_temperature = bounded_action_float(
            action,
            "target_temperature_K",
            state.temperature_K,
            minimum=250.0,
            maximum=520.0,
        )
        reactor_settings = equipment_settings(state.equipment, "batch_reactor")
        stirring_speed = bounded_action_float(
            action,
            "stirring_speed_rpm",
            float(reactor_settings.get("stirring_speed_rpm", 600.0)),
            minimum=100.0,
            maximum=1200.0,
        )
        if state.quenched and jacket_program is None:
            working_state, dissolved = state, 0.0
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
                    ledger=advance_ledger(
                        held_state.ledger,
                        0.0,
                        energy_deltas={"energy_jacket_J": energy, "heat_reaction_J": energy},
                    )
                )
            return ModelProviderResult(
                outputs={},
                diagnostics={},
                state=held_state,
                actual_duration_s=duration,
            )
        working_state, redissolved_target = state, 0.0
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
        except ReactionValidityError as exc:
            return ModelProviderResult(
                outputs={},
                diagnostics=exc.diagnostics,
                outcome="rejected",
                success=False,
                failure_reason="reaction_validity_domain_exceeded",
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                outputs={},
                diagnostics={"solver_error": str(exc), "error_type": type(exc).__name__},
                outcome="numerical_failure",
                success=False,
                failure_reason="reaction_solver_failed",
            )
        if result is None:
            return ModelProviderResult(
                outputs={},
                diagnostics={"solver_returned_no_result": True},
                outcome="numerical_failure",
                success=False,
                failure_reason="reaction_solver_failed",
            )
        dissolution_heat_J = 20_000.0 * redissolved_target
        ledger = advance_ledger(
            working_state.ledger,
            result.duration_s,
            cost_delta=result.cost_delta,
            energy_deltas={
                "energy_jacket_J": result.energy_jacket_J + dissolution_heat_J,
                "heat_reaction_J": result.heat_reaction_J + dissolution_heat_J,
                "heat_loss_J": result.heat_loss_J,
            },
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
            phase_records = working_state.phases.phases.copy()
            phase_records[active_phase_id] = replace(
                phase_records[active_phase_id],
                species_amounts_mol=result.species_amounts,
            )
            phases = PhaseLedger(phase_records)
            species_amounts = phases.total_amounts_mol()
        candidate = working_state.replace(
            species_amounts=species_amounts,
            temperature_K=result.temperature_K,
            ledger=ledger,
            equipment=equipment,
            process=process,
            phases=phases,
        )
        return ModelProviderResult(
            outputs={},
            state=candidate,
            actual_duration_s=result.duration_s,
            outcome="success" if result.termination_reason == "duration_reached" else "stopped",
            diagnostics={"termination_reason": result.termination_reason},
        )

    def with_risk_and_pressure(self, state: WorldState) -> WorldState:
        selected = self.advancer.strategy(state)
        if selected is None:
            return state
        reconcile = getattr(selected[1], "reconcile_pressure", None)
        return state if reconcile is None else reconcile(state)

    def _legacy_risk_and_pressure(self, state: WorldState) -> WorldState:
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
