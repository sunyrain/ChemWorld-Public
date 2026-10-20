"""Operation-record assembly for the transactional runtime."""

from __future__ import annotations

from typing import Any

from chemworld.foundation import (
    OperationRecord,
    PhysicalConstitution,
    WorldState,
    equipment_settings,
)
from chemworld.foundation.balances import TransitionBalance
from chemworld.world.operations import instrument_name

ELECTROCHEMICAL_SUMMARY_KEYS = (
    "equilibrium_potential_V",
    "measured_potential_V",
    "interfacial_potential_V",
    "overpotential_V",
    "actual_current_A",
    "charge_C",
    "faradaic_charge_C",
    "faradaic_efficiency",
    "electrical_work_J",
    "interfacial_work_J",
    "ohmic_loss_J",
    "total_resistance_ohm",
    "uncompensated_voltage_drop_V",
    "voltage_window_exceeded",
)


class ChemWorldOperationRecorder:
    """Build auditable operation records from pre/post transaction states."""

    def __init__(self, constitution: PhysicalConstitution, *, modules=None) -> None:
        self.constitution = constitution
        self.modules = modules

    def record(
        self,
        operation: str,
        before: WorldState,
        after: WorldState,
        preconditions: dict[str, bool],
        action: dict[str, Any] | None = None,
        *,
        constitution_checks: tuple[dict[str, object], ...] | None = None,
        material_balance: TransitionBalance | None = None,
    ) -> OperationRecord:
        action = action or {}
        # Transaction callers supply their exact candidate checks, including
        # failures. Rechecking the rolled-back state would erase the reason.
        checks = constitution_checks
        if checks is None:
            checks = tuple(
                check.to_dict()
                for check in (
                    *self.constitution.check_state(after).checks,
                    self.constitution.check_material_conservation(before, after),
                )
            )
        instrument = None
        measurement_cost = 0.0
        sample_consumed = after.ledger.sample_consumed_L - before.ledger.sample_consumed_L
        preconditions_passed = all(preconditions.values())
        if operation == "measure":
            instrument = instrument_name(action.get("instrument", "hplc"), modules=self.modules)
            if preconditions_passed:
                measurement_cost = after.ledger.cost - before.ledger.cost
        return OperationRecord(
            operation_type=operation,
            preconditions=preconditions,
            state_delta_summary=self.state_delta_summary(operation, before, after),
            constitution_checks=list(checks),
            instrument=instrument,
            terminal_assay=(
                instrument in self.constitution.instruments
                and self.constitution.instruments[instrument].requires_terminated
            ),
            measurement_cost=measurement_cost,
            sample_consumed_L=sample_consumed,
            material_balance=material_balance or TransitionBalance(),
        )

    @staticmethod
    def state_delta_summary(
        operation: str,
        before: WorldState,
        after: WorldState,
    ) -> dict[str, float]:
        state_delta_summary = {
            "delta_time_s": after.ledger.time_s - before.ledger.time_s,
            "delta_cost": after.ledger.cost - before.ledger.cost,
            "delta_risk": after.ledger.risk - before.ledger.risk,
            "delta_temperature_K": after.temperature_K - before.temperature_K,
            "delta_volume_L": after.volume_L - before.volume_L,
        }
        if operation == "set_potential":
            settings = equipment_settings(after.equipment, "electrochemical_cell")
            for source, target in (
                ("potential_V", "configured_potential_V"),
                ("current_mA", "configured_current_mA"),
            ):
                if source in settings:
                    state_delta_summary[target] = float(settings[source])
        if operation == "electrolyze":
            process_metrics = {} if after.process is None else after.process.metrics
            for key in ELECTROCHEMICAL_SUMMARY_KEYS:
                if key in process_metrics:
                    state_delta_summary[key] = float(process_metrics[key])
        return state_delta_summary


__all__ = [
    "ELECTROCHEMICAL_SUMMARY_KEYS",
    "ChemWorldOperationRecorder",
]
