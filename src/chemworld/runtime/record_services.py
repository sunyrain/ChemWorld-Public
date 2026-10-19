"""Operation-record assembly for the transactional runtime."""

from __future__ import annotations

from typing import Any

from chemworld.foundation import (
    OperationRecord,
    PhysicalConstitution,
    WorldState,
    equipment_settings,
)
from chemworld.foundation.constitution import CheckResult
from chemworld.world.operations import instrument_name

MATERIAL_DELTA_ALLOWED_OPERATIONS = frozenset(
    {
        "add_reagent",
        "add_component",
        "add_catalyst",
        "add_solvent",
        "sample",
        "measure",
        "add_phase",
        "add_extractant",
        "seed_crystals",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
    }
)

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
THERMAL_TRANSITION_OPERATIONS = frozenset({"quench", "cool_crystallize"})


class ChemWorldOperationRecorder:
    """Build auditable operation records from pre/post transaction states."""

    def __init__(self, constitution: PhysicalConstitution) -> None:
        self.constitution = constitution

    def record(
        self,
        operation: str,
        before: WorldState,
        after: WorldState,
        preconditions: dict[str, bool],
        action: dict[str, Any] | None = None,
    ) -> OperationRecord:
        action = action or {}
        checks = self._constitution_checks(operation, before, after)
        instrument = None
        measurement_cost = 0.0
        sample_consumed = after.ledger.sample_consumed_L - before.ledger.sample_consumed_L
        preconditions_passed = all(preconditions.values())
        if operation == "measure":
            instrument = instrument_name(action.get("instrument", "hplc"))
            if preconditions_passed:
                measurement_cost = after.ledger.cost - before.ledger.cost
        return OperationRecord(
            operation_type=operation,
            preconditions=preconditions,
            state_delta_summary=self._state_delta_summary(operation, before, after),
            constitution_checks=[check.to_dict() for check in checks],
            instrument=instrument,
            measurement_cost=measurement_cost,
            sample_consumed_L=sample_consumed,
        )

    def _constitution_checks(
        self,
        operation: str,
        before: WorldState,
        after: WorldState,
    ) -> list[CheckResult]:
        report = self.constitution.check_state(after)
        material_check = self.constitution.check_material_conservation(before, after)
        if operation in MATERIAL_DELTA_ALLOWED_OPERATIONS:
            material_check = CheckResult(
                "material_conservation",
                True,
                "declared open-system material input/output operation",
                value=0.0,
                tolerance=self.constitution.tolerance,
            )
        checks = [*report.checks, material_check]
        if operation in THERMAL_TRANSITION_OPERATIONS:
            transition = after.metadata.get("last_energy_transition", {})
            residual = (
                float(transition.get("energy_balance_residual_J", float("inf")))
                if isinstance(transition, dict) and transition.get("operation") == operation
                else float("inf")
            )
            checks.append(
                CheckResult(
                    "operation_energy_conservation",
                    abs(residual) <= 1.0e-8,
                    "signed sensible, jacket, environmental, and phase-change heat close",
                    value=residual,
                    tolerance=1.0e-8,
                )
            )
        return checks

    @staticmethod
    def _state_delta_summary(
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
    "MATERIAL_DELTA_ALLOWED_OPERATIONS",
    "THERMAL_TRANSITION_OPERATIONS",
    "ChemWorldOperationRecorder",
]
