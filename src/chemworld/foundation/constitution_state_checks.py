"""State-invariant checks for the executable physical constitution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Number
from typing import Any, cast

from chemworld.foundation.constitution_reports import CheckResult
from chemworld.foundation.ledger_audit import (
    PHASE_PRIMARY_METADATA_KEYS,
    audit_ledger_single_source_of_truth,
)
from chemworld.foundation.state import WorldState
from chemworld.foundation.units import canonical_unit


def check_nonnegative(constitution: Any, state: WorldState) -> list[CheckResult]:
    values = {
        **{f"amount:{key}": value for key, value in state.species_amounts.items()},
        "volume_L": state.volume_L,
        "temperature_K": state.temperature_K,
        "pressure_Pa": state.pressure_Pa,
        "ledger.time_s": state.ledger.time_s,
        "ledger.cost": state.ledger.cost,
        "ledger.risk": state.ledger.risk,
        "ledger.sample_consumed_L": state.ledger.sample_consumed_L,
    }
    return [
        CheckResult(
            f"nonnegative:{key}",
            isfinite(value) and value >= -constitution.tolerance,
            f"{key}={value}",
            value=value,
            tolerance=constitution.tolerance,
        )
        for key, value in values.items()
    ]


def check_species_registry(constitution: Any, state: WorldState) -> CheckResult:
    registered = set(constitution.substances)
    present = set(state.species_amounts)
    for sample in state.samples.samples.values():
        present.update(sample.species_amounts_mol)
    if state.species is not None:
        present.update(state.species.species_roles)
        present.update(state.species.initial_amounts_mol)
    if state.phases is not None:
        for phase in state.phases.phases.values():
            present.update(phase.species_amounts_mol)
    unknown_count = len(present - registered)
    return CheckResult(
        "species_registry_membership",
        unknown_count == 0,
        "All material-ledger species must belong to the compiled mechanism registry.",
        value=float(unknown_count),
        tolerance=0.0,
    )


def check_units(constitution: Any) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for key, unit in constitution.required_state_units.items():
        try:
            canonical_unit(unit)
            passed = True
        except ValueError:
            passed = False
        checks.append(CheckResult(f"unit:{key}", passed, unit))
    return checks


def check_vessel_bounds(constitution: Any, state: WorldState) -> list[CheckResult]:
    vessel = None
    if state.vessels is not None:
        vessel = state.vessels.vessels.get(state.vessel_id)
    max_volume_L = (
        constitution.vessel.max_volume_L
        if vessel is None
        else vessel.max_volume_L
    )
    max_temperature_K = (
        constitution.vessel.max_temperature_K
        if vessel is None
        else vessel.max_temperature_K
    )
    max_pressure_Pa = (
        constitution.vessel.max_pressure_Pa
        if vessel is None
        else vessel.max_pressure_Pa
    )
    return [
        CheckResult(
            "vessel_volume_bound",
            state.volume_L <= max_volume_L + constitution.tolerance,
            f"volume={state.volume_L}",
            state.volume_L,
            constitution.tolerance,
        ),
        CheckResult(
            "vessel_temperature_bound",
            state.temperature_K
            <= max_temperature_K + constitution.tolerance,
            f"temperature_K={state.temperature_K}",
            state.temperature_K,
            constitution.tolerance,
        ),
        CheckResult(
            "vessel_pressure_bound",
            state.pressure_Pa <= max_pressure_Pa + constitution.tolerance,
            f"pressure_Pa={state.pressure_Pa}",
            state.pressure_Pa,
            constitution.tolerance,
        ),
    ]


def check_typed_ledgers(constitution: Any, state: WorldState) -> list[CheckResult]:
    initial_amount_metadata_keys = {
        key
        for key in state.metadata
        if key == "initial_reactant_mol"
        or (key.startswith("initial_") and key.endswith("_mol"))
    }
    checks: list[CheckResult] = [
        CheckResult(
            "state_numeric_values_finite",
            _numeric_tree_is_finite(state.to_dict(include_hidden=True)),
            "Every numeric value in the hidden state and typed ledgers must be finite.",
        ),
        CheckResult(
            "metadata_no_primary_phase_ledger",
            "phase_ledger" not in state.metadata,
            "Phase material state must live in typed PhaseLedger.",
        ),
        CheckResult(
            "metadata_no_primary_reactor_settings",
            constitution.primary_reactor_metadata_keys.isdisjoint(state.metadata),
            "Batch-reactor operation settings must live in typed EquipmentLedger.",
        ),
        CheckResult(
            "metadata_no_primary_phase_status",
            constitution.primary_phase_metadata_keys.isdisjoint(state.metadata),
            "Phase-system readiness, settled status, and selection must "
            "live in typed PhaseLedger.",
        ),
        CheckResult(
            "metadata_no_primary_vessel_bounds",
            constitution.primary_vessel_metadata_keys.isdisjoint(state.metadata),
            "Vessel operating bounds must live in typed VesselLedger.",
        ),
        CheckResult(
            "metadata_no_primary_instrument_status",
            constitution.primary_instrument_metadata_keys.isdisjoint(state.metadata),
            "Instrument completion state must live in typed EquipmentLedger.",
        ),
        CheckResult(
            "metadata_no_primary_crystallizer_seed_status",
            constitution.primary_crystallizer_metadata_keys.isdisjoint(state.metadata),
            "Crystallizer seed status and seed mass must live in typed EquipmentLedger.",
        ),
        CheckResult(
            "metadata_no_primary_crystallization_output",
            constitution.primary_crystallization_output_metadata_keys.isdisjoint(
                state.metadata
            ),
            "Crystallized material amounts must live in typed PhaseLedger.",
        ),
        CheckResult(
            "metadata_no_primary_distillation_output",
            constitution.primary_distillation_output_metadata_keys.isdisjoint(
                state.metadata
            ),
            "Distillate material amounts must live in typed PhaseLedger.",
        ),
        CheckResult(
            "metadata_no_primary_downstream_operation_status",
            constitution.primary_downstream_operation_metadata_keys.isdisjoint(
                state.metadata
            ),
            "Downstream operation status and diagnostics must live in typed "
            "EquipmentLedger or ProcessLedger.",
        ),
        CheckResult(
            "metadata_no_primary_process_metrics",
            constitution.primary_process_metric_metadata_keys.isdisjoint(state.metadata),
            "Derived process metrics must live in typed ProcessLedger.",
        ),
        CheckResult(
            "metadata_no_primary_initial_amounts",
            not initial_amount_metadata_keys,
            "Initial charged material amounts must live in typed SpeciesLedger.",
        ),
    ]
    if state.phases is not None:
        for phase_id, phase in state.phases.phases.items():
            checks.append(
                CheckResult(
                    f"phase_volume_nonnegative:{phase_id}",
                    isfinite(phase.volume_L)
                    and phase.volume_L >= -constitution.tolerance,
                    f"volume_L={phase.volume_L}",
                    phase.volume_L,
                    constitution.tolerance,
                )
            )
            for species_id, amount in phase.species_amounts_mol.items():
                checks.append(
                    CheckResult(
                        f"phase_amount_nonnegative:{phase_id}:{species_id}",
                        isfinite(amount) and amount >= -constitution.tolerance,
                        f"amount={amount}",
                        amount,
                        constitution.tolerance,
                    )
                )
            checks.append(
                CheckResult(
                    f"phase_metadata_no_primary_material_or_process_state:{phase_id}",
                    PHASE_PRIMARY_METADATA_KEYS.isdisjoint(phase.metadata),
                    "Phase material amounts must live in typed phase species "
                    "amounts, and process losses must live in typed ProcessLedger.",
                )
            )
            if state.vessels is not None:
                checks.append(
                    CheckResult(
                        f"phase_attached_vessel_exists:{phase_id}",
                        phase.vessel_id in state.vessels.vessels,
                        f"vessel_id={phase.vessel_id}",
                    )
                )
    if state.vessels is not None:
        known_phases = set(state.phases.phases) if state.phases is not None else set()
        for vessel_id, vessel in state.vessels.vessels.items():
            missing = sorted(set(vessel.phase_ids) - known_phases)
            checks.append(
                CheckResult(
                    f"vessel_phase_reverse_index:{vessel_id}",
                    not missing,
                    "" if not missing else f"missing phase ids: {missing}",
                )
            )
    if state.equipment is not None and state.vessels is not None:
        for equipment_id, equipment in state.equipment.equipment.items():
            checks.append(
                CheckResult(
                    f"equipment_attached_vessel_exists:{equipment_id}",
                    equipment.attached_vessel_id in state.vessels.vessels,
                    f"attached_vessel_id={equipment.attached_vessel_id}",
                )
            )
    return checks


def _numeric_tree_is_finite(value: Any) -> bool:
    if value is None or isinstance(value, str | bytes | bytearray | bool):
        return True
    if isinstance(value, Number):
        try:
            resolved = float(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            return False
        return isfinite(resolved)
    if isinstance(value, Mapping):
        return all(_numeric_tree_is_finite(child) for child in value.values())
    if isinstance(value, Sequence):
        return all(_numeric_tree_is_finite(child) for child in value)
    return True


def check_ledger_single_source(constitution: Any, state: WorldState) -> list[CheckResult]:
    primary_metadata_keys = (
        constitution.primary_reactor_metadata_keys
        | constitution.primary_phase_metadata_keys
        | constitution.primary_vessel_metadata_keys
        | constitution.primary_instrument_metadata_keys
        | constitution.primary_crystallizer_metadata_keys
        | constitution.primary_crystallization_output_metadata_keys
        | constitution.primary_distillation_output_metadata_keys
        | constitution.primary_downstream_operation_metadata_keys
        | constitution.primary_process_metric_metadata_keys
    )
    findings = audit_ledger_single_source_of_truth(
        state,
        tolerance=constitution.tolerance,
        primary_metadata_keys=primary_metadata_keys,
        phase_primary_metadata_keys=PHASE_PRIMARY_METADATA_KEYS,
    )
    return [
        CheckResult(
            finding.name,
            finding.passed,
            finding.message,
            finding.value,
            finding.tolerance,
        )
        for finding in findings
    ]


def check_risk_range(constitution: Any, state: WorldState) -> CheckResult:
    return CheckResult(
        "risk_range",
        0.0 <= state.ledger.risk <= 1.0,
        f"risk={state.ledger.risk}",
        state.ledger.risk,
        constitution.tolerance,
    )


__all__ = [
    "check_ledger_single_source",
    "check_nonnegative",
    "check_risk_range",
    "check_species_registry",
    "check_typed_ledgers",
    "check_units",
    "check_vessel_bounds",
]
