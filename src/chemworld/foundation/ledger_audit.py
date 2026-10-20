"""Audits for typed-ledger single-source-of-truth invariants."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from chemworld.foundation.state import WorldState

PHASE_PRIMARY_METADATA_KEYS = frozenset(
    {
        "product_mol",
        "impurity_mol",
        "solvent_loss",
    }
)

DEFAULT_PRIMARY_METADATA_KEYS = frozenset(
    {
        "catalyst",
        "solvent",
        "stirring_speed_rpm",
        "phase_ledger",
        "phase_system",
        "phase_settled",
        "selected_phase",
        "max_volume_L",
        "max_temperature_K",
        "max_pressure_Pa",
        "final_assay_done",
        "final_assay_time_s",
        "crystal_seeded",
        "crystal_seed_mass_g",
        "crystallization_active",
        "crystal_product_mol",
        "crystal_impurity_mol",
        "distillation_active",
        "distillate_product_mol",
        "distillate_impurity_mol",
        "crystals_filtered",
        "distillation_model",
        "distillation_kernel",
        "fraction_collected",
        "last_observation",
        "last_observed_mask",
        "pre_separation_product_mol",
        "purity",
        "recovery",
        "solvent_loss",
        "crystal_yield",
        "crystal_purity",
        "crystal_size",
        "distillate_purity",
        "distillate_recovery",
        "flow_conversion",
        "flow_campaign_time_s",
        "flow_throughput_mL",
        "electrochemical_model",
        "selective_product_yield",
        "electrochemical_conversion",
        "electrochemical_selectivity",
        "faradaic_efficiency",
        "energy_efficiency",
        "equilibrium_potential_V",
        "measured_potential_V",
        "interfacial_potential_V",
        "overpotential_V",
        "kinetic_current_A",
        "actual_current_A",
        "charge_C",
        "faradaic_charge_C",
        "electrical_work_J",
        "interfacial_work_J",
        "ohmic_loss_J",
        "electrolyte_resistance_ohm",
        "contact_resistance_ohm",
        "total_resistance_ohm",
        "uncompensated_voltage_drop_V",
        "voltage_window_exceeded",
    }
)


@dataclass(frozen=True)
class LedgerAuditFinding:
    """One ledger single-source-of-truth audit finding."""

    name: str
    passed: bool
    message: str = ""
    value: float | None = None
    tolerance: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "value": self.value,
            "tolerance": self.tolerance,
        }


def _max_mapping_error(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> tuple[float, str | None]:
    max_error = 0.0
    max_key: str | None = None
    for key in sorted(set(left) | set(right)):
        error = abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
        if error > max_error:
            max_error = error
            max_key = key
    return max_error, max_key


def _metadata_initial_amount_keys(metadata: Mapping[str, Any]) -> set[str]:
    return {
        key
        for key in metadata
        if key == "initial_reactant_mol"
        or (key.startswith("initial_") and key.endswith("_mol"))
    }


def _metadata_process_metric_keys(metadata: Mapping[str, Any]) -> set[str]:
    return {
        key
        for key in metadata
        if key.endswith("_purity")
        or key.endswith("_recovery")
        or key.endswith("_efficiency")
        or key.endswith("_current_A")
        or key.endswith("_charge_C")
        or key.endswith("_work_J")
        or key.endswith("_loss_J")
        or key.endswith("_resistance_ohm")
        or key.endswith("_potential_V")
    }


def _present(name: str, value: object | None) -> LedgerAuditFinding:
    return LedgerAuditFinding(
        name=f"ledger_present:{name}",
        passed=value is not None,
        message=f"{name} ledger must exist.",
    )


def _forbidden_metadata_findings(
    state: WorldState,
    *,
    primary_metadata_keys: Iterable[str],
    phase_primary_metadata_keys: Iterable[str],
) -> list[LedgerAuditFinding]:
    primary_keys = set(primary_metadata_keys)
    forbidden = sorted(
        (primary_keys & set(state.metadata))
        | _metadata_initial_amount_keys(state.metadata)
        | _metadata_process_metric_keys(state.metadata)
    )
    findings = [
        LedgerAuditFinding(
            "metadata_no_primary_structured_state",
            not forbidden,
            "" if not forbidden else f"forbidden metadata keys: {forbidden}",
        )
    ]
    if state.phases is not None:
        phase_forbidden = set(phase_primary_metadata_keys)
        for phase_id, phase in state.phases.phases.items():
            leaked = sorted(phase_forbidden & set(phase.metadata))
            findings.append(
                LedgerAuditFinding(
                    f"phase_metadata_no_primary_state:{phase_id}",
                    not leaked,
                    "" if not leaked else f"forbidden phase metadata keys: {leaked}",
                )
            )
    return findings


def audit_ledger_single_source_of_truth(
    state: WorldState,
    *,
    tolerance: float = 1.0e-8,
    primary_metadata_keys: Iterable[str] = DEFAULT_PRIMARY_METADATA_KEYS,
    phase_primary_metadata_keys: Iterable[str] = PHASE_PRIMARY_METADATA_KEYS,
) -> list[LedgerAuditFinding]:
    """Audit that typed ledgers, not metadata, are the primary state source.

    ``WorldState.species_amounts`` and ``WorldState.ledger`` are still kept as
    compatibility views for older scoring and reporting code. This audit treats
    ``PhaseLedger`` and ``ProcessLedger`` as the authoritative sources and
    verifies that the compatibility views remain synchronized.
    """

    findings: list[LedgerAuditFinding] = [
        _present("species", state.species),
        _present("phases", state.phases),
        _present("vessels", state.vessels),
        _present("equipment", state.equipment),
        _present("thermal", state.thermal),
        _present("process", state.process),
    ]

    if state.phases is None:
        findings.append(
            LedgerAuditFinding(
                "ledger_material_single_source",
                False,
                "PhaseLedger is required as the primary material ledger.",
            )
        )
    else:
        phase_totals = state.phases.total_amounts_mol()
        max_error, max_key = _max_mapping_error(phase_totals, state.species_amounts)
        findings.append(
            LedgerAuditFinding(
                "ledger_material_single_source",
                max_error <= tolerance,
                f"max |phase totals - compatibility species_amounts|={max_error:.3g}"
                + ("" if max_key is None else f" at {max_key}"),
                value=max_error,
                tolerance=tolerance,
            )
        )

    if state.process is None:
        findings.append(
            LedgerAuditFinding(
                "ledger_process_single_source",
                False,
                "ProcessLedger is required as the primary process ledger.",
            )
        )
    else:
        process_view = {
            "time_s": state.process.time_s,
            "cost": state.process.cost,
            "risk": state.process.risk,
            "sample_consumed_L": state.process.sample_consumed_L,
        }
        legacy_view = {
            "time_s": state.ledger.time_s,
            "cost": state.ledger.cost,
            "risk": state.ledger.risk,
            "sample_consumed_L": state.ledger.sample_consumed_L,
        }
        max_error, max_key = _max_mapping_error(process_view, legacy_view)
        findings.append(
            LedgerAuditFinding(
                "ledger_process_single_source",
                max_error <= tolerance,
                f"max |process ledger - compatibility ledger|={max_error:.3g}"
                + ("" if max_key is None else f" at {max_key}"),
                value=max_error,
                tolerance=tolerance,
            )
        )

    if state.phases is not None:
        phase_ids = set(state.phases.phases)
        vessel_ids = set(state.vessels.vessels) if state.vessels is not None else set()
        for phase_id, phase in state.phases.phases.items():
            findings.append(
                LedgerAuditFinding(
                    f"phase_attached_vessel_exists:{phase_id}",
                    phase.vessel_id in vessel_ids,
                    f"vessel_id={phase.vessel_id}",
                )
            )
            if state.vessels is not None and phase.vessel_id in state.vessels.vessels:
                vessel = state.vessels.vessels[phase.vessel_id]
                findings.append(
                    LedgerAuditFinding(
                        f"phase_listed_in_attached_vessel:{phase_id}",
                        phase_id in set(vessel.phase_ids),
                        f"vessel_id={phase.vessel_id}",
                    )
                )
            findings.append(
                LedgerAuditFinding(
                    f"phase_volume_finite_nonnegative:{phase_id}",
                    isfinite(phase.volume_L) and phase.volume_L >= -tolerance,
                    f"volume_L={phase.volume_L}",
                    value=phase.volume_L,
                    tolerance=tolerance,
                )
            )
            for species_id, amount in phase.species_amounts_mol.items():
                findings.append(
                    LedgerAuditFinding(
                        f"phase_amount_finite_nonnegative:{phase_id}:{species_id}",
                        isfinite(float(amount)) and float(amount) >= -tolerance,
                        f"amount={amount}",
                        value=float(amount),
                        tolerance=tolerance,
                    )
                )
        if state.vessels is not None:
            for vessel_id, vessel in state.vessels.vessels.items():
                missing = sorted(set(vessel.phase_ids) - phase_ids)
                findings.append(
                    LedgerAuditFinding(
                        f"vessel_phase_reverse_index:{vessel_id}",
                        not missing,
                        "" if not missing else f"missing phase ids: {missing}",
                    )
                )

    if state.equipment is not None and state.vessels is not None:
        vessel_ids = set(state.vessels.vessels)
        for equipment_id, equipment in sorted(state.equipment.equipment.items()):
            findings.append(
                LedgerAuditFinding(
                    f"equipment_attached_vessel_exists:{equipment_id}",
                    equipment.attached_vessel_id in vessel_ids,
                    f"attached_vessel_id={equipment.attached_vessel_id}",
                )
            )

    if state.thermal is not None:
        vessel_ids = set(state.vessels.vessels) if state.vessels is not None else set()
        for vessel_id, thermal in state.thermal.vessels.items():
            findings.append(
                LedgerAuditFinding(
                    f"thermal_attached_vessel_exists:{vessel_id}",
                    vessel_id in vessel_ids,
                    f"vessel_id={vessel_id}",
                )
            )
            for field_name in ("energy_jacket_J", "heat_reaction_J", "heat_loss_J"):
                value = float(getattr(thermal, field_name))
                findings.append(
                    LedgerAuditFinding(
                        f"thermal_value_finite:{vessel_id}:{field_name}",
                        isfinite(value),
                        f"{field_name}={value}",
                        value=value,
                        tolerance=tolerance,
                    )
                )

    findings.extend(
        _forbidden_metadata_findings(
            state,
            primary_metadata_keys=primary_metadata_keys,
            phase_primary_metadata_keys=phase_primary_metadata_keys,
        )
    )
    return findings


__all__ = [
    "DEFAULT_PRIMARY_METADATA_KEYS",
    "PHASE_PRIMARY_METADATA_KEYS",
    "LedgerAuditFinding",
    "audit_ledger_single_source_of_truth",
]
