"""Typed ledger helper functions for ChemWorld state."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
)

if TYPE_CHECKING:
    from chemworld.foundation.state import WorldState


def attached_device_record(state: WorldState, equipment_type: str) -> EquipmentRecord | None:
    """Find one device of the requested kind attached to this local vessel."""
    # WorldState materializes optional ledger inputs during construction.
    equipment = cast(EquipmentLedger, state.equipment)
    records = [r for r in equipment.equipment.values()
               if r.equipment_type == equipment_type and r.attached_vessel_id == state.vessel_id]
    if len(records) > 1:
        raise ValueError("a local system has one device of each declared kind")
    return records[0] if records else None


def has_phase_system(phases: PhaseLedger | None) -> bool:
    """Return whether a state has an explicit downstream phase system."""

    if phases is None:
        return False
    return any(phase_id != "reactor_liquid" for phase_id in phases.phases)


def phases_are_settled(phases: PhaseLedger | None) -> bool:
    """Return whether all non-reactor phases are marked settled."""

    if not has_phase_system(phases) or phases is None:
        return False
    downstream_phases = [
        phase for phase_id, phase in phases.phases.items() if phase_id != "reactor_liquid"
    ]
    return bool(downstream_phases) and all(phase.settled for phase in downstream_phases)


def selected_phase_id(phases: PhaseLedger | None) -> str | None:
    """Return the selected phase id, if one is marked in the typed ledger."""

    if phases is None:
        return None
    for phase_id, phase in phases.phases.items():
        if phase.selected:
            return phase_id
    return None


def scale_phase_ledger(
    phases: PhaseLedger | None,
    *,
    amount_factor: float,
    volume_factor: float | None = None,
) -> PhaseLedger | None:
    """Return a phase ledger scaled by destructive sampling or volume removal."""

    if phases is None:
        return None
    volume_factor = amount_factor if volume_factor is None else volume_factor
    return PhaseLedger(
        {
            phase_id: PhaseRecord(
                phase_id=phase.phase_id,
                vessel_id=phase.vessel_id,
                phase_type=phase.phase_type,
                volume_L=phase.volume_L * volume_factor,
                species_amounts_mol={
                    species_id: amount * amount_factor
                    for species_id, amount in phase.species_amounts_mol.items()
                },
                settled=phase.settled,
                selected=phase.selected,
                metadata=phase.metadata,
            )
            for phase_id, phase in phases.phases.items()
        }
    )


def equipment_settings(
    equipment: EquipmentLedger | None,
    equipment_id: str,
    *,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Copy settings, optionally projecting fields before copying nested data.

    Scalar admission checks must not traverse unrelated, growing execution
    histories. Both forms retain defensive-copy semantics for selected values.
    """

    if equipment is None or equipment_id not in equipment.equipment:
        return {}
    settings = equipment.equipment[equipment_id].settings
    if fields is not None:
        return deepcopy({key: settings[key] for key in fields if key in settings})
    return deepcopy(settings)


def equipment_setting_truth(
    equipment: EquipmentLedger | None,
    equipment_id: str,
    field: str,
) -> bool:
    """Read presence/truth without copying or exposing a mutable setting."""

    if equipment is None or equipment_id not in equipment.equipment:
        return False
    return bool(equipment.equipment[equipment_id].settings.get(field))


def equipment_status(
    equipment: EquipmentLedger | None,
    equipment_id: str,
) -> str | None:
    """Return a typed equipment record status, if present."""

    if equipment is None or equipment_id not in equipment.equipment:
        return None
    return equipment.equipment[equipment_id].status


INSTRUMENT_DEVICE_SELECTION = "_instrument_device_selection"


def instrument_equipment_id(instrument_id: str, *, state: WorldState | None = None) -> str:
    """Resolve an explicitly bound real instance, or the legacy single instance."""

    selection = {} if state is None else state.metadata.get(INSTRUMENT_DEVICE_SELECTION, {})
    if state is not None and selection.get("instrument") == instrument_id:
        identity = cast(str, selection.get("device_id"))
        record = cast(EquipmentLedger, state.equipment).equipment.get(identity)
        if (record is None or record.equipment_type != "instrument"
                or record.settings.get("lab_analyzer_instrument") != instrument_id):
            raise ValueError("selected analyzer does not own the declared instrument contract")
        return cast(str, identity)
    return f"instrument:{instrument_id}"


def instrument_completed(
    equipment: EquipmentLedger | None,
    instrument_id: str,
) -> bool:
    """Return whether an instrument record is marked completed."""

    return equipment_status(equipment, instrument_equipment_id(instrument_id)) == "completed"


def upsert_equipment_record(
    equipment: EquipmentLedger | None,
    *,
    equipment_id: str,
    equipment_type: str,
    attached_vessel_id: str,
    status: str = "configured",
    settings: dict[str, Any] | None = None,
    batch_reset: str | None = None,
    private_settings: tuple[str, ...] | None = None,
) -> EquipmentLedger:
    """Insert or update a typed equipment record with merged settings."""

    records = {} if equipment is None else equipment.equipment.copy()
    previous = records.get(equipment_id)
    merged_settings = {} if previous is None else previous.settings.copy()
    if settings:
        merged_settings.update(deepcopy(settings))
    records[equipment_id] = EquipmentRecord(
        equipment_id=equipment_id,
        equipment_type=equipment_type,
        attached_vessel_id=attached_vessel_id,
        status=status,
        settings=merged_settings,
        batch_reset=batch_reset or (previous.batch_reset if previous else "initial"),
        private_settings=(private_settings if private_settings is not None
                          else previous.private_settings if previous else ()),
    )
    return EquipmentLedger(records)


__all__ = [
    "attached_device_record",
    "equipment_settings",
    "equipment_status",
    "has_phase_system",
    "instrument_completed",
    "instrument_equipment_id",
    "phases_are_settled",
    "scale_phase_ledger",
    "selected_phase_id",
    "upsert_equipment_record",
]
