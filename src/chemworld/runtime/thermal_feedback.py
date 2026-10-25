"""Original heater-owned feedback memory; no inventory, solver or control clock."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.coupled_system import CoupledState, CoupledSystem
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.thermal_feedback import (
    ThermalFeedbackSpec,
    augment_thermal_feedback,
    feedback_key,
    thermal_feedback_point,
)

STATE_TYPE = "thermal-feedback-memory-v1"


def feedback_slot_spec(equipment_id: str) -> StateSlotSpec:
    return StateSlotSpec(
        f"{equipment_id}:thermal-feedback",
        "equipment",
        equipment_id,
        STATE_TYPE,
        (
            StateFieldSpec("integral_K", "K", mixing="equal", minimum=None, public=True),
            StateFieldSpec(
                "filtered_temperature_K", "K", mixing="equal", minimum=1e-9, public=True
            ),
        ),
        mechanism_id="finite-thermal-feedback-v1",
        batch_reset="preserve",
    )


def install_thermal_feedback(
    state: WorldState,
    equipment_id: str,
    specification: ThermalFeedbackSpec,
    *,
    initial_integral_K: float,
    initial_filtered_temperature_K: float,
) -> WorldState:
    """Build initial declaration and one explicit device-owned memory slot."""
    if state.equipment is None:
        raise ValueError("feedback installation requires original equipment")
    record = state.equipment.equipment[equipment_id]
    if (
        record.equipment_type != "thermal_device"
        or record.attached_vessel_id != state.vessel_id
        or "thermal_feedback" in record.settings
        or any(
            s.spec.owner_id == equipment_id and s.spec.state_type == STATE_TYPE
            for s in state.state_slots.slots
        )
    ):
        raise ValueError("install feedback once on the original connected thermal device")
    device = ThermalDeviceSpec.from_dict(record.settings["thermal_device"])
    thermal_feedback_point(
        specification,
        device,
        sample_temperature_K=state.temperature_K,
        integral_K=initial_integral_K,
        filtered_temperature_K=initial_filtered_temperature_K,
    )
    slot = StateSlot(
        feedback_slot_spec(equipment_id),
        state.vessel_id,
        {
            "integral_K": initial_integral_K,
            "filtered_temperature_K": initial_filtered_temperature_K,
        },
    )
    records = dict(state.equipment.equipment)
    records[equipment_id] = replace(
        record,
        settings={
            **record.settings,
            "thermal_feedback": specification.to_dict(),
        },
    )
    return state.replace(
        equipment=EquipmentLedger(records),
        state_slots=StateSlots((*state.state_slots.slots, slot)),
    )


def _binding(
    state: WorldState,
    physics: Any = None,
) -> tuple[EquipmentRecord, ThermalFeedbackSpec, ThermalDeviceSpec, StateSlot] | None:
    records = {} if state.equipment is None else state.equipment.equipment
    for slot in state.state_slots.slots:
        if slot.spec.state_type == STATE_TYPE and (
            slot.spec.owner_id not in records
            or "thermal_feedback" not in records[slot.spec.owner_id].settings
        ):
            raise ValueError("feedback memory has no original heater declaration")
    if physics is not None:
        originals = {
            key: record
            for source in physics.initial_states.values()
            for key, record in source.equipment.equipment.items()
        }
        for key, record in records.items():
            current = record.settings.get("thermal_feedback")
            previous = (
                None if key not in originals else originals[key].settings.get("thermal_feedback")
            )
            # Explicit equipment replacement introduces a new identity and
            # its own declaration/initial slot through the existing lifecycle.
            # Only an original installed identity is bound to its old model.
            if key in originals and current != previous:
                raise ValueError(
                    "thermal feedback model differs from original equipment declaration"
                )
    installed = [
        r
        for r in records.values()
        if r.attached_vessel_id == state.vessel_id
        and r.status != "parked"
        and "thermal_feedback" in r.settings
    ]
    if not installed:
        return None
    if len(installed) != 1:
        raise ValueError("local thermal feedback requires exactly one active controller")
    record = installed[0]
    if record.equipment_type != "thermal_device":
        raise ValueError("thermal feedback must belong to the original heater")
    slots = [
        s
        for s in state.state_slots.slots
        if s.spec.owner_id == record.equipment_id and s.spec.state_type == STATE_TYPE
    ]
    if (
        len(slots) != 1
        or slots[0].spec != feedback_slot_spec(record.equipment_id)
        or slots[0].location_id != state.vessel_id
    ):
        raise ValueError("heater feedback requires its unique declared local memory slot")
    spec = ThermalFeedbackSpec.from_dict(record.settings["thermal_feedback"])
    device = ThermalDeviceSpec.from_dict(record.settings["thermal_device"])
    if physics is not None:
        model = physics.models_of("thermal")[state.vessel_id]
        if (
            not model.minimum_temperature_K
            <= spec.minimum_command_K
            < spec.maximum_command_K
            <= model.maximum_temperature_K
        ):
            raise ValueError("feedback command interval leaves the original thermal domain")
    return record, spec, device, slots[0]


def with_thermal_feedback(
    state: WorldState,
    system: CoupledSystem,
    initial: CoupledState,
    *,
    physics: Any = None,
) -> tuple[CoupledSystem, CoupledState]:
    bound = _binding(state, physics)
    if bound is None:
        return system, initial
    record, spec, device, slot = bound
    return augment_thermal_feedback(
        system,
        initial,
        spec,
        device,
        record.equipment_id,
        integral_K=slot.values["integral_K"],
        filtered_temperature_K=slot.values["filtered_temperature_K"],
    )


def store_thermal_feedback(state: WorldState, final: CoupledState) -> WorldState:
    bound = _binding(state)
    if bound is None:
        return state
    record, _, _, slot = bound
    updated = replace(
        slot,
        values={
            field.name: final.coordinates[feedback_key(record.equipment_id, field.name)]
            for field in slot.spec.fields
        },
    )
    return state.replace(
        state_slots=StateSlots(
            tuple(
                updated if s.spec.slot_id == slot.spec.slot_id else s
                for s in state.state_slots.slots
            )
        )
    )


def thermal_feedback_report(state: WorldState) -> dict[str, Any]:
    bound = _binding(state)
    if bound is None:
        return {}
    record, spec, device, slot = bound
    return {
        "thermal_feedback": {
            "equipment_id": record.equipment_id,
            "memory_slot": slot.spec.slot_id,
            **thermal_feedback_point(
                spec, device, sample_temperature_K=state.temperature_K, **slot.values
            ),
        }
    }
