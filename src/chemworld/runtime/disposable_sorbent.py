"""Explicit finite unused-media stock for the legacy disposable contact.

Native reusable carriers own their loading separately. This legacy kit supplies
an actual fresh packet, which the existing contact removes to the spent outlet.
Neither an operation nor a material reset replenishes the kit.
"""

from dataclasses import replace
from math import isclose, isfinite

from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)

MEDIA_STATE_TYPE = "unused-drying-media-v1"
MEDIA_QUANTITY = "drying-media:mass"
LEGACY_KIT_MASS_KG = 0.02


def make_media_supply(vessel_id, *, mass_kg=LEGACY_KIT_MASS_KG, owner_id=None):
    if not isfinite(mass_kg) or mass_kg < 0:
        raise ValueError("unused drying media requires finite nonnegative kg")
    owner_id = f"{vessel_id}:drying-media-supply" if owner_id is None else owner_id
    record = EquipmentRecord(
        owner_id, "disposable_sorbent_supply", vessel_id, batch_reset="preserve"
    )
    slot = StateSlot(
        StateSlotSpec(
            f"{owner_id}:unused-media",
            "equipment",
            owner_id,
            MEDIA_STATE_TYPE,
            (
                StateFieldSpec(
                    "available_mass_kg", "kg", public=True, conserved_quantity=MEDIA_QUANTITY
                ),
            ),
            batch_reset="preserve",
            time_evolution="static",
        ),
        vessel_id,
        {"available_mass_kg": mass_kg},
    )
    return record, slot


def media_slot(state):
    slots = [s for s in state.state_slots.slots if s.spec.state_type == MEDIA_STATE_TYPE]
    if len(slots) != 1:
        raise ValueError("disposable drying requires one explicitly declared finite media supply")
    slot = slots[0]
    expected_record, expected_slot = make_media_supply(
        state.vessel_id,
        mass_kg=slot.values.get("available_mass_kg", -1),
        owner_id=slot.spec.owner_id,
    )
    record = state.equipment.equipment.get(slot.spec.owner_id)
    if (
        record is None
        or record.equipment_type != expected_record.equipment_type
        or record.attached_vessel_id != state.vessel_id
        or slot.spec != expected_slot.spec
    ):
        raise ValueError("unused drying media ownership/units disagree with its supply declaration")
    return slot


def install_legacy_media_supplies(state):
    locals_ = {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}
    for name, local in locals_.items():
        if any(s.spec.state_type == MEDIA_STATE_TYPE for s in local.state_slots.slots):
            media_slot(local)
            continue
        record, slot = make_media_supply(name)
        locals_[name] = local.replace(
            equipment=EquipmentLedger({**local.equipment.equipment, record.equipment_id: record}),
            state_slots=StateSlots((*local.state_slots.slots, slot)),
        )
    active = locals_.pop(state.vessel_id)
    return active.replace(inactive_vessels=locals_)


def debit_media(state, mass_kg):
    slot = media_slot(state)
    available = slot.values["available_mass_kg"]
    if not isfinite(mass_kg) or mass_kg <= 0 or mass_kg > available:
        raise ValueError("finite unused drying medium is exhausted")
    remaining = available - mass_kg
    updated = replace(slot, values={"available_mass_kg": remaining})
    candidate = state.replace(
        state_slots=StateSlots(
            tuple(
                updated if s.spec.slot_id == slot.spec.slot_id else s
                for s in state.state_slots.slots
            )
        )
    )
    return candidate, {
        "source_equipment": slot.spec.owner_id,
        "quantity": MEDIA_QUANTITY,
        "unit": "kg",
        "used_mass_kg": mass_kg,
        "before_mass_kg": available,
        "after_mass_kg": remaining,
        "boundary_model": "finite_unused_media_to_disposable_contact_then_spent_sink",
    }


def validate_media_packet(before, after, packet):
    from chemworld.foundation import selected_phase_id
    from chemworld.runtime.vnext_downstream import PhaseSlice, runtime_sorbent_bed

    old, new = media_slot(before), media_slot(after)
    amount = float(packet["used_mass_kg"])
    phase = before.phases.phases[selected_phase_id(before.phases)]
    expected = runtime_sorbent_bed(PhaseSlice(0.0, 0.0, phase.volume_L, 0.0)).sorbent_mass_kg
    if (
        packet.get("source_equipment") != old.spec.owner_id
        or old.spec != new.spec
        or packet.get("quantity") != MEDIA_QUANTITY
        or packet.get("unit") != "kg"
        or amount <= 0
        or amount > old.values["available_mass_kg"]
        or not isclose(amount, expected, rel_tol=0, abs_tol=1e-12)
        or not isclose(
            packet["before_mass_kg"], old.values["available_mass_kg"], rel_tol=0, abs_tol=1e-12
        )
        or not isclose(
            packet["after_mass_kg"], new.values["available_mass_kg"], rel_tol=0, abs_tol=1e-12
        )
        or not isclose(
            old.values["available_mass_kg"] - new.values["available_mass_kg"],
            amount,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("disposable sorbent outlet disagrees with actual finite media debit")
    return amount
