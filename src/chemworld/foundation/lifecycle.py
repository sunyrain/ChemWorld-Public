"""Experiment-boundary initialization, distinct from in-world material operations.

These functions are evaluator/session lifecycle APIs, not free Agent operations.
They record removed and declared replacement material rather than pretending a
new batch is a closed physical transition. In-world flushing uses material routing.
"""

from dataclasses import dataclass, replace

from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, ProcessLedger
from chemworld.foundation.state_ownership import StateSlots


@dataclass(frozen=True)
class LifecycleTransition:
    state: WorldState
    receipt: dict


def _local_states(state):
    return {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}


def _batch_equipment(equipment, initial):
    # Completion means "this batch has a terminal readout". Usage, elapsed
    # instrument time, calibration and drift belong to the physical device.
    records = {}
    for key, record in equipment.equipment.items():
        if record.batch_reset == "preserve":
            records[key] = (
                replace(record, status="idle") if record.equipment_type == "instrument" else record
            )
        elif key in initial.equipment:
            records[key] = initial.equipment[key]
    return EquipmentLedger(records)


def initialize_entities(current, initial, *, scope="batch", vessel_ids=None, preserve_vessels=()):
    """Apply declared initialization while preserving states owned by retained entities.

    batch: slot-specific reset; material: replace contents and retain stationary
    states; vessel: replace contents/container states, retain attached equipment.
    Equipment replacement is explicit through replace_equipment below.
    """
    if scope not in {"batch", "material", "vessel"}:
        raise ValueError("unknown entity initialization scope")
    shared = current.lab_clock is not None
    if shared and scope == "batch":
        raise ValueError(
            "shared initialization requires explicit material, vessel or equipment scope"
        )
    old, templates = _local_states(current), _local_states(initial)
    selected = set(old) - set(preserve_vessels) if vessel_ids is None else set(vessel_ids)
    if not selected <= old.keys() or not selected <= templates.keys():
        raise ValueError("initialization must refer to declared vessel locations")
    result, receipts = dict(old), {}
    for name in sorted(selected):
        previous, template = old[name], templates[name]
        if scope == "batch":
            slots = previous.state_slots.reset()
        elif scope == "material":
            slots = StateSlots((*previous.state_slots.stationary().slots,
                                *template.state_slots.material().slots))
        else:
            slots = StateSlots((
                *(s for s in previous.state_slots.slots if s.spec.owner_kind == "equipment"),
                *(s for s in template.state_slots.slots if s.spec.owner_kind != "equipment"),
            ))
        result[name] = template.replace(
            state_slots=slots,
            equipment=(_batch_equipment(previous.equipment, template.equipment)
                       if scope == "batch" else previous.equipment),
            process=ProcessLedger(),
            **({"ledger": previous.ledger, "thermal": previous.thermal,
                "vessels": previous.vessels if scope == "material" else template.vessels}
               if shared else {}),
        )
        receipts[name] = {
            "removed_bulk_mol": dict(previous.species_amounts),
            "introduced_bulk_mol": dict(template.species_amounts),
            "previous_slots": previous.state_slots.to_dict(),
            "initialized_slots": slots.to_dict(),
            "previous_temperature_K": previous.temperature_K,
            "initialized_temperature_K": template.temperature_K,
        }
    active_id = initial.vessel_id if scope == "batch" else current.vessel_id
    active = result.pop(active_id)
    generation = current.samples.batch_generation + 1
    samples = replace(
        current.samples,
        active_lineage=() if active_id in selected else current.samples.active_lineage,
        active_reference_sources=({} if active_id in selected
                                  else current.samples.active_reference_sources),
        active_reference_shares=({} if active_id in selected
                                 else current.samples.active_reference_shares),
        batch_generation=generation,
    )
    state = active.replace(
        samples=samples,
        vessel_elapsed_s={
            **active.vessel_elapsed_s,
            **{name: age for name, age in current.vessel_elapsed_s.items() if name not in selected},
        },
        inactive_vessels={
            name: local.replace(samples=replace(
                local.samples, batch_generation=generation,
                active_lineage=(f"batch-{generation:04d}-{name}",),
            )) if name in selected else local
            for name, local in result.items()
        },
    )
    if scope != "batch":
        state = state.replace(
            ledger=current.ledger, thermal=current.thermal,
            vessel_elapsed_s=current.vessel_elapsed_s,
            lab_clock=current.lab_clock,
        )
    return LifecycleTransition(state, {
        "scope": scope,
        "kind": "experiment_boundary_initialization",
        "vessels": receipts,
        "preserved_vessels": sorted(set(old) - selected),
        "thermal_semantics": "declared_initial_condition_not_a_physical_heating_operation",
    })


def replace_equipment(state, equipment_id, replacement, *, slots=None):
    """Replace one installed device, retaining the specimen and other devices."""
    slots = slots or StateSlots()
    locals_ = _local_states(state)
    matches = [name for name, local in locals_.items() if equipment_id in local.equipment.equipment]
    if len(matches) != 1:
        raise ValueError("replacement must identify exactly one installed device")
    name = matches[0]
    if replacement.attached_vessel_id != name:
        raise ValueError("replacement must attach to the same declared location")
    if replacement.equipment_id == equipment_id or any(
        replacement.equipment_id in local.equipment.equipment for local in locals_.values()
    ):
        raise ValueError("a replacement device needs a new unique identity")
    if any(
        s.spec.owner_kind != "equipment" or s.spec.owner_id != replacement.equipment_id
        or s.location_id != name for s in slots.slots
    ):
        raise ValueError("replacement slots must belong to the new device")
    local = locals_[name]
    removed = local.equipment.equipment[equipment_id]
    if replacement.equipment_type != removed.equipment_type:
        raise ValueError("replacement must retain the installed equipment type")
    role = removed.settings.get("lab_analyzer_instrument")
    if role is not None:
        if replacement.settings.get("instrument_id", role) != role:
            raise ValueError("replacement analyzer must support its declared instrument contract")
        replacement = replace(replacement, settings={
            **replacement.settings, "lab_analyzer_instrument": role,
        })
    records = {k: v for k, v in local.equipment.equipment.items() if k != equipment_id}
    records[replacement.equipment_id] = replacement
    retained = tuple(s for s in local.state_slots.slots if not (
        s.spec.owner_kind == "equipment" and s.spec.owner_id == equipment_id
    ))
    locals_[name] = local.replace(
        equipment=EquipmentLedger(records), state_slots=StateSlots((*retained, *slots.slots)),
        process=replace(local.process, last_observation_record=None),
    )
    active = locals_.pop(state.vessel_id)
    return LifecycleTransition(active.replace(inactive_vessels=locals_), {
        "scope": "equipment", "kind": "experiment_boundary_initialization",
        "removed_device": local.equipment.equipment[equipment_id].to_dict(),
        "introduced_device": replacement.to_dict(),
        "removed_slots": StateSlots(tuple(s for s in local.state_slots.slots if (
            s.spec.owner_kind == "equipment" and s.spec.owner_id == equipment_id
        ))).to_dict(),
        "introduced_slots": slots.to_dict(),
    })
