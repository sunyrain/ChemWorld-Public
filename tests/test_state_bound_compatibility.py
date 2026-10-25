"""Optional upper bounds preserve the original state field and JSON contracts."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state_ownership import StateFieldSpec, StateSlot, StateSlotSpec


def test_all_original_field_positional_arguments_keep_their_meanings():
    field = StateFieldSpec("bound", "mol", "sum", None, 0.0, True, "element:C", "C")
    assert field.public is True
    assert field.conserved_quantity == "element:C"
    assert field.species_id == "C"
    assert field.maximum is None


def legacy_payload():
    return {
        "spec": {
            "slot_id": "surface", "owner_kind": "material", "owner_id": "catalyst",
            "state_type": "adsorption", "fields": [{
                "name": "bound", "unit": "mol", "mixing": "sum", "weight_field": None,
                "minimum": 0.0, "public": True, "conserved_quantity": "element:C",
                "species_id": "C",
            }],
            "mechanism_id": "binding", "phase_id": "solid", "batch_reset": "clear",
            "heat_capacity_field": None, "capacities": [], "time_evolution": "coupled",
            "energy_interactions": [],
        },
        "location_id": "vessel", "values": {"bound": 0.2}, "initial_values": {"bound": 0.2},
    }


def test_old_unbounded_slot_json_and_public_observation_are_unchanged():
    payload = legacy_payload()
    restored = StateSlot.from_dict(payload)
    assert json.loads(json.dumps(restored.to_dict())) == payload
    assert restored.to_dict(include_hidden=False) == {"bound": {"value": .2, "unit": "mol"}}
    explicit_null = legacy_payload()
    explicit_null["spec"]["fields"][0]["maximum"] = None
    assert StateSlot.from_dict(explicit_null) == restored


def test_declared_upper_bound_roundtrips_and_still_rejects_invalid_inventory():
    field = StateFieldSpec("h", "dimensionless", mixing="equal", minimum=-1.0, maximum=1.0)
    specification = StateSlotSpec("hysteresis", "equipment", "cell", "memory", (field,))
    slot = StateSlot(specification, "vessel", {"h": 1.0})
    payload = json.loads(json.dumps(slot.to_dict()))
    assert payload["spec"]["fields"][0]["maximum"] == 1.0
    assert StateSlot.from_dict(payload) == slot
    for h in (-1.01, 1.01):
        with pytest.raises(ValueError, match="bound"):
            replace(slot, values={"h": h})
        with pytest.raises(ValueError, match="bound"):
            replace(slot, initial_values={"h": h})
