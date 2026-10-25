"""A cleared mobile carrier is optional; a missing fixed surface is invalid."""

import json

import pytest

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec


def test_cleared_mobile_surface_restores_and_advances_without_inventing_carrier():
    instance = ChemWorldRuntime(composition=surface_world_spec(mobile=True))
    instance.reset(seed=12)
    instance.reset(options={"scope": "batch"})
    assert not instance._state.state_slots.material().has_inventory()
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(instance.snapshot().to_dict())))
    action = {"operation": "wait", "duration_s": 1}
    for session in (instance, restored):
        result = session.execute(action)
        assert result.info["transaction_status"] == "committed"
        assert not session._state.state_slots.material().has_inventory()
    assert instance._state.to_dict() == restored._state.to_dict()


def test_missing_fixed_surface_still_fails_original_snapshot_validation():
    instance = ChemWorldRuntime(composition=surface_world_spec())
    instance.reset(seed=12)
    payload = json.loads(json.dumps(instance.snapshot().to_dict()))
    payload["world_state"]["state_slots"] = {}
    payload["world_state"]["state_slot_order"] = []
    with pytest.raises(ValueError, match="owned surface inventory"):
        ChemWorldRuntime.from_snapshot(payload)
