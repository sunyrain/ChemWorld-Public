"""Snapshot identities survive mapping reordering and reject unknown contracts."""

import json
from copy import deepcopy

import pytest

from chemworld.foundation import WorldState
from chemworld.foundation.state_serialization import restore_sample
from chemworld.runtime import ChemWorldRuntime, SessionSnapshot
from chemworld.runtime.material_routing import _active
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.mineral_world import mineral_world_spec
from test_native_domains_end_to_end import (
    test_snapshot_resume_in_middle_of_real_workflow as check_resume,
)
from test_physics_model_declarations import local_registry, make_world
from test_state_ownership import state, support


@pytest.mark.parametrize("name", (
    "heater", "distillation", "crystal", "flash", "liquid", "concentration",
))
def test_sorted_model_input_preserves_authored_phase_associations(name):
    world = make_world(name)
    restored = WorldCompositionSpec.from_dict(
        json.loads(json.dumps(world.to_dict(), sort_keys=True))
    )
    for name, before in world.physics.initial_states.items():
        after = restored.physics.initial_states[name]
        assert after.vessels.vessels[name].phase_ids == before.vessels.vessels[name].phase_ids
        assert list(after.phases.phases) == list(before.phases.phases)


def test_sorted_state_and_retained_sample_preserve_material_slot_coordinate_order():
    before = state(support("z-mobile"), support("b-mobile"), support("a-fixed", fixed=True))
    after = WorldState.from_dict(json.loads(json.dumps(before.to_dict(), sort_keys=True)))
    assert [slot.spec.slot_id for slot in after.state_slots.slots] == [
        slot.spec.slot_id for slot in before.state_slots.slots
    ]
    sample = _active(before)
    restored = restore_sample(json.loads(json.dumps(sample.to_dict(), sort_keys=True)))
    assert [slot.spec.slot_id for slot in restored.contents.state_slots.slots] == [
        slot.spec.slot_id for slot in sample.contents.state_slots.slots
    ]


def test_unknown_continuation_field_cannot_replace_the_live_executor():
    session = ChemWorldRuntime(composition=mineral_world_spec())
    try:
        session.reset(seed=7)
        before = session.snapshot().to_dict()
        bad = session.snapshot().to_dict()
        bad["continuation"]["runtime"] = "not-an-executor"
        with pytest.raises(ValueError, match="continuation"):
            session.restore(SessionSnapshot.from_dict(bad))
        assert session.snapshot().to_dict() == before
    finally:
        session.close()


@pytest.mark.parametrize("field", ["root", "resource", "missing", "v1", "slot_order"])
def test_unknown_or_incomplete_snapshot_contract_is_atomic(field):
    session = ChemWorldRuntime(composition=mineral_world_spec())
    try:
        session.reset(seed=7)
        before = session.snapshot().to_dict()
        bad = deepcopy(before)
        if field == "root":
            bad["load_module"] = "untrusted.loader"
        elif field == "resource":
            bad["resource_ledgers"]["runtime"] = None
        elif field == "missing":
            bad["continuation"].pop("_operation_id")
        elif field == "v1":
            bad["schema_version"] = "chemworld-session-v1"
        else:
            bad["world_state"].pop("state_slot_order")
        with pytest.raises(ValueError):
            session.restore(bad)
        assert session.snapshot().to_dict() == before
    finally:
        session.close()


@pytest.mark.parametrize("value", [True, -1, 1.0])
def test_snapshot_counters_keep_integer_types(value):
    session = ChemWorldRuntime(composition=mineral_world_spec())
    try:
        session.reset(seed=7)
        bad = session.snapshot().to_dict()
        bad["continuation"]["_operation_id"] = value
        with pytest.raises(ValueError, match="exact nonnegative integers"):
            SessionSnapshot.from_dict(bad)
    finally:
        session.close()


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_json_save_keeps_full_precision_rng_and_next_real_measurement(tmp_path, noise):
    session = ChemWorldRuntime(
        composition=mineral_world_spec(), observation_noise_mode=noise,
        observation_seed_override=12,
    )
    try:
        session.reset(seed=7)
        original = session.snapshot().to_dict()
        assert original["rng_state"]["state"]["state"] > 2**53
        path = tmp_path / "rng.json"
        session.snapshot().save(path)
        loaded = SessionSnapshot.load(path)
        assert loaded.to_dict()["rng_state"] == original["rng_state"]
        restored = ChemWorldRuntime.from_snapshot(loaded)
        try:
            action = {"operation": "measure", "instrument": "aqueous_ph"}
            assert restored.execute(action) == session.execute(action)
            assert restored._state.to_dict() == session._state.to_dict()
        finally:
            restored.close()
    finally:
        session.close()


def test_snapshot_consumes_the_installed_instance_model_codecs(tmp_path):
    from chemworld.world.surface_world import surface_world_spec

    modules = local_registry()
    payload = surface_world_spec().to_dict()
    payload["physics"]["models"]["local-envelope"] = {
        "version": "test-1", "bindings": {"reactor": {"minimum": 300, "unit": "K"}},
    }
    world = WorldCompositionSpec.from_dict(payload, modules=modules)
    session = ChemWorldRuntime(composition=world)
    try:
        session.reset(seed=7)
        path = tmp_path / "local.json"
        session.snapshot().save(path)
        with pytest.raises(ValueError, match="unknown physical model type"):
            ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
        restored = ChemWorldRuntime.from_snapshot(
            SessionSnapshot.load(path), domain_modules=modules
        )
        try:
            action = {"operation": "wait", "duration_s": 1}
            assert restored.execute(action) == session.execute(action)
            assert restored._state.to_dict() == session._state.to_dict()
        finally:
            restored.close()
    finally:
        session.close()


def legacy_v1(snapshot):
    def remove_order(value):
        if isinstance(value, dict):
            return {key: remove_order(item) for key, item in value.items()
                    if key not in {"phase_order", "state_slot_order"}}
        if isinstance(value, list):
            return [remove_order(item) for item in value]
        return value

    payload = remove_order(json.loads(json.dumps(snapshot, sort_keys=True)))
    payload["schema_version"] = "chemworld-session-v1"
    return payload


def test_explicit_v1_migration_preserves_unambiguous_real_history_and_configuration():
    session = ChemWorldRuntime(composition=mineral_world_spec())
    try:
        session.reset(seed=7)
        assert session.execute({"operation": "wait", "duration_s": 10}).info[
            "transaction_status"
        ] == "committed"
        legacy = legacy_v1(session.snapshot().to_dict())
        unchanged = deepcopy(legacy)
        migrated = SessionSnapshot.migrate_v1(legacy)
        assert legacy == unchanged
        restored = ChemWorldRuntime.from_snapshot(migrated)
        try:
            action = {"operation": "measure", "instrument": "aqueous_ph"}
            expected = session.execute(action)
            assert restored.execute(action) == expected
            session.restore(migrated)
            assert session.execute(action) == expected
            assert session._state.to_dict() == restored._state.to_dict()
        finally:
            restored.close()
    finally:
        session.close()


@pytest.mark.parametrize("missing", ["semantics", "slot_order"])
def test_v1_migration_refuses_unknown_semantics_or_missing_assembly_information(missing):
    session = ChemWorldRuntime(composition=mineral_world_spec())
    try:
        session.reset(seed=7)
        legacy = legacy_v1(session.snapshot().to_dict())
        if missing == "semantics":
            legacy["execution"]["runtime"] = "retired-physical-contract"
        else:
            slots = legacy["world_state"]["state_slots"]
            slots["another-family"] = deepcopy(next(iter(slots.values())))
            slots["another-family"]["spec"]["slot_id"] = "another-family"
        with pytest.raises(ValueError):
            SessionSnapshot.migrate_v1(legacy)
    finally:
        session.close()


def extra_recipe(name):
    world = make_world(name)
    if name == "flash":
        actions = [
            {"operation": "add_component", "component": "mixture-feed", "amount_mol": 1e-4},
            {"operation": "heat", "target_temperature_K": 410, "duration_s": 20,
             "stirring_speed_rpm": 600},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "route_material", "connection": "reactor-to-receiver",
             "transfer_fraction": 0.2, "mixing": 1},
            {"operation": "terminate"},
        ]
    elif name == "liquid":
        actions = [
            {"operation": "mix", "duration_s": 2, "stirring_speed_rpm": 600},
            {"operation": "wash", "wash_volume_L": 1e-5},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "route_material", "connection": "reactor-to-receiver",
             "transfer_fraction": 0.2, "mixing": 1},
            {"operation": "terminate"},
        ]
    else:
        actions = [
            {"operation": "concentrate", "duration_s": 10},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "pause_control"},
            {"operation": "wait", "duration_s": 2, "stirring_speed_rpm": 600},
            {"operation": "terminate"},
        ]
    return world, actions


@pytest.mark.parametrize("name", (
    "heater", "membrane", "mineral", "shared-clock", "distillation", "crystal", "solid",
    "flash", "liquid", "concentration",
))
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_sorted_json_continues_each_physical_history_in_a_new_process(name, noise, tmp_path):
    recipe = extra_recipe(name) if name in {"flash", "liquid", "concentration"} else None
    check_resume(name, noise, tmp_path, recipe=recipe, sort_keys=True)
