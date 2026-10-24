"""Directional OCP owns real persistent state in the existing battery session."""

import json
from dataclasses import replace
from math import exp

import pytest

from chemworld.examples.battery_hysteresis import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import F
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def session(spec):
    result = ChemWorldRuntime(composition=battery_world_spec(spec))
    result.reset(seed=4)
    return result


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def expected_h(h0, signed_current, duration, capacity):
    if signed_current == 0:
        return h0
    sign = 1. if signed_current > 0 else -1.
    decay = 40. if signed_current > 0 else 20.
    return sign+(h0-sign)*exp(-decay*abs(signed_current)*duration/capacity)


@pytest.mark.parametrize("many_particle", [False, True])
def test_one_h_per_electrode_follows_actual_clock_and_total_current(many_particle):
    model = default_spec(many_particle=many_particle, cells=4)
    runtime = session(model)
    initial = unpack_battery(runtime._state)[3]
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .2})
    execute(runtime, {"operation": "wait", "duration_s": 4.})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, slot, spec, final, _ = unpack_battery(runtime._state)
    assert final.time_s == pytest.approx(5.)  # meter acquisition advances one second
    assert len(spec.hysteresis_keys) == 2
    assert tuple(final.coordinates) == spec.coordinate_keys
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    for electrode, signed_current in (("negative", .2), ("positive", -.2)):
        key = f"battery:{electrode}:hysteresis"
        capacity = F*getattr(spec, electrode).capacity_mol
        assert final.coordinates[key] == pytest.approx(
            expected_h(initial.coordinates[key], signed_current, 5., capacity), abs=1e-8)
        field = next(field for field in slot.spec.fields if field.name == key)
        assert field.unit == "dimensionless" and field.mixing == "equal" and not field.public
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-12)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(1., abs=1e-8)
    assert {c.channel_id for c in result.observation.channels} == {
        "terminal_potential", "electrical_current"}


@pytest.mark.parametrize("many_particle", [False, True])
def test_memory_survives_rest_material_reset_and_snapshot_then_reverses(many_particle):
    runtime = session(default_spec(many_particle=many_particle, cells=4))
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .3})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    _, _, spec, physical, _ = unpack_battery(runtime._state)
    memory = {k: physical.coordinates[k] for k in spec.hysteresis_keys}
    execute(runtime, {"operation": "configure_electrical_control", "current_A": 0.})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert {k: unpack_battery(runtime._state)[3].coordinates[k] for k in memory} == memory
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "configure_electrical_control", "current_A": -.2})
        execute(active, {"operation": "wait", "duration_s": 4.})
    assert runtime._state.to_dict() == restored._state.to_dict()
    final = unpack_battery(runtime._state)[3]
    assert final.coordinates[spec.hysteresis_keys[0]] < memory[spec.hysteresis_keys[0]]
    assert final.coordinates[spec.hysteresis_keys[1]] > memory[spec.hysteresis_keys[1]]
    # Replacing the actual equipment, unlike replacing surrounding material,
    # uses the new device's declared initial memory exactly once.
    record = unpack_battery(runtime._state)[0]
    replacement = replace(record, equipment_id="replacement-cell")
    fresh = initial_battery_slot(replacement, spec)
    runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                           "replacement": replacement, "slots": StateSlots((fresh,))})
    assert unpack_battery(runtime._state)[1] == fresh


def test_stage_changes_preserve_memory_and_rest_after_completion():
    model = default_spec(cells=4)
    runtime = session(model)
    segments = ((.2, 3.), (0., 2.), (-.1, 4.))
    for current, duration in segments:
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "wait", "duration_s": 9.})
    assert runtime._state.to_dict() == restored._state.to_dict()
    record, _, spec, final, program = unpack_battery(runtime._state)
    assert record.settings["electrical_control"]["stage_index"] == 3
    # A completed programme stops at its true boundary; a subsequent wait
    # advances the idle source. It does not silently consume the extra second.
    assert final.time_s == 9. and program.value_at(9.) == 0.
    for electrode, direction in (("negative", 1), ("positive", -1)):
        key = f"battery:{electrode}:hysteresis"
        expected = model.initial_state().coordinates[key]
        for current, duration in segments:
            expected = expected_h(expected, direction*current, duration,
                                  F*getattr(spec, electrode).capacity_mol)
        assert final.coordinates[key] == pytest.approx(expected, abs=1e-8)
    memory = {key: final.coordinates[key] for key in spec.hysteresis_keys}
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    rested = unpack_battery(runtime._state)[3]
    assert rested.time_s == 10.
    assert {key: rested.coordinates[key] for key in memory} == memory


def test_failed_advance_rolls_back_memory_with_inventory_and_time(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = session(default_spec(cells=4))
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected hysteresis integration failure")

    monkeypatch.setattr(adapter, "advance_battery", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize("bad_h", [-1.1, 1.1])
def test_invalid_snapshot_memory_is_rejected_without_projection(bad_h):
    runtime = session(default_spec(cells=4))
    _, slot, spec, _, _ = unpack_battery(runtime._state)
    values = {**slot.values, spec.hysteresis_keys[0]: bad_h}
    with pytest.raises(ValueError):
        corrupted = replace(slot, values=values)
        runtime._state = runtime._state.replace(state_slots=StateSlots((corrupted,)))
        ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))


@pytest.mark.parametrize("mode", ["axen", "current_sigmoid"])
@pytest.mark.parametrize("many_particle", [False, True])
def test_real_example_and_exact_replay(tmp_path, mode, many_particle):
    model = default_spec(mode=mode, many_particle=many_particle, cells=4)
    assert len(model.hysteresis_keys) == (2 if mode == "axen" else 0)
    result = run(tmp_path/f"{mode}-{many_particle}.jsonl", spec=model)
    assert result["steps"] == result["replay_steps"] == 11
