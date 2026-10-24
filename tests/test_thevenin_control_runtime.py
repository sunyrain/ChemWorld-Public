"""Named targets use the original shared clock, ownership and replay contracts."""

import json

import pytest

from chemworld.examples.native_thevenin import control_spec, default_spec, diffusion_spec, run
from chemworld.physchem.battery_thevenin import SOC, WORK, thevenin_fields
from chemworld.physchem.battery_thevenin_control import CONTROLLER_CURRENT
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.thevenin import unpack_thevenin
from chemworld.world.thevenin_world import thevenin_world_spec
from test_thevenin_runtime import execute, session


@pytest.mark.parametrize("mode,extra", [
    ("power", {"power_W": .5, "branch_lower_A": 0., "branch_upper_A": 1.}),
    ("resistance", {"resistance_Ohm": 10.}),
    ("differential_power", {"power_W": .5}),
    ("differential_resistance", {"resistance_Ohm": 35.,
                                "branch_lower_A": .01, "branch_upper_A": 1.}),
    ("cccv", {"potential_V": 3.8, "current_limit_A": -.2}),
])
def test_actual_named_operation_advances_one_owned_state(mode, extra):
    runtime = session(control_spec())
    before = unpack_thevenin(runtime._state)[3]
    execute(runtime, {"operation": "configure_electrical_control", "control_mode": mode, **extra})
    assert runtime._state.ledger.time_s == 0.
    execute(runtime, {"operation": "wait", "duration_s": .2})
    _, slot, model, after, program = unpack_thevenin(runtime._state)
    assert program.control_mode == mode
    assert runtime._state.ledger.time_s == pytest.approx(.2)
    assert slot.values == after.coordinates and slot.spec.owner_kind == "equipment"
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(-after.coordinates[WORK])
    fields = thevenin_fields(model, after, program)
    if mode == "power":
        assert fields["current_A"]*fields["voltage_V"] == pytest.approx(.5, abs=1e-8)
    if mode == "resistance":
        assert fields["voltage_V"]/fields["current_A"] == pytest.approx(10., abs=1e-8)
    if mode in {"power", "resistance"}:
        assert after.coordinates[CONTROLLER_CURRENT] == before.coordinates[CONTROLLER_CURRENT]
    else:
        assert after.coordinates[CONTROLLER_CURRENT] != before.coordinates[CONTROLLER_CURRENT]


def test_control_memory_survives_cc_pause_material_reset_and_exact_snapshot_replay():
    runtime = session(control_spec(diffusion_spec(cells=4)))
    execute(runtime, {"operation": "configure_electrical_control",
                      "control_mode": "differential_power", "power_W": .8})
    execute(runtime, {"operation": "wait", "duration_s": .2})
    memory = unpack_thevenin(runtime._state)[3].coordinates[CONTROLLER_CURRENT]
    execute(runtime, {"operation": "configure_electrical_control", "current_A": 0.})
    execute(runtime, {"operation": "wait", "duration_s": .2})
    assert unpack_thevenin(runtime._state)[3].coordinates[CONTROLLER_CURRENT] == memory
    runtime.reset(options={"scope": "material", "vessel_ids": [runtime._state.vessel_id]})
    assert unpack_thevenin(runtime._state)[3].coordinates[CONTROLLER_CURRENT] == memory
    payload = json.loads(json.dumps(runtime.snapshot().to_dict(), sort_keys=True))
    restored = ChemWorldRuntime.from_snapshot(payload)
    for current in (runtime, restored):
        execute(current, {"operation": "configure_electrical_control",
                          "control_mode": "differential_power", "power_W": .2})
        execute(current, {"operation": "wait", "duration_s": .3})
    assert runtime._state.to_dict() == restored._state.to_dict()
    assert unpack_thevenin(runtime._state)[3].coordinates[CONTROLLER_CURRENT] != memory


def test_queued_named_modes_activate_on_shared_clock_without_reinitializing_controller():
    runtime = session(control_spec())
    execute(runtime, {"operation": "queue_electrical_stage", "control_mode": "differential_power",
                      "power_W": .7, "stage_duration_s": .3})
    execute(runtime, {"operation": "queue_electrical_stage", "control_mode": "cccv",
                      "potential_V": 3.8, "current_limit_A": -.2, "stage_duration_s": .4})
    execute(runtime, {"operation": "wait", "duration_s": .1})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for current in (runtime, restored):
        execute(current, {"operation": "wait", "duration_s": .6})
    assert runtime._state.to_dict() == restored._state.to_dict()
    record, _, _, state, program = unpack_thevenin(runtime._state)
    assert record.settings["electrical_control"]["stage_index"] == 2
    assert program.control_mode == "current" and program.value_at(state.time_s) == 0.
    assert state.time_s == pytest.approx(.7)
    assert state.coordinates[CONTROLLER_CURRENT] != .1


@pytest.mark.parametrize("future", [False, True])
def test_unreachable_queue_activation_cannot_commit_an_unrestorable_snapshot(future):
    runtime = session(control_spec())
    if future:
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": .1,
                          "stage_duration_s": .1})
    unreachable = {"operation": "queue_electrical_stage", "control_mode": "power",
                   "power_W": 90., "branch_lower_A": 0., "branch_upper_A": 1.,
                   "stage_duration_s": 1.}
    before = runtime._state
    if future:
        execute(runtime, unreachable)
        before = runtime._state
        outcome = runtime.execute({"operation": "wait", "duration_s": .1})
    else:
        outcome = runtime.execute(unreachable)
    assert outcome.info["transaction_status"] != "committed"
    assert runtime._state.equipment == before.equipment
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s
    payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(payload)
    assert restored._state.to_dict() == runtime._state.to_dict()


@pytest.mark.parametrize("extra", [
    {"control_mode": "power", "power_W": .5},
    {"control_mode": "power", "power_W": 90., "branch_lower_A": 0., "branch_upper_A": 1.},
    {"control_mode": "resistance", "resistance_Ohm": 0.},
    {"control_mode": "differential_resistance", "resistance_Ohm": 10.,
     "branch_lower_A": -.5, "branch_upper_A": .5},
    {"control_mode": "cccv", "potential_V": 3.8, "current_limit_A": .2},
    {"control_mode": "power", "power_W": .5, "branch_lower_A": 0.,
     "branch_upper_A": 1., "ramp_duration_s": 1.},
])
def test_invalid_named_targets_reject_without_changing_physical_state(extra):
    runtime = session(control_spec())
    original = runtime._state
    outcome = runtime.execute({"operation": "configure_electrical_control", **extra})
    assert outcome.info["transaction_status"] != "committed"
    assert runtime._state.state_slots == original.state_slots
    assert runtime._state.equipment == original.equipment
    assert runtime._state.ledger.time_s == original.ledger.time_s


def test_dynamic_control_requires_installed_memory_but_algebraic_power_does_not():
    runtime = session(default_spec())
    bad = runtime.execute({"operation": "configure_electrical_control",
                           "control_mode": "differential_power", "power_W": .5})
    assert bad.info["transaction_status"] != "committed"
    execute(runtime, {"operation": "configure_electrical_control", "control_mode": "power",
                      "power_W": .5, "branch_lower_A": 0., "branch_upper_A": 1.})
    execute(runtime, {"operation": "wait", "duration_s": .1})
    assert CONTROLLER_CURRENT not in unpack_thevenin(runtime._state)[3].coordinates


def test_named_program_ramp_keeps_mode_and_never_interpolates_control_memory():
    runtime = session(control_spec())
    execute(runtime, {"operation": "configure_electrical_control",
                      "control_mode": "differential_power", "power_W": .5})
    execute(runtime, {"operation": "configure_electrical_control",
                      "control_mode": "differential_power", "power_W": 1., "ramp_duration_s": 2.})
    _, _, _, state, program = unpack_thevenin(runtime._state)
    assert program.value_at(1.) == .75
    assert state.coordinates[CONTROLLER_CURRENT] == .1


def test_stationary_analysis_ignores_frozen_controller_without_resetting_it():
    world = thevenin_world_spec(control_spec(), include_impedance_analysis=True)
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=7)
    original = runtime._state.state_slots
    execute(runtime, {"operation": "measure", "instrument": "thevenin_small_signal_analysis"})
    assert runtime._state.state_slots == original
    assert unpack_thevenin(runtime._state)[3].coordinates[SOC] == .7


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_named_control_example_has_complete_trajectory_and_replay(tmp_path, noise):
    report = run(tmp_path / (noise+".jsonl"), noise=noise, include_named_controls=True)
    assert report["steps"] == report["replay_steps"] == 17
