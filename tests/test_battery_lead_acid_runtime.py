"""Real six-species electrochemistry uses the existing clock, meter and queue."""

import json
from copy import deepcopy

import pytest

from chemworld.examples.battery_lead_acid import default_spec, run
from chemworld.physchem.battery_models import F, battery_control_current, battery_readout
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec


@pytest.fixture
def runtime():
    result = ChemWorldRuntime(composition=battery_world_spec(default_spec()))
    result.reset(seed=4)
    try:
        yield result
    finally:
        result.close()


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_meter_moves_six_actual_stocks_and_counts_terminal_work_once(runtime):
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .5})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, slot, _, final, _ = unpack_battery(runtime._state)
    assert final.time_s == 4.
    assert final.coordinates["discharged_charge_C"] == pytest.approx(2., abs=1e-8)
    coefficients = {"negative:Pb": -1, "negative:PbSO4": 1,
                    "positive:PbO2": -1, "positive:PbSO4": 1,
                    "electrolyte:H2SO4": -2, "electrolyte:H2O": 2}
    assert set(final.amounts_mol) == set(coefficients)
    for key, coefficient in coefficients.items():
        assert final.amounts_mol[key] - initial.amounts_mol[key] == (
            pytest.approx(coefficient * 2. / (2 * F), abs=1e-11))
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        -final.coordinates["delivered_work_J"], abs=1e-8)
    assert runtime._state.ledger.heat_reaction_J == 0.
    assert runtime._state.ledger.energy_jacket_J == 0.
    assert runtime._state.temperature_K == spec.temperature_K
    assert slot.spec.state_type == "battery-lead-acid-inventory-and-charge-v1"
    assert {c.channel_id for c in result.observation.channels} == {
        "terminal_potential", "electrical_current"}


def test_cv_uses_real_acid_water_and_electrode_response(runtime):
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    target = battery_readout(spec, initial, .5).voltage_V
    execute(runtime, {"operation": "configure_electrical_control", "potential_V": target})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, program = unpack_battery(runtime._state)
    current = battery_control_current(spec, final, program)
    assert 0 < current < spec.maximum_current_A
    assert battery_readout(spec, final, current).voltage_V == pytest.approx(target, abs=1e-7)
    assert final.coordinates["delivered_work_J"] == pytest.approx(
        target * final.coordinates["discharged_charge_C"], abs=1e-8)
    assert final.time_s == 3.


def test_queue_json_and_material_reset_preserve_six_stocks(runtime):
    for current, duration in ((.12, 3.), (0., 2.), (-.1, 2.)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    try:
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 8.})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        restored.close()
    record, slot, _, physical, _ = unpack_battery(runtime._state)
    assert physical.time_s == pytest.approx(7.)
    assert physical.coordinates["discharged_charge_C"] == pytest.approx(.16, abs=1e-8)
    assert record.settings["electrical_control"]["stage_index"] == 3
    before = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == before
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    assert unpack_battery(runtime._state)[3].time_s == 8.


def test_failed_candidate_commits_neither_chemistry_nor_clock(runtime, monkeypatch):
    import chemworld.runtime.battery as adapter

    before = deepcopy(runtime._state)

    def fail(*args, **kwargs):
        raise RuntimeError("injected lead-acid candidate failure")

    monkeypatch.setattr(adapter, "advance_battery", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_actual_lead_acid_example_and_replay(tmp_path, noise):
    result = run(tmp_path / f"lead-acid-{noise}.jsonl", noise=noise)
    assert result["steps"] == result["replay_steps"] == 11


def test_six_stocks_share_time_with_other_systems_and_reject_undeclared_heat():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=default_spec()))
    runtime.reset(seed=4)
    restored = None
    try:
        before = local_states(runtime._state)
        advancer = runtime.runtime.domain_services.reaction_thermal.advancer
        expected = {key: advancer.batch(local, {"operation": "wait", "duration_s": 2.},
                                       heat=False) for key, local in before.items()}
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, 2.)
        for key, local in local_states(runtime._state).items():
            assert expected[key].success
            for field in ("species_amounts", "temperature_K", "pressure_Pa", "volume_L",
                          "phases", "state_slots", "equipment"):
                assert getattr(local, field) == getattr(expected[key].state, field)
        execute(runtime, {"operation": "select_vessel", "vessel": "cell"})
        saved = runtime._state
        rejected = runtime.execute({"operation": "queue_lab_heat", "target_temperature_K": 310.,
                                    "duration_s": 2.})
        assert rejected.info["transaction_status"] != "committed"
        assert runtime._state.lab_clock == saved.lab_clock
        assert runtime._state.state_slots == saved.state_slots
        assert runtime._state.temperature_K == saved.temperature_K
        assert runtime._state.ledger.time_s == saved.ledger.time_s
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict())))
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 1.})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        if restored is not None:
            restored.close()
        runtime.close()
