"""MSMR predictions consume the original cell inventory and session clock."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.battery_msmr import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import (
    F,
    battery_control_current,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def session(spec=None):
    runtime = ChemWorldRuntime(composition=battery_world_spec(
        default_spec(cells=4) if spec is None else spec))
    runtime.reset(seed=4)
    return runtime


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def amounts(spec, state, electrode):
    return sum(state.amounts_mol[key] for key in getattr(spec, electrode).keys(electrode))


def test_meter_advances_unique_lithium_and_terminal_work_once():
    runtime = session()
    _, slot, spec, initial, _ = unpack_battery(runtime._state)
    assert battery_spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert len(slot.values) == len(spec.inventory_keys) + 2
    assert spec.coordinate_keys == ("discharged_charge_C", "delivered_work_J")
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .12})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, _ = unpack_battery(runtime._state)
    assert final.time_s == 4.
    assert final.coordinates["discharged_charge_C"] == pytest.approx(.48, abs=1e-8)
    for electrode, sign in (("negative", -1), ("positive", 1)):
        assert amounts(spec, final, electrode)-amounts(spec, initial, electrode) == (
            pytest.approx(sign*.48/F, abs=1e-11))
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-11)
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        -final.coordinates["delivered_work_J"], abs=1e-9)
    assert runtime._state.ledger.energy_jacket_J == 0.
    assert runtime._state.ledger.heat_reaction_J == 0.
    assert runtime._state.temperature_K == spec.temperature_K
    assert {c.channel_id for c in result.observation.channels} == {
        "terminal_potential", "electrical_current"}
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"


def test_voltage_control_uses_actual_msmr_state():
    runtime = session()
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    target = battery_readout(spec, initial, .12).voltage_V
    execute(runtime, {"operation": "configure_electrical_control", "potential_V": target})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, program = unpack_battery(runtime._state)
    current = battery_control_current(spec, final, program)
    assert 0 < current < spec.maximum_current_A
    assert battery_readout(spec, final, current).voltage_V == pytest.approx(target, abs=2e-6)
    assert final.time_s == 3.


def test_midqueue_json_snapshot_and_material_reset_keep_msmr_inventory():
    runtime = session()
    for current, duration in ((.12, 3.), (0., 2.), (-.1, 4.)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "wait", "duration_s": 9.})
    assert runtime._state.to_dict() == restored._state.to_dict()
    record, slot, _, physical, program = unpack_battery(runtime._state)
    assert physical.time_s == pytest.approx(9.)
    assert physical.coordinates["discharged_charge_C"] == pytest.approx(-.04, abs=1e-8)
    assert program.value_at(physical.time_s) == 0.
    assert record.settings["electrical_control"]["stage_index"] == 3
    before = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == before
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    final = unpack_battery(runtime._state)[3]
    assert final.time_s == 10.
    assert final.coordinates["discharged_charge_C"] == pytest.approx(-.04, abs=1e-8)


def test_replacement_installs_fresh_material_declaration_and_inventory_once():
    runtime = session()
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .12})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    record, _, old, previous, _ = unpack_battery(runtime._state)
    new = replace(old, initial_negative_fraction=.5)
    replacement = replace(record, equipment_id="new-msmr", settings={
        **record.settings, "physical_model": new.to_dict()})
    fresh = initial_battery_slot(replacement, new)
    runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                           "replacement": replacement, "slots": StateSlots((fresh,))})
    _, slot, spec, physical, _ = unpack_battery(runtime._state)
    assert slot == fresh and spec == new
    assert physical.amounts_mol == new.initial_state().amounts_mol
    assert physical.amounts_mol != previous.amounts_mol


def test_solver_failure_rolls_back_every_shell_and_clock(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = session()
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected MSMR solver failure")

    monkeypatch.setattr(adapter, "advance_battery", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_world_rejects_undefined_thermal_coupling():
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    with pytest.raises(ValueError, match="MSMR"):
        battery_world_spec(default_spec(cells=4), thermal_device=ThermalDeviceSpec())


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_msmr_example_replays_both_noise_modes(tmp_path, noise):
    result = run(tmp_path/f"msmr-{noise}.jsonl", spec=default_spec(cells=4), noise=noise)
    assert result["steps"] == result["replay_steps"] == 11


def test_shared_clock_and_movable_heater_refusal_preserve_one_state():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=default_spec(cells=4)))
    runtime.reset(seed=4)
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
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "wait", "duration_s": 1.})
    assert runtime._state.to_dict() == restored._state.to_dict()
