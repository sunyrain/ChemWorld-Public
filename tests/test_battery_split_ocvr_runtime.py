"""Separate electrode capacities retain original sessions, owners and controls."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.battery_split_ocvr import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import (
    F,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_control_current,
    battery_dissipation_w,
    battery_readout,
    battery_sei_affinity_j_mol,
    battery_spec_from_dict,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
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


def test_meter_advances_only_two_lithium_stores_and_counts_terminal_work_once(runtime):
    _, slot, spec, initial, _ = unpack_battery(runtime._state)
    assert battery_spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert set(slot.values) == {
        "negative:Li", "positive:Li", "discharged_charge_C", "delivered_work_J"}
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .12})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, _ = unpack_battery(runtime._state)
    assert final.time_s == 4.
    assert final.coordinates["discharged_charge_C"] == pytest.approx(.48, abs=1e-8)
    for electrode, sign in (("negative", -1), ("positive", 1)):
        key = electrode + ":Li"
        assert final.amounts_mol[key] - initial.amounts_mol[key] == (
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
    record = unpack_battery(runtime._state)[0]
    assert record.equipment_type == "electrochemical_battery"
    assert slot.spec.state_type == "battery-lithium-and-charge-v1"
    assert set(initial.amounts_mol) == {"negative:Li", "positive:Li"}
    assert not hasattr(battery_readout(spec, final, .12), "negative_surface_fraction")


def test_voltage_control_and_work_follow_the_same_series_resistance(runtime):
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
    assert final.coordinates["delivered_work_J"] == pytest.approx(
        target * final.coordinates["discharged_charge_C"], abs=1e-8)


def test_queue_midstage_json_and_material_reset_keep_the_same_two_stores(runtime):
    for current, duration in ((.12, 3.), (0., 2.), (-.1, 4.)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    try:
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 9.})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        restored.close()
    record, slot, _, physical, _ = unpack_battery(runtime._state)
    assert physical.time_s == pytest.approx(9.)
    assert physical.coordinates["discharged_charge_C"] == pytest.approx(-.04, abs=1e-8)
    assert record.settings["electrical_control"]["stage_index"] == 3
    before = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == before
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    final = unpack_battery(runtime._state)[3]
    assert final.time_s == 10.
    assert final.coordinates["discharged_charge_C"] == pytest.approx(-.04, abs=1e-8)


def test_replacement_installs_new_average_material_once(runtime):
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .12})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    record, _, old, previous, _ = unpack_battery(runtime._state)
    new = replace(old, initial_negative_fraction=.5)
    replacement = replace(record, equipment_id="new-split", settings={
        **record.settings, "physical_model": new.to_dict()})
    fresh = initial_battery_slot(replacement, new)
    runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                           "replacement": replacement, "slots": StateSlots((fresh,))})
    _, slot, spec, physical, _ = unpack_battery(runtime._state)
    assert slot == fresh and spec == new
    assert physical.amounts_mol == new.initial_state().amounts_mol
    assert physical.amounts_mol != previous.amounts_mol
    assert physical.time_s == 3.


def test_solver_failure_leaves_inventory_and_clock_uncommitted(runtime, monkeypatch):
    import chemworld.runtime.battery as adapter

    before = deepcopy(runtime._state)

    def fail(*args, **kwargs):
        raise RuntimeError("injected Split OCV-R solver failure")

    monkeypatch.setattr(adapter, "advance_battery", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_no_undefined_thermal_energy_or_sei_fallback(runtime):
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    _, _, spec, state, _ = unpack_battery(runtime._state)
    calls = (
        lambda: battery_world_spec(spec, thermal_device=ThermalDeviceSpec()),
        lambda: battery_chemical_energy_j(spec, state),
        lambda: battery_chemical_potentials(spec, state),
        lambda: battery_dissipation_w(spec, state, 0.),
        lambda: battery_sei_affinity_j_mol(spec, state, 0.),
    )
    for call in calls:
        with pytest.raises(ValueError, match="Split OCV-R"):
            call()


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_split_ocvr_example_replays_both_noise_modes(tmp_path, noise):
    result = run(tmp_path/f"split-{noise}.jsonl", spec=default_spec(), noise=noise)
    assert result["steps"] == result["replay_steps"] == 11


def test_shared_clock_and_heater_rejection_preserve_original_owners():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(
        battery_spec=default_spec()))
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


def test_zero_resistance_cc_works_but_undefined_cv_preserves_stocks():
    spec = replace(default_spec(), resistance_ohm=0.)
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=4)
    try:
        execute(runtime, {"operation": "configure_electrical_control", "current_A": .1})
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        _, _, _, state, _ = unpack_battery(runtime._state)
        assert state.coordinates["discharged_charge_C"] == pytest.approx(.2, abs=1e-8)
        saved = runtime._state
        potential = battery_readout(spec, state, 0.).voltage_V
        result = runtime.execute({"operation": "configure_electrical_control",
                                  "potential_V": potential})
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.state_slots == saved.state_slots
        assert runtime._state.ledger.time_s == saved.ledger.time_s
        rejected_stage = runtime.execute({"operation": "queue_electrical_stage",
                                          "potential_V": potential, "stage_duration_s": 2.})
        assert rejected_stage.info["transaction_status"] != "committed"
        assert runtime._state.state_slots == saved.state_slots
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": .1,
                          "stage_duration_s": 2., "stop_voltage_below_V": 3.0})
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        assert unpack_battery(runtime._state)[3].coordinates["discharged_charge_C"] == (
            pytest.approx(.4, abs=1e-8))
    finally:
        runtime.close()
