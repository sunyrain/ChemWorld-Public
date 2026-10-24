"""Thickness-field state follows the same cell lifetime and shared clock."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.examples.battery_spatial_thermal import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import (
    F,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_control_current,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_spatial_thermal import spatial_thermal_internal_energy_j
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def fixture_spec():
    spec = default_spec(cells=4, thermal_cells=2)
    return replace(spec, initial_temperatures_K=tuple(
        np.linspace(300., 310., len(spec.heat_stack.node_ids))))


def session(spec=None):
    runtime = ChemWorldRuntime(composition=battery_world_spec(
        fixture_spec() if spec is None else spec))
    runtime.reset(seed=4)
    return runtime


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def temperature_field(spec, state):
    return spec.heat_stack.temperatures(state.temperature_K, state.coordinates)


def test_meter_advances_heat_field_and_books_terminal_work_once():
    runtime = session()
    _, slot, spec, initial, _ = unpack_battery(runtime._state)
    assert battery_spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert runtime._state.volume_L == spec.volume_l
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    initial_energy = spatial_thermal_internal_energy_j(spec, initial)
    execute(runtime, {"operation": "configure_electrical_control", "current_A": 1.})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, _ = unpack_battery(runtime._state)
    assert final.time_s == 4.
    assert final.coordinates["discharged_charge_C"] == pytest.approx(4., abs=1e-8)
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-11)
    assert not np.allclose(temperature_field(spec, final), temperature_field(spec, initial))
    ledger = runtime._state.ledger
    assert ledger.energy_jacket_J == 0.
    assert ledger.flow_electrical_work_J == pytest.approx(
        -final.coordinates["delivered_work_J"], abs=1e-8)
    assert spatial_thermal_internal_energy_j(spec, final)-initial_energy == pytest.approx(
        ledger.flow_electrical_work_J-ledger.heat_loss_J, abs=2e-6)
    assert ledger.heat_reaction_J == pytest.approx(
        battery_chemical_energy_j(spec, initial)-battery_chemical_energy_j(spec, final), abs=1e-6)
    assert {c.channel_id for c in result.observation.channels} == {
        "terminal_potential", "electrical_current"}


def test_voltage_control_uses_evolving_electrode_temperatures():
    runtime = session()
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    target = battery_readout(spec, initial, 1.).voltage_V
    execute(runtime, {"operation": "configure_electrical_control", "potential_V": target})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    _, _, _, final, program = unpack_battery(runtime._state)
    current = battery_control_current(spec, final, program)
    assert 0 < current < spec.maximum_current_A
    assert battery_readout(spec, final, current).voltage_V == pytest.approx(target, abs=2e-6)
    assert final.time_s == 3.


def test_midqueue_snapshot_and_material_reset_preserve_local_temperatures():
    runtime = session()
    for current, duration in ((1., 3.), (0., 2.), (-.8, 4.)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "wait", "duration_s": 9.})
    assert runtime._state.to_dict() == restored._state.to_dict()
    record, slot, spec, physical, programme = unpack_battery(runtime._state)
    assert physical.time_s == 9.
    assert physical.coordinates["discharged_charge_C"] == pytest.approx(-.2, abs=1e-8)
    assert programme.value_at(physical.time_s) == 0.
    assert record.settings["electrical_control"]["stage_index"] == 3
    temperatures = temperature_field(spec, physical).copy()
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    _, after_slot, _, after, _ = unpack_battery(runtime._state)
    assert dict(after_slot.values) == dict(slot.values)
    np.testing.assert_array_equal(temperature_field(spec, after), temperatures)
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    rested = unpack_battery(runtime._state)[3]
    assert rested.time_s == 10.
    assert rested.coordinates["discharged_charge_C"] == pytest.approx(-.2, abs=1e-8)
    assert not np.allclose(temperature_field(spec, rested), temperatures)


def test_replacement_installs_new_temperature_field_and_inventory_once():
    runtime = session()
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    record, _, old, previous, _ = unpack_battery(runtime._state)
    new = replace(old, initial_negative_fraction=.6,
                  initial_temperatures_K=tuple(reversed(old.initial_temperatures_K)))
    replacement = replace(record, equipment_id="new-thermal-cell", settings={
        **record.settings, "physical_model": new.to_dict()})
    fresh = initial_battery_slot(replacement, new)
    runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                           "replacement": replacement, "slots": StateSlots((fresh,))})
    _, slot, spec, physical, _ = unpack_battery(runtime._state)
    assert slot == fresh and spec == new
    assert physical.amounts_mol == new.initial_state().amounts_mol
    np.testing.assert_array_equal(temperature_field(spec, physical),
                                  temperature_field(new, new.initial_state()))
    assert physical.temperature_K == new.temperature_K
    assert physical.amounts_mol != previous.amounts_mol


def test_solver_failure_rolls_back_every_temperature_coordinate_and_clock(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = session()
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected spatial thermal solver failure")

    monkeypatch.setattr(adapter, "advance_battery", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.temperature_K == before.temperature_K
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_undefined_external_thermal_contact_is_rejected_at_world_boundary():
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    with pytest.raises(ValueError, match="spatial thermal"):
        battery_world_spec(fixture_spec(), thermal_device=ThermalDeviceSpec())


def test_generic_chemical_energy_matches_declared_constant_ocv():
    spec = fixture_spec()
    state = spec.initial_state()
    expected = sum(-F*spec.ocv(e, .5)*state.amounts_mol[k]
                   for e in ("negative", "positive") for k in getattr(spec, e).keys(e))
    assert battery_chemical_energy_j(spec, state) == pytest.approx(expected, abs=1e-8)
    potentials = battery_chemical_potentials(spec, state)
    for e in ("negative", "positive"):
        assert all(potentials[k] == pytest.approx(-F*spec.ocv(e, .5))
                   for k in getattr(spec, e).keys(e))


@pytest.mark.parametrize("operation", ["impedance", "soh", "dissipation", "sei", "soc"])
def test_undefined_analysis_refuses_without_mutating_temperature_field(operation):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import battery_dissipation_w, battery_sei_affinity_j_mol
    from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery

    spec = fixture_spec()
    state = spec.initial_state()
    original = (dict(state.amounts_mol), dict(state.coordinates), state.temperature_K, state.time_s)
    calls = {
        "impedance": lambda: battery_impedance(spec, [1.], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "dissipation": lambda: battery_dissipation_w(spec, state, 1.),
        "sei": lambda: battery_sei_affinity_j_mol(spec, state, 1.),
        "soc": lambda: battery_spec_at_soc(spec, None, .5),
    }
    with pytest.raises(ValueError, match="spatial"):
        calls[operation]()
    assert (dict(state.amounts_mol), dict(state.coordinates), state.temperature_K, state.time_s
            ) == original


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_spatial_thermal_example_replays_both_noise_modes(tmp_path, noise):
    result = run(tmp_path/f"spatial-thermal-{noise}.jsonl",
                 spec=default_spec(cells=4, thermal_cells=2), noise=noise)
    assert result["steps"] == result["replay_steps"] == 11


def test_shared_clock_advances_actual_field_and_refuses_undefined_movable_heater():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=fixture_spec()))
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
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        sum(item.state.ledger.flow_electrical_work_J for item in expected.values()), abs=1e-8)
    assert runtime._state.ledger.heat_loss_J == pytest.approx(
        sum(item.state.ledger.heat_loss_J for item in expected.values()), abs=1e-8)
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
