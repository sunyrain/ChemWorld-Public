"""Finite metal shares the existing cell lifetime, clock, stops and replay."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.battery_half_cell import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import F, battery_readout, battery_spec_from_dict
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def session(spec=None):
    result = ChemWorldRuntime(composition=battery_world_spec(
        default_spec(cells=4) if spec is None else spec))
    result.reset(seed=4)
    return result


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_real_meter_consumes_actual_lithium_and_time_without_negative_particle():
    runtime = session()
    _, slot, spec, initial, _ = unpack_battery(runtime._state)
    assert not hasattr(spec, "negative")
    assert all(not key.startswith("negative:") for key in slot.values)
    assert battery_spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .2})
    execute(runtime, {"operation": "wait", "duration_s": 4.})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, final_slot, _, final, _ = unpack_battery(runtime._state)
    assert final.time_s == pytest.approx(5.)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(1., abs=1e-8)
    assert final.amounts_mol["metal:Li"] == pytest.approx(
        initial.amounts_mol["metal:Li"]-1/F, abs=1e-12)
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-12)
    assert final_slot.spec.owner_kind == "equipment"
    assert final_slot.spec.batch_reset == "preserve"
    reading = battery_readout(spec, final, .2)
    assert reading.metal_amount_mol == final.amounts_mol["metal:Li"]
    assert not hasattr(reading, "negative_surface_fraction")
    assert {c.channel_id for c in result.observation.channels} == {
        "terminal_potential", "electrical_current"}


def test_voltage_control_and_meter_use_actual_half_cell_port():
    runtime = session()
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    target = battery_readout(spec, initial, .2).voltage_V
    execute(runtime, {"operation": "configure_electrical_control", "potential_V": target})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, program = unpack_battery(runtime._state)
    from chemworld.physchem.battery_models import battery_control_current

    current = battery_control_current(spec, final, program)
    assert 0 < current < spec.maximum_current_A
    assert battery_readout(spec, final, current).voltage_V == pytest.approx(target, abs=2e-6)
    assert final.time_s == 3.
    assert final.amounts_mol["metal:Li"] < initial.amounts_mol["metal:Li"]


def test_midqueue_snapshot_and_material_reset_preserve_real_metal():
    runtime = session()
    for current, duration in ((.2, 3.), (0., 2.), (-.1, 4.)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "wait", "duration_s": 9.})
    assert runtime._state.to_dict() == restored._state.to_dict()
    record, slot, spec, physical, programme = unpack_battery(runtime._state)
    assert physical.time_s == pytest.approx(9.)
    assert programme.value_at(physical.time_s) == 0.
    assert record.settings["electrical_control"]["stage_index"] == 3
    assert physical.amounts_mol["metal:Li"] == pytest.approx(
        spec.metal.initial_amount_mol-.2/F, abs=1e-12)
    before = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == before
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    after = unpack_battery(runtime._state)[3]
    assert after.time_s == 10.
    assert after.amounts_mol["metal:Li"] == physical.amounts_mol["metal:Li"]


def test_replacing_equipment_uses_new_finite_inventory_once():
    runtime = session()
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .2})
    execute(runtime, {"operation": "wait", "duration_s": 3.})
    record, _, _, previous, _ = unpack_battery(runtime._state)
    # Existing equipment replacement retains the declared housing volume. A
    # fresh compatible cell may have a different working stoichiometry, while
    # replacement with a different physical volume requires a new housing.
    new = replace(default_spec(cells=4), initial_positive_fraction=.4)
    replacement = replace(record, equipment_id="new-half-cell", settings={
        **record.settings, "physical_model": new.to_dict()})
    fresh = initial_battery_slot(replacement, new)
    runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                           "replacement": replacement, "slots": StateSlots((fresh,))})
    _, slot, spec, physical, _ = unpack_battery(runtime._state)
    assert slot == fresh and spec == new
    assert physical.amounts_mol["metal:Li"] == new.metal.initial_amount_mol
    assert physical.amounts_mol["metal:Li"] > previous.amounts_mol["metal:Li"]


def test_depleted_snapshot_stays_valid_and_reverse_current_replates():
    runtime = session(default_spec(cells=4, metal_thickness_m=1e-8))
    spec = unpack_battery(runtime._state)[2]
    total = sum(spec.initial_state().amounts_mol.values())
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .2})
    execute(runtime, {"operation": "wait", "duration_s": 20.})
    final = unpack_battery(runtime._state)[3]
    expected_time = F*spec.metal.initial_amount_mol/.2
    assert final.time_s == pytest.approx(expected_time, abs=1e-5)
    assert final.amounts_mol["metal:Li"] == pytest.approx(0., abs=1e-13)
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    assert unpack_battery(runtime._state)[3].time_s == final.time_s
    for active in (runtime, restored):
        execute(active, {"operation": "configure_electrical_control", "current_A": -.1})
        execute(active, {"operation": "wait", "duration_s": 2.})
        physical = unpack_battery(active._state)[3]
        assert physical.time_s == pytest.approx(expected_time+2., abs=1e-5)
        assert physical.amounts_mol["metal:Li"] == pytest.approx(.2/F, abs=1e-12)
        assert sum(physical.amounts_mol.values()) == pytest.approx(total, abs=1e-12)
    # The extra zero-duration stopped wait is an observed action; its resources
    # may differ, while the physical inventories and coordinates agree.
    left, right = (unpack_battery(r._state)[3] for r in (runtime, restored))
    assert left.amounts_mol == right.amounts_mol
    assert left.coordinates == right.coordinates


def test_half_cell_solver_failure_rolls_back_metal_particle_and_clock(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = session()
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected half-cell solver failure")

    monkeypatch.setattr(adapter, "advance_battery", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_half_cell_world_rejects_undefined_thermal_coupling():
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    with pytest.raises(ValueError, match=r"half.cell"):
        battery_world_spec(default_spec(cells=4), thermal_device=ThermalDeviceSpec())


def test_jacket_program_refusal_is_structured_and_does_not_advance():
    from chemworld.runtime.battery import BatteryRuntime

    runtime = session()
    before = runtime._state.to_dict()
    result = BatteryRuntime().advance(runtime._state, {"duration_s": 1.},
                                      jacket_program=object())
    assert not result.success and result.outcome == "rejected"
    assert result.failure_reason == "battery_requires_shared_thermal_device"
    assert result.diagnostics == {}
    assert runtime._state.to_dict() == before


def test_legacy_full_cell_snapshot_without_pressure_boundary_work_still_continues():
    from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec

    spec = BatterySPMSpec(negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=4))
    original = session(spec)
    execute(original, {"operation": "configure_electrical_control", "current_A": .1})
    payload = json.loads(json.dumps(original.snapshot().to_dict()))

    def remove_new_ledger_field(value):
        if isinstance(value, dict):
            value.pop("boundary_work_J", None)
            for item in value.values():
                remove_new_ledger_field(item)
        elif isinstance(value, list):
            for item in value:
                remove_new_ledger_field(item)

    remove_new_ledger_field(payload)
    restored = ChemWorldRuntime.from_snapshot(payload)
    for runtime in (original, restored):
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        assert runtime._state.ledger.boundary_work_J == 0.
    assert original._state.to_dict() == restored._state.to_dict()


@pytest.mark.parametrize("operation", [
    "soh", "energy", "dissipation", "chemical_potentials", "sei_affinity",
    "soc_configuration",
])
def test_undefined_full_cell_analysis_refuses_without_mutating_state(operation):
    from chemworld.physchem.battery_models import (
        battery_chemical_energy_j,
        battery_chemical_potentials,
        battery_dissipation_w,
        battery_sei_affinity_j_mol,
    )
    from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery

    spec = default_spec(cells=4)
    state = spec.initial_state()
    original = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    calls = {
        "soh": lambda: soh_from_battery(spec, state),
        "energy": lambda: battery_chemical_energy_j(spec, state),
        "dissipation": lambda: battery_dissipation_w(spec, state, .1),
        "chemical_potentials": lambda: battery_chemical_potentials(spec, state),
        "sei_affinity": lambda: battery_sei_affinity_j_mol(spec, state, .1),
        "soc_configuration": lambda: battery_spec_at_soc(spec, None, .5),
    }
    with pytest.raises(ValueError, match=r"half.cell"):
        calls[operation]()
    assert (dict(state.amounts_mol), dict(state.coordinates), state.time_s) == original


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_half_cell_example_replays_both_measurement_noise_modes(tmp_path, noise):
    result = run(tmp_path/f"half-cell-{noise}.jsonl", spec=default_spec(cells=4), noise=noise)
    assert result["steps"] == result["replay_steps"] == 11
