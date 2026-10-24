"""Half-cell EIS consumes original finite inventories without expanding chemistry."""

import json
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.examples.battery_half_cell import default_spec
from chemworld.physchem.battery_half_cell import METAL_KEY, BatteryHalfCellSPMSpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, F, battery_spec_from_dict
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def snapshot(state):
    return {item.name: dict(value) if isinstance(value, Mapping) else value
            for item in fields(state) for value in (getattr(state, item.name),)}


def test_codec_keeps_finite_metal_source_and_complete_original_response_layout():
    spec = default_spec(cells=4)
    payload = json.loads(json.dumps(spec.to_dict()))
    restored = battery_spec_from_dict(payload)
    assert isinstance(restored, BatteryHalfCellSPMSpec)
    assert restored == spec and restored.to_dict() == payload
    assert restored.metal.parameters == spec.metal.parameters
    assert restored.positive_ocv.source == spec.positive_ocv.source
    assert "numerical_form" not in payload and restored.numerical_form == "residual"
    assert not hasattr(restored, "negative")
    assert restored.inventory_keys == (METAL_KEY, *restored.positive.keys("positive"))
    assert restored.inventory_species == dict.fromkeys(restored.inventory_keys, "Li")
    assert restored.coordinate_keys == ("discharged_charge_C", "delivered_work_J")
    state = restored.initial_state()
    before = snapshot(state)
    system, initial = battery_impedance_system(restored, state)
    names = tuple(v.name for v in system.variables)
    assert names == (*spec.inventory_keys, *spec.coordinate_keys, *BATTERY_ALGEBRAIC)
    assert tuple(v.unit for v in system.variables) == (
        *("mol" for _ in spec.inventory_keys), "C", "J", "A", "V")
    assert "half-cell:control" in {e.name for e in system.equations}
    assert not initial.flags.writeable
    first = battery_impedance(spec, [0.003, 0.2], state=spec.initial_state())
    second = battery_impedance(restored, [0.003, 0.2], state=state)
    assert first.success and second.success
    assert first.response.to_dict() == second.response.to_dict()
    assert first.to_dict() == second.to_dict()
    assert all(not point.response.flags.writeable for point in second.response.points)
    assert json.loads(json.dumps(second.response.to_dict(), allow_nan=False))["success"]
    assert snapshot(state) == before


def test_frequency_analysis_keeps_the_existing_equipment_slot_and_housing():
    spec = default_spec(cells=4)
    world = battery_world_spec(spec)
    original = world.physics.initial_states["reactor"]
    before = original.to_dict()
    record, slot, decoded, state, _ = unpack_battery(original)
    expected = initial_battery_slot(record, decoded)
    assert slot == expected
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    assert tuple(slot.values) == (*spec.inventory_keys, *spec.coordinate_keys)
    assert slot.spec.state_type == "battery-lithium-and-charge-v1"
    assert state.volume_L == spec.volume_l
    assert state.volume_L == (spec.metal.volume_m3 + spec.positive.active_volume_m3) * 1000
    assert battery_impedance(decoded, [0.01], state=state).success
    assert original.to_dict() == before
    assert unpack_battery(original)[1] == expected


def test_valid_noninitial_equilibrium_preserves_passed_charge_work_and_time():
    spec = default_spec(cells=4)
    initial = spec.initial_state()
    # A declared uniform equilibrium consistent with both original Faraday
    # invariants; no integration or hidden relaxation manufactures this state.
    moved = 1e-5
    amounts = dict(initial.amounts_mol)
    amounts[METAL_KEY] -= moved
    for key, volume in zip(spec.positive.keys("positive"),
                           spec.positive.mesh.cell_volumes_m3, strict=True):
        amounts[key] += moved * volume / spec.positive.active_volume_m3
    state = replace(initial, amounts_mol=amounts, time_s=7., coordinates={
        "discharged_charge_C": F * moved, "delivered_work_J": 3.1 * F * moved,
    })
    before = snapshot(state)
    assert battery_impedance(spec, [0.003, 0.2], state=state).success
    assert snapshot(state) == before
    assert state.coordinates != initial.coordinates
    assert state.amounts_mol[METAL_KEY] != initial.amounts_mol[METAL_KEY]


@pytest.mark.parametrize("bad", [
    "missing_metal", "missing_positive", "extra_negative", "missing_charge", "missing_work",
    "extra_hysteresis", "metal_faraday", "working_faraday", "wrong_temperature", "wrong_volume",
    "negative_inventory", "nonfinite_inventory",
])
def test_analysis_refuses_invalid_layout_or_faradaic_state_without_repair(bad):
    spec = default_spec(cells=4)
    state = spec.initial_state()
    amounts, coordinates = dict(state.amounts_mol), dict(state.coordinates)
    key = spec.positive.keys("positive")[0]
    if bad == "missing_metal":
        amounts.pop(METAL_KEY)
    elif bad == "missing_positive":
        amounts.pop(key)
    elif bad == "extra_negative":
        amounts["negative:Li:0"] = 0.
    elif bad == "missing_charge":
        coordinates.pop("discharged_charge_C")
    elif bad == "missing_work":
        coordinates.pop("delivered_work_J")
    elif bad == "extra_hysteresis":
        coordinates["battery:positive:hysteresis"] = 0.
    elif bad == "metal_faraday":
        amounts[METAL_KEY] *= 0.99
    elif bad == "working_faraday":
        amounts[key] *= 0.99
    elif bad == "wrong_temperature":
        state = replace(state, temperature_K=state.temperature_K + 1.)
    elif bad == "wrong_volume":
        state = replace(state, volume_L=state.volume_L * 1.01)
    elif bad == "negative_inventory":
        amounts[METAL_KEY] = -1e-6
    else:
        amounts[METAL_KEY] = np.inf
    if bad in {"negative_inventory", "nonfinite_inventory"}:
        before = snapshot(state)
        with pytest.raises(ValueError, match=r"negative|finite"):
            replace(state, amounts_mol=amounts, coordinates=coordinates)
        assert snapshot(state) == before
        return
    state = replace(state, amounts_mol=amounts, coordinates=coordinates)
    before = snapshot(state)
    with pytest.raises(ValueError):
        battery_impedance(spec, [0.01], state=state)
    assert snapshot(state) == before


@pytest.mark.parametrize("frequencies", [[], [0.], [-1.], [np.nan], [np.inf]])
def test_invalid_frequency_requests_do_not_change_finite_inventory(frequencies):
    spec = default_spec(cells=4)
    state = spec.initial_state()
    before = snapshot(state)
    with pytest.raises(ValueError, match="finite positive Hz"):
        battery_impedance(spec, frequencies, state=state)
    assert snapshot(state) == before


@pytest.mark.parametrize("undeclared", [
    "thermal", "sei", "plating", "positive_ocp", "electrolyte", "negative",
])
def test_eis_does_not_add_undeclared_half_cell_chemistries_to_codec(undeclared):
    payload = json.loads(json.dumps(default_spec(cells=4).to_dict()))
    payload[undeclared] = None
    before = json.loads(json.dumps(payload))
    with pytest.raises((TypeError, ValueError), match=undeclared):
        battery_spec_from_dict(payload)
    assert payload == before
