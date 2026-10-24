"""MSMR EIS consumes sourced reactions and the existing unique Li inventories."""

import json
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.examples.battery_msmr import default_spec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, F, battery_spec_from_dict
from chemworld.physchem.battery_msmr import BatteryMSMRSPMSpec
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def snapshot(state):
    return {item.name: dict(value) if isinstance(value, Mapping) else value
            for item in fields(state) for value in (getattr(state, item.name),)}


def test_codec_keeps_reaction_identity_and_one_inventory_per_radial_shell():
    spec = default_spec(cells=4)
    payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    restored = battery_spec_from_dict(payload)
    assert isinstance(restored, BatteryMSMRSPMSpec)
    assert restored == spec and restored.to_dict() == payload
    assert "numerical_form" not in payload and restored.numerical_form == "residual"
    for electrode in ("negative", "positive"):
        original_material = getattr(spec, electrode + "_material")
        material = getattr(restored, electrode + "_material")
        assert material.parameters == original_material.parameters
        assert material.material_id == original_material.material_id
        assert material.reaction_row_ids == original_material.reaction_row_ids
        assert material.reaction_parameters == original_material.reaction_parameters
        assert len(material.reaction_row_ids) == 2
    assert restored.inventory_keys == (
        *spec.negative.keys("negative"), *spec.positive.keys("positive"),
    )
    assert restored.inventory_species == dict.fromkeys(restored.inventory_keys, "Li")
    assert restored.coordinate_keys == ("discharged_charge_C", "delivered_work_J")
    assert not any("reaction" in key or "potential" in key for key in restored.inventory_keys)

    state = restored.initial_state()
    before = snapshot(state)
    system, initial = battery_impedance_system(restored, state)
    assert tuple(v.name for v in system.variables) == (
        *spec.inventory_keys, *spec.coordinate_keys, *BATTERY_ALGEBRAIC,
    )
    assert tuple(v.unit for v in system.variables) == (
        *("mol" for _ in spec.inventory_keys), "C", "J", "A", "V",
    )
    assert "msmr:control" in {equation.name for equation in system.equations}
    assert not initial.flags.writeable
    first = battery_impedance(spec, [0.003, 0.2], state=spec.initial_state())
    second = battery_impedance(restored, [0.003, 0.2], state=state)
    assert first.success and second.success
    assert first.response.to_dict() == second.response.to_dict()
    assert first.to_dict() == second.to_dict()
    assert all(not point.response.flags.writeable for point in second.response.points)
    assert json.loads(json.dumps(second.response.to_dict(), allow_nan=False))["success"]
    assert snapshot(state) == before


def test_eis_preserves_original_equipment_slot_and_does_not_add_reaction_housing():
    spec = default_spec(cells=4)
    world = battery_world_spec(spec)
    original = world.physics.initial_states["reactor"]
    before = original.to_dict()
    record, slot, decoded, state, _ = unpack_battery(original)
    expected = initial_battery_slot(record, decoded)
    assert slot == expected
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    assert slot.spec.state_type == "battery-lithium-and-charge-v1"
    assert tuple(slot.values) == (*spec.inventory_keys, *spec.coordinate_keys)
    assert len(slot.values) == spec.negative.cells + spec.positive.cells + 2
    assert state.volume_L == (
        spec.negative.active_volume_m3 + spec.positive.active_volume_m3
    ) * 1000
    assert battery_impedance(decoded, [0.01], state=state).success
    assert original.to_dict() == before
    assert unpack_battery(original)[1] == expected


@pytest.mark.parametrize("moved", [-1e-5, 1e-5])
def test_declared_noninitial_rest_preserves_both_faradaic_balances_and_receipts(moved):
    spec = default_spec(cells=4)
    initial = spec.initial_state()
    amounts = dict(initial.amounts_mol)
    # An explicitly supplied uniform state with two consistent Faraday
    # inventories, not a relaxation or an alteration performed by analysis.
    for electrode, sign in (("negative", -1.), ("positive", 1.)):
        particle = getattr(spec, electrode)
        for key, volume in zip(particle.keys(electrode),
                               particle.mesh.cell_volumes_m3, strict=True):
            amounts[key] += sign * moved * volume / particle.active_volume_m3
    state = replace(initial, amounts_mol=amounts, time_s=7., coordinates={
        "discharged_charge_C": F * moved, "delivered_work_J": 3.8 * F * moved,
    })
    before = snapshot(state)
    assert battery_impedance(spec, [0.003, 0.2], state=state).success
    assert snapshot(state) == before
    assert state.coordinates != initial.coordinates
    assert state.amounts_mol != initial.amounts_mol


@pytest.mark.parametrize("bad", [
    "missing_negative", "missing_positive", "extra_reaction_stock", "missing_charge",
    "missing_work", "extra_potential_coordinate", "negative_faraday", "positive_faraday",
    "wrong_temperature", "wrong_volume", "negative_inventory", "nonfinite_inventory",
])
def test_eis_rejects_invalid_original_state_without_repair(bad):
    spec = default_spec(cells=4)
    state = spec.initial_state()
    amounts, coordinates = dict(state.amounts_mol), dict(state.coordinates)
    negative = spec.negative.keys("negative")[0]
    positive = spec.positive.keys("positive")[0]
    if bad == "missing_negative":
        amounts.pop(negative)
    elif bad == "missing_positive":
        amounts.pop(positive)
    elif bad == "extra_reaction_stock":
        amounts["negative:reaction:0:Li"] = 0.
    elif bad == "missing_charge":
        coordinates.pop("discharged_charge_C")
    elif bad == "missing_work":
        coordinates.pop("delivered_work_J")
    elif bad == "extra_potential_coordinate":
        coordinates["negative:potential:0"] = 0.2
    elif bad == "negative_faraday":
        amounts[negative] *= 0.99
    elif bad == "positive_faraday":
        amounts[positive] *= 0.99
    elif bad == "wrong_temperature":
        state = replace(state, temperature_K=state.temperature_K + 1.)
    elif bad == "wrong_volume":
        state = replace(state, volume_L=state.volume_L * 1.01)
    elif bad == "negative_inventory":
        amounts[negative] = -1e-6
    else:
        amounts[negative] = np.inf
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
def test_invalid_frequency_request_preserves_every_shell_and_receipt(frequencies):
    spec = default_spec(cells=4)
    state = spec.initial_state()
    before = snapshot(state)
    with pytest.raises(ValueError, match="finite positive Hz"):
        battery_impedance(spec, frequencies, state=state)
    assert snapshot(state) == before


@pytest.mark.parametrize("undeclared", [
    "thermal", "sei", "plating", "negative_ocp", "positive_ocp", "electrolyte",
    "numerical_form",
])
def test_eis_does_not_expand_msmr_codec_to_undeclared_chemistry(undeclared):
    payload = json.loads(json.dumps(default_spec(cells=4).to_dict()))
    payload[undeclared] = None
    before = json.loads(json.dumps(payload))
    with pytest.raises((TypeError, ValueError), match=undeclared):
        battery_spec_from_dict(payload)
    assert payload == before
