"""NT frequency analysis retains two mean stocks and the original electrical layout."""

import json
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.examples.battery_newman_tobias import default_spec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, F, battery_spec_from_dict
from chemworld.physchem.battery_newman_tobias import BatteryNewmanTobiasSpec
from chemworld.physchem.battery_porous_electrical import PHI_E
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def snapshot(state):
    return {item.name: dict(value) if isinstance(value, Mapping) else value
            for item in fields(state) for value in (getattr(state, item.name),)}


def test_sourced_codec_and_original_thickness_algebraics_survive_eis_roundtrip():
    source = InputSource(DataSourceProvenance(
        "nt-eis-compat-synthetic", 0, "Authored compatibility OCV", "1", "CC0-1.0",
    ), "synthetic")
    spec = replace(default_spec(cells=(3, 2, 5)), negative_ocv=ElectrodeOCV(
        "nt-negative", source, polynomial_V=(0.25, -0.15, 0.02),
        supported_models=("NewmanTobias",),
    ))
    payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    restored = battery_spec_from_dict(payload)
    assert isinstance(restored, BatteryNewmanTobiasSpec)
    assert restored == spec and restored.to_dict() == payload
    assert restored.negative_ocv == spec.negative_ocv
    assert restored.negative_ocv.source == source
    assert restored.geometry == spec.geometry
    assert "numerical_form" not in payload and restored.numerical_form == "residual"
    assert restored.inventory_keys == ("negative:Li", "positive:Li")
    assert restored.inventory_species == dict.fromkeys(restored.inventory_keys, "Li")
    assert restored.coordinate_keys == ("discharged_charge_C", "delivered_work_J")
    assert not hasattr(restored.negative, "cells")
    assert not hasattr(restored.positive, "diffusivity_m2_s")

    state = restored.initial_state()
    before = snapshot(state)
    system, initial = battery_impedance_system(restored, state)
    expected_reactions = (
        *(f"battery:negative:reaction:{i}_A" for i in range(2)),
        *(f"battery:positive:reaction:{i}_A" for i in range(4)),
    )
    assert tuple(v.name for v in system.variables) == (
        *spec.inventory_keys, *spec.coordinate_keys, *expected_reactions, PHI_E,
        *BATTERY_ALGEBRAIC,
    )
    assert tuple(v.unit for v in system.variables) == (
        "mol", "mol", "C", "J", *("A" for _ in expected_reactions), "V", "A", "V",
    )
    assert "battery:control" in {equation.name for equation in system.equations}
    assert not initial.flags.writeable
    first = battery_impedance(spec, [0.003, 0.2], state=spec.initial_state())
    second = battery_impedance(restored, [0.003, 0.2], state=state)
    assert first.success and second.success
    assert first.response.to_dict() == second.response.to_dict()
    assert first.to_dict() == second.to_dict()
    assert all(not point.response.flags.writeable for point in second.response.points)
    assert json.loads(json.dumps(second.response.to_dict(), allow_nan=False))["success"]
    assert snapshot(state) == before
    # Derived electrical caches and ephemeral circuit solutions are not new
    # serializable model inputs or persistent inventories.
    assert restored.to_dict() == payload
    assert state.algebraic_values == {}


def test_eis_preserves_original_mean_li_slot_and_total_porous_housing():
    spec = default_spec(cells=(3, 2, 4))
    original = battery_world_spec(spec).physics.initial_states["reactor"]
    before = original.to_dict()
    record, slot, decoded, state, _ = unpack_battery(original)
    expected = initial_battery_slot(record, decoded)
    assert slot == expected
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    assert slot.spec.state_type == "battery-lithium-and-charge-v1"
    assert tuple(slot.values) == (*spec.inventory_keys, *spec.coordinate_keys)
    assert len(slot.values) == 4
    assert state.volume_L == spec.geometry.area_m2 * sum(spec.geometry.lengths_m) * 1000
    assert battery_impedance(decoded, [0.01], state=state).success
    assert original.to_dict() == before
    assert unpack_battery(original)[1] == expected


@pytest.mark.parametrize("moved", [-1e-5, 1e-5])
def test_noninitial_mean_state_and_signed_receipts_are_not_reinitialized(moved):
    spec = default_spec(cells=(3, 2, 4))
    initial = spec.initial_state()
    amounts = dict(initial.amounts_mol)
    amounts["negative:Li"] -= moved
    amounts["positive:Li"] += moved
    state = replace(initial, amounts_mol=amounts, time_s=7., coordinates={
        "discharged_charge_C": F * moved, "delivered_work_J": 3.8 * F * moved,
    })
    before = snapshot(state)
    assert battery_impedance(spec, [0.003, 0.2], state=state).success
    assert snapshot(state) == before
    assert state.coordinates != initial.coordinates
    assert state.amounts_mol != initial.amounts_mol


@pytest.mark.parametrize("bad", [
    "missing_negative", "missing_positive", "extra_radial_stock", "extra_salt_stock",
    "missing_charge", "missing_work", "extra_electrical_coordinate", "negative_faraday",
    "positive_faraday", "wrong_temperature", "wrong_volume", "negative_inventory",
    "nonfinite_inventory",
])
def test_analysis_refuses_bad_mean_inventory_or_layout_without_repair(bad):
    spec = default_spec(cells=(3, 2, 4))
    state = spec.initial_state()
    amounts, coordinates = dict(state.amounts_mol), dict(state.coordinates)
    if bad == "missing_negative":
        amounts.pop("negative:Li")
    elif bad == "missing_positive":
        amounts.pop("positive:Li")
    elif bad == "extra_radial_stock":
        amounts["negative:Li:0"] = 0.
    elif bad == "extra_salt_stock":
        amounts["electrolyte:salt:0"] = 0.
    elif bad == "missing_charge":
        coordinates.pop("discharged_charge_C")
    elif bad == "missing_work":
        coordinates.pop("delivered_work_J")
    elif bad == "extra_electrical_coordinate":
        coordinates[PHI_E] = -0.13
    elif bad == "negative_faraday":
        amounts["negative:Li"] *= 0.99
    elif bad == "positive_faraday":
        amounts["positive:Li"] *= 0.99
    elif bad == "wrong_temperature":
        state = replace(state, temperature_K=state.temperature_K + 1.)
    elif bad == "wrong_volume":
        state = replace(state, volume_L=state.volume_L * 1.01)
    elif bad == "negative_inventory":
        amounts["negative:Li"] = -1e-6
    else:
        amounts["negative:Li"] = np.inf
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
def test_bad_frequencies_cannot_change_means_or_circuit_state(frequencies):
    spec = default_spec(cells=(3, 2, 4))
    state = spec.initial_state()
    before = snapshot(state)
    with pytest.raises(ValueError, match="finite positive Hz"):
        battery_impedance(spec, frequencies, state=state)
    assert snapshot(state) == before


@pytest.mark.parametrize("undeclared", [
    "thermal", "sei", "plating", "negative_ocp", "positive_ocp", "electrolyte",
    "numerical_form",
])
def test_eis_support_does_not_expand_nt_chemical_or_numerical_schema(undeclared):
    payload = json.loads(json.dumps(default_spec(cells=(3, 2, 4)).to_dict()))
    payload[undeclared] = None
    before = json.loads(json.dumps(payload))
    with pytest.raises((TypeError, ValueError), match=undeclared):
        battery_spec_from_dict(payload)
    assert payload == before


@pytest.mark.parametrize("electrode", ["negative", "positive"])
@pytest.mark.parametrize("undeclared", ["cells", "diffusivity_m2_s"])
def test_eis_does_not_promote_mean_electrodes_to_radial_models(electrode, undeclared):
    payload = json.loads(json.dumps(default_spec(cells=(3, 2, 4)).to_dict()))
    payload[electrode][undeclared] = 4 if undeclared == "cells" else 1e-13
    before = json.loads(json.dumps(payload))
    with pytest.raises((TypeError, ValueError), match=undeclared):
        battery_spec_from_dict(payload)
    assert payload == before
