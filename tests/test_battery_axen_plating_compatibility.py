"""Persistent Axen memory and finite metal through the original battery owner."""

import json
from dataclasses import replace
from math import fsum

import pytest

from chemworld.foundation.state_ownership import StateSlot, StateSlots
from chemworld.physchem.battery_degradation import SEISpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import (
    BatteryThermalSpec,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_dissipation_w,
    battery_readout,
    battery_sei_affinity_j_mol,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec
from test_battery_hysteresis_analysis import operational
from test_battery_partial_plating_compatibility import partial
from test_battery_plating_compatibility import METAL, old_spec, plating

COMBINATIONS = (("axen", "single"), ("single", "axen"), ("axen", "axen"))


def model(modes=("axen", "axen"), **plating_options):
    return replace(old_spec(), plating=plating(**plating_options), **{
        name + "_ocp": operational(name, mode, initial_h=h)
        for name, mode, h in zip(("negative", "positive"), modes, (.2, -.3), strict=True)
    })


@pytest.mark.parametrize("modes", COMBINATIONS)
def test_supported_single_axen_combinations_roundtrip_all_original_declarations(modes):
    spec = model(modes)
    encoded = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    restored = battery_spec_from_dict(encoded)
    assert restored.to_dict() == encoded
    assert restored.plating.parameters.to_dict() == spec.plating.parameters.to_dict()
    assert restored.plating.mode == "reversible"
    assert restored.plating.to_dict()["model_version"] == "pybamm-25.4.2-spm-plating-v1"
    assert len(restored.plating.native_parameters) == 4
    assert restored.initial_state() == spec.initial_state()
    expected_keys = tuple(f"battery:{name}:hysteresis" for name, mode in
                          zip(("negative", "positive"), modes, strict=True) if mode == "axen")
    assert spec.hysteresis_keys == expected_keys
    assert set(spec.coordinate_keys) == {"discharged_charge_C", "delivered_work_J", *expected_keys}
    assert spec.inventory_keys == (*old_spec().inventory_keys, *METAL)
    assert spec.initial_state().amounts_mol == replace(
        old_spec(), plating=spec.plating).initial_state().amounts_mol
    for name, mode in zip(("negative", "positive"), modes, strict=True):
        original, decoded = getattr(spec, name + "_ocp"), getattr(restored, name + "_ocp")
        assert decoded.to_dict() == original.to_dict()
        if mode == "axen":
            assert decoded.parameters.to_dict() == original.parameters.to_dict()
            with pytest.raises(TypeError):
                decoded._native["initial_h"] = .9
    assert battery_readout(restored, restored.initial_state(), .05) == battery_readout(
        spec, spec.initial_state(), .05)


@pytest.mark.parametrize("modes", COMBINATIONS)
def test_hysteresis_and_metal_have_one_original_equipment_owner(modes):
    spec = model(modes)
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=46)
    restored = None
    try:
        before = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
        restored = ChemWorldRuntime.from_snapshot(before)
        record, slot, decoded, state, _ = unpack_battery(restored._state)
        assert record.equipment_type == "single_particle_battery"
        assert slot.spec.state_type == "battery-lithium-and-charge-v1"
        assert slot.spec.owner_kind == "equipment" and slot.spec.owner_id == record.equipment_id
        assert slot.spec.batch_reset == "preserve"
        assert set(slot.values) == set(spec.inventory_keys) | set(spec.coordinate_keys)
        fields = {field.name: field for field in slot.spec.fields}
        for key in spec.hysteresis_keys:
            assert fields[key].unit == "dimensionless" and fields[key].species_id is None
            assert fields[key].minimum == -1 and fields[key].maximum == 1
            assert state.coordinates[key] == spec.initial_state().coordinates[key]
        for key in METAL:
            assert fields[key].unit == "mol" and fields[key].species_id == "Li"
            assert state.amounts_mol[key] == spec.plating.initial_amounts()[key]
        inventory = restored.constitution.material_inventory(restored._state)
        assert inventory == StateSlots((slot,)).species_amounts()
        assert inventory == pytest.approx(
            {"Li": fsum(state.amounts_mol.values())}, rel=0, abs=2e-17)
        assert restored.constitution.element_totals(inventory) == inventory
        assert all(value == 0 for value in restored._state.species_amounts.values())
        assert all(not phase.species_amounts_mol
                   for phase in restored._state.phases.phases.values())
        for _ in range(2):
            assert initial_battery_slot(record, decoded) == slot
            assert unpack_battery(restored._state)[3] == state
        assert json.loads(json.dumps(restored.snapshot().to_dict())) == before
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("h", (1.000001, float("inf"), float("nan")))
def test_equipment_slot_rejects_hysteresis_above_declared_upper_bound(h):
    spec = model()
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=46)
    try:
        _, slot, _, _, _ = unpack_battery(runtime._state)
        payload = slot.to_dict()
        payload["values"][spec.hysteresis_keys[0]] = h
        with pytest.raises(ValueError, match=r"upper bound|finite"):
            StateSlot.from_dict(payload)
    finally:
        runtime.close()


@pytest.mark.parametrize("defect", (
    "missing-negative-h", "missing-positive-h", "extra-h", "h-high", "h-low", "h-nan",
    "missing-plated", "missing-dead", "negative-plated", "wrong-faraday",
))
def test_invalid_accepted_state_is_rejected_without_h_or_inventory_projection(defect):
    spec = model()
    state = spec.initial_state()
    coordinates, amounts = dict(state.coordinates), dict(state.amounts_mol)
    if defect.startswith("missing-") and defect.endswith("-h"):
        del coordinates[f"battery:{defect.split('-')[1]}:hysteresis"]
    elif defect == "extra-h":
        coordinates["battery:invented:hysteresis"] = 0.
    elif defect.startswith("h-"):
        coordinates[spec.hysteresis_keys[0]] = {"h-high": 1.01, "h-low": -1.01,
                                               "h-nan": float("nan")}[defect]
    elif defect in {"missing-plated", "missing-dead"}:
        del amounts["plating:Li" if defect == "missing-plated" else "dead:Li"]
    elif defect == "negative-plated":
        amounts["plating:Li"] = -1e-8
    else:
        amounts["plating:Li"] += 1e-5
    with pytest.raises(ValueError):
        bad = replace(state, coordinates=coordinates, amounts_mol=amounts)
        battery_readout(spec, bad, 0.)
    assert state == spec.initial_state()


@pytest.mark.parametrize("h", (-1., 1.))
def test_valid_h_endpoints_remain_exact_accepted_states(h):
    spec = model()
    state = spec.initial_state()
    coordinates = {**state.coordinates, **dict.fromkeys(spec.hysteresis_keys, h)}
    endpoint = replace(state, coordinates=coordinates)
    battery_readout(spec, endpoint, 0.)
    assert all(endpoint.coordinates[key] == h for key in spec.hysteresis_keys)
    assert endpoint.amounts_mol == state.amounts_mol


@pytest.mark.parametrize("unsupported", (
    "negative-sigmoid", "positive-sigmoid", "irreversible", "partial",
    "MPM", "DFN", "SPMe", "thermal", "sei", "ode",
))
def test_joint_scope_is_not_inherited_by_unimplemented_modes_or_families(unsupported):
    base = old_spec(unsupported) if unsupported in {"MPM", "DFN", "SPMe"} else old_spec()
    options = {"negative_ocp": operational("negative", "axen"),
               "positive_ocp": operational("positive", "axen"), "plating": plating()}
    if unsupported.endswith("-sigmoid"):
        name = unsupported.split("-")[0]
        options[name + "_ocp"] = operational(name, "current_sigmoid")
    elif unsupported == "irreversible":
        options["plating"] = plating(mode="irreversible")
    elif unsupported == "partial":
        options["plating"] = partial()
    elif unsupported == "thermal":
        options["thermal"] = BatteryThermalSpec()
    elif unsupported == "sei":
        options["sei"] = SEISpec()
    elif unsupported == "ode":
        options["numerical_form"] = "ode"
    with pytest.raises(ValueError):
        replace(base, **options)


@pytest.mark.parametrize("analysis", (
    "eis-system", "eis", "soh", "soc", "energy", "chemical-potentials", "dissipation", "sei",
))
def test_declared_joint_model_refuses_undeclared_analysis_even_at_zero_plating_rate(analysis):
    spec = model(rate=0., plated=0., dead=0.)
    state, encoded = spec.initial_state(), spec.to_dict()
    calls = {
        "eis-system": lambda: battery_impedance_system(spec, state),
        "eis": lambda: battery_impedance(spec, [1.], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "soc": lambda: battery_spec_at_soc(spec, None, .5),
        "energy": lambda: battery_chemical_energy_j(spec, state),
        "chemical-potentials": lambda: battery_chemical_potentials(spec, state),
        "dissipation": lambda: battery_dissipation_w(spec, state, 0.),
        "sei": lambda: battery_sei_affinity_j_mol(spec, state, 0.),
    }
    with pytest.raises(ValueError):
        calls[analysis]()
    assert state == spec.initial_state() and spec.to_dict() == encoded
