"""Finite metal Li belongs to the original device, not to an analysis shortcut."""

import json
from dataclasses import replace
from math import fsum

import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_degradation import SEISpec
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    ParticleSpec,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_dissipation_w,
    battery_sei_affinity_j_mol,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec
from test_battery_hysteresis_analysis import operational

METAL = ("plating:Li", "dead:Li")
OLD_FIELDS = {
    "negative", "positive", "initial_negative_fraction", "initial_positive_fraction",
    "negative_ocv_intercept_V", "negative_ocv_slope_V", "positive_ocv_intercept_V",
    "positive_ocv_slope_V", "temperature_K", "surface_margin", "minimum_voltage_V",
    "maximum_voltage_V", "maximum_current_A", "series_resistance_ohm", "thermal", "sei",
    "provenance", "numerical_form",
}


def plating(*, mode="reversible", rate=1e-10, plated=2e-5, dead=3e-5):
    from chemworld.physchem.battery_plating import LithiumPlatingSpec

    source = InputSource(DataSourceProvenance(
        "plating-compatibility-synthetic", 0, "Explicit finite metal compatibility inputs",
        "1", "CC0-1.0",
    ), "synthetic")
    values = {
        "k_m_s": (rate, "m/s"),
        "alpha": (0.45, "dimensionless"),
        "ce_mol_m3": (1000., "mol/m^3"),
        "negative_electrode_volume_m3": (2e-6, "m^3"),
    }
    parameters = ScientificTable(
        "plating-compatibility", "parameters",
        tuple(QuantityColumn(name, unit, "parameter", (0., 1e6), "Explicit synthetic input")
              for name, (_, unit) in values.items()),
        (source,),
        (ScientificRow("metal", source.provenance.source_id, "group", "sample", "parameters",
                       {name: InputValue(value, unit, None, "Fixed synthetic input")
                        for name, (value, unit) in values.items()}),),
    )
    return LithiumPlatingSpec(parameters, "metal", mode, plated, dead)


def old_spec(family="SPM"):
    particles = {"negative": ParticleSpec(cells=3), "positive": ParticleSpec(cells=4)}
    if family == "MPM":
        return BatteryMPMSpec(**particles)
    if family == "DFN":
        return BatteryDFNSpec(**particles, electrolyte=BatteryElectrolyteSpec(cells=(2, 2, 2)))
    if family == "SPMe":
        return BatterySPMSpec(**particles, electrolyte=BatteryElectrolyteSpec(cells=(2, 2, 2)))
    return BatterySPMSpec(**particles)


@pytest.mark.parametrize("family", ("SPM", "MPM", "SPMe", "DFN"))
def test_absent_plating_keeps_old_codec_geometry_and_device_slots(family):
    spec = old_spec(family)
    old_fields = set(OLD_FIELDS)
    if family in {"SPMe", "DFN"}:
        old_fields.add("electrolyte")
    if family == "MPM":
        old_fields.update(("negative_distribution", "positive_distribution", "model_family"))
    if family == "DFN":
        old_fields.add("model_family")
    payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    assert set(payload) == old_fields
    assert "plating" not in payload
    assert replace(spec, plating=None).to_dict() == payload
    restored = battery_spec_from_dict(payload)
    assert type(restored) is type(spec) and restored.to_dict() == payload
    expected_volume = (
        spec.electrolyte.area_m2 * sum(spec.electrolyte.lengths_m) * 1000
        if spec.electrolyte is not None
        else (spec.negative.active_volume_m3 + spec.positive.active_volume_m3) * 1000
    )
    assert restored.volume_l == expected_volume
    state = battery_world_spec(restored).physics.initial_states["reactor"]
    record, slot, decoded, physical, _ = unpack_battery(state)
    assert record.equipment_type == "single_particle_battery"
    assert slot.spec.state_type == "battery-lithium-and-charge-v1"
    assert slot.spec.owner_kind == "equipment" and slot.spec.owner_id == record.equipment_id
    assert slot.spec.batch_reset == "preserve" and decoded.to_dict() == payload
    assert set(slot.values) == set(spec.inventory_keys) | {
        "discharged_charge_C", "delivered_work_J"}
    assert not set(METAL) & set(slot.values)
    assert physical.amounts_mol == spec.initial_state().amounts_mol


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
def test_metal_source_codec_roundtrips_without_dataclass_copying_immutable_tables(mode):
    declaration = plating(mode=mode)
    spec = replace(old_spec(), plating=declaration)
    payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    restored = battery_spec_from_dict(payload)
    assert restored.to_dict() == payload == spec.to_dict()
    assert restored.plating.parameters.to_dict() == declaration.parameters.to_dict()
    assert restored.plating.parameter_row_id == "metal"
    assert restored.plating.native_parameters == declaration.native_parameters
    assert restored.initial_state() == spec.initial_state()
    assert spec.initial_state().amounts_mol["plating:Li"] == 2e-5
    assert spec.initial_state().amounts_mol["dead:Li"] == 3e-5
    assert spec.volume_l == (2e-6 + spec.positive.active_volume_m3) * 1000
    assert spec.negative.capacity_mol == old_spec().negative.capacity_mol
    assert spec.positive.capacity_mol == old_spec().positive.capacity_mol
    corrupted = json.loads(json.dumps(payload))
    corrupted["plating"]["undeclared_decay"] = 1.0
    with pytest.raises(ValueError):
        battery_spec_from_dict(corrupted)
    assert spec.to_dict() == payload


def test_metal_has_one_equipment_owner_and_repeated_declaration_does_not_add_lithium():
    spec = replace(old_spec(), plating=plating())
    world = battery_world_spec(spec)
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=9)
    try:
        state = runtime._state
        before = runtime.snapshot().to_dict()
        record, slot, decoded, physical, _ = unpack_battery(state)
        assert tuple(spec.inventory_keys[-2:]) == METAL
        assert all(decoded.inventory_species[key] == "Li" for key in METAL)
        fields = {field.name: field for field in slot.spec.fields}
        for key in METAL:
            assert fields[key].unit == "mol" and fields[key].species_id == "Li"
            assert slot.values[key] == spec.initial_state().amounts_mol[key]
        assert slot.spec.owner_kind == "equipment" and slot.spec.owner_id == record.equipment_id
        assert slot.spec.batch_reset == "preserve"
        assert set(physical.coordinates) == {"discharged_charge_C", "delivered_work_J"}
        assert all(value == 0 for value in state.species_amounts.values())
        assert all(not phase.species_amounts_mol for phase in state.phases.phases.values())
        expected = fsum(physical.amounts_mol.values())
        inventory = runtime.constitution.material_inventory(state)
        assert inventory == StateSlots((slot,)).species_amounts()
        assert inventory == pytest.approx({"Li": expected}, rel=0, abs=2e-17)
        assert runtime.constitution.element_totals(inventory) == pytest.approx(
            {"Li": expected}, rel=0, abs=2e-17)
        for _ in range(3):
            assert initial_battery_slot(record, decoded) == slot
            assert unpack_battery(state)[3].amounts_mol == physical.amounts_mol
        assert runtime.snapshot().to_dict() == before
    finally:
        runtime.close()


@pytest.mark.parametrize("unsupported", (
    "DFN", "SPMe", "thermal", "sei", "current_sigmoid", "ode",
))
def test_plating_rejects_unimplemented_combinations_before_state_construction(unsupported):
    spec = old_spec(unsupported) if unsupported in {"MPM", "DFN", "SPMe"} else old_spec()
    options = {"plating": plating()}
    if unsupported == "thermal":
        options["thermal"] = BatteryThermalSpec()
    elif unsupported == "sei":
        options["sei"] = SEISpec()
    elif unsupported == "current_sigmoid":
        options["negative_ocp"] = operational("negative", unsupported)
    elif unsupported == "ode":
        options["numerical_form"] = "ode"
    with pytest.raises(ValueError, match=r"(?i)plating"):
        replace(spec, **options)


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
@pytest.mark.parametrize("analysis", ("impedance_system", "impedance", "soh", "soc"))
def test_declared_plating_rejects_automatic_analysis_even_with_zero_rate_and_zero_metal(
    mode, analysis
):
    spec = replace(old_spec(), plating=plating(mode=mode, rate=0., plated=0., dead=0.))
    state = spec.initial_state()
    configuration = spec.to_dict()
    calls = {
        "impedance_system": lambda: battery_impedance_system(spec, state),
        "impedance": lambda: battery_impedance(spec, [1.], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "soc": lambda: battery_spec_at_soc(spec, None, .5),
    }
    with pytest.raises(ValueError, match=r"(?i)plating"):
        calls[analysis]()
    assert state == spec.initial_state() and spec.to_dict() == configuration


def test_old_chemical_energy_and_sei_helpers_do_not_silently_omit_metal_potential():
    spec = replace(old_spec(), plating=plating())
    state = spec.initial_state()
    for analyze in (
        lambda: battery_chemical_energy_j(spec, state),
        lambda: battery_chemical_potentials(spec, state),
        lambda: battery_dissipation_w(spec, state, 0.),
        lambda: battery_sei_affinity_j_mol(spec, state, 0.),
    ):
        with pytest.raises(ValueError, match=r"(?i)plating"):
            analyze()
    assert state == spec.initial_state()


def test_old_equilibrium_analyses_and_sei_exclusions_remain_distinct():
    plain = old_spec()
    explicit_none = replace(plain, plating=None)
    left = battery_impedance(plain, [.1, 1.])
    right = battery_impedance(explicit_none, [.1, 1.])
    assert left.success and right.success
    assert left.to_dict() == right.to_dict()
    assert soh_from_battery(plain).to_dict() == soh_from_battery(explicit_none).to_dict()
    ageing = replace(plain, thermal=BatteryThermalSpec(), sei=SEISpec())
    state = ageing.initial_state()
    assert "plating" not in ageing.to_dict() and not set(METAL) & set(state.amounts_mol)
    with pytest.raises(ValueError, match="without side reactions"):
        battery_impedance(ageing, [1.], state=state)
    result = soh_from_battery(ageing, state)
    active = (*ageing.negative.keys("negative"), *ageing.positive.keys("positive"))
    assert result.spec.known_capacity_C == pytest.approx(
        F * fsum(state.amounts_mol[key] for key in active), rel=2e-15)
    assert state == ageing.initial_state()
