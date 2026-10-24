"""MPM side-current compatibility through the original public battery consumers.

Small limiting cases complement, rather than replace, the fixed independent
reference block. No trajectory generated here is stored as a golden reference.
"""

import json
from dataclasses import asdict, replace
from math import exp, fsum

import numpy as np
import pytest

from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_degradation import SEISpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    ParticleSpec,
    advance_battery,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_dissipation_w,
    battery_particle_flux,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.battery_plating import plating_fields
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.physchem.electrochemistry import R_J_PER_MOL_K as R
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec
from test_battery_hysteresis_analysis import operational
from test_battery_plating_compatibility import METAL, OLD_FIELDS, old_spec, plating


def distribution(radii, weights, basis="volume"):
    return ParticleSizeDistribution.from_weights(
        radii, weights, basis=basis, source=plating().parameters.sources[0]
    )


def model(*, mode="reversible", rate=1e-10, single_bin=False, **changes):
    options = {
        "negative": ParticleSpec(radius_m=5e-6, diffusivity_m2_s=1e-12, cells=3),
        "positive": ParticleSpec(
            radius_m=6e-6, diffusivity_m2_s=1e-12, active_volume_m3=8e-7, cells=4
        ),
        "plating": plating(mode=mode, rate=rate),
    }
    if not single_bin:
        options.update(
            negative_distribution=distribution((4e-6, 8e-6), (1/9, 8/9)),
            positive_distribution=distribution((6e-6, 12e-6, 18e-6), (.2, .3, .5)),
        )
    return BatteryMPMSpec(**{**options, **changes})


def program(current, start=0.):
    return ScalarControlProgram(((start, current),), "A")


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
def test_single_bin_mpm_reaches_the_existing_spm_plating_limit(mode):
    mpm = model(mode=mode, single_bin=True)
    spm = BatterySPMSpec(negative=mpm.negative, positive=mpm.positive, plating=mpm.plating)
    initial_m, initial_s = mpm.initial_state(), spm.initial_state()
    # Different public inventory identities represent the same two real particles.
    key_pairs = tuple(
        (old, new)
        for name, _, particle, keys, _ in mpm.particle_bins
        for old, new in zip(particle.keys(name), keys, strict=True)
    ) + tuple((key, key) for key in METAL)
    np.testing.assert_allclose(
        [initial_m.amounts_mol[new] for _, new in key_pairs],
        [initial_s.amounts_mol[old] for old, _ in key_pairs], rtol=0, atol=2e-18,
    )
    assert asdict(battery_readout(mpm, initial_m, .1)) == pytest.approx(
        asdict(battery_readout(spm, initial_s, .1)), rel=1e-11, abs=1e-12
    )
    result_m = advance_battery(mpm, initial_m, program(.1), .25)
    result_s = advance_battery(spm, initial_s, program(.1), .25)
    assert result_m.outcome == result_s.outcome == "success"
    np.testing.assert_allclose(
        [result_m.final_state.amounts_mol[new] for _, new in key_pairs],
        [result_s.final_state.amounts_mol[old] for old, _ in key_pairs],
        rtol=2e-8, atol=2e-12,
    )
    assert result_m.final_state.coordinates == pytest.approx(
        result_s.final_state.coordinates, rel=2e-8, abs=1e-10
    )
    assert asdict(battery_readout(mpm, result_m.final_state, .1)) == pytest.approx(
        asdict(battery_readout(spm, result_s.final_state, .1)), rel=2e-8, abs=2e-9
    )
    assert initial_m == mpm.initial_state() and initial_s == spm.initial_state()


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
def test_zero_rate_preserves_old_mpm_particles_and_passive_unique_metal(mode):
    combined = model(mode=mode, rate=0.)
    old = replace(combined, plating=None)
    actual, expected = combined.initial_state(), old.initial_state()
    for current in (.08, -.08):
        result = advance_battery(combined, actual, program(current, actual.time_s), .2)
        original = advance_battery(old, expected, program(current, expected.time_s), .2)
        assert result.outcome == original.outcome == "success"
        actual, expected = result.final_state, original.final_state
        np.testing.assert_allclose(
            [actual.amounts_mol[k] for k in old.inventory_keys],
            [expected.amounts_mol[k] for k in old.inventory_keys], rtol=2e-8, atol=2e-12,
        )
        a, b = battery_readout(combined, actual, current), battery_readout(old, expected, current)
        for field in (
            "voltage_V", "negative_surface_fraction", "positive_surface_fraction",
            "negative_average_fraction", "positive_average_fraction",
            "negative_overpotential_V", "positive_overpotential_V",
        ):
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=2e-8, abs=2e-9)
        assert actual.coordinates == pytest.approx(expected.coordinates, rel=2e-8, abs=1e-10)
        assert a.stripping_current_A == 0.
    assert {key: actual.amounts_mol[key] for key in METAL} == combined.plating.initial_amounts()


@pytest.mark.parametrize("basis, weights", (
    ("number", (1., 1.)), ("area", (1., 4.)), ("volume", (1., 8.)),
))
def test_three_input_bases_drive_the_same_actual_surface_area_and_side_current(basis, weights):
    psd = distribution((4e-6, 8e-6), weights, basis)
    spec = model(negative_distribution=psd)
    state = spec.initial_state()
    f = plating_fields(spec, state, current_A=.1)
    assert psd.source_basis == basis
    assert psd.volume_weights == pytest.approx((1/9, 8/9), rel=2e-15)
    # Number-equivalent spheres have area weights 1:4, not volume weights 1:8.
    area = 3*spec.negative.active_volume_m3*(1/(9*4e-6)+8/(9*8e-6))
    assert f["negative_active_area_m2"] == pytest.approx(area, rel=2e-15)
    assert not np.isclose(area, spec.negative.area_m2, rtol=.01)
    np.testing.assert_allclose(f["particle_volume_weights"][:2], (1/9, 8/9), rtol=2e-15)
    np.testing.assert_allclose(f["particle_area_weights"][:2], (1/5, 4/5), rtol=2e-15)
    np.testing.assert_allclose(f["particle_radii_m"][:2], (4e-6, 8e-6), rtol=0, atol=0)
    assert f["particle_electrodes"] == ("negative", "negative", "positive", "positive", "positive")
    assert fsum(f["particle_area_weights"][2:]) == pytest.approx(1., abs=2e-16)
    # Independent source formula uses the full negative geometry, not active Li volume.
    values = spec.plating.native_parameters
    beta = F*f["negative_electrode_potential_V"]/(R*spec.temperature_K)
    expected = area*F*values["k_m_s"]*(
        state.amounts_mol["plating:Li"]/values["negative_electrode_volume_m3"]
        * exp((1-values["alpha"])*beta) - values["ce_mol_m3"]*exp(-values["alpha"]*beta)
    )
    assert f["stripping_current_A"] == pytest.approx(expected, rel=2e-9, abs=2e-12)
    assert battery_particle_flux(spec, state, "negative", .1) == pytest.approx(
        fsum(f["reaction_current_A"][:2])/F, rel=2e-12, abs=1e-16
    )
    assert battery_particle_flux(spec, state, "positive", .1) == pytest.approx(-.1/F, abs=1e-16)
    assert f["negative_intercalation_current_A"] + f["stripping_current_A"] == pytest.approx(
        .1, abs=2e-12
    )
    canonical = model()
    assert asdict(battery_readout(spec, state, .1)) == pytest.approx(
        asdict(battery_readout(canonical, canonical.initial_state(), .1)), rel=2e-10, abs=2e-12
    )


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
def test_mpm_codec_and_original_equipment_slot_have_one_metal_inventory(mode):
    spec = model(mode=mode)
    old = replace(spec, plating=None)
    payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False, sort_keys=True))
    restored = battery_spec_from_dict(payload)
    assert type(restored) is BatteryMPMSpec
    assert restored.to_dict() == payload == spec.to_dict()
    assert payload["model_family"] == "mpm-isothermal-v1"
    assert restored.model_id == "mpm-plating-isothermal-v1"
    assert set(payload) == set(old.to_dict()) | {"plating"}
    assert restored.plating.parameters.to_dict() == spec.plating.parameters.to_dict()
    assert restored.negative_distribution == spec.negative_distribution
    assert spec.inventory_keys == (*old.inventory_keys, *METAL)
    assert tuple(spec.initial_state().coordinates) == ("discharged_charge_C", "delivered_work_J")
    assert spec.volume_l == (2e-6+spec.positive.active_volume_m3)*1000
    runtime = ChemWorldRuntime(composition=battery_world_spec(restored))
    runtime.reset(seed=19)
    try:
        before = runtime.snapshot().to_dict()
        state = runtime._state
        record, slot, decoded, physical, _ = unpack_battery(state)
        assert record.equipment_type == "single_particle_battery"
        assert slot.spec.state_type == "battery-lithium-and-charge-v1"
        assert slot.spec.owner_kind == "equipment" and slot.spec.owner_id == record.equipment_id
        assert slot.spec.batch_reset == "preserve"
        assert set(slot.values) == set(spec.inventory_keys) | set(spec.coordinate_keys)
        assert not any("size" in key for key in slot.values if "plating" in key or "dead" in key)
        for key in METAL:
            field = next(field for field in slot.spec.fields if field.name == key)
            assert field.unit == "mol" and field.species_id == "Li"
            assert slot.values[key] == spec.plating.initial_amounts()[key]
        expected = (
            spec.negative.capacity_mol*spec.initial_negative_fraction
            + spec.positive.capacity_mol*spec.initial_positive_fraction
            + fsum(spec.plating.initial_amounts().values())
        )
        inventory = runtime.constitution.material_inventory(state)
        assert inventory == StateSlots((slot,)).species_amounts()
        assert inventory == pytest.approx({"Li": expected}, rel=0, abs=2e-17)
        assert runtime.constitution.element_totals(inventory) == pytest.approx(
            {"Li": expected}, rel=0, abs=2e-17
        )
        assert all(value == 0 for value in state.species_amounts.values())
        assert all(not phase.species_amounts_mol for phase in state.phases.phases.values())
        for _ in range(2):
            assert initial_battery_slot(record, decoded) == slot
            assert unpack_battery(state)[3].amounts_mol == physical.amounts_mol
        assert runtime.snapshot().to_dict() == before
    finally:
        runtime.close()


def test_absent_mpm_plating_keeps_codec_unknowns_and_all_original_slot_keys():
    spec = model(plating=None)
    explicit_none = replace(spec, plating=None)
    expected_fields = OLD_FIELDS | {
        "model_family", "negative_distribution", "positive_distribution"
    }
    assert set(spec.to_dict()) == expected_fields
    assert "plating" not in spec.to_dict()
    assert spec.to_dict() == explicit_none.to_dict()
    expected_inventory = tuple(
        f"{name}:size:{size}:Li:{cell}"
        for name, count, cells in (("negative", 2, 3), ("positive", 3, 4))
        for size in range(count) for cell in range(cells)
    )
    assert spec.inventory_keys == expected_inventory
    assert spec.inventory_species == dict.fromkeys(expected_inventory, "Li")
    assert not set(METAL) & set(spec.initial_state().amounts_mol)
    assert "battery:plating:stripping-current_A" not in spec.algebraic_names
    assert spec.particle_side_reaction is None
    assert spec.model_id == "mpm-isothermal-v1"
    assert spec.volume_l == (spec.negative.active_volume_m3+spec.positive.active_volume_m3)*1000
    restored = battery_spec_from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.initial_state() == spec.initial_state()
    assert restored.algebraic_names == spec.algebraic_names
    np.testing.assert_array_equal(restored.reaction_map, spec.reaction_map)


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
@pytest.mark.parametrize("analysis", ("eis_system", "eis", "soh", "soc"))
def test_mpm_plating_remains_explicitly_unsupported_by_equilibrium_analyses(mode, analysis):
    spec = model(mode=mode, plating=plating(mode=mode, rate=0., plated=0., dead=0.))
    state, encoded = spec.initial_state(), spec.to_dict()
    calls = {
        "eis_system": lambda: battery_impedance_system(spec, state),
        "eis": lambda: battery_impedance(spec, [1.], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "soc": lambda: battery_spec_at_soc(spec, None, .5),
    }
    with pytest.raises(ValueError, match=r"(?i)plating"):
        calls[analysis]()
    assert state == spec.initial_state() and spec.to_dict() == encoded


@pytest.mark.parametrize("unsupported", (
    "thermal", "sei", "electrolyte", "current_sigmoid", "axen", "ode",
))
def test_mpm_extension_does_not_enable_undeclared_joint_physics(unsupported):
    updates = {
        "thermal": lambda: {"thermal": BatteryThermalSpec()},
        "sei": lambda: {"sei": SEISpec()},
        "electrolyte": lambda: {"electrolyte": BatteryElectrolyteSpec(cells=(2, 2, 2))},
        "current_sigmoid": lambda: {"negative_ocp": operational("negative", "current_sigmoid")},
        "axen": lambda: {"negative_ocp": operational("negative", "axen")},
        "ode": lambda: {"numerical_form": "ode"},
    }
    with pytest.raises(ValueError, match=r"(?i)plating"):
        model(**updates[unsupported]())


@pytest.mark.parametrize("family", ("DFN", "composite", "MSMR"))
def test_other_public_family_codecs_do_not_silently_drop_plating(family):
    if family == "composite":
        from chemworld.examples.battery_composite import default_spec
        base = default_spec(cells=3)
    elif family == "MSMR":
        from chemworld.examples.battery_msmr import default_spec
        base = default_spec(cells=3)
    else:
        base = old_spec("DFN")
    payload = json.loads(json.dumps(base.to_dict(), allow_nan=False))
    payload["plating"] = plating().to_dict()
    with pytest.raises((ValueError, TypeError)):
        battery_spec_from_dict(payload)
    assert "plating" not in base.to_dict()


def test_mpm_energy_consumers_do_not_assign_an_undeclared_metal_potential():
    spec = model()
    state = spec.initial_state()
    for call in (
        lambda: battery_chemical_energy_j(spec, state),
        lambda: battery_chemical_potentials(spec, state),
        lambda: battery_dissipation_w(spec, state, .1),
    ):
        with pytest.raises(ValueError, match=r"(?i)plating"):
            call()
    assert state == spec.initial_state()
