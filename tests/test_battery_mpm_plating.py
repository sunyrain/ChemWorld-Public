"""Shared metallic Li, independent size fluxes and the original residual system."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_models import F, ParticleSpec, battery_spec_from_dict
from chemworld.physchem.battery_mpm import BatteryMPMSpec, mpm_fields, mpm_readout
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.battery_plating import (
    PLATING_SPECIES,
    advance_plating,
    plating_algebraic,
    plating_control_mode,
    plating_fields,
    plating_particle_flux,
    plating_readout,
    plating_system,
    validate_plating_state,
)
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.residual_system import finite_jacobian
from test_battery_plating import declaration, program


def model(*, mode="reversible", plated=4e-5, dead=2e-5, rate=1e-9, **changes):
    plating = declaration(mode=mode, plated=plated, dead=dead, k_m_s=rate)
    source = plating.parameters.sources[0]
    values = {
        "negative": ParticleSpec(radius_m=5e-6, diffusivity_m2_s=1e-12, cells=4),
        "positive": ParticleSpec(radius_m=5e-6, diffusivity_m2_s=1e-12, cells=5),
        "negative_distribution": ParticleSizeDistribution.from_weights(
            (2e-6, 5e-6, 12e-6), (.2, .35, .45), basis="volume", source=source),
        "positive_distribution": ParticleSizeDistribution.from_weights(
            (3e-6, 9e-6), (.4, .6), basis="volume", source=source),
        "maximum_current_A": .5, "plating": plating,
    }
    return BatteryMPMSpec(**{**values, **changes})


def test_json_psd_decodes_before_base_validation_and_inventory_is_unique():
    s = model()
    restored = battery_spec_from_dict(json.loads(json.dumps(s.to_dict(), allow_nan=False)))
    assert restored == s
    assert restored.initial_state() == s.initial_state()
    old = replace(s, plating=None)
    assert s.inventory_keys == (*old.inventory_keys, *PLATING_SPECIES)
    assert s.coordinate_keys == old.coordinate_keys
    assert s.model_id == "mpm-plating-isothermal-v1"
    assert old.model_id == "mpm-isothermal-v1"
    assert s.initial_state().amounts_mol["plating:Li"] == 4e-5
    assert s.volume_l == pytest.approx(.003)


def test_actual_area_and_bin_flux_are_not_a_typical_radius_or_fixed_current_split():
    s, current = model(negative=ParticleSpec(
        radius_m=5e-6, diffusivity_m2_s=1e-14, cells=4)), .2
    initial = s.initial_state()
    f = plating_fields(s, initial, current_A=current)
    mask = np.array([n == "negative" for n in f["particle_electrodes"]])
    expected_area = 3*s.negative.active_volume_m3*sum(
        w/r for w, r in zip(s.negative_distribution.volume_weights,
                           s.negative_distribution.radii_m, strict=True))
    assert f["negative_active_area_m2"] == pytest.approx(expected_area, rel=2e-15)
    assert abs(expected_area-s.negative.area_m2) > .01
    expected_side = s.plating.current_derivatives(
        f["negative_electrode_potential_V"], initial.amounts_mol["plating:Li"],
        s.temperature_K, expected_area)[0]
    assert f["stripping_current_A"] == pytest.approx(expected_side, abs=1e-12)
    assert sum(f["reaction_current_A"][mask])+expected_side == pytest.approx(current, abs=1e-12)
    assert sum(f["reaction_current_A"][~mask]) == pytest.approx(-current, abs=1e-13)
    assert np.ptp(f["reaction_current_A"][mask]/f["area_m2"][mask]) > 1e-4
    for name, sel in (("negative", mask), ("positive", ~mask)):
        assert sum(f["particle_volume_weights"][sel]) == pytest.approx(1.)
        assert sum(f["particle_area_weights"][sel]) == pytest.approx(1.)
        assert plating_particle_flux(s, initial, name, current) == pytest.approx(
            sum(f["reaction_current_A"][sel])/F, abs=1e-20)
    assert abs(plating_particle_flux(s, initial, "negative", current)
               -f["reaction_current_A"][0]/F) > 1e-8
    readout = plating_readout(s, initial, current)
    assert mpm_readout(s, initial, current) == readout
    assert readout.negative_surface_fraction == pytest.approx(np.average(
        f["surface_fraction"][mask], weights=f["area_m2"][mask]))
    assert readout.negative_average_fraction == pytest.approx(np.average(
        f["average_fraction"][mask], weights=f["particle_volume_weights"][mask]))
    assert mpm_fields(s, initial, current_A=current)["negative_active_area_m2"] == pytest.approx(
        expected_area, rel=2e-15)


@pytest.mark.parametrize("mode", ["reversible", "irreversible"])
@pytest.mark.parametrize("value,unit", [(0., "A"), (-.3, "A"), (.2, "A"),
                                       (3.78, "V"), (2.6, "V"), (4.4, "V")])
def test_full_multibin_residual_jacobian_includes_metal_phi_all_bins_and_cv(mode, value, unit):
    s = model(mode=mode)
    p, initial = program(value, unit), s.initial_state()
    f = plating_fields(s, initial, program=p)
    closure = plating_algebraic(s, p)
    algebraic = tuple(v.name for v in closure.variables)
    names = (*s.inventory_keys, "temperature", *s.coordinate_keys, *algebraic)
    assert algebraic == s.algebraic_names
    y = np.r_[list(initial.amounts_mol.values()), initial.temperature_K,
              list(initial.coordinates.values()), [f["algebraic_values"][k] for k in algebraic]]
    n = len(s.inventory_keys)
    scales = np.r_[[1e-3]*n, 300., 1000., 4000.,
                   [v.scale for v in closure.variables]]

    def view(z):
        return CoupledStateView(dict(zip(s.inventory_keys, z[:n], strict=True)), z[n], s.volume_l,
            coordinates=dict(zip(s.coordinate_keys, z[n+1:n+3], strict=True)),
            algebraic_values=dict(zip(algebraic, z[n+3:], strict=True)))

    system = plating_system(s, p)

    def rhs(z):
        v = view(z)
        rates = assemble_coupled_rates(v, processes=system.processes,
            thermodynamics=system.thermodynamics, isothermal=True, heat_flux=system.heat_flux)
        return np.r_[[rates.amount_rates_mol_s[k] for k in s.inventory_keys], 0.,
                     [rates.coordinate_rates[k] for k in s.coordinate_keys],
                     -np.asarray(system.algebraic.residual(v))]

    actual = system.residual_rhs_jacobian(view(y), names).toarray()
    expected = finite_jacobian(rhs, y, scales)
    row_scale = np.r_[[1e-3]*n, 300., 1., 4., [1.]*len(algebraic)]
    np.testing.assert_allclose(actual*scales/row_scale[:, None],
                              expected*scales/row_scale[:, None], rtol=8e-6, atol=3e-5)


def test_rest_has_real_internal_exchange_and_no_external_current_charge_or_work():
    s = model(maximum_current_A=.001, plated=.001)
    initial = s.initial_state()
    f = plating_fields(s, initial, current_A=0.)
    assert f["stripping_current_A"] > s.maximum_current_A
    assert f["negative_intercalation_current_A"] == pytest.approx(-f["stripping_current_A"])
    result = advance_plating(s, initial, program(0.), 4.)
    final = result.final_state
    assert result.outcome == "success"
    assert final.amounts_mol["plating:Li"] < initial.amounts_mol["plating:Li"]
    assert final.amounts_mol["dead:Li"] == initial.amounts_mol["dead:Li"]
    keys = [k for n, _, _, keys, _ in s.particle_bins if n == "negative" for k in keys]
    assert sum(final.amounts_mol[k] for k in keys) > sum(initial.amounts_mol[k] for k in keys)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(0., abs=1e-14)
    assert final.coordinates["delivered_work_J"] == pytest.approx(0., abs=1e-14)
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-12)
    validate_plating_state(s, final)


@pytest.mark.parametrize("mode", ["reversible", "irreversible"])
def test_zero_metal_charge_rest_reverse_with_one_shared_metal_inventory(mode):
    s = model(mode=mode, plated=0., dead=0.)
    initial, state = s.initial_state(), s.initial_state()
    for current in (-.3, 0., .2):
        result = advance_plating(s, state, program(current, start=state.time_s), 2.)
        state = result.final_state
        assert result.outcome == "success"
        validate_plating_state(s, state)
        assert sum(state.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-12)
    changed, fixed = (("plating:Li", "dead:Li") if mode == "reversible"
                      else ("dead:Li", "plating:Li"))
    assert state.amounts_mol[changed] > 0
    assert state.amounts_mol[fixed] == 0
    assert state.coordinates["discharged_charge_C"] == pytest.approx(-.2, abs=1e-10)


def test_cv_evolves_same_bins_and_metal_then_both_limits():
    s = model(maximum_current_A=.3)
    state = advance_plating(s, s.initial_state(), program(-.2), 2.).final_state
    target = (plating_readout(s, state, .2).voltage_V
              +plating_readout(s, state, -.2).voltage_V)/2
    p = program(target, "V", start=state.time_s)
    result = advance_plating(s, state, p, 2.)
    assert result.outcome == "success"
    final = result.final_state
    assert plating_control_mode(s, final, p)[0] == "constant_voltage"
    assert plating_fields(s, final, program=p)["voltage_V"] == pytest.approx(target, abs=1e-8)
    validate_plating_state(s, final)
    for voltage, mode, current in ((2.6, "positive_current_limit", .3),
                                  (4.4, "negative_current_limit", -.3)):
        p = program(voltage, "V", start=final.time_s)
        assert plating_control_mode(s, final, p)[0] == mode
        assert plating_fields(s, final, program=p)["current_A"] == pytest.approx(current)


def test_one_bin_capacity_cannot_be_masked_by_whole_electrode_average():
    s = model()
    state = s.initial_state()
    amounts = dict(state.amounts_mol)
    _, _, p, keys, _ = s.particle_bins[0]
    amount = 1.01*p.mesh.cell_volumes_m3[-1]*p.maximum_concentration_mol_m3
    excess = amount-amounts[keys[-1]]
    amounts[keys[-1]] = amount
    amounts[s.particle_bins[1][3][-1]] -= excess
    with pytest.raises(ValueError, match="finite capacity"):
        plating_fields(s, replace(state, amounts_mol=amounts), current_A=0.)


def test_rare_bin_surface_event_precedes_average_and_allows_inward_restart():
    base = model(rate=1e-12, negative_ocv_slope_V=0.)
    distribution = ParticleSizeDistribution.from_weights(
        (2e-6, 16e-6), (.02, .98), basis="volume",
        source=base.plating.parameters.sources[0])
    s = replace(base, negative_distribution=distribution, initial_negative_fraction=.15)
    state = s.initial_state()
    amounts = dict(state.amounts_mol)
    bins = [b for b in s.particle_bins if b[0] == "negative"]
    removed = 0.
    for key in bins[0][3]:
        old = amounts[key]
        amounts[key] *= .025/.15
        removed += old-amounts[key]
    p, keys = bins[1][2:4]
    for key, weight in zip(keys, p.mesh.cell_volumes_m3/p.active_volume_m3, strict=True):
        amounts[key] += removed*weight
    state = replace(state, amounts_mol=amounts)
    validate_plating_state(s, state)
    result = advance_plating(s, state, program(.2), 100.)
    assert result.outcome == "stopped"
    assert result.stopping_event == "battery:negative:depleted"
    final = result.final_state
    f = plating_fields(s, final, current_A=.2)
    assert f["surface_fraction"][0] == pytest.approx(s.surface_margin, abs=1e-7)
    assert plating_readout(s, final, .2).negative_average_fraction > .1
    resumed = advance_plating(s, final, program(-.2, start=final.time_s), 1.)
    assert resumed.outcome == "success"
    validate_plating_state(s, resumed.final_state)
