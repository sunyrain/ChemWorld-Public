"""Finite metal conservation, simultaneous current closure and true continuation."""

import json
from dataclasses import replace
from math import exp

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    F,
    ParticleSpec,
    advance_battery,
    battery_readout,
)
from chemworld.physchem.battery_plating import (
    PLATING_SPECIES,
    PLATING_UNITS,
    LithiumPlatingSpec,
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
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.electrochemistry import R_J_PER_MOL_K as R
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def declaration(*, mode="reversible", plated=4e-5, dead=2e-5, **changes):
    values = {"k_m_s": 1e-9, "alpha": .45, "ce_mol_m3": 1000.,
              "negative_electrode_volume_m3": 2e-6, **changes}
    source = InputSource(DataSourceProvenance(
        "plating-physical-synthetic", 0, "Authored finite-metal regression parameters",
        "1", "CC0-1.0"), "synthetic")
    table = ScientificTable("plating-physical", "parameters",
        tuple(QuantityColumn(key, unit, "parameter", (-1e6, 1e6), "Explicit test input")
              for key, unit in PLATING_UNITS.items()), (source,),
        (ScientificRow("kinetics", source.provenance.source_id, "group", "sample", "parameters",
            {key: InputValue(value, PLATING_UNITS[key], None, "Fixed synthetic input")
             for key, value in values.items()}),))
    return LithiumPlatingSpec(table, "kinetics", mode, plated, dead)


def model(*, mode="reversible", plated=4e-5, dead=2e-5, rate=1e-9, **kwargs):
    return BatterySPMSpec(
        negative=ParticleSpec(radius_m=5e-6, diffusivity_m2_s=1e-12, cells=5),
        positive=ParticleSpec(radius_m=5e-6, diffusivity_m2_s=1e-12, cells=6),
        plating=declaration(mode=mode, plated=plated, dead=dead, k_m_s=rate), **kwargs)


def program(value=.2, unit="A", start=0):
    return ScalarControlProgram(((start, value),), unit)


def test_parameter_identity_codec_and_immutable_concentration_geometry():
    spec = declaration()
    restored = LithiumPlatingSpec.from_dict(json.loads(json.dumps(spec.to_dict(), allow_nan=False)))
    assert restored == spec
    assert spec.initial_amounts() == {"plating:Li": 4e-5, "dead:Li": 2e-5}
    assert restored.parameters.sources[0].provenance.source_id == "plating-physical-synthetic"
    with pytest.raises(TypeError):
        spec.native_parameters["k_m_s"] = 1
    with pytest.raises(ValueError, match="fields"):
        LithiumPlatingSpec.from_dict({**spec.to_dict(), "hidden_rate": 1})


@pytest.mark.parametrize("changes", [
    {"k_m_s": -1e-9}, {"alpha": 0}, {"alpha": 1}, {"ce_mol_m3": 0},
    {"negative_electrode_volume_m3": 0}, {"mode": "partially reversible"}, {"plated": -1},
])
def test_invalid_plating_equations_and_real_inventory_refuse(changes):
    with pytest.raises(ValueError):
        declaration(**changes)


@pytest.mark.parametrize("mode", ["reversible", "irreversible"])
def test_source_kinetic_formula_and_every_analytic_partial(mode):
    p = declaration(mode=mode)
    potential, amount, temperature, area = .025, 4e-5, 315., .6
    parameters = p.native_parameters
    factor, gain = F/(R*temperature), area*F*parameters["k_m_s"]
    c = amount / parameters["negative_electrode_volume_m3"]
    expected = -gain*parameters["ce_mol_m3"]*exp(-parameters["alpha"]*factor*potential)
    if mode == "reversible":
        expected += gain*c*exp((1-parameters["alpha"])*factor*potential)
    value, *derivatives = p.current_derivatives(potential, amount, temperature, area)
    assert value == pytest.approx(expected, rel=2e-15)
    inputs = [potential, amount, temperature]
    for i, step in enumerate((1e-7, 1e-9, 1e-3)):
        lo, hi = inputs.copy(), inputs.copy()
        lo[i] -= step
        hi[i] += step
        measured = (p.current_derivatives(*hi, area)[0]
                    - p.current_derivatives(*lo, area)[0])/(2*step)
        assert derivatives[i] == pytest.approx(measured, rel=2e-8, abs=2e-12)
    # Explicit total-electrode volume, not active-particle volume, sets c_plated.
    doubled = declaration(mode=mode, negative_electrode_volume_m3=4e-6)
    assert doubled.current_derivatives(potential, amount*2, temperature, area)[0] == value


def test_overflow_and_nonphysical_kinetics_refuse_without_clipping():
    p = declaration()
    with pytest.raises(ValueError, match="representable"):
        p.current_derivatives(1000., 4e-5, 298.15, .6)
    with pytest.raises(ValueError, match="physical inventory"):
        p.current_derivatives(.1, -1e-8, 298.15, .6)
    assert declaration(k_m_s=0).current_derivatives(1000., 4e-5, 298.15, .6) == (0.,)*4


def test_one_bin_adapter_keeps_original_radial_keys_and_adds_two_actual_li_stocks():
    s = model()
    old = replace(s, plating=None)
    assert s.inventory_keys == (*old.inventory_keys, *PLATING_SPECIES)
    assert s.coordinate_keys == old.coordinate_keys
    initial = s.initial_state()
    assert set(initial.amounts_mol)-set(old.inventory_keys) == set(PLATING_SPECIES)
    f = plating_fields(s, initial, current_A=.2)
    assert f["reaction_current_A"][0]+f["stripping_current_A"] == pytest.approx(.2, abs=1e-14)
    assert f["reaction_current_A"][1] == pytest.approx(-.2, abs=1e-14)
    assert plating_particle_flux(s, initial, "negative", .2) == pytest.approx(
        f["negative_intercalation_current_A"]/F)
    assert plating_particle_flux(s, initial, "positive", .2) == pytest.approx(-.2/F)
    assert s.volume_l == pytest.approx(.003)


@pytest.mark.parametrize("mode", ["reversible", "irreversible"])
@pytest.mark.parametrize("value,unit", [(0., "A"), (-.3, "A"), (.2, "A"),
                                       (3.78, "V"), (2.6, "V"), (4.4, "V")])
def test_complete_shared_jacobian_includes_side_current_inventory_and_control(mode, value, unit):
    s = model(mode=mode, maximum_current_A=.4)
    p, initial = program(value, unit), s.initial_state()
    f = plating_fields(s, initial, program=p)
    closure = plating_algebraic(s, p)
    algebraic = tuple(v.name for v in closure.variables)
    names = (*s.inventory_keys, "temperature", *s.coordinate_keys, *algebraic)
    y = np.r_[list(initial.amounts_mol.values()), initial.temperature_K,
              list(initial.coordinates.values()), [f["algebraic_values"][k] for k in algebraic]]
    n = len(s.inventory_keys)
    scales = np.r_[[1e-3]*n, 300., 1000., 4000., .4, 1., .4, 1.]

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


def test_rest_internal_exchange_is_real_not_limited_by_external_source():
    s = model(maximum_current_A=.001, plated=.001)
    initial = s.initial_state()
    f = plating_fields(s, initial, current_A=0.)
    assert f["stripping_current_A"] > s.maximum_current_A
    assert f["negative_intercalation_current_A"] == pytest.approx(-f["stripping_current_A"])
    result = advance_plating(s, initial, program(0.), 8.)
    final = result.final_state
    assert final.amounts_mol["plating:Li"] < initial.amounts_mol["plating:Li"]
    assert sum(final.amounts_mol[k] for k in s.negative.keys("negative")) > sum(
        initial.amounts_mol[k] for k in s.negative.keys("negative"))
    assert final.amounts_mol["dead:Li"] == initial.amounts_mol["dead:Li"]
    assert final.coordinates["discharged_charge_C"] == pytest.approx(0, abs=1e-14)
    assert final.coordinates["delivered_work_J"] == pytest.approx(0, abs=1e-14)
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-13)
    assert result.energy_balance_residual_J is None


@pytest.mark.parametrize("mode", ["reversible", "irreversible"])
def test_zero_metal_is_admissible_and_charge_rest_reverse_preserve_faraday(mode):
    s = model(mode=mode, plated=0., dead=0.)
    initial, state = s.initial_state(), s.initial_state()
    for value in (-.3, 0., .2):
        result = advance_plating(s, state, program(value, start=state.time_s), 4.)
        state = result.final_state
        assert result.outcome == "success"
        validate_plating_state(s, state)
        assert sum(state.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-12)
    if mode == "reversible":
        assert state.amounts_mol["plating:Li"] > 0 and state.amounts_mol["dead:Li"] == 0
    else:
        assert state.amounts_mol["dead:Li"] > 0 and state.amounts_mol["plating:Li"] == 0
    assert state.coordinates["discharged_charge_C"] == pytest.approx(-.4, abs=1e-10)


@pytest.mark.parametrize("mode", ["reversible", "irreversible"])
def test_zero_rate_reduces_to_original_spm_with_unchanged_passive_metal(mode):
    s = model(mode=mode, rate=0.)
    old = replace(s, plating=None)
    new_state, old_state = s.initial_state(), old.initial_state()
    for current in (.2, -.3):
        new_state = advance_plating(
            s, new_state, program(current, start=new_state.time_s), 5.).final_state
        old_state = advance_battery(
            old, old_state, program(current, start=old_state.time_s), 5.).final_state
        np.testing.assert_allclose([new_state.amounts_mol[k] for k in old.inventory_keys],
                                   list(old_state.amounts_mol.values()), rtol=2e-8, atol=2e-11)
        assert plating_readout(s, new_state, current).voltage_V == pytest.approx(
            battery_readout(old, old_state, current).voltage_V, abs=2e-8)
    assert {k: new_state.amounts_mol[k] for k in PLATING_SPECIES} == s.plating.initial_amounts()


def test_finite_cv_and_fresh_reverse_control_use_the_same_real_state():
    s = model(maximum_current_A=.3)
    first = advance_plating(s, s.initial_state(), program(-.2), 3.).final_state
    forward = plating_readout(s, first, .2)
    reverse = plating_readout(s, first, -.2)
    target = (forward.voltage_V+reverse.voltage_V)/2
    p = program(target, "V", start=first.time_s)
    result = advance_plating(s, first, p, 3.)
    assert plating_control_mode(s, result.final_state, p)[0] == "constant_voltage"
    assert plating_fields(s, result.final_state, program=p)["voltage_V"] == pytest.approx(
        target, abs=1e-8)
    for voltage, mode, current in ((2.6, "positive_current_limit", .3),
                                  (4.4, "negative_current_limit", -.3)):
        p = program(voltage, "V", start=result.final_state.time_s)
        assert plating_control_mode(s, result.final_state, p)[0] == mode
        assert plating_fields(s, result.final_state, program=p)["current_A"] == pytest.approx(
            current)


def test_finite_surface_stop_and_inward_restart_preserve_all_metal():
    s = model(initial_negative_fraction=.045, rate=1e-11)
    result = advance_plating(s, s.initial_state(), program(.3), 1000.)
    assert result.outcome == "stopped"
    assert result.stopping_event == "battery:negative:depleted"
    assert 0 < result.final_state.time_s < 1000.
    resumed = advance_plating(s, result.final_state,
                              program(-.2, start=result.final_state.time_s), 2.)
    assert resumed.outcome == "success"
    validate_plating_state(s, resumed.final_state)


def test_uncertified_material_or_inconsistent_inventory_cannot_enter_closure():
    with pytest.raises(ValueError, match="nonincreasing OCV"):
        model(negative_ocv_slope_V=.1)
    s = model()
    initial = s.initial_state()
    amounts = dict(initial.amounts_mol)
    amounts["plating:Li"] *= 2
    with pytest.raises(ValueError, match="Faradaic"):
        plating_readout(s, replace(initial, amounts_mol=amounts), 0.)


def test_finite_metal_amount_cannot_overflow_its_real_faradaic_inventory():
    with pytest.raises(ValueError, match="Faradaic inventory"):
        model(dead=1e306)
