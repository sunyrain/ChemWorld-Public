"""Constant metal deactivation joins the original current and inventory residual."""

from dataclasses import replace
from math import exp

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_models import F
from chemworld.physchem.battery_plating import (
    PARTIAL_PLATING_UNITS,
    LithiumPlatingSpec,
    advance_plating,
    plating_algebraic,
    plating_control_mode,
    plating_fields,
    plating_inventory_ledger,
    plating_readout,
    plating_system,
    validate_plating_state,
)
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.residual_system import finite_jacobian
from test_battery_plating import declaration as old_declaration
from test_battery_plating import model as old_model
from test_battery_plating import program


def declaration(*, gamma=.03, plated=4e-5, dead=2e-5, rate=1e-9):
    base = old_declaration(plated=plated, dead=dead, k_m_s=rate)
    source = base.parameters.sources[0]
    values = {**base.native_parameters, "gamma_s_inv": gamma}
    table = ScientificTable("partial-plating-physical", "parameters",
        tuple(QuantityColumn(key, unit, "parameter", (-1e308, 1e308), "Explicit test input")
              for key, unit in PARTIAL_PLATING_UNITS.items()), (source,),
        (ScientificRow("kinetics", source.provenance.source_id, "group", "sample", "parameters",
            {key: InputValue(value, PARTIAL_PLATING_UNITS[key], None, "Fixed synthetic input")
             for key, value in values.items()}),))
    return LithiumPlatingSpec(table, "kinetics", "partially reversible", plated, dead)


def model(*, gamma=.03, plated=4e-5, dead=2e-5, rate=1e-9, **changes):
    return replace(old_model(**changes), plating=declaration(
        gamma=gamma, plated=plated, dead=dead, rate=rate))


@pytest.mark.parametrize("gamma", [-.01, float("nan"), float("inf")])
def test_nonphysical_constant_gamma_refuses(gamma):
    with pytest.raises(ValueError):
        declaration(gamma=gamma)


def test_nonfinite_initial_or_later_deactivation_refuses_without_clipping():
    with pytest.raises(ValueError, match=r"deactivation rate.*representable"):
        declaration(gamma=1e308, plated=2.)
    spec = declaration(gamma=1e308, plated=0.)
    with pytest.raises(ValueError, match=r"deactivation rate.*representable"):
        spec.deactivation_rate_mol_s(2.)
    with pytest.raises(ValueError, match="finite nonnegative"):
        spec.deactivation_rate_mol_s(-1e-20)
    assert spec.deactivation_rate_mol_s(0.) == 0.


@pytest.mark.parametrize("value,unit", [(0., "A"), (-.3, "A"), (.2, "A"),
                                       (3.78, "V"), (2.6, "V"), (4.4, "V")])
def test_full_residual_jacobian_has_deactivation_but_no_new_electron_or_work_rate(value, unit):
    s = model(maximum_current_A=.4)
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
    index = {name: i for i, name in enumerate(names)}
    assert actual[index["dead:Li"], index["plating:Li"]] == .03
    assert actual[index["plating:Li"], index["plating:Li"]] == -.03
    assert actual[index["discharged_charge_C"], index["plating:Li"]] == 0.
    assert actual[index["delivered_work_J"], index["plating:Li"]] == 0.
    rates = rhs(y)
    assert rates[index["dead:Li"]] == pytest.approx(.03*initial.amounts_mol["plating:Li"])
    assert rates[index["plating:Li"]]+rates[index["dead:Li"]] == pytest.approx(
        -f["stripping_current_A"]/F, abs=1e-20)
    assert f["negative_intercalation_current_A"] == pytest.approx(
        f["current_A"]-f["stripping_current_A"], abs=1e-15)


def test_zero_electrochemical_rate_preserves_real_analytic_deactivation():
    s = model(rate=0., gamma=.02)
    initial = s.initial_state()
    times = [0., 1., 5., 20.]
    result = advance_plating(s, initial, program(0.), 20., evaluation_times_s=times)
    assert result.outcome == "success"
    for state in result.trajectory:
        expected = initial.amounts_mol["plating:Li"]*exp(-.02*state.time_s)
        assert state.amounts_mol["plating:Li"] == pytest.approx(expected, rel=2e-6, abs=2e-11)
        assert state.amounts_mol["dead:Li"] == pytest.approx(
            initial.amounts_mol["dead:Li"]+initial.amounts_mol["plating:Li"]-expected,
            rel=2e-6, abs=2e-11)
        for name in ("negative", "positive"):
            np.testing.assert_allclose(
                [state.amounts_mol[k] for k in getattr(s, name).keys(name)],
                [initial.amounts_mol[k] for k in getattr(s, name).keys(name)],
                rtol=4*np.finfo(float).eps, atol=0.)
        assert state.coordinates == initial.coordinates
        assert plating_fields(s, state, current_A=0.)["stripping_current_A"] == 0.
        validate_plating_state(s, state)
    assert result.energy_balance_residual_J is None


def test_inventory_ledger_exposes_declared_constant_decay_without_relabeling_plating():
    s = model(rate=0., gamma=.02)
    initial = s.initial_state()
    before = plating_inventory_ledger(s, initial)
    result = advance_plating(s, initial, program(0.), 2.)
    after = plating_inventory_ledger(s, result.final_state)
    expected_dead = s.plating.initial_plated_mol * (1 - exp(-.02 * 2.0))
    assert before["deactivated_mol"] == pytest.approx(0.0, abs=1e-14)
    assert before["deactivation_rate_mol_s"] == pytest.approx(.02 * 4e-5)
    assert after["deactivated_mol"] == pytest.approx(expected_dead, rel=2e-6)
    assert after["deactivation_rate_mol_s"] == pytest.approx(
        .02 * result.final_state.amounts_mol["plating:Li"])
    assert after["plated_mol"] + after["deactivated_mol"] == pytest.approx(
        s.plating.initial_plated_mol, rel=2e-6)


def test_public_readout_exposes_partial_deactivation_rate():
    s = model(rate=0., gamma=.02)
    readout = plating_readout(s, s.initial_state(), 0.)
    assert readout.deactivation_rate_mol_s == pytest.approx(.02 * 4e-5)

    # Existing plating modes retain the compatible zero default.
    legacy = old_model()
    assert plating_readout(legacy, legacy.initial_state(), 0.).deactivation_rate_mol_s == 0.


def test_gamma_zero_does_not_insert_structural_zero_in_shared_jacobian():
    s = model(gamma=0.)
    old = replace(s, plating=old_declaration())
    state, old_state = s.initial_state(), old.initial_state()
    assert state == old_state
    p = program(.2)
    algebraic = tuple(v.name for v in plating_algebraic(s, p).variables)
    names = (*s.inventory_keys, "temperature", *s.coordinate_keys, *algebraic)
    fields = plating_fields(s, state, program=p)
    v = CoupledStateView(state.amounts_mol, state.temperature_K, state.volume_L,
                        coordinates=state.coordinates, algebraic_values=fields["algebraic_values"])
    actual = plating_system(s, p).residual_rhs_jacobian(v, names).tocsc()
    expected = plating_system(old, p).residual_rhs_jacobian(v, names).tocsc()
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)


def test_no_plated_lithium_is_legal_and_charge_rest_reverse_gives_real_dead_inventory():
    s = model(plated=0., dead=0.)
    initial, state = s.initial_state(), s.initial_state()
    assert plating_fields(s, state, current_A=0.)["deactivation_rate_mol_s"] == 0.
    for current in (-.3, 0., .2):
        result = advance_plating(s, state, program(current, start=state.time_s), 3.)
        previous = state
        state = result.final_state
        assert result.outcome == "success"
        assert state.amounts_mol["dead:Li"] >= previous.amounts_mol["dead:Li"]
        validate_plating_state(s, state)
        assert sum(state.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-12)
    assert state.amounts_mol["plating:Li"] > 0 and state.amounts_mol["dead:Li"] > 0
    assert state.coordinates["discharged_charge_C"] == pytest.approx(-.3, abs=1e-10)


def test_finite_cv_and_two_current_limits_include_same_deactivating_state():
    s = model(maximum_current_A=.3)
    first = advance_plating(s, s.initial_state(), program(-.2), 2.).final_state
    target = (plating_readout(s, first, .2).voltage_V
              +plating_readout(s, first, -.2).voltage_V)/2
    p = program(target, "V", start=first.time_s)
    result = advance_plating(s, first, p, 2.)
    assert result.outcome == "success"
    final = result.final_state
    assert final.amounts_mol["dead:Li"] > first.amounts_mol["dead:Li"]
    assert plating_control_mode(s, final, p)[0] == "constant_voltage"
    assert plating_fields(s, final, program=p)["voltage_V"] == pytest.approx(target, abs=1e-8)
    for voltage, mode, current in ((2.6, "positive_current_limit", .3),
                                  (4.4, "negative_current_limit", -.3)):
        p = program(voltage, "V", start=final.time_s)
        assert plating_control_mode(s, final, p)[0] == mode
        assert plating_fields(s, final, program=p)["current_A"] == pytest.approx(current)


def test_surface_stop_and_reverse_restart_keep_real_dead_lithium():
    s = model(initial_negative_fraction=.045, rate=1e-11)
    result = advance_plating(s, s.initial_state(), program(.3), 1000.)
    assert result.outcome == "stopped"
    assert result.stopping_event == "battery:negative:depleted"
    final = result.final_state
    assert 0 < final.time_s < 1000.
    assert final.amounts_mol["dead:Li"] > s.plating.initial_dead_mol
    resumed = advance_plating(s, final, program(-.2, start=final.time_s), 2.)
    assert resumed.outcome == "success"
    validate_plating_state(s, resumed.final_state)


def test_dead_inventory_cannot_be_silently_reactivated_even_with_total_li_preserved():
    s = model()
    state = s.initial_state()
    amounts = dict(state.amounts_mol)
    amounts["dead:Li"] -= 1e-5
    amounts["plating:Li"] += 1e-5
    with pytest.raises(ValueError, match="inactive lithium"):
        plating_fields(s, replace(state, amounts_mol=amounts), current_A=0.)
