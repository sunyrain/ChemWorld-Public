"""Main-reaction driven Axen memory in the original reversible metal residual."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_models import F, advance_battery, battery_readout
from chemworld.physchem.battery_ocp import hysteresis_key
from chemworld.physchem.battery_plating import (
    STRIPPING_CURRENT,
    advance_plating,
    plating_algebraic,
    plating_control_mode,
    plating_fields,
    plating_readout,
    plating_system,
    validate_plating_state,
)
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.residual_system import finite_jacobian
from test_battery_hysteresis import ocp
from test_battery_plating import model as old_model
from test_battery_plating import program


def model(*, negative=True, positive=True, h_negative=.2, h_positive=-.3, **changes):
    return replace(old_model(**changes),
        negative_ocp=ocp(h=h_negative) if negative else None,
        positive_ocp=ocp("positive", h=h_positive) if positive else None)


def rates(spec, state, p):
    f = plating_fields(spec, state, program=p)
    view = CoupledStateView(state.amounts_mol, state.temperature_K, state.volume_L, state.time_s,
                           coordinates=state.coordinates, algebraic_values=f["algebraic_values"])
    system = plating_system(spec, p)
    return f, assemble_coupled_rates(view, processes=system.processes,
        thermodynamics=system.thermodynamics, isothermal=True, heat_flux=system.heat_flux)


@pytest.mark.parametrize("value,unit", [(0., "A"), (-.3, "A"), (.2, "A"),
                                       (3.78, "V"), (2.6, "V"), (4.4, "V")])
def test_complete_residual_jacobian_has_true_main_current_and_bv_h_derivatives(value, unit):
    # At exact terminal rest the positive Axen current derivative is not
    # two-sided. Use a single positive OCP for this case, while retaining the
    # nonzero internal negative reaction and both Axen states for all others.
    s = model(maximum_current_A=.4, positive=not (value == 0 and unit == "A"))
    p, initial = program(value, unit), s.initial_state()
    f = plating_fields(s, initial, program=p)
    algebraic = tuple(v.name for v in plating_algebraic(s, p).variables)
    names = (*s.inventory_keys, "temperature", *s.coordinate_keys, *algebraic)
    n, nc = len(s.inventory_keys), len(s.coordinate_keys)
    y = np.r_[list(initial.amounts_mol.values()), initial.temperature_K,
              list(initial.coordinates.values()), [f["algebraic_values"][k] for k in algebraic]]
    scales = np.r_[[1e-3]*n, 300., 1000., 4000., [1.]*len(s.hysteresis_keys), .4, 1., .4, 1.]

    def view(z):
        return CoupledStateView(dict(zip(s.inventory_keys, z[:n], strict=True)), z[n], s.volume_l,
            coordinates=dict(zip(s.coordinate_keys, z[n+1:n+1+nc], strict=True)),
            algebraic_values=dict(zip(algebraic, z[n+1+nc:], strict=True)))

    system = plating_system(s, p)

    def rhs(z):
        v = view(z)
        answer = assemble_coupled_rates(v, processes=system.processes,
            thermodynamics=system.thermodynamics, isothermal=True, heat_flux=system.heat_flux)
        return np.r_[[answer.amount_rates_mol_s[k] for k in s.inventory_keys], 0.,
                     [answer.coordinate_rates[k] for k in s.coordinate_keys],
                     -np.asarray(system.algebraic.residual(v))]

    actual = system.residual_rhs_jacobian(view(y), names).toarray()
    expected = finite_jacobian(rhs, y, scales)
    row_scale = np.r_[[1e-3]*n, 300., 1., 4., [1.]*len(s.hysteresis_keys), [1.]*len(algebraic)]
    np.testing.assert_allclose(actual*scales/row_scale[:, None],
                              expected*scales/row_scale[:, None], rtol=8e-6, atol=3e-5)
    index = {name: i for i, name in enumerate(names)}
    row = index[hysteresis_key("negative")]
    # dq_n/dI=1 and dq_n/dq_strip=-1 in the *same* residual, never a
    # precomputed terminal-current rate detached from the electrical closure.
    assert actual[row, index["battery:current_A"]] != 0
    assert actual[row, index[STRIPPING_CURRENT]] == -actual[row, index["battery:current_A"]]
    assert actual[index[STRIPPING_CURRENT], row] != 0  # negative BV equation
    assert actual[index["discharged_charge_C"], row] == 0
    assert actual[index["delivered_work_J"], row] == 0


def test_rest_internal_stripping_drives_negative_memory_without_terminal_charge():
    s = model(plated=.001, maximum_current_A=.001)
    initial = s.initial_state()
    f, answer = rates(s, initial, program(0.))
    qn, qp = f["reaction_current_A"]
    assert f["stripping_current_A"] > s.maximum_current_A
    assert qn == pytest.approx(-f["stripping_current_A"], abs=1e-15)
    assert qp == 0
    hn, hp = s.hysteresis_keys
    # Independent source equation, using the true particle reaction current.
    expected = 20*abs(qn)/F/s.negative.capacity_mol*(-1-initial.coordinates[hn])
    assert answer.coordinate_rates[hn] == pytest.approx(expected, rel=2e-15)
    assert answer.coordinate_rates[hp] == 0
    result = advance_plating(s, initial, program(0.), 4.)
    final = result.final_state
    assert result.outcome == "success"
    assert final.coordinates[hn] < initial.coordinates[hn]
    assert final.coordinates[hp] == initial.coordinates[hp]
    assert final.coordinates["discharged_charge_C"] == 0
    assert final.coordinates["delivered_work_J"] == 0
    assert final.amounts_mol["plating:Li"] < initial.amounts_mol["plating:Li"]
    assert final.amounts_mol["dead:Li"] == initial.amounts_mol["dead:Li"]
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-12)
    validate_plating_state(s, final)


@pytest.mark.parametrize("current", [-.2, .2])
def test_zero_plating_rate_recovers_original_axen_physics(current):
    s = model(rate=0.)
    original = replace(s, plating=None)
    actual = advance_plating(s, s.initial_state(), program(current), 2.).final_state
    expected = advance_battery(original, original.initial_state(), program(current), 2.).final_state
    for key in original.inventory_keys:
        assert actual.amounts_mol[key] == pytest.approx(expected.amounts_mol[key], abs=2e-12)
    for key in s.coordinate_keys:
        assert actual.coordinates[key] == pytest.approx(expected.coordinates[key], abs=3e-8)
    assert plating_readout(s, actual, current).voltage_V == pytest.approx(
        battery_readout(original, expected, current).voltage_V, abs=2e-9)


def test_identical_branches_recover_single_valued_plating_with_independent_memory():
    s = model()
    for name in ("negative", "positive"):
        directional = getattr(s, name + "_ocp")
        curve = replace(directional.lithiation,
            polynomial_V=(getattr(s, name + "_ocv_intercept_V"),
                          getattr(s, name + "_ocv_slope_V")))
        s = replace(s, **{
            name + "_ocp": replace(directional, lithiation=curve, delithiation=curve)})
    original = replace(s, negative_ocp=None, positive_ocp=None)
    a, b = s.initial_state(), original.initial_state()

    def accurate_step(spec, state, current):
        system = plating_system(spec, program(current, start=state.time_s))
        # Extra passive h variables can change adaptive error control even
        # when both voltage branches coincide. Resolve both paths more finely
        # while retaining the original physical equality thresholds below.
        return replace(system, policy=replace(system.policy, rtol=1e-10, atol=1e-13)).advance(
            state, .5).final_state

    for current in (-.2, 0., .2):
        a = accurate_step(s, a, current)
        b = accurate_step(original, b, current)
        for key in s.inventory_keys:
            assert a.amounts_mol[key] == pytest.approx(b.amounts_mol[key], abs=2e-12)
        for key in original.coordinate_keys:
            assert a.coordinates[key] == pytest.approx(b.coordinates[key], abs=3e-8)
        assert plating_readout(s, a, current).voltage_V == pytest.approx(
            plating_readout(original, b, current).voltage_V, abs=2e-9)
    key = s.hysteresis_keys[0]
    assert a.coordinates[key] != s.initial_state().coordinates[key]


@pytest.mark.parametrize("negative,positive", [(True, False), (False, True), (True, True)])
def test_branch_mixture_enters_real_bv_and_finite_cv(negative, positive):
    s = model(negative=negative, positive=positive, maximum_current_A=.3)
    state = s.initial_state()
    f = plating_fields(s, state, current_A=.1)
    for i, name in enumerate(("negative", "positive")):
        expected = s.operational_ocp(name, f["surface_fraction"][i], .1, state)
        actual = (f["solid_potential_V"][i]-f["electrolyte_potential_V"]
                  -f["kinetic_overpotential_V"][i])
        assert actual == pytest.approx(expected, abs=2e-12)
    p = program(f["voltage_V"], "V")
    assert plating_control_mode(s, state, p)[0] == "constant_voltage"
    result = advance_plating(s, state, p, 1.)
    assert result.outcome == "success"
    assert plating_fields(s, result.final_state, program=p)["voltage_V"] == pytest.approx(
        f["voltage_V"], abs=2e-9)
    for target, mode, current in ((2.6, "positive_current_limit", .3),
                                  (4.4, "negative_current_limit", -.3)):
        p = program(target, "V", start=result.final_state.time_s)
        assert plating_control_mode(s, result.final_state, p)[0] == mode
        assert plating_fields(s, result.final_state, program=p)["current_A"] == pytest.approx(
            current)


@pytest.mark.parametrize("current,h", [(-.3, -1.), (.3, 1.)])
def test_legal_h_endpoints_remain_invariant_under_actual_reaction_direction(current, h):
    s = model(h_negative=h, h_positive=-h, rate=1e-11)
    initial = s.initial_state()
    f, answer = rates(s, initial, program(current))
    assert np.sign(f["reaction_current_A"][0]) == h
    assert all(answer.coordinate_rates[key] == 0 for key in s.hysteresis_keys)
    result = advance_plating(s, initial, program(current), 1.)
    assert result.outcome == "success"
    assert all(result.final_state.coordinates[key] == initial.coordinates[key]
               for key in s.hysteresis_keys)


def test_exact_zero_main_reaction_uses_existing_nonsmooth_rest_convention():
    s = model(rate=0.)
    _, answer = rates(s, s.initial_state(), program(0.))
    assert all(answer.coordinate_rates[key] == 0 for key in s.hysteresis_keys)
    for name in ("negative", "positive"):
        assert getattr(s, name + "_ocp").reaction_rate_derivatives(
            0., .2, getattr(s, name).capacity_mol) == (0., 0., 0.)


def test_charge_rest_reverse_preserves_all_lithium_and_shared_charge_work():
    s = model()
    initial, state = s.initial_state(), s.initial_state()
    for current, duration in ((-.3, 2.), (0., 2.), (.2, 3.)):
        result = advance_plating(s, state, program(current, start=state.time_s), duration)
        assert result.outcome == "success"
        state = result.final_state
        validate_plating_state(s, state)
        assert sum(state.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-12)
        assert all(-1 <= state.coordinates[k] <= 1 for k in s.hysteresis_keys)
    assert state.time_s == 7
    assert state.coordinates["discharged_charge_C"] == pytest.approx(0., abs=1e-10)
    assert state.coordinates["delivered_work_J"] != 0
    assert result.energy_balance_residual_J is None


def test_surface_cutoff_and_reverse_recovery_preserve_h_and_real_metals():
    s = model(initial_negative_fraction=.045, rate=1e-11)
    result = advance_plating(s, s.initial_state(), program(.3), 1000.)
    assert result.outcome == "stopped"
    assert result.stopping_event == "battery:negative:depleted"
    assert 0 < result.final_state.time_s < 1000
    resumed = advance_plating(s, result.final_state,
        program(-.2, start=result.final_state.time_s), 1.)
    assert resumed.outcome == "success"
    validate_plating_state(s, resumed.final_state)


def test_nonincreasing_branch_certificate_cannot_be_bypassed_by_side_adapter():
    s = model()
    curve = replace(s.negative_ocp.delithiation, polynomial_V=(.1, .2))
    with pytest.raises(ValueError, match="certified nonincreasing"):
        replace(s, negative_ocp=replace(s.negative_ocp, delithiation=curve))
