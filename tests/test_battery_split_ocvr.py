"""Independent finite-inventory/closed-form checks of the common SplitOCVR residual."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, F
from chemworld.physchem.battery_split_ocvr import (
    BatterySplitOCVRSpec,
    advance_split_ocvr,
    split_ocvr_control_mode,
    split_ocvr_fields,
    split_ocvr_readout,
    split_ocvr_system,
    validate_split_ocvr_state,
)
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def program(value=0.2, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


def test_two_real_capacities_sourced_curves_and_json_native_codec():
    spec = BatterySplitOCVRSpec()
    assert spec.supported_control_units == ("A", "V")
    payload = spec.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    restored = BatterySplitOCVRSpec.from_dict(json.loads(json.dumps(payload)))
    assert restored == spec and restored.initial_state() == spec.initial_state()
    assert spec.negative_ocv.source.data_kind == "synthetic"
    assert not hasattr(spec, "negative") and not hasattr(spec, "surface_margin")
    assert tuple(spec.initial_state().amounts_mol) == ("negative:Li", "positive:Li")
    assert spec.initial_state().amounts_mol == pytest.approx(
        {"negative:Li": 70 / F, "positive:Li": 36 / F}
    )
    reading = split_ocvr_readout(spec, spec.initial_state(), 0.2)
    assert not hasattr(reading, "negative_surface_fraction")
    assert reading.voltage_V == pytest.approx(3.81)
    assert reading.negative_average_fraction == pytest.approx(0.7)
    payload["negative_ocv"]["polynomial_V"][0] = 8.0
    assert restored.negative_ocv.polynomial_V == (0.2, -0.1)


@pytest.mark.parametrize("current,resistance", [(0.0, 0.1), (0.2, 0.1), (-0.2, 0.1), (0.2, 0.0)])
def test_cc_closed_form_inventory_charge_and_work_without_fake_heat(current, resistance):
    spec = BatterySplitOCVRSpec(resistance_ohm=resistance)
    initial = spec.initial_state()
    result = advance_split_ocvr(spec, initial, program(current), 8.0)
    final = result.final_state
    assert final.time_s == 8.0
    q = current * 8.0
    slope = -0.1 / spec.negative_capacity_C - 0.8 / spec.positive_capacity_C
    w = current * ((3.83 - current * resistance) * 8 + 0.5 * slope * current * 8**2)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(q, abs=2e-12)
    assert final.coordinates["delivered_work_J"] == pytest.approx(w, abs=2e-8)
    for e, sign in (("negative", -1), ("positive", 1)):
        assert F * (final.amounts_mol[e + ":Li"] - initial.amounts_mol[e + ":Li"]) == pytest.approx(
            sign * q, abs=2e-11
        )
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=2e-18
    )
    assert final.temperature_K == initial.temperature_K
    assert result.final_state.heat_loss_J == 0.0
    assert "no heat closure" in result.solver_diagnostic["energy_scope"]


def test_interior_cv_follows_independent_exponential_and_work_is_voltage_times_charge():
    spec = BatterySplitOCVRSpec()
    target, duration = 3.81, 8.0
    result = advance_split_ocvr(spec, spec.initial_state(), program(target, "V"), duration)
    final = result.final_state
    slope = -0.1 / 100 - 0.8 / 120
    q = (3.83 - target) / slope * np.expm1(slope / spec.resistance_ohm * duration)
    f = split_ocvr_fields(spec, final, program=program(target, "V"))
    assert f["voltage_V"] == pytest.approx(target, abs=2e-12)
    assert f["current_A"] == pytest.approx(0.2 * np.exp(slope / 0.1 * duration), abs=5e-7)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(q, abs=3e-6)
    assert final.coordinates["delivered_work_J"] == pytest.approx(target * q, abs=1.2e-5)


@pytest.mark.parametrize(
    "target,expected,mode",
    [(2.6, 2.0, "positive_current_limit"), (4.4, -2.0, "negative_current_limit")],
)
def test_cv_limit_switches_real_current_and_does_not_reuse_previous_stage(target, expected, mode):
    spec = BatterySplitOCVRSpec()
    state = advance_split_ocvr(spec, spec.initial_state(), program(0.1), 1.0).final_state
    control = program(target, "V")
    f = split_ocvr_fields(spec, state, program=control)
    assert f["current_A"] == expected
    assert split_ocvr_control_mode(spec, state, control) == (mode, expected)
    assert f["voltage_V"] != target
    final = advance_split_ocvr(spec, state, control, 0.5).final_state
    assert final.coordinates["discharged_charge_C"] == pytest.approx(
        0.1 + 0.5 * expected, abs=2e-11
    )


def test_tiny_resistance_uses_endpoint_comparisons_without_division_overflow():
    spec = BatterySplitOCVRSpec(resistance_ohm=1e-320)
    for target, current in ((2.6, 2.0), (4.4, -2.0)):
        f = split_ocvr_fields(spec, spec.initial_state(), program=program(target, "V"))
        assert f["current_A"] == current and np.isfinite(f["voltage_V"])


@pytest.mark.parametrize("target", [3.83, 3.5])
def test_zero_resistance_cv_rejects_both_nonunique_and_unreachable_cases(target):
    spec = BatterySplitOCVRSpec(resistance_ohm=0.0)
    assert spec.supported_control_units == ("A",)
    with pytest.raises(ValueError, match="nonunique"):
        split_ocvr_system(spec, program(target, "V"))


@pytest.mark.parametrize(
    "value,unit", [(0.2, "A"), (0.0, "A"), (3.81, "V"), (2.6, "V"), (4.4, "V")]
)
def test_analytic_jacobian_of_actual_rhs_including_nonlinear_ocv_and_arbitrary_order(value, unit):
    base = BatterySplitOCVRSpec()
    spec = replace(
        base,
        negative_ocv=replace(base.negative_ocv, polynomial_V=(0.2, -0.1, 0.03)),
        positive_ocv=replace(base.positive_ocv, polynomial_V=(4.2, -0.8, -0.05)),
    )
    control, state = program(value, unit), spec.initial_state()
    system = split_ocvr_system(spec, control)
    initial = system.algebraic.initial_values(state)
    names = (*spec.inventory_keys, "temperature", *spec.coordinate_keys, *BATTERY_ALGEBRAIC)
    y = np.array([*state.amounts_mol.values(), state.temperature_K, 0.0, 0.0, *initial.values()])
    scales = np.array([100 / F, 120 / F, 300.0, 100.0, 400.0, 2.0, 1.0])

    def view(values):
        return CoupledStateView(
            dict(zip(spec.inventory_keys, values[:2], strict=True)),
            values[2],
            spec.volume_l,
            coordinates=dict(zip(spec.coordinate_keys, values[3:5], strict=True)),
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, values[5:], strict=True)),
        )

    def rhs(values):
        candidate = view(values)
        rates = assemble_coupled_rates(
            candidate,
            thermodynamics=system.thermodynamics,
            processes=system.processes,
            heat_flux=system.heat_flux,
            isothermal=True,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in spec.inventory_keys],
            0.0,
            [rates.coordinate_rates[k] for k in spec.coordinate_keys],
            -system.algebraic.residual(candidate),
        ]

    analytic = system.residual_rhs_jacobian(view(y), names).toarray()
    expected = finite_jacobian(rhs, y, scales, relative_step=1e-6)
    np.testing.assert_allclose(analytic * scales, expected * scales, atol=2e-8, rtol=2e-7)
    permutation = np.array([6, 2, 0, 5, 3, 1, 4])
    permuted = system.residual_rhs_jacobian(view(y), tuple(names[i] for i in permutation)).toarray()
    np.testing.assert_array_equal(permuted, analytic[permutation][:, permutation])


@pytest.mark.parametrize(
    "electrode,bound,current",
    [
        ("negative", "minimum", 0.2),
        ("negative", "maximum", -0.2),
        ("positive", "minimum", -0.2),
        ("positive", "maximum", 0.2),
    ],
)
def test_four_independent_fraction_boundaries_stop_then_reverse_inward(electrode, bound, current):
    base = BatterySplitOCVRSpec(minimum_voltage_V=-100.0, maximum_voltage_V=100.0)
    limit = 0.2 if bound == "minimum" else 0.8
    theta = limit + (0.002 if bound == "minimum" else -0.002)
    spec = replace(
        base,
        **{
            electrode + "_" + bound + "_fraction": limit,
            "initial_" + electrode + "_fraction": theta,
        },
    )
    result = advance_split_ocvr(spec, spec.initial_state(), program(current), 10.0)
    expected_time = 0.002 * getattr(spec, electrode + "_capacity_C") / abs(current)
    assert result.final_state.time_s == pytest.approx(expected_time, abs=2e-7)
    assert result.stopping_event == "split-ocvr:" + electrode + ":" + bound + "-fraction"
    state = result.final_state
    validate_split_ocvr_state(spec, state)
    held = advance_split_ocvr(spec, state, program(current), 1.0)
    assert held.final_state.time_s == state.time_s
    reverse = advance_split_ocvr(spec, state, program(-current), 1.0).final_state
    assert reverse.time_s == state.time_s + 1.0
    fraction = split_ocvr_fields(spec, reverse, current_A=0.0)[electrode + "_average_fraction"]
    assert fraction > limit if bound == "minimum" else fraction < limit


def test_voltage_stop_keeps_actual_partial_work_and_reversed_inward_recovery():
    # Zero resistance isolates the OCV boundary from an instantaneous IR jump.
    spec = BatterySplitOCVRSpec(resistance_ohm=0.0, minimum_voltage_V=3.82)
    result = advance_split_ocvr(spec, spec.initial_state(), program(0.2), 20.0)
    expected_time = 0.01 / (0.2 * (0.1 / 100 + 0.8 / 120))
    assert result.stopping_event == "battery:minimum-voltage"
    assert result.final_state.time_s == pytest.approx(expected_time, abs=3e-6)
    assert (
        advance_split_ocvr(spec, result.final_state, program(0.2), 1.0).final_state.time_s
        == result.final_state.time_s
    )
    reverse = advance_split_ocvr(spec, result.final_state, program(-0.2), 1.0)
    assert reverse.final_state.time_s == result.final_state.time_s + 1.0


def test_legal_initial_boundary_is_not_clipped_and_supports_inward_start():
    spec = BatterySplitOCVRSpec(
        initial_negative_fraction=0.0001, minimum_voltage_V=-100.0, maximum_voltage_V=100.0
    )
    state = spec.initial_state()
    assert split_ocvr_fields(spec, state, current_A=0.0)[
        "negative_average_fraction"
    ] == pytest.approx(0.0001, abs=1e-19)
    assert advance_split_ocvr(spec, state, program(0.2), 1.0).final_state.time_s == 0.0
    assert advance_split_ocvr(spec, state, program(-0.2), 1.0).final_state.time_s == 1.0


def test_directional_recovery_does_not_accept_initial_voltage_outside_the_window():
    spec = BatterySplitOCVRSpec(resistance_ohm=0.0, minimum_voltage_V=3.9)
    # Charging points towards the window, but the initial voltage is already
    # outside it. Only a legitimate stopping boundary may recover inward.
    result = advance_split_ocvr(spec, spec.initial_state(), program(-0.2), 1.0)
    assert result.actual_duration_s == 0.0
    assert result.stopping_event == "battery:minimum-voltage"


@pytest.mark.parametrize(
    "change",
    [
        {"negative_capacity_C": 0.0},
        {"positive_capacity_C": float("inf")},
        {"resistance_ohm": -0.1},
        {"housing_volume_L": 0.0},
        {"negative_minimum_fraction": 0.8},
        {"positive_maximum_fraction": 0.2},
        {"temperature_K": True},
        {"numerical_form": "ode"},
    ],
)
def test_invalid_declarations_are_rejected(change):
    with pytest.raises(ValueError):
        BatterySplitOCVRSpec(**change)


def test_ocv_identity_domain_and_invalid_accepted_state_are_explicit():
    base = BatterySplitOCVRSpec()
    for curve in (
        replace(base.negative_ocv, supported_models=("SPM",)),
        replace(base.negative_ocv, stoichiometry_domain=(0.01, 0.99)),
    ):
        with pytest.raises(ValueError):
            replace(base, negative_ocv=curve)
    state = base.initial_state()
    bad = replace(
        state,
        amounts_mol={
            **state.amounts_mol,
            "negative:Li": state.amounts_mol["negative:Li"] + 0.01 / F,
        },
    )
    with pytest.raises(ValueError, match="Faraday"):
        validate_split_ocvr_state(base, bad)
    with pytest.raises(ValueError, match="thermal"):
        split_ocvr_system(base, program(), thermal_device=object())
