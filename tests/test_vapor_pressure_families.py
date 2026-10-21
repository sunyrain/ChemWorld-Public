"""Actual pressure declarations, source-coordinate derivatives and inverse consumers."""

from dataclasses import replace
from math import inf, log

import pytest

from chemworld.foundation.dimensions import (
    dimension_definition,
    unmapped_supported_unit_dimensions,
)
from chemworld.physchem import (
    PropertyCorrelation,
    antoine_coefficients_from_point,
    arrhenius_pressure_extrapolation,
    dippr101_coefficients_from_point,
    evaluate_correlation,
    pure_saturation_temperature_report,
    vapor_pressure_parameter_jacobian,
    vapor_pressure_report,
    vapor_pressure_temperature_derivative,
)


def antoine():
    return PropertyCorrelation(
        "synthetic",
        "antoine",
        {"A": 8, "B": 1500, "C": 200},
        {"temperature": "degC"},
        "mmHg",
        "vapor_pressure",
        {"temperature": (1, 100)},
    )


def test_legacy_temperature_and_pressure_basis_survives_all_three_derivatives():
    curve = antoine()
    T = 350
    P = evaluate_correlation(curve, temperature_K=T).value
    D = T - 273.15 + 200
    l1, l2, l3 = log(10) * 1500 / D**2, -2 * log(10) * 1500 / D**3, 6 * log(10) * 1500 / D**4
    expected = (P * l1, P * (l1 * l1 + l2), P * (l1**3 + 3 * l1 * l2 + l3))
    factor = evaluate_correlation(curve, temperature_K=T).to("Pa").value / P
    for order, value in enumerate(expected, 1):
        result = vapor_pressure_temperature_derivative(curve, temperature_K=T, order=order)
        unit = "Pa/K" + (f"^{order}" if order > 1 else "")
        assert result.value == pytest.approx(value)
        assert result.to(unit).value == pytest.approx(value * factor)
        assert result.quantity().value == result.value
        assert (
            dimension_definition(f"pressure_temperature_derivative_{order}").vector.temperature
            == -order
        )
    assert unmapped_supported_unit_dimensions() == ()


def test_antoine_jacobian_uses_explicit_source_coordinate_columns():
    curve = antoine()
    T = 350
    P = evaluate_correlation(curve, temperature_K=T).value
    D = T - 273.15 + 200
    report = vapor_pressure_parameter_jacobian(curve, (T,), parameters=("A", "B", "C"))
    assert report.parameters == ("A", "B", "C")
    assert report.values[0] == pytest.approx(
        (P * log(10), -P * log(10) / D, P * log(10) * 1500 / D**2)
    )
    assert report.to_dict()["pressure_unit"] == "mmHg"
    with pytest.raises(ValueError):
        vapor_pressure_parameter_jacobian(curve, (T,), parameters=("A", "A"))


def test_local_coefficient_matching_and_explicit_extrapolation():
    curve = PropertyCorrelation(
        "point",
        "antoine",
        {"A": 8, "B": 1500, "C": -30},
        {"temperature": "K"},
        "Pa",
        "vapor_pressure",
        {"temperature": (300, 400)},
    )
    T = 350
    pressure = evaluate_correlation(curve, temperature_K=T).value
    first, second = (
        vapor_pressure_temperature_derivative(curve, temperature_K=T, order=k).value for k in (1, 2)
    )
    point = {"temperature_K": T, "pressure_Pa": pressure, "first_derivative_Pa_K": first}
    coefficients = antoine_coefficients_from_point(**point, second_derivative_Pa_K2=second)
    assert coefficients == pytest.approx({**curve.coefficients, "base": 10})
    for equation, values, orders in (
        ("antoine", antoine_coefficients_from_point(**point), (1,)),
        (
            "dippr101",
            dippr101_coefficients_from_point(**point, second_derivative_Pa_K2=second),
            (1, 2),
        ),
        ("dippr101", dippr101_coefficients_from_point(**point, second_derivative_Pa_K2=inf), (1,)),
    ):
        local = replace(curve, equation_id=equation, coefficients=values)
        assert evaluate_correlation(local, temperature_K=T).value == pytest.approx(pressure)
        for order in orders:
            actual = vapor_pressure_temperature_derivative(
                local, temperature_K=T, order=order
            ).value
            assert actual == pytest.approx((first, second)[order - 1])
    extrapolated = arrhenius_pressure_extrapolation(
        curve, anchor_temperature_K=T, temperature_bounds_K=(200, 500)
    )
    assert extrapolated.metadata["extrapolation"] == "explicit_lnP_inverseT"
    assert evaluate_correlation(extrapolated, temperature_K=T).value == pytest.approx(pressure)
    assert vapor_pressure_report(extrapolated, temperature_K=T).dp_dt_pa_per_k == pytest.approx(
        first
    )
    with pytest.raises(ValueError):
        evaluate_correlation(curve, temperature_K=250, validity_policy="raise")
    assert evaluate_correlation(extrapolated, temperature_K=250, validity_policy="raise").value > 0


@pytest.mark.parametrize(
    "method,coefficients,bounds,T",
    (
        ("iapws_saturation", {}, (273.15, 647), 373.15),
        ("wagner25", {"Tc": 600, "Pc": 3e6, "a": -7, "b": 1, "c": -1, "d": -2}, (250, 550), 400),
        ("yaws_psat", {"A": 8, "B": -1500, "C": 0, "D": 0, "E": 0}, (250, 550), 400),
        ("tde_pv_expansion", {"a1": 20, "a2": -3000, "a3": 0}, (250, 550), 400),
        ("lee_kesler_pressure", {"Tc": 600, "Pc": 3e6, "omega": 0.3}, (250, 550), 400),
    ),
)
def test_new_pressure_formulas_feed_existing_saturation_solver(method, coefficients, bounds, T):
    curve = PropertyCorrelation(
        method,
        method,
        coefficients,
        {"temperature": "K"},
        "Pa",
        "vapor_pressure",
        {"temperature": bounds},
    )
    report = vapor_pressure_report(curve, temperature_K=T)
    inverse = pure_saturation_temperature_report(curve, pressure_Pa=report.pressure_pa)
    assert inverse.converged
    assert inverse.saturation_temperature_K == pytest.approx(T, abs=1e-6)
    assert vapor_pressure_temperature_derivative(curve, temperature_K=T, order=2).valid


def test_trc_value_does_not_require_a_nonexistent_derivative_at_the_branch():
    curve = PropertyCorrelation(
        "branch",
        "trc_antoine_extended",
        {"Tc": 600, "to": 26.85, "A": 8, "B": 1500, "C": 0, "n": 0.5, "E": 0, "F": 0},
        {"temperature": "K"},
        "Pa",
        "vapor_pressure",
        {"temperature": (250, 450)},
    )
    assert evaluate_correlation(curve, temperature_K=300).value == pytest.approx(1000)
    with pytest.raises(ValueError, match="does not exist"):
        vapor_pressure_temperature_derivative(curve, temperature_K=300)
    with pytest.raises(ValueError, match="does not exist"):
        vapor_pressure_parameter_jacobian(curve, (300,), parameters=("to",))


def test_inverse_stays_within_source_even_when_evaluation_policy_ignores_warnings():
    curve = antoine()
    pressure = vapor_pressure_report(curve, temperature_K=350).pressure_pa
    for bounds in ((100, 350), (350, 1000), (float("nan"), 400)):
        with pytest.raises(ValueError):
            pure_saturation_temperature_report(
                curve, pressure_Pa=pressure, temperature_bounds_K=bounds, validity_policy="ignore"
            )
    with pytest.raises(ValueError, match="required"):
        pure_saturation_temperature_report(
            replace(curve, validity_ranges={}), pressure_Pa=pressure, critical_temperature_K=600
        )


def test_inverse_branch_selection_does_not_claim_a_globally_unique_root():
    curve = PropertyCorrelation(
        "two_roots",
        "yaws_psat",
        {"A": 5, "B": 0, "C": 0, "D": -0.02, "E": 2e-5},
        {"temperature": "K"},
        "Pa",
        "vapor_pressure",
        {"temperature": (300, 700)},
    )
    pressure = evaluate_correlation(curve, temperature_K=400).value
    with pytest.raises(ValueError, match="outside the saturation bracket"):
        pure_saturation_temperature_report(curve, pressure_Pa=pressure)
    for bracket, expected in (((300, 500), 400), ((500, 700), 600)):
        answer = pure_saturation_temperature_report(
            curve, pressure_Pa=pressure, temperature_bounds_K=bracket
        )
        assert answer.saturation_temperature_K == pytest.approx(expected)
        assert "bracketed_root_only_uniqueness_not_certified" in answer.warnings


def test_legacy_wagner_default_inverse_reports_its_open_source_endpoint():
    from math import nextafter

    curve = PropertyCorrelation(
        "open",
        "wagner",
        {"Tc": 600, "Pc": 3e6, "a": -7, "b": 1, "c": -1, "d": -2},
        {"temperature": "K"},
        "Pa",
        "vapor_pressure",
        {"temperature": (250, 600)},
    )
    pressure = evaluate_correlation(curve, temperature_K=400).value
    answer = pure_saturation_temperature_report(curve, pressure_Pa=pressure)
    assert answer.saturation_temperature_K == pytest.approx(400, abs=1e-6)
    assert answer.bracket_temperature_K == (250, nextafter(600, 0))
    assert curve.validity_ranges["temperature"] == (250, 600)
    assert "open_critical_endpoint_uses_next_representable_temperature" in answer.warnings
