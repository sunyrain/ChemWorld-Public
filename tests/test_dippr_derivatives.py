"""Derivative domains, source coordinates, local identifiability and consumer units."""

import json
from dataclasses import replace
from itertools import permutations
from math import exp, log

import pytest

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.dippr import (
    dippr106_coefficients_from_point,
    dippr_mixed_derivative,
    dippr_parameter_jacobian,
    dippr_temperature_response,
)
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.specs import PropertyCorrelation, property_equation_contracts


def capacity(equation="dippr107", coefficients=None, **updates):
    return replace(PropertyCorrelation(
        "derivative-capacity", equation,
        {"A": 25., "B": 30., "C": 800., "D": 10., "E": 400.}
        if coefficients is None else coefficients,
        {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
        {"temperature": (200, 500)}, "Synthetic functional test"), **updates)


def test_mixed_chain_rules_retain_all_coordinate_seeds_and_permutations():
    f = lambda p: p["x"]**2*p["y"]**3  # noqa: E731
    for coords in set(permutations(("x", "y", "y"))):
        assert partial_derivative(f, {"x": 2, "y": 3}, coords) == 72
    assert partial_derivative(f, {"x": 2, "y": 3}, ("x", "x")) == 54
    assert partial_derivative(f, {"x": 2, "y": 3}, ("x", "x", "x")) == 0
    for coords in ((), ("z",), ("x",)*4):
        with pytest.raises(ValueError):
            partial_derivative(f, {"x": 2, "y": 3}, coords)


@pytest.mark.parametrize("theta", (0., 1e-12, -1e-12, 1e9, -1e9))
@pytest.mark.parametrize("equation", ("dippr107", "dippr127"))
def test_oscillator_extremes_keep_finite_analytic_response(theta, equation):
    parameters = {"A": 25., "B": 10., "C": theta, "D": 0., "E": 1.}
    if equation == "dippr127":
        parameters.update(F=0., G=1.)
    card = capacity(equation, parameters)
    response = dippr_temperature_response(card, temperature_K=300)
    expected = 25. if abs(theta) == 1e9 else 35.
    assert response.values == pytest.approx((expected, 0, 0, 0), abs=1e-24)
    assert dippr_mixed_derivative(card, temperature_K=300, coordinates=("C",)).value == (
        pytest.approx(0, abs=1e-15))
    assert dippr_parameter_jacobian(card, [300], parameters=["A"]).values == ((1.,),)


def test_zero_oscillator_second_parameter_derivative_and_cross_derivative_are_not_dropped():
    card = capacity(coefficients={"A": 25, "B": 30, "C": 0, "D": 10, "E": 0})
    assert dippr_mixed_derivative(card, temperature_K=300, coordinates=("C", "C")).value == (
        pytest.approx(-2*30/(3*300**2)))
    assert dippr_mixed_derivative(card, temperature_K=300, coordinates=("E", "E")).value == (
        pytest.approx(2*10/300**2))
    assert dippr_mixed_derivative(card, temperature_K=300,
        coordinates=("temperature", "C", "C")).value == pytest.approx(4*30/(3*300**3))


def test_output_scaling_changes_amplitudes_but_not_temperature_coefficients():
    original = capacity()
    scaled = replace(original, output_unit="kJ/(mol*K)", coefficients={
        **original.coefficients, "A": .025, "B": .030, "D": .010})
    response = dippr_temperature_response(scaled, temperature_K=300).to("J/(mol*K)")
    assert response.values == pytest.approx(dippr_temperature_response(
        original, temperature_K=300).values, rel=1e-13)
    j = dippr_parameter_jacobian(original, [300], parameters=("A", "C"))
    js = dippr_parameter_jacobian(scaled, [300], parameters=("A", "C"))
    assert js.values[0] == pytest.approx((j.values[0][0], j.values[0][1]/1000))
    for report in (response, js, dippr_mixed_derivative(scaled, temperature_K=300,
                                                       coordinates=("C", "temperature"))):
        json.dumps(report.to_dict(), allow_nan=False)


def test_optional_parameter_must_be_declared_and_domain_policy_is_preserved():
    card = capacity("dippr100", {"A": 25})
    for names in ((), ("B",), ("A", "A")):
        with pytest.raises(ValueError, match="coefficient"):
            dippr_parameter_jacobian(card, [300], parameters=names)
    with pytest.raises(ValueError, match="temperature"):
        dippr_parameter_jacobian(card, [])
    with pytest.raises(ValueError, match="outside validity"):
        dippr_temperature_response(card, temperature_K=550)
    warned = dippr_temperature_response(card, temperature_K=550, validity_policy="warn")
    assert warned.warnings and warned.values == (25, 0, 0, 0)
    assert not dippr_temperature_response(card, temperature_K=550,
                                          validity_policy="ignore").warnings
    for order in (-1, 4, True, 1.0):
        with pytest.raises(ValueError):
            dippr_temperature_response(card, temperature_K=300, order=order)


def test_subcritical_domain_is_not_relaxed_by_derivative_or_warning_policy():
    card = PropertyCorrelation("latent", "dippr106", {"Tc": 650, "A": 50000, "B": .4},
                                {"temperature": "K"}, "J/mol", "heat_of_vaporization")
    for T in (0, 650, 700, float("nan")):
        with pytest.raises(ValueError):
            dippr_temperature_response(card, temperature_K=T, validity_policy="ignore")
    response = dippr_temperature_response(card, temperature_K=300)
    assert response.values[0] == evaluate_correlation(card, temperature_K=300).value
    assert property_equation_contracts()["dippr106"]["mixed_derivative_order"] == 3


@pytest.mark.parametrize("C", (None, -.03, .1))
def test_local_recovery_matches_value_and_derivatives_without_fitting_a_range(C):
    T, Tc, A, B = 320., 650., 50000., .4
    c = C or 0
    tau, distance = 1-T/Tc, Tc-T
    exponent = B+c*T/Tc
    value = A*tau**exponent
    slope = c/Tc*log(tau)-exponent/distance
    second = value*(slope*slope - 2*c/(Tc*distance)-exponent/distance**2)
    coefficients = dippr106_coefficients_from_point(temperature_K=T,
        critical_temperature_K=Tc, value=value, first_derivative=value*slope,
        second_derivative=None if C is None else second)
    assert coefficients == pytest.approx({"Tc": Tc, "A": A, "B": B, "C": c, "D": 0, "E": 0})


def test_local_abc_rejects_rank_deficiency_while_ab_remains_defined():
    kwargs = {"temperature_K": 650*(1-exp(-2)), "critical_temperature_K": 650,
              "value": 100., "first_derivative": -.2}
    with pytest.raises(ValueError, match="rank deficient"):
        dippr106_coefficients_from_point(**kwargs, second_derivative=.01)
    assert dippr106_coefficients_from_point(**kwargs)["A"] > 0
    for update in ({"value": 0}, {"temperature_K": 650}, {"first_derivative": float("inf")},
                   {"second_derivative": True}):
        with pytest.raises(ValueError):
            dippr106_coefficients_from_point(**{**kwargs, **update})
