"""Domains, units and real property consumers; no chemicals dependency."""

import json
from dataclasses import replace
from math import log

import pytest

from chemworld.physchem.correlation_families import CORRELATION_FAMILIES
from chemworld.physchem.curated_properties import curated_property_package
from chemworld.physchem.enthalpy import (
    caloric_correlation_change,
    caloric_temperature,
    phase_sensible_enthalpy_report,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
    integrate_correlation,
)
from chemworld.physchem.property_packages import (
    ComponentPropertyPackage,
    SpeciesCaloricModel,
    evaluate_caloric_correlation,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K, CaloricReference
from chemworld.physchem.specs import PropertyCorrelation, property_equation_contracts
from chemworld.physchem.vapor_pressure import vapor_pressure_report


def capacity(equation="dippr107", coefficients=None, **updates):
    card = PropertyCorrelation(
        "synthetic-cp", equation,
        {"A": 30, "B": 20, "C": 600, "D": 10, "E": 300} if coefficients is None else coefficients,
        {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
        {"temperature": (200, 600)}, "Synthetic test, no material accuracy claim")
    return replace(card, **updates)


def test_equation_metadata_and_json_preserve_the_single_declaration():
    card = capacity()
    restored = PropertyCorrelation.from_dict(json.loads(json.dumps(card.to_dict())))
    assert restored == card
    contracts = property_equation_contracts()
    assert set(CORRELATION_FAMILIES) <= contracts.keys()
    for name, family in CORRELATION_FAMILIES.items():
        assert contracts[name]["reference"] == family.reference
        assert contracts[name]["output_dimensions_by_property"] == dict(family.property_dimensions)
    json.dumps(contracts, allow_nan=False)


@pytest.mark.parametrize("updates", (
    {"input_units": {"temperature": "degC"}}, {"output_unit": "Pa"},
    {"property_id": "vapor_pressure"}, {"property_id": ""},
    {"validity_ranges": {"temperature": (100, float("nan"))}},
    {"coefficients": {"A": 20, "unknown": 1}}, {"coefficients": {"B": 20}},
    {"coefficients": {"A": True}}, {"coefficients": {"A": float("inf")}},
    {"equation_id": "dippr999"},
))
def test_invalid_declarations_fail_before_evaluation(updates):
    with pytest.raises(ValueError):
        capacity(**updates)


@pytest.mark.parametrize("temperature", (0, -1, float("nan"), float("inf")))
def test_invalid_physical_temperature(temperature):
    with pytest.raises(ValueError):
        evaluate_correlation(capacity(), temperature_K=temperature)


def test_range_policy_preserves_warnings_and_never_relaxes_mathematical_domains():
    card = capacity()
    with pytest.raises(ValueError, match="outside validity"):
        integrate_correlation(card, initial_temperature_K=190, final_temperature_K=300)
    result = integrate_correlation(card, initial_temperature_K=190, final_temperature_K=300,
                                   validity_policy="warn")
    assert len(result.warnings) == 1
    with pytest.raises(ValueError, match="validity policy"):
        evaluate_correlation(card, temperature_K=300, validity_policy="typo")
    critical = capacity("dippr114", {"Tc": 600, "A": 2, "B": 50, "C": .1, "D": .1})
    for T in (600, 650):
        with pytest.raises(ValueError, match="T < Tc"):
            evaluate_correlation(critical, temperature_K=T, validity_policy="ignore")


@pytest.mark.parametrize("equation,coefficients,property_id,unit", (
    ("dippr100_reciprocal", {"A": 105000, "B": -650, "C": 1}, "liquid_molar_volume", "m^3/mol"),
    ("dippr102", {"A": 1, "B": 0, "C": -650, "D": 105000}, "gas_viscosity", "Pa*s"),
))
def test_integral_detects_interior_poles_even_with_positive_endpoints(
    equation, coefficients, property_id, unit,
):
    card = PropertyCorrelation("poles", equation, coefficients, {"temperature": "K"}, unit,
                                property_id)
    for T in (250, 400):
        assert evaluate_correlation(card, temperature_K=T).value > 0
    with pytest.raises(ValueError, match="crosses a pole"):
        integrate_correlation(card, initial_temperature_K=250, final_temperature_K=400)


def test_positive_endpoints_do_not_hide_negative_heat_capacity():
    # A very narrow negative pocket at 333 K, easily missed by quadrature nodes.
    card = capacity("dippr100", {"A": 333**2 - 1e-4, "B": -666, "C": 1})
    for T in (250, 400):
        assert evaluate_correlation(card, temperature_K=T).value > 0
    with pytest.raises(ValueError, match="nonpositive"):
        integrate_correlation(card, initial_temperature_K=250, final_temperature_K=400)


def test_oscillator_interval_bounds_cover_interior_cosh_peak_and_mixed_signs():
    card = capacity(coefficients={"A": 1, "D": -3, "E": 360})
    assert evaluate_correlation(card, temperature_K=100).value > 0
    assert evaluate_correlation(card, temperature_K=1000).value > 0
    with pytest.raises(ValueError, match=r"nonpositive|negative"):
        integrate_correlation(card, initial_temperature_K=100, final_temperature_K=1000,
                              validity_policy="ignore")
    mixed = capacity(coefficients={"A": 20, "B": 30, "C": 500, "D": -5, "E": 400})
    result = integrate_correlation(mixed, initial_temperature_K=200, final_temperature_K=500)
    assert result.value > 0


@pytest.mark.parametrize("theta", (0, 1e-12, 1e9))
@pytest.mark.parametrize("equation", ("dippr107", "dippr127"))
def test_oscillator_limits_are_finite_without_exponential_overflow(equation, theta):
    parameters = {"A": 25, "B": 10, "C": theta, "D": 0, "E": 1}
    if equation == "dippr127":
        parameters.update(F=0, G=1)
    card = capacity(equation, parameters)
    expected = 25 if theta == 1e9 else 35
    assert evaluate_correlation(card, temperature_K=300).value == pytest.approx(expected)
    assert correlation_temperature_derivative(card, temperature_K=300).value == pytest.approx(
        0, abs=1e-24)


def test_integral_orientation_zero_width_and_error_report():
    card = capacity("dippr100", {"A": 30})
    for initial, final in ((250, 400), (400, 250), (300, 300)):
        result = integrate_correlation(card, initial_temperature_K=initial,
                                       final_temperature_K=final)
        assert result.value == pytest.approx(30 * (final - initial))
        result_s = integrate_correlation(card, initial_temperature_K=initial,
                                         final_temperature_K=final, over_temperature=True)
        assert result_s.value == pytest.approx(30 * log(final / initial))
        assert result.evaluations == 0 if initial == final else result.evaluations > 0
        assert result.absolute_error_estimate >= 0
        json.dumps(result.to_dict(), allow_nan=False)


def test_impossible_quadrature_tolerance_reports_failure():
    with pytest.raises(ValueError, match="quadrature failed"):
        integrate_correlation(capacity(), initial_temperature_K=250, final_temperature_K=400,
                              epsabs=1e-25, epsrel=1e-25)


def test_output_unit_conversion_does_not_rescale_shape_parameters():
    original = capacity()
    # B/D/A scale; characteristic temperatures C/E do not.
    scaled = capacity(coefficients={"A": .030, "B": .020, "C": 600, "D": .010, "E": 300},
                      output_unit="kJ/(mol*K)")
    for entropy in (False, True):
        request = {"initial_temperature_K": 250, "final_temperature_K": 450, "entropy": entropy}
        assert caloric_correlation_change(scaled, **request).value == pytest.approx(
            caloric_correlation_change(original, **request).value, rel=1e-12)


def test_old_dippr101_and_new_pressure_reports_share_value_and_slope():
    old = PropertyCorrelation(
        "old", "dippr101_vapor_pressure", {"A": 22, "B": -3000, "C": .2, "D": 1e-5, "E": 1.5},
        {"temperature": "K"}, "Pa", "vapor_pressure")
    new = replace(old, equation_id="dippr101")
    for T in (250, 300, 400):
        a, b = (vapor_pressure_report(card, temperature_K=T) for card in (old, new))
        assert a.pressure.value == b.pressure.value
        assert a.temperature_derivative_value == b.temperature_derivative_value


def test_phase_path_and_caloric_temperature_use_new_capacity():
    card = capacity(property_id="liquid_heat_capacity")
    reference = CaloricReference(phase="liquid", temperature_K=250, energy_J_mol=-1000)

    def enthalpy(T):
        return evaluate_caloric_correlation(card, reference=reference, temperature_K=T,
                                            quantities=("enthalpy",))["enthalpy"].value

    phase = phase_sensible_enthalpy_report(
        component_id="synthetic", phase="liquid", heat_capacity_correlation=card,
        initial_temperature_K=280, final_temperature_K=430, reference_temperature_K=250)
    assert phase.total_enthalpy_J_mol == pytest.approx(enthalpy(430) - enthalpy(280))
    assert caloric_temperature(enthalpy, enthalpy(430), (250, 500),
                               energy_basis="enthalpy") == pytest.approx(430)
    # A condensed-phase Cp integral does not supply the missing U/Cv closure.
    with pytest.raises(ValueError, match="phase/EOS closure"):
        SpeciesCaloricModel(card, reference)


def test_component_package_uses_new_correlations_with_explicit_reference():
    original = curated_property_package("methane")
    card = capacity()
    component = replace(original.component, allowed_property_correlations=())
    package = ComponentPropertyPackage(component, (card,))
    reference = CaloricReference(phase="gas", temperature_K=250, entropy_J_mol_K=100)
    result = package.evaluate_state(temperature_K=400, pressure_Pa=101325, phase="gas",
                                    quantities=("cp", "enthalpy", "entropy"), reference=reference)
    assert result.properties["cp"].value == pytest.approx(
        evaluate_correlation(card, temperature_K=400).value)
    assert result.properties["enthalpy"].value > 0
    assert result.properties["entropy"].value > 100


@pytest.mark.parametrize("equation", ("cp_polynomial", "dippr100", "dippr107"))
def test_zero_cv_gas_still_has_positive_cp_and_entropy_change(equation):
    card = capacity(equation, {"a" if equation == "cp_polynomial" else "A": 0},
                    property_id="isochoric_heat_capacity")
    reference = CaloricReference(phase="gas", energy_basis="internal_energy",
                                 temperature_K=250, entropy_J_mol_K=10)
    result = evaluate_caloric_correlation(card, reference=reference, temperature_K=400,
                                         quantities=("cp", "entropy", "internal_energy"))
    assert result["cp"].value == R_J_PER_MOL_K
    assert result["internal_energy"].value == 0
    assert result["entropy"].value == pytest.approx(10 + R_J_PER_MOL_K * log(400 / 250))
