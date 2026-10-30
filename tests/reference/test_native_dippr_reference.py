"""Fixed development block defined in NATIVE_CONV22A_EXPERIMENT.md."""

from dataclasses import replace
from math import log

import chemicals
import numpy as np
import pytest
from chemicals import dippr

from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
    integrate_correlation,
)
from chemworld.physchem.property_packages import evaluate_caloric_correlation
from chemworld.physchem.property_reports import R_J_PER_MOL_K, CaloricReference
from chemworld.physchem.specs import PropertyCorrelation

# Synthetic coefficients: no material accuracy is inferred from agreement.
CASES = {
    "100": ({"A": 25, "B": .01, "C": 2e-5, "D": -1e-8, "E": 1e-11, "F": 2e-15, "G": 1e-18},
            {"A": 45, "B": -.02, "C": 8e-5, "D": 2e-8, "E": -1e-11, "F": 1e-15, "G": 2e-18}),
    "101": ({"A": 22, "B": -3000, "C": .2, "D": 1e-5, "E": 1.5},
            {"A": 30, "B": -4200, "C": -.5, "D": 2e-6, "E": 2}),
    "102": ({"A": 1e-6, "B": .7, "C": 50, "D": 200}, {"A": 2e-6, "B": .6, "C": -20, "D": 800}),
    "104": ({"A": 1e-4, "B": -.1, "C": -400, "D": 1e13, "E": -1e15},
            {"A": 2e-4, "B": -.2, "C": 250, "D": -5e12, "E": 1e14}),
    "105": ({"A": 200, "B": .3, "C": 700, "D": .3}, {"A": 300, "B": .4, "C": 620, "D": .28}),
    "106": ({"Tc": 650, "A": 50000, "B": .4, "C": .1, "D": -.02, "E": .01},
            {"Tc": 700, "A": 60000, "B": .38, "C": -.03, "D": .01, "E": -.005}),
    "107": ({"A": 25, "B": 30, "C": 800, "D": 10, "E": 400},
            {"A": 35, "B": 45, "C": 1000, "D": -5, "E": 700}),
    "114": ({"Tc": 650, "A": 2, "B": 80, "C": .2, "D": .1},
            {"Tc": 700, "A": 3, "B": 60, "C": -.1, "D": .3}),
    "115": ({"A": 20, "B": -3000, "C": .5, "D": 1e-6, "E": 3000},
            {"A": 25, "B": -4000, "C": -.2, "D": -1e-6, "E": -2000}),
    "116": ({"Tc": 650, "A": 10, "B": 20, "C": 3, "D": -2, "E": 1},
            {"Tc": 700, "A": 15, "B": 30, "C": -2, "D": 4, "E": -.5}),
    "127": ({"A": 25, "B": 30, "C": 600, "D": 20, "E": 1200, "F": 10, "G": 2400},
            {"A": 35, "B": 40, "C": 900, "D": -5, "E": 1600, "F": 20, "G": 3000}),
}
for _base in ("100", "105", "106"):
    CASES[_base + "_reciprocal"] = CASES[_base]

_PROPERTY_UNITS = {
    "100": ("liquid_heat_capacity", "J/(mol*K)"),
    "101": ("vapor_pressure", "Pa"), "102": ("gas_viscosity", "Pa*s"),
    "104": ("second_virial_coefficient", "m^3/mol"),
    "105": ("liquid_density", "kg/m^3"),
    "106": ("heat_of_vaporization", "J/mol"),
    "107": ("ideal_gas_heat_capacity", "J/(mol*K)"),
    "114": ("liquid_heat_capacity", "J/(mol*K)"),
    "115": ("vapor_pressure", "Pa"), "116": ("liquid_density", "kg/m^3"),
    "127": ("ideal_gas_heat_capacity", "J/(mol*K)"),
}


def correlation(number, parameter_set):
    property_id, unit = _PROPERTY_UNITS.get(number, ("liquid_molar_volume", "m^3/mol"))
    return PropertyCorrelation(
        f"synthetic-{number}-{parameter_set}", "dippr" + number, CASES[number][parameter_set],
        {"temperature": "K"}, unit, property_id, {"temperature": (220, 450)},
        "Fixed synthetic coefficients; chemicals 1.5.2 equation comparison")


@pytest.fixture(autouse=True)
def pinned_reference():
    assert chemicals.__version__ == "1.5.2"


@pytest.mark.parametrize("number", tuple(CASES))
@pytest.mark.parametrize("parameter_set", (0, 1))
@pytest.mark.parametrize("temperature", (240, 320, 400))
def test_values_and_slopes(number, parameter_set, temperature):
    card = correlation(number, parameter_set)
    function = getattr(dippr, "EQ" + number)
    expected = function(temperature, **card.coefficients)
    expected_slope = function(temperature, **card.coefficients, order=1)
    assert evaluate_correlation(card, temperature_K=temperature).value == pytest.approx(
        expected, rel=2e-12, abs=1e-12)
    actual_slope = correlation_temperature_derivative(card, temperature_K=temperature).value
    assert actual_slope == pytest.approx(expected_slope, rel=3e-11, abs=1e-11)


@pytest.mark.parametrize("number", tuple(CASES))
@pytest.mark.parametrize("parameter_set", (0, 1))
@pytest.mark.parametrize("over_temperature", (False, True))
def test_definite_integrals(number, parameter_set, over_temperature):
    card = correlation(number, parameter_set)
    function = getattr(dippr, "EQ" + number)
    if number in {"100", "104", "107", "114", "116", "127"}:
        order = -10 if over_temperature else -1
        expected = (function(380, **card.coefficients, order=order)
                    - function(260, **card.coefficients, order=order))
    else:
        nodes, weights = np.polynomial.legendre.leggauss(96)
        points = 320 + 60 * nodes
        values = [function(T, **card.coefficients) / (T if over_temperature else 1)
                  for T in points]
        expected = 60 * np.dot(weights, values)
    result = integrate_correlation(card, initial_temperature_K=260, final_temperature_K=380,
                                   over_temperature=over_temperature)
    assert result.value == pytest.approx(expected, rel=5e-9, abs=1e-9)
    assert result.evaluations > 0
    assert 0 <= result.absolute_error_estimate <= max(1e-10, abs(result.value) * 1e-10)


@pytest.mark.parametrize("basis", ("enthalpy", "internal_energy"))
def test_caloric_reference(basis):
    card = correlation("107", 0)
    if basis == "internal_energy":
        card = replace(card, property_id="isochoric_heat_capacity")
    ref = CaloricReference(phase="gas", temperature_K=260, pressure_Pa=1e5,
                           energy_basis=basis, energy_J_mol=-1500, entropy_J_mol_K=100)
    actual = evaluate_caloric_correlation(
        card, reference=ref, temperature_K=380, pressure_Pa=2e5,
        quantities=("cp", "cv", "enthalpy", "internal_energy", "entropy"))
    capacity = dippr.EQ107(380, **card.coefficients)
    energy = -1500 + (dippr.EQ107(380, **card.coefficients, order=-1)
                      - dippr.EQ107(260, **card.coefficients, order=-1))
    entropy = (100 + dippr.EQ107(380, **card.coefficients, order=-10)
               - dippr.EQ107(260, **card.coefficients, order=-10)
               - R_J_PER_MOL_K * log(2))
    if basis == "enthalpy":
        expected = {"cp": capacity, "cv": capacity - R_J_PER_MOL_K, "enthalpy": energy,
                        "internal_energy": energy - R_J_PER_MOL_K * 380, "entropy": entropy}
    else:
        expected = {"cv": capacity, "cp": capacity + R_J_PER_MOL_K, "internal_energy": energy,
                        "enthalpy": energy + R_J_PER_MOL_K * 380,
                        "entropy": entropy + R_J_PER_MOL_K * log(380 / 260)}
    for quantity, value in expected.items():
        assert actual[quantity].value == pytest.approx(value, rel=5e-10, abs=1e-8)
