"""Physical closures, coordinate semantics, boundaries and original consumers."""

import json
from dataclasses import replace
from math import exp, log

import numpy as np
import pytest

from chemworld.physchem.activity_thermodynamics import (
    activity_infinite_dilution,
    activity_parameter_jacobian,
    activity_thermodynamic_report,
)
from chemworld.physchem.equilibrium import ActivityModelSpec, activity_coefficients


def binary(model="nrtl"):
    if model == "nrtl":
        p = {
            "tau_b:A|B": -61.024979,
            "tau_b:B|A": 673.2359,
            "alpha:A|B": 0.2974,
            "alpha:B|A": 0.2974,
        }
    elif model == "wilson":
        p = {"lambda_a:A|B": log(0.154), "lambda_a:B|A": log(0.888)}
    elif model == "uniquac":
        p = {
            "r:A": 2.1055,
            "r:B": 0.92,
            "q:A": 1.972,
            "q:B": 1.4,
            "tau_b:A|B": 80.0,
            "tau_b:B|A": -300.0,
        }
    elif model == "margules":
        p = {"A:A|B": 1.2, "A:B|A": 0.8}
    else:
        p = {}
    return ActivityModelSpec(model, ("A", "B"), model, p)


@pytest.mark.parametrize("model", ("ideal", "margules", "wilson", "nrtl", "uniquac"))
def test_excess_derivatives_partial_properties_and_original_gamma_share_one_potential(model):
    spec, T, x = binary(model), 343.15, {"A": 0.252, "B": 0.748}
    report = activity_thermodynamic_report(spec, x, temperature_K=T)
    assert report.activity_coefficients == pytest.approx(
        activity_coefficients(spec, x, temperature_K=T)
    )
    for molar, partial in (
        (report.gibbs_excess_J_mol, report.partial_molar_gibbs_excess_J_mol),
        (report.enthalpy_excess_J_mol, report.partial_molar_enthalpy_excess_J_mol),
        (report.entropy_excess_J_mol_K, report.partial_molar_entropy_excess_J_mol_K),
    ):
        assert sum(x[k] * v for k, v in partial.items()) == pytest.approx(molar, abs=1e-10)
    jac = np.array(report.dln_gamma_dns)
    assert jac == pytest.approx(jac.T, abs=1e-12)
    assert np.array(list(x.values())) @ jac == pytest.approx(np.zeros(2), abs=1e-12)
    h = 0.01
    reports = [activity_thermodynamic_report(spec, x, temperature_K=T + dt) for dt in (-h, h)]
    assert (reports[1].gibbs_excess_J_mol - reports[0].gibbs_excess_J_mol) / (
        2 * h
    ) == pytest.approx(report.dGE_dT, rel=1e-7, abs=1e-9)
    assert (reports[1].enthalpy_excess_J_mol - reports[0].enthalpy_excess_J_mol) / (
        2 * h
    ) == pytest.approx(report.heat_capacity_excess_J_mol_K, rel=1e-7, abs=1e-9)
    json.dumps(report.to_dict(), allow_nan=False)
    with pytest.raises(TypeError):
        report.composition["A"] = 0.5
    with pytest.raises(TypeError):
        report.parameter_contract["parameters"][next(iter(spec.parameters), "x")] = 0


def test_asymmetric_margules_uses_standard_three_suffix_and_missing_pair_fails():
    spec = binary("margules")
    x, y, a, b = 0.2, 0.8, 1.2, 0.8
    actual = activity_coefficients(spec, {"A": x, "B": y}, temperature_K=300)
    assert actual == pytest.approx(
        {"A": exp(y * y * (a + 2 * (b - a) * x)), "B": exp(x * x * (b + 2 * (a - b) * y))}
    )
    symmetric = replace(spec, parameters={"A:A|B": 1.2})
    report = activity_thermodynamic_report(symmetric, {"A": x, "B": y}, temperature_K=300)
    assert report.activity_coefficients["A"] == pytest.approx(exp(a * y * y))
    assert report.parameter_contract["margules_symmetric_single_direction"] == ("A:A|B",)
    with pytest.raises(ValueError, match="explicit A"):
        replace(spec, parameters={})


def test_infinite_dilution_uses_exact_zero_and_preserves_mixed_solvent_ratios():
    spec = ActivityModelSpec(
        "ternary", ("A", "B", "C"), "margules", {"A:A|B": 1.2, "A:A|C": 0.8, "A:B|C": 0.3}
    )
    result = activity_infinite_dilution(
        spec, {"A": 0.2, "B": 0.3, "C": 0.5}, temperature_K=300, solute_id="A"
    )
    assert result.solvent_compositions["A"] == pytest.approx({"A": 0.0, "B": 0.375, "C": 0.625})
    assert result.solvent_compositions["A"]["A"] == 0.0
    # For symmetric pairwise GE, partial solute = sum(A_ij*x_j)-gE/RT.
    expected = 1.2 * 0.375 + 0.8 * 0.625 - 0.3 * 0.375 * 0.625
    assert result.log_activity_coefficients["A"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="remaining solvent"):
        activity_infinite_dilution(binary(), {"A": 1.0, "B": 0.0}, temperature_K=300, solute_id="A")


@pytest.mark.parametrize("quantity", ("GE", "HE", "SE", "CpE", "ln_gamma:A"))
def test_parameter_sensitivity_commutes_with_temperature_derivatives(quantity):
    from chemworld.physchem.activity_kernels import observable

    spec = binary()
    name = "tau_b:A|B"
    actual = activity_parameter_jacobian(
        spec, {"A": 0.3, "B": 0.7}, temperature_K=320, parameters=(name,), quantities=(quantity,)
    )[quantity][name]
    h = 0.001
    values = [
        observable(
            replace(spec, parameters={**spec.parameters, name: spec.parameters[name] + d}),
            (0.3, 0.7),
            320,
            quantity,
        )
        for d in (-h, h)
    ]
    assert actual == pytest.approx((values[1] - values[0]) / (2 * h), rel=1e-7, abs=1e-9)


def test_parameter_shadowing_invalid_temperature_and_zero_state_are_explicit():
    spec = replace(binary("wilson"), parameters={**binary("wilson").parameters, "lambda:A|B": 0.4})
    with pytest.raises(ValueError, match="active"):
        activity_parameter_jacobian(
            spec,
            {"A": 0.3, "B": 0.7},
            temperature_K=320,
            parameters=("lambda_a:A|B",),
            quantities=("GE",),
        )
    for T in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            activity_thermodynamic_report(spec, {"A": 0.3, "B": 0.7}, temperature_K=T)
    with pytest.raises(ValueError):
        activity_thermodynamic_report(spec, {"A": 0.0, "B": 0.0}, temperature_K=300)
