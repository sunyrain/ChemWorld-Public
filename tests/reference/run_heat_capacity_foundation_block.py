"""Prespecified 348-unit 22C1 development comparison; retain every failure."""

import argparse
import json
import math
import runpy
import time
import traceback
from pathlib import Path

import chemicals
import numpy as np
from chemicals import heat_capacity as reference
from quasi_entropy_reference import quasi_entropy_primitive

from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity, temperature_from_caloric_change
from chemworld.physchem.heat_capacity_kernels import vibrational_temperature
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
    integrate_correlation,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.specs import PropertyCorrelation

FIXTURES = runpy.run_path(str(Path(__file__).parents[1] / "test_heat_capacity_families.py"))
CASES = FIXTURES["CASES"]
REFERENCES = dict(zip(CASES, (
    "Poling", "Shomate", "TRCCp", "PPDS2", "PPDS15", "TDE_CSExpansion", "Rowlinson_Poling",
    "Rowlinson_Bondi", "Zabransky_cubic", "Zabransky_quasi_polynomial", "Perry_151",
    "Lastovka_Shaw", "Dadgostar_Shaw", "Lastovka_solid", "Cpg_statistical_mechanics",
), strict=True))
GAUSS = {n: np.polynomial.legendre.leggauss(n) for n in (64, 128)}
TOTAL = 348


def parameters(equation, variant):
    c = dict(CASES[equation][0])
    if not variant:
        return c
    if equation in {"poling_cp", "shomate_cp"}:
        factor = 1.2 if equation == "poling_cp" else .8
        return {key: value * factor for key, value in c.items()}
    changes = {
        "trc_cp": {"a0": 1.1, "a3": .8},
        "ppds2_cp": {"C_low": 1.1, "C_inf": 1.2},
        "ppds15_cp": {"a0": 1.2, "a1": 1.05},
        "tde_cs_cp": {"b": 1.2, "a1": 1.1},
        "rowlinson_poling_cp": {"omega": .8, "Cpgm": 1.1},
        "rowlinson_bondi_cp": {"omega": .8, "Cpgm": 1.1},
        "dadgostar_shaw_cp": {"similarity_variable": .9, "MW": 1.1},
        "lastovka_solid_cp": {"similarity_variable": .9, "MW": 1.1},
    }
    for key, scale in changes.get(equation, {}).items():
        c[key] *= scale
    if equation == "zabransky_cubic_cp":
        c["a1"] += 2
    elif equation == "zabransky_quasi_cp":
        c["a3"] += 2
    elif equation == "perry151_cp":
        c["a"] += 1
    elif equation == "lastovka_shaw_cp":
        c["cyclic_aliphatic"] = 1
    elif equation == "statistical_cpg":
        c = {"linear": 1, "theta_0": 3000}
    return c


def make_card(equation, coefficients, bounds=None):
    _, phase, T, _ = CASES[equation]
    return PropertyCorrelation("fixed:" + equation, equation, coefficients, {"temperature": "K"},
        "J/(mol*K)", phase + "_heat_capacity", {"temperature": bounds or (.8*T, 1.2*T)},
        "Fixed 22C1 equation reproduction, not material validation")


def reference_keywords(equation, c):
    if equation == "statistical_cpg":
        return {"linear": bool(c["linear"]),
                "thetas": [c[f"theta_{i}"] for i in range(len(c) - 1)]}
    return dict(c)


def ref_value(equation, c, T):
    return getattr(reference, REFERENCES[equation])(T, **reference_keywords(equation, c))


def near(actual, expected, rtol, atol):
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (actual, expected)
    return abs(actual - expected)


def slope_reference(equation, c, T):
    h = T * 1e-4
    def function(t):
        return ref_value(equation, c, t)
    if equation == "trc_cp" and c["a7"] == T:
        answer = (25/12*function(T) - 4*function(T-h) + 3*function(T-2*h)
                  - 4/3*function(T-3*h) + .25*function(T-4*h)) / h
        return answer, h, "five_point_backward"
    if equation == "trc_cp" and c["a7"] < T:
        h = min(h, (T - c["a7"]) / 4)
    answer = (function(T-2*h) - 8*function(T-h) + 8*function(T+h) - function(T+2*h)) / (12*h)
    return answer, h, "five_point_centered"


def compare_point(curve, T, *, derivative=True):
    equation, c = curve.equation_id, curve.coefficients
    expected = ref_value(equation, c, T)
    actual = evaluate_correlation(curve, temperature_K=T).value
    metrics = {"temperature_K": T, "value": actual, "reference": expected,
               "value_error": near(actual, expected, 2e-11, 1e-10)}
    if derivative:
        expected, step, mode = slope_reference(equation, c, T)
        actual = correlation_temperature_derivative(curve, temperature_K=T).value
        metrics.update(slope=actual, reference_slope=expected,
            slope_error=near(actual, expected, 3e-7, 2e-9), difference_step_K=step,
            derivative_reference=mode)
    return metrics


def reference_integral(equation, c, lower, upper, entropy, quasi_reference="source"):
    if equation == "zabransky_quasi_cp" and entropy and quasi_reference == "series":
        high, high_tail = quasi_entropy_primitive(upper, c)
        low, low_tail = quasi_entropy_primitive(lower, c)
        return high-low, "independent_quasi_series_primitive", high_tail+low_tail
    suffix = "_integral_over_T" if entropy else "_integral"
    function = getattr(reference, REFERENCES[equation] + suffix, None)
    if function:
        kwargs = reference_keywords(equation, c)
        return function(upper, **kwargs) - function(lower, **kwargs), "source_primitive", 0.0
    if lower == upper:
        return 0.0, "zero_interval", 0.0
    midpoint, half = (upper + lower) / 2, (upper - lower) / 2
    answers = []
    for order in (64, 128):
        nodes, weights = GAUSS[order]
        answers.append(half * math.fsum(float(w) * ref_value(equation, c, midpoint + half*float(x))
                                       / ((midpoint + half*float(x)) if entropy else 1)
                                       for x, w in zip(nodes, weights, strict=True)))
    difference = abs(answers[1] - answers[0])
    assert difference <= max(2e-8, 2e-11*abs(answers[1])), difference
    return answers[1], "independent_gauss_legendre_128", difference


def compare_integral(curve, lower, upper, entropy, quasi_reference="source"):
    expected, mode, convergence = reference_integral(curve.equation_id, curve.coefficients,
                                                     lower, upper, entropy, quasi_reference)
    actual = integrate_correlation(curve, initial_temperature_K=lower, final_temperature_K=upper,
        over_temperature=entropy, epsabs=1e-9, epsrel=1e-11)
    diagnostics = {}
    if mode == "independent_quasi_series_primitive":
        original = reference_integral(curve.equation_id, curve.coefficients,
                                      lower, upper, entropy)[0]
        diagnostics = {"source_primitive_difference": original-expected,
                       "series_tail_bound": convergence}
    else:
        diagnostics["reference_order_difference"] = convergence
    return {"value": actual.value, "reference": expected, "reference_mode": mode,
            **diagnostics,
            "error": near(actual.value, expected, 2e-10, 2e-8),
            "reported_error_estimate": actual.absolute_error_estimate,
            "evaluations": actual.evaluations}


def inverse(curve, center, entropy, quasi_reference="source"):
    target, mode, _ = reference_integral(curve.equation_id, curve.coefficients,
                                        center, 1.1*center, entropy, quasi_reference)
    actual = temperature_from_caloric_change(curve, target, reference_temperature_K=center,
        temperature_bounds_K=(.85*center, 1.15*center), entropy=entropy)
    return {"temperature_K": actual, "reference_mode": mode,
            "temperature_error_K": near(actual, 1.1*center, 0, 2e-7)}


def piecewise(T, entropy):
    curve = PiecewiseHeatCapacity("two-ranges", FIXTURES["segments"](), "Fixed synthetic gas")
    middle = min(T, 400)
    if entropy:
        expected = R*(4*math.log(middle/300) + .001*(middle-300))
        if T > 400:
            expected += R*(5*math.log(T/400) + .002*(T-400))
    else:
        expected = R*(4*(middle-300) + .0005*(middle**2-300**2))
        if T > 400:
            expected += R*(5*(T-400) + .001*(T**2-400**2))
    actual = integrate_correlation(curve, initial_temperature_K=300, final_temperature_K=T,
                                    over_temperature=entropy)
    root = temperature_from_caloric_change(curve, expected, reference_temperature_K=300,
                                           entropy=entropy)
    return {"integral_error": near(actual.value, expected, 2e-10, 2e-8),
            "temperature_error_K": near(root, T, 0, 2e-7)}


def vibration_limit(curve, temperature):
    baseline = 3.5 if curve.coefficients["linear"] else 4
    modes = len(curve.coefficients) - 1
    expected = (baseline + (modes if temperature == 1e10 else 0)) * R
    actual = evaluate_correlation(curve, temperature_K=temperature).value
    return {"value": actual, "reference": expected, "reference_mode": "analytic_vibration_limit",
            "value_error": near(actual, expected, 2e-11, 1e-10)}


def run(output, quasi_reference="source"):
    if output.exists():
        raise FileExistsError("preserve previous results; select a new output")
    assert chemicals.__version__ == "1.5.2"
    units = []
    for equation in CASES:
        center = CASES[equation][2]
        for variant in (0, 1):
            curve = make_card(equation, parameters(equation, variant))
            prefix = f"{equation}:{variant}"
            for fraction in (.85, 1, 1.15):
                units.append((prefix + f":point:{fraction}", lambda c=curve, t=center*fraction:
                              compare_point(c, t)))
            for entropy in (False, True):
                for low, high in ((.85*center, 1.15*center), (1.15*center, .85*center),
                                  (center, center)):
                    units.append((prefix + f":integral:{entropy}:{low}:{high}",
                                  lambda c=curve, a=low, b=high, s=entropy:
                                  compare_integral(c, a, b, s, quasi_reference)))
                units.append((prefix + f":inverse:{entropy}", lambda c=curve, t=center, s=entropy:
                              inverse(c, t, s, quasi_reference)))
    for temperature in (300, 400, 500):
        for entropy in (False, True):
            units.append((f"piecewise:{temperature}:{entropy}", lambda t=temperature, s=entropy:
                          piecewise(t, s)))
    for temperature in (150, 201, 201.0001, 1000):
        curve = make_card("trc_cp", parameters("trc_cp", 0), (100, 1200))
        units.append((f"TRC-branch:{temperature}",
                      lambda c=curve, t=temperature: compare_point(c, t)))
    for variant in (0, 1):
        curve = make_card("statistical_cpg", parameters("statistical_cpg", variant), (1, 1e10))
        for temperature in (1, 1e10):
            units.append((f"vibration-limit:{variant}:{temperature}",
                          lambda c=curve, t=temperature: vibration_limit(c, t)))
    for temperature in (350, 1000):
        coefficients = {**parameters("lastovka_shaw_cp", 0), "term_A": .8}
        curve = make_card("lastovka_shaw_cp", coefficients, (300, 1200))
        units.append((f"Lastovka-term-A:{temperature}", lambda c=curve, t=temperature:
                      compare_point(c, t)))
    for scale in (1, .9365):
        units.append((f"frequency:{scale}", lambda s=scale: {"temperature_error_K": near(
            vibrational_temperature(667, scale=s),
            6.62607015e-34 * 299792458 / 1.380649e-23 * 66700 * s, 2e-14, 1e-12)}))
    assert len(units) == TOTAL
    started = time.perf_counter()
    results = []
    for identity, action in units:
        row = {"identity": identity}
        try:
            row.update(status="passed", metrics=action())
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        results.append(row)
        if len(results) % 20 == 0 or len(results) == TOTAL:
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = len(results)/elapsed
            print(f"caloric block {len(results)}/{TOTAL} "
                  f"failed={sum(r['status']=='failed' for r in results)} "
                  f"rate={rate:.1f}/s ETA={(TOTAL-len(results))/rate:.1f}s", flush=True)
    failures = [r for r in results if r["status"] == "failed"]
    summary = {"block": "22C1-heat-capacity", "mode": "development", "reference": "chemicals 1.5.2",
        "total": TOTAL, "passed": TOTAL-len(failures), "failed": len(failures),
        "elapsed_seconds": time.perf_counter()-started,
        "quasi_entropy_reference": quasi_reference,
        "denominators": {"points": 90, "integrals": 180, "inverses": 60, "piecewise": 6,
                         "specials": 12}, "results": results, "failures": failures}
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fields = ("total", "passed", "failed", "elapsed_seconds")
    print(json.dumps({k: summary[k] for k in fields}))
    return int(bool(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quasi-entropy-reference", choices=("source", "series"), default="source")
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.quasi_entropy_reference))
