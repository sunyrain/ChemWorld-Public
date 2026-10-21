"""Caloric equations derived from published formulas, in J/(mol K).

Method contracts live only in correlation_families. No source implementation
or reference package is imported. Similarity methods require MW in g/mol and
alpha in mol/g; explicit validity ranges carry their empirical applicability.
"""

from math import exp, isfinite

import numpy as np

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    Interval,
    einstein,
    exponential,
    logarithm,
    positive_part,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R


def validate_caloric_parameters(equation, c):
    for key in ("MW", "similarity_variable", "Ts"):
        if key in c and c[key] <= 0:
            raise ValueError(f"{equation} requires positive {key}")
    for key in ("linear", "cyclic_aliphatic"):
        if key in c and c[key] not in (0, 1):
            raise ValueError(f"{key} must be explicitly encoded as 0 or 1")
    if equation == "statistical_cpg":
        modes = sorted(int(key[6:]) for key in c if key.startswith("theta_"))
        if (modes != list(range(len(modes))) or not modes
                or any(f"theta_{i}" not in c for i in modes)):
            raise ValueError("vibrational temperatures require consecutive theta_0, theta_1, ...")
        if any(c[f"theta_{i}"] <= 0 for i in modes):
            raise ValueError("vibrational temperatures must be positive")


def caloric_domain(equation, c, lower, upper):
    if not 0 < lower <= upper or not isfinite(upper):
        raise ValueError("heat-capacity interval requires finite positive temperatures")
    if "Tc" in c and upper >= c["Tc"]:
        raise ValueError(f"{equation} requires T < Tc")
    if equation == "trc_cp" and lower <= -c["a6"] <= upper and -c["a6"] > c["a7"]:
        raise ValueError("TRC interval crosses a pole in its active branch")


def _polynomial(T, coefficients):
    result = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        result = result * T + coefficient
    return result


def _polynomial_data(equation, c):
    """(lowest power, ascending coefficients), shared by evaluation and sign bounds."""
    if equation == "poling_cp":
        return 0, [R * c[key] for key in "abcde"]
    if equation == "shomate_cp":
        return -2, [c["E"], 0, c["A"], c["B"], c["C"], c["D"]]
    if equation == "perry151_cp":
        return -2, [4.184 * v for v in (c["c"], 0, c["a"], c["b"], c["d"])]
    if equation == "zabransky_cubic_cp":
        return 0, [R * c[f"a{i+1}"] / 100**i for i in range(4)]
    if equation == "dadgostar_shaw_cp":
        alpha, MW = c["similarity_variable"], c["MW"]
        return 0, [MW * value for value in (
            24.5 * (-.3416 * alpha + 2.2671 * alpha**2),
            .1064 * alpha - .3874 * alpha**2, -9.8231e-5 * alpha + 4.182e-4 * alpha**2,
        )]
    return None


def _kernel(equation, T, c):
    polynomial = _polynomial_data(equation, c)
    if polynomial:
        power, coefficients = polynomial
        return _polynomial(T, coefficients) * T**power
    if equation == "trc_cp":
        # Factored powers remove the removable T=a7 singularity, including its slope.
        u = positive_part(T - c["a7"])
        active = (T.upper > c["a7"] if isinstance(T, Interval) else
                  T.value > c["a7"] if isinstance(T, FirstDerivative) else c["a7"] < T)
        result = c["a0"] + c["a1"] / T**2 * exponential(-c["a2"] / T)
        if active:
            denominator = T + c["a6"]
            result += (c["a3"] * (u / denominator)**2
                       + (c["a4"] * u**8 - c["a5"] * u**6) / denominator**8)
        return R * result
    if equation == "ppds2_cp":
        y = T / (T + c["Ts"])
        polynomial = _polynomial(y, [c[f"a{i}"] for i in range(1, 6)])
        return R * (c["C_low"] + (c["C_inf"] - c["C_low"]) * y**2
                    * (1 + (y - 1) * polynomial))
    if equation == "ppds15_cp":
        tau = 1 - T / c["Tc"]
        return R * (c["a0"] / tau + _polynomial(tau, [c[f"a{i}"] for i in range(1, 6)]))
    if equation == "tde_cs_cp":
        return c["b"] / (1 - T / c["Tc"]) + _polynomial(
            T, [c.get(f"a{i}", 0) for i in range(1, 5)])
    if equation in {"rowlinson_poling_cp", "rowlinson_bondi_cp"}:
        reduced = T / c["Tc"]
        tau = 1 - reduced
        a, b = (1.586, .49) if equation == "rowlinson_poling_cp" else (1.45, .45)
        return c["Cpgm"] + R * (a + b / tau + c["omega"] * (
            4.2775 + 6.3 * tau**(1/3) / reduced + .4355 / tau))
    if equation == "zabransky_quasi_cp":
        reduced = T / c["Tc"]
        tau = 1 - reduced
        return R * (c["a1"] * logarithm(tau) + c["a2"] / tau
                    + _polynomial(reduced, [c[f"a{i}"] for i in range(3, 7)]))
    if equation == "statistical_cpg":
        return R * (3.5 if c["linear"] else 4.0) + R * sum(
            einstein(theta, T) for key, theta in c.items() if key.startswith("theta_"))
    alpha, MW = c["similarity_variable"], c["MW"]
    if equation == "lastovka_shaw_cp":
        if "term_A" in c:
            term_A = c["term_A"]
        elif c["cyclic_aliphatic"]:
            term_A = -.1793547 + 3.86944439 * alpha
        else:
            exponent = (alpha - .17338003) / .014
            logistic = (exp(-exponent) / (1 + exp(-exponent)) if exponent >= 0
                        else 1 / (1 + exp(exponent)))
            term_A = 1.25 + (.58 - 1.25) * logistic
        return MW * (term_A + (.73917383 + 8.88308889 * alpha)
                     * einstein(1188.28051 + 1813.04613 * alpha, T)
                     + (.0483019 + 4.35656721 * alpha)
                     * einstein(2897.01927 + 5987.80407 * alpha, T))
    if equation == "lastovka_solid_cp":
        return MW * (3 * (.013183 * alpha + .249381 * alpha**2) * R * einstein(151.8675, T)
                     + (.026526 * alpha - .024942 * alpha**2) * T
                     + (.000025 * alpha - .000123 * alpha**2) * T**2)
    raise ValueError(f"unknown caloric equation {equation}")


def caloric_value_slope(equation, temperature, coefficients):
    caloric_domain(equation, coefficients, temperature, temperature)
    try:
        result = _kernel(equation, FirstDerivative(temperature, 1), coefficients)
        if not isinstance(result, FirstDerivative):
            result = FirstDerivative(result)
        if not isfinite(result.value) or not isfinite(result.slope):
            raise ValueError("nonfinite heat-capacity value or derivative")
        return result.value, result.slope
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{equation} exceeds its finite mathematical domain") from exc


def validate_caloric_capacity(equation, coefficients, lower, upper, *, minimum=0, allow_zero=False):
    """Bound Cp-minimum throughout the interval, splitting conservative ranges."""
    caloric_domain(equation, coefficients, lower, upper)

    def valid(value):
        return isfinite(value) and (value >= minimum if allow_zero else value > minimum)

    for temperature in (lower, upper):
        if not valid(caloric_value_slope(equation, temperature, coefficients)[0]):
            raise ValueError("heat capacity is below its required minimum")
    polynomial = _polynomial_data(equation, coefficients)
    if polynomial:
        power, values = polynomial
        # T**(-power) is positive: the transformed polynomial has the same sign.
        values[-power] -= minimum
        poly = np.polynomial.Polynomial(values)
        points = [lower, upper, *(root.real for root in poly.deriv().roots()
                                 if abs(root.imag) < 1e-8 and lower < root.real < upper)]
        if all(isfinite(poly(T)) and (poly(T) >= 0 if allow_zero else poly(T) > 0) for T in points):
            return
        raise ValueError("heat capacity violates its minimum at a polynomial extremum")
    intervals = [(lower, upper)]
    if equation == "trc_cp" and lower < coefficients["a7"] < upper:
        intervals = [(lower, coefficients["a7"]), (coefficients["a7"], upper)]
    for _ in range(8192):
        if not intervals:
            return
        low, high = intervals.pop()
        bound = _kernel(equation, Interval(low, high), coefficients)
        lower_bound = bound.lower if isinstance(bound, Interval) else bound
        if valid(lower_bound):
            continue
        mid = low + (high - low) / 2
        if mid in (low, high) or not valid(caloric_value_slope(equation, mid, coefficients)[0]):
            break
        intervals.extend(((low, mid), (mid, high)))
    raise ValueError("heat capacity is below its minimum or cannot be bounded on the interval")


def vibrational_temperature(frequency_cm_inverse, *, scale=1.0):
    """h c wavenumber / k, using exact SI constants; scale is always explicit."""
    if any(isinstance(v, bool) or not isfinite(v) or v <= 0
           for v in (frequency_cm_inverse, scale)):
        raise ValueError("vibrational frequency and scale must be positive finite numbers")
    return 100 * 6.62607015e-34 * 299792458 / 1.380649e-23 * frequency_cm_inverse * scale
