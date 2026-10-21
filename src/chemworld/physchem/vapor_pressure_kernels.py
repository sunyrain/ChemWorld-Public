"""Native pressure formulas composed from the published mathematical relations.

No source lookup, ranking, automatic extrapolation or runtime state lives here.
The correlation registry owns parameter/property contracts; this pure layer
accepts native derivative arithmetic so temperature and parameter sensitivities
use the same mathematical expression as the pressure.
"""

from math import isfinite

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    derivative_values,
    exponential,
    logarithm,
    scalar_value,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.water_saturation import water_saturation_pressure


def _positive(value, name):
    value = scalar_value(value)
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _wagner_numerator(tau, values, powers):
    return sum(a * tau**power for a, power in zip(values, powers, strict=True)
               if scalar_value(a) or isinstance(a, FirstDerivative))


def pressure_logarithm(equation, temperature, coefficients):
    """Natural log of pressure in the coefficients' declared pressure unit.

    The legacy Antoine input may use Celsius; its denominator is validated in
    that convention. All other formulas here use absolute kelvin.
    """
    T, c = temperature, coefficients
    if equation != "antoine":
        _positive(T, "temperature")
    if equation == "henry_pressure":
        # The six-parameter Henry log fit is exactly the existing TDE subset.
        return pressure_logarithm("tde_pv_expansion", T, {
            "a1": c["A"], "a2": c.get("B", 0), "a3": c.get("C", 0),
            "a4": c.get("D", 0), "a5": c.get("F", 0), "a6": c.get("E", 0),
        })
    if equation in {"dippr101", "dippr115"}:
        value = c["A"]+c["B"]/T+c.get("C", 0)*logarithm(T)
        if equation == "dippr101":
            # exp(E ln T) also allows analytic derivatives with respect to E.
            D, E = c.get("D", 0), c.get("E", 0)
            if scalar_value(D) or isinstance(D, FirstDerivative):
                power = exponential(E*logarithm(T)) if isinstance(E, FirstDerivative) else T**E
                value += D*power
        else:
            value += c.get("D", 0)*T*T+c.get("E", 0)/T**2
        return value
    if equation in {"antoine", "trc_antoine_extended"}:
        _positive(T + c["C"], "Antoine T+C denominator")
        base = c.get("base", 10.0) if equation == "antoine" else 10.0
        _positive(base, "logarithm base")
        if scalar_value(base) == 1:
            raise ValueError("Antoine logarithm base cannot be one")
        reduced = c["A"] - c["B"] / (T + c["C"])
        if equation == "trc_antoine_extended":
            _positive(c["Tc"], "Tc")
            _positive(c["n"], "TRC exponent n")
            x = (T - c["to"] - 273.15) / c["Tc"]
            if scalar_value(x) > 0:
                reduced += (
                    0.43429 * exponential(c["n"] * logarithm(x)) + c["E"] * x**8 + c["F"] * x**12
                )
        return logarithm(base) * reduced
    if equation == "yaws_psat":
        return logarithm(10) * (c["A"] + c["B"] / T + c["D"] * T + c["E"] * T * T) + c[
            "C"
        ] * logarithm(T)
    if equation == "tde_pv_expansion":
        return (
            c["a1"]
            + c["a2"] / T
            + c["a3"] * logarithm(T)
            + c.get("a4", 0) * T
            + c.get("a5", 0) * T * T
            + c.get("a6", 0) / T**2
            + c.get("a7", 0) * T**6
            + c.get("a8", 0) / T**4
        )
    if equation == "arrhenius_pressure":
        _positive(c["T_ref"], "T_ref")
        _positive(c["P_ref"], "P_ref")
        return logarithm(c["P_ref"]) + c["slope"] * (1 / T - 1 / c["T_ref"])
    if equation == "clapeyron_sublimation":
        _positive(c["Tt"], "Tt")
        _positive(c["Pt"], "Pt")
        _positive(c["Hsub_t"], "Hsub_t")
        return logarithm(c["Pt"]) - c["Hsub_t"] / R * (1 / T - 1 / c["Tt"])
    if equation == "iapws_saturation":
        return logarithm(water_saturation_pressure(T))
    _positive(c["Tc"], "Tc")
    _positive(c["Pc"], "Pc")
    Tr = T / c["Tc"]
    tau = 1 - Tr
    if scalar_value(Tr) > 1 or (equation == "wagner" and scalar_value(Tr) == 1):
        raise ValueError("pressure formula requires subcritical temperature")
    if equation in {"wagner", "wagner25"}:
        powers = (1, 1.5, 3, 6) if equation == "wagner" else (1, 1.5, 2.5, 5)
        reduced = _wagner_numerator(tau, [c[k] for k in "abcd"], powers) / Tr
    elif equation == "boiling_critical_pressure":
        _positive(c["Tb"], "Tb")
        if scalar_value(c["Tb"]) >= scalar_value(c["Tc"]):
            raise ValueError("normal boiling temperature must be below Tc")
        Tbr = c["Tb"] / c["Tc"]
        reduced = Tbr * logarithm(c["Pc"] / 101325) / (1 - Tbr) * (1 - 1 / Tr)
    elif equation == "lee_kesler_pressure":
        reduced = (
            5.92714
            - 6.09648 / Tr
            - 1.28862 * logarithm(Tr)
            + 0.169347 * Tr**6
            + c["omega"] * (15.2518 - 15.6875 / Tr - 13.4721 * logarithm(Tr) + 0.43577 * Tr**6)
        )
    elif equation == "ambrose_walton_pressure":
        parameters = (
            (-5.97616, 1.29874, -0.60394, -1.06841),
            (-5.03365, 1.11505, -5.41217, -7.46628),
            (-0.64771, 2.41539, -4.26979, 3.25259),
        )
        reduced = sum(
            c["omega"] ** i * _wagner_numerator(tau, row, (1, 1.5, 2.5, 5)) / Tr
            for i, row in enumerate(parameters)
        )
    elif equation == "sanjari_pressure":
        parameters = (
            (6.83377, -5.76051, 0.90654, -1.16906),
            (5.32034, -28.1460, -58.0352, 23.57466),
            (18.19967, 16.33839, 65.6995, -35.9739),
        )
        reduced = sum(
            c["omega"] ** i * (a + b / Tr + d * logarithm(Tr) + e * Tr**1.9)
            for i, (a, b, d, e) in enumerate(parameters)
        )
    elif equation == "edalat_pressure":
        omega = c["omega"]
        d = 1 / (-0.4893 - 0.9912 * omega + 3.1551 * omega**2)
        values = (
            -6.1559 - 4.0855 * omega,
            1.5737 - 1.0540 * omega - 4.4365e-3 * d,
            -0.8747 - 7.8874 * omega,
            d,
        )
        reduced = _wagner_numerator(tau, values, (1, 1.5, 3, 6)) / Tr
    else:
        raise ValueError(f"unsupported pressure equation {equation!r}")
    return logarithm(c["Pc"]) + reduced


def pressure_derivatives(equation, temperature, coefficients, *, order=1):
    """Value and analytic temperature derivatives, with no finite-difference step."""
    try:
        if (equation == "trc_antoine_extended" and temperature == coefficients["to"]+273.15
                and order > 0 and coefficients["n"] <= order):
            raise ValueError("requested TRC derivative does not exist at the branch temperature")
        return derivative_values(
            lambda T: exponential(pressure_logarithm(equation, T, coefficients)),
            temperature,
            order=order,
        )
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("pressure or derivative is outside the finite formula domain") from exc
