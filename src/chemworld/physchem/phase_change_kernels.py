"""Latent-heat equations from published phase-change formula specifications.

SMK (Sivaraman et al., 1984) and MK (Morgan and Kobayashi, 1994) parameter
tables use the full precision scientific literals distributed with chemicals
1.5.2; no upstream executable expressions are used. The common power-series
evaluation and derivatives below are independent and use native chain rules.
All new methods use kelvin, pascal, and J/mol. Watson retains its legacy
caller-declared enthalpy scale and zero-above-critical convention.
"""

from math import isfinite, log, pi

from chemworld.physchem.caloric_math import derivative_values, logarithm, scalar_value
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R

_SMK_POWERS = (1 / 3, 5 / 6, 29 / 24, 1, 2, 3)
# MK's fitted exponents are the published decimal parameters, not exact
# fractions. Substituting SMK's rational exponents changes the fitted model.
_MK_POWERS = (0.3333, 0.8333, 1.2083, 1, 2, 3)
_SMK = (
    (6.536924, -2.466698, -77.52141, 59.63435, 36.09887, -14.60567),
    (-0.132584, -28.21525, -82.9582, 99.00008, 19.10458, -2.79566),
)
_MK = (
    (5.2804, 12.865, 1.171, -13.116, 0.4858, -1.088),
    (0.080022, 273.23, 465.08, -638.51, -145.12, 74.049),
    (7.2543, -346.45, -610.48, 839.89, 160.05, -50.711),
)


def _positive(value, name):
    if not isfinite(scalar_value(value)) or scalar_value(value) <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _series(tau, coefficients, powers):
    return sum(a * tau**p for a, p in zip(coefficients, powers, strict=True) if a != 0)


def latent_heat_value(equation, temperature, coefficients):
    """One expression per method; supports the existing nested derivative scalar."""
    T, c = temperature, coefficients
    _positive(T, "temperature")
    if equation == "constant_phase_change_enthalpy":
        return c["delta_h"]
    if equation == "phase_change_point":
        from chemworld.physchem.property_points import source_point_value

        return source_point_value(T,c["T_ref"],c["delta_h"])
    Tc = c["Tc"]
    _positive(Tc, "Tc")
    t = scalar_value(T)
    if equation == "watson_hvap":
        _positive(c["T_ref"], "T_ref")
        if c["T_ref"] >= Tc or c["Hvap_ref"] < 0:
            raise ValueError("Watson requires T_ref < Tc and nonnegative reference enthalpy")
        exponent = c.get("exponent", 0.38)
        _positive(exponent, "Watson exponent")
        if t >= Tc:
            return 0.0
        return c["Hvap_ref"] * ((Tc - T) / (Tc - c["T_ref"])) ** exponent
    if t > Tc:
        raise ValueError("latent-heat equation requires T <= Tc; no supercritical extension")
    tau = 1 - T / Tc
    if equation == "clapeyron_hvap":
        _positive(c["Pc"], "Pc")
        _positive(c["Psat"], "Psat")
        _positive(c["dZ"], "dZ")
        if t == Tc or c["Psat"] >= c["Pc"]:
            raise ValueError("Clapeyron estimate requires T < Tc and Psat < Pc")
        return R * T * c["dZ"] * logarithm(c["Pc"] / c["Psat"]) / tau
    if equation == "pitzer_hvap":
        reduced = 7.08 * tau**0.354 + 10.95 * c["omega"] * tau**0.456
    elif equation == "smk_hvap":
        reduced = _series(tau, _SMK[0], _SMK_POWERS) + (c["omega"] - 0.212) / (
            0.461 - 0.212
        ) * _series(tau, _SMK[1], _SMK_POWERS)
    elif equation == "mk_hvap":
        reduced = sum(c["omega"] ** i * _series(tau, row, _MK_POWERS) for i, row in enumerate(_MK))
    elif equation == "velasco_hvap":
        reduced = (7.2729 + 10.4962 * c["omega"] + 0.6061 * c["omega"] ** 2) * tau**0.38
    elif equation == "ppds12_hvap":
        reduced = _series(tau, [c[k] for k in "ABCDE"], (1 / 3, 2 / 3, 1, 2, 6))
    elif equation == "alibakhshi_hvap":
        # N_A is the exact SI Avogadro constant. This empirical model does
        # not tend to zero at Tc; its fitted range must be declared separately.
        return (
            (4.5 * pi * 6.02214076e23) ** (1 / 3) * 4.2e-7 * (Tc - 6)
            - 0.5 * R * T * logarithm(T)
            + c["C"] * T
        )
    else:
        raise ValueError(f"unknown latent-heat method {equation!r}")
    return R * Tc * reduced


def latent_heat_derivatives(equation, temperature, coefficients, *, order=1):
    """Value and analytic temperature derivatives at fixed declared parameters."""
    if equation == "phase_change_point" and order > 0:
        raise ValueError("a single observation has no temperature derivative")
    if (
        equation == "watson_hvap"
        and order > 0
        and temperature == coefficients["Tc"]
        and coefficients["Hvap_ref"] != 0
        and coefficients.get("exponent", 0.38) <= order
    ):
        raise ValueError("requested Watson derivative does not exist at the critical branch")
    try:
        return derivative_values(
            lambda T: latent_heat_value(equation, T, coefficients), temperature, order=order
        )
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("latent heat or derivative is outside the finite formula domain") from exc


def boiling_vaporization_enthalpy(
    *,
    method,
    boiling_temperature_K,
    critical_temperature_K,
    critical_pressure_Pa,
    vetere_factor=1.0,
):
    """Estimate Hvap only at the stated normal boiling temperature, in J/mol.

    This is an anchor estimate, not a constant enthalpy curve. An explicit
    Watson model is needed to transport the estimate to another temperature.
    """
    Tb, Tc, Pc = boiling_temperature_K, critical_temperature_K, critical_pressure_Pa
    for name, value in (("Tb", Tb), ("Tc", Tc), ("Pc", Pc), ("vetere_factor", vetere_factor)):
        _positive(value, name)
    if Tb >= Tc:
        raise ValueError("normal boiling temperature must be below Tc")
    tr, pbar = Tb / Tc, Pc / 1e5
    tau = 1 - tr
    try:
        if method == "Riedel":
            reduced = 1.093 * (log(pbar) - 1.013) / (0.930 - tr)
        elif method == "Chen":
            reduced = (3.978 * tr - 3.958 + 1.555 * log(pbar)) / (1.07 - tr)
        elif method == "Liu":
            reduced = (
                (Tb / 220) ** 0.0627 * tau**0.38 * log(Pc / 101325) / (tau + 0.38 * tr * log(tr))
            )
        elif method == "Vetere":
            reduced = (
                tau**0.38
                * (log(pbar) - 0.513 + 0.5066 / (pbar * tr**2))
                / (tau + vetere_factor * (1 - tau**0.38) * log(tr))
            )
        else:
            raise ValueError("method must be Riedel, Chen, Liu or Vetere")
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("boiling-point enthalpy estimate is singular") from exc
    result = R * Tb * reduced
    if not isfinite(result) or result <= 0:
        raise ValueError("boiling-point method gives no finite positive latent heat")
    return result


def watson_exponent(
    *, temperature_1_K, temperature_2_K, enthalpy_1_J_mol, enthalpy_2_J_mol, critical_temperature_K
):
    """Recover the Watson exponent from two distinct subcritical observations."""
    T1, T2, Tc = temperature_1_K, temperature_2_K, critical_temperature_K
    h1, h2 = enthalpy_1_J_mol, enthalpy_2_J_mol
    for name, value in (("T1", T1), ("T2", T2), ("Tc", Tc), ("Hvap1", h1), ("Hvap2", h2)):
        _positive(value, name)
    if T1 == T2 or max(T1, T2) >= Tc:
        raise ValueError("Watson exponent requires distinct subcritical temperatures")
    exponent = (log(h1) - log(h2)) / (log(Tc - T1) - log(Tc - T2))
    _positive(exponent, "inferred Watson exponent")
    return exponent
