"""Native viscosity equations from published formulae and public API documentation.

All results are Pa s; T is K, Pc Pa, MW g/mol. Coefficients of fits must
already carry the Pa s scale. Density-dependent models do not calculate an
EOS. Analytic temperature derivatives hold every supplied parameter fixed.
"""

from math import isfinite, log

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    derivative_values,
    exponential,
    scalar_value,
)
from chemworld.physchem.water_transport import water_viscosity_background

# method -> required parameters, optional parameters, applicable pure phases
VISCOSITY_METHODS = {
    "PPDS5": ("Tc a0 a1 a2", "", "gas"),
    "PPDS9": ("A B C D E", "", "liquid"),
    "mu_TDE": ("A B C D", "", "liquid"),
    "mu_Yaws": ("A B C D", "", "liquid"),
    "Viswanath_Natarajan_2": ("A B", "", "liquid"),
    "Viswanath_Natarajan_2_exponential": ("C D", "", "liquid"),
    "Viswanath_Natarajan_3": ("A B C", "", "liquid"),
    "Letsou_Stiel": ("MW Tc Pc omega", "", "liquid"),
    "Yoon_Thodos": ("Tc Pc MW", "", "gas"),
    "Stiel_Thodos": ("Tc Pc MW", "", "gas"),
    "viscosity_gas_Gharagheizi": ("Tc Pc MW", "low_temperature_limit", "gas"),
    "Lucas_gas": ("Tc Pc Zc MW", "dipole quantum_Q", "gas"),
    "mu_air_lemmon": ("rho", "", "gas"),
    "mu_IAPWS": ("rho", "", "gas liquid"),
    "Twu_1985": ("Tb rho_60F", "rho_at_T", "liquid"),
    "Przedziecki_Sridhar": ("Tm Tc Pc Vc Vf omega MW", "Vscale", "liquid"),
}


def _positive(value, label, *, zero=False):
    v = scalar_value(value)
    if (
        isinstance(v, bool)
        or not isinstance(v, int | float)
        or not isfinite(v)
        or (v < 0 if zero else v <= 0)
    ):
        raise ValueError(f"{label} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _absolute(value):
    v = scalar_value(value)
    if v == 0 and isinstance(value, FirstDerivative):
        raise ValueError("viscosity derivative is undefined at an absolute-value branch")
    return value if v >= 0 else -value


def _formula(method, T, p):
    if method == "Przedziecki_Sridhar":
        from chemworld.physchem.dense_viscosity import przedziecki_sridhar_temperature_viscosity

        return przedziecki_sridhar_temperature_viscosity(
            T,
            melting_volume_m3_mol=p["Vf"],
            melting_temperature_K=p["Tm"],
            critical_temperature_K=p["Tc"],
            critical_pressure_Pa=p["Pc"],
            critical_volume_m3_mol=p["Vc"],
            acentric_factor=p["omega"],
            molecular_weight_g_mol=p["MW"],
            volume_scale_m3_mol=p.get("Vscale"),
        )
    if method == "Twu_1985":
        from chemworld.physchem.petroleum_viscosity import twu_1985_viscosity

        return twu_1985_viscosity(T, p["Tb"], p["rho_60F"], density_at_T_kg_m3=p.get("rho_at_T"))
    if method == "mu_IAPWS":
        return water_viscosity_background(T, p["rho"])
    if method == "PPDS5":
        tr = T / p["Tc"]
        denominator = 1 + p["a1"] * tr ** p["a2"] * (tr - 1)
        return p["a0"] * tr / _positive(denominator, "PPDS5 denominator") ** (1 / 6)
    if method == "PPDS9":
        if scalar_value(T) == p["D"]:
            raise ValueError("PPDS9 has a pole at T=D")
        x = (p["C"] - T) / (T - p["D"])
        if scalar_value(x) == 0:
            if isinstance(T, FirstDerivative):
                raise ValueError("PPDS9 derivatives require a side away from T=C")
            return p["E"]
        root = x ** (1 / 3) if scalar_value(x) > 0 else -((-x) ** (1 / 3))
        return p["E"] * exponential(p["A"] * root + p["B"] * x * root)
    if method == "mu_TDE":
        return exponential(p["A"] + p["B"] / T + p["C"] / T**2 + p["D"] / T**3)
    if method == "mu_Yaws":
        return exponential(log(10) * (p["A"] + p["B"] / T + p["C"] * T + p["D"] * T**2))
    if method == "Viswanath_Natarajan_2":
        return exponential(p["A"] + p["B"] / T)
    if method == "Viswanath_Natarajan_2_exponential":
        return p["C"] * T ** p["D"]
    if method == "Viswanath_Natarajan_3":
        # Public numerical example and coefficient tables use C-T, not the
        # erroneous T+C printed in the chemicals 1.5.2 equation documentation.
        if scalar_value(T) == p["C"]:
            raise ValueError("Viswanath-Natarajan has a pole at T=C")
        return exponential(log(10) * (p["A"] + p["B"] / (p["C"] - T)))
    if method == "mu_air_lemmon":
        from chemworld.physchem.air_transport import air_viscosity_background

        return air_viscosity_background(T, p["rho"])
    tr = T / p["Tc"]
    if method in {"Letsou_Stiel", "Yoon_Thodos"}:
        # Independent published/source coefficient conventions: Yoon's
        # numerical coefficient has one more decimal than its printed docs.
        coefficient = 2173.424 if method == "Letsou_Stiel" else 2173.4241
        xi = coefficient * p["Tc"] ** (1 / 6) / (p["MW"] ** 0.5 * p["Pc"] ** (2 / 3))
        if method == "Letsou_Stiel":
            return (
                (
                    (1.5174 - 2.135 * tr + 0.75 * tr**2)
                    + p["omega"] * (4.2552 - 7.674 * tr + 3.4 * tr**2)
                )
                * 1e-5
                / xi
            )
        return (
            (
                46.10 * tr**0.618
                - 20.40 * exponential(-0.449 * tr)
                + 19.40 * exponential(-4.058 * tr)
                + 1
            )
            * 1e-8
            / xi
        )
    if method == "Stiel_Thodos":
        xi = p["Tc"] ** (1 / 6) / (p["MW"] ** 0.5 * (p["Pc"] / 101325) ** (2 / 3))
        if scalar_value(tr) == 1.5 and isinstance(T, FirstDerivative):
            raise ValueError("Stiel-Thodos derivative at the 1.5 Tc branch is undefined")
        # These coefficient magnitudes give centipoise, despite the public
        # documentation's statement "poise". See its numerical CCl4 example.
        return (
            (17.78e-5 * (4.58 * tr - 1.67) ** 0.625 if scalar_value(tr) > 1.5 else 34e-5 * tr**0.94)
            / xi
            * 0.001
        )
    if method == "viscosity_gas_Gharagheizi":
        if (
            p.get("low_temperature_limit", 0)
            and scalar_value(tr) == 0.2
            and isinstance(T, FirstDerivative)
        ):
            raise ValueError("Gharagheizi derivative at limiting branch requires a side")
        if scalar_value(tr) < 0.2:
            if not p.get("low_temperature_limit", 0):
                raise ValueError(
                    "Gharagheizi gas requires T/Tc >= 0.2; no implicit temperature clamp"
                )
            # Explicit reference compatibility. A constant jet has zero slope.
            T = T * 0 + 0.2 * p["Tc"]
            tr = T / p["Tc"]
        mw, pc = p["MW"], p["Pc"]
        return 1e-7 * _absolute(
            1e-5 * pc * tr
            + (0.091 - 0.477 / mw) * T
            + mw * (1e-5 * pc - 8 * mw**2 / T**2) * (10.7639 / p["Tc"] - 4.1929 / T)
        )
    if method == "Lucas_gas":
        pc_bar = p["Pc"] / 1e5
        xi = 0.176 * (p["Tc"] / (p["MW"] ** 3 * pc_bar**4)) ** (1 / 6)
        reduced_dipole = 52.46 * p.get("dipole", 0) ** 2 * pc_bar / p["Tc"] ** 2
        polar = 1
        if reduced_dipole >= 0.022:
            _positive(0.292 - p["Zc"], "Lucas polar 0.292-Zc", zero=True)
            polar += (
                30.55
                * (0.292 - p["Zc"]) ** 1.72
                * (_absolute(0.96 + 0.1 * (tr - 0.7)) if reduced_dipole >= 0.075 else 1)
            )
        quantum = 1
        if p.get("quantum_Q", 0):
            offset = tr - 12
            if scalar_value(offset) == 0:
                if isinstance(T, FirstDerivative):
                    raise ValueError("Lucas quantum derivative at Tr=12 requires a side")
                quantum = 1.22 * p["quantum_Q"] ** 0.15
            else:
                sign = 1 if scalar_value(offset) > 0 else -1
                quantum = (
                    1.22
                    * p["quantum_Q"] ** 0.15
                    * (1 + 0.00385 * (offset**2) ** (1 / p["MW"]) * sign)
                )
        return (
            (
                0.807 * tr**0.618
                - 0.357 * exponential(-0.449 * tr)
                + 0.340 * exponential(-4.058 * tr)
                + 0.018
            )
            * polar
            * quantum
            / xi
            * 1e-7
        )
    raise ValueError(f"unknown viscosity method {method!r}")


def viscosity_derivatives(method, T, parameters, *, order=1):
    """Value and analytic T derivatives; fixed supplied density/other parameters.

    These are mathematical equations, not automatic validity estimates. The
    application/source interval belongs to the original PropertyCorrelation.
    Lucas quantum_Q is explicit: He=1.38, H2=.76, D2=.52; zero disables it.
    Lemmon rho is mol/m³ (not kg/m³); its dilute coefficient is 0.0266958,
    correcting the 0.9266958 typographic error in the public formula.
    """
    if method not in VISCOSITY_METHODS:
        raise ValueError(f"unknown viscosity method {method!r}")
    required, optional, _ = VISCOSITY_METHODS[method]
    if set(required.split()) - parameters.keys() or parameters.keys() - set(
        (required + " " + optional).split()
    ):
        raise ValueError(f"{method} requires {required}; optional: {optional}")
    _positive(T, "temperature")
    for key, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key == "low_temperature_limit" and value not in (0, 1):
            raise ValueError("low_temperature_limit must be 0 or 1")
        if key in {"Tc", "Pc", "MW", "Zc", "Tb", "rho_60F", "rho_at_T", "Tm", "Vc", "Vf", "Vscale"}:
            _positive(value, key)
        elif key in {"rho", "dipole", "quantum_Q"}:
            _positive(value, key, zero=True)
    try:
        values = derivative_values(lambda t: _formula(method, t, parameters), T, order=order)
    except (ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{method} is singular or outside its numerical domain") from exc
    _positive(values[0], "viscosity")
    return values


def viscosity_value(method, T, **parameters):
    """Evaluate the named native equation without requesting a derivative."""
    return viscosity_derivatives(method, T, parameters, order=0)[0]


def viscosity_yaws_parameter_jacobian(temperatures_K, *, A, B, C, D):
    """Analytic rows d(mu)/d(A,B,C,D), with unmodified Pa s fit coefficients."""
    rows = []
    for T in temperatures_K:
        value = viscosity_value("mu_Yaws", T, A=A, B=B, C=C, D=D)
        rows.append(tuple(value * log(10) * factor for factor in (1, 1 / T, T, T * T)))
    return tuple(rows)
