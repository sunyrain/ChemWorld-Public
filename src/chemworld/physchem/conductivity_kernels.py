"""Native thermal-conductivity equations from published mathematical APIs.

T/Tc/Tb/Tm are K, Pc Pa, MW g/mol, Cvm J/(mol K), mu Pa s, Vc m³/mol,
Hfus J/mol; output W/(m K). Temperature derivatives hold supplied Cvm/mu
and all other parameters fixed: this module does not infer a thermodynamic
path or an EOS. Source/application validity belongs to PropertyCorrelation.
"""

from math import isfinite

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    derivative_values,
    logarithm,
    scalar_value,
)
from chemworld.physchem.viscosity_kernels import _positive

R = 8.31446261815324
CONDUCTIVITY_METHODS = {
    "Sheffy_Johnson": ("MW Tm", "", "liquid"),
    "Sato_Riedel": ("MW Tb Tc", "", "liquid"),
    "Lakshmi_Prasad": ("MW", "", "liquid"),
    "Bahadori_liquid": ("MW", "", "liquid"),
    "Gharagheizi_liquid": ("MW Tb Pc omega", "", "liquid"),
    "Nicola": ("MW Tc Pc omega", "", "liquid"),
    "Nicola_original": ("MW Tc omega Hfus", "", "liquid"),
    "kl_Mersmann_Kind": ("MW Tc Vc na", "", "liquid"),
    "PPDS3": ("Tc a1 a2 a3", "", "gas"),
    "PPDS8": ("Tc a0 a1 a2 a3", "", "liquid"),
    "Chemsep_16": ("A B C D E", "", "liquid"),
    "Eucken": ("MW Cvm mu", "", "gas"),
    "Eucken_modified": ("MW Cvm mu", "", "gas"),
    "DIPPR9B_monoatomic": ("MW Cvm mu", "", "gas"),
    "DIPPR9B_linear": ("MW Cvm mu Tc", "", "gas"),
    "DIPPR9B_nonlinear": ("MW Cvm mu", "", "gas"),
    "Chung": ("MW Tc omega Cvm mu", "", "gas"),
    "Bahadori_gas": ("MW", "", "gas"),
    "k_IAPWS": ("rho", "", "gas liquid"),
    "k_air_lemmon": ("rho", "", "gas"),
    "Eli_Hanley": ("MW Tc Vc Zc omega Cvm", "", "gas"),
    "Chung_dense": ("MW Tc Vc omega Cvm Vm mu dipole", "association", "gas"),
    "Gharagheizi_gas": ("MW Tb Pc omega", "", "gas"),
    "Stiel_Thodos_dense": ("MW Tc Pc Vc Zc Vm kg", "", "gas"),
    "Eli_Hanley_dense": ("MW Tc Vc Zc omega Cvm Vm", "", "gas"),
    "Ely_Hanley_dense_translational": ("MW Tc Vc Zc omega Cvm Vm", "", "gas"),
}

# Methane reference-viscosity polynomial, Ely-Hanley (1983), n=1..9.
# Scientific coefficients only, independently evaluated in powers (n-4)/3.
_ELY_METHANE_VISCOSITY = (
    2907741.307,
    -3312874.033,
    1608101.838,
    -433190.4871,
    70624.8133,
    -7116.62075,
    432.51744,
    -14.4591121,
    0.2037119479,
)

# Bahadori and Mokhatab (2008), scientific polynomial coefficient tables.
# Only numeric literals were extracted from the reference distribution;
# no reference algorithms or executable interpolation coefficients are used.
_BAHADORI = {
    "liquid": (
        (-0.0648326, 0.002715015, -1.0858e-5, 9.853917e-9),
        (0.01565612, -0.000155833, 5.051114e-7, -4.6803e-10),
        (-0.000180304, 1.758693e-6, -5.55224e-9, 5.201365e-12),
        (5.880443e-7, -5.65898e-9, 1.764384e-11, -1.65944e-14),
    ),
    "gas": (
        (0.43931323468, -0.0388001122207, 0.000928616040136, -6.57828995724e-6),
        (-0.0029624238519, 0.00026795614582, -6.40171884139e-6, 4.48579040207e-8),
        (7.54249790107e-6, -6.46636219509e-7, 1.5124510261e-8, -1.0376480449e-10),
        (-6.0988433456e-9, 5.20752132076e-10, -1.19425545729e-11, 8.0136464085e-14),
    ),
}


def _tau(T, tc):
    tau = 1 - T / tc
    if scalar_value(tau) < 0:
        raise ValueError("liquid conductivity requires T <= Tc")
    if scalar_value(tau) == 0 and isinstance(T, FirstDerivative):
        raise ValueError("fractional-power critical derivative requires a side")
    return tau


def _chung_psi(T, tc, omega, cvm):
    """Shared dilute/dense Chung internal-degree-of-freedom correction."""
    a = cvm / R - 1.5
    b = 0.7862 - 0.7109 * omega + 1.3168 * omega**2
    z = 2 + 10.5 * (T / tc) ** 2
    denominator = _positive(0.6366 + b * z + 1.061 * a * b, "Chung denominator")
    return 1 + a * (0.215 + 0.28288 * a - 1.061 * b + 0.26665 * z) / denominator


def _ely_methane_viscosity(T):
    """Methane dilute viscosity in Pa s from the shared scientific table."""
    result = 1e-7 * sum(c * T ** ((n - 4) / 3) for n, c in enumerate(_ELY_METHANE_VISCOSITY, 1))
    return _positive(result, "methane reference viscosity")


def _ely_dilute_parts(T, p):
    """Translational and internal dilute contributions, both in W/(m K)."""
    tr = T / p["Tc"]
    if scalar_value(tr) == 2 and isinstance(T, FirstDerivative):
        raise ValueError("Ely-Hanley shape derivative at Tr=2 requires a side")
    if scalar_value(tr) > 2:
        tr = 2.0
    theta = 1 + (p["omega"] - 0.011) * (0.56553 - 0.86276 * logarithm(tr) - 0.69852 / tr)
    psi = (1 + (p["omega"] - 0.011) * (0.38560 - 1.1617 * logarithm(tr))) * 0.288 / p["Zc"]
    f = _positive(p["Tc"] / 190.4 * theta, "Ely-Hanley temperature shape")
    h = _positive(p["Vc"] / 9.92e-5 * psi, "Ely-Hanley volume shape")
    t0 = T / f  # Actual T continues to change above the reduced shape cutoff.
    eta0 = _ely_methane_viscosity(t0)
    factor = (16.04 / p["MW"]) ** 0.5 * f**0.5 / h ** (2 / 3)
    eta = eta0 * factor * p["MW"] / 16.04
    return 1944 * eta0 * factor, eta / (p["MW"] / 1000) * 1.32 * (p["Cvm"] - 1.5 * R)


def _formula(method, T, p):
    if method == "Eli_Hanley":
        return sum(_ely_dilute_parts(T, p))
    if method in {"Eli_Hanley_dense", "Ely_Hanley_dense_translational"}:
        from chemworld.physchem.dense_conductivity import ely_hanley_dense_conductivity

        scope = "dilute" if method == "Eli_Hanley_dense" else "translational"
        return ely_hanley_dense_conductivity(T, correction_scope=scope, **p)
    if method == "Stiel_Thodos_dense":
        from chemworld.physchem.dense_conductivity import stiel_thodos_dense_conductivity

        return stiel_thodos_dense_conductivity(T, **p)
    if method == "Chung_dense":
        from chemworld.physchem.dense_conductivity import chung_dense_conductivity

        return chung_dense_conductivity(T, **p)
    if method == "k_IAPWS":
        from chemworld.physchem.water_transport import water_conductivity_background

        return water_conductivity_background(T, p["rho"])
    if method == "k_air_lemmon":
        from chemworld.physchem.air_transport import air_conductivity_background

        return air_conductivity_background(T, p["rho"])
    mw = p.get("MW")
    if method == "Gharagheizi_gas":
        # Public Gharagheizi expression, completed by OpenPNM gas_pure_gismr.
        # u/d regroup the mathematical equation; no upstream execution path.
        u = 2 * p["omega"] + 3.2825
        d = T * (1 - u / p["Tb"])
        if scalar_value(d + u) == 0:
            raise ValueError("Gharagheizi gas B denominator is singular")
        b = d + 2 - u / (d + u)
        pc_bar = p["Pc"] / 1e5
        a = (d + u) * (pc_bar + 1.9876 * (b + 2 * p["omega"]) + 6.5243) ** 2
        a /= mw * pc_bar * T
        return 7.9505e-4 + 3.989e-5 * (T + a) - 5.419e-5 * mw
    if method == "Sheffy_Johnson":
        return 1.951 * (1 - 0.00126 * (T - p["Tm"])) / p["Tm"] ** 0.216 / mw**0.3
    if method == "Sato_Riedel":
        tau = _tau(T, p["Tc"])
        if p["Tb"] >= p["Tc"]:
            raise ValueError("Sato-Riedel requires Tb < Tc")
        return (
            1.1053
            / mw**0.5
            * (3 + 20 * tau ** (2 / 3))
            / (3 + 20 * (1 - p["Tb"] / p["Tc"]) ** (2 / 3))
        )
    if method == "Lakshmi_Prasad":
        return 0.0655 - 0.0005 * T + (1.3855 - 0.00197 * T) / mw**0.5
    if method.startswith("Bahadori_"):
        phase = method.split("_")[1]
        x, y = (mw, T) if phase == "liquid" else (T, mw)
        return sum(
            c * x**i * y**j for i, row in enumerate(_BAHADORI[phase]) for j, c in enumerate(row)
        )
    if method == "Gharagheizi_liquid":
        b = 16.0407 * mw + 2 * p["Tb"] - 27.9074
        if b == 0:
            raise ValueError("Gharagheizi liquid B is singular")
        ratio = mw / b
        return 1e-4 * (
            10 * p["omega"]
            + 2 * p["Pc"] / 1e5
            - 2 * T
            + 4
            + 1.908 * (p["Tb"] + 1.009 / ratio**2)
            + 3.9287 * ratio**4
            + 3.8588 * ratio**8 * (1.0045 * b + 6.5152 * mw - 8.9756)
        )
    if method == "Nicola":
        # The public numerical example requires MW**(-.2999); the printed
        # (1/MW)**(-.2999) is inconsistent with it.
        return 0.5147 * (
            -0.2537 * T / p["Tc"] + 0.0017 * p["Pc"] / 1e5 + 0.1501 * p["omega"] + mw**-0.2999
        )
    if method == "Nicola_original":
        return (
            -0.5694
            - 0.1436 * T / p["Tc"]
            + 5.4893e-7 * p["Hfus"]
            + 0.0508 * p["omega"]
            + mw**-0.0622
        )
    if method == "kl_Mersmann_Kind":
        reduced = 2 / 3 * (p["na"] + 40 * _tau(T, p["Tc"]) ** 0.5)
        return (
            reduced
            * (1.380649e-23 * p["Tc"]) ** 1.5
            * 6.02214076e23 ** (7 / 6)
            / (p["Vc"] ** (2 / 3) * p["Tc"] * (mw / 1000) ** 0.5)
        )
    if method == "PPDS3":
        tr = T / p["Tc"]
        denominator = sum(p[f"a{i}"] / tr**i for i in (1, 2, 3))
        return tr**0.5 / _positive(denominator, "PPDS3 denominator")
    if method == "PPDS8":
        tau = _tau(T, p["Tc"])
        return p["a0"] * (1 + sum(p[f"a{i}"] * tau ** (i / 3) for i in (1, 2, 3)))
    if method in {"Eucken", "Eucken_modified"}:
        a, b = (1, 2.25) if method == "Eucken" else (1.32, 1.77)
        return p["mu"] / (mw / 1000) * (a * p["Cvm"] + b * R)
    if method.startswith("DIPPR9B_"):
        cv = p["Cvm"] * 1000
        if method == "DIPPR9B_monoatomic":
            term = 2.5 * cv
        elif method == "DIPPR9B_nonlinear":
            term = 1.15 * cv + 16903.36
        else:
            term = 1.30 * cv + 14644 - 2928.80 * p["Tc"] / T
        return p["mu"] / mw * term
    if method == "Chung":
        psi = _chung_psi(T, p["Tc"], p["omega"], p["Cvm"])
        return 3.75 * psi * R * p["mu"] / (mw / 1000)
    raise ValueError(f"unknown conductivity method {method!r}")


def conductivity_derivatives(method, T, parameters, *, order=1):
    """Value and partial T derivatives, holding every supplied parameter fixed."""
    if method not in CONDUCTIVITY_METHODS:
        raise ValueError(f"unknown conductivity method {method!r}")
    required, optional, _ = CONDUCTIVITY_METHODS[method]
    if set(required.split()) - parameters.keys() or parameters.keys() - set(
        (required + " " + optional).split()
    ):
        raise ValueError(f"{method} requires {required}; optional {optional}")
    _positive(T, "temperature")
    for key, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key in {"MW", "Tm", "Tb", "Tc", "Pc", "Vc", "Zc", "Vm", "na", "Cvm", "mu", "kg"}:
            _positive(value, key)
        if key in {"Hfus", "rho", "dipole", "association"}:
            _positive(value, key, zero=True)
    try:
        if method == "Chemsep_16":
            from chemworld.physchem.correlation_families import family_value_and_slope

            if isinstance(order, bool) or order not in (0, 1):
                raise ValueError("shared Chemsep16 kernel supports first derivative only")
            result = family_value_and_slope("chemsep16", T, parameters)[: order + 1]
        else:
            result = derivative_values(lambda t: _formula(method, t, parameters), T, order=order)
    except (ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{method} is singular or outside its numerical domain") from exc
    _positive(result[0], "thermal conductivity")
    return result


def conductivity_value(method, T, **parameters):
    return conductivity_derivatives(method, T, parameters, order=0)[0]
