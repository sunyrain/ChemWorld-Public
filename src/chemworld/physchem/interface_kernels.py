"""Surface-tension equations on the common analytic derivative arithmetic.

Inputs: T/Tc/Tb/Tm K, Pc Pa, Vc m³/mol, MW g/mol; output N/m.
Aleem alone takes rhol kg/m³, Hvap_Tb J/kg and Cpl J/(kg K).
All supplied state properties are held fixed in temperature derivatives.
Equations follow public mathematical descriptions; scientific fit constants
follow chemicals 1.5.2 (Miqueu .25, Hakim 1.656 differ from its docstrings).
No upstream algorithm or runtime is used.
"""

from math import isfinite

from chemworld.physchem.caloric_math import (
    derivative_depth as _depth,
)
from chemworld.physchem.caloric_math import (
    derivative_values,
    exponential,
    logarithm,
    scalar_value,
)
from chemworld.physchem.caloric_math import (
    nonnegative_power as _power,
)
from chemworld.physchem.viscosity_kernels import _positive

INTERFACE_METHODS = {
    "source_point": ("T_ref sigma_ref", ""),
    "sigma_IAPWS": ("", ""),
    "REFPROP_sigma": ("Tc sigma0 n0", "sigma1 n1 sigma2 n2"),
    "Somayajulu": ("Tc A B C", ""),
    "Jasper": ("a b", ""),
    "Brock_Bird": ("Tb Tc Pc", ""),
    "Pitzer_sigma": ("Tc Pc omega", ""),
    "Zuo_Stenby": ("Tc Pc omega", ""),
    "Sastri_Rao": ("Tb Tc Pc", ""),
    "Sastri_Rao_alcohol": ("Tb Tc Pc", ""),
    "Sastri_Rao_acid": ("Tb Tc Pc", ""),
    "Hakim_Steinberg_Stiel": ("Tc Pc omega", "StielPolar"),
    "Miqueu": ("Tc Vc omega", ""),
    "sigma_Gharagheizi_1": ("Tc MW omega", ""),
    "sigma_Gharagheizi_2": ("Tb Tc Pc Vc", ""),
    "Aleem": ("MW Tb rhol Hvap_Tb Cpl", ""),
    "Mersmann_Kind_sigma": ("Tm Tb Tc Pc", "n_associated"),
    "API10A32": ("Tc K_W", ""),
    "PPDS14": ("Tc a0 a1 a2", ""),
    "Watson_sigma": ("Tc a1 a2", "a3 a4 a5"),
    "ISTExpansion": ("Tc a1 a2", "a3 a4 a5"),
}
_SASTRI = {
    "Sastri_Rao": (0.158, 0.5, -1.5, 1.85, 11 / 9),
    "Sastri_Rao_acid": (0.125, 0.5, -1.5, 1.85, 11 / 9),
    "Sastri_Rao_alcohol": (2.28, 0.25, 0.175, 0, 0.8),
}


def _series(tau, terms):
    combined = {}
    for a, n in terms:
        combined[n] = combined.get(n, 0) + a
    return sum(a * _power(tau, n) for n, a in combined.items() if a != 0)


def _formula(method, T, p):
    if method == "source_point":
        from chemworld.physchem.property_points import source_point_value

        return source_point_value(T,p["T_ref"],p["sigma_ref"])
    if method == "Jasper":
        return 0.001 * (p["a"] - p["b"] * (T - 273.15))
    if method == "Aleem":
        mw = p["MW"] / 1000  # The complete Aleem equation uses kg/mol, including phi.
        phi = 1 - 0.0047 * mw + 0.0000068 * mw**2
        return (
            phi
            * mw ** (1 / 3)
            / (6 * 6.02214076e23 ** (1 / 3))
            * p["rhol"] ** (2 / 3)
            * (p["Hvap_Tb"] + p["Cpl"] * (p["Tb"] - T))
        )
    tc = 647.096 if method == "sigma_IAPWS" else p["Tc"]
    tau = 1 - T / tc
    if scalar_value(tau) < 0:
        raise ValueError("surface tension requires T <= Tc")
    if method == "sigma_IAPWS":
        return _series(tau, ((0.2358, 1.256), (-0.2358 * 0.625, 2.256)))
    if method == "REFPROP_sigma":
        return _series(tau, ((p.get(f"sigma{i}", 0), p.get(f"n{i}", 0)) for i in range(3)))
    if method == "Somayajulu":
        return 0.001 * _series(tau, ((p[k], i + 1.25) for i, k in enumerate("ABC")))
    if method == "PPDS14":
        return _series(tau, ((p["a0"], p["a1"]), (p["a0"] * p["a2"], p["a1"] + 1)))
    if method == "ISTExpansion":
        return _series(tau, ((p.get(f"a{i}", 0), i) for i in range(1, 6)))
    if method == "Watson_sigma":
        tr = T / tc
        if scalar_value(tau) == 0:
            a2, a3, a4, a5 = (p.get(f"a{i}", 0) for i in range(2, 6))
            b = a2 + a3 + a4 + a5
            # tau**b * exp(log(tau) * sum(d_j*tau**j)). The first
            # nonzero log term has no finite derivative at order b+j.
            for j, d in enumerate((-a3 - 2 * a4 - 3 * a5, a4 + 3 * a5, -a5), 1):
                if d != 0 and b + j <= _depth(T):
                    raise ValueError(
                        "requested critical derivative contains a logarithmic singularity"
                    )
            return exponential(p["a1"]) * _power(tau, b)
        exponent = sum(p.get(f"a{i + 2}", 0) * tr**i for i in range(4))
        return exponential(p["a1"] + logarithm(tau) * exponent)
    if method == "sigma_Gharagheizi_1":
        a = tc - T - p["omega"]
        # A vanishes below Tc for positive omega. This is a formula domain,
        # not a declaration that a real fluid has lost its interface.
        if p["omega"] < 0 or scalar_value(a) < 0:
            raise ValueError("Gharagheizi_1 requires omega >= 0 and T <= Tc-omega")
        if p["omega"] == 0:
            return 0.0
        return 8.948226e-4 * p["omega"] ** 0.25 / p["MW"] ** 0.75 * _power(a, 1.25)
    if method == "Miqueu":
        return (
            1.380649e-23
            * tc
            * (6.02214076e23 / p["Vc"]) ** (2 / 3)
            * (4.35 + 4.14 * p["omega"])
            * _series(tau, ((1, 1.26), (0.19, 1.76), (-0.25, 2.26)))
        )
    pc = p["Pc"] / 1e5 if "Pc" in p else None
    if method == "API10A32":
        # API coefficient gives mN/m (the reference example is 29.577 mN/m).
        return 0.6737 / p["K_W"] * _power(tau, 1.232)
    if method == "Brock_Bird":
        tb = p["Tb"] / tc
        q = 0.1196 * (1 + tb * logarithm(pc / 1.01325) / (1 - tb)) - 0.279
        return 0.001 * pc ** (2 / 3) * tc ** (1 / 3) * q * _power(tau, 11 / 9)
    if method == "Pitzer_sigma":
        w = p["omega"]
        base = _positive((3.75 + 0.91 * w) / (0.291 - 0.08 * w), "Pitzer ratio")
        return (
            0.001
            * pc ** (2 / 3)
            * tc ** (1 / 3)
            * (1.86 + 1.18 * w)
            / 19.05
            * base ** (2 / 3)
            * _power(tau, 11 / 9)
        )
    if method == "Zuo_Stenby":
        reduced = []
        for a, n, t_ref, p_ref in ((40.520, 1.287, 190.56, 45.99), (52.095, 1.21548, 568.7, 24.9)):
            reduced.append(
                logarithm(1 + a * _power(tau, n) / (t_ref ** (1 / 3) * p_ref ** (2 / 3)))
            )
        value = reduced[0] + (p["omega"] - 0.012) / (0.4 - 0.012) * (reduced[1] - reduced[0])
        return 0.001 * tc ** (1 / 3) * pc ** (2 / 3) * (exponential(value) - 1)
    if method in _SASTRI:
        k, x, y, z, m = _SASTRI[method]
        return 0.001 * k * pc**x * p["Tb"] ** y * tc**z * _power(tau, m) / (1 - p["Tb"] / tc) ** m
    if method == "Hakim_Steinberg_Stiel":
        w, c = p["omega"], p.get("StielPolar", 0)
        q = 0.1574 + 0.359 * w - 1.769 * c - 13.69 * c * c - 0.51 * w * w + 1.298 * w * c
        m = 1.21 + 0.5385 * w - 14.61 * c - 32.07 * c * c - 1.656 * w * w + 22.03 * w * c
        return 0.001 * (p["Pc"] / 101325) ** (2 / 3) * tc ** (1 / 3) * q * _power(tau / 0.4, m)
    if method == "sigma_Gharagheizi_2":
        tb = p["Tb"] / tc
        return (
            1e-4
            * pc ** (2 / 3)
            * tc ** (1 / 3)
            * _power(tau, 11 / 9)
            * (7.728729 * tb + 2.476318 * (tb**3 + 1000 * p["Vc"]))
        )
    if method == "Mersmann_Kind_sigma":
        scale = ((p["Tb"] - p["Tm"]) / p["Tm"]) ** (1 / 3)
        return (
            scale
            * _series(tau, ((6.25, 1), (31.3, 4 / 3)))
            * (1.380649e-23 * tc) ** (1 / 3)
            * p["Tm"]
            / tc
            * p["Pc"] ** (2 / 3)
            / p.get("n_associated", 1) ** (1 / 3)
        )
    raise ValueError(f"unknown interface method {method!r}")


def interface_derivatives(method, T, parameters, *, order=1, above_critical="raise"):
    """Value and partial T derivatives through order 3, N/(m K**order).

    Critical derivatives are liquid-side limits. The optional zero extension
    above Tc is a computational convention, not a physical phase assertion;
    it does not override a subcritical fit's negative/undefined result.
    """
    if method not in INTERFACE_METHODS:
        raise ValueError(f"unknown interface method {method!r}")
    required, optional = INTERFACE_METHODS[method]
    if method == "source_point" and order > 0:
        raise ValueError("a single interface observation has no temperature derivative")
    if set(required.split()) - parameters.keys() or parameters.keys() - set(
        (required + " " + optional).split()
    ):
        raise ValueError(f"{method} requires {required}; optional {optional}")
    _positive(T, "temperature")
    for key, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key in {
            "Tc",
            "T_ref",
            "Tb",
            "Tm",
            "Pc",
            "MW",
            "Vc",
            "rhol",
            "Hvap_Tb",
            "Cpl",
            "n_associated",
            "K_W",
        }:
            _positive(value, key)
    p = parameters
    if "Tb" in p and "Tc" in p and p["Tb"] >= p["Tc"]:
        raise ValueError("surface-tension method requires Tb < Tc")
    if "Tm" in p and p["Tm"] >= p["Tb"]:
        raise ValueError("Mersmann-Kind requires Tm < Tb")
    if above_critical not in {"raise", "zero"}:
        raise ValueError("above_critical must be raise or zero")
    tc = 647.096 if method == "sigma_IAPWS" else p.get("Tc")
    if above_critical == "zero" and tc is None:
        raise ValueError("this formula has no critical-temperature extension")
    try:
        values = derivative_values(
            (lambda _: 0.0)
            if tc is not None and tc < T and above_critical == "zero"
            else lambda t: _formula(method, t, p),
            T,
            order=order,
        )
    except (OverflowError, ZeroDivisionError, TypeError) as exc:
        raise ValueError("surface tension or derivative is outside the real finite domain") from exc
    if values[0] < 0:
        raise ValueError("negative surface-tension prediction; no implicit clipping")
    return values


def interface_value(method, T, *, above_critical="raise", **parameters):
    return interface_derivatives(method, T, parameters, order=0, above_critical=above_critical)[0]
