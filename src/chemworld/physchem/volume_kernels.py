"""Pure and compressed-liquid equations from published mathematical descriptions.

T/Tc/Tb/Tt/Tm K, Pc/P Pa, Vc/Vml m³/mol, MW g/mol, dipole debye,
rho kg/m³. Raw output follows the documented scientific API: most methods
return molar volume; density fits return density without MW; Rackett_fit
returns mass specific volume without MW. Explicit output conversion is also
available. No reference implementation is imported at runtime.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from types import MappingProxyType

from scipy.interpolate import CubicSpline

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    derivative_depth,
    derivative_values,
    exponential,
    logarithm,
    scalar_value,
)
from chemworld.physchem.caloric_math import (
    nonnegative_power as power,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class VolumeMethod:
    required: str
    optional: str = ""
    phase: str = "liquid"
    native_output: str = "molar_volume"
    relation: str = "saturated_liquid"

    @property
    def uses_pressure(self):
        return "P" in self.required.split()


VOLUME_METHODS = MappingProxyType(
    {
        "Rackett": VolumeMethod("Tc Pc Zc"),
        "ideal_gas": VolumeMethod("P", phase="gas", relation="ideal_gas"),
        "Tait": VolumeMethod(
            "P P_ref rho_ref B C", native_output="density", relation="compressed_liquid"
        ),
        "Tait_molar": VolumeMethod("P P_ref V_ref B C", relation="compressed_liquid"),
        "COSTALD_compressed": VolumeMethod(
            "P Tc Pc omega", "Psat Vs Vc estimate_saturation", relation="compressed_liquid"
        ),
        "Yen_Woods_saturation": VolumeMethod("Tc Vc Zc"),
        "Yamada_Gunn": VolumeMethod("Tc Pc omega"),
        "Townsend_Hales": VolumeMethod("Tc Vc omega"),
        "Bhirud_normal": VolumeMethod("Tc Pc omega"),
        "COSTALD": VolumeMethod("Tc Vc omega"),
        "SNM0": VolumeMethod("Tc Vc omega", "delta_SRK"),
        "Campbell_Thodos": VolumeMethod("Tb Tc Pc MW", "dipole has_hydroxyl hydroxyl_polar_term"),
        "Goodman": VolumeMethod("Tt Vml", phase="solid", relation="solid_below_triple"),
        "PPDS17": VolumeMethod("Tc a0 a1 a2", native_output="density"),
        "TDE_VDNS_rho": VolumeMethod("Tc rhoc a1 a2 a3 a4", native_output="density"),
        "Rackett_fit": VolumeMethod("Tc rhoc b n", native_output="specific_volume"),
        "CRC_inorganic": VolumeMethod(
            "rho0 k Tm", native_output="density", relation="molten_linear_expansion"
        ),
        "volume_VDI_PPDS": VolumeMethod("Tc rhoc a b c d", native_output="density"),
    }
)

# Scientific table, not upstream interpolation code or generated spline state.
# Bhirud (1978), DOI 10.1002/aic.690240630; public numerical arrays distributed
# by chemicals 1.5.2, MIT. Interpolation is a native SciPy not-a-knot cubic.
BHIRUD_TABLE = (
    (0.980, 0.982, 0.984, 0.986, 0.988, 0.990, 0.992, 0.994, 0.996, 0.998, 0.999, 1.0),
    (
        -1.6198,
        -1.604,
        -1.59,
        -1.578,
        -1.564,
        -1.548,
        -1.533,
        -1.515,
        -1.489,
        -1.454,
        -1.425,
        -1.243,
    ),
    (
        -0.4626,
        -0.459,
        -0.451,
        -0.441,
        -0.428,
        -0.412,
        -0.392,
        -0.367,
        -0.337,
        -0.302,
        -0.283,
        -0.2629,
    ),
)


@lru_cache(maxsize=1)
def _bhirud_splines():
    return tuple(CubicSpline(BHIRUD_TABLE[0], ys, extrapolate=False) for ys in BHIRUD_TABLE[1:])


def _polynomial(x, coefficients):
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def _spline_value(spline, x):
    # Evaluate the local cubic on the common derivative arithmetic, including
    # Tc/parameter sensitivities. SciPy owns spline construction, not state.
    import bisect

    i = min(len(spline.x) - 2, max(0, bisect.bisect_right(spline.x, scalar_value(x)) - 1))
    if derivative_depth(x) >= 3 and scalar_value(x) in spline.x[1:-1]:
        raise ValueError("third derivative at an interior spline knot is not two-sided")
    return _polynomial(x - spline.x[i], tuple(float(v) for v in spline.c[:, i][::-1]))


def _positive(value, name):
    if not isfinite(scalar_value(value)) or scalar_value(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _formula(method, T, p):
    if method == "ideal_gas":
        return R * T / p["P"]
    if method in {"Tait", "Tait_molar"}:
        factor = _tait_factor(p["P"], p["P_ref"], p["B"], p["C"])
        return p["rho_ref"] / factor if method == "Tait" else p["V_ref"] * factor
    if method == "COSTALD_compressed":
        tau = 1 - T / p["Tc"]
        if scalar_value(tau) <= 0:
            raise ValueError("compressed COSTALD requires T < Tc")
        omega = p["omega"]
        e = exponential(4.79594 + 0.250047 * omega + 1.14188 * omega**2)
        b = p["Pc"] * (
            -1 - 9.070217 * power(tau, 1 / 3) + 62.45326 * power(tau, 2 / 3)
            - 135.1102 * tau + e * power(tau, 4 / 3)
        )
        c = 0.0861488 + 0.0344483 * omega
        psat = p["Psat"] if "Psat" in p else _riedel_pressure(T, p)
        vs = p["Vs"] if "Vs" in p else _formula("COSTALD", T, p)
        return _positive(vs, "saturation volume") * _tait_factor(p["P"], psat, b, c)
    if method == "Goodman":
        if scalar_value(T) > scalar_value(p["Tt"]):
            raise ValueError("Goodman solid requires T <= Tt")
        return p["Vml"] / (1.28 - 0.16 * T / p["Tt"])
    if method == "CRC_inorganic":
        if scalar_value(T) < scalar_value(p["Tm"]):
            raise ValueError("CRC molten liquid requires T >= Tm")
        return p["rho0"] - p["k"] * (T - p["Tm"])
    tr = T / p["Tc"]
    tau = 1 - tr
    if scalar_value(tau) < 0:
        raise ValueError("saturated liquid volume requires T <= Tc")
    if method in {"Rackett", "Yamada_Gunn", "Campbell_Thodos"}:
        if method == "Rackett":
            z = p["Zc"]
        elif method == "Yamada_Gunn":
            z = 0.29056 - 0.08775 * p["omega"]
        else:
            pc_atm = p["Pc"] / 101325
            tbr = p["Tb"] / p["Tc"]
            s = tbr * logarithm(pc_atm) / (1 - tbr)
            lam = pc_atm ** (1 / 3) / (p["MW"] ** 0.5 * p["Tc"] ** (5 / 6))
            alpha = 0.3883 - 0.0179 * s
            beta = 0.00318 * s - 0.0211 + 0.625 * lam**1.35
            theta = pc_atm * p.get("dipole", 0) ** 2 / p["Tc"] ** 2
            if p.get("has_hydroxyl", 0):
                alpha = (0.690 * tbr - 0.3342 + 5.79e-10 / tbr**32.75) * pc_atm**0.145
                beta += 5.90 * power(theta, 0.835)
                # The 1.5.2 public API retains the polar beta contribution;
                # its rendered hydroxyl equation omits it. Explicit choices
                # use the documented terms, never a fitted discrepancy offset.
                if p.get("hydroxyl_polar_term", 1):
                    beta += 9.74e6 * power(theta, 3.38)
            else:
                alpha -= 130540 * power(theta, 2.41)
                beta += 9.74e6 * power(theta, 3.38)
            z = alpha + beta * tau
        _positive(z, "Rackett base")
        return R * p["Tc"] / p["Pc"] * exponential((1 + power(tau, 2 / 7)) * logarithm(z))
    if method == "Yen_Woods_saturation":
        z = p["Zc"]
        if isinstance(z, FirstDerivative) and scalar_value(z) == 0.26:
            raise ValueError("Yen-Woods Zc branch has no two-sided parameter derivative")
        a = _polynomial(z, (17.4425, -214.578, 989.625, -1522.06))
        b = _polynomial(
            z,
            (-3.28257, 13.6377, 107.4844, -384.211)
            if scalar_value(z) <= 0.26
            else (60.2091, -402.063, 501, 641),
        )
        return p["Vc"] / (
            1 + a * power(tau, 1 / 3) + b * power(tau, 2 / 3) + (0.93 - b) * power(tau, 4 / 3)
        )
    if method == "Townsend_Hales":
        return p["Vc"] / (1 + 0.85 * tau + (1.692 + 0.986 * p["omega"]) * power(tau, 1 / 3))
    if method == "Bhirud_normal":
        if scalar_value(tr) == 0.98 and derivative_depth(tr):
            raise ValueError("Bhirud polynomial/table join has no two-sided derivative")
        if scalar_value(tr) <= 0.98:
            u0 = _polynomial(tr, (1.39644, -24.076, 102.615, -255.719, 355.805, -256.671, 75.1088))
            u1 = _polynomial(
                tr, (13.4412, -135.7437, 533.380, -1091.453, 1231.43, -728.227, 176.737)
            )
        else:
            u0, u1 = (_spline_value(spline, tr) for spline in _bhirud_splines())
        return R * T / p["Pc"] * exponential(u0 + p["omega"] * u1)
    if method == "COSTALD":
        v0 = (
            1
            - 1.52816 * power(tau, 1 / 3)
            + 1.43907 * power(tau, 2 / 3)
            - 0.81446 * tau
            + 0.190454 * power(tau, 4 / 3)
        )
        vd = _polynomial(tr, (-0.296123, 0.386914, -0.0427258, -0.0480645)) / (tr - 1.00001)
        return p["Vc"] * v0 * (1 - p["omega"] * vd)
    if method == "SNM0":
        w = p["omega"]
        alpha = (1 + (0.480 + 1.574 * w - 0.176 * w * w) * (1 - tr**0.5)) ** 2
        u = 1 - tr / _positive(alpha, "SRK alpha")
        denominator = (
            1
            + 1.169 * power(u, 1 / 3)
            + 1.818 * power(u, 2 / 3)
            - 2.658 * u
            + 2.161 * power(u, 4 / 3)
        )
        if p.get("delta_SRK", 0) != 0:
            denominator *= 1 + p["delta_SRK"] * power(alpha - 1, 1 / 3)
        return p["Vc"] / _positive(denominator, "SNM0 denominator")
    if method == "PPDS17":
        base = _positive(p["a1"] + p["a2"] * tau, "PPDS17 base")
        return 1 / (p["a0"] * exponential((1 + power(tau, 2 / 7)) * logarithm(base)))
    if method == "Rackett_fit":
        return exponential(power(tau, p["n"]) * logarithm(p["b"])) / p["rhoc"]
    if method == "TDE_VDNS_rho":
        return p["rhoc"] + sum(
            p[k] * power(tau, n)
            for k, n in zip(("a1", "a2", "a3", "a4"), (0.35, 1, 2, 3), strict=True)
            if p[k] != 0
        )
    if method == "volume_VDI_PPDS":
        return p["rhoc"] + sum(
            p[k] * power(tau, n)
            for k, n in zip("abcd", (0.35, 2 / 3, 1, 4 / 3), strict=True)
            if p[k] != 0
        )
    raise ValueError(f"unknown volume method {method!r}")


def _tait_factor(P, P_ref, B, C):
    # No upstream saturation clipping: a supplied pressure is never replaced.
    if scalar_value(P) < scalar_value(P_ref):
        raise ValueError("compressed liquid requires P >= reference/saturation pressure")
    _positive(C, "Tait C")
    numerator = _positive(B + P, "Tait B + P")
    denominator = _positive(B + P_ref, "Tait B + reference pressure")
    return _positive(1 - C * logarithm(numerator / denominator), "Tait volume factor")


def _riedel_pressure(T, p):
    """Generalized Riedel, API 6A3.4; explicit model selection at the boundary."""
    tr = T / p["Tc"]
    log10tr = logarithm(tr) / logarithm(10.0)
    alpha = 35 - 36 / tr - 96.736 * log10tr + tr**6
    log10pr = 5.8031817 * log10tr + 0.07608141 * alpha
    log10pr += p["omega"] * 4.86601 * (log10tr + 0.03721754 * alpha)
    return p["Pc"] * exponential(logarithm(10.0) * log10pr)


def _validate(method, T, parameters):
    if method not in VOLUME_METHODS:
        raise ValueError(f"unknown volume method {method!r}")
    spec = VOLUME_METHODS[method]
    required = set(spec.required.split())
    optional = {*spec.optional.split(), "MW"}
    if required - parameters.keys() or parameters.keys() - required - optional:
        raise ValueError(f"{method}: missing or unexpected parameters")
    for name, value in {"T": T, **parameters}.items():
        if name in {"has_hydroxyl", "hydroxyl_polar_term", "estimate_saturation"}:
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        elif isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    for name in ("T", "Tc", "Pc", "P", "Vc", "Zc", "MW", "Tb", "Tt", "Vml", "Tm", "rho0", "rhoc",
                 "P_ref", "Psat", "Vs", "V_ref", "rho_ref"):
        if name in {"T": T, **parameters}:
            _positive(T if name == "T" else parameters[name], name)
    if parameters.get("dipole", 0) < 0:
        raise ValueError("dipole must be nonnegative")
    if method == "Campbell_Thodos" and parameters["Tb"] >= parameters["Tc"]:
        raise ValueError("Campbell-Thodos requires Tb < Tc")
    if method == "Campbell_Thodos" and parameters.get("has_hydroxyl", 0):
        _positive(parameters.get("dipole", 0), "hydroxyl dipole")
    if method == "PPDS17":
        _positive(parameters["a0"], "a0")
    if method == "Rackett_fit":
        _positive(parameters["b"], "b")
        _positive(parameters["n"], "n")
    if method == "COSTALD_compressed":
        estimated = parameters.get("estimate_saturation", 0)
        if not estimated and (not {"Psat", "Vs"} <= parameters.keys() or "Vc" in parameters):
            raise ValueError("explicit Psat and Vs required; explicitly select estimation")
        if estimated and ("Vs" in parameters or "Vc" not in parameters):
            raise ValueError("estimated saturation requires Vc, not a frozen Vs")
    return spec


def _converted(method, T, parameters, output):
    spec = VOLUME_METHODS[method]
    value = _positive(_formula(method, T, parameters), "volume/density")
    if output is None:
        output = (
            "molar_volume"
            if "MW" in parameters and spec.native_output != "molar_volume"
            else spec.native_output
        )
    if output not in {"molar_volume", "density", "specific_volume"}:
        raise ValueError("output must be molar_volume, density or specific_volume")
    if output == spec.native_output:
        return value
    mw = parameters.get("MW")
    if "molar_volume" in (output, spec.native_output):
        if mw is None:
            raise ValueError("mass/molar conversion requires explicit MW in g/mol")
        _positive(mw, "MW")
    if spec.native_output == "molar_volume":
        specific = value / (mw / 1000)
    elif spec.native_output == "density":
        specific = 1 / value
    else:
        specific = value
    return (
        specific * mw / 1000
        if output == "molar_volume"
        else 1 / specific
        if output == "density"
        else specific
    )


def volume_derivatives(method, T, parameters, *, order=1, output=None):
    """Value and analytic T derivatives through order 3, holding inputs fixed.

    Saturation derivatives follow the saturation fit, not an isobaric liquid
    path. Explicit Psat/Vs and Tait coefficients are held fixed; their physical
    temperature dependence must be supplied by the caller or by an explicitly
    selected saturation estimator. A finite critical value does not imply finite derivatives. Table
    knots have at most two continuous derivatives and the .98 join has none.
    """
    _validate(method, T, parameters)
    return derivative_values(lambda t: _converted(method, t, parameters, output), T, order=order)


def volume_value(method, T, *, output=None, **parameters):
    """Evaluate the chosen pure equation; output=None follows documented units.

    Campbell hydroxyl_polar_term=1 (default) matches the 1.5.2 public API;
    =0 follows its rendered hydroxyl beta equation. Neither is a fitted
    correction. This convention is a discrete model parameter.
    """
    return volume_derivatives(method, T, parameters, order=0, output=output)[0]


def volume_pressure_derivatives(method, T, parameters, *, order=1, output=None):
    """Value and P derivatives through third order, at fixed T/reference inputs.

    At the compression lower boundary these are derivatives from the liquid
    side. Saturation fits never acquire an artificial independent P response.
    """
    spec = _validate(method, T, parameters)
    if not spec.uses_pressure:
        raise ValueError("pressure derivative requires a pressure-dependent volume method")
    return derivative_values(
        lambda p: _converted(method, T, {**parameters, "P": p}, output),
        parameters["P"], order=order)


def volume_parameter_jacobian(method, T, parameters, *, output=None, names=None):
    """Analytic derivatives w.r.t. explicitly supplied continuous parameters.

    No derivative of the discrete hydroxyl selector; branch/knot derivatives
    must exist. Units are output units per parameter unit.
    """
    _validate(method, T, parameters)
    discrete = {"has_hydroxyl", "hydroxyl_polar_term", "estimate_saturation"}
    names = tuple(k for k in parameters if k not in discrete) if names is None else tuple(names)
    if (
        len(set(names)) != len(names)
        or set(names) - parameters.keys()
        or discrete.intersection(names)
    ):
        raise ValueError("Jacobian names must identify unique supplied continuous parameters")
    result = {}
    for key in names:
        values = {**parameters, key: FirstDerivative(parameters[key], 1.0)}
        value = _converted(method, T, values, output)
        slope = scalar_value(value.slope) if isinstance(value, FirstDerivative) else 0.0
        if not isfinite(slope):
            raise ValueError("nonfinite volume parameter derivative")
        result[key] = slope
    return result
