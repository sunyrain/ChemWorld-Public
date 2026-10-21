"""Complete historical efflux scale conversions using scientific tables.

One log-log interpolant per tabulated scale defines both directions;
inversion uses the original shared root solver. Reference runtime algorithms
and precomputed spline code are neither imported nor copied.
"""

from functools import lru_cache
from math import exp, log

import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline

from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root
from chemworld.physchem.viscosity_kernels import _positive
from chemworld.physchem.viscosity_scale_data import VISCOSITY_LINEAR_SCALES, VISCOSITY_SCALE_TABLES

VISCOSITY_SCALES = tuple(sorted(set(VISCOSITY_SCALE_TABLES) | set(VISCOSITY_LINEAR_SCALES)))


def _name(name):
    if not isinstance(name, str):
        raise ValueError("viscosity scale must be a name")
    name = " ".join(name.lower().replace("degrees", "").replace("seconds", "").split())
    if name not in VISCOSITY_SCALES:
        raise ValueError(f"unknown viscosity scale {name!r}")
    return name


@lru_cache(maxsize=46)
def _table_spline(name, interpolation):
    scale, nu = VISCOSITY_SCALE_TABLES[name]
    order = np.argsort(scale)
    x, y = np.log(np.asarray(scale)[order]), np.log(np.asarray(nu)[order])
    spline = (
        PchipInterpolator(x, y) if interpolation == "monotone" else make_interp_spline(x, y, k=3)
    )
    return x, y, spline


def _check_interval(value, low, high, *, extrapolate, label):
    # Permit only a small endpoint roundoff from reversible inversion.
    if not extrapolate and (value < low * (1 - 1e-12) or value > high * (1 + 1e-12)):
        raise ValueError(f"{label} outside valid interval [{low}, {high}]")


def _linear_tail(point, x, y):
    """Continue the endpoint secant in log coordinates; None inside the table."""
    if x[0] <= point <= x[-1]:
        return None
    i = 0 if point < x[0] else len(x) - 2
    return float(y[i] + (point - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]))


def _saybolt(nu):
    polynomial = 3930.2 + 262.7 * nu + 23.97 * nu**2 + 1.646 * nu**3
    return 4.6324 * nu + (1 + 0.03264 * nu) / (polynomial * 1e-5)


def _invert_saybolt(value):
    def residual(point):
        return [_saybolt(exp(point[0])) - value]

    def jacobian(point):
        nu = exp(point[0])
        p = 3930.2 + 262.7 * nu + 23.97 * nu**2 + 1.646 * nu**3
        dp = 262.7 + 47.94 * nu + 4.938 * nu**2
        slope = 4.6324 + 1e5 * (0.03264 * p - (1 + 0.03264 * nu) * dp) / p**2
        return [[slope * nu]]

    if value <= _saybolt(0):
        raise ValueError("Saybolt formula has no positive kinematic-viscosity solution")
    result = solve_root(
        residual,
        [log(value / 4.6324)],
        jacobian=jacobian,
        variables=(ResidualVariable("log_nu_cSt", "dimensionless", 1),),
        residual_scales=[value],
        policy=RootPolicy(tolerance=2e-14),
    )
    if not result.success:
        raise ValueError(f"Saybolt inversion failed: {result.failure_reason}")
    return exp(result.value[0])


def _to_cst(value, name, extrapolate, interpolation):
    if name == "kinematic viscosity":
        return value * 1e6
    if name in VISCOSITY_SCALE_TABLES:
        values, _ = VISCOSITY_SCALE_TABLES[name]
        _check_interval(value, min(values), max(values), extrapolate=extrapolate, label=name)
        if name == "saybolt universal":
            return _invert_saybolt(value)
        x, y, spline = _table_spline(name, interpolation)
        tail = (
            _linear_tail(log(value), x, y) if extrapolate and interpolation == "monotone" else None
        )
        return exp(float(spline(log(value))) if tail is None else tail)
    coefficient, minimum = VISCOSITY_LINEAR_SCALES[name]
    _check_interval(value, minimum, float("inf"), extrapolate=extrapolate, label=name)
    return value * coefficient


def _from_cst(nu, name, extrapolate, interpolation):
    if name == "kinematic viscosity":
        return nu * 1e-6
    if name in VISCOSITY_SCALE_TABLES:
        values, nus = VISCOSITY_SCALE_TABLES[name]
        if name == "saybolt universal":
            result = _saybolt(nu)
        else:
            _check_interval(nu, min(nus), max(nus), extrapolate=extrapolate, label=name + " cSt")
            x, y, spline = _table_spline(name, interpolation)
            target = log(nu)
            order = np.argsort(y)
            if extrapolate and interpolation == "monotone":
                tail = _linear_tail(target, y[order], x[order])
                if tail is not None:
                    return exp(tail)
            initial = float(np.interp(target, y[order], x[order]))
            result = solve_root(
                lambda point: [float(spline(point[0])) - target],
                [initial],
                jacobian=lambda point: [[float(spline(point[0], nu=1))]],
                variables=(
                    ResidualVariable(
                        "log_efflux",
                        "dimensionless",
                        1,
                        lower=None if extrapolate else float(x[0]),
                        upper=None if extrapolate else float(x[-1]),
                    ),
                ),
                residual_scales=[1],
                policy=RootPolicy(tolerance=2e-14),
            )
            if not result.success:
                raise ValueError(f"{name} inversion failed: {result.failure_reason}")
            result = exp(result.value[0])
        _check_interval(result, min(values), max(values), extrapolate=extrapolate, label=name)
        return result
    coefficient, minimum = VISCOSITY_LINEAR_SCALES[name]
    result = nu / coefficient
    _check_interval(result, minimum, float("inf"), extrapolate=extrapolate, label=name)
    return result


def viscosity_converter(
    value, old_scale, new_scale, *, extrapolate=False, interpolation="monotone"
):
    """Convert all 47 named scales; kinematic viscosity uses m²/s, efflux s.

    Engler and Barbey use degrees. This does not convert dynamic viscosity;
    use dynamic_viscosity/kinematic_viscosity with an explicit state density.
    Outside empirical table/linear limits conversion requires opt-in.
    The default shape-preserving log-log interpolation is reversible within
    the tables. Explicit interpolation='reference_cubic' reproduces the
    reference cubic curve, including Barbey's nonmonotone overshoot; that
    mode cannot guarantee an inverse for every forward value. Saybolt uses
    its analytic API equation in both modes, not an interpolated table.
    Monotone-mode extrapolation uses the endpoint log-log secant, so a
    zero endpoint PCHIP slope cannot create a noninvertible cubic tail.
    """
    _positive(value, "viscosity scale value")
    if not isinstance(extrapolate, bool):
        raise ValueError("extrapolate must be boolean")
    if interpolation not in {"monotone", "reference_cubic"}:
        raise ValueError("unknown viscosity scale interpolation")
    old, new = _name(old_scale), _name(new_scale)
    try:
        result = _from_cst(
            _to_cst(value, old, extrapolate, interpolation), new, extrapolate, interpolation
        )
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("viscosity scale conversion outside numerical domain") from exc
    return _positive(result, "converted viscosity scale value")
