"""Rounded/miter bends and coils, using existing native friction kernels."""

from math import isfinite, log10, pi, radians, sin, sqrt

import numpy as np

from chemworld.physchem.curved_friction import helical_friction_factor
from chemworld.physchem.fitting_curves import FittingCurves
from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.local_fittings import _curve, _fd
from chemworld.physchem.straight_friction import crane_fully_turbulent_friction


def _miller_outlet(base: float, length_ratio: float, curves: FittingCurves | None) -> float:
    if curves is None or "miller_outlet_limit" not in curves.curves:
        raise ValueError("caller Miller outlet correction curves required")
    points = curves.curves["miller_outlet_limit"].x
    clipped = min(max(base, points[0]), points[-1])
    length = min(length_ratio, 30)
    upper = min(len(points) - 1, max(1, int(np.searchsorted(points, clipped))))
    values = []
    for i in (upper - 1, upper):
        if length >= _curve(curves, "miller_outlet_limit", points[i]):
            value = 1.0
        elif length <= 0.01:
            value = _curve(curves, "miller_outlet_low", points[i])
        else:
            value = _curve(curves, f"miller_outlet_{i}", length)
        values.append(value)
    return float(np.interp(clipped, points[upper - 1 : upper + 1], values))


def _miller_roughness(reynolds: float, diameter: float, roughness: float) -> float:
    re = min(reynolds, 1e6)
    if re < 2040:
        return 1.0
    return _fd(diameter, roughness, re, None) / _fd(diameter, 0, re, None)


def bend_loss_coefficient(
    shape: str,
    *,
    diameter_m: float,
    angle_degrees: float,
    method: str = "Rennels",
    radius_m: float | None = None,
    reynolds: float | None = None,
    roughness_m: float = 0,
    friction_factor: float | None = None,
    unimpeded_length_m: float | None = None,
    curves: FittingCurves | None = None,
) -> float:
    _positive(diameter_m, "diameter")
    if (
        shape not in {"rounded", "miter"}
        or not isfinite(angle_degrees)
        or not 0 <= angle_degrees <= 180
    ):
        raise ValueError("rounded/miter bend and angle in [0,180] required")
    if not isfinite(roughness_m) or not 0 <= roughness_m < diameter_m:
        raise ValueError("roughness must be nonnegative and below diameter")
    if angle_degrees == 0:
        return 0.0
    radius = 5 * diameter_m if radius_m is None else radius_m
    _positive(radius, "bend radius")
    ratio = radius / diameter_m
    theta = radians(angle_degrees)
    sine = sin(theta / 2)
    if method == "Rennels":
        if shape == "miter":
            return 0.42 * sine + 2.56 * sine**3
        friction = _fd(diameter_m, roughness_m, reynolds, friction_factor)
        return (
            friction * theta * ratio
            + (0.1 + 2.4 * friction) * sine
            + 6.6 * friction * (sqrt(sine) + sine) / ratio ** (4 * theta / pi)
        )
    if method == "Swamee" and shape == "rounded":
        return (0.0733 + 0.923 / ratio**3.5) * sqrt(theta)
    if method in {"Crane", "Crane standard"}:
        friction = crane_fully_turbulent_friction(diameter_m)
        if shape == "miter":
            return friction * _curve(curves, "bend_miter_Crane", angle_degrees)
        if method == "Crane standard":
            return friction * _curve(curves, "bend_standard_Crane", angle_degrees)
        ratio = min(20, max(1, ratio))
        base = friction * _curve(curves, "bend_rounded_Crane", (ratio - 10.5) * 2 / 19)
        return base + (angle_degrees / 90 - 1) * (0.25 * pi * friction * ratio + 0.5 * base)
    if reynolds is None:
        raise ValueError("selected bend method requires Reynolds number")
    _positive(reynolds, "Reynolds number")
    if method == "Blevins" and shape == "miter":
        return _curve(curves, "bend_miter_Blevins", angle_degrees) * (2e5 / reynolds) ** 0.2
    if method == "Ito" and shape == "rounded":
        alpha = 1.0
        if ratio <= 50:
            alpha = float(
                np.interp(
                    angle_degrees,
                    (45, 90, 180),
                    (
                        1 + 5.13 / ratio**1.47,
                        0.95 + 4.42 / ratio**1.96 if ratio < 9.85 else 1.0,
                        1 + 5.06 / ratio**4.52,
                    ),
                )
            )
        if reynolds / ratio**2 > 360:
            return 0.00431 * alpha * angle_degrees * reynolds**-0.17 * ratio**0.84
        # Ito's low-Dean branch specifies smooth White/Srinivasan correlations;
        # it does not define an additional roughness correction.
        friction = helical_friction_factor(
            reynolds,
            diameter_m,
            2 * radius,
            roughness_m=0,
            transition_method="Srinivasan",
            laminar_method="White",
            turbulent_method="Srinivasan turbulent",
            check_recommended_ranges=False,
        ).friction_factor
        return 0.0175 * alpha * friction * angle_degrees * ratio
    if method == "Miller":
        ratio = min(10, max(0.5, ratio))
        if shape == "miter":
            base = _curve(curves, "bend_miter_Miller", (min(120, angle_degrees) - 60) / 60)
        else:
            base = _curve(curves, "bend_rounded_Miller", ratio, max(10, angle_degrees))
        re = min(1e8, max(1e4, reynolds))
        one = (
            _curve(curves, "bend_reynolds_Miller", log10(re), 1.0)
            if re < 207956.58904584477
            else 1.0
        )
        if shape == "miter" or (ratio <= 0.7 and base >= 0.4):
            factor = base / (base - 0.2 * one + 0.2)
            if factor > 2.2 or factor < 0:
                factor = 2.2
        elif ratio <= 1:
            factor = one
        elif ratio >= 2:
            factor = (
                0.4196741237602154
                if re == 1e8
                else 2.1775876405173977
                if re == 1e4
                else _curve(curves, "bend_reynolds_Miller", log10(re), 2.0)
            )
        elif re > 1048884.4656835075 or re > _curve(curves, "bend_reynolds_limit_Miller", ratio):
            factor = 1.0
        else:
            factor = _curve(curves, "bend_reynolds_Miller", log10(re), ratio)
        length = 20 * diameter_m if unimpeded_length_m is None else unimpeded_length_m
        if not isfinite(length) or length < 0:
            raise ValueError("nonnegative unimpeded length required")
        return (
            base
            * factor
            * _miller_roughness(reynolds, diameter_m, roughness_m)
            * _miller_outlet(base, length / diameter_m, curves)
        )
    raise ValueError("unknown bend shape/method")


def coil_loss_coefficient(
    shape: str,
    *,
    diameter_m: float,
    friction_factor: float,
    pitch_m: float,
    radius_m: float,
    turns: float = 1,
    outer_radius_m: float | None = None,
) -> float:
    for value in (diameter_m, friction_factor, pitch_m, radius_m, turns):
        _positive(value, "coil parameter")
    if shape == "helix":
        return turns * (
            friction_factor * sqrt((2 * pi * radius_m) ** 2 + pitch_m**2) / diameter_m
            + 0.2
            + 4.8 * friction_factor
        )
    if shape == "spiral":
        if outer_radius_m is None or outer_radius_m <= radius_m:
            raise ValueError("spiral outer radius must exceed inner radius")
        return (outer_radius_m - radius_m) / pitch_m * (
            friction_factor * pi * (outer_radius_m + radius_m) / diameter_m
            + 0.2
            + 4.8 * friction_factor
        ) + 13.2 * friction_factor / (radius_m / diameter_m) ** 2
    raise ValueError("coil shape must be helix or spiral")
