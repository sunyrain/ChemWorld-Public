"""Entrances and diameter transitions on their explicit velocity basis.

Chart methods require caller FittingCurves. Formula methods are mathematical
correlations; valid installation and phase scope remain caller declarations.
"""

from math import atan, cos, isfinite, log, pi, radians, sin, sqrt, tan

from chemworld.physchem.fitting_curves import FittingCurves
from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.straight_friction import straight_friction_correlation
from chemworld.physchem.valve_fittings import change_loss_velocity_basis


def _curve(curves: FittingCurves | None, name: str, x: float, y: float | None = None) -> float:
    if curves is None:
        raise ValueError(f"caller fitting curves required for {name}")
    return curves.evaluate(name, x, y)


def _fd(diameter: float, roughness: float, reynolds: float | None, provided: float | None) -> float:
    if provided is not None:
        _positive(provided, "Darcy friction factor")
        return provided
    if reynolds is None:
        raise ValueError("Reynolds number or explicit friction factor required")
    return straight_friction_correlation(
        reynolds, roughness / diameter, check_recommended_ranges=False
    )


def entrance_loss_coefficient(
    shape: str,
    *,
    diameter_m: float = 1,
    method: str = "Rennels",
    thickness_m: float = 0,
    projection_m: float | None = None,
    rounding_m: float = 0,
    bevel_length_m: float = 0,
    angle_degrees: float = 45,
    orifice_diameter_m: float | None = None,
    curves: FittingCurves | None = None,
) -> float:
    """K on pipe velocity; beveled-orifice K uses the orifice velocity."""
    _positive(diameter_m, "diameter")
    if any(not isfinite(v) or v < 0 for v in (thickness_m, rounding_m, bevel_length_m)):
        raise ValueError("finite nonnegative entrance geometry required")
    if not isfinite(angle_degrees) or not 0 <= angle_degrees <= 90:
        raise ValueError("entrance angle must be in [0,90] degrees")
    if shape == "exit":
        return 1.0
    if shape == "sharp":
        fixed = {
            "Rennels": 0.57,
            "Miller": 0.5092676683721356,
            "Swamee": 0.5,
            "Blevins": 0.5,
            "Crane": 0.5,
            "Idelchik": 0.5,
        }
        if method not in fixed:
            raise ValueError("unknown sharp entrance method")
        return fixed[method]
    if shape == "angled":
        if method != "Idelchik":
            raise ValueError("angled entrance uses Idelchik")
        c = cos(radians(angle_degrees))
        return 0.57 + 0.3 * c + 0.2 * c * c
    if shape in {"distance", "distance45"}:
        ratio = thickness_m / diameter_m
        if shape == "distance45":
            return _curve(curves, "entrance_distance45_Miller", (min(ratio, 0.3) - 0.15) * 20 / 3)
        if method == "Crane":
            return 0.78
        if method == "Rennels":
            t = min(ratio, 0.05)
            return 1.12 - 22 * t + 216 * t * t + 80 * t**3
        if method == "Miller":
            return _curve(curves, "entrance_distance_Miller", (min(ratio, 0.3) - 0.15) * 20 / 3)
        if method == "Harris":
            return _curve(curves, "entrance_distance_Harris", min(ratio, 0.289145))
        if method == "Idelchik":
            projection = diameter_m if projection_m is None else projection_m
            if not isfinite(projection) or projection < 0:
                raise ValueError("nonnegative entrance projection required")
            return max(
                0.0,
                _curve(
                    curves,
                    "entrance_distance_Idelchik",
                    min(projection / diameter_m, 10),
                    min(ratio, 1),
                ),
            )
    elif shape in {"beveled", "beveled_orifice"}:
        bore = diameter_m if orifice_diameter_m is None else orifice_diameter_m
        _positive(bore, "entrance orifice")
        if bore > diameter_m:
            raise ValueError("orifice must fit pipe")
        ratio = bevel_length_m / (bore if shape == "beveled_orifice" else diameter_m)
        if method == "Idelchik" and shape == "beveled":
            return _curve(curves, "entrance_beveled_Idelchik", 2 * angle_degrees, ratio)
        if method == "Rennels":
            fraction = angle_degrees / 90
            cb = (1 - fraction) * fraction ** (1 / (1 + ratio))
            multiplier = 1 if shape == "beveled_orifice" else 1.5
            contraction = 1 + 0.622 * (1 - multiplier * cb * ratio ** ((1 - ratio**0.25) / 2))
            target = (bore / diameter_m) ** 2 if shape == "beveled_orifice" else 1
            return 0.0696 * (1 - cb * ratio) * contraction**2 + (contraction - target) ** 2
    elif shape == "rounded":
        ratio = rounding_m / diameter_m
        if method == "Rennels":
            if ratio > 1:
                return 0.03
            contraction = 1 + 0.622 * (1 - 0.3 * sqrt(ratio) - 0.7 * ratio) ** 4
            return 0.0696 * (1 - 0.569 * ratio) * contraction**2 + (contraction - 1) ** 2
        if method == "Swamee":
            return 0.5 / (1 + 36 * ratio**1.2)
        if method == "Miller":
            return _curve(curves, "entrance_rounded_Miller", (min(ratio, 0.3) - 0.15) * 20 / 3)
        if method == "Crane":
            return 0.04 if ratio > 0.15 else _curve(curves, "entrance_rounded_Crane", ratio)
        if method == "Harris":
            return 0.0 if ratio > 0.16 else _curve(curves, "entrance_rounded_Harris", ratio)
        if method == "Idelchik":
            return _curve(curves, "entrance_rounded_Idelchik", min(ratio, 0.2))
    raise ValueError("unknown entrance shape/method combination")


def _transition_geometry(
    inlet: float, outlet: float, length: float | None, angle: float | None
) -> tuple[float, float]:
    if angle is not None:
        if not isfinite(angle) or not 0 < angle <= 180:
            raise ValueError("transition angle in (0,180] required")
        theta = radians(angle)
        return theta, abs(outlet - inlet) / (2 * tan(theta / 2))
    if length is None or not isfinite(length) or length < 0:
        raise ValueError("positive transition angle or nonnegative length required")
    return pi if length == 0 else 2 * atan(abs(outlet - inlet) / (2 * length)), length


def transition_loss_coefficient(
    shape: str,
    *,
    inlet_diameter_m: float,
    outlet_diameter_m: float,
    method: str = "Rennels",
    length_m: float | None = None,
    angle_degrees: float | None = None,
    rounding_m: float = 0,
    reynolds: float | None = None,
    roughness_m: float = 0,
    friction_factor: float | None = None,
    outlet_friction_factor: float | None = None,
    curves: FittingCurves | None = None,
) -> float:
    """All contraction/diffuser losses use the smaller-pipe velocity basis."""
    d1, d2 = inlet_diameter_m, outlet_diameter_m
    for d in (d1, d2):
        _positive(d, "diameter")
    if d1 == d2:
        return 0.0
    contraction = shape.startswith("contraction")
    if (contraction and d2 > d1) or (not contraction and d1 > d2):
        raise ValueError("diameter ordering conflicts with transition shape")
    if not isfinite(rounding_m) or rounding_m < 0:
        raise ValueError("nonnegative rounding required")
    beta = min(d1, d2) / max(d1, d2)
    deficit = 1 - beta**2
    if shape in {
        "contraction conical",
        "contraction beveled",
        "diffuser conical",
        "diffuser curved",
        "diffuser pipe reducer",
    }:
        theta, length = _transition_geometry(d1, d2, length_m, angle_degrees)
    else:
        theta, length = pi, 0.0
    degrees = theta * 180 / pi
    s = sin(theta / 2)
    if shape == "contraction rounded":
        radius = rounding_m / d2
        if method == "Miller":
            return max(0.0, _curve(curves, "contraction_rounded_Miller", beta**2, min(radius, 0.1)))
        if method == "Idelchik":
            return _curve(curves, "contraction_rounded_Idelchik", radius) * deficit
        if method == "Rennels":
            lam = 1 + 0.622 * (1 - 0.3 * sqrt(radius) - 0.7 * radius) ** 4 * (
                1 - 0.215 * beta**2 - 0.785 * beta**5
            )
            return (
                0.0696 * (1 - 0.569 * radius) * (1 - sqrt(radius) * beta) * (1 - beta**5) * lam**2
                + (lam - 1) ** 2
            )
    elif shape == "contraction beveled":
        if length_m is None or angle_degrees is None:
            raise ValueError("beveled transition needs length and angle")
        cb = length_m / d2 * 2 * beta * tan(theta / 2) / (1 - beta)
        lam = 1 + 0.622 * (1 + cb * ((theta / pi) ** 0.8 - 1)) * (
            1 - 0.215 * beta**2 - 0.785 * beta**5
        )
        return 0.0696 * (1 + cb * (s - 1)) * (1 - beta**5) * lam**2 + (lam - 1) ** 2
    elif shape in {"contraction sharp", "contraction conical"}:
        sharp = shape.endswith("sharp")
        if method == "Rennels":
            lam = 1 + 0.622 * (theta / pi) ** 0.8 * (1 - 0.215 * beta**2 - 0.785 * beta**5)
            local = 0.0696 * s * (1 - beta**5) * lam**2 + (lam - 1) ** 2
            return (
                local
                if sharp
                else local
                + _fd(d2, roughness_m, reynolds, friction_factor) * (1 - beta**4) / (8 * s)
            )
        if method == "Crane":
            return (0.8 * s if degrees < 45 else 0.5 * sqrt(s)) * deficit
        if method == "Hooper":
            if reynolds is None:
                raise ValueError("Hooper requires Reynolds")
            _positive(reynolds, "Reynolds number")
            upstream = (
                (1.2 + 160 / reynolds) * (beta**-4 - 1)
                if reynolds <= 2500
                else (0.6 + 0.48 * _fd(d1, roughness_m, reynolds, friction_factor))
                * beta**-2
                * (beta**-2 - 1)
            )
            modifier = 1 if sharp else sqrt(s) if degrees > 45 else 1.6 * s
            return upstream * modifier * beta**4
        if not sharp:
            if method == "Swamee":
                return 0.315 * theta ** (1 / 3)
            if method == "Miller":
                return min(
                    0.5,
                    max(
                        0.0,
                        _curve(
                            curves,
                            "contraction_conical_Miller",
                            log(min(10, max(0.1, 2 * length / d2))),
                            log(min(4, max(1.1, beta**-2))),
                        ),
                    ),
                )
            if method == "Blevins":
                return _curve(
                    curves,
                    "contraction_conical_Blevins",
                    min(length / d2, 0.6),
                    min(10, max(1.2, beta**-2)),
                )
            if method == "Idelchik":
                return _curve(
                    curves, "entrance_beveled_Idelchik", degrees, length / d2
                ) * deficit + _curve(
                    curves,
                    "transition_friction_Idelchik",
                    min(20, max(2, degrees)),
                    min(0.6, max(0.05, beta**2)),
                )
    elif shape in {"diffuser sharp", "diffuser conical"}:
        sharp = shape.endswith("sharp")
        if method == "Hooper":
            if reynolds is None:
                raise ValueError("Hooper requires Reynolds")
            _positive(reynolds, "Reynolds number")
            base = (
                2 * (1 - beta**4)
                if reynolds < 4000
                else (1 + 0.8 * _fd(d1, roughness_m, reynolds, friction_factor)) * deficit**2
            )
            return base if sharp or degrees > 45 else base * 2.6 * s
        if method == "Rennels":
            if sharp:
                return deficit**2
            friction = _fd(d2, roughness_m, reynolds, friction_factor) * (1 - beta**4) / (8 * s)
            if degrees <= 20:
                return 8.3 * tan(theta / 2) ** 1.75 * deficit**2 + friction
            if degrees <= 60:
                base = 1.366 * sqrt(sin(radians(2 * (degrees - 15)))) - 0.17
                if beta < 0.5:
                    base -= 3.28 * (0.0625 - beta**4) * sqrt(0.025 * (degrees - 20))
                return base * deficit**2 + friction
            base = (
                1.205 - 3.28 * (0.0625 - beta**4) - 12.8 * beta**6 * sqrt((degrees - 60) / 120)
                if beta < 0.5
                else 1.205 - 0.2 * sqrt((degrees - 60) / 120)
            )
            return base * deficit**2
        if not sharp:
            if method == "Crane":
                return (2.6 * s if degrees < 45 else 1) * deficit**2
            if method == "Miller":
                return max(
                    0.0,
                    _curve(
                        curves,
                        "diffuser_conical_Miller",
                        log(min(20, max(0.1, 2 * length / d1))),
                        log(min(4, max(1.1, beta**-2))),
                    ),
                )
            if method == "Idelchik":
                return _curve(
                    curves,
                    "transition_friction_Idelchik",
                    min(20, max(2, degrees)),
                    min(0.6, max(0.05, beta**2)),
                ) + _curve(curves, "diffuser_conical_Idelchik", min(0.6, beta**2), max(3, degrees))
            if method == "Swamee":
                return 1 / sqrt(
                    0.25
                    * theta**-3
                    * (1 + 0.6 * beta**1.67 * (pi - theta) / theta) ** (0.533 / beta - 2.6)
                )
    elif shape == "diffuser curved":
        ratio = length / d1
        return (
            (1.01 - 0.624 * ratio + 0.3 * ratio**2 - 0.074 * ratio**3 + 0.007 * ratio**4)
            * (1.43 - 1.3 * beta**2)
            * deficit**2
        )
    elif shape == "diffuser pipe reducer":
        _positive(length, "reducer length")
        fd1 = _fd(d1, roughness_m, reynolds, friction_factor)
        fd2 = fd1 if outlet_friction_factor is None else outlet_friction_factor
        theta = 2 * atan((d2 - d1) / (1.2 * length))
        return (
            fd1 * 0.2 * length / d1
            + fd1 * (1 - beta) / (8 * sin(theta / 2))
            + fd2 * 0.2 * length / d2 * beta**4
        )
    raise ValueError("unknown diameter transition shape/method")


def staged_diffuser_loss(
    diameters_m: tuple[float, ...],
    lengths_m: tuple[float, ...],
    *,
    friction_factor: float,
    method: str = "Rennels",
    curves: FittingCurves | None = None,
    common_velocity_basis: bool = True,
) -> float:
    """Sum stage losses on inlet velocity; optionally expose source raw-K convention."""
    if len(diameters_m) != len(lengths_m) + 1 or not lengths_m:
        raise ValueError("one length per diameter transition required")
    total = 0.0
    for inlet, outlet, length in zip(diameters_m[:-1], diameters_m[1:], lengths_m, strict=True):
        value = transition_loss_coefficient(
            "diffuser conical",
            inlet_diameter_m=inlet,
            outlet_diameter_m=outlet,
            length_m=length,
            friction_factor=friction_factor,
            method=method,
            curves=curves,
        )
        total += (
            change_loss_velocity_basis(value, from_diameter_m=inlet, to_diameter_m=diameters_m[0])
            if common_velocity_basis
            else value
        )
    return total
