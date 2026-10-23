"""Four chevron plate friction models; factors use straight port-to-port length."""

from math import cos, isfinite, log, log10, pi, radians, sin, sqrt, tan

from chemworld.physchem.flow_meters import _positive

PLATE_FRICTION_METHODS = ("Kumar", "Martin 1999", "Martin VDI", "Muley Manglik")
# Published piecewise fitted equation coefficients, not the original plotted observations.
_KUMAR = (
    (30, (10, 100), (50, 19.4, 2.99), (1, 0.589, 0.183)),
    (45, (15, 300), (47, 18.29, 1.441), (1, 0.652, 0.206)),
    (50, (20, 300), (34, 11.25, 0.772), (1, 0.631, 0.161)),
    (60, (40, 400), (24, 3.24, 0.76), (1, 0.457, 0.215)),
    (65, (50, 500), (24, 2.8, 0.639), (1, 0.451, 0.213)),
)


def plate_friction_factor(
    reynolds: float,
    chevron_angle_degrees: float,
    *,
    method: str,
    enlargement_factor: float = 1.0,
    darcy: bool = True,
    check_recommended_ranges: bool = True,
) -> float:
    """Kumar piecewise fit, Martin1999/VDI, or Muley-Manglik sinusoidal plates.

    Kumar has empirical discontinuities at coefficient boundaries. This does
    not imply discontinuous physical pressure. Enlargement enters only Muley.
    """
    _positive(reynolds, "Reynolds number")
    if not isfinite(chevron_angle_degrees) or not 0 <= chevron_angle_degrees < 90:
        raise ValueError("chevron angle must lie in [0,90) degrees")
    _positive(enlargement_factor, "plate enlargement factor")
    if method not in PLATE_FRICTION_METHODS:
        raise ValueError("unknown plate friction method")
    if method != "Muley Manglik" and enlargement_factor != 1:
        raise ValueError("only Muley-Manglik accepts an enlargement correction")
    r, beta = reynolds, chevron_angle_degrees
    if method == "Kumar":
        if check_recommended_ranges and not (0.1 <= r <= 1e4 and 30 <= beta <= 65):
            raise ValueError("Kumar development domain Re.1-10000 and angle30-65 degrees")
        row = next((row for row in _KUMAR if beta <= row[0]), _KUMAR[-1])
        _, thresholds, coefficients, powers = row
        branch = 0 if r <= thresholds[0] else 1 if r <= thresholds[1] else 2
        factor = 4 * coefficients[branch] * r ** -powers[branch]
    elif method == "Muley Manglik":
        if check_recommended_ranges and not (
            r > 1000 and 30 <= beta <= 60 and 1 <= enlargement_factor <= 1.5
        ):
            raise ValueError("Muley-Manglik requires Re>1000, angle30-60, enlargement1-1.5")
        phi = enlargement_factor
        angle_fit = 2.917 - 0.1277 * beta + 0.002016 * beta**2
        area_fit = 5.474 - 19.02 * phi + 18.93 * phi**2 - 5.341 * phi**3
        factor = 4 * angle_fit * area_fit * r ** (-0.2 - 0.0577 * sin(pi * beta / 45 + 2.1))
    else:
        angle = radians(beta)
        if method == "Martin 1999":
            base = 16 / r if r < 2000 else (1.56 * log(r) - 3) ** -2
            correction = 149 / r + 0.9625 if r < 2000 else 9.75 * r**-0.289
            b, c, multiplier = 0.045, 0.09, 4
        else:
            base = 64 / r if r < 2000 else (1.8 * log10(r) - 1.5) ** -2
            correction = 597 / r + 3.85 if r < 2000 else 39 * r**-0.289
            b, c, multiplier = 0.18, 0.36, 1
        inverse = cos(angle) / sqrt(b * tan(angle) + c * sin(angle) + base / cos(angle))
        inverse += (1 - cos(angle)) / sqrt(3.8 * correction)
        factor = multiplier / inverse**2
    _positive(factor, "plate friction result")
    return factor if darcy else factor / 4
