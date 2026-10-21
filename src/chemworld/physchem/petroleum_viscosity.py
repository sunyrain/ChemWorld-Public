"""Petroleum viscosity relations with explicit kinematic/mass-density bases.

Twu (1985), doi:10.1021/i200031a064, equations 3-22; ASTM D2270.
Twu predicts kinematic viscosity. Its 60 F density determines specific gravity;
conversion to dynamic viscosity at another temperature needs that state's
density, which must not be confused with the characterization density.
"""

from bisect import bisect_left
from math import exp, log

from chemworld.physchem.caloric_math import exponential, logarithm
from chemworld.physchem.viscosity_kernels import _positive
from chemworld.physchem.viscosity_scale_data import (
    VISCOSITY_INDEX_H,
    VISCOSITY_INDEX_L,
    VISCOSITY_INDEX_NU,
)


def kinematic_viscosity(dynamic_viscosity_Pa_s, density_kg_m3):
    """Convert Pa s to m²/s using the density at the same physical state."""
    return _positive(dynamic_viscosity_Pa_s, "dynamic viscosity") / _positive(
        density_kg_m3, "density"
    )


def dynamic_viscosity(kinematic_viscosity_m2_s, density_kg_m3):
    """Convert m²/s to Pa s using the density at the same physical state."""
    return _positive(kinematic_viscosity_m2_s, "kinematic viscosity") * _positive(
        density_kg_m3, "density"
    )


def twu_1985_kinematic_viscosity(T, Tb, density_60F_kg_m3):
    """Twu petroleum-fraction estimate in m²/s, T/Tb K; supports T jets.

    The reference fluid uses Rankine and centistokes internally. SG is the
    60 F/60 F convention, water density 999.0170824078306 kg/m³. Explicitly
    characterize the fraction at 60 F, not at the requested T.
    """
    for value, label in (
        (T, "temperature"),
        (Tb, "boiling temperature"),
        (density_60F_kg_m3, "density at 60 F"),
    ):
        _positive(value, label)
    tb = Tb * 1.8
    tc0 = tb / (
        0.533272
        + 0.191017e-3 * tb
        + 0.779681e-7 * tb**2
        - 0.284376e-10 * tb**3
        + 0.959468e28 / tb**13
    )
    alpha = 1 - tb / tc0
    nu2 = exp(4.73227 - 27.0975 * alpha + 49.4491 * alpha**2 - 50.4706 * alpha**4) - 1.5
    _positive(nu2, "Twu reference viscosity")
    nu1 = exp(0.801621 + 1.37179 * log(nu2))
    sg0 = 0.843593 - 0.128624 * alpha - 3.36159 * alpha**3 - 13749.5 * alpha**12
    delta_sg = density_60F_kg_m3 / 999.0170824078306 - sg0
    x = abs(1.99873 - 56.7394 / tb**0.5)
    f2 = x * delta_sg - 21.1141 * delta_sg**2 / tb**0.5
    f1 = 1.33932 * x * delta_sg - 21.1141 * delta_sg**2 / tb**0.5
    offset = 450 / tb
    points = []
    for nu, correction in ((nu1, f1), (nu2, f2)):
        if 1 - 2 * correction == 0:
            raise ValueError("Twu specific-gravity correction is singular")
        shifted = (
            exp(log(nu + offset) * ((1 + 2 * correction) / (1 - 2 * correction)) ** 2) - offset
        )
        _positive(shifted, "Twu corrected viscosity")
        z = shifted + 0.7 + exp(-1.47 - 1.84 * shifted - 0.51 * shifted**2)
        _positive(z - 1, "Twu log-log coordinate minus one")
        points.append(log(log(z)))
    slope = (points[0] - points[1]) / log(559.67 / 669.67)
    z = exponential(exponential(points[0] + slope * logarithm(T * 1.8 / 559.67)))
    a = z - 0.7
    # Correct grouping from original eq. 19; public API doc has broken parentheses.
    nu = a - exponential(-0.7487 - 3.295 * a + 0.6119 * a**2 - 0.3193 * a**3)
    return _positive(nu, "Twu kinematic viscosity") * 1e-6


def twu_1985_viscosity(T, Tb, density_60F_kg_m3, *, density_at_T_kg_m3=None):
    """Pa s. Omitted state density selects chemicals' reference-density convention.

    Supplying density_at_T_kg_m3 gives the physical dynamic viscosity at T.
    The omitted-density compatibility convention is an approximation, not a
    thermal-expansion model; all characterization still uses the 60 F density.
    """
    density = density_60F_kg_m3 if density_at_T_kg_m3 is None else density_at_T_kg_m3
    return dynamic_viscosity(twu_1985_kinematic_viscosity(T, Tb, density_60F_kg_m3), density)


def viscosity_index(nu_40_m2_s, nu_100_m2_s, *, rounding=False):
    """ASTM D2270 viscosity index; input kinematic viscosity (not Pa s).

    Below 2 cSt at 100 C the standard has no index and None is returned.
    Standard rounding of a half-integer uses the nearest even integer.
    """
    _positive(nu_40_m2_s, "40 C kinematic viscosity")
    _positive(nu_100_m2_s, "100 C kinematic viscosity")
    if not isinstance(rounding, bool):
        raise ValueError("rounding must be boolean")
    u, v = nu_40_m2_s * 1e6, nu_100_m2_s * 1e6
    if v < 2:
        return None
    if v >= 70:
        low, high = 0.8353 * v * v + 14.67 * v - 216, 0.1684 * v * v + 11.85 * v - 97
    else:
        index = bisect_left(VISCOSITY_INDEX_NU, v)
        if VISCOSITY_INDEX_NU[index] == v:
            low, high = VISCOSITY_INDEX_L[index], VISCOSITY_INDEX_H[index]
        else:
            a, b = VISCOSITY_INDEX_NU[index - 1 : index + 1]
            fraction = (v - a) / (b - a)
            low, high = (
                values[index - 1] + fraction * (values[index] - values[index - 1])
                for values in (VISCOSITY_INDEX_L, VISCOSITY_INDEX_H)
            )
    value = (
        100 * (low - u) / (low - high)
        if u > high
        else (10 ** ((log(high) - log(u)) / log(v)) - 1) / 0.00715 + 100
    )
    return round(value) if rounding else value
