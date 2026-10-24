"""IAPWS R8-97 static dielectric correlation, independently implemented.

Published 1997 physical constants and Eq. 1-6/Table 2 are used unchanged.
Density is an explicit argument; this correlation does not select a water phase.
"""

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any

from chemworld.physchem.caloric_math import scalar_value
from chemworld.physchem.second_order_pair import SecondOrderPair

_TERMS = (
    (0.978224486826, 1, 0.25),
    (-0.957771379375, 1, 1),
    (0.237511794148, 1, 2.5),
    (0.714692244396, 2, 1.5),
    (-0.298217036956, 3, 1.5),
    (-0.108863472196, 3, 2.5),
    (0.0949327488264, 4, 2),
    (-0.00980469816509, 5, 2),
    (0.0000165167634970, 6, 5),
    (0.0000937359795772, 7, 0.5),
    (-0.000000000123179218720, 10, 10),
)
WATER_DIELECTRIC_MOLAR_MASS_KG_MOL = 0.018015268


def water_relative_permittivity(temperature_K: Any, mass_density_kg_m3: Any) -> Any:
    """Relative permittivity; 238-873 K and 0-1250 kg/m3, finite density input.

    The rectangular computational range is not a stable-phase or pressure claim.
    """
    T, rho = temperature_K, mass_density_kg_m3
    if any(
        isinstance(scalar_value(v), bool) or not isfinite(scalar_value(v)) for v in (T, rho)
    ) or not (238 <= scalar_value(T) <= 873 and 0 <= scalar_value(rho) <= 1250):
        raise ValueError("dielectric correlation requires 238-873 K and 0-1250 kg/m3")
    delta, tau = rho / 322, 647.096 / T
    g = 1 + sum(n * delta**i * tau**j for n, i, j in _TERMS)
    g += 0.00196096504426 * delta * (T / 228 - 1) ** -1.2
    epsilon0 = 1 / (4e-7 * pi * 299792458**2)
    molar_density = rho / WATER_DIELECTRIC_MOLAR_MASS_KG_MOL
    A = 6.0221367e23 * 6.138e-30**2 * molar_density * g / (epsilon0 * 1.380658e-23 * T)
    B = 6.0221367e23 * 1.636e-40 * molar_density / (3 * epsilon0)
    return (1 + A + 5 * B + (9 + 2 * A + 18 * B + A * A + 10 * A * B + 9 * B * B) ** 0.5) / (
        4 - 4 * B
    )


@dataclass(frozen=True)
class WaterDielectricResponse:
    relative_permittivity: float
    d_temperature_K: float
    d_density_kg_m3: float
    d2_temperature_K2: float
    d2_temperature_density: float
    d2_density: float


def water_dielectric_response(
    temperature_K: float, mass_density_kg_m3: float
) -> WaterDielectricResponse:
    """Value and analytic first/second partials at explicit T and mass density."""
    value = water_relative_permittivity(
        SecondOrderPair(temperature_K, dx=1), SecondOrderPair(mass_density_kg_m3, dy=1)
    )
    return WaterDielectricResponse(value.value, value.dx, value.dy, value.dxx, value.dxy, value.dyy)
