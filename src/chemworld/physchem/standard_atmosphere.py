"""US1976 lower atmosphere and refracted air columns from explicit profiles.

Standard layer boundaries and lapse rates are equation definitions. Layer
pressures are integrated from sea-level conditions, not loaded source arrays.
Temperature offset changes density/transport, not the standard pressure curve.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import cos, exp, isfinite, log, pi, sqrt

from scipy.integrate import quad

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.transport import FluidState

STANDARD_EARTH_RADIUS_M = 6356766.0
STANDARD_GRAVITY_M_S2 = 9.80665
STANDARD_AIR_MOLAR_MASS_KG_MOL = 0.0289644
STANDARD_GAS_CONSTANT_J_MOL_K = 8.31432
_HEIGHTS = (0, 11000, 20000, 32000, 47000, 51000, 71000, 84852)
_LAPSES = (-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002, 0)
_TEMPERATURES = [288.15]
_PRESSURES = [101325.0]
for _index in range(1, len(_HEIGHTS)):
    _delta = _HEIGHTS[_index] - _HEIGHTS[_index - 1]
    _lapse = _LAPSES[_index - 1]
    _temperature = _TEMPERATURES[-1] + _lapse * _delta
    _factor = STANDARD_GRAVITY_M_S2 * STANDARD_AIR_MOLAR_MASS_KG_MOL / STANDARD_GAS_CONSTANT_J_MOL_K
    _pressure = (
        _PRESSURES[-1] * exp(-_factor * _delta / _temperature)
        if _lapse == 0
        else (_PRESSURES[-1] * (_TEMPERATURES[-1] / _temperature) ** (_factor / _lapse))
    )
    _TEMPERATURES.append(_temperature)
    _PRESSURES.append(_pressure)


def standard_air_properties(temperature_K: float, pressure_Pa: float) -> dict[str, float]:
    """US1976 air density, Sutherland viscosity, conductivity and sound speed."""
    _positive(temperature_K, "temperature")
    _positive(pressure_Pa, "pressure")
    t = temperature_K
    specific_R = STANDARD_GAS_CONSTANT_J_MOL_K / STANDARD_AIR_MOLAR_MASS_KG_MOL
    return {
        "density_kg_m3": pressure_Pa / (specific_R * t),
        "viscosity_Pa_s": 1.458e-6 * t**1.5 / (t + 110.4),
        "conductivity_W_m_K": 2.64638e-3 * t**1.5 / (t + 245.4 * exp(-12 * log(10) / t)),
        "sound_speed_m_s": sqrt(1.4 * specific_R * t),
    }


def standard_gravity(elevation_m: float) -> float:
    if not isfinite(elevation_m) or elevation_m <= -STANDARD_EARTH_RADIUS_M:
        raise ValueError("elevation must be above the model Earth center")
    return (
        STANDARD_GRAVITY_M_S2
        * (STANDARD_EARTH_RADIUS_M / (STANDARD_EARTH_RADIUS_M + elevation_m)) ** 2
    )


@dataclass(frozen=True)
class StandardAtmosphereState:
    elevation_m: float
    geopotential_height_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    viscosity_Pa_s: float
    conductivity_W_m_K: float
    sound_speed_m_s: float
    gravity_m_s2: float
    temperature_offset_K: float
    extrapolated: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)

    def fluid_state(self) -> FluidState:
        """Same local air as existing single-phase transport property consumer."""
        cp = (1.4 / 0.4) * STANDARD_GAS_CONSTANT_J_MOL_K / STANDARD_AIR_MOLAR_MASS_KG_MOL
        return FluidState(self.density_kg_m3, self.viscosity_Pa_s, cp, self.conductivity_W_m_K)


def standard_atmosphere(
    elevation_m: float,
    temperature_offset_K: float = 0,
    *,
    extrapolate: bool = False,
) -> StandardAtmosphereState:
    """Declared geometric altitude -610..86000m; extrapolation is explicit."""
    if not isfinite(elevation_m) or not isfinite(temperature_offset_K):
        raise ValueError("finite altitude and temperature offset required")
    outside = not -610 <= elevation_m <= 86000
    if outside and not extrapolate:
        raise ValueError("altitude outside US1976 lower-atmosphere -610..86000m domain")
    gravity = standard_gravity(elevation_m)
    height = STANDARD_EARTH_RADIUS_M * elevation_m / (STANDARD_EARTH_RADIUS_M + elevation_m)
    index = max(0, min(7, bisect_left(_HEIGHTS, height) - 1))
    offset = height - _HEIGHTS[index]
    base_t, base_p, lapse = _TEMPERATURES[index], _PRESSURES[index], _LAPSES[index]
    temperature = base_t + lapse * offset
    _positive(temperature, "standard layer temperature")
    factor = STANDARD_GRAVITY_M_S2 * STANDARD_AIR_MOLAR_MASS_KG_MOL / STANDARD_GAS_CONSTANT_J_MOL_K
    pressure = (
        base_p * exp(-factor * offset / base_t)
        if lapse == 0
        else (base_p * (base_t / temperature) ** (factor / lapse))
    )
    actual_t = temperature + temperature_offset_K
    properties = standard_air_properties(actual_t, pressure)
    return StandardAtmosphereState(
        elevation_m,
        height,
        actual_t,
        pressure,
        properties["density_kg_m3"],
        properties["viscosity_Pa_s"],
        properties["conductivity_W_m_K"],
        properties["sound_speed_m_s"],
        gravity,
        temperature_offset_K,
        outside,
    )


def standard_pressure_column(
    temperature_K: float, pressure_Pa: float, elevation_change_m: float
) -> float:
    """Integrate rho*g over the shifted standard profile above the matching P.

    This is the source's pressure integral, not a new hydrostatic pressure
    profile fitted to the temperature offset. Negative height gives a signed
    negative integral. Both endpoints remain in the declared altitude range.
    """
    _positive(temperature_K, "temperature")
    _positive(pressure_Pa, "pressure")
    if not isfinite(elevation_change_m):
        raise ValueError("finite elevation change required")
    lower, upper = -610.0, 86000.0
    if (
        not standard_atmosphere(upper).pressure_Pa
        <= pressure_Pa
        <= standard_atmosphere(lower).pressure_Pa
    ):
        raise ValueError("pressure outside the declared standard-atmosphere altitude domain")

    def residual(height: float) -> float:
        return log(standard_atmosphere(height).pressure_Pa / pressure_Pa)

    for _ in range(16):
        midpoint = (lower + upper) / 2
        if residual(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    start = _engineering_root(
        residual, (lower + upper) / 2, lower, upper, name="standard_altitude", unit="m", scale=10000
    )
    finish = start + elevation_change_m
    if not -610 <= finish <= 86000:
        raise ValueError("pressure integral endpoint outside standard-atmosphere domain")
    offset = temperature_K - standard_atmosphere(start).temperature_K

    def integrand(height: float) -> float:
        state = standard_atmosphere(height, offset)
        return state.density_kg_m3 * state.gravity_m_s2

    value, _ = quad(integrand, start, finish, epsabs=1e-9, epsrel=1e-10)
    return float(value)


@dataclass(frozen=True)
class AirColumnMass:
    mass_kg_m2: float
    quadrature_error_kg_m2: float
    upper_elevation_m: float
    elevation_angle_degrees: float


def air_column_mass(
    density_profile: Callable[[float], float],
    elevation_angle_degrees: float,
    *,
    upper_elevation_m: float = 86400,
    planet_radius_m: float = 6371229,
    refractive_index: float = 1.000276,
) -> AirColumnMass:
    """Refracted spherical path integral; profile altitude domain belongs to caller."""
    if not isfinite(elevation_angle_degrees) or not 0 <= elevation_angle_degrees <= 90:
        raise ValueError("elevation angle must lie in [0,90] degrees")
    _positive(upper_elevation_m, "upper integration height")
    _positive(planet_radius_m, "planet radius")
    if not isfinite(refractive_index) or refractive_index < 1:
        raise ValueError("refractive index must be finite and at least one")
    surface = density_profile(0)
    _positive(surface, "surface density")
    cosine = cos(elevation_angle_degrees * pi / 180)

    def integrand(root_height: float) -> float:
        height = root_height**2
        density = density_profile(height)
        if not isfinite(density) or density < 0:
            raise ValueError("profile density must be finite and nonnegative")
        factor = (
            1
            - (1 + 2 * (refractive_index - 1) * (1 - density / surface))
            * (cosine / (1 + height / planet_radius_m)) ** 2
        )
        if factor <= 0:
            raise ValueError("profile/refraction produces a nonpropagating ray")
        return 2 * root_height * density / sqrt(factor)

    # h=u² removes the integrable horizon endpoint singularity.
    value, error = quad(integrand, 0, sqrt(upper_elevation_m), epsabs=1e-6, epsrel=1e-10, limit=200)
    return AirColumnMass(float(value), float(error), upper_elevation_m, elevation_angle_degrees)
