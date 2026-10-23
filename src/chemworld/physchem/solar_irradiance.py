"""Explicit local clear-sky irradiation and surface energy accounting.

Ineichen/Perez with Kasten-Young relative air mass and isotropic sky. Linke
turbidity and atmospheric state are caller observations/parameters. No implicit
pvlib database lookup or hidden empirical-atmosphere evaluation is performed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import cos, exp, isfinite, pi, radians, sin

from chemworld.physchem.solar_ephemeris import SolarEphemeris, SolarPosition, _utc


def extraterrestrial_irradiance(
    moment: datetime,
    *,
    solar_constant_W_m2: float = 1366.1,
    method: str = "spencer",
    ephemeris: SolarEphemeris | None = None,
) -> float:
    if not isfinite(solar_constant_W_m2) or solar_constant_W_m2 <= 0:
        raise ValueError("positive finite solar constant required")
    day = _utc(moment).timetuple().tm_yday
    angle = 2 * pi * (day - 1) / 365
    if method == "spencer":
        factor = (
            1.00011
            + 0.034221 * cos(angle)
            + 0.00128 * sin(angle)
            + 0.000719 * cos(2 * angle)
            + 0.000077 * sin(2 * angle)
        )
    elif method == "asce":
        factor = 1 + 0.033 * cos(2 * pi * day / 365)
    elif method == "nrel":
        if ephemeris is None:
            raise ValueError("NREL distance method requires explicit solar ephemeris")
        factor = (149597870700 / ephemeris.earth_sun_distance_m(moment)) ** 2
    else:
        raise ValueError("extraterrestrial method must be spencer, asce or nrel")
    return solar_constant_W_m2 * factor


@dataclass(frozen=True)
class SolarIrradiance:
    global_horizontal_W_m2: float
    direct_normal_W_m2: float
    diffuse_horizontal_W_m2: float
    plane_total_W_m2: float
    plane_direct_W_m2: float
    plane_diffuse_W_m2: float
    plane_sky_W_m2: float
    plane_ground_W_m2: float
    extraterrestrial_W_m2: float
    convention: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)

    def absorbed_power_w(self, area_m2: float, absorptivity: float) -> float:
        if (
            not isfinite(area_m2)
            or area_m2 < 0
            or not isfinite(absorptivity)
            or not 0 <= absorptivity <= 1
        ):
            raise ValueError("finite nonnegative area and absorptivity0..1 required")
        return area_m2 * absorptivity * self.plane_total_W_m2


def clear_sky_irradiance(
    position: SolarPosition,
    moment: datetime,
    *,
    pressure_Pa: float,
    elevation_m: float,
    linke_turbidity: float,
    surface_tilt_degrees: float,
    surface_azimuth_degrees: float,
    albedo: float = 0.25,
    solar_constant_W_m2: float = 1366.1,
    extra_method: str = "spencer",
    ephemeris: SolarEphemeris | None = None,
    convention: str = "seasonal",
    perez_enhancement: bool = True,
) -> SolarIrradiance:
    """Source uses constant solar flux even after calculating seasonal correction.

    Default seasonal applies the chosen correction; source_constant explicitly
    reproduces source behavior. External state enters through position and P.
    The empirical domain is0..8000m and Linke>=1; no weather/cloud attenuation.
    """
    values = (
        pressure_Pa,
        elevation_m,
        linke_turbidity,
        surface_tilt_degrees,
        surface_azimuth_degrees,
        albedo,
        position.apparent_zenith_degrees,
        position.azimuth_degrees,
    )
    if not all(isfinite(v) for v in values) or pressure_Pa <= 0:
        raise ValueError("finite geometry and positive pressure required")
    if not (
        0 <= elevation_m <= 8000
        and linke_turbidity >= 1
        and 0 <= albedo <= 1
        and 0 <= surface_tilt_degrees <= 180
        and 0 <= position.apparent_zenith_degrees <= 180
    ):
        raise ValueError("clear-sky empirical inputs outside declared domain")
    if convention not in {"seasonal", "source_constant"}:
        raise ValueError("explicit seasonal or source_constant convention required")
    extra = extraterrestrial_irradiance(
        moment, solar_constant_W_m2=solar_constant_W_m2, method=extra_method, ephemeris=ephemeris
    )
    flux = extra if convention == "seasonal" else solar_constant_W_m2
    zenith, tilt = radians(position.apparent_zenith_degrees), radians(surface_tilt_degrees)
    cosine = cos(zenith)
    if cosine <= 0:
        return SolarIrradiance(0, 0, 0, 0, 0, 0, 0, 0, extra, convention)
    relative_mass = 1 / (
        cosine + 0.50572 * (96.07995 - position.apparent_zenith_degrees) ** -1.6364
    )
    air_mass = relative_mass * pressure_Pa / 101325
    altitude_factor = exp(-elevation_m / 8000)
    scale = exp(-elevation_m / 1250)
    attenuation = (
        -(0.0387 + 3.92e-5 * elevation_m)
        * air_mass
        * (altitude_factor + scale * (linke_turbidity - 1))
    )
    if perez_enhancement:
        attenuation += 0.01 * air_mass**1.8
    global_horizontal = (0.868 + 5.09e-5 * elevation_m) * flux * cosine * exp(attenuation)
    beam = (0.664 + 0.163 / altitude_factor) * flux * exp(-0.09 * air_mass * (linke_turbidity - 1))
    correction = 1 - (0.1 - 0.2 * exp(-linke_turbidity)) / (0.1 + 0.882 / altitude_factor)
    direct_normal = min(beam, global_horizontal * correction / cosine)
    diffuse_horizontal = global_horizontal - direct_normal * cosine
    projection = cos(tilt) * cosine + sin(tilt) * sin(zenith) * cos(
        radians(position.azimuth_degrees - surface_azimuth_degrees)
    )
    direct = direct_normal * max(0.0, projection)
    sky = diffuse_horizontal * (1 + cos(tilt)) / 2
    ground = global_horizontal * albedo * (1 - cos(tilt)) / 2
    return SolarIrradiance(
        global_horizontal,
        direct_normal,
        diffuse_horizontal,
        direct + sky + ground,
        direct,
        sky + ground,
        sky,
        ground,
        extra,
        convention,
    )
