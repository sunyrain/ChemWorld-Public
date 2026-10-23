"""Solar civil-day events by solving the same ephemeris at the actual UTC time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from math import asin, cos, degrees, isfinite, radians, sin

from scipy.optimize import minimize_scalar

from chemworld.physchem.flow_meters import _engineering_root
from chemworld.physchem.solar_ephemeris import SolarEphemeris, _timestamp, solar_delta_t_seconds


@dataclass(frozen=True)
class SolarEvents:
    sunrise: datetime | None
    sunset: datetime | None
    transit: datetime | None
    status: str
    horizon_degrees: float

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "sunrise": None if self.sunrise is None else self.sunrise.isoformat(),
            "sunset": None if self.sunset is None else self.sunset.isoformat(),
            "transit": None if self.transit is None else self.transit.isoformat(),
            "status": self.status,
            "horizon_degrees": self.horizon_degrees,
        }


def solar_events(
    ephemeris: SolarEphemeris,
    moment: datetime,
    latitude_degrees: float,
    longitude_degrees: float,
    *,
    horizon_degrees: float = -0.8333,
    delta_t_seconds: float | None = None,
) -> SolarEvents:
    """Events in moment's local civil date/timezone; naive input means UTC.

    Default horizon follows SPA geocentric -0.8333degree convention. Numerical
    roots replace daily quadratic interpolation and retain polar/one-event days.
    Local days can be23/25hours. No reversed UTC-offset conversion is applied.
    """
    if not all(isfinite(v) for v in (latitude_degrees, longitude_degrees, horizon_degrees)):
        raise ValueError("finite solar-event geometry required")
    if not (
        -90 <= latitude_degrees <= 90
        and -180 <= longitude_degrees <= 180
        and -90 < horizon_degrees < 90
    ):
        raise ValueError("solar-event geometry outside domain")
    zone = moment.tzinfo or UTC
    start = datetime.combine(moment.date(), time(), tzinfo=zone)
    stop = datetime.combine(moment.date() + timedelta(days=1), time(), tzinfo=zone)
    origin, duration = _timestamp(start), _timestamp(stop) - _timestamp(start)
    utc_noon = (start + timedelta(hours=12)).astimezone(UTC)
    delta = (
        solar_delta_t_seconds(utc_noon.year, utc_noon.month)
        if delta_t_seconds is None
        else delta_t_seconds
    )
    if not isfinite(delta):
        raise ValueError("finite delta-T required")
    lat = radians(latitude_degrees)

    def geometry(fraction: float) -> tuple[float, float]:
        sun = ephemeris._geocentric(origin + fraction * duration, delta)
        hour = (sun.sidereal_angle + longitude_degrees - sun.right_ascension + 180) % 360 - 180
        dec = radians(sun.declination)
        elevation = degrees(asin(sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(radians(hour))))
        return elevation - horizon_degrees, hour

    coordinates = [i / 96 for i in range(97)]
    geometry_values = [geometry(x) for x in coordinates]
    # Refine extrema so brief polar-circle days/nights cannot fall between grid samples.
    extra = []
    for i in range(1, 96):
        a, b, c = (geometry_values[k][0] for k in (i - 1, i, i + 1))
        if (a < b > c) or (a > b < c):
            sign = -1 if a < b else 1
            result = minimize_scalar(
                lambda f, sign=sign: sign * geometry(float(f))[0],
                bounds=(coordinates[i - 1], coordinates[i + 1]),
                method="bounded",
                options={"xatol": 1e-13},
            )
            if not result.success:
                raise ValueError("solar-event extremum refinement failed")
            extra.append(float(result.x))
    coordinates = sorted(coordinates + extra)
    geometry_values = [geometry(x) for x in coordinates]
    rise = setting = transit = None
    for index in range(len(coordinates) - 1):
        left, right = coordinates[index : index + 2]
        for component in (0, 1):
            a, b = geometry_values[index][component], geometry_values[index + 1][component]
            if a == 0:
                root = left
            elif a * b < 0 and (component == 0 or abs(a - b) < 180):

                def residual(f: float, component: int = component) -> float:
                    return geometry(f)[component] / 360

                root = _engineering_root(
                    residual,
                    (left + right) / 2,
                    left,
                    right,
                    name="solar_event_day_fraction",
                    unit="dimensionless",
                    scale=1,
                )
            else:
                continue
            event = (
                datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=origin + root * duration)
            ).astimezone(zone)
            if component == 1:
                transit = event
            elif b > a:
                rise = event
            else:
                setting = event
    if rise is not None and setting is not None:
        status = "normal"
    elif rise is not None or setting is not None:
        status = "transition"
    else:
        status = "polar_day" if min(v[0] for v in geometry_values) >= 0 else "polar_night"
    return SolarEvents(rise, setting, transit, status, horizon_degrees)
