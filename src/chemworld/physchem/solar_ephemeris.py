"""Reda/Andreas solar position from caller-supplied VSOP87/nutation terms.

UTC is explicit (naive datetimes mean UTC). Positions preserve microseconds;
all angular outputs are degrees, equation of time seconds, distance meters.
No source backend or empirical table is bundled in production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import asin, atan, atan2, cos, degrees, isfinite, radians, sin, tan
from typing import TYPE_CHECKING, Any

from numpy.polynomial.polynomial import polyval

if TYPE_CHECKING:
    from chemworld.data.scientific_inputs import ScientificTable

SOLAR_PARAMETER_SHAPES = {
    **{f"L{i}": (n, 3) for i, n in enumerate((64, 34, 20, 7, 3, 1))},
    **{f"B{i}": (n, 3) for i, n in enumerate((5, 2))},
    **{f"R{i}": (n, 3) for i, n in enumerate((40, 10, 6, 2, 1))},
    "nutation": (63, 9),
}
ASTRONOMICAL_UNIT_M = 149597870700.0


def solar_delta_t_seconds(year: int, month: int) -> float:
    """Espenak/Meeus piecewise estimate TT-UT, model rather than observed UT1."""
    if not isinstance(year, int) or not isinstance(month, int) or not 1 <= month <= 12:
        raise ValueError("integer calendar year and month1..12 required")
    y = year + (month - 0.5) / 12
    # Published polynomial intervals: upper year, origin, scale, ascending coefficients.
    intervals = (
        (
            500,
            0,
            100,
            (10583.6, -1014.41, 33.78311, -5.952053, -0.1798452, 0.022174192, 0.0090316521),
        ),
        (
            1600,
            1000,
            100,
            (1574.2, -556.01, 71.23472, 0.319781, -0.8503463, -0.005050998, 0.0083572073),
        ),
        (1700, 1600, 1, (120, -0.9808, -0.01532, 1 / 7129)),
        (1800, 1700, 1, (8.83, 0.1603, -0.0059285, 0.00013336, -1 / 1174000)),
        (
            1860,
            1800,
            1,
            (
                13.72,
                -0.332447,
                0.0068612,
                0.0041116,
                -0.00037436,
                0.0000121272,
                -0.0000001699,
                0.000000000875,
            ),
        ),
        (1900, 1860, 1, (7.62, 0.5737, -0.251754, 0.01680668, -0.0004473624, 1 / 233174)),
        (1920, 1900, 1, (-2.79, 1.494119, -0.0598939, 0.0061966, -0.000197)),
        (1941, 1920, 1, (21.20, 0.84493, -0.076100, 0.0020936)),
        (1961, 1950, 1, (29.07, 0.407, -1 / 233, 1 / 2547)),
        (1986, 1975, 1, (45.45, 1.067, -1 / 260, -1 / 718)),
        (2005, 2000, 1, (63.86, 0.3345, -0.060374, 0.0017275, 0.000651814, 0.00002373599)),
        (2050, 2000, 1, (62.92, 0.32217, 0.005589)),
    )
    if year < -500 or year >= 2150:
        return -20 + 32 * ((y - 1820) / 100) ** 2
    if year >= 2050:
        return -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y)
    for upper, origin, scale, coefficients in intervals:
        if year < upper:
            return float(polyval((y - origin) / scale, coefficients))
    raise AssertionError("unreachable delta-T interval")


def _utc(moment: datetime) -> datetime:
    value = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    if value.year > 6000:
        raise ValueError("solar ephemeris calendar domain ends at year6000")
    return value


def _timestamp(moment: datetime) -> float:
    return (_utc(moment) - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()


class SolarParameters:
    """1152 encoded coefficient values on the common scientific-input contract.

    Flatten each SOLAR_PARAMETER_SHAPES array row-major into index/value cells.
    Nutation rows are five angular multipliers followed by A,B,C,D amplitudes.
    """

    def __init__(self, table: ScientificTable) -> None:
        if table.kind != "parameters":
            raise ValueError("solar scientific parameter table required")
        groups: dict[str, dict[int, float]] = {}
        for row in table.rows:
            index, value = (
                row.values["index"].to("dimensionless"),
                row.values["value"].to("dimensionless"),
            )
            if index.quality != "valid" or value.quality != "valid":
                raise ValueError("valid solar parameter cells required")
            i, v = float(index.value), float(value.value)
            if not isfinite(i) or not i.is_integer() or not isfinite(v):
                raise ValueError("finite coefficient and integer parameter index required")
            group = groups.setdefault(row.group_id, {})
            if int(i) in group:
                raise ValueError("duplicate solar coefficient index")
            group[int(i)] = v
        if set(groups) != set(SOLAR_PARAMETER_SHAPES):
            raise ValueError("complete VSOP87 and nutation arrays required")
        self.arrays = {}
        for name, (rows, width) in SOLAR_PARAMETER_SHAPES.items():
            if set(groups[name]) != set(range(rows * width)):
                raise ValueError(f"incomplete solar array: {name}")
            self.arrays[name] = tuple(
                tuple(groups[name][r * width + c] for c in range(width)) for r in range(rows)
            )
        self.table = table

    def to_dict(self) -> dict[str, Any]:
        return self.table.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolarParameters:
        from chemworld.data.scientific_inputs import ScientificTable

        return cls(ScientificTable.from_dict(data))


@dataclass(frozen=True)
class SolarPosition:
    apparent_zenith_degrees: float
    zenith_degrees: float
    apparent_elevation_degrees: float
    elevation_degrees: float
    azimuth_degrees: float
    equation_of_time_seconds: float
    earth_sun_distance_m: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class _GeocentricSun:
    right_ascension: float
    declination: float
    sidereal_angle: float
    radius_au: float
    equation_of_time_seconds: float


class SolarEphemeris:
    def __init__(self, parameters: SolarParameters) -> None:
        self.parameters = parameters

    def _geocentric(self, timestamp: float, delta_t: float) -> _GeocentricSun:
        # Subtract the epoch before division: adding/subtracting a 2.4million-day
        # Julian date loses tens of microseconds and defeats event root solving.
        days_since_epoch = (timestamp - 946728000.0) / 86400
        century = days_since_epoch / 36525
        ephemeris_century = (days_since_epoch + delta_t / 86400) / 36525
        millennium = ephemeris_century / 10
        arrays = self.parameters.arrays

        def series(prefix: str, count: int) -> float:
            terms = [
                sum(a * cos(b + c * millennium) for a, b, c in arrays[f"{prefix}{i}"])
                for i in range(count)
            ]
            return float(polyval(millennium, terms)) * 1e-8

        longitude = degrees(series("L", 6)) % 360 + 180
        latitude = -series("B", 2)
        distance = series("R", 5)
        fundamental_coefficients = (
            (297.85036, 445267.11148, -0.0019142, 1 / 189474),
            (357.52772, 35999.05034, -0.0001603, -1 / 300000),
            (134.96298, 477198.867398, 0.0086972, 1 / 56250),
            (93.27191, 483202.017538, -0.0036825, 1 / 327270),
            (125.04452, -1934.136261, 0.0020708, 1 / 450000),
        )
        fundamental = [
            radians(float(polyval(ephemeris_century, coefficients)))
            for coefficients in fundamental_coefficients
        ]
        dlong = dobliq = 0.0
        for row in arrays["nutation"]:
            phase = sum(a * b for a, b in zip(row[:5], fundamental, strict=True))
            dlong += (row[5] + row[6] * ephemeris_century) * sin(phase)
            dobliq += (row[7] + row[8] * ephemeris_century) * cos(phase)
        dlong /= 36000000
        dobliq /= 36000000
        mean_obliquity = float(
            polyval(
                millennium / 10,
                (
                    84381.448,
                    -4680.93,
                    -1.55,
                    1999.25,
                    -51.38,
                    -249.67,
                    -39.05,
                    7.12,
                    27.87,
                    5.79,
                    2.45,
                ),
            )
        )
        obliquity = radians(mean_obliquity / 3600 + dobliq)
        apparent_longitude = radians(longitude + dlong - 20.4898 / (3600 * distance))
        ascension = (
            degrees(
                atan2(
                    sin(apparent_longitude) * cos(obliquity) - tan(latitude) * sin(obliquity),
                    cos(apparent_longitude),
                )
            )
            % 360
        )
        declination = degrees(
            asin(
                sin(latitude) * cos(obliquity)
                + cos(latitude) * sin(obliquity) * sin(apparent_longitude)
            )
        )
        sidereal = (
            280.46061837
            + 360.98564736629 * days_since_epoch
            + 0.000387933 * century**2
            - century**3 / 38710000
        ) % 360 + dlong * cos(obliquity)
        mean_longitude = float(
            polyval(
                millennium,
                (280.4664567, 360007.6982779, 0.03032028, 1 / 49931, -1 / 15300, -1 / 2000000),
            )
        )
        angle = (mean_longitude - 0.0057183 - ascension + dlong * cos(obliquity) + 180) % 360 - 180
        return _GeocentricSun(ascension, declination, sidereal, distance, angle * 240)

    def earth_sun_distance_m(
        self, moment: datetime, *, delta_t_seconds: float | None = None
    ) -> float:
        utc = _utc(moment)
        delta = (
            solar_delta_t_seconds(utc.year, utc.month)
            if delta_t_seconds is None
            else delta_t_seconds
        )
        if not isfinite(delta):
            raise ValueError("finite delta-T required")
        return self._geocentric(_timestamp(utc), delta).radius_au * ASTRONOMICAL_UNIT_M

    def position(
        self,
        moment: datetime,
        latitude_degrees: float,
        longitude_degrees: float,
        *,
        elevation_m: float = 0,
        temperature_K: float = 298.15,
        pressure_Pa: float = 101325,
        atmospheric_refraction_degrees: float = 0.5667,
        delta_t_seconds: float | None = None,
    ) -> SolarPosition:
        values = (
            latitude_degrees,
            longitude_degrees,
            elevation_m,
            temperature_K,
            pressure_Pa,
            atmospheric_refraction_degrees,
        )
        if not all(isfinite(v) for v in values):
            raise ValueError("finite solar observing inputs required")
        if not (-90 <= latitude_degrees <= 90 and -180 <= longitude_degrees <= 180):
            raise ValueError("geographic solar coordinates out of range")
        if temperature_K <= 0.15 or pressure_Pa < 0 or atmospheric_refraction_degrees < 0:
            raise ValueError(
                "positive atmospheric temperature and nonnegative pressure/refraction required"
            )
        utc = _utc(moment)
        delta = (
            solar_delta_t_seconds(utc.year, utc.month)
            if delta_t_seconds is None
            else delta_t_seconds
        )
        if not isfinite(delta):
            raise ValueError("finite delta-T required")
        sun = self._geocentric(_timestamp(utc), delta)
        lat, dec = radians(latitude_degrees), radians(sun.declination)
        hour = radians((sun.sidereal_angle + longitude_degrees - sun.right_ascension) % 360)
        reduced_latitude = atan(0.99664719 * tan(lat))
        x = cos(reduced_latitude) + elevation_m / 6378140 * cos(lat)
        y = 0.99664719 * sin(reduced_latitude) + elevation_m / 6378140 * sin(lat)
        parallax = sin(radians(8.794 / (3600 * sun.radius_au)))
        correction = atan2(-x * parallax * sin(hour), cos(dec) - x * parallax * cos(hour))
        local_declination = atan2(
            (sin(dec) - y * parallax) * cos(correction), cos(dec) - x * parallax * cos(hour)
        )
        local_hour = hour - correction
        elevation = degrees(
            asin(
                sin(lat) * sin(local_declination)
                + cos(lat) * cos(local_declination) * cos(local_hour)
            )
        )
        refraction = 0.0
        if elevation >= -(0.26667 + atmospheric_refraction_degrees) and pressure_Pa > 0:
            refraction = (
                pressure_Pa
                / 101000
                * 283
                / (temperature_K - 0.15)
                * 1.02
                / (60 * tan(radians(elevation + 10.3 / (elevation + 5.11))))
            )
        azimuth = (
            degrees(
                atan2(
                    sin(local_hour), cos(local_hour) * sin(lat) - tan(local_declination) * cos(lat)
                )
            )
            + 180
        ) % 360
        return SolarPosition(
            90 - elevation - refraction,
            90 - elevation,
            elevation + refraction,
            elevation,
            azimuth,
            sun.equation_of_time_seconds,
            sun.radius_au * ASTRONOMICAL_UNIT_M,
        )
