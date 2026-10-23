"""Fixed solar scientific matrix, source differences retained by input."""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta, timezone
from itertools import product
from math import cos, isclose, isfinite, radians
from pathlib import Path

import fluids.atmosphere as ref
from fluids.optional import spa
from solar_source_parameters import source_parameters

from chemworld.physchem.solar_ephemeris import (
    SolarEphemeris,
    SolarParameters,
    SolarPosition,
    solar_delta_t_seconds,
)
from chemworld.physchem.solar_events import solar_events
from chemworld.physchem.solar_irradiance import clear_sky_irradiance

LOCATIONS = ((-80, -150), (-45, -70), (0, 0), (39.742476, -105.1786), (65, 110), (85, 179))
DATES = (
    (1000, 1, 1),
    (1600, 7, 1),
    (1900, 2, 1),
    (2000, 1, 1),
    (2050, 7, 1),
    (2024, 1, 1),
    (2024, 3, 20),
    (2024, 6, 21),
    (2024, 9, 22),
    (2024, 12, 21),
)
SEASONS = DATES[6:]


def run():
    start = time.perf_counter()
    raw = source_parameters()
    parameters = SolarParameters.from_dict(json.loads(json.dumps(raw.to_dict())))
    model = SolarEphemeris(parameters)
    inputs = comparisons = passed = 0
    failures, exceptions = [], []

    def check(case, name, actual, expected, rtol=1e-8, atol=1e-7):
        nonlocal comparisons, passed
        comparisons += 1
        success = (
            (actual is None and expected is None)
            if actual is None or expected is None
            else isclose(actual, expected, rel_tol=rtol, abs_tol=atol)
        )
        if success:
            passed += 1
        else:
            failures.append(
                {"input": case, "property": name, "native": actual, "reference": expected}
            )

    for row, old in zip(parameters.table.rows, raw.table.rows, strict=True):
        inputs += 1
        check(
            [row.row_id], "coefficient", row.values["value"].value, old.values["value"].value, 0, 0
        )
    for date, (lat, lon), (z, t, p) in product(
        DATES, LOCATIONS, ((0, 298.15, 101325), (2000, 280, 80000))
    ):
        inputs += 1
        moment = datetime(*date, 12, 30, tzinfo=UTC)
        case = [moment.isoformat(), lat, lon, z, t, p]
        try:
            native = model.position(moment, lat, lon, elevation_m=z, temperature_K=t, pressure_Pa=p)
            expected = [
                *ref.solar_position(moment, lat, lon, Z=z, T=t, P=p),
                ref.earthsun_distance(moment),
            ]
            for index, (name, value) in enumerate(native.to_dict().items()):
                check(
                    case,
                    name,
                    value,
                    expected[index],
                    1e-11 if index == 6 else 1e-8,
                    1e-3 if index == 6 else 1e-7,
                )
        except Exception as error:
            exceptions.append({"input": case, "error": repr(error)})
    inputs += 1
    golden = model.position(
        datetime(2003, 10, 17, 19, 30, 30, tzinfo=UTC),
        39.742476,
        -105.1786,
        elevation_m=1830.14,
        temperature_K=284.15,
        pressure_Pa=82000,
        delta_t_seconds=67,
    )
    check(["NRELpublished"], "zenith", golden.apparent_zenith_degrees, 50.111622, 0, 1e-6)
    check(["NRELpublished"], "azimuth", golden.azimuth_degrees, 194.340241, 0, 1e-6)
    for year, month in product(
        (
            -600,
            -500,
            0,
            500,
            1600,
            1700,
            1800,
            1860,
            1900,
            1920,
            1941,
            1961,
            1986,
            2005,
            2050,
            2150,
        ),
        (1, 6, 12),
    ):
        inputs += 1
        check(
            ["deltaT", year, month],
            "seconds",
            solar_delta_t_seconds(year, month),
            spa.calculate_deltat(year, month),
            1e-12,
            1e-9,
        )
    for date, hours in product(SEASONS, (8, -7)):
        inputs += 1
        moment = datetime(*date, 23, 30, tzinfo=UTC)
        local = moment.astimezone(timezone(timedelta(hours=hours)))
        a, b = model.position(local, 35, 110), model.position(moment, 35, 110)
        for (name, x), y in zip(a.to_dict().items(), b.to_dict().values(), strict=True):
            check(["timezone", moment.isoformat(), hours], name, x, y)
    print("solar positions/time complete", flush=True)
    for date, (lat, lon) in product(SEASONS, LOCATIONS):
        inputs += 1
        moment = datetime(*date, tzinfo=UTC)
        case = ["events", moment.isoformat(), lat, lon]
        try:
            actual = solar_events(model, moment, lat, lon)
            midnight = (moment - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()
            delta = solar_delta_t_seconds(moment.year, moment.month)
            candidates = [
                spa.transit_sunrise_sunset(midnight + shift * 86400, lat, lon, delta)
                for shift in (-1, 0, 1)
            ]
            for name, idx in (("sunrise", 1), ("sunset", 2), ("transit", 0)):
                reference = [
                    values[idx]
                    for values in candidates
                    if isfinite(values[idx]) and midnight <= values[idx] < midnight + 86400
                ]
                expected = reference[0] if reference else None
                native = getattr(actual, name)
                timestamp = (
                    None
                    if native is None
                    else (native - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()
                )
                check(case, name, timestamp, expected, 0, 10)
                if native is None:
                    check(
                        case,
                        name + "_status",
                        actual.status in ("polar_day", "polar_night", "transition"),
                        True,
                        0,
                        0,
                    )
                else:
                    sun = model._geocentric(timestamp, delta)
                    hour = (sun.sidereal_angle + lon - sun.right_ascension + 180) % 360 - 180
                    if name == "transit":
                        residual = hour
                    else:
                        from math import asin, degrees, sin

                        residual = (
                            degrees(
                                asin(
                                    sin(radians(lat)) * sin(radians(sun.declination))
                                    + cos(radians(lat))
                                    * cos(radians(sun.declination))
                                    * cos(radians(hour))
                                )
                            )
                            + 0.8333
                        )
                    check(case, name + "_residual", residual, 0, 0, 1e-7)
        except Exception as error:
            exceptions.append({"input": case, "error": repr(error)})
    print("solar civil-day events complete", flush=True)
    for zenith, tilt, z, linke, date in product(
        (0, 30, 60, 85, 95, 120),
        (0, 30, 90, 180),
        (0, 2000),
        (2, 3, 6),
        ((2024, 1, 1), (2024, 7, 1)),
    ):
        inputs += 1
        moment = datetime(*date, 12, tzinfo=UTC)
        case = ["irradiance", zenith, tilt, z, linke, moment.isoformat()]
        try:
            position = SolarPosition(zenith, zenith, 90 - zenith, 90 - zenith, 180, 0, 149597870700)
            native = clear_sky_irradiance(
                position,
                moment,
                pressure_Pa=101325,
                elevation_m=z,
                linke_turbidity=linke,
                surface_tilt_degrees=tilt,
                surface_azimuth_degrees=180,
                convention="source_constant",
            )
            seasonal = clear_sky_irradiance(
                position,
                moment,
                pressure_Pa=101325,
                elevation_m=z,
                linke_turbidity=linke,
                surface_tilt_degrees=tilt,
                surface_azimuth_degrees=180,
            )
            expected = ref.solar_irradiation(
                0,
                0,
                z,
                moment,
                tilt,
                180,
                T=298.15,
                P=101325,
                linke_turbidity=linke,
                cache={"zenith": zenith, "apparent_zenith": zenith, "azimuth": 180},
            )
            extra = ref._get_extra_radiation_shim(moment.timetuple().tm_yday, solar_constant=1366.1)
            for i, name in enumerate(
                (
                    "plane_total_W_m2",
                    "plane_direct_W_m2",
                    "plane_diffuse_W_m2",
                    "plane_sky_W_m2",
                    "plane_ground_W_m2",
                )
            ):
                check(case, name, getattr(native, name), expected[i], 1e-10, 1e-8)
                check(
                    case,
                    name + "_seasonal",
                    getattr(seasonal, name),
                    expected[i] * extra / 1366.1,
                    1e-10,
                    1e-8,
                )
            check(case, "extra", native.extraterrestrial_W_m2, extra, 1e-10, 1e-8)
            check(
                case,
                "horizontal_closure",
                native.global_horizontal_W_m2,
                native.direct_normal_W_m2 * max(0, cos(radians(zenith)))
                + native.diffuse_horizontal_W_m2,
                1e-10,
                1e-8,
            )
            check(
                case,
                "plane_closure",
                native.plane_total_W_m2,
                native.plane_direct_W_m2 + native.plane_diffuse_W_m2,
                1e-10,
                1e-8,
            )
        except Exception as error:
            exceptions.append({"input": case, "error": repr(error)})
    assert inputs == 1641
    return {
        "inputs": inputs,
        "planned_comparisons": 5986,
        "comparisons": comparisons,
        "passed": passed,
        "failures": failures,
        "exceptions": exceptions,
        "elapsed_s": time.perf_counter() - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({k: v for k, v in result.items() if k not in ("failures", "exceptions")})
    print("differences", len(result["failures"]), "exceptions", len(result["exceptions"]))
