"""Independent unit and actual-ephemeris roots for retained source differences."""

import argparse
import json
from datetime import UTC, datetime
from math import asin, cos, degrees, isclose, radians, sin
from pathlib import Path

import fluids.atmosphere as ref
import mpmath as mp
from fluids.optional import spa
from run_solar_block import DATES
from solar_source_parameters import source_parameters

from chemworld.physchem.solar_ephemeris import SolarEphemeris, solar_delta_t_seconds


def run(original):
    mp.mp.dps = 80
    model = SolarEphemeris(source_parameters())
    rows = []
    for date in DATES:
        moment = datetime(*date, 12, 30, tzinfo=UTC)
        seconds = (moment - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()
        delta = solar_delta_t_seconds(moment.year, moment.month)
        millennium = ((mp.mpf(seconds) - 946728000) / 86400 + mp.mpf(delta) / 86400) / 365250
        radius = sum(
            sum(
                mp.mpf(str(a)) * mp.cos(mp.mpf(str(b)) + mp.mpf(str(c)) * millennium)
                for a, b, c in model.parameters.arrays[f"R{i}"]
            )
            * millennium**i
            for i in range(5)
        ) * mp.mpf("1e-8")
        actual = model.earth_sun_distance_m(moment)
        expected = float(radius * 149597870700)
        converted = ref.earthsun_distance(moment) * 149597870700 / 149597870691
        rows.append(
            {
                "input": moment.isoformat(),
                "actual": actual,
                "independent": expected,
                "source_AU_corrected": converted,
                "checks": [
                    isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-3),
                    isclose(actual, converted, rel_tol=1e-11, abs_tol=1e-3),
                ],
            }
        )
    for failure in original["failures"]:
        if failure["property"] not in ("sunrise", "sunset"):
            continue
        _, date, lat, lon = failure["input"]
        moment = datetime.fromisoformat(date)
        delta = solar_delta_t_seconds(moment.year, moment.month)

        def residual(timestamp, delta=delta, lon=lon, lat=lat):
            sidereal, alpha, declination = spa.solar_position(
                timestamp, 0, 0, 0, 0, 0, delta, 0, sst=True
            )
            hour = radians((sidereal + lon - alpha) % 360)
            return (
                degrees(
                    asin(
                        sin(radians(lat)) * sin(radians(declination))
                        + cos(radians(lat)) * cos(radians(declination)) * cos(hour)
                    )
                )
                + 0.8333
            )

        lower, upper = failure["reference"] - 3600, failure["reference"] + 3600
        flo, fhi = residual(lower), residual(upper)
        assert flo * fhi < 0
        for _ in range(80):
            middle = (lower + upper) / 2
            if middle in (lower, upper):
                break
            value = residual(middle)
            if value * flo > 0:
                lower, flo = middle, value
            else:
                upper = middle
        independent = (lower + upper) / 2
        error = residual(failure["native"])
        rows.append(
            {
                "input": failure["input"],
                "event": failure["property"],
                "native": failure["native"],
                "independent": independent,
                "source_residual": error,
                "checks": [abs(failure["native"] - independent) <= 0.001, abs(error) <= 5e-7],
            }
        )
    assert len(rows) == 21
    return {
        "inputs": len(rows),
        "comparisons": 42,
        "passed": sum(sum(row["checks"]) for row in rows),
        "rows": rows,
        "failures": [row for row in rows if not all(row["checks"])],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.original.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({k: v for k, v in result.items() if k not in ("rows", "failures")})
