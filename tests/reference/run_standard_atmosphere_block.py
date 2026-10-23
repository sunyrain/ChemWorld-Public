"""Fixed standard atmosphere comparison; reference imports remain test-only."""

import argparse
import json
import time
from math import exp, isclose
from pathlib import Path

import fluids.atmosphere as ref

from chemworld.physchem.standard_atmosphere import (
    STANDARD_EARTH_RADIUS_M,
    air_column_mass,
    standard_air_properties,
    standard_atmosphere,
    standard_pressure_column,
)


def run():
    started = time.perf_counter()
    rows = []
    failures = []
    comparisons = 0
    passed = 0

    def check(key, values):
        nonlocal comparisons, passed
        rows.append(key)
        for label, actual, expected in values:
            comparisons += 1
            if isclose(actual, expected, rel_tol=3e-10, abs_tol=1e-7):
                passed += 1
            else:
                failures.append(
                    {"input": key, "label": label, "native": actual, "reference": expected}
                )

    heights = [
        -610,
        0,
        100,
        1000,
        5000,
        10000,
        11000,
        12000,
        19000,
        20000,
        21000,
        32000,
        40000,
        47000,
        51000,
        60000,
        71000,
        75000,
        80000,
        84852,
        86000,
    ]
    radius = STANDARD_EARTH_RADIUS_M
    heights += [
        height * radius / (radius - height) + delta
        for height in (11000, 20000, 32000, 47000, 51000, 71000, 84852)
        for delta in (-0.001, 0, 0.001)
    ]
    properties = {
        "temperature_K": "T",
        "pressure_Pa": "P",
        "density_kg_m3": "rho",
        "viscosity_Pa_s": "mu",
        "conductivity_W_m_K": "k",
        "sound_speed_m_s": "v_sonic",
        "gravity_m_s2": "g",
        "geopotential_height_m": "H",
    }
    for height in heights:
        for offset in (-10, 0, 15):
            native = standard_atmosphere(height, offset)
            reference = ref.ATMOSPHERE_1976(height, offset)
            check(
                ["state", height, offset],
                [
                    (name, getattr(native, name), getattr(reference, attr))
                    for name, attr in properties.items()
                ],
            )
    for temperature in (200, 250, 300, 350):
        native = standard_air_properties(temperature, 101325)
        for name, value in (
            ("density_kg_m3", ref.ATMOSPHERE_1976.density(temperature, 101325)),
            ("viscosity_Pa_s", ref.ATMOSPHERE_1976.viscosity(temperature)),
            ("conductivity_W_m_K", ref.ATMOSPHERE_1976.thermal_conductivity(temperature)),
            ("sound_speed_m_s", ref.ATMOSPHERE_1976.sonic_velocity(temperature)),
        ):
            check(["property", temperature, name], [(name, native[name], value)])
    for height in (0, 1000, 10000, 50000):
        for offset in (-10, 0, 10):
            state = standard_atmosphere(height, offset)
            for delta in (-100, 100):
                actual = standard_pressure_column(state.temperature_K, state.pressure_Pa, delta)
                expected = ref.ATMOSPHERE_1976.pressure_integral(
                    state.temperature_K, state.pressure_Pa, delta
                )
                check(["pressure", height, offset, delta], [("pressure_Pa", actual, expected)])
    for scale in (7000, 9000):

        def density(h, scale=scale):
            return 1.225 * exp(-h / scale)

        for angle in (0, 1, 5, 10, 30, 45, 60, 90):
            actual = air_column_mass(density, angle).mass_kg_m2
            values = [("column", actual, ref.airmass(density, angle))]
            if angle == 90:
                values.append(("analytic", actual, 1.225 * scale * (1 - exp(-86400 / scale))))
            check(["column", scale, angle], values)
    assert len(rows) == 182 and comparisons == 1066
    return {
        "inputs": len(rows),
        "comparisons": comparisons,
        "passed": passed,
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "input_inventory": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_inventory", "failures")}))
    print("failures", len(result["failures"]))
