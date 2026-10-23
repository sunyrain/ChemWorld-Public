"""Prospective 342-input/447-comparison isolated-particle development block."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

import fluids
import fluids.drag as reference
import mpmath as mp

from chemworld.physchem.particle_drag import (
    PARTICLE_DRAG_RANGES,
    ParticleDragSpec,
    ParticleMotionState,
    advance_particle_motion,
    particle_drag_coefficient,
    particle_terminal_velocity,
    time_to_stokes_terminal,
)
from chemworld.physchem.transport import FluidState


def inputs():
    for method, reynolds in product(PARTICLE_DRAG_RANGES, (0.1, 1, 10, 100, 1000, 1e4)):
        yield {"family": "coefficient", "method": method, "Re": reynolds}
    for method, boundaries in (
        ("Clift", (0.01, 20, 260, 1500, 12000, 44000, 338000, 400000)),
        ("Morsi_Alexander", (0.1, 1, 10, 100, 1000, 5000, 10000)),
        ("auto", (0.01, 0.1, 212963.26847812787, 1e6)),
    ):
        for boundary, factor in product(boundaries, (0.999999, 1, 1.000001)):
            yield {"family": "boundary", "method": method, "Re": boundary * factor}
    for shape, area, reynolds in product((0.5, 0.8, 1), (0.6, 1, 1.5), (1, 100, 1000)):
        yield {"family": "shape", "shape": shape, "area": area, "Re": reynolds}
    for method, diameter in product(PARTICLE_DRAG_RANGES, (0.0001, 0.001)):
        yield {"family": "terminal", "method": method, "D": diameter, "rhop": 3000}
    for density, diameter in product((500, 1000, 3000), (1e-6, 1e-4, 1e-3)):
        yield {"family": "terminal", "method": "auto", "D": diameter, "rhop": density}
    for diameter, initial, tolerance in product(
        (1e-6, 1e-5, 1e-4), (0, 0.01, 0.1), (1e-3, 1e-6, 1e-9)
    ):
        yield {"family": "time", "D": diameter, "V0": initial, "tol": tolerance}
    for method, diameter, duration, initial in product(
        ("auto", "Stokes"), (1e-6, 1e-4, 1e-3), (0.001, 0.1, 1), (0, 0.1)
    ):
        yield {"family": "motion", "method": method, "D": diameter, "t": duration, "V0": initial}
    for diameter, duration, initial in product((1e-6, 1e-4, 1e-3), (0.001, 0.1, 1), (0, 0.1)):
        yield {
            "family": "independent",
            "method": "Stokes",
            "D": diameter,
            "t": duration,
            "V0": initial,
        }


def compare(quantity, native, expected, rtol=2e-9, atol=1e-10):
    return {
        "quantity": quantity,
        "native": float(native),
        "reference": float(expected),
        "rtol": rtol,
        "atol": atol,
        "passed": isclose(native, expected, rel_tol=rtol, abs_tol=atol),
    }


def evaluate(case):
    family = case["family"]
    if family in ("coefficient", "boundary"):
        method, reynolds = case["method"], case["Re"]
        native = particle_drag_coefficient(
            reynolds,
            method=method,
            check_recommended_ranges=False,
            cap_barati_high_at_million=method == "auto" and reynolds > 1e6,
        )
        expected = (
            reference.drag_sphere(reynolds)
            if method == "auto"
            else getattr(reference, method)(reynolds)
        )
        return [compare("Cd", native, expected)]
    if family == "shape":
        native = particle_drag_coefficient(
            case["Re"],
            method="Song_Xu",
            sphericity=case["shape"],
            projected_area_ratio=case["area"],
            check_recommended_ranges=False,
        )
        return [compare("Cd", native, reference.Song_Xu(case["Re"], case["shape"], case["area"]))]
    spec = ParticleDragSpec(
        case["D"],
        case.get("rhop", 3000),
        FluidState(1000, 0.001),
        method=case.get("method", "Stokes"),
        check_recommended_ranges=False,
    )
    common = {"D": spec.diameter_m, "rhop": spec.particle_density_kg_m3, "rho": 1000, "mu": 0.001}
    if family == "terminal":
        native = particle_terminal_velocity(spec)
        expected = reference.v_terminal(
            **common, Method=None if spec.method == "auto" else spec.method
        )
        force = abs(spec.particle_mass_kg * spec.driving_acceleration_m_s2)
        return [
            compare("velocity", native.velocity_m_s, expected),
            compare(
                "normalized_force_residual",
                native.force_residual_N / force if force else 0,
                0,
                0,
                1e-9,
            ),
        ]
    if family == "time":
        return [
            compare(
                "relaxation_time",
                time_to_stokes_terminal(spec, case["V0"], relative_tolerance=case["tol"]),
                reference.time_v_terminal_Stokes(**common, V0=case["V0"], tol=case["tol"]),
                1e-7,
                1e-12,
            )
        ]
    result = advance_particle_motion(spec, ParticleMotionState(velocity_m_s=case["V0"]), case["t"])
    if family == "motion":
        expected = reference.integrate_drag_sphere(
            **common,
            t=case["t"],
            V=case["V0"],
            Method=None if spec.method == "auto" else spec.method,
            distance=True,
        )
        rtol, atol = 1e-5, 1e-9
    else:
        with mp.workdps(80):
            diameter, duration, initial = (mp.mpf(str(case[key])) for key in ("D", "t", "V0"))
            tau = mp.mpf(3000) * diameter**2 / (18 * mp.mpf("0.001"))
            terminal = mp.mpf("9.80665") * (1 - mp.mpf(1000) / 3000) * tau
            expected = (
                float(terminal + (initial - terminal) * mp.exp(-duration / tau)),
                float(terminal * duration - (initial - terminal) * tau * mp.expm1(-duration / tau)),
            )
        rtol, atol = 2e-11, 1e-13
    return [
        compare("velocity", result.state.velocity_m_s, expected[0], rtol, atol),
        compare("displacement", result.displacement_m, expected[1], rtol, atol),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior attempts; choose a new output path")
    cases, records = list(inputs()), []
    assert len(cases) == 342
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        expected_count = 2 if case["family"] in ("terminal", "motion", "independent") else 1
        try:
            comparisons = evaluate(case)
            assert len(comparisons) == expected_count
            records.append({"input": case, "comparisons": comparisons})
        except Exception as error:
            records.append(
                {
                    "input": case,
                    "exception": f"{type(error).__name__}: {error}",
                    "expected_comparisons": expected_count,
                }
            )
        if index % 25 == 0 or index == len(cases):
            elapsed = time.monotonic() - started
            print(
                f"particle drag {index}/{len(cases)} {index / max(elapsed, 1e-6):.1f} inputs/s "
                f"ETA {(len(cases) - index) * elapsed / index:.1f}s",
                flush=True,
            )
    comparisons = [item for record in records for item in record.get("comparisons", [])]
    failures = [
        record
        for record in records
        if "exception" in record
        or any(not item["passed"] for item in record.get("comparisons", []))
    ]
    summary = {
        "reference_version": fluids.__version__,
        "inputs": len(cases),
        "expected_comparisons": 447,
        "completed_comparisons": len(comparisons),
        "passed_comparisons": sum(item["passed"] for item in comparisons),
        "failed_comparisons": sum(not item["passed"] for item in comparisons),
        "exceptions": sum("exception" in record for record in records),
        "failures": failures,
        "elapsed_s": time.monotonic() - started,
        "records": records,
    }
    assert (
        len(comparisons) + sum(record.get("expected_comparisons", 0) for record in records) == 447
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("records", "failures")}
        )
    )


if __name__ == "__main__":
    main()
