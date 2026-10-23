"""Prospective 594-input / 639-comparison pressure correlation block."""

import argparse
import json
import time
from itertools import product
from math import isclose, log10, pi, sqrt
from pathlib import Path

import fluids
from fluids.friction import friction_factor as reference_friction
from fluids.two_phase import two_phase_correlations, two_phase_dP

from chemworld.physchem.friction_correlations import pipe_friction_factor
from chemworld.physchem.two_phase_holdup import TwoPhaseProperties
from chemworld.physchem.two_phase_pressure import two_phase_pressure_drop


def inputs():
    for reynolds, roughness in product(
        (50, 500, 2039.9, 2040, 2040.1, 1e4, 1e5, 1e6, 1e8),
        (0, 1e-6, 1e-4, 0.01, 0.05),
    ):
        yield {"family": "friction", "Re": reynolds, "eD": roughness}
    for method, quality, flux, diameter in product(
        two_phase_correlations, (0.1, 0.5, 0.9), (50, 250, 2000), (0.0005, 0.005)
    ):
        yield {"method": method, "x": quality, "G": flux, "D": diameter}
    for quality, flux, angle, acceleration in product(
        (0.1, 0.5, 0.9), (50, 250, 1000), (-10, 0, 10), (False, True)
    ):
        yield {
            "method": "Beggs-Brill",
            "x": quality,
            "G": flux,
            "D": 0.005,
            "angle": angle,
            "acceleration": acceleration,
        }
    for method, density, flux in product(
        ("Chisholm", "Chisholm rough", "Baroczy_Chisholm"),
        (0.1, 1, 10),
        (100, 500, 600, 1000, 1900, 2500),
    ):
        yield {
            "method": method,
            "x": 0.5,
            "G": flux,
            "D": 0.005,
            "rhog": density,
            "roughness": 1e-5 if method == "Chisholm rough" else 0,
        }
    for quality, flux in product((0.001, 0.2, 0.95), (10, 500, 5000)):
        yield {"method": "Kim_Mudawar", "x": quality, "G": flux, "D": 0.005}


def measurement(name, native, reference):
    return {
        "quantity": name,
        "native": native,
        "reference": reference,
        "relative_tolerance": 2e-9,
        "absolute_tolerance": 1e-9,
        "passed": isclose(native, reference, rel_tol=2e-9, abs_tol=1e-9),
    }


def evaluate(case):
    if case.get("family") == "friction":
        reynolds, roughness = case["Re"], case["eD"]
        native = pipe_friction_factor(reynolds, roughness)
        residual = (
            native - 64 / reynolds
            if reynolds < 2040
            else 1 / sqrt(native) + 2 * log10(roughness / 3.7 + 2.51 / (reynolds * sqrt(native)))
        )
        return [
            measurement("Darcy friction", native, reference_friction(reynolds, roughness)),
            measurement("constitutive residual", residual, 0.0),
        ]
    p = TwoPhaseProperties(900, case.get("rhog", 10), 0.0005, 1.5e-5, 0.05, 1e7, 5e7)
    mass = case["G"] * pi * case["D"] ** 2 / 4
    native = two_phase_pressure_drop(
        p,
        method=case["method"],
        mass_flow_kg_s=mass,
        diameter_m=case["D"],
        vapor_quality=case["x"],
        roughness_m=case.get("roughness", 0),
        inclination_degrees=case.get("angle", 0),
        beggs_brill_acceleration=case.get("acceleration", False),
        check_recommended_ranges=False,
        reference_only=True,
    )
    kwargs = {
        "m": mass,
        "x": case["x"],
        "rhol": 900,
        "rhog": p.vapor_density_kg_m3,
        "mul": 0.0005,
        "mug": 1.5e-5,
        "sigma": 0.05,
        "P": 1e7,
        "Pc": 5e7,
        "D": case["D"],
        "L": 1,
        "roughness": case.get("roughness", 0),
        "angle": case.get("angle", 0),
    }
    if case["method"] == "Beggs-Brill":
        from fluids.two_phase import Beggs_Brill

        kwargs.pop("Pc")
        reference = Beggs_Brill(**kwargs, acceleration=case.get("acceleration", False))
    else:
        reference = two_phase_dP(**kwargs, Method=case["method"])
    return [measurement("pressure_Pa", native.pressure_drop_Pa, reference)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a new attempt directory; never overwrite a result")
    cases = list(inputs())
    assert len(cases) == 594
    records, failures = [], []
    start = time.perf_counter()
    for index, case in enumerate(cases, 1):
        record = {"unit": index, "input": case}
        try:
            record["measurements"] = evaluate(case)
            record["passed"] = all(item["passed"] for item in record["measurements"])
        except Exception as exc:
            record["measurements"] = []
            record["passed"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if not record["passed"]:
            failures.append(record)
        if index % 50 == 0 or index == len(cases):
            elapsed = time.perf_counter() - start
            print(
                f"pressure {index}/594 rate={index / max(elapsed, 1e-9):.1f}/s "
                f"ETA={elapsed / index * (594 - index):.1f}s failures={len(failures)}",
                flush=True,
            )
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 594,
        "expected_comparisons": 639,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": failures,
        "records": records,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))


if __name__ == "__main__":
    main()
