"""Prospective 459 inputs/781 comparisons, with empirical domain checks explicit."""

import argparse
import json
import time
from itertools import product
from math import isclose, log10, sqrt
from pathlib import Path

import fluids
import fluids.friction as reference

from chemworld.physchem.curved_friction import (
    HELICAL_FRICTION_METHODS,
    HELICAL_TRANSITION_RANGES,
    helical_friction_factor,
    helical_transition_reynolds,
)
from chemworld.physchem.friction_correlations import pipe_friction_factor
from chemworld.physchem.plate_friction import PLATE_FRICTION_METHODS, plate_friction_factor
from chemworld.physchem.straight_friction import (
    crane_fully_turbulent_friction,
    friction_transmission_factor,
    straight_friction_correlation,
)


def inputs():
    for method, reynolds, roughness in product(reference.fmethods, (1e4, 1e5, 1e6), (1e-4, 0.001)):
        yield {"family": "straight", "method": method, "Re": reynolds, "eD": roughness}
    for method, reynolds in product(("Blasius", "Prandtl_von_Karman_Nikuradse"), (4000, 1e4, 1e5)):
        yield {"family": "smooth", "method": method, "Re": reynolds}
    for reynolds in (1, 100, 2039.9):
        yield {"family": "laminar", "Re": reynolds}
    for roughness in (1e-6, 1e-4, 0.01):
        yield {"family": "fully-rough", "eD": roughness}
    for factor in (0.005, 0.02, 0.1):
        yield {"family": "transmission", "f": factor}
    for diameter in (0.005, 0.01, 0.1, 1):
        yield {"family": "Crane", "D": diameter}
    for method, ratio in product(HELICAL_TRANSITION_RANGES, (0.01, 0.05, 0.1)):
        yield {"family": "transition", "method": method, "ratio": ratio}
    for method, reynolds, ratio in product(
        HELICAL_FRICTION_METHODS, (100, 1000, 1e4, 21999, 22000, 1e5), (0.01, 0.05)
    ):
        yield {"family": "helical", "method": method, "Re": reynolds, "ratio": ratio}
    for method, roughness in product(
        ("Ju", "Mandal Nigam", "Prasad", "Schmidt turbulent"), (1e-5, 1e-4)
    ):
        yield {
            "family": "helical-rough",
            "method": method,
            "Re": 5e4,
            "ratio": 0.05,
            "roughness": roughness,
        }
    for transition, reynolds in product(HELICAL_TRANSITION_RANGES, (100, 20000)):
        yield {"family": "helical-auto", "transition": transition, "Re": reynolds, "ratio": 0.05}
    for method, reynolds, angle in product(
        PLATE_FRICTION_METHODS, (10, 100, 1999, 2000, 1e4), (30, 45, 60)
    ):
        yield {"family": "plate", "method": method, "Re": reynolds, "angle": angle}
    for angle, thresholds in (
        (30, (10, 100)),
        (45, (15, 300)),
        (50, (20, 300)),
        (60, (40, 400)),
        (65, (50, 500)),
    ):
        for threshold, position in product(thresholds, (0.999999, 1, 1.000001)):
            yield {"family": "plate", "method": "Kumar", "Re": threshold * position, "angle": angle}


def compare(name, native, expected, relative_tolerance=2e-9, absolute_tolerance=1e-10):
    return {
        "quantity": name,
        "native": native,
        "reference": expected,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "passed": isclose(native, expected, rel_tol=relative_tolerance, abs_tol=absolute_tolerance),
    }


def evaluate(case):
    family = case["family"]
    if family in {"straight", "smooth"}:
        method, reynolds, roughness = case["method"], case["Re"], case.get("eD", 0)
        function = getattr(reference, method)
        expected = function(reynolds) if family == "smooth" else function(reynolds, roughness)
        result = [
            compare(
                "Darcy",
                straight_friction_correlation(
                    reynolds, roughness, method=method, check_recommended_ranges=False
                ),
                expected,
            )
        ]
        if family == "straight":
            result.append(
                compare(
                    "Fanning",
                    straight_friction_correlation(
                        reynolds,
                        roughness,
                        method=method,
                        darcy=False,
                        check_recommended_ranges=False,
                    ),
                    expected / 4,
                )
            )
        return result
    if family == "laminar":
        reynolds = case["Re"]
        return [
            compare(
                "direct", straight_friction_correlation(reynolds, method="laminar"), 64 / reynolds
            ),
            compare(
                "automatic",
                pipe_friction_factor(reynolds, 0.001),
                reference.friction_factor(reynolds, 0.001),
            ),
        ]
    if family == "fully-rough":
        roughness = case["eD"]
        return [
            compare(
                "Darcy",
                straight_friction_correlation(1e8, roughness, method="von_Karman"),
                reference.von_Karman(roughness),
            )
        ]
    if family == "transmission":
        transmission = friction_transmission_factor(darcy_friction_factor=case["f"])
        return [
            compare("F", transmission, reference.transmission_factor(fd=case["f"])),
            compare(
                "inverse",
                friction_transmission_factor(transmission_factor=transmission),
                reference.transmission_factor(F=transmission),
            ),
        ]
    if family == "Crane":
        d = case["D"]
        factor = crane_fully_turbulent_friction(d)
        roughness = 3.4126825352925e-5 * d**-1.0112
        residual = 1 / sqrt(factor) + 2 * log10(roughness / 3.7 + 2.51 / (7.5e6 * d * sqrt(factor)))
        return [
            compare("Crane source fast approximation", factor, reference.ft_Crane(d), 3e-5),
            compare("Colebrook residual", residual, 0, absolute_tolerance=1e-9),
        ]
    if family == "transition":
        method, coil = case["method"], 0.02 / case["ratio"]
        return [
            compare(
                "critical_Re",
                helical_transition_reynolds(
                    0.02, coil, method=method, check_recommended_ranges=False
                ),
                reference.helical_Re_crit(0.02, coil, Method=method),
            )
        ]
    if family.startswith("helical"):
        method, transition, reynolds, coil = (
            case.get("method"),
            case.get("transition", "Schmidt"),
            case["Re"],
            0.02 / case["ratio"],
        )
        roughness = case.get("roughness", 0)
        expected = reference.friction_factor_curved(
            reynolds, 0.02, coil, roughness, Method=method, Rec_method=transition
        )
        native = helical_friction_factor(
            reynolds,
            0.02,
            coil,
            roughness_m=roughness,
            method=method,
            transition_method=transition,
            check_recommended_ranges=False,
        )
        values = [compare("Darcy", native.friction_factor, expected)]
        if family == "helical":
            fanning = helical_friction_factor(
                reynolds, 0.02, coil, method=method, darcy=False, check_recommended_ranges=False
            )
            values.append(compare("Fanning", fanning.friction_factor, expected / 4))
        return values
    method, reynolds, angle = case["method"], case["Re"], case["angle"]
    function = getattr(reference, "friction_plate_" + method.replace(" ", "_"))
    if method == "Muley Manglik":
        expected, enlargement = function(reynolds, angle, 1.2), 1.2
    else:
        expected, enlargement = function(reynolds, angle), 1
    return [
        compare(
            "Darcy",
            plate_friction_factor(
                reynolds,
                angle,
                method=method,
                enlargement_factor=enlargement,
                check_recommended_ranges=False,
            ),
            expected,
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite an attempt")
    cases = list(inputs())
    assert len(cases) == 459
    records, failures = [], []
    start = time.perf_counter()
    for index, case in enumerate(cases, 1):
        record = {"unit": index, "input": case}
        try:
            record["measurements"] = evaluate(case)
            record["passed"] = all(m["passed"] for m in record["measurements"])
        except Exception as exc:
            record.update(measurements=[], passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        if not record["passed"]:
            failures.append(record)
        if index % 50 == 0 or index == 459:
            elapsed = time.perf_counter() - start
            print(
                f"friction {index}/459 rate={index / max(elapsed, 1e-9):.1f}/s "
                f"ETA={elapsed / index * (459 - index):.2f}s failures={len(failures)}",
                flush=True,
            )
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 459,
        "expected_comparisons": 781,
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
    print(json.dumps({k: v for k, v in summary.items() if k not in {"failures", "records"}}))


if __name__ == "__main__":
    main()
