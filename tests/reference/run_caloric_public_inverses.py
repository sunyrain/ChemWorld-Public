"""Fixed direct public H/S inversion comparisons; no new caloric solver."""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import chemicals
from chemicals import heat_capacity as reference

from chemworld.physchem.heat_capacity import temperature_from_caloric_change
from chemworld.physchem.specs import PropertyCorrelation


def inputs():
    for cyclic, origin, target, entropy, factor in product(
        (False, True), (298.15, 500.0), (250.0, 500.0, 1100.0), (False, True), (0.8, 1.0, 1.2)
    ):
        yield {
            "coefficients": {
                "similarity_variable": 0.22491,
                "MW": 142.28,
                "cyclic_aliphatic": cyclic,
            },
            "reference_temperature_K": origin,
            "target_temperature_K": target,
            "entropy": entropy,
            "factor": factor,
        }
    for entropy, target, weight, similarity in (
        (False, 55000.0, 80.0, 0.23), (True, 112.8, 72.151, 0.2356)
    ):
        yield {
            "coefficients": {
                "similarity_variable": similarity,
                "MW": weight,
                "cyclic_aliphatic": False,
            },
            "reference_temperature_K": 298.15,
            "target_temperature_K": None,
            "target_change": target,
            "entropy": entropy,
            "factor": 1.0,
        }


def compare(spec):
    row = {"inputs": spec, "checks": [], "errors": {}}
    coefficients = spec["coefficients"]
    origin, target = spec["reference_temperature_K"], spec["target_temperature_K"]
    entropy, factor = spec["entropy"], spec["factor"]
    change = spec.get("target_change")
    if change is None:
        primitive = (
            reference.Lastovka_Shaw_integral_over_T if entropy else reference.Lastovka_Shaw_integral
        )
        change = factor * (primitive(target, **coefficients) - primitive(origin, **coefficients))
    row["target_change"] = change
    row["target_unit"] = "J/(mol*K)" if entropy else "J/mol"
    curve = PropertyCorrelation(
        "lastovka-direct-inverse", "lastovka_shaw_cp",
        {name: int(value) if isinstance(value, bool) else value
         for name, value in coefficients.items()},
        {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
        {"temperature": (200.0, 1200.0)}, "Synthetic public caloric inversion qualification",
    )
    try:
        row["native_temperature_K"] = temperature_from_caloric_change(
            curve, change / factor, reference_temperature_K=origin, entropy=entropy,
        )
    except Exception as exc:
        row["errors"]["native"] = f"{type(exc).__name__}: {exc}"
    try:
        function = reference.Lastovka_Shaw_T_for_Sm if entropy else reference.Lastovka_Shaw_T_for_Hm
        row["reference_temperature_K"] = function(
            change, **coefficients, T_ref=origin, factor=factor,
        )
    except Exception as exc:
        row["errors"]["reference"] = f"{type(exc).__name__}: {exc}"

    def check(name, actual, expected):
        error = abs(actual - expected)
        row["checks"].append({
            "name": name, "actual_K": actual, "expected_K": expected,
            "absolute_error_K": error, "passed": error <= 2e-7,
        })

    if not row["errors"]:
        check("native_vs_public", row["native_temperature_K"], row["reference_temperature_K"])
    if target is not None:
        for kind in ("native", "reference"):
            if kind not in row["errors"]:
                check(f"{kind}_vs_target", row[f"{kind}_temperature_K"], target)
    row["passed"] = not row["errors"] and all(c["passed"] for c in row["checks"])
    return row


def run(output):
    if output.exists():
        raise ValueError("preserve earlier attempts; use a new output path")
    started = time.perf_counter()
    results = []
    for index, spec in enumerate(inputs(), 1):
        try:
            result = compare(spec)
        except Exception as exc:
            result = {
                "inputs": spec, "checks": [], "passed": False,
                "errors": {"setup": f"{type(exc).__name__}: {exc}"},
            }
        results.append(result)
        if index % 12 == 0 or index == 74:
            elapsed = time.perf_counter() - started
            print(
                f"Completed {index}/74, {index / elapsed:.1f} states/s, "
                f"ETA {(74 - index) * elapsed / index:.2f}s", flush=True,
            )
    checks = [c for row in results for c in row["checks"]]
    report = {
        "schema": "chemworld-public-caloric-inverses-1", "mode": "development",
        "reference_versions": {"chemicals": chemicals.__version__},
        "expected": {"states": 74, "comparisons": 218}, "completed": len(results),
        "passed_count": sum(row["passed"] for row in results), "comparisons": len(checks),
        "passed_comparisons": sum(c["passed"] for c in checks), "results": results,
        "failures": [row for row in results if not row["passed"]],
        "seconds": time.perf_counter() - started,
    }
    report["passed"] = report["passed_count"] == 74 and report["passed_comparisons"] == 218
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in (
        "completed", "comparisons", "passed_count", "passed_comparisons", "passed", "seconds",
    )}), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
