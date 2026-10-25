"""Fixed complete source-energy and independent Dufal endpoint discriminators."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from itertools import product
from pathlib import Path

import mpmath as mp
import numpy as np
from quantum_molecular_controls import grid, models
from run_dufal import CASES
from run_dufal import models as association_models
from run_dufal_high_precision import Oracle

from chemworld.physchem.molecular_eos import molecular_residual_chemical_potentials

STEPS = (1e-3, 3e-4, 1e-4)


def compare(actual, reference, *, atol=2e-7, rtol=3e-6):
    bound = atol + rtol * max(abs(actual), abs(reference))
    finite = math.isfinite(actual) and math.isfinite(reference)
    return {
        "native": actual if math.isfinite(actual) else None,
        "reference": reference if math.isfinite(reference) else None,
        "atol": atol,
        "rtol": rtol,
        "absolute_error": abs(actual - reference) if finite else None,
        "bound": bound if finite else None,
        "passed": finite and abs(actual - reference) <= bound,
    }


def amount_derivative(reference, t, volume, amounts, index, step):
    endpoint = amounts[index] == 0
    nodes = (0, 1, 2, 3, 4) if endpoint else (-2, -1, 1, 2)
    weights = (-25, 48, -36, 16, -3) if endpoint else (1, -8, 8, -1)
    values = []
    for offset in nodes:
        perturbed = np.array(amounts, dtype=float)
        perturbed[index] += offset * step
        # The source returns extensive J for actual mol numbers at fixed V.
        values.append(float(reference.helmholtz_tv(t, volume, perturbed, property_flag="R")[0]))
    result = math.fsum(w * v for w, v in zip(weights, values, strict=True)) / (12 * step)
    return result, {
        "step_mol": step,
        "nodes": nodes,
        "energies_J": values,
        "derivative_J_mol": result,
    }


def run_source(sources, stream):
    cases = grid("association")
    assert len(cases) == 324
    cache = {}
    failures = []
    count = passed = components = analytic_differences = 0
    started = time.perf_counter()
    for unit, case in enumerate(cases, 1):
        key = tuple((k, str(v)) for k, v in case.items() if k not in ("t", "rho", "x"))
        if key not in cache:
            cache[key] = models(case, sources)
        native, reference = cache[key]
        t, rho, x = case["t"], case["rho"], case["x"]
        actual = molecular_residual_chemical_potentials(native, t, rho, x)
        analytic = reference.helmholtz_tv(t, 1 / rho, x, dadn=True, property_flag="R")[1]
        records = []
        for i, value in enumerate(actual):
            component = {
                "component": i,
                "id": case["names"][i],
                "absent": x[i] == 0,
                "source_analytic": float(analytic[i]),
                "steps": [],
            }
            analytic_ok = compare(value, float(analytic[i]))["passed"]
            component["analytic_agrees"] = analytic_ok
            analytic_differences += not analytic_ok
            for step in STEPS:
                try:
                    target, diagnostic = amount_derivative(reference, t, 1 / rho, x, i, step)
                    row = {**diagnostic, **compare(value, target)}
                except Exception as exc:
                    row = {
                        "step_mol": step,
                        "exception": f"{type(exc).__name__}: {exc}",
                        "passed": False,
                    }
                component["steps"].append(row)
                if step != STEPS[0]:
                    count += 1
                    passed += row["passed"]
                    if not row["passed"]:
                        failures.append({"unit": unit, **case, "component": i, **row})
            components += 1
            records.append(component)
        stream.write(
            json.dumps({"unit": unit, **case, "components": records}, allow_nan=False) + "\n"
        )
        stream.flush()
        if unit % 12 == 0 or unit == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"Source-energy: {unit}/324; {unit / elapsed:.2f} states/s; "
                f"ETA {(324 - unit) * elapsed / unit:.1f}s; failures={len(failures)}",
                flush=True,
            )
    assert components == 612 and count == 1224
    return {
        "units": 324,
        "component_derivatives": components,
        "comparisons": count,
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
        "source_analytic_disagreements": analytic_differences,
    }


def run_independent(sources, stream):
    mp.mp.dps = 80
    cases = list(product(range(9), (300.0, 600.0), (100.0, 5000.0)))
    cache = {}
    failures = []
    count = passed = 0
    started = time.perf_counter()
    for unit, (case, t, rho) in enumerate(cases, 1):
        if case not in cache:
            native, _, _ = association_models(case, "thermopack", source_topology=True)
            oracle = Oracle(
                sources / "saftvrmie_association.f90", case, "segment", source_topology=True
            )
            cache[case] = native, oracle
        native, oracle = cache[case]
        fractions = CASES[case][1]
        xs = tuple(mp.mpf(str(v)) for v in fractions)
        actual = molecular_residual_chemical_potentials(native, t, rho, fractions)
        records = []
        for i, value in enumerate(actual):

            def extensive(delta, index=i, xs=xs, oracle=oracle, t=t, rho=rho):
                amounts = list(xs)
                amounts[index] += delta
                total = mp.fsum(amounts)
                return total * oracle.alpha(mp.mpf(t), rho * total, [v / total for v in amounts])

            try:
                target = float(oracle.R * t * mp.diff(extensive, mp.mpf(0)))
                row = {"component": i, **compare(value, target, atol=2e-5, rtol=2e-7)}
            except Exception as exc:
                row = {"component": i, "exception": f"{type(exc).__name__}: {exc}", "passed": False}
            count += 1
            passed += row["passed"]
            if not row["passed"]:
                failures.append({"unit": unit, "case": case, "t": t, "rho": rho, **row})
            records.append(row)
        stream.write(
            json.dumps(
                {
                    "unit": unit,
                    "case": case,
                    "t": t,
                    "rho": rho,
                    "fractions": fractions,
                    "comparisons": records,
                },
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        if unit % 3 == 0 or unit == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"Independent Dufal: {unit}/36; {unit / elapsed:.2f} states/s; "
                f"ETA {(36 - unit) * elapsed / unit:.1f}s; failures={len(failures)}",
                flush=True,
            )
    assert count == 68
    return {
        "units": 36,
        "comparisons": count,
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
        "decimal_precision": 80,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("source-energy", "independent"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (args.output / "records.jsonl").open("w", encoding="utf8") as stream:
        result = (run_source if args.block == "source-energy" else run_independent)(
            args.sources, stream
        )
    result.update(
        mode="development",
        block=args.block,
        seconds=time.perf_counter() - started,
        source=importlib.metadata.version("thermopack"),
        production_model_changed=False,
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}), flush=True)
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
