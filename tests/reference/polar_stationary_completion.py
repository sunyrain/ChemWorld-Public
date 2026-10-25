# ruff: noqa: B023 -- all differentiation callbacks are evaluated within the current unit.
"""Fixed original-polarization adjudication, source replay and strong-polarity block."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import mpmath as mp
from polar_stationary_oracle import PolarOracle
from run_saft_polar_extensions import CONTROLS, ORDERS, models

from chemworld.physchem.molecular_eos import (
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)


def grid(block):
    if block == "envelope":
        return [
            {"case": c, "control": k, "multiplier": m, "T": t, "rho": r}
            for c, k, m, (t, r) in product(
                (3, 6, 7), range(8), (1.0, 5.0, 20.0), ((280.0, 4000.0), (480.0, 100.0))
            )
        ]
    return [
        {"case": c, "control": k, "multiplier": 1.0, "T": 350.0, "rho": r}
        for c, k, r in product((0, 2, 3, 4, 6), range(8), (100.0, 1500.0, 4000.0))
    ]


def run(block, source, output):
    output.mkdir(parents=True, exist_ok=False)
    mp.mp.dps = 80
    cases = grid(block)
    failures, classifications, inputs = [], [], []
    count = rejected = planned = 0
    started = time.perf_counter()
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, case in enumerate(cases, 1):
            t, rho = case["T"], case["rho"]
            native, ref, zero, x = models(case["case"], CONTROLS[5 + case["control"]], True)
            if case["multiplier"] != 1:
                spec = replace(
                    native.polar,
                    moments=tuple(
                        replace(
                            p,
                            alpha_parallel_m3=p.alpha_parallel_m3 * case["multiplier"],
                            alpha_perpendicular_m3=p.alpha_perpendicular_m3 * case["multiplier"],
                        )
                        for p in native.polar.moments
                    ),
                )
                native = replace(native, polar=spec)
            inputs.append(native.to_dict())
            oracle = PolarOracle(native, t, source, source_replay=block == "source-replay")
            xx = tuple(mp.mpf(str(v)) for v in x)
            tt, rr = mp.mpf(t), mp.mpf(rho)
            expected, actual, errors, diagnostic = {}, {}, {}, {}
            polar = SimpleNamespace(
                alphar=lambda t, r, x, native=native: native.contributions(t, r, x)[
                    "correlation_polar"
                ],
                component_ids=native.component_ids,
                gas_constant_J_mol_K=native.gas_constant_J_mol_K,
            )
            quantities = [f"Ar{i}{j}" for i, j in ORDERS]
            if block != "source-replay":
                quantities += [f"mu_{i}" for i in range(len(x))]
            planned += len(quantities)
            try:
                diagnostic = oracle.evaluate(tt, rr, xx, diagnostic=True)
                expected = {
                    f"Ar{i}{j}": float(
                        mp.diff(
                            lambda a, b: oracle.evaluate(tt / (1 + a), rr * (1 + b), xx),
                            (mp.mpf(0), mp.mpf(0)),
                            (i, j),
                        )
                    )
                    for i, j in ORDERS
                }
                if block != "source-replay":
                    for i in range(len(x)):

                        def extensive(z, index=i):
                            amounts = [v + z if k == index else v for k, v in enumerate(xx)]
                            total = mp.fsum(amounts)
                            return total * oracle.evaluate(
                                tt, rr * total, [v / total for v in amounts]
                            )

                        expected[f"mu_{i}"] = float(
                            mp.mpf(str(native.gas_constant_J_mol_K))
                            * tt
                            * mp.diff(extensive, mp.mpf(0))
                        )
            except Exception as exc:
                errors["independent_exception"] = f"{type(exc).__name__}: {exc}"
            try:
                if block == "source-replay":
                    actual = {
                        f"Ar{i}{j}": ref.get_Arxy(i, j, t, rho, x) - zero.get_Arxy(i, j, t, rho, x)
                        for i, j in ORDERS
                    }
                else:
                    actual = {
                        f"Ar{i}{j}": molecular_scaled_derivative(
                            polar, t, rho, x, inverse_temperature_order=i, density_order=j
                        )
                        for i, j in ORDERS
                    }
                    actual.update(
                        {
                            f"mu_{i}": v
                            for i, v in enumerate(
                                molecular_residual_chemical_potentials(polar, t, rho, x)
                            )
                        }
                    )
            except Exception as exc:
                errors["native_exception" if block != "source-replay" else "source_exception"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            comparisons = []
            proven_nonminimum = (
                block != "source-replay"
                and diagnostic
                and diagnostic["normalized_stationarity"] < 1e-20
                and diagnostic["minimum_curvature"] < 0
            )
            classification_pass = bool(diagnostic) and (
                block == "source-replay"
                or (
                    diagnostic["normalized_stationarity"] < 1e-20
                    and (diagnostic["minimum_curvature"] > 0 or "native_exception" in errors)
                )
            )
            classifications.append(
                dict(
                    unit=unit,
                    **case,
                    **diagnostic,
                    passed=classification_pass,
                    proven_nonminimum=bool(proven_nonminimum),
                )
            )
            if proven_nonminimum and "native_exception" in errors:
                rejected += 1
            else:
                for quantity in quantities:
                    a, b = actual.get(quantity), expected.get(quantity)
                    a = float(a) if a is not None and math.isfinite(a) else None
                    b = float(b) if b is not None and math.isfinite(b) else None
                    atol = 2e-5 if quantity.startswith("mu") else 2e-7
                    passed = (
                        a is not None
                        and b is not None
                        and math.isclose(a, b, rel_tol=2e-5, abs_tol=atol)
                    )
                    row = dict(
                        quantity=quantity,
                        native=a,
                        reference=b,
                        atol=atol,
                        rtol=2e-5,
                        passed=passed,
                        **errors,
                    )
                    comparisons.append(row)
                    count += 1
                    if not passed:
                        failures.append(dict(unit=unit, **case, **row))
            stream.write(
                json.dumps(
                    dict(
                        unit=unit,
                        **case,
                        diagnostic=diagnostic,
                        classification_pass=classification_pass,
                        comparisons=comparisons,
                        **errors,
                    )
                )
                + "\n"
            )
            stream.flush()
            elapsed = time.perf_counter() - started
            print(
                f"Polar {block}: {unit}/{len(cases)}, {unit / elapsed:.3f} units/s, "
                f"ETA {(len(cases) - unit) * elapsed / unit:.1f}s, failed={len(failures)}, "
                f"refused_nonminima={rejected}",
                flush=True,
            )
    summary = {
        "mode": "development",
        "block": block,
        "units": len(cases),
        "comparisons": count,
        "planned_comparisons": planned,
        "not_applicable_comparisons": planned - count,
        "passed": count - len(failures),
        "failed": len(failures),
        "all_failures": failures,
        "classifications": classifications,
        "classification_failed": sum(not r["passed"] for r in classifications),
        "refused_nonminimum_units": rejected,
        "inputs": inputs,
        "seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("all_failures", "classifications", "inputs")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("original", "source-replay", "envelope"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.block, args.source, args.output)
