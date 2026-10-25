"""Prospective 33C finite-start completion; original single-start block retained."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path
from time import monotonic

from chemworld.runtime.molecular_process_design import optimize_molecular_process
from reference.molecular_design33_cases import specification
from reference.molecular_design33_reference import CASES, optimum, point, self_check
from reference.run_molecular_design33 import (
    compare,
    consumer_unit,
    design_unit,
    negative_unit,
)


def multistart(index, reverse=False):
    starts = tuple(specification(index, s).starts[0] for s in (0.2, 0.8))
    return replace(specification(index), starts=starts[::-1] if reverse else starts)


def design(index, reverse, progress):
    spec = multistart(index, reverse)
    row = design_unit(index, 0.2, progress, declared_spec=spec)
    result = row["result"]["optimization"]
    attempts, trials = result["attempts"], result["trials"]
    compare(
        row["checks"],
        "both_starts_and_accounting",
        len(attempts) == 2
        and [a["initial"] for a in attempts] == list(spec.starts)
        and sum(a["executed_evaluations"] for a in attempts) + 1 == len(trials),
        [{k: a[k] for k in ("initial", "success", "executed_evaluations")} for a in attempts],
    )
    costs = [trials[a["verified_trial"]]["objective"] for a in attempts if a["success"]]
    selected_cost = result["point"]["objective"] if result["point"] else None
    compare(
        row["checks"],
        "best_verified_converged_attempt",
        selected_cost is not None and bool(costs) and abs(selected_cost - min(costs)) <= 5e-5,
        selected_cost,
        costs,
        5e-5,
    )
    return row


def local_branch(progress):
    result = optimize_molecular_process(specification(7, 0.8), progress=progress)
    values, selected = result.optimization.values, result.optimization.point
    independent, global_best = point(CASES[7], 6.5e5), optimum(CASES[7])
    # Cooling derivative in W is lambda - ((Pout/P)^a - 1)/eta2.
    x, _, pout, _, eta2, _, price, _, _ = CASES[7]
    cooling_derivative = price - ((pout / 6.5e5) ** (1 / (4.5 - x)) - 1) / eta2
    checks = []
    compare(
        checks,
        "upper_pressure_zero_cooling_corner",
        result.success
        and abs(values["interstage_pressure"] - 6.5e5) <= 1
        and abs(values["cooling_duty"]) <= 5e-4,
        values,
    )
    compare(
        checks,
        "independent_local_cost",
        selected is not None and abs(selected.objective - independent["cost"]) <= 5e-5,
        selected.objective if selected else None,
        independent["cost"],
        5e-5,
    )
    compare(
        checks,
        "feasible_direction_derivatives",
        independent["d_cost_d_log_pressure"] < 0 and cooling_derivative > 0,
        [independent["d_cost_d_log_pressure"], cooling_derivative],
    )
    compare(
        checks,
        "strict_global_gap_preserved",
        selected is not None and selected.objective - global_best["cost"] > 1,
        selected.objective - global_best["cost"] if selected else None,
        ">1 currency/s",
    )
    return {
        "checks": checks,
        "result": result.to_dict(),
        "local_reference": independent,
        "global_reference": global_best,
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    reference_checks, began, evaluations = self_check(), monotonic(), 0
    completed = []

    def progress(_):
        nonlocal evaluations
        evaluations += 1

    units = [
        (
            f"design-{i + 1}-{'reverse' if reverse else 'forward'}",
            10,
            lambda i=i, reverse=reverse: design(i, reverse, progress),
        )
        for i in range(8)
        for reverse in (False, True)
    ]
    units += [
        (
            "world-wait",
            8,
            lambda: consumer_unit(0, False, progress, output, declared_spec=multistart(0)),
        ),
        (
            "world-event-resume",
            8,
            lambda: consumer_unit(3, True, progress, output, declared_spec=multistart(3)),
        ),
        (
            "infeasible",
            8,
            lambda: negative_unit("infeasible", progress, declared_spec=multistart(0)),
        ),
        ("budget", 8, lambda: negative_unit("budget", progress, declared_spec=multistart(0))),
        ("single_start_local_branch", 4, lambda: local_branch(progress)),
    ]
    for name, count, function in units:
        before = evaluations
        try:
            row = function()
            assert len(row["checks"]) == count
        except Exception:
            row = {"checks": [], "exception": traceback.format_exc()}
        row.update(unit=name, declared_comparisons=count, physical_evaluations=evaluations - before)
        (output / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        completed.append(
            {k: v for k, v in row.items() if k not in {"result", "state", "continuation"}}
        )
        elapsed = max(monotonic() - began, 1e-9)
        print(
            json.dumps(
                {
                    "stage": name,
                    "completed": len(completed),
                    "total": len(units),
                    "passed": sum(c["passed"] for c in row["checks"]),
                    "physical_evaluations": evaluations,
                    "units_per_s": len(completed) / elapsed,
                    "eta_s": (len(units) - len(completed)) * elapsed / len(completed),
                }
            ),
            flush=True,
        )
    checks = [c for row in completed for c in row["checks"]]
    summary = {
        "scope": "development_finite_declared_starts_not_global_optimizer",
        "declared_units": 21,
        "executed_units": len(completed),
        "declared_comparisons": 196,
        "executed_comparisons": len(checks),
        "passed_comparisons": sum(c["passed"] for c in checks),
        "passed_units": sum(
            len(r["checks"]) == r["declared_comparisons"] and all(c["passed"] for c in r["checks"])
            for r in completed
        ),
        "unexecuted_comparisons": 196 - len(checks),
        "physical_evaluations": evaluations,
        "elapsed_s": monotonic() - began,
        "reference_self_checks": reference_checks,
        "units": completed,
        "failures": [
            {"unit": r["unit"], **c} for r in completed for c in r["checks"] if not c["passed"]
        ],
        "exceptions": [
            {"unit": r["unit"], "exception": r["exception"]} for r in completed if "exception" in r
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "units"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
