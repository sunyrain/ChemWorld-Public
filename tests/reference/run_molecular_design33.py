"""Fixed 33C 20-unit/160-comparison development block; preserve every failure."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.examples.molecular_process_design import design_receiver
from chemworld.physchem.bounded_design import BoundedDesignPolicy, DesignConstraint
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_process_design import (
    MolecularProcessDesignSpec,
    optimize_molecular_process,
    optimized_process_receiver,
)
from chemworld.world.molecular_process_world import build_molecular_process_world
from reference.molecular_design33_cases import specification
from reference.molecular_design33_reference import CASES, optimum, receiver, self_check


def compare(rows, name, passed, actual=None, expected=None, tolerance=None):
    rows.append(
        {
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "tolerance": tolerance,
        }
    )


def close(rows, name, actual, expected, tolerance):
    compare(rows, name, abs(actual - expected) <= tolerance, actual, expected, tolerance)


def design_unit(index, start, progress, *, declared_spec=None):
    spec = specification(index, start) if declared_spec is None else declared_spec
    before = spec.to_dict()
    counts = []

    def evaluated(row):
        counts.append(row)
        progress(row)

    result = optimize_molecular_process(spec, progress=evaluated)
    checks, expected = [], optimum(CASES[index])
    compare(checks, "verified_convergence", result.success, result.success, True)
    values, point = result.optimization.values, result.optimization.point
    if values is not None and point is not None:
        close(
            checks,
            "pressure",
            values["interstage_pressure"],
            expected["pressure"],
            2e-4 * expected["pressure"],
        )
        close(checks, "heat_duty", values["cooling_duty"], -expected["cooling"], 5e-4)
        close(checks, "cost", point.objective, expected["cost"], 5e-5)
        violations = [c.violation(point.observations[c.metric]) for c in spec.constraints]
        compare(
            checks,
            "physical_constraints",
            all(v <= c.tolerance for v, c in zip(violations, spec.constraints, strict=True)),
            violations,
            [c.tolerance for c in spec.constraints],
        )
        close(
            checks,
            "outlet_temperature",
            point.observations["outlet_temperature_K"],
            expected["temperature"],
            0.003,
        )
        close(checks, "energy", point.observations["energy_residual_W"], 0.0, 2e-6)
    else:
        for name in (
            "pressure",
            "heat_duty",
            "cost",
            "physical_constraints",
            "outlet_temperature",
            "energy",
        ):
            compare(checks, name, False, result.optimization.failure_reason)
    compare(
        checks,
        "accounting_and_input",
        len(counts) == len(result.optimization.trials)
        and spec.to_dict() == before
        and all(
            0 <= c.variable.encode(t["values"][c.variable.parameter_id]) <= 1
            for t in result.optimization.trials
            for c in spec.controls
        ),
        {"callbacks": len(counts), "trials": len(result.optimization.trials)},
    )
    return {"checks": checks, "reference": expected, "result": result.to_dict()}


def consumer_unit(index, event, progress, output, *, declared_spec=None):
    spec = specification(index) if declared_spec is None else declared_spec
    template, initial = design_receiver(spec, maximum_amount_mol=0.065 if event else 0.3)
    world = build_molecular_process_world(
        template, initial, formulas={"A": "Ar", "B": "C3H8"}, duration_range_s=(0.5, 14400.0)
    )
    runtime = ChemWorldRuntime(composition=world)
    try:
        runtime.reset(seed=7)
        before = runtime._state.to_dict(include_hidden=True)
        result = optimize_molecular_process(spec, progress=progress)
        unchanged = before == runtime._state.to_dict(include_hidden=True)
    finally:
        runtime.close()
    checks = []
    if not result.success:
        return {
            "checks": [
                {
                    "name": f"consumer_{i}",
                    "passed": False,
                    "actual": result.optimization.failure_reason,
                }
                for i in range(8)
            ],
            "result": result.to_dict(),
        }
    selected = optimized_process_receiver(result, template)
    world = build_molecular_process_world(
        selected, initial, formulas={"A": "Ar", "B": "C3H8"}, duration_range_s=(0.5, 14400.0)
    )
    runtime = ChemWorldRuntime(composition=world)
    continuation = None
    try:
        runtime.reset(seed=7)
        outcome = runtime.execute({"operation": "wait", "duration_s": 3.0 if event else 1.0})
        state = runtime._state
        expected = receiver(CASES[index], optimum(CASES[index]), 1.5 if event else 1.0)
        compare(
            checks,
            "outcome",
            outcome.info["transaction_status"] == "committed"
            and (not event or outcome.info["operation_outcome"] == "stopped"),
            outcome.info,
        )
        close(checks, "material", sum(state.species_amounts.values()), expected["amount"], 2e-10)
        close(checks, "temperature", state.temperature_K, expected["temperature"], 0.003)
        close(checks, "pressure", state.pressure_Pa, expected["pressure"], 0.1)
        compare(
            checks,
            "utilities_once",
            abs(state.ledger.flow_heat_J - expected["heat"]) <= 2e-5
            and abs(state.ledger.flow_shaft_work_J - expected["work"]) <= 2e-5,
            [state.ledger.flow_heat_J, state.ledger.flow_shaft_work_J],
            [expected["heat"], expected["work"]],
            2e-5,
        )
        close(checks, "actual_time", state.ledger.time_s, 1.5 if event else 1.0, 2e-7)
        compare(
            checks,
            "no_analysis_mutation_or_debit",
            unchanged
            and result.optimization.to_dict()["experiment_resource_debit"] is None
            and result.optimization.to_dict()["physical_time_s"] is None,
        )
        actual = state.to_dict(include_hidden=True)
    finally:
        runtime.close()
    if event:
        resumed = ChemWorldRuntime(composition=world)
        try:
            resumed.reset(seed=7)
            first = resumed.execute({"operation": "wait", "duration_s": 0.5})
            snapshot, continuation_file = output / "snapshot.json", output / "continued.json"
            resumed.snapshot().save(snapshot)
            second = resumed.execute({"operation": "wait", "duration_s": 3.0})
            code = (
                "import json,sys; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                "r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text())); "
                "o=r.execute({'operation':'wait','duration_s':3.}); "
                "Path(sys.argv[2]).write_text(json.dumps({'status':o.info['transaction_status'],"
                "'state':r._state.to_dict(include_hidden=True)})); r.close()"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code, str(snapshot), str(continuation_file)],
                env={**os.environ},
                capture_output=True,
                text=True,
                check=True,
            )
            continuation = json.loads(continuation_file.read_text())
            same = continuation["state"] == json.loads(
                json.dumps(resumed._state.to_dict(include_hidden=True))
            )
            compare(
                checks,
                "cross_process_exact_continuation",
                same
                and first.info["transaction_status"]
                == second.info["transaction_status"]
                == continuation["status"]
                == "committed"
                and abs(resumed._state.ledger.time_s - 1.5) <= 2e-7,
                {
                    "same": same,
                    "stderr": completed.stderr,
                    "first": first.info,
                    "second": second.info,
                },
            )
        finally:
            resumed.close()
    else:
        restored = MolecularProcessDesignSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        compare(checks, "original_spec_json_roundtrip", restored.to_dict() == spec.to_dict())
    return {
        "checks": checks,
        "result": result.to_dict(),
        "state": actual,
        "reference": expected,
        "continuation": continuation,
    }


def negative_unit(kind, progress, *, declared_spec=None):
    spec = specification(0) if declared_spec is None else declared_spec
    if kind == "budget":
        spec = replace(spec, policy=BoundedDesignPolicy(maximum_evaluations=2))
    else:
        spec = replace(
            spec,
            constraints=(
                *spec.constraints,
                DesignConstraint("device:1:temperature_K", "K", 300.0, 2e-5, upper=200.0),
            ),
            policy=BoundedDesignPolicy(maximum_iterations=30),
        )
    before = spec.to_dict()
    template, state = design_receiver(spec)
    initial = state.to_dict(include_hidden=True)
    result = optimize_molecular_process(spec, progress=progress)
    optimization, checks = result.optimization, []
    compare(checks, "rejected", not result.success)
    compare(checks, "no_executable_path", result.selected_path is None)
    compare(
        checks, "no_selected_values", optimization.values is None and optimization.point is None
    )
    compare(
        checks,
        "input_unchanged",
        spec.to_dict() == before and state.to_dict(include_hidden=True) == initial,
    )
    compare(
        checks,
        "no_trial_debits",
        optimization.to_dict()["physical_time_s"] is None
        and optimization.to_dict()["experiment_resource_debit"] is None,
    )
    compare(
        checks,
        "truthful_counts",
        len(optimization.trials) == sum(a["executed_evaluations"] for a in optimization.attempts)
        and (kind != "budget" or len(optimization.trials) == 2),
    )
    refused = False
    try:
        optimized_process_receiver(result, template)
    except ValueError:
        refused = True
    compare(checks, "consumer_refuses_failure", refused)
    compare(
        checks,
        "explicit_failure_retained",
        bool(optimization.failure_reason)
        and all(not a["success"] and a.get("failure_reason") for a in optimization.attempts)
        and all("objective" not in t for t in optimization.trials if not t["success"]),
        optimization.failure_reason,
    )
    return {"checks": checks, "result": result.to_dict()}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    reference_checks = self_check()
    began, completed, evaluations, current = monotonic(), [], 0, "starting"
    stop = Event()

    def status():
        elapsed = max(monotonic() - began, 1e-9)
        rate = len(completed) / elapsed
        return {
            "stage": current,
            "completed": len(completed),
            "total": 20,
            "physical_evaluations": evaluations,
            "elapsed_s": elapsed,
            "units_per_s": rate,
            "eta_s": (20 - len(completed)) / rate if rate else None,
        }

    def heartbeat():
        while not stop.wait(30):
            print(json.dumps({"heartbeat": status()}), flush=True)

    def progress(row):
        nonlocal evaluations
        evaluations += 1
        if row["physical_evaluations"] % 25 == 0:
            print(json.dumps({"evaluation": row, **status()}), flush=True)

    heartbeat_thread = Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    units = [
        (f"design-{i + 1}-start-{s}", lambda i=i, s=s: design_unit(i, s, progress))
        for i in range(8)
        for s in (0.2, 0.8)
    ]
    units += [
        ("world-wait", lambda: consumer_unit(0, False, progress, output)),
        ("world-event-resume", lambda: consumer_unit(3, True, progress, output)),
        ("infeasible", lambda: negative_unit("infeasible", progress)),
        ("budget", lambda: negative_unit("budget", progress)),
    ]
    try:
        for current, function in units:
            print(json.dumps({"unit_start": current, **status()}), flush=True)
            start = monotonic()
            try:
                unit = function()
            except Exception as exc:
                unit = {
                    "checks": [
                        {"name": f"unavailable_{i}", "passed": False, "actual": str(exc)}
                        for i in range(8)
                    ],
                    "exception": traceback.format_exc(),
                }
            unit.update(unit=current, elapsed_s=monotonic() - start)
            (output / f"{current}.json").write_text(json.dumps(unit, indent=2), encoding="utf-8")
            summary = {
                k: v for k, v in unit.items() if k not in {"result", "state", "continuation"}
            }
            summary["physical_evaluations"] = (
                unit.get("result", {}).get("optimization", {}).get("physical_evaluations", 0)
            )
            completed.append(summary)
            print(
                json.dumps(
                    {
                        "unit_done": current,
                        "passed": sum(c["passed"] for c in unit["checks"]),
                        **status(),
                    }
                ),
                flush=True,
            )
    finally:
        stop.set()
        heartbeat_thread.join()
    summary = {
        "scope": "development_authored_ideal_local_design_and_original_consumers",
        "declared_units": 20,
        "executed_units": len(completed),
        "declared_comparisons": 160,
        "executed_comparisons": sum(len(u["checks"]) for u in completed),
        "passed_units": sum(all(c["passed"] for c in u["checks"]) for u in completed),
        "passed_comparisons": sum(c["passed"] for u in completed for c in u["checks"]),
        "physical_evaluations": evaluations,
        "elapsed_s": monotonic() - began,
        "reference_self_checks": reference_checks,
        "units": completed,
        "failures": [
            {"unit": u["unit"], **c} for u in completed for c in u["checks"] if not c["passed"]
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"units", "failures"}}), flush=True
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
