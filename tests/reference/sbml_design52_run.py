"""Fixed 14-unit objective-design and original-world consumer qualification."""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from sbml_design52_cases import BOUNDARY_CASES, OPTIMIZATION_CASES, inputs
from sbml_design52_reference import optimum

from chemworld.examples.sbml_design import design_spec, design_world
from chemworld.physchem.sbml_design import SBMLDesignSpec, optimize_sbml_design
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_design import analyze_sbml_design_world
from chemworld.runtime.sbml_models import validate_model_state


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def work(report):
    trials = report["optimization"]["trials"]
    segments = [s for t in trials for s in t.get("diagnostics", {}).get("continuous_solves", [])]
    roots = [
        t["diagnostics"]["steady_analysis"]
        for t in trials
        if t.get("diagnostics", {}).get("steady_analysis") is not None
    ]
    return {
        "candidate_evaluations": len(trials),
        "continuous_solves": len(segments),
        "RHS": sum(s["nfev"] for s in segments),
        "Jacobian": sum(s["njev"] for s in segments),
        "LU": sum(s["nlu"] for s in segments),
        "steady_roots": len(roots),
        "steady_residual_evaluations": sum(s["root_diagnostic"]["evaluations"] for s in roots),
        "failed_candidates": sum(not t["success"] for t in trials),
    }


class Unit:
    def __init__(self, name):
        self.name, self.checks, self.details = name, [], {}

    def check(self, label, actual, expected, *, atol=2e-6, rtol=2e-6, exact=False):
        if exact:
            passed, error = actual == expected, None
        else:
            actual, expected = np.asarray(actual, float), np.asarray(expected, float)
            passed = actual.shape == expected.shape and bool(
                np.allclose(actual, expected, atol=atol, rtol=rtol)
            )
            error = float(np.max(np.abs(actual - expected)))
            actual, expected = actual.tolist(), expected.tolist()
        self.checks.append(
            {
                "label": label,
                "actual": actual,
                "expected": expected,
                "passed": bool(passed),
                "atol": None if exact else atol,
                "rtol": None if exact else rtol,
                "maximum_absolute_error": error,
            }
        )


def optimize_unit(unit, path):
    model, current, spec = inputs(unit.name)
    before = dict(current)
    report = optimize_sbml_design(model, current, spec)
    write(path / "report.json", report)
    write(path / "source-model.json", model.to_dict())
    (path / "source.xml").write_text(model.model.to_xml(), encoding="utf-8")
    expected, result = optimum(unit.name), report["optimization"]
    unit.details["work"] = work(report)
    unit.check("converged", result["success"], True, exact=True)
    names = [c.variable.parameter_id for c in spec.controls]
    unit.check(
        "optimal_controls_Hz",
        [result["values"][n] for n in names],
        [expected["controls"][n] for n in names],
        atol=2e-5,
        rtol=0.0,
    )
    unit.check("raw_objective", report["objective"]["value"], expected["objective"])
    metric_names = [m.metric_id for m in spec.metrics]
    unit.check(
        "actual_metrics",
        [result["point"]["observations"][m] for m in metric_names],
        [expected["metrics"][m] for m in metric_names],
    )
    last = result["trials"][-1]
    unit.check(
        "fresh_feasibility",
        last["fresh_verification"]
        and last["feasible"]
        and all(
            violation <= c.tolerance
            for c, violation in zip(spec.constraints, last["violations"], strict=True)
        ),
        True,
        exact=True,
    )
    factor = 60.0 if unit.name.startswith("scaled_") else 1.0
    unit.check(
        "recommended_source_controls",
        list(report["recommended_source_parameters"].values()),
        [expected["controls"][n] * factor for n in names],
    )
    unit.check(
        "actual_work_records",
        result["physical_evaluations"] == len(result["trials"])
        and all("diagnostics" in t for t in result["trials"]),
        True,
        exact=True,
    )
    unit.check("input_unchanged", current, before, exact=True)


def boundary_unit(unit, path):
    model, current, spec = inputs(unit.name)
    before = dict(current)
    report = optimize_sbml_design(model, current, spec)
    write(path / "report.json", report)
    result = report["optimization"]
    unit.details["work"] = work(report)
    unit.details["failure_reason"] = result["failure_reason"]
    unit.check("failed_optimization", result["success"], False, exact=True)
    unit.check(
        "no_recommendation",
        [report["recommended_source_parameters"], report["objective"]["value"]],
        [None, None],
        exact=True,
    )
    unit.check(
        "all_attempts_retained",
        bool(result["attempts"])
        and bool(result["trials"])
        and result["physical_evaluations"] == len(result["trials"]),
        True,
        exact=True,
    )
    failures = [t for t in result["trials"] if not t["success"]]
    reasons = [a.get("failure_reason", "") for a in result["attempts"]]
    expected_reason = (
        "design_physical_evaluation_budget_exhausted"
        if unit.name == "evaluation_budget"
        else "no_verified_converged_candidate"
    )
    unit.check("failure_reason", result["failure_reason"], expected_reason, exact=True)
    unit.check(
        "no_fabricated_penalty",
        all("objective" not in t for t in failures)
        and (
            bool(failures)
            if unit.name in {"physical_failure", "steady_seed_failure"}
            else bool(reasons)
        ),
        True,
        exact=True,
    )
    unit.check("input_unchanged", current, before, exact=True)


def actual_actions(runtime, report):
    replies = []
    for action in (
        {"operation": "set_model_parameters", "values": report["recommended_source_parameters"]},
        {"operation": "wait", "duration_s": 2.0},
        {"operation": "measure", "instrument": "formal_quantity_meter"},
    ):
        reply = runtime.execute(action)
        replies.append(
            {
                "action": action,
                "info": reply.info,
                "channels": [c.to_dict() for c in reply.observation.channels],
            }
        )
    return replies


def world_unit(unit, path):
    scaled = unit.name == "world_scaled"
    composition = design_world(scaled=scaled, initial_assignment=True)
    runtime, restored = ChemWorldRuntime(composition=composition), None
    try:
        runtime.reset(seed=52)
        runtime.execute({"operation": "wait", "duration_s": 0.4})
        before = json.loads(json.dumps(runtime.snapshot().to_dict()))
        write(path / "checkpoint.json", before)
        model = composition.physics.models_of("sbml")["formal_workspace"]
        _, current, _ = validate_model_state(runtime._state, model)
        spec = design_spec("trajectory_min")
        report = analyze_sbml_design_world(runtime._state, composition, spec)
        write(path / "report.json", report)
        write(path / "design.json", spec.to_dict())
        unit.details["work"] = work(report)
        unit.check("design_success", report["optimization"]["success"], True, exact=True)
        after_query = json.loads(json.dumps(runtime.snapshot().to_dict()))
        write(path / "after-query.json", after_query)
        unit.check("complete_query_nonmutation", after_query == before, True, exact=True)
        candidate = report["optimization"]["point"]["diagnostics"][
            "initial_coordinates_after_control"
        ]
        unit.check(
            "initialized_amounts_preserved",
            [candidate["N:A"], candidate["N:B"]],
            [current["N:A"], current["N:B"]],
            atol=2e-12,
            rtol=0.0,
        )
        expected = optimum("trajectory_min", initial_a=current["N:A"], initial_b=current["N:B"])
        unit.check(
            "current_state_optimum_Hz",
            report["optimization"]["values"]["forward_rate"],
            expected["k1"],
            atol=2e-5,
            rtol=0.0,
        )
        unit.check(
            "source_recommendation",
            report["recommended_source_parameters"]["k1"],
            expected["k1"] * (60.0 if scaled else 1.0),
        )
        initial_cost = runtime._state.ledger.cost
        actions = actual_actions(runtime, report)
        write(path / "actions.json", actions)
        unit.check(
            "normal_actions_committed",
            [r["info"]["transaction_status"] for r in actions],
            ["committed"] * 3,
            exact=True,
        )
        observation = next(c for c in actions[-1]["channels"] if c["channel_id"] == "amount_B")
        unit.check("public_endpoint_amount_mol", observation["value"], 0.6, atol=2e-6, rtol=0.0)
        unit.check("actual_clock_s", runtime._state.ledger.time_s, 2.4, atol=1e-12, rtol=0.0)
        unit.check(
            "actual_resource_debit",
            runtime._state.ledger.cost - initial_cost,
            sum(a["info"]["cost_delta"] for a in actions),
            atol=1e-15,
            rtol=0.0,
        )
        final = json.loads(json.dumps(runtime.snapshot().to_dict()))
        write(path / "final.json", final)
        restored = ChemWorldRuntime.from_snapshot(before)
        replay = analyze_sbml_design_world(
            restored._state,
            restored.compiled_composition.spec,
            SBMLDesignSpec.from_dict(json.loads(json.dumps(spec.to_dict()))),
        )
        write(path / "replay-report.json", replay)
        unit.check("JSON_design_replay", replay == report, True, exact=True)
        actual_actions(restored, replay)
        replay_final = json.loads(json.dumps(restored.snapshot().to_dict()))
        write(path / "restored.json", replay_final)
        unit.check("complete_in_process_continuation", replay_final == final, True, exact=True)
        child = subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                str(Path(__file__).resolve()),
                "--resume",
                str(path / "checkpoint.json"),
                "--design",
                str(path / "design.json"),
                "--output",
                str(path / "fresh.json"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        (path / "fresh.log").write_text(child.stdout + child.stderr, encoding="utf-8")
        fresh = json.loads((path / "fresh.json").read_text()) if child.returncode == 0 else None
        unit.check(
            "complete_fresh_process_continuation",
            fresh is not None
            and fresh["report"] == json.loads(json.dumps(report))
            and fresh["snapshot"] == final,
            True,
            exact=True,
        )
        unit.details.update(
            actual_cost=runtime._state.ledger.cost - initial_cost,
            actual_time_s=runtime._state.ledger.time_s,
            public_product_mol=observation["value"],
            actual_runtime_actions=10,
        )
    finally:
        runtime.close()
        if restored:
            restored.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(json.loads(args.resume.read_text()))
        try:
            spec = SBMLDesignSpec.from_dict(json.loads(args.design.read_text()))
            report = analyze_sbml_design_world(
                runtime._state, runtime.compiled_composition.spec, spec
            )
            actual_actions(runtime, report)
            write(args.output, {"report": report, "snapshot": runtime.snapshot().to_dict()})
        finally:
            runtime.close()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cases = (
        [(name, 8, optimize_unit) for name in OPTIMIZATION_CASES]
        + [(name, 6, boundary_unit) for name in BOUNDARY_CASES]
        + [(name, 12, world_unit) for name in ("world", "world_scaled")]
    )
    start, records = time.monotonic(), []
    for index, (name, count, callback) in enumerate(cases, 1):
        path = args.output / name
        path.mkdir()
        unit, failure = Unit(name), None
        print(f"52 design {index}/14 starting {name}", flush=True)
        try:
            callback(unit, path)
        except Exception:
            failure = traceback.format_exc()
        passed = sum(c["passed"] for c in unit.checks)
        record = {
            "name": name,
            "planned": count,
            "executed": len(unit.checks),
            "passed": passed,
            "unevaluated": count - len(unit.checks),
            "failure": failure,
            "checks": unit.checks,
            "details": unit.details,
            "success": failure is None and passed == count and len(unit.checks) == count,
        }
        records.append(record)
        write(path / "unit.json", record)
        elapsed = time.monotonic() - start
        rate = f"{index / elapsed:.2f} units/s" if elapsed > 0 else "clock tick pending"
        print(
            f"52 design {index}/14 completed {passed}/{count}; {rate} "
            f"ETA {(14 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    summary = {
        "scope": "development_parent52_native_objective_design",
        "units_planned": 14,
        "units_passed": sum(u["success"] for u in records),
        "comparisons_planned": 112,
        "comparisons_executed": sum(u["executed"] for u in records),
        "comparisons_passed": sum(u["passed"] for u in records),
        "comparisons_unevaluated": sum(u["unevaluated"] for u in records),
        "elapsed_s": time.monotonic() - start,
        "units": records,
    }
    write(args.output / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "units"}, flush=True)
    raise SystemExit(0 if summary["units_passed"] == 14 else 1)


if __name__ == "__main__":
    main()
