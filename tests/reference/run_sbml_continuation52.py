"""Fixed 52F eighteen-unit/180-comparison development qualification."""

import argparse
import json
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from sbml_continuation52_cases import CASES, build
from sbml_continuation52_reference import exact, numeric, reference_checks

from chemworld.physchem.sbml_continuation import sbml_steady_branch
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_continuation import analyze_sbml_steady_branch_world
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.sbml_world import sbml_world


def persist(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def failure_checks(report, case, unchanged):
    kind = case["failure"]
    expected = {
        "parameter": "corrector_retry_budget",
        "evaluation": "variational_evaluation_budget",
        "seed": "seed_root_failed: iteration_limit",
        "rank": "branch Jacobian",
    }[kind]
    rows, attempts, counts = report["records"], report["attempts"], report["costs"]
    witness = (
        counts["variational_rate_calls"] == 2
        if kind == "evaluation"
        else counts["root_iterations"] == 1
        if kind == "seed"
        else counts["tangent_svd_calls"] == 1 and report["initial_coordinates"]["P:k1"] == 0
        if kind == "rank"
        else all(0.05 <= r["parameter_value"] <= 0.25 for r in rows)
    )
    conditions = [
        ("declared-stop", expected in report["stop_reason"]),
        (
            "partial-status",
            report["status"] == ("failed" if kind in {"seed", "rank"} else "partial"),
        ),
        (
            "rejected-attempt-retained",
            report["rejected_attempts"] > 0 and bool(attempts[-1]["reason"]),
        ),
        (
            "fixed-budget",
            counts["variational_rate_calls"] <= report["policy"]["maximum_evaluations"],
        ),
        ("point-accounting", len(attempts) == len(rows) + report["rejected_attempts"]),
        ("input-immutable", unchanged),
        ("finite-json", json.loads(json.dumps(report, allow_nan=False)) == report),
        ("mathematical-or-budget-witness", witness),
        ("retained-roots-valid", all(r["full_scaled_rhs_norm"] < 1e-8 for r in rows)),
        (
            "remaining-points-explicit",
            len(rows) < report["requested_points"] and report["completed_points"] == len(rows),
        ),
    ]
    return [exact(name, {name: bool(value)}) for name, value in conditions]


def flux_readout_checks(report, case):
    expected_flux, actual_flux, expected_readout, actual_readout = [], [], [], []
    seconds, amount = (60.0, 0.001) if case.get("scaled") else (1.0, 1.0)
    for row in report["records"]:
        values = row["coordinates"]
        if case.get("closed"):
            a, b = values["N:A"], values["N:B"]
            flux = [values["P:k1"] / seconds * a, values["P:k2"] / seconds * b]
            quantity = {"amount_A": a, "concentration_B": b / values["V:V"]}
            quantity.update({k: flux[0] for k in row["readout"] if k.startswith("reaction_rate_")})
        else:
            x = values["N:X"]
            feed = values["P:mu"] * amount / seconds * values["N:H"] / 2
            flux = [feed, 3 * x * x, 2.5 * x, x**3]
            if case.get("coupled"):
                flux += [values["P:k"] / seconds * x, 0.4 * values["N:Y"]]
            quantity = {
                "amount_X": x,
                "concentration_X": x / values["V:V"],
                "reaction_rate_feed": feed,
            }
        actual_flux.append(list(row["reaction_fluxes_mol_s"].values()))
        expected_flux.append(flux)
        actual_readout.append(list(row["readout"].values()))
        expected_readout.append([quantity[k] for k in row["readout"]])
    return [
        numeric("original-fluxes", actual_flux, expected_flux),
        numeric("native-readouts", actual_readout, expected_readout),
    ]


def replay(input_path, output_path):
    item = json.loads(input_path.read_text(encoding="utf-8"))
    composition = WorldCompositionSpec.from_dict(item["composition"])
    runtime = ChemWorldRuntime.from_snapshot(item["snapshot"])
    try:
        before = runtime.snapshot().to_dict()
        report = analyze_sbml_steady_branch_world(runtime._state, composition, **item["options"])
        persist(
            output_path, {"analysis": report, "unchanged": runtime.snapshot().to_dict() == before}
        )
    finally:
        runtime.close()


def run_unit(case, record, directory):
    spec, current, options = build(case)
    original = dict(current)
    record.update(
        source=spec.to_dict(),
        current=current,
        options={k: asdict(v) if k == "root_policy" else v for k, v in options.items()},
        analysis_costs=[],
    )
    if case.get("world"):
        composition = sbml_world(
            spec.model,
            controlled_parameters=spec.controlled_parameters,
            observables=spec.observables,
        )
        runtime, restored = ChemWorldRuntime(composition=composition), None
        try:
            runtime.reset(seed=52)
            multiplier = 60000 if case.get("scaled") else 1
            seconds = 60 if case.get("scaled") else 1
            transaction_status = []
            for action in (
                {
                    "operation": "set_model_parameters",
                    "values": {"mu": 0.23 * multiplier, "k": 0.8 * seconds},
                },
                {"operation": "wait", "duration_s": 0.1},
            ):
                transaction_status.append(runtime.execute(action).info["transaction_status"])
            before = runtime.snapshot().to_dict()
            report = analyze_sbml_steady_branch_world(runtime._state, composition, **options)
            record["analysis"] = report
            record["analysis_costs"].append(report["costs"])
            unchanged = runtime.snapshot().to_dict() == before
            restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(before)))
            again = analyze_sbml_steady_branch_world(restored._state, composition, **options)
            record["analysis_costs"].append(again["costs"])
            replay_input = directory / (case["name"] + "-snapshot.json")
            replay_output = directory / (case["name"] + "-fresh-process.json")
            persist(
                replay_input,
                {"composition": composition.to_dict(), "snapshot": before, "options": options},
            )
            process = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    __file__,
                    "--replay",
                    str(replay_input),
                    "--output",
                    str(replay_output),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            record["fresh_process"] = {
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
            fresh = (
                json.loads(replay_output.read_text(encoding="utf-8"))
                if replay_output.exists()
                else {}
            )
            if fresh:
                record["analysis_costs"].append(fresh["analysis"]["costs"])
            preserved = exact(
                "world-input-and-replay",
                {
                    "actions_committed": transaction_status == ["committed", "committed"],
                    "complete_snapshot_unchanged": unchanged,
                    "same_process_replay": report == again,
                    "fresh_process_replay": fresh.get("analysis") == report,
                    "fresh_snapshot_unchanged": fresh.get("unchanged") is True,
                    "world_time": report["start_time_s"] == 0.1,
                    "current_other_control": report["initial_coordinates"]["P:k"] == 0.8 * seconds,
                    "query_uncharged": report["world_query"]["live_resources_charged"] is False,
                },
            )
        finally:
            runtime.close()
            if restored:
                restored.close()
    else:
        report = sbml_steady_branch(spec, current, start_time_s=0, **options)
        record["analysis"] = report
        record["analysis_costs"].append(report["costs"])
        unchanged = current == original
        if case.get("failure"):
            record["comparisons"] = failure_checks(report, case, unchanged)
            return
        rebuilt = SBMLSystemSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        again = sbml_steady_branch(
            rebuilt, json.loads(json.dumps(current)), start_time_s=0, **options
        )
        record["analysis_costs"].append(again["costs"])
        preserved = exact(
            "source-input-and-replay",
            {
                "unchanged": unchanged,
                "exact_json_replay": again == report,
                "current_seed_preserved": report["initial_coordinates"] == current,
            },
        )
    checks = reference_checks(
        report, closed=case.get("closed", False), scaled=case.get("scaled", False)
    )
    checks.extend(flux_readout_checks(report, case))
    checks.append(preserved)
    checks.append(
        exact(
            "initial-current-parameter",
            {
                "seed_parameter": report["records"][0]["parameter_value"]
                == report["initial_coordinates"]["P:" + report["parameter"]],
                "seed_arclength": report["records"][0]["predictor_arclength"] == 0,
                "no_reinitialization": report["initial_coordinates"]
                == (record["current"] if not case.get("world") else report["initial_coordinates"]),
                "explicit_physical_boundary": "undeclared" in report["energy_scope"],
            },
        )
    )
    record["comparisons"] = checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        replay(args.replay, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results, started = [], time.perf_counter()
    for index, case in enumerate(CASES, 1):
        record = {"unit": case["name"], "comparisons": []}
        try:
            run_unit(case, record, args.output)
        except Exception as error:
            record.update(exception=str(error), traceback=traceback.format_exc())
        while len(record["comparisons"]) < 10:
            record["comparisons"].append(
                {"name": "unexecuted-slot-" + str(len(record["comparisons"])), "passed": False}
            )
        record["passed"] = len(record["comparisons"]) == 10 and all(
            c["passed"] for c in record["comparisons"]
        )
        results.append(record)
        persist(args.output / f"{index:02}-{case['name']}.json", record)
        elapsed = time.perf_counter() - started
        print(
            f"52F {index}/18 {case['name']}; pass={record['passed']}; "
            f"{index / elapsed:.2f} unit/s; ETA={(18 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    costs = [cost for r in results for cost in r.get("analysis_costs", [])]
    summary = {
        "mode": "development",
        "units": 18,
        "comparisons": 180,
        "units_passed": sum(r["passed"] for r in results),
        "comparisons_passed": sum(c["passed"] for r in results for c in r["comparisons"]),
        "elapsed_s": time.perf_counter() - started,
        "analysis_calls": len(costs),
        "costs": {key: sum(c[key] for c in costs) for key in costs[0]} if costs else {},
        "source_points": sum(r.get("analysis", {}).get("completed_points", 0) for r in results),
        "source_rejected_attempts": sum(
            r.get("analysis", {}).get("rejected_attempts", 0) for r in results
        ),
        "failures": [
            {"unit": r["unit"], "comparison": c["name"], "exception": r.get("exception")}
            for r in results
            for c in r["comparisons"]
            if not c["passed"]
        ],
        "unit_outcomes": [
            {
                "unit": r["unit"],
                "passed": r["passed"],
                "points": r.get("analysis", {}).get("completed_points"),
                "status": r.get("analysis", {}).get("status"),
                "stop_reason": r.get("analysis", {}).get("stop_reason"),
                "fold_brackets": len(r.get("analysis", {}).get("fold_brackets", [])),
            }
            for r in results
        ],
        "scope": (
            "finite connected branch; synthetic analytic references, no global or COPASI claim"
        ),
    }
    persist(args.output / "summary.json", summary)
    print(
        json.dumps({k: summary[k] for k in ("units_passed", "comparisons_passed", "elapsed_s")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
