"""Fixed independent closed-form and actual public-history qualification for 52B."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from math import isclose
from pathlib import Path

from sbml_calibration52_cases import CASES
from sbml_calibration52_reference import reference

from chemworld.data.sbml_calibration import SBMLCalibration
from chemworld.examples.sbml_calibration import TIMES, fixed_inputs, pool_model
from chemworld.physchem.sbml_sensitivity import sbml_trajectory_sensitivities
from chemworld.physchem.sbml_system import SBMLObservable, SBMLSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import validate_model_state


class Record:
    def __init__(self, unit):
        self.unit, self.checks, self.evidence = unit, [], {}

    def check(self, label, actual, expected, *, numeric=False, atol=2e-8, rtol=2e-6):
        passed = (
            isclose(actual, expected, rel_tol=rtol, abs_tol=atol) if numeric else actual == expected
        )
        self.checks.append(
            {"label": label, "actual": actual, "expected": expected, "passed": bool(passed)}
        )


def progress(record):
    if record["completed_forward_groups"] % 10 == 0 or record["failure"]:
        print(json.dumps(record), flush=True)


def independent_inputs(**kwargs):
    """Replace every authored response with the separate reference implementation."""
    table, parameters, experiments = fixed_inputs(**kwargs)
    rows = []
    for row in table.rows:
        group = int(row.group_id.rsplit("-", 1)[-1])
        expected = reference(
            row.values["time"].value,
            kind=kwargs.get("kind", "decay"),
            initial_a=(0.8, 0.45, 1.1)[group],
            initial_b=0.2 if kwargs.get("kind") == "reversible" else 0.0,
        )
        rows.append(
            replace(
                row,
                values={
                    key: replace(value, value=expected[key][0]) if key in expected else value
                    for key, value in row.values.items()
                },
            )
        )
    return replace(table, rows=tuple(rows)), parameters, experiments


def trajectory(record, case):
    kind, scaled = case["kind"], case.get("scaled", False)
    table, parameters, experiments = independent_inputs(kind=kind, scaled=scaled)
    calibration = SBMLCalibration(table, parameters, experiments)
    values = {p.parameter_id: {"k1": 0.55, "k2": 0.23}[p.parameter_id] for p in parameters}
    native = {**calibration.forward(values, "train"), **calibration.forward(values, "validation")}
    analyses = []
    for group, experiment in enumerate(experiments):
        analysis = calibration.sensitivity(experiment, values)
        analyses.append(analysis)
        record.check(f"group{group}/status", analysis["status"], "complete")
        for index, entry in enumerate(analysis["records"]):
            expected = reference(
                entry["time_s"],
                kind=kind,
                initial_a=(0.8, 0.45, 1.1)[group],
                initial_b=0.2 if kind == "reversible" else 0.0,
                source_seconds=60 if scaled else 1.0,
            )
            for channel, reading in entry["readouts"].items():
                label = f"group{group}/t{index}/{channel}"
                value, derivatives = expected[channel]
                record.check(label + "/analysis", reading["value"], value, numeric=True)
                record.check(
                    label + "/public",
                    native[(f"pools-{group}-{index}", channel)],
                    value,
                    numeric=True,
                )
                for name, slope in derivatives.items():
                    record.check(
                        label + "/derivative/" + name,
                        reading["derivative"][name],
                        slope,
                        numeric=True,
                    )
                    expected_normalized = (
                        slope * analysis["controls"][name] / value if value else None
                    )
                    record.check(
                        label + "/normalized/" + name,
                        reading["normalized"][name],
                        expected_normalized,
                        numeric=value != 0,
                    )
    x = [p.encode(p.initial) for p in parameters]
    weighted = calibration.training_jacobian(x)
    for row_index, (row_id, quantity) in enumerate(calibration.pairs["train"]):
        row = calibration.rows[row_id]
        group = int(row.group_id.rsplit("-", 1)[-1])
        expected = reference(
            row.values["time"].value,
            kind=kind,
            initial_a=(0.8, 0.45, 1.1)[group],
            initial_b=0.2 if kind == "reversible" else 0.0,
            k1=0.3,
            k2=0.2,
        )
        for column, p in enumerate(parameters):
            slope = expected[quantity][1][p.source_id] * p.decode_derivative(x[column]) / 0.002
            record.check(
                f"weighted/{row_index}/{column}",
                float(weighted[row_index, column]),
                slope,
                numeric=True,
            )
    resources = {
        **calibration._resource_report(),
        "standalone_tangent_groups": len(analyses),
        "standalone_tangent_rhs_evaluations": sum(a["rhs_evaluations"] for a in analyses),
    }
    record.evidence.update(
        analyses=analyses,
        native_runs=calibration.runs,
        jacobian_runs=calibration.jacobian_runs,
        resources=resources,
        source_xml=[
            e.composition.physics.models_of("sbml")["formal_workspace"].model.to_xml()
            for e in experiments
        ],
    )


def zero(record):
    model = SBMLSystemSpec(
        pool_model(),
        observables=(SBMLObservable("amount", "B"),),
        controlled_parameters=("k1",),
        readout_relative_sd=0.0,
        rtol=2e-10,
        atol=1e-12,
    )
    analysis = sbml_trajectory_sensitivities(model, model.frame.initial_values, {"k1": 0.55}, TIMES)
    record.check("status", analysis["status"], "complete")
    for index, entry in enumerate(analysis["records"]):
        actual = entry["readouts"]["amount_B"]
        value, slopes = reference(entry["time_s"])["amount_B"]
        record.check(f"t{index}/value", actual["value"], value, numeric=True)
        record.check(f"t{index}/derivative", actual["derivative"]["k1"], slopes["k1"], numeric=True)
        normalized = slopes["k1"] * 0.55 / value if value else None
        record.check(
            f"t{index}/normalized", actual["normalized"]["k1"], normalized, numeric=value != 0
        )
    record.evidence.update(
        analysis=analysis,
        resources={
            "standalone_tangent_groups": 1,
            "standalone_tangent_rhs_evaluations": analysis["rhs_evaluations"],
        },
    )


def fitted(record, case):
    kind, boundary = case["kind"], case.get("boundary", False)
    inputs = independent_inputs(kind=kind, boundary=boundary)
    result = SBMLCalibration(*inputs, progress=progress).fit()
    record.evidence.update(fit=result, resources=result["resources"], table=inputs[0].to_dict())
    record.check("fit_status", result["status"], "fitted")
    if result["status"] != "fitted":
        return
    if boundary:
        record.check(
            "bounded_estimate", result["estimate"]["k1"], 0.4, numeric=True, atol=2e-5, rtol=0.0
        )
        record.check("upper_bound_active", result["active_bounds"]["k1"], 1)
    else:
        for split in ("training", "validation"):
            record.check(split + "_rmse", result[split]["weighted_rmse"] <= 0.01, True)
        if kind == "confounded":
            record.check(
                "sum_recovery",
                sum(result["estimate"].values()),
                0.78,
                numeric=True,
                atol=2e-6,
                rtol=2e-4,
            )
        else:
            for name, value in result["estimate"].items():
                record.check(
                    name + "_recovery",
                    value,
                    {"k1": 0.55, "k2": 0.23}[name],
                    numeric=True,
                    atol=2e-6,
                    rtol=2e-4,
                )
    record.check(
        "local_rank",
        result["identifiability"]["rank"],
        1 if kind == "confounded" else len(inputs[1]),
    )
    record.check(
        "heldout_not_used_in_optimizer",
        all(run["stage"] == "heldout" for run in result["runs"] if run["split"] == "validation"),
        True,
    )
    record.check(
        "heldout_not_used_in_tangents",
        all(g["group_id"] != "pools-2" for r in result["jacobian_runs"] for g in r["groups"]),
        True,
    )
    record.check(
        "actual_public_packets",
        all(reading["native"] is not None for run in result["runs"] for reading in run["readouts"]),
        True,
    )
    record.check(
        "actual_control_actions",
        all(
            run["steps"][0]["action"]["operation"] == "set_model_parameters"
            for run in result["runs"]
        ),
        True,
    )


def scope(record):
    table, parameters, experiments = independent_inputs()
    experiment = experiments[0]
    scenarios = {}
    for defect in ("noise", "second_control", "wrong_unit", "unknown_control"):
        trial_parameters, trial_experiments = parameters, experiments
        if defect == "noise":
            physics = experiment.composition.physics
            model = physics.models_of("sbml")["formal_workspace"]
            composition = replace(
                experiment.composition,
                physics=replace(
                    physics,
                    models={
                        **physics.models,
                        "sbml": {"formal_workspace": replace(model, readout_relative_sd=0.01)},
                    },
                ),
            )
            trial_experiments = (replace(experiment, composition=composition), *experiments[1:])
        elif defect == "second_control":
            actions = list(experiment.actions)
            actions[2] = actions[0]
            trial_experiments = (replace(experiment, actions=tuple(actions)), *experiments[1:])
        elif defect == "wrong_unit":
            trial_parameters = (
                replace(parameters[0], column=replace(parameters[0].column, unit="K")),
            )
        else:
            trial_parameters = (replace(parameters[0], source_id="undeclared"),)
        error = None
        try:
            SBMLCalibration(table, trial_parameters, trial_experiments)
        except ValueError as exc:
            error = str(exc)
        scenarios[defect] = error
        record.check(defect + "_rejected", error is not None, True)
    record.evidence["rejections"] = scenarios


def budget(record):
    table, parameters, experiments = independent_inputs()
    row = table.rows[1]
    table = replace(
        table,
        rows=(
            table.rows[0],
            replace(
                row,
                values={
                    **row.values,
                    "amount_A": replace(row.values["amount_A"], quality="missing", value=None),
                },
            ),
            *table.rows[2:],
        ),
    )
    result = SBMLCalibration(table, parameters, experiments, max_forward_groups=1).fit()
    record.check("failure_status", result["status"], "forward_failed")
    record.check("executed_group_count", result["resources"]["forward_groups"], 1)
    record.check("operations_retained", result["resources"]["committed_operations"], 12)
    record.check("excluded_count", len(result["excluded_observations"]), 1)
    record.check("budget_reason", any("budget" in e for e in result["failures"]), True)
    record.evidence.update(fit=result, resources=result["resources"])


def holdout(record):
    table, parameters, experiments = independent_inputs()
    first = SBMLCalibration(table, parameters, experiments, progress=progress).fit()
    perturbed = replace(
        table,
        rows=tuple(
            replace(
                row,
                values={
                    **row.values,
                    "amount_A": replace(
                        row.values["amount_A"], value=row.values["amount_A"].value + 0.2
                    ),
                },
            )
            if row.split == "validation"
            else row
            for row in table.rows
        ),
    )
    second = SBMLCalibration(perturbed, parameters, experiments, progress=progress).fit()
    record.check("first_status", first["status"], "fitted")
    record.check("second_status", second["status"], "fitted")
    if first["status"] == second["status"] == "fitted":
        record.check(
            "unchanged_training_estimate",
            first["estimate"]["k1"],
            second["estimate"]["k1"],
            numeric=True,
            atol=1e-10,
            rtol=0,
        )
        record.check("heldout_error_increases", second["validation"]["weighted_rmse"] > 50, True)
    record.evidence.update(
        fits=[first, second],
        resources={
            key: first["resources"][key] + value for key, value in second["resources"].items()
        },
    )


def immutable(record):
    inputs = independent_inputs(initial_assignment=True)
    calibration = SBMLCalibration(*inputs)
    experiment = inputs[2][0]
    before = experiment.composition.to_dict()
    table_before = inputs[0].to_dict()
    actions_before = json.dumps(experiment.actions, sort_keys=True)
    model, _ = calibration._model_initial(experiment)
    analysis = calibration.sensitivity(experiment, {"k1": 0.8})
    spec, actions = calibration._candidate(experiment, {"k1": 0.8})
    runtime = ChemWorldRuntime(composition=spec)
    try:
        runtime.reset(seed=0)
        initial_snapshot = runtime.snapshot().to_dict()
        result = runtime.execute(actions[0])
        record.check("control_committed", result.info["transaction_status"], "committed")
        _, coordinates, _ = validate_model_state(runtime._state, model)
        record.check(
            "actual_runtime_tangent_initial_identity",
            coordinates,
            analysis["initial_coordinates_after_control"],
        )
        record.check("initial_assignment_not_recomputed", coordinates["N:A"], 0.8, numeric=True)
        record.check("control_installed", coordinates["P:k1"], 0.8, numeric=True)
        before_analysis = runtime.snapshot().to_dict()
        another = sbml_trajectory_sensitivities(model, coordinates, {"k1": 0.8}, TIMES)
        record.check("read_only_runtime_snapshot", runtime.snapshot().to_dict(), before_analysis)
        record.check(
            "initial_tangent",
            another["records"][0]["readouts"]["amount_A"]["derivative"],
            {"k1": 0.0},
        )
        record.evidence.update(
            initial_snapshot=initial_snapshot,
            control_result=result.info,
            analysis=analysis,
            actual_state_analysis=another,
            resources={
                "standalone_tangent_groups": 2,
                "standalone_tangent_rhs_evaluations": analysis["rhs_evaluations"]
                + another["rhs_evaluations"],
                "standalone_control_operations": 1,
            },
        )
    finally:
        runtime.close()
    record.check("world_unchanged", experiment.composition.to_dict() == before, True)
    record.check("table_unchanged", inputs[0].to_dict() == table_before, True)
    record.check(
        "history_unchanged", json.dumps(experiment.actions, sort_keys=True) == actions_before, True
    )


def run_block(output, summary):
    if output.exists() or summary.exists():
        raise ValueError("attempt files must be new; never overwrite results")
    started = time.perf_counter()
    records, failures = [], []
    for index, case in enumerate(CASES):
        record = Record(case["name"])
        unit_started = time.perf_counter()
        try:
            if case["type"] == "trajectory":
                trajectory(record, case)
            elif case["type"] == "fit":
                fitted(record, case)
            else:
                {
                    "zero": zero,
                    "scope": scope,
                    "budget": budget,
                    "holdout": holdout,
                    "immutable": immutable,
                }[case["type"]](record)
        except Exception as exc:
            record.check("unexpected_exception", f"{type(exc).__name__}: {exc}", None)
        failed = [{"unit": record.unit, **check} for check in record.checks if not check["passed"]]
        failures.extend(failed)
        records.append(
            {
                "unit": record.unit,
                "checks": record.checks,
                "evidence": record.evidence,
                "elapsed_s": time.perf_counter() - unit_started,
                "passed": not failed,
            }
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": "fixed_52B_block",
                    "completed": index + 1,
                    "total": len(CASES),
                    "units_per_s": (index + 1) / elapsed,
                    "eta_s": elapsed / (index + 1) * (len(CASES) - index - 1),
                    "unit": record.unit,
                    "passed": not failed,
                }
            ),
            flush=True,
        )
    totals = {}
    for entry in records:
        for key, value in entry["evidence"].get("resources", {}).items():
            totals[key] = totals.get(key, 0) + value
    fit_outcomes = []
    for entry in records:
        evidence = entry["evidence"]
        fits = [evidence["fit"]] if "fit" in evidence else evidence.get("fits", [])
        for index, fit in enumerate(fits):
            fit_outcomes.append(
                {
                    "unit": entry["unit"],
                    "fit_index": index,
                    "status": fit["status"],
                    "dataset": fit["dataset"],
                    "estimate": fit.get("estimate"),
                    "active_bounds": fit.get("active_bounds"),
                    "local_rank": fit.get("identifiability", {}).get("rank"),
                    "parameter_count": len(fit["parameters"]),
                    "training_weighted_rmse": fit.get("training", {}).get("weighted_rmse"),
                    "validation_weighted_rmse": fit.get("validation", {}).get("weighted_rmse"),
                    "excluded_observations": fit["excluded_observations"],
                    "execution_failures": fit["failures"],
                }
            )
    compact = {
        "mode": "development",
        "status": "failed" if failures else "passed",
        "units": len(records),
        "passed_units": sum(r["passed"] for r in records),
        "comparisons": sum(len(r["checks"]) for r in records),
        "passed_comparisons": sum(c["passed"] for r in records for c in r["checks"]),
        "failures": failures,
        "resources": totals,
        "fit_outcomes": fit_outcomes,
        "scope_rejections": {
            entry["unit"]: entry["evidence"]["rejections"]
            for entry in records
            if "rejections" in entry["evidence"]
        },
        "elapsed_s": time.perf_counter() - started,
        "per_unit": [
            {
                "unit": r["unit"],
                "passed": r["passed"],
                "comparisons": len(r["checks"]),
                "elapsed_s": r["elapsed_s"],
                "resources": r["evidence"].get("resources", {}),
            }
            for r in records
        ],
        "full_output": str(output.resolve()),
        "boundary": (
            "Independent analytic development qualification; no empirical or COPASI parity claim"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": compact, "units": records}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(compact, indent=2))
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = run_block(args.output, args.summary)
    if result["status"] != "passed":
        raise SystemExit(1)
