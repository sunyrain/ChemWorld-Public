"""Run all fixed 52D units, retaining original actions and every failure."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from sbml_profile52_cases import CASES, SLOTS
from sbml_profile52_reference import affine_baseline, affine_point, affine_sse, pool_point

from chemworld.data.sbml_calibration import SBMLCalibration
from chemworld.data.sbml_profile import GAUSSIAN_KNOWN_SIGMA, sbml_profile_likelihood
from chemworld.examples.sbml_calibration import fixed_inputs
from chemworld.examples.sbml_profile import affine_inputs


def snapshot(blueprint):
    return {
        "table": blueprint.table.to_dict(),
        "parameters": [p.to_dict() for p in blueprint.parameters],
        "experiments": [
            {
                "composition": e.composition.to_dict(),
                "actions": e.actions,
                "readouts": [vars(r) for r in e.readouts],
                "seed": e.seed,
            }
            for e in blueprint.experiments
        ],
        "runs": list(blueprint.runs),
        "jacobian_runs": list(blueprint.jacobian_runs),
        "fit_started": blueprint._fit_started,
    }


class Unit:
    def __init__(self, case):
        self.case = case
        kind = case[1] if case[1] in SLOTS else "affine"
        self.record = {
            "unit": case[0],
            "checks": [
                {"label": slot, "status": "not_executed", "passed": False} for slot in SLOTS[kind]
            ],
            "profiles": [],
            "exceptions": [],
        }
        self.last_progress = time.monotonic()

    def check(self, label, actual, expected=True, *, numeric=False, atol=2e-5, rtol=0):
        slot = next(item for item in self.record["checks"] if item["label"] == label)
        passed = (
            bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
            if numeric
            else actual == expected
        )
        slot.update(status="evaluated", passed=bool(passed), actual=actual, expected=expected)

    def progress(self, data):
        if (
            time.monotonic() - self.last_progress >= 20
            or data.get("profile_stage") == "point_complete"
        ):
            print(json.dumps({"unit": self.case[0], **data}), flush=True)
            self.last_progress = time.monotonic()

    def run(self, inputs, *, grid=None, parameter=None, per_fit=300, **options):
        blueprint = SBMLCalibration(*inputs, max_forward_groups=per_fit)
        before = snapshot(blueprint)
        result = sbml_profile_likelihood(
            blueprint,
            parameter or self.case[2],
            self.case[3] if grid is None else grid,
            noise_model=GAUSSIAN_KNOWN_SIGMA,
            progress=self.progress,
            **options,
        )
        self.record["profiles"].append(result)
        return result, snapshot(blueprint) == before


def fits(report):
    return [report["baseline"], *(point["fit"] for point in report["points"] if point["fit"])]


def resource_consistent(report):
    sources = fits(report)
    for key, total in report["resources"].items():
        if not np.isclose(
            sum(fit["resources"][key] for fit in sources), total, rtol=1e-12, atol=1e-12
        ):
            return False
    for fit in sources:
        steps = [step for run in fit["runs"] for step in run["steps"]]
        resources = fit["resources"]
        if resources["forward_groups"] != len(fit["runs"]) or resources[
            "attempted_operations"
        ] != len(steps):
            return False
        if resources["committed_operations"] != sum(
            s["transaction_status"] == "committed" for s in steps
        ):
            return False
        for key in ("actual_duration_s", "analysis_effort_s", "cost_delta", "sample_delta_L"):
            if not np.isclose(sum(s[key] for s in steps), resources[key], rtol=1e-12, atol=1e-12):
                return False
        if resources["tangent_rhs_evaluations"] != sum(
            g["analysis"]["rhs_evaluations"] for run in fit["jacobian_runs"] for g in run["groups"]
        ):
            return False
    return True


def train_only(report):
    return all(
        run["split"] == "train" and run["stage"] != "heldout"
        for fit in fits(report)
        for run in fit["runs"]
    )


def numerical(unit):
    _, kind, parameter, grid = unit.case
    pool = kind in {"decay", "reversible", "confounded"}
    inputs = (
        fixed_inputs(kind=kind)
        if pool
        else affine_inputs(
            scaled=kind == "scaled",
            coordinate="log" if kind == "scaled" else "linear",
            b_upper=0.55 if kind == "bound" else 2.0,
        )
    )
    report, immutable = unit.run(inputs)
    baseline = 0.0 if pool else affine_baseline(0.55 if kind == "bound" else 2.0)["sse"]
    expected = [
        pool_point(kind, value)
        if pool
        else affine_point(parameter, value, 0.55 if kind == "bound" else 2.0)
        for value in grid
    ]
    atol, rtol = (2e-3, 2e-5) if pool else (2e-5, 0)
    unit.check("baseline_sse", report["baseline"]["objective"], baseline, numeric=True, atol=2e-5)
    unit.check(
        "point_sse",
        [p["weighted_SSE"] for p in report["points"]],
        [e[0] for e in expected],
        numeric=True,
        atol=atol,
        rtol=rtol,
    )
    unit.check(
        "delta",
        [p["delta_chi2"] for p in report["points"]],
        [e[0] - baseline for e in expected],
        numeric=True,
        atol=2e-5 if kind == "confounded" else atol,
        rtol=rtol,
    )
    unit.check("resources", resource_consistent(report))
    unit.check("immutable", immutable)
    sides = {c["side"] for c in report["crossing_brackets"]}
    if kind == "decay":
        unit.check(
            "no_optimizer", all(p["fit"]["optimizer"]["nfev"] == 0 for p in report["points"])
        )
        unit.check(
            "no_tangents",
            all(p["fit"]["resources"]["jacobian_calls"] == 0 for p in report["points"]),
        )
        unit.check(
            "fixed_values", [p["fit"]["estimate"][parameter] for p in report["points"]], list(grid)
        )
        unit.check("crossings", sides == {"left", "right"})
        unit.check("no_heldout", train_only(report))
    elif kind == "confounded":
        unit.check(
            "nuisance_sum",
            [sum(p["fit"]["estimate"].values()) for p in report["points"]],
            [0.78] * len(grid),
            numeric=True,
        )
        unit.check("rank_deficient", "baseline_rank_deficient" in report["limitations"])
        unit.check("flat", "profile_flat_on_declared_grid" in report["limitations"])
        unit.check(
            "missing_crossings", "threshold_not_bracketed_on_both_sides" in report["limitations"]
        )
        unit.check("no_ci", report["confidence_interval"] is None)
    else:
        nuisance = "k2" if pool else "b" if parameter == "a" else "a"
        unit.check(
            "nuisance",
            [p["fit"]["estimate"][nuisance] for p in report["points"]],
            [e[1] for e in expected],
            numeric=True,
        )
        unit.check(
            "fixed_values", [p["fit"]["estimate"][parameter] for p in report["points"]], list(grid)
        )
        unit.check(
            "converged",
            report["baseline"]["status"] == "fitted" and report["counts"]["converged"] == len(grid),
        )
        unit.check(
            "crossings",
            "nuisance_has_active_bounds" in report["limitations"]
            if kind == "bound"
            else sides == {"left", "right"},
        )
        unit.check(
            "public_histories",
            train_only(report)
            and all(
                run["steps"][0]["action"]["operation"] == "set_model_parameters"
                and all(r["native"] is not None for r in run["readouts"])
                for fit in fits(report)
                for run in fit["runs"]
            ),
        )


def grids(unit):
    inputs = affine_inputs()
    narrow, unchanged1 = unit.run(inputs)
    full_grid = (0.01, 0.5, 1.0, 2.0, 3.0)
    full, unchanged2 = unit.run(inputs, grid=full_grid)
    for label, result in (("narrow_sse", narrow), ("full_sse", full)):
        unit.check(
            label,
            [p["weighted_SSE"] for p in result["points"]],
            [affine_point("a", p["value"])[0] for p in result["points"]],
            numeric=True,
        )
    unit.check("no_narrow_crossings", narrow["crossing_brackets"] == [])
    unit.check("limited", all(r["status"] == "limited_discrete_profile" for r in (narrow, full)))
    unit.check(
        "boundary_values",
        [p["value"] for p in full["points"] if p["at_parameter_bound"]],
        [0.01, 3.0],
    )
    unit.check("boundary_flags", "grid_reaches_parameter_bound" in full["limitations"])
    unit.check("denominator", [narrow["counts"]["converged"], full["counts"]["converged"]], [3, 5])
    unit.check("no_ci", all(r["confidence_interval"] is None for r in (narrow, full)))
    unit.check("resources", all(resource_consistent(r) for r in (narrow, full)))
    unit.check("immutable", unchanged1 and unchanged2)


def poison(unit):
    inputs = affine_inputs()
    table, parameters, experiments = inputs
    altered = replace(
        table,
        rows=tuple(
            replace(
                row,
                values={
                    name: replace(value, value=value.value + 100, uncertainty=0.0001)
                    if name != "time"
                    else value
                    for name, value in row.values.items()
                },
            )
            if row.split == "validation"
            else row
            for row in table.rows
        ),
    )
    clean, unchanged1 = unit.run(inputs)
    poisoned, unchanged2 = unit.run((altered, parameters, experiments))
    unit.check("same_baseline", clean["baseline"]["objective"], poisoned["baseline"]["objective"])
    for label, field in (("same_points", "weighted_SSE"), ("same_delta", "delta_chi2")):
        unit.check(
            label, [p[field] for p in clean["points"]], [p[field] for p in poisoned["points"]]
        )
    unit.check(
        "same_nuisance",
        [p["fit"]["estimate"] for p in clean["points"]],
        [p["fit"]["estimate"] for p in poisoned["points"]],
    )
    unit.check("same_crossings", clean["crossing_brackets"], poisoned["crossing_brackets"])
    unit.check("train_only", train_only(clean) and train_only(poisoned))
    unit.check(
        "train_tangents",
        all(
            g["group_id"] == "train"
            for report in (clean, poisoned)
            for fit in fits(report)
            for run in fit["jacobian_runs"]
            for g in run["groups"]
        ),
    )
    unit.check("same_resources", clean["resources"], poisoned["resources"])
    unit.check(
        "heldout_preserved",
        len([r for r in altered.rows if r.split == "validation"]) == 5
        and all(
            r.values["reaction_rate_signal"].value > 100
            for r in altered.rows
            if r.split == "validation"
        ),
    )
    unit.check("immutable", unchanged1 and unchanged2)


def baseline(unit):
    report, unchanged = unit.run(affine_inputs(), max_nfev=1)
    base, point = report["baseline"], report["points"][0]
    unit.check("not_converged", base["status"], "optimizer_not_converged")
    unit.check("baseline_sse", base["objective"], affine_sse(0.3, 0.2), numeric=True)
    unit.check("point_sse", point["weighted_SSE"], affine_sse(1.0, 0.2), numeric=True)
    unit.check(
        "negative_delta",
        point["delta_chi2"] < 0
        and point["delta_chi2"] == point["weighted_SSE"] - base["objective"],
    )
    unit.check("baseline_flag", any("improves_baseline" in flag for flag in report["limitations"]))
    unit.check("limited", report["status"], "limited_discrete_profile")
    unit.check("estimate_retained", base["estimate"], {"a": 0.3, "b": 0.2})
    unit.check("no_ci", report["confidence_interval"] is None)
    unit.check("resources", resource_consistent(report))
    unit.check("immutable", unchanged)


def budgets(unit):
    reports, unchanged = [], []
    for options in (
        {"max_total_forward_groups": 1},
        {"max_total_forward_groups": 9},
        {"per_fit": 1},
    ):
        report, same = unit.run(affine_inputs(), **options)
        reports.append(report)
        unchanged.append(same)
    one, nine, perfit = reports
    unit.check(
        "total_cap",
        [one["resources"]["forward_groups"], nine["resources"]["forward_groups"]],
        [1, 9],
    )
    unit.check("per_fit_cap", all(fit["resources"]["forward_groups"] <= 1 for fit in fits(perfit)))
    unit.check(
        "denominator", all(r["counts"]["planned"] == 5 and len(r["points"]) == 5 for r in reports)
    )
    unit.check(
        "unattempted", [one["counts"]["not_attempted"], nine["counts"]["not_attempted"]], [5, 4]
    )
    unit.check("baseline_failure", one["baseline"]["status"], "forward_failed")
    unit.check("point_failure", nine["points"][0]["status"], "forward_failed")
    unit.check(
        "no_fake_objective",
        all(
            p["weighted_SSE"] is None
            for r in reports
            for p in r["points"]
            if p["status"] == "not_attempted"
        ),
    )
    unit.check("no_ci", all(r["confidence_interval"] is None for r in reports))
    unit.check("resources", all(resource_consistent(r) for r in reports))
    unit.check("immutable", all(unchanged))


def invalid(unit):
    blueprint = SBMLCalibration(*affine_inputs())
    attempts = (
        ("unknown_parameter", {"parameter_id": "unknown"}),
        ("empty_grid", {"grid": ()}),
        ("duplicate_grid", {"grid": (1.0, 1.0)}),
        ("unsorted_grid", {"grid": (1.0, 0.5)}),
        ("outside_bounds", {"grid": (4.0,)}),
        ("nonfinite_grid", {"grid": (float("inf"),)}),
        ("noise_model", {"noise_model": "unknown_sigma"}),
        ("threshold", {"delta_chi2_threshold": 0}),
    )
    for label, changes in attempts:
        options = {
            "calibration": blueprint,
            "parameter_id": "a",
            "grid": unit.case[3],
            "noise_model": GAUSSIAN_KNOWN_SIGMA,
            **changes,
        }
        failure = None
        try:
            sbml_profile_likelihood(**options)
        except ValueError as error:
            failure = str(error)
        unit.check(label, failure is not None and not blueprint.runs)
        unit.record.setdefault("rejections", []).append({"case": label, "failure": failure})
    table, parameters, experiments = affine_inputs()
    first = experiments[0]
    physics = first.composition.physics
    model = physics.models_of("sbml")["formal_workspace"]
    noisy = replace(
        first,
        composition=replace(
            first.composition,
            physics=replace(
                physics,
                models={
                    **physics.models,
                    "sbml": {"formal_workspace": replace(model, readout_relative_sd=0.01)},
                },
            ),
        ),
    )
    failure = None
    try:
        SBMLCalibration(table, parameters, (noisy, *experiments[1:]))
    except ValueError as error:
        failure = str(error)
    unit.check("noisy_gauge", failure is not None)
    unit.record.setdefault("rejections", []).append({"case": "noisy_gauge", "failure": failure})
    constrained = replace(
        first,
        composition=replace(first.composition, task=replace(first.composition.task, budget=1)),
    )
    failed, _ = unit.run((table, parameters, (constrained, *experiments[1:])))
    unit.check(
        "runtime_failure",
        failed["baseline"]["status"] == "forward_failed"
        and failed["resources"]["attempted_operations"] > 0
        and bool(failed["baseline"]["runs"][0]["failure"])
        and resource_consistent(failed),
    )


def compact_report(report):
    result = {key: value for key, value in report.items() if key not in {"baseline", "points"}}

    def compact_fit(fit):
        if fit is None:
            return None
        return {
            key: value
            for key, value in fit.items()
            if key not in {"runs", "jacobian_runs", "dataset", "sources", "parameters"}
        }

    result["baseline"] = compact_fit(report["baseline"])
    result["points"] = [
        {**{key: value for key, value in p.items() if key != "fit"}, "fit": compact_fit(p["fit"])}
        for p in report["points"]
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise ValueError("preserve every attempt; use new output paths")
    started = time.monotonic()
    payload = {
        "schema": "chemworld-52D-fixed-development-qualification-1",
        "planned_units": 12,
        "planned_comparisons": 120,
        "units": [],
        "evidence_mode": "development",
    }
    for index, case in enumerate(CASES):
        unit = Unit(case)
        print(
            json.dumps(
                {
                    "stage": "unit_start",
                    "unit": case[0],
                    "completed_units": index,
                    "planned_units": 12,
                }
            ),
            flush=True,
        )
        try:
            handler = {
                "grid": grids,
                "poison": poison,
                "baseline": baseline,
                "budgets": budgets,
                "invalid": invalid,
            }.get(case[1], numerical)
            handler(unit)
        except Exception as error:
            unit.record["exceptions"].append(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        payload["units"].append(unit.record)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        elapsed = time.monotonic() - started
        print(
            json.dumps(
                {
                    "stage": "unit_complete",
                    "unit": case[0],
                    "completed_units": index + 1,
                    "planned_units": 12,
                    "passed": sum(c["passed"] for c in unit.record["checks"]),
                    "comparisons": 10,
                    "elapsed_s": elapsed,
                    "units_per_s": (index + 1) / elapsed,
                    "eta_s": elapsed * (11 - index) / (index + 1),
                }
            ),
            flush=True,
        )
    checks = [c for unit in payload["units"] for c in unit["checks"]]
    payload.update(
        elapsed_s=time.monotonic() - started,
        attempted_units=len(payload["units"]),
        passed_units=sum(all(c["passed"] for c in unit["checks"]) for unit in payload["units"]),
        evaluated_comparisons=sum(c["status"] == "evaluated" for c in checks),
        passed_comparisons=sum(c["passed"] for c in checks),
        failed_comparisons=sum(c["status"] == "evaluated" and not c["passed"] for c in checks),
        not_executed_comparisons=sum(c["status"] == "not_executed" for c in checks),
    )
    payload["failures"] = [
        {"unit": unit["unit"], **check}
        for unit in payload["units"]
        for check in unit["checks"]
        if not check["passed"]
    ]
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        **payload,
        "full_evidence": str(args.output),
        "units": [
            {**unit, "profiles": [compact_report(report) for report in unit["profiles"]]}
            for unit in payload["units"]
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"units", "failures"}}
        ),
        flush=True,
    )
    if payload["passed_comparisons"] != 120:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
