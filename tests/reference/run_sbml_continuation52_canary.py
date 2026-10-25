"""Fixed three-unit, eighteen-comparison 52F canary, all attempts retained."""

import argparse
import json
import time
import traceback
from pathlib import Path

from sbml_continuation52_reference import reference_checks

from chemworld.examples.sbml_calibration import pool_model
from chemworld.examples.sbml_continuation import branch_options, example_report, fold_model
from chemworld.physchem.sbml_continuation import sbml_steady_branch
from chemworld.physchem.sbml_system import SBMLSystemSpec


def run_unit(index, record):
    if index == 2:
        consumer = example_report()
        record["consumer"] = consumer
        report = consumer["analysis"]
        record["analysis"] = report
        checks = reference_checks(report)
        checks[4]["passed"] &= all(consumer[k] for k in ("world_unchanged", "json_replay_exact"))
        checks[4]["passed"] &= consumer["transactions"] == ["committed", "committed"]
        checks[4]["passed"] &= report["initial_coordinates"]["P:k"] == 0.8
        checks[4]["passed"] &= report["start_time_s"] == 0.1
    elif index == 1:
        spec = SBMLSystemSpec(
            pool_model(kind="reversible", initial_b=0.2), controlled_parameters=("k1", "k2")
        )
        current = dict(spec.frame.initial_values)
        current["N:A"], current["N:B"] = 0.9, 0.4
        report = sbml_steady_branch(
            spec,
            current,
            parameter="k1",
            start_time_s=0,
            amount_scales_mol={"A": 1, "B": 1},
            parameter_scale=1,
            parameter_bounds=(0.05, 5),
            maximum_points=12,
        )
        record["source"] = spec.to_dict()
        record["analysis"] = report
        checks = reference_checks(report, closed=True)
    else:
        spec = SBMLSystemSpec(fold_model(), controlled_parameters=("mu",))
        report = sbml_steady_branch(
            spec, spec.frame.initial_values, start_time_s=0, **branch_options()
        )
        record["source"] = spec.to_dict()
        record["analysis"] = report
        checks = reference_checks(report)
    record["analysis"], record["comparisons"] = report, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results, started = [], time.perf_counter()
    for index, name in enumerate(("two-folds", "current-conserved-pool", "actual-world")):
        record = {"unit": name, "comparisons": []}
        try:
            run_unit(index, record)
        except Exception as error:
            record.update(exception=str(error), traceback=traceback.format_exc())
        while len(record["comparisons"]) < 6:
            record["comparisons"].append({"name": "unexecuted", "passed": False})
        record["passed"] = all(c["passed"] for c in record["comparisons"])
        results.append(record)
        (args.output / (name + ".json")).write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
        )
        elapsed = time.perf_counter() - started
        print(
            f"52F canary {index + 1}/3 pass={record['passed']}; "
            f"{(index + 1) / elapsed:.2f} unit/s; ETA={(2 - index) * elapsed / (index + 1):.1f}s",
            flush=True,
        )
    summary = {
        "units": 3,
        "comparisons": 18,
        "units_passed": sum(r["passed"] for r in results),
        "comparisons_passed": sum(c["passed"] for r in results for c in r["comparisons"]),
        "elapsed_s": time.perf_counter() - started,
        "failures": [
            {"unit": r["unit"], "comparison": c["name"], "exception": r.get("exception")}
            for r in results
            for c in r["comparisons"]
            if not c["passed"]
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
