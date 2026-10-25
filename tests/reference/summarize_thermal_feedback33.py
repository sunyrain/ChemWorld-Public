"""Generate readable 33D evidence without duplicating full runtime snapshots."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    attempts = []
    for identity in ("canary01", "canary02", "canary03", "attempt01", "attempt02"):
        raw = json.loads((args.attempt_root / identity / "results.json").read_text(encoding="utf8"))
        attempt = {k: v for k, v in raw.items() if k != "rows"}
        attempt.update(
            name=identity, artifact_directory=f"%TEMP%/ChemWorld-33D/{identity}", rows=[]
        )
        for row in raw["rows"]:
            compact = {
                k: row[k]
                for k in (
                    "id",
                    "input",
                    "checks",
                    "failures",
                    "passed",
                    "elapsed_s",
                    "reference",
                    "errors",
                    "shared_continuation",
                    "replacement_continuation",
                )
                if k in row
            }
            if "native" in row:
                native = row["native"]
                compact["native_samples"] = native["samples"]
                compact["native_stopping_event"] = native["stopping_event"]
                compact["native_pauses"] = native["paused"]
                solvers = [r["solver"] for r in native["reports"]]
                compact["native_resources"] = {
                    "advances": len(solvers),
                    **{
                        key: sum(s[key] for s in solvers)
                        for key in ("solver_calls", "nfev", "njev", "nlu")
                    },
                }
                compact["native_energy_residuals_J"] = [
                    r["energy_balance_residual_J"] for r in native["reports"]
                ]
            attempt["rows"].append(compact)
        attempts.append(attempt)
    final = attempts[-1]
    native = [r["native_resources"] for r in final["rows"] if "native_resources" in r]
    refs = [
        resource
        for r in final["rows"]
        if "reference" in r
        for resource in r["reference"]["resources"]
    ]
    worst = max(
        (e["maximum_absolute_error"] / e["tolerance"], r["id"], key)
        for r in final["rows"]
        for key, e in r.get("errors", {}).items()
        if "maximum_absolute_error" in e
    )
    result = {
        "schema": "chemworld-33D-development-summary-v1",
        "development_only": True,
        "final": {
            k: final[k]
            for k in (
                "units_planned",
                "units_completed",
                "units_passed",
                "checks_planned",
                "checks_passed",
                "elapsed_s",
            )
        },
        "reference_scope": {
            "linear": "independent constant-capacity thermal/PID matrix exponential",
            "nonlinear": "independent DOP853 actuator-limited plant and controller",
            "IDAES": (
                "actual IDAES2.13.0 public P/PI controller expressions "
                "consumed by independent DOP853"
            ),
            "full_IDAES_DAE_or_IPOPT_execution": False,
            "IDAES_hard_limit_or_filtered_PID_equivalence": False,
        },
        "recorded_native_solver_resources": {
            key: sum(r[key] for r in native)
            for key in ("advances", "solver_calls", "nfev", "njev", "nlu")
        },
        "additional_actual_world_advance_actions": 23,
        "additional_actual_world_solver_counts": None,
        "resource_note": (
            "23 extra heat/wait actions include continuations and shared systems; "
            "their solver counts are not aggregated, not zero"
        ),
        "reference_solver_resources": {
            "integrations": len(refs),
            "rhs": sum(r["rhs"] for r in refs),
            "steps": sum(r["steps"] for r in refs),
        },
        "maximum_error_tolerance_ratio": {"ratio": worst[0], "unit": worst[1], "field": worst[2]},
        "functional_tests": {
            "distinct_passed": 19,
            "runs": [
                {"record": "%TEMP%/chemworld-33D-functional01.xml", "passed": 17},
                {"record": "%TEMP%/chemworld-33D-functional02.xml", "passed": 1},
                {"record": "%TEMP%/chemworld-33D-functional03.xml", "passed": 1},
            ],
        },
        "checks_by_group": {
            name: {
                "units": len(rows),
                "passed_units": sum(r["passed"] for r in rows),
                "checks": sum(len(r["checks"]) for r in rows),
                "passed_checks": sum(sum(r["checks"].values()) for r in rows),
            }
            for name, rows in (
                ("dynamics", final["rows"][:20]),
                ("actual_IDAES_control_equations", final["rows"][20:22]),
                ("world_continuation", final["rows"][22:25]),
                ("equipment_lifecycle", final["rows"][25:]),
            )
        },
        "attempts": attempts,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("attempts", "functional_tests")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
