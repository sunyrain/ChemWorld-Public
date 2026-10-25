"""Publish bounded machine-readable 31D results without copying world snapshots."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    attempts = []
    for name in ("canary01", "canary02", "canary03", "attempt01", "attempt02"):
        source = read(args.attempt_root / name / "results.json")
        compact = {k: v for k, v in source.items() if k != "rows"}
        compact["name"] = name
        compact["artifact_directory"] = f"%TEMP%/ChemWorld-31D/{name}"
        compact["rows"] = []
        for row in source["rows"]:
            value = {
                k: row[k]
                for k in ("id", "checks", "passed", "failures", "elapsed_s", "errors")
                if k in row
            }
            if "reference" in row:
                value["reference"] = row["reference"]
            if "native" in row:
                native = row["native"]
                if "samples" in native:
                    value["native_samples"] = native["samples"]
                    value["native_stopping_event"] = native["stopping_event"]
                else:
                    value["native_operating_point"] = native
            resources = []
            for key in ("native", "baseline", "alternative"):
                resources.extend(row.get(key, {}).get("resources", []))
            if resources:
                value["native_resources"] = {
                    "advances": len(resources),
                    **{
                        k: sum(r.get(k, 0) for r in resources)
                        for k in ("solver_calls", "nfev", "njev", "nlu")
                    },
                }
            if "final_state" in row:
                value["world_waits_completed"] = 4
            compact["rows"].append(value)
        attempts.append(compact)
    final = attempts[-1]
    resources = [r["native_resources"] for r in final["rows"] if "native_resources" in r]
    summary = {
        "schema": "chemworld-31D-development-summary-v1",
        "development_only": True,
        "qualification_scope": "finite synthetic nonreactive ideal-gas two-tank shaft device",
        "reference": {
            "cantera_version": "3.2.0",
            "constant_cp": "independent analytic PS/eta/PH and n/U DOP853 integration",
            "variable_cp": (
                "actual Cantera SP/HP setters, raw T retained, independent brentq refinement "
                "using Cantera property equations"
            ),
            "default_cantera_setters_alone_passed_all_declared_tolerances": False,
        },
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
        "final_direct_native_resources": {
            k: sum(r[k] for r in resources)
            for k in ("advances", "solver_calls", "nfev", "njev", "nlu")
        },
        "final_world_waits_completed": sum(
            r.get("world_waits_completed", 0) for r in final["rows"]
        ),
        "reference_ODE_resources": {
            "integrations": 16,
            "rhs_evaluations": sum(
                r.get("reference", {}).get("rhs_evaluations", 0) for r in final["rows"]
            ),
            "steps": sum(r.get("reference", {}).get("steps", 0) for r in final["rows"]),
        },
        "checks_by_group": {
            group: {
                "units": len(rows),
                "passed_units": sum(r["passed"] for r in rows),
                "checks": sum(len(r["checks"]) for r in rows),
                "passed_checks": sum(sum(r["checks"].values()) for r in rows),
            }
            for group, rows in (
                ("dynamics_and_events", final["rows"][:16]),
                ("static", final["rows"][16:24]),
                ("invariance", final["rows"][24:28]),
                ("actual_world", final["rows"][28:]),
            )
        },
        "attempts": attempts,
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "final",
                    "final_direct_native_resources",
                    "final_world_waits_completed",
                    "reference_ODE_resources",
                    "checks_by_group",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
