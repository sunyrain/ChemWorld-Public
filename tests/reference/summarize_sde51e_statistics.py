"""Retain every completed fixed SDE block, with all failed checks and exact counts."""

import argparse
import json
from pathlib import Path


def attempt(directory):
    result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    units = []
    for unit in result["units"]:
        engines = {}
        for name, data in unit["engines"].items():
            engines[name] = {key: value for key, value in data.items() if key not in {"failures"}}
            engines[name]["failures"] = [
                {key: value for key, value in row.items() if key != "traceback"}
                for row in data["failures"]
            ]
            # Old blocks counted only returned records; a killed active job has
            # no final row. Preserve those raw counts as lower bounds, never zero
            # attempted simulations when a timeout left the count unobserved.
            engines[name].setdefault(
                "simulation_attempt_count_exact", unit["processes"][name]["status"] != "timeout"
            )
        units.append(
            {
                "id": unit["id"],
                "kind": unit["kind"],
                "step_s": unit["step_s"],
                "duration_s": unit["duration_s"],
                "passed": unit["passed"],
                "failed_checks": [c["name"] for c in unit["checks"] if not c["passed"]],
                "processes": unit["processes"],
                "engines": engines,
                "cdf": unit["cdf"],
            }
        )
    totals = {
        engine: {
            key: sum(u["engines"][engine][key] for u in units)
            for key in ("planned", "recorded", "completed", "simulations_attempted", "unrecorded")
        }
        for engine in ("native", "copasi")
    }
    for value in totals.values():
        if value["planned"] != 22530 or value["recorded"] + value["unrecorded"] != 22530:
            raise ValueError("statistics denominator changed")
    for engine, value in totals.items():
        value["simulation_attempt_count_exact"] = all(
            u["engines"][engine]["simulation_attempt_count_exact"] for u in units
        )
    return {
        "attempt": directory.name,
        "output_policy": result.get("output_policy", "linear"),
        "passed": result["passed"],
        "units_planned": 24,
        "units_recorded": len(units),
        "units_passed": sum(u["passed"] for u in units),
        "elapsed_s": result["elapsed_s"],
        "totals": totals,
        "cdf_limit": result["cdf_limit"],
        "cdf_horizontal_tolerance_particles": result["cdf_horizontal_tolerance_particles"],
        "units": units,
        "convergence": result["convergence"],
        "raw_evidence": str(directory),
        "full_tracebacks_retained_in_raw": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="conv51e-statistics-*")
    args = parser.parse_args()
    directories = sorted(p for p in args.evidence.glob(args.pattern) if p.is_dir())
    unfinished = [str(p) for p in directories if not (p / "summary.json").is_file()]
    result = {
        "schema": "chemworld-sde51e-statistics-history-v1",
        "attempts": [attempt(p) for p in directories if (p / "summary.json").is_file()],
        "unfinished_attempts": unfinished,
        "scope": "fixed 24-unit development qualification; not full parent 51",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"completed_attempts": len(result["attempts"]), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
