"""All source wiring/distribution attempts, with exact starts, failures and metadata."""

import argparse
import json
from pathlib import Path

STRATEGIES = ("direct", "next", "coarse", "fine")


def history(root, mode):
    attempts, unfinished = [], []
    for directory in sorted(root.glob("conv51h-source-" + mode + "-*")):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if not path.is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        engines = [e for u in data["units"] for e in u["engines"].values()]
        aggregate = {
            key: sum(e[key] for e in engines)
            for key in (
                "planned",
                "recorded",
                "started",
                "simulation_attempted",
                "simulation_completed",
                "successful",
                "unstarted",
                "checked_output_points",
            )
        }
        aggregate.update(
            models_passed=sum(u["passed"] for u in data["units"]),
            strategy_units_passed=sum(e["passed"] for e in engines),
            failed_calls=sum(len(e["failures"]) for e in engines),
            invalid_trajectories=sum(len(e["invalid_trajectories"]) for e in engines),
            strategies={},
        )
        for strategy in STRATEGIES:
            selected = [u["engines"][strategy] for u in data["units"]]
            checks = [c for e in selected for c in (e.get("statistics") or {}).get("checks", [])]
            cdf = [c for c in checks if c["name"].startswith(("cdf_", "joint_cdf_"))]
            aggregate["strategies"][strategy] = {
                "models_passed": sum(e["passed"] for e in selected),
                "planned": sum(e["planned"] for e in selected),
                "simulation_completed": sum(e["simulation_completed"] for e in selected),
                "statistical_checks": len(checks),
                "statistical_checks_passed": sum(c["passed"] for c in checks),
                "cdf_checks": len(cdf),
                "maximum_cdf_difference": max((c["absolute_error"] for c in cdf), default=None),
                "minimum_cdf_limit": min((c["limit"] for c in cdf), default=None),
            }
        attempts.append(
            {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                **data,
                "aggregate": aggregate,
            }
        )
    return {
        "schema": "chemworld-integer51h-source-" + mode + "-history-v1",
        "scope": (
            "Actual SDK25/26/27, fixed complete source matrix; "
            "native qualification remains separate"
        ),
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "all_raw_cps_xml_reports_series_messages_and_failures_retained": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for mode, filename in (
        ("wiring", "SOURCE_WIRING_RESULTS.json"),
        ("distribution", "SOURCE_DISTRIBUTION_RESULTS.json"),
    ):
        result = history(args.evidence, mode)
        (args.output / filename).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "attempts": len(result["attempts"]),
                    "unfinished": result["unfinished_attempts"],
                    "aggregates": [a["aggregate"] for a in result["attempts"]],
                }
            )
        )
    diagnostic = args.evidence / "conv51h-source-column-diagnostic-first"
    if (diagnostic / "inspection.json").is_file():
        inspection = json.loads((diagnostic / "inspection.json").read_text(encoding="utf-8"))
        inspection.pop("raw_table", None)
        result = {
            "schema": "chemworld-integer51h-source-column-diagnostic-v1",
            "scope": "One unchanged saved-CPS execution, not ensemble qualification",
            "raw_evidence": str(diagnostic),
            "process": json.loads((diagnostic / "process.json").read_text(encoding="utf-8")),
            **inspection,
            "raw_table_retained": True,
            "correction": (
                "Actual source CReport at all grid points; "
                "no reconstructed invariant column"
            ),
        }
        (args.output / "SOURCE_COLUMN_DIAGNOSTIC.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
