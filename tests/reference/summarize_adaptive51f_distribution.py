"""Every completed or unfinished fixed distribution attempt, without deleting failures."""

import argparse
import json
from pathlib import Path


def cycle_reactivation(directory, unit):
    raw = json.loads((directory / unit["id"] / "unit.json").read_text(encoding="utf-8"))
    participants = [
        set(r["orders"]) | {k for k, n in r["changes"].items() if n} for r in raw["reactions"]
    ]
    initial_low = {k for k, v in raw["initial"].items() if v < (raw["lower"] + raw["upper"]) / 2}
    result = {}
    for engine in ("native", "reference"):
        paused_paths = resumed_paths = 0
        resumes = dict.fromkeys((r["id"] for r in raw["reactions"]), 0)
        path = directory / unit["id"] / engine / "trajectories.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] != "ok":
                continue
            active = [bool(p & initial_low) for p in participants]
            paused, resumed = set(), set()
            for change in row["partition_changes"]:
                after = [bool(p & set(change["after"])) for p in participants]
                for index, (was, now) in enumerate(zip(active, after, strict=True)):
                    if was and not now:
                        paused.add(index)
                    if now and not was and index in paused:
                        resumed.add(index)
                        resumes[raw["reactions"][index]["id"]] += 1
                active = after
            paused_paths += bool(paused)
            resumed_paths += bool(resumed)
        result[engine] = {
            "trajectories_with_channel_paused": paused_paths,
            "trajectories_with_channel_returned": resumed_paths,
            "return_transitions_by_reaction": resumes,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51f-distribution-*")):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if not path.is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for unit in data["units"]:
            if unit["family"] == "cycle":
                unit["channel_reactivation"] = cycle_reactivation(directory, unit)
        totals = {}
        for engine in ("native", "reference"):
            totals[engine] = {
                field: sum(u["engines"][engine][field] for u in data["units"])
                for field in (
                    "planned",
                    "recorded",
                    "completed",
                    "simulations_attempted",
                    "unstarted",
                    "unrecorded",
                    "events",
                    "trajectories_with_transitions",
                )
            }
            totals[engine]["max_pool_error_particles"] = max(
                u["engines"][engine]["max_pool_error_particles"] for u in data["units"]
            )
        failed = [
            {"unit": u["id"], **check}
            for u in data["units"]
            for check in u["checks"]
            if not check["passed"]
        ]
        attempts.append(
            {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                **data,
                "totals": totals,
                "passed_units": sum(u["passed"] for u in data["units"]),
                "failed_checks": failed,
            }
        )
    result = {
        "schema": "chemworld-adaptive51f-distribution-history-v1",
        "scope": (
            "Finite-grid frozen-propensity distribution; not continuous-hazard or source parity"
        ),
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "raw_endpoints_changes_counts_errors_and_source_snapshots_retained": True,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
