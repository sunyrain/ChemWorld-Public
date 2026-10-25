"""All same-network resource attempts, preserving failed trials and unstarted units."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51f-resource-*")):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if not path.is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        totals = {
            field: sum(e[field] for c in data["conditions"] for e in c["engines"].values())
            for field in ("planned", "recorded", "completed", "started", "unstarted", "unrecorded")
        }
        attempts.append(
            {"attempt": directory.name, "raw_evidence": str(directory), **data, "totals": totals}
        )
    result = {
        "schema": "chemworld-adaptive51f-resource-history-v1",
        "scope": "Same-network approximation / measured work and time; no full CME equivalence",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "all_timing_observations_and_allocation_trials_retained": True,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
