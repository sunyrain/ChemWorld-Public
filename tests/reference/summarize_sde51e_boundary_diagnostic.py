"""Preserve every tolerance diagnostic, distinguishing failed startup from live timeouts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51e-boundary-diagnostic-*")):
        if not directory.is_dir():
            continue
        if not (directory / "summary.json").is_file():
            unfinished.append(str(directory))
            continue
        source = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        units = []
        for unit in source["units"]:
            row = {key: value for key, value in unit.items() if key != "rows"}
            row.setdefault("planned", 1)
            row.setdefault("unstarted", 1 - row["simulations_attempted"])
            row.setdefault("unrecorded", 1 - row["recorded"])
            row["endpoint_evaluated"] = bool(unit.get("rows") and unit["rows"][0].get("rows"))
            stderr = (directory / unit["id"] / "stderr.log").read_text(encoding="utf-8")
            row["startup_error"] = "FileExistsError" if "FileExistsError" in stderr else None
            units.append(row)
        attempts.append(
            {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                "planned": 2,
                "recorded": len(units),
                "units": units,
            }
        )
    result = {
        "schema": "chemworld-sde51e-boundary-diagnostic-history-v1",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "scope": "60-second diagnosis only; no qualification claim",
        "raw_source_outputs_and_all_startup_errors_retained": True,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
