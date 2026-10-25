"""Compact actual installed SBML evidence; raw module paths stay outside Git."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.evidence / "summary.json").read_text(encoding="utf-8"))
    target = (args.evidence / "installed").resolve()
    protocol = raw.get("protocol_summary", {})
    rows = protocol.get("protocols", [])
    parent = protocol.get("module_origins", {})
    origins_valid = bool(parent) and all(Path(p).is_relative_to(target) for p in parent.values())
    children = []
    line_counts = []
    for row in rows:
        directory = args.evidence / "protocols"
        child = json.loads((directory / (row["id"] + "-fresh.json")).read_text(encoding="utf-8"))
        origins = child["module_origins"]
        children.append(
            {
                "id": row["id"],
                "loaded_modules": len(origins),
                "all_from_installed_target": bool(origins)
                and all(Path(p).is_relative_to(target) for p in origins.values()),
            }
        )
        line_counts.append(
            sum(
                bool(line.strip())
                for line in (directory / (row["id"] + ".jsonl"))
                .read_text(encoding="utf-8")
                .splitlines()
            )
        )
    passed = (
        raw["passed"]
        and len(rows) == 12
        and all(row["passed"] for row in rows)
        and all(row["recorded_steps"] == row["replayed_steps"] == 8 for row in rows)
        and line_counts == [8] * 12
        and origins_valid
        and all(row["all_from_installed_target"] for row in children)
        and protocol["sdk_importable"] is False
    )
    result = {
        "schema": "chemworld-sbml51i-package-summary-v1",
        "raw_block": args.evidence.name,
        "passed": passed,
        "stages": raw["stages"],
        "artifacts": raw.get("artifacts"),
        "elapsed_s": raw["elapsed_s"],
        "protocols_planned": 12,
        "protocols": rows,
        "recorded_steps": sum(row["recorded_steps"] for row in rows),
        "zero_tolerance_replayed_steps": sum(row["replayed_steps"] for row in rows),
        "trajectory_records": sum(line_counts),
        "new_processes": children,
        "parent_loaded_modules": len(parent),
        "all_parent_modules_installed": origins_valid,
        "sdk_importable": protocol["sdk_importable"],
        "unexecuted_stages": raw["unexecuted_stages"],
        "independent_distribution_qualified": False,
        "parent51_complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "passed",
                    "recorded_steps",
                    "zero_tolerance_replayed_steps",
                    "parent_loaded_modules",
                )
            }
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
