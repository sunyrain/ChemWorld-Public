"""Retain every fixed 51F SDK wiring attempt and every material-balance failure."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51f-adapter-*")):
        if not directory.is_dir():
            continue
        if not (directory / "summary.json").is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for unit in data["units"]:
            for engine in unit["engines"].values():
                engine["failures"] = [
                    {k: v for k, v in row.items() if k not in {"traceback", "rows"}}
                    for row in engine["failures"]
                ]
            source_rows = [
                json.loads(line)
                for line in (directory / unit["id"] / "copasi/trajectories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            own_rows = [
                json.loads(line)
                for line in (directory / unit["id"] / "native/trajectories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            unit["source_versions"] = sorted({r["version"] for r in source_rows})
            unit["selected_methods"] = sorted(
                {r["method_name"] for r in source_rows if "method_name" in r}
            )
            unit["native_activity"] = [
                {
                    "id": r["id"],
                    "events": len(r["events"]),
                    "partition_changes": r["partition_changes"],
                    "work": r["work"],
                }
                for r in own_rows
                if r["status"] == "ok"
            ]
        attempts.append({"attempt": directory.name, "raw_evidence": str(directory), **data})
    result = {
        "schema": "chemworld-adaptive51f-adapter-history-v1",
        "scope": (
            "Eight fixed wiring units and material-policy characterization; "
            "no CDF or resource qualification"
        ),
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "all_source_material_errors_retained": True,
        "raw_particle_grids_parameters_rng_and_tracebacks_retained": True,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
