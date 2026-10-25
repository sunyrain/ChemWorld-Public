"""All actual method-31 adapter attempts with exact denominators and setup failures."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51g-adapter-*")):
        if not directory.is_dir():
            continue
        if not (directory / "summary.json").is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for unit in data["units"]:
            for engine in unit["engines"].values():
                engine["failures"] = [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"traceback", "rows", "initial", "initial_fluxes"}
                    }
                    for row in engine["failures"]
                ]
            lines = (
                (directory / unit["id"] / "copasi/trajectories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            rows = [json.loads(line) for line in lines if line.strip()]
            unit["source_versions"] = sorted({row["version"] for row in rows})
            unit["selected_methods"] = sorted(
                {row["method_name"] for row in rows if row["status"] == "ok"}
            )
            valid = [row for row in rows if row["status"] == "ok"]
            unit["actual_partition_sample"] = valid[0]["deterministic_reactions"] if valid else None
        attempts.append({"attempt": directory.name, "raw_evidence": str(directory), **data})
    result = {
        "schema": "chemworld-hybrid51g-adapter-history-v1",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "raw_tracebacks_and_particle_rows_retained": True,
        "scope": "Fixed four-unit wiring block; not distribution or resource qualification",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
