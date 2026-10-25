"""Keep all old-code preparations and later exact consumers without rebuilding expectations."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51g-legacy-*")):
        if not directory.is_dir():
            continue
        source = directory / "summary.json"
        if not source.is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads(source.read_text(encoding="utf-8"))
        rows = []
        for row in data["units"]:
            compact = {"id": row["id"], "passed": row["passed"]}
            if not row["passed"]:
                match = re.search(r"'error_message': '([^']*)'", row["error"])
                compact["error"] = match.group(1) if match else row["error"][:500]
                compact["full_error_and_traceback"] = str(source)
            rows.append(compact)
        attempts.append(
            {
                **{key: value for key, value in data.items() if key != "units"},
                "profile": data.get("profile", "subsecond"),
                "attempt": directory.name,
                "planned": 4,
                "recorded": len(rows),
                "units_passed": sum(row["passed"] for row in rows),
                "units": rows,
                "raw_evidence": str(directory),
            }
        )
    result = {
        "schema": "chemworld-hybrid51g-legacy-history-v1",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "scope": "Old-code baseline/consumer only; not RK45 qualification",
        "all_raw_errors_and_tracebacks_retained": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
