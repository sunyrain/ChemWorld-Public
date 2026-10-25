"""All original integer-world baselines and their later consumption attempts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51h-legacy-*")):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if not path.is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        attempts.append({"attempt": directory.name, "raw_evidence": str(directory), **data})
    result = {
        "schema": "chemworld-integer51h-legacy-history-v1",
        "scope": "Original supported models; immutable pre-change state/info expectations",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
