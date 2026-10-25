"""Keep every fixed world/integration attempt and exact historical-snapshot result."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for pattern in ("conv51f-world-*", "conv51f-integration-*"):
        for directory in sorted(args.evidence.glob(pattern)):
            if not directory.is_dir():
                continue
            required = ("junit.xml", "planned.json", "process.json")
            if not all((directory / name).is_file() for name in required):
                unfinished.append(str(directory))
                continue
            planned = json.loads((directory / "planned.json").read_text(encoding="utf-8"))
            process = json.loads((directory / "process.json").read_text(encoding="utf-8"))
            suites = ET.parse(directory / "junit.xml").getroot().findall("testsuite")
            row = {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                "planned": planned["total"],
                "files": planned["files"],
                "process": process,
                "time_s": sum(float(s.get("time", 0)) for s in suites),
            }
            for key in ("tests", "failures", "errors", "skipped"):
                row[key] = sum(int(s.get(key, 0)) for s in suites)
            row["failure_details"] = [
                {
                    "classname": case.get("classname"),
                    "name": case.get("name"),
                    "kind": tag,
                    "text": detail.text,
                }
                for suite in suites
                for case in suite.findall("testcase")
                for tag in ("failure", "error", "skipped")
                for detail in case.findall(tag)
            ]
            row["passed"] = (
                row["tests"] == row["planned"]
                and row["failures"] == row["errors"] == row["skipped"] == 0
                and process["exit_code"] == 0
            )
            attempts.append(row)
    compatibility = []
    for directory in sorted(args.evidence.glob("conv51e-fractional-compat-*")):
        if not directory.is_dir():
            continue
        if not (directory / "summary.json").is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        compatibility.append({"attempt": directory.name, "raw_evidence": str(directory), **data})
    result = {
        "schema": "chemworld-adaptive51f-world-functional-v1",
        "attempts": attempts,
        "historical_sde_compatibility": compatibility,
        "unfinished_attempts": unfinished,
        "overlap": "72 includes prior 52 pure nodes; regression attempts overlap; do not sum",
        "scope": "Functional world integration; scientific, reference and installation remain open",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "attempts": len(attempts),
                "compatibility": len(compatibility),
                "unfinished": unfinished,
            }
        )
    )


if __name__ == "__main__":
    main()
