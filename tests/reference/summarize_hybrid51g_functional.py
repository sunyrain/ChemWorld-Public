"""Every fixed RK45 functional attempt, including failures and unfinished processes."""

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
    for pattern in ("conv51g-functional-*", "conv51g-integration-*"):
        for directory in sorted(args.evidence.glob(pattern)):
            if not directory.is_dir():
                continue
            if not all(
                (directory / name).is_file()
                for name in ("junit.xml", "planned.json", "process.json")
            ):
                unfinished.append(str(directory))
                continue
            plan = json.loads((directory / "planned.json").read_text(encoding="utf-8"))
            process = json.loads((directory / "process.json").read_text(encoding="utf-8"))
            suites = ET.parse(directory / "junit.xml").getroot().findall("testsuite")
            row = {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                "planned": plan["total"],
                "files": plan["files"],
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
    result = {
        "schema": "chemworld-hybrid51g-functional-history-v1",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "overlap": "141 integration nodes include 46 new and 95 old; do not sum attempts",
        "scope": "Functional RK45/analytic-path checks; not full source distribution qualification",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
