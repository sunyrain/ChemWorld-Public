"""Complete optional-output functional/compatibility history, with overlapping counts separate."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="conv51e-fractional-*")
    parser.add_argument("--schema", default="chemworld-sde51e-fractional-functional-v1")
    args = parser.parse_args()
    tests, compatibility, unfinished = [], [], []
    for directory in sorted(args.evidence.glob(args.pattern)):
        if not directory.is_dir():
            continue
        if "-compat-" in directory.name:
            if (directory / "summary.json").is_file():
                result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                compatibility.append(
                    {"attempt": directory.name, "raw_evidence": str(directory), **result}
                )
            else:
                unfinished.append(str(directory))
            continue
        if not any(name in directory.name for name in ("-unit-", "-integration-")):
            continue
        xml = directory / "execution/junit.xml"
        if not xml.is_file():
            xml = directory / "junit.xml"
        if not xml.is_file():
            unfinished.append(str(directory))
            continue
        suites = ET.parse(xml).getroot().findall("testsuite")
        plan = json.loads((directory / "planned.json").read_text(encoding="utf-8"))
        planned = plan.get("total", plan["nodes"])
        row = {"attempt": directory.name, "planned": planned, "raw_evidence": str(directory)}
        for key in ("tests", "failures", "errors", "skipped"):
            row[key] = sum(int(s.get(key, 0)) for s in suites)
        row["time_s"] = sum(float(s.get("time", 0)) for s in suites)
        row["failure_details"] = [
            {"name": case.get("name"), "kind": tag, "text": detail.text}
            for suite in suites
            for case in suite.findall("testcase")
            for tag in ("failure", "error", "skipped")
            for detail in case.findall(tag)
        ]
        row["passed"] = (
            row["tests"] == planned and row["failures"] == row["errors"] == row["skipped"] == 0
        )
        tests.append(row)
    result = {
        "schema": args.schema,
        "test_attempts": tests,
        "legacy_compatibility_attempts": compatibility,
        "unfinished_attempts": unfinished,
        "overlap": (
            "Integration includes the single-file nodes; attempts are overlapping, not additive"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "test_attempts": len(tests),
                "compatibility_attempts": len(compatibility),
                "unfinished": unfinished,
            }
        )
    )


if __name__ == "__main__":
    main()
