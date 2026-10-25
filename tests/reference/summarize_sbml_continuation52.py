"""Compact retained 52F machine summaries without copying full branch arrays."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    attempts = []
    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        summary_path = directory / "summary.json"
        failure_path = directory / "FAILURE.txt"
        if summary_path.exists():
            attempts.append(
                {
                    "attempt": directory.name,
                    "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                }
            )
        elif failure_path.exists():
            attempts.append(
                {
                    "attempt": directory.name,
                    "failure": failure_path.read_text(encoding="utf-8-sig").strip(),
                }
            )
    latest = sorted(p for p in args.root.glob("attempt*") if (p / "summary.json").exists())[-1]
    maxima = {}
    for unit in sorted(latest.glob("[0-9][0-9]-*.json")):
        data = json.loads(unit.read_text(encoding="utf-8"))
        for comparison in data["comparisons"]:
            if "max_absolute_error" in comparison:
                key = comparison["name"]
                maxima[key] = max(maxima.get(key, 0), comparison["max_absolute_error"])
            if "polynomial_root_max_error" in comparison:
                key = "independent-real-cubic-root-distance"
                maxima[key] = max(maxima.get(key, 0), comparison["polynomial_root_max_error"])
    tests = []
    for xml in sorted(args.root.glob("*.xml")):
        suites = ET.parse(xml).getroot().iter("testsuite")
        entries = list(suites)
        tests.append(
            {
                "run": xml.stem,
                **{
                    key: sum(int(s.get(key, "0")) for s in entries)
                    for key in ("tests", "failures", "errors", "skipped")
                },
            }
        )
    result = {
        "format": "chemworld-52f-development-summary-v1",
        "raw_directory": str(args.root),
        "final_attempt": latest.name,
        "attempts": attempts,
        "test_runs": tests,
        "largest_absolute_errors_final": maxima,
        "interpretation": (
            "18 fixed units/180 checks. Intended partial/failed branches are boundary checks; "
            "a passing check does not make those branches complete. Synthetic analytic "
            "development qualification, no empirical/all-branches/COPASI parity claim."
        ),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps({"final_attempt": latest.name, "test_runs": tests, "max_errors": maxima}),
        flush=True,
    )


if __name__ == "__main__":
    main()
