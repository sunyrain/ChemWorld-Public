"""Preserve every SBML integer functional attempt and immutable suffix check."""

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET


def read_text(path):
    raw = path.read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-integration", action="store_true")
    args = parser.parse_args()
    attempts = []
    for name in (
        "conv51i-functional-first",
        "conv51i-functional-second",
        "conv51i-functional-third",
        "conv51i-regression-first",
        "conv51i-biochemical-regression-first",
    ):
        directory = args.evidence / name
        text = read_text(directory / "pytest.log")
        match = re.findall(r"(?:(\d+) failed, )?(\d+) passed in ([\d.]+)s", text)
        if len(match) != 1:
            raise ValueError("missing or ambiguous terminal pytest summary: " + name)
        failed, passed, elapsed = match[0]
        row = {
            "id": name,
            "passed": int(passed),
            "failed": int(failed or 0),
            "total": int(passed) + int(failed or 0),
            "pytest_elapsed_s": float(elapsed),
            "failed_nodes": re.findall(r"^FAILED (.*)$", text, re.M),
        }
        if (directory / "junit.xml").exists():
            root = ET.parse(directory / "junit.xml").getroot()
            row["junit"] = {
                key: sum(int(s.get(key, 0)) for s in root.findall("testsuite"))
                for key in ("tests", "failures", "errors", "skipped")
            }
            if row["junit"]["tests"] != row["total"]:
                raise ValueError("JUnit and terminal counts disagree: " + name)
        if name == "conv51i-regression-first":
            row["supervisor_failure"] = (
                "Elapsed time was zero at the first progress print, raising ZeroDivisionError. "
                "The child remained live (PID 37352), was observed without restart, "
                "and finished all 274 tests. Final pytest text and JUnit agree. "
                "The departed supervisor did not observe its return code."
            )
            row["child_returncode_observed"] = None
        attempts.append(row)
    legacy = []
    legacy_names = (
        "conv51i-legacy-first",
        "conv51i-legacy-consume-first",
        "conv51i-integer-legacy-consume-first",
        "conv51i-legacy-metadata-first",
        "conv51i-integer-legacy-metadata-first",
    )
    if args.final_integration:
        legacy_names += ("conv51i-final-sbml-legacy-first", "conv51i-final-integer-legacy-first")
    for name in legacy_names:
        payload = json.loads(read_text(args.evidence / name / "summary.json"))
        legacy.append(
            {
                "id": name,
                "mode": payload["mode"],
                "baseline": payload.get("baseline"),
                "total": len(payload["units"]),
                "passed": payload["passed"],
                "elapsed_s": payload["elapsed_s"],
                "failures": [r for r in payload["units"] if not r["passed"]],
            }
        )
    types = {}
    for filename in (
        "mypy-new-module.log",
        "mypy-legacy-policy.log",
        "mypy-original-shadow.log",
        "ruff.log",
    ):
        types[filename] = read_text(
            args.evidence / "conv51i-functional-third" / filename
        ).splitlines()
    result = {
        "schema": "chemworld-sbml51i-functional-v1",
        "pytest_attempts": attempts,
        "legacy_attempts": legacy,
        "static_checks": types,
        "latest_new_matrix_passed": attempts[2]["passed"] == 62 and attempts[2]["failed"] == 0,
        "regression_total": attempts[3]["total"] + attempts[4]["total"],
        "independent_distribution_qualified": False,
        "installed_package_qualified": False,
        "parent51_complete": False,
    }
    if args.final_integration:
        protocols = []
        for name in (
            "conv51i-merged-exports-integration-first",
            "conv51i-final-shared-integration-first",
        ):
            text = read_text(args.evidence / name / "pytest.log")
            matches = re.findall(r"(\d+) passed, (\d+) deselected in ([\d.]+)s", text)
            if len(matches) != 1:
                raise ValueError("missing exact integration terminal summary: " + name)
            passed, deselected, elapsed = matches[0]
            protocols.append(
                {
                    "id": name,
                    "passed": int(passed),
                    "deselected": int(deselected),
                    "pytest_elapsed_s": float(elapsed),
                    "complete_fixed_protocols": int(passed) == 12 and int(deselected) == 50,
                }
            )
        result["post_merge_protocol_attempts"] = protocols
        result["post_merge_scope"] = (
            "Repeated original twelve world protocols, not new functional coverage"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({key: result[key] for key in ("latest_new_matrix_passed", "regression_total")})
    )


if __name__ == "__main__":
    main()
