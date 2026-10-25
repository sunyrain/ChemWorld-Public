"""Compact all retained parent-47 attempts without copying source variable dumps."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def compact(value):
    if isinstance(value, dict):
        return {
            key: compact(item) for key, item in value.items() if key not in {"values", "traceback"}
        }
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def read(path):
    return compact(json.loads(path.read_text(encoding="utf8")))


def summarize(root, output):
    attempts, diagnostics = {}, {}
    for folder in sorted(root.glob("ChemWorld-47-close-*")):
        if not folder.is_dir():
            continue
        summary = folder / "summary.json"
        if summary.exists():
            row = read(summary)
            if isinstance(row, dict) and "total" in row:
                attempts[folder.name] = row
                continue
        files = sorted(folder.glob("*.json")) if summary.exists() else []
        if summary.exists():
            files = [summary]
        else:
            files = sorted(folder.glob("*.json")) + sorted(folder.glob("unit-*/summary.json"))
        diagnostics[folder.name] = {
            str(path.relative_to(folder)): read(path)
            for path in files
            if path.name != "native.json"
        } or {"status": "no-machine-result-retained; not a passing unit"}
    for folder in sorted(root.glob("chemworld47-runtime-attempt*")):
        attempts[folder.name] = read(folder / "SUMMARY.json")
    mesh = read(root / "ChemWorld-47-close-S-explicit-02" / "mesh-summary.json")
    suite = ET.parse(root / "ChemWorld-47-close-final-regression.xml").getroot().find("testsuite")
    if suite is None:
        raise ValueError("regression JUnit suite missing")
    regression = {
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
        "junit_elapsed_s": float(suite.attrib["time"]),
        "cases": [dict(case.attrib) for case in suite.findall("testcase")],
    }
    final = {
        "G": attempts["ChemWorld-47-close-G01"],
        "S_explicit": attempts["ChemWorld-47-close-S-explicit-02"],
        "R": attempts["chemworld47-runtime-attempt03"],
    }
    if any(row["total"] != row.get("completed", row["total"]) for row in final.values()):
        raise ValueError("qualification has an unfinished block")
    report = {
        "label": "development-qualification; not a publication freeze",
        "source_versions": {
            "watertap": "1.8.0",
            "idaes-pse": "2.13.0",
            "pyomo": "6.10.1",
            "ipopt": "3.14.20",
            "ASL": "20241111",
        },
        "current_blocks": {
            key: {k: v for k, v in row.items() if k != "rows" and k != "cases"}
            for key, row in final.items()
        },
        "source_default_attempt_remains_failed": True,
        "final_compatibility_regression": regression,
        "source_mesh_postprocessing": mesh,
        "comparison_count": sum(row.get("comparisons", 0) for row in final["S_explicit"]["rows"]),
        "common_G_check_count": sum(row.get("comparisons", 0) for row in final["G"]["rows"]),
        "all_current_blocks_passed": all(row["failed"] == 0 for row in final.values())
        and mesh["mesh_passed"],
        "attempts": attempts,
        "diagnostics_excluded_from_qualification_denominators": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"attempts", "diagnostics_excluded_from_qualification_denominators"}
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source_root, args.output)
