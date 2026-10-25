"""Compact fixed-attempt histories, including every failed or unexecuted unit."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def failures(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not row["passed"]:
            grouped[(row.get("error", "failed gate"), row.get("traceback", ""))].append(row["id"])
    return [
        {"error": error, "traceback": trace, "ids": ids} for (error, trace), ids in grouped.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for label in (
        "functional-first",
        "functional-second",
        "original-integration-first",
        "boundaries-first",
    ):
        directory = args.evidence / ("conv51j-" + label)
        data = load(directory / "summary.json")
        failed = []
        for node in ET.parse(directory / "junit.xml").getroot().iter("testcase"):
            for error in (*node.findall("failure"), *node.findall("error")):
                failed.append(
                    {
                        "node": node.get("classname") + "::" + node.get("name"),
                        "message": error.get("message"),
                        "text": error.text,
                    }
                )
        records[label] = {**data, "failed_nodes": failed, "raw_directory": str(directory)}
    (args.output / "FUNCTIONAL_RESULTS.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    records = {}
    for label in ("legacy-first", "legacy-consume-first", "legacy-consume-second"):
        directory = args.evidence / ("conv51j-" + label)
        data = load(directory / "summary.json")
        records[label] = {k: v for k, v in data.items() if k != "units"}
        records[label].update(
            raw_directory=str(directory),
            planned=48,
            recorded=len(data["units"]),
            successful=sum(r["passed"] for r in data["units"]),
            failures=failures(data["units"]),
        )
    (args.output / "LEGACY_RESULTS.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    records = {}
    for label in ("wiring-first", "wiring-second"):
        directory = args.evidence / ("conv51j-" + label)
        data = load(directory / "summary.json")
        rows, cells = data["units"], data["cells"]
        records[label] = {
            "passed": data["passed"],
            "planned": 448,
            "recorded": len(rows),
            "successful": sum(r["passed"] for r in rows),
            "raw_directory": str(directory),
            "planned_cells": 112,
            "recorded_cells": len(cells),
            "successful_cells": sum(c["passed"] for c in cells),
            "output_points": sum(r.get("output_points", 0) for r in rows),
            "propensity_comparisons": data["propensity_comparisons"],
            "elapsed_s": data["elapsed_s"],
            "failures": failures(rows),
        }
    (args.output / "WIRING_RESULTS.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    records = {}
    for label in ("package-first", "package-second"):
        directory = args.evidence / ("conv51j-" + label)
        data = load(directory / "summary.json")
        rows = data.get("protocol_summary", {}).get("protocols", [])
        records[label] = {
            "passed": data["passed"],
            "planned": 12,
            "recorded": len(rows),
            "successful": sum(r["passed"] for r in rows),
            "raw_directory": str(directory),
            "unexecuted": 12 - len(rows),
            "elapsed_s": data["elapsed_s"],
            "stages": data["stages"],
            "error": data.get("error"),
            "traceback": data.get("traceback"),
            "probe_stderr": (directory / "probe.stderr").read_text(encoding="utf-8"),
            "artifacts": data.get("artifacts"),
            "failures": failures(rows),
            "protocols": rows,
            "module_origins": data.get("protocol_summary", {}).get("module_origins"),
        }
    (args.output / "PACKAGE_RESULTS.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps({"written": 4, "output": str(args.output)}))


if __name__ == "__main__":
    main()
