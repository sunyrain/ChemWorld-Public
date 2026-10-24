"""Compact readable 42D records from retained outputs, without rerunning science."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def test_summary(path):
    suites = ET.parse(path).getroot().findall("testsuite")
    total = sum(int(s.attrib["tests"]) for s in suites)
    failed = sum(
        int(s.attrib.get("failures", 0))
        + int(s.attrib.get("errors", 0))
        + int(s.attrib.get("skipped", 0))
        for s in suites
    )
    return {
        "passed": total - failed,
        "total": total,
        "elapsed_s": sum(float(s.attrib["time"]) for s in suites),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.root / "attempt01/results.json").read_text(encoding="utf8"))
    reference = json.loads((args.root / "attempt01/reference.json").read_text(encoding="utf8"))
    groups = raw["groups"]
    maxima = {}
    for name in ("reference", "shared_resolution"):
        maxima[name] = {
            key: max(row["errors"].get(key, 0.0) for row in groups[name])
            for key in ("phase_compositions", "phase_fractions", "G", "H", "mu_active")
        }
    maxima["element_balance"] = max(row["balance_error"] for row in groups["reference"])
    canaries = []
    for name in ("canary01", "canary02", "canary03"):
        directory = args.root / name
        if (directory / "summary.json").exists():
            record = json.loads((directory / "summary.json").read_text(encoding="utf8"))
            canaries.append(
                {
                    "attempt": name,
                    "stage": record["stage"],
                    "maps": record["maps"],
                    "elapsed_s": record["elapsed_s"],
                    "inventory_success": record["inventory"]["success"],
                    "read_only_query": record["world"].get("read_only"),
                    "snapshot_exact": record["world"].get("replay_equal"),
                    "heat_committed": record["world"].get("heat", {}).get("transaction_status"),
                    "failures": record["failures"],
                }
            )
        else:
            record = json.loads((directory / "INTERRUPTED.json").read_text(encoding="utf-8-sig"))
            record.pop("command", None)
            canaries.append({"attempt": name, **record})
    worlds = []
    for row in groups["world"]:
        queries = [row[k] for k in ("before_query", "after_query")]
        worlds.append(
            {
                "grid_intervals": row["grid_intervals"],
                "checks": row["checks"],
                "queries": [
                    {
                        "points": len(q["diagram"]["points"]),
                        "successful_points": sum(p["success"] for p in q["diagram"]["points"]),
                        **q["world_query"],
                    }
                    for q in queries
                ],
                "heat_info": row["heat_info"],
                "continuation_info": row["continuation_info"],
            }
        )
    payload = {
        "summary": raw["summary"],
        "reference": {k: v for k, v in reference.items() if k != "cases"},
        "maximum_absolute_errors": maxima,
        "canaries": canaries,
        "scientific_attempts": [{"attempt": "attempt01", "summary": raw["summary"]}],
        "functional_validation": {
            "new_tests": test_summary(args.root / "functional.xml"),
            "coordinator_shared_boundary_and_42B_tests": test_summary(
                args.root.parent / "chemworld-42d-zero-derivatives.xml"
            ),
        },
        "points": groups["reference"],
        "maps": groups["map_geometry"],
        "shared_resolution": groups["shared_resolution"],
        "inventory": [
            {
                "id": row["id"],
                "checks": row["checks"],
                "installed_phase_ids": row["result"]["installed_phase_ids"],
                "failure_reason": row["result"]["failure_reason"],
            }
            for row in groups["inventory"]
        ],
        "worlds": worlds,
        "limitations": [
            "Finite declared samples and starts; no global continuous phase map",
            "Neutral vacancy-free CEF; reference covers synthetic substitutional models",
            "Three-phase PH canary interrupted; runtime evidence uses ideal single phase",
            "Composition-grid refinement does not increase original TP search budget",
            "Absent-element chemical potentials are undefined and not compared",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                "maxima": maxima,
                "world_query_nodes": sum(q["points"] for w in worlds for q in w["queries"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
