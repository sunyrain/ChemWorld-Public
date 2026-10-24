"""Derive honest availability counts from preserved 32B raw unit attempts."""

import argparse
import json
from pathlib import Path


def assessed_checks(index, raw):
    names = list(raw["checks"])
    if index <= 16:
        available = (
            set(names)
            if raw.get("native", {}).get("success")
            else {names[0], names[9]}
            if "native" in raw
            else {names[0]}
        )
    elif index in (17, 18):
        available = (
            set(names)
            if raw.get("info", {}).get("transaction_status") == "committed"
            else set(names[:2])
            if "final_inventory" in raw
            else {names[0]}
        )
    elif index == 19:
        available = set(names) if "final_inventory" in raw else {names[0]}
    else:
        available = set(names) if "final_inventory" in raw else {names[0]}
    result = {}
    for name, check in raw["checks"].items():
        evidence = check["detail"]
        if index in (17, 18) and name in names[:2] and "final_inventory" in raw:
            evidence = {
                "actual_outcome": raw["info"]["operation_outcome"],
                "actual_time_s": raw["final_inventory"]["time_s"],
                "expected_time_s": 1.0 if index == 17 else 2.0,
                "expected_outcome": "success" if index == 17 else "stopped",
            }
        elif index > 16 and "final_inventory" not in raw and name == names[0]:
            evidence = {
                "actual": "initialization_rejected",
                "reason": raw["failure_reason"],
                "expected": "committed_receiver_event_and_fresh_process_continuation",
            }
        result[name] = {
            "assessed": name in available,
            "passed": bool(check["passed"]) if name in available else False,
            "evidence": evidence
            if name in available
            else "no accepted physical solution or trajectory",
        }
    return result


def derive(raw_directory):
    original = json.loads((raw_directory / "summary.json").read_text(encoding="utf-8"))
    units = []
    for index in range(1, 21):
        raw = json.loads((raw_directory / f"unit{index:02d}.json").read_text(encoding="utf-8"))
        rows = assessed_checks(index, raw)
        units.append(
            {
                "id": raw["id"],
                "passed": raw["passed"],
                "elapsed_s": raw["elapsed_s"],
                "checks": rows,
                "failure_reason": raw.get(
                    "failure_reason", raw.get("native", {}).get("failure_reason", "")
                ),
            }
        )
    checks = [c for unit in units for c in unit["checks"].values()]
    return {
        "task": "32B",
        "mode": "development",
        "attempt": "block01",
        "raw_directory": str(raw_directory.resolve()),
        "units_attempted": len(units),
        "units_planned": 20,
        "units_fully_passed": sum(u["passed"] for u in units),
        "comparison_slots_planned": 200,
        "comparisons_assessed": sum(c["assessed"] for c in checks),
        "comparisons_passed": sum(c["passed"] for c in checks),
        "comparisons_assessed_failed": sum(c["assessed"] and not c["passed"] for c in checks),
        "comparisons_unexecuted_unavailable": sum(not c["assessed"] for c in checks),
        "elapsed_s": original["elapsed_s"],
        "raw_count_correction": (
            "Raw comparisons_executed=200 counted filled check slots. It is preserved, "
            "not used as the scientific execution denominator."
        ),
        "units": units,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = derive(args.raw)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "units"}, indent=2))


if __name__ == "__main__":
    main()
