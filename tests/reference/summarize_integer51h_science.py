"""Every fixed integer chemistry science attempt and explicit native comparison."""

import argparse
import json
from pathlib import Path

from integer51h_distribution_cases import STRATEGIES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def distribution_aggregate(directory, data):
    result = {
        "models_passed": sum(u["passed"] for u in data["units"]),
        "reference_laws_passed": sum(u["reference"]["passed"] for u in data["units"]),
        "strategies": {},
    }
    for strategy in STRATEGIES:
        rows = [u["engines"][strategy] for u in data["units"]]
        checks = [c for r in rows for c in r.get("statistics", {}).get("checks", [])]
        cdf = [c for c in checks if c["name"].startswith(("cdf_", "joint_cdf_"))]
        result["strategies"][strategy] = {
            **{
                key: sum(r[key] for r in rows)
                for key in (
                    "planned",
                    "recorded",
                    "successful",
                    "simulation_completed",
                    "simulation_attempted",
                    "unrecorded",
                )
            },
            "models_passed": sum(r["passed"] for r in rows),
            "invalid_trajectories": sum(len(r["invalid_trajectories"]) for r in rows),
            "failed_calls": sum(len(r["failures"]) for r in rows),
            "statistical_checks": len(checks),
            "statistical_checks_passed": sum(c["passed"] for c in checks),
            "cdf_checks": len(cdf),
            "maximum_cdf_difference": max((c["absolute_error"] for c in cdf), default=None),
            "minimum_cdf_limit": min((c["limit"] for c in cdf), default=None),
        }
    for unit in data["units"]:
        path = directory / unit["id"] / "reference/law.json"
        if path.is_file():
            law = read(path)
            unit["reference"]["evaluator"] = law.get(
                "reference_evaluator", "dense-expm-and-uniformization-v1"
            )
    data["aggregate"] = result


def resource_aggregate(data):
    engines = [e for u in data["units"] for e in u["engines"].values()]
    checks = [c for e in engines for c in e["checks"]]
    data["aggregate"] = {
        "models_passed": sum(u["passed"] for u in data["units"]),
        "strategy_units_passed": sum(e["passed"] for e in engines),
        "planned": sum(e["planned"] for e in engines),
        "recorded": sum(e["recorded"] for e in engines),
        "successful": sum(e["successful"] for e in engines),
        "checks": len(checks),
        "checks_passed": sum(c["passed"] for c in checks),
    }


def history(root, kind):
    attempts, unfinished = [], []
    for directory in sorted(root.glob("conv51h-" + kind + "-*")):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if not path.is_file():
            unfinished.append(str(directory))
            continue
        data = read(path)
        if kind == "distribution":
            distribution_aggregate(directory, data)
        elif kind == "resource":
            resource_aggregate(data)
        else:
            for unit in data["units"]:
                for key in ("action_probability", "ode_probability"):
                    unit.pop(key, None)
            data["aggregate"] = {
                "models_passed": sum(u["passed"] for u in data["units"]),
                "maximum_action_error": max(
                    (u.get("action_vs_uniformization", 0) for u in data["units"]), default=None
                ),
                "maximum_ode_error": max(
                    (u.get("ode_vs_uniformization", 0) for u in data["units"]), default=None
                ),
            }
        attempts.append({"attempt": directory.name, "raw_evidence": str(directory), **data})
    return {
        "schema": "chemworld-integer51h-" + kind + "-history-v1",
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "all_raw_vectors_trajectories_and_failures_retained": True,
    }


def compare_native(first, second):
    left, right = read(first / "summary.json"), read(second / "summary.json")
    if [u["id"] for u in left["units"]] != [u["id"] for u in right["units"]]:
        raise ValueError("native comparison needs the same complete model matrix")
    units = []
    for unit in left["units"]:
        for strategy in STRATEGIES:
            first_path = first / unit["id"] / strategy / "trajectories.jsonl"
            second_path = second / unit["id"] / strategy / "trajectories.jsonl"
            byte_exact = first_path.read_bytes() == second_path.read_bytes()
            a = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
            b = [json.loads(line) for line in second_path.read_text(encoding="utf-8").splitlines()]
            expected = unit["engines"][strategy]["planned"]
            differences = [x.get("id") for x, y in zip(a, b, strict=False) if x != y]
            units.append(
                {
                    "id": unit["id"],
                    "strategy": strategy,
                    "planned": expected,
                    "first_recorded": len(a),
                    "second_recorded": len(b),
                    "full_row_differences": differences,
                    "jsonl_bytes_exact": byte_exact,
                    "passed": len(a) == len(b) == expected and not differences,
                }
            )
    return {
        "scope": (
            "All complete saved native rows, including endpoint/counts/work/full continuation; "
            "no unstored event times claimed"
        ),
        "first": str(first),
        "second": str(second),
        "rows_planned": sum(u["planned"] for u in units),
        "all_jsonl_bytes_exact": all(u["jsonl_bytes_exact"] for u in units),
        "units": units,
        "passed": len(units) == 56 and all(u["passed"] for u in units),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-first", type=Path)
    parser.add_argument("--compare-second", type=Path)
    args = parser.parse_args()
    for kind, filename in (
        ("distribution", "DISTRIBUTION_RESULTS.json"),
        ("reference-diagnostic", "REFERENCE_DIAGNOSTIC.json"),
        ("resource", "RESOURCE_RESULTS.json"),
    ):
        data = history(args.evidence, kind)
        if kind == "distribution" and args.compare_first and args.compare_second:
            data["native_full_row_comparison"] = compare_native(
                args.compare_first, args.compare_second
            )
        (args.output / filename).write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "kind": kind,
                    "attempts": len(data["attempts"]),
                    "unfinished": data["unfinished_attempts"],
                    "aggregates": [a["aggregate"] for a in data["attempts"]],
                }
            )
        )


if __name__ == "__main__":
    main()
