"""Preserve every complete/unfinished G distribution or resource attempt."""

import argparse
import json
from collections import Counter
from pathlib import Path


def distribution_diagnostics(directory, data):
    aggregates = {}
    for engine in ("native", "reference", "copasi"):
        summaries = [u["engines"][engine] for u in data["units"]]
        totals = {
            key: sum(s[key] for s in summaries)
            for key in (
                "planned",
                "recorded",
                "successful",
                "simulation_attempted",
                "simulation_completed",
                "unstarted",
                "checked_output_points",
            )
        }
        totals["invalid_trajectories"] = sum(len(s["invalid_trajectories"]) for s in summaries)
        totals["failed_calls"] = sum(len(s["failed_calls"]) for s in summaries)
        aggregates[engine] = totals
    for unit in data["units"]:
        path = directory / unit["id"] / "copasi" / "trajectories.jsonl"
        versions, methods, negative_points, minima, partial = set(), set(), 0, {}, 0
        rows_with_negatives, negative_species = 0, Counter()
        partition = None
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if "version" in row:
                        versions.add(row["version"])
                    if "method_name" in row:
                        methods.add(row["method_name"])
                    partition = partition or row.get("deterministic_reactions")
                    points = row.get("rows", row.get("partial_rows", []))
                    partial += len(row.get("partial_rows", []))
                    negative = False
                    for point in points:
                        bad_point = False
                        for key, value in point["populations"].items():
                            if not isinstance(value, (float, int)):
                                continue
                            minima[key] = min(minima.get(key, value), value)
                            if value < 0:
                                negative_species[key] += 1
                                negative = bad_point = True
                        negative_points += bad_point
                    rows_with_negatives += negative
        unit["source_output_diagnostic"] = {
            "scope": (
                "Descriptive raw-output inspection; qualification gates unchanged, "
                "no clipping or discarded rows"
            ),
            "versions": sorted(versions),
            "methods": sorted(methods),
            "negative_points": negative_points,
            "trajectories_with_negative_points": rows_with_negatives,
            "negative_entries_by_species": dict(negative_species),
            "minimum_particles_by_species": minima,
            "partial_output_points": partial,
            "actual_partition_sample": partition,
        }
    comparisons = [c for u in data["units"] for c in u["checks"]["native"]]
    cdfs = [c for c in comparisons if c["name"].startswith("cdf_")]
    data["aggregate"] = {
        "engines": aggregates,
        "native_units_passed": sum(u["native_passed"] for u in data["units"]),
        "source_units_passed": sum(u["source_passed"] for u in data["units"]),
        "native_comparison_checks": len(comparisons),
        "native_comparison_checks_passed": sum(c["passed"] for c in comparisons),
        "native_cdf_checks": len(cdfs),
        "maximum_native_cdf_difference": max((c["value"] for c in cdfs), default=None),
        "minimum_native_cdf_limit": min((c["limit"] for c in cdfs), default=None),
        "rare_event_checks": {
            u["id"]: {
                e: d.get("rare_event_check", {"not_evaluated": "fixed sample failed validity gate"})
                for e, d in u["engines"].items()
            }
            for u in data["units"]
            if u["case_index"] == 2 and u["kind"] == "cross"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--kind", choices=("distribution", "resource"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51g-" + args.kind + "-*")):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if not path.is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if args.kind == "distribution":
            distribution_diagnostics(directory, data)
        else:
            data["aggregate"] = {
                "models_passed": sum(u["passed"] for u in data["units"]),
                "native_method_units": 2 * len(data["units"]),
                "native_advances_recorded": sum(
                    u["engines"][e]["recorded"] for u in data["units"] for e in ("RK45", "DOP853")
                ),
                "native_advances_successful": sum(
                    u["engines"][e]["successful"] for u in data["units"] for e in ("RK45", "DOP853")
                ),
                "independent_evaluations_recorded": sum(
                    u["engines"]["reference"]["recorded"] for u in data["units"]
                ),
                "checks": sum(len(u["checks"]) for u in data["units"]),
                "checks_passed": sum(c["passed"] for u in data["units"] for c in u["checks"]),
            }
        attempts.append({"attempt": directory.name, "raw_evidence": str(directory), **data})
    result = {
        "schema": "chemworld-hybrid51g-" + args.kind + "-history-v1",
        "scope": (
            "Development qualification; all fixed denominators/failures retained; "
            "native/source and performance boundaries separate"
        ),
        "attempts": attempts,
        "unfinished_attempts": unfinished,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "attempts": len(attempts),
                "unfinished": unfinished,
                "aggregates": [a["aggregate"] for a in attempts],
            }
        )
    )


if __name__ == "__main__":
    main()
