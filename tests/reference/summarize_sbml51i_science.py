"""Complete compact histories of the fixed original-SBML science blocks."""

import argparse
import json
from collections import Counter
from pathlib import Path


def compact_engine(engine):
    result = {k: v for k, v in engine.items() if k not in {"failures", "generator"}}
    result["failures"] = [
        {
            k: r[k]
            for k in (
                "id",
                "seed",
                "stage",
                "error",
                "simulation_attempted",
                "simulation_completed",
            )
            if k in r
        }
        for r in engine["failures"]
    ]
    result["generator"] = engine["generator"]
    return result


def summarize(path):
    raw = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    result = {k: v for k, v in raw.items() if k != "units"}
    engines = [e for u in raw["units"] for e in u["engines"].values()]
    result.update(
        raw_block=path.name,
        encoding_model_units_planned=28,
        encoding_model_units_passed=sum(u["passed"] for u in raw["units"]),
        method_cells_planned=112,
        method_cells_recorded=len(engines),
        method_cells_passed=sum(e["passed"] for e in engines),
        simulation_attempted=sum(e["simulation_attempted"] for e in engines),
        simulation_completed=sum(e["simulation_completed"] for e in engines),
        report_points=sum(e.get("report_points", 0) for e in engines),
        unrecorded=sum(e["unrecorded"] for e in engines),
        trajectory_errors=sum(len(e["failures"]) for e in engines),
        invalid_trajectories=sum(len(e["invalid_trajectories"]) for e in engines),
        malformed_rows=sum(len(e["malformed_rows"]) for e in engines),
        generator_comparisons=sum(e["generator"].get("comparisons", 0) for e in engines),
        generator_passed=sum(e["generator"]["passed"] for e in engines),
    )
    result["unstarted_time_courses"] = raw["planned_trajectories"] - result["simulation_attempted"]
    checks = [c for e in engines for c in e.get("statistics", {}).get("checks", [])]
    result["statistics"] = {
        "family_size": 2000,
        "alpha_per_family": 0.005,
        "declared_weak_bias_allowances": {"direct": 0.0, "next": 0.0, "coarse": 0.08, "fine": 0.01},
        "reference_numeric_allowance": 1e-11,
        "comparisons": len(checks),
        "passed": sum(c["passed"] for c in checks),
        "failed_comparisons": [c for c in checks if not c["passed"]],
    }
    result["strategy_statistics"] = {}
    for strategy in ("direct", "next", "coarse", "fine"):
        cells = [(u, u["engines"][strategy]) for u in raw["units"]]
        comparisons = [c for _, e in cells for c in e.get("statistics", {}).get("checks", [])]
        cdfs = [
            {"unit": u["id"], "encoding": u["encoding"], **c}
            for u, e in cells
            for c in e.get("statistics", {}).get("checks", [])
            if "cdf" in c["name"]
        ]
        result["strategy_statistics"][strategy] = {
            "encoding_model_cells_passed": sum(e["passed"] for _, e in cells),
            "statistical_comparisons": len(comparisons),
            "statistical_passed": sum(c["passed"] for c in comparisons),
            "cdf_comparisons": len(cdfs),
            "maximum_cdf_difference": max((c["actual"] for c in cdfs), default=None),
            "cdf_limit_range": [min(c["limit"] for c in cdfs), max(c["limit"] for c in cdfs)]
            if cdfs
            else None,
            "worst_cdf": max(cdfs, key=lambda c: c["actual"]) if cdfs else None,
        }
    if raw["mode"] == "resource":
        observations = [e["observations"] for e in engines if "observations" in e]
        result["timed_trials"] = sum(len(o["wall_s"]) for o in observations)
        result["zero_cpu_deltas"] = sum(o["cpu_zero_deltas"] for o in observations)
        result["allocation_trials"] = len(observations)
        result["timing_scope"] = (
            "original_SBML_integer_advance_excluding_XML_spec_IO_and_outer_runtime"
        )
        expected = [(997, "warmup")] + [(s, "timed") for s in (1, 7, 17, 37, 73, 101, 251, 509)]
        expected += [(17, "allocation")]
        purposes, failures = Counter(), []
        for unit in raw["units"]:
            for strategy in unit["engines"]:
                records = [
                    json.loads(line)
                    for line in (
                        path
                        / (unit["id"] + "-" + unit["encoding"])
                        / strategy
                        / "trajectories.jsonl"
                    )
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                purposes.update(r.get("purpose") for r in records)
                if [(r.get("seed"), r.get("purpose")) for r in records] != expected:
                    failures.append(
                        {"unit": unit["id"], "encoding": unit["encoding"], "strategy": strategy}
                    )
        result["trial_purpose_counts"] = dict(purposes)
        result["trial_schedule_failures"] = failures
        result["trial_schedule_verified"] = not failures and purposes == {
            "warmup": 112,
            "timed": 896,
            "allocation": 112,
        }
        result["passed"] = result["passed"] and result["trial_schedule_verified"]
    result["units"] = [
        {
            **{k: v for k, v in u.items() if k != "engines"},
            "engines": {s: compact_engine(e) for s, e in u["engines"].items()},
        }
        for u in raw["units"]
    ]
    if raw["mode"] == "wiring":
        for unit in result["units"]:
            for strategy, engine in unit["engines"].items():
                filename = (
                    path / (unit["id"] + "-" + unit["encoding"]) / strategy / "trajectories.jsonl"
                )
                records = [
                    json.loads(line) for line in filename.read_text(encoding="utf-8").splitlines()
                ]
                good = [r for r in records if r["status"] == "ok"]
                engine["actual_source"] = {
                    "versions": sorted({r["version"] for r in records}),
                    "method_names": sorted({r["method_name"] for r in good}),
                    "method_enums": sorted({r["method_enum"] for r in records}),
                    "configured_avogadro": sorted(
                        {r["avogadro"] for r in records if "avogadro" in r}
                    ),
                    "imported_avogadro": sorted(
                        {r["imported_avogadro"] for r in records if "imported_avogadro" in r}
                    ),
                    "quantity_to_number": sorted(
                        {r["quantity_to_number"] for r in records if "quantity_to_number" in r}
                    ),
                    "source_functions": good[0]["functions"] if good else None,
                    "typed_parameter_readbacks": [r["parameter_readback"] for r in good],
                    "initial_particle_quantities": [
                        r["initial"] for r in records if "initial" in r
                    ],
                    "initial_macroscopic_particle_fluxes": [
                        r["initial_fluxes"] for r in records if "initial_fluxes" in r
                    ],
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-item-wiring", type=Path)
    parser.add_argument("--diagnostics", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    attempts = [summarize(p) for p in args.attempts]
    result = {
        "schema": "chemworld-sbml51i-science-history-v1",
        "attempts": attempts,
        "diagnostics": [
            {
                "raw_block": p.name,
                "records": json.loads((p / "summary.json").read_text(encoding="utf-8")),
            }
            for p in args.diagnostics
        ],
        "parent51_complete": False,
        "retained_source_fine_tau_S12_failure": (
            "H source full distribution remains 13/14 models, 55/56 methods; "
            "wiring does not replace it"
        ),
    }
    if args.old_item_wiring:
        current = args.attempts[-1]
        if not all(
            json.loads((p / "summary.json").read_text(encoding="utf-8"))["passed"]
            for p in (args.old_item_wiring, current)
        ):
            raise ValueError("compatibility views require two previously admitted complete blocks")
        comparisons, failures = 0, []
        for model in range(1, 15):
            identity = f"S{model:02d}"
            for strategy in ("direct", "next", "coarse", "fine"):
                paths = (
                    args.old_item_wiring / identity / strategy / "trajectories.jsonl",
                    current / (identity + "-item_amount") / strategy / "trajectories.jsonl",
                )
                records = [
                    [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
                    for p in paths
                ]
                if [len(r) for r in records] != [4, 4]:
                    failures.append(
                        {"unit": identity, "strategy": strategy, "error": "incomplete source grids"}
                    )
                    continue
                for index, (old, new) in enumerate(zip(*records, strict=True)):

                    def view(row):
                        return [
                            {
                                "time_s": p["time_s"],
                                "counts": {k: round(v) for k, v in p["populations"].items()},
                            }
                            for p in row.get("rows", [])
                        ]

                    ok = (
                        old["status"] == new["status"] == "ok"
                        and len(view(old)) == 25
                        and old["seed"] == new["seed"]
                        and view(old) == view(new)
                        and old["rng_prefix"] == new["rng_prefix"]
                    )
                    comparisons += 1
                    if not ok:
                        failures.append({"unit": identity, "strategy": strategy, "index": index})
        result["original_item_wiring_compatibility"] = {
            "planned": 224,
            "compared": comparisons,
            "failures": failures,
            "passed": comparisons == 224 and not failures,
            "scope": "admitted integer report views and RNG prefixes; no bitwise float claim",
        }
    result["attempt_outcomes"] = dict(Counter(a["passed"] for a in attempts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "attempts": len(attempts),
                "latest_passed": attempts[-1]["passed"],
                "latest_recorded": attempts[-1]["recorded_trajectories"],
            }
        )
    )


if __name__ == "__main__":
    main()
