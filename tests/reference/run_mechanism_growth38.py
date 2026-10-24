"""Fixed 13-unit development block for actual native core-edge mechanism growth."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from math import isclose, sqrt
from pathlib import Path

from mechanism_growth38_cases import (
    KEYS,
    ROWS,
    SPECIES,
    independent_trajectory,
    scalar_derivatives,
    scalar_rates,
)

from chemworld.examples.native_mechanism_growth import example_growth_inputs, run_example
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    CoreEdgeGrowthPolicy,
    GrowthResourceLimits,
    MechanismGrowthCheckpoint,
)
from chemworld.physchem.reaction_generation import GenerationPolicy


class Record:
    def __init__(self, unit):
        self.unit, self.comparisons, self.evidence = unit, [], {}

    def check(self, label, actual, expected, *, numeric=False):
        if hasattr(actual, "item"):
            actual = actual.item()
        if hasattr(expected, "item"):
            expected = expected.item()
        passed = bool(
            isclose(actual, expected, rel_tol=3e-7, abs_tol=2e-10)
            if numeric
            else actual == expected
        )
        self.comparisons.append(
            {"label": label, "actual": actual, "expected": expected, "passed": passed}
        )


def summarize_report(report, artifact_path):
    """Keep complete denominators/decisions once; full checkpoints stay external."""
    first_growth = next(
        unit["evidence"]["growth"] for unit in report["units"] if "growth" in unit["evidence"]
    )

    def growth_summary(result, *, include_rounds):
        state = result["checkpoint"]["state"]
        return {
            "status": result["status"],
            "complete": result["complete"],
            "reasons": result["reasons"],
            "final_species": [s["species_id"] for s in result["network"]["species"]],
            "final_reactions": [r["reaction_id"] for r in result["network"]["reactions"]],
            "completed_rounds": state["completed_rounds"],
            "condition_solves": state["condition_solves"],
            "successful_solve_nfev": state["nfev"],
            "resource_history": state["resource_history"],
            "stops": state["stops"],
            "rounds": [
                {
                    "round": item["round"],
                    "core_ids": item["core_ids"],
                    "core_reactions": [r["reaction_id"] for r in item["core_network"]["reactions"]],
                    "edge_reactions": [r["reaction_id"] for r in item["edge_network"]["reactions"]],
                    "generation": {
                        "species_count": len(item["generation"]["species"]),
                        "reaction_count": len(item["generation"]["reactions"]),
                        "raw_matches": item["generation"]["raw_matches"],
                        "complete": item["generation"]["complete"],
                        "truncated_reasons": item["generation"]["truncated_reasons"],
                        "rejected": item["generation"]["rejected"],
                    },
                    "decisions": item["decisions"],
                    "conditions": [
                        {
                            "condition_id": case["condition_id"],
                            "sample_count": len(case["samples"]),
                            "solver_diagnostic": {
                                k: v
                                for k, v in case["trajectory"]["solver_diagnostic"].items()
                                if k
                                in {
                                    "success",
                                    "nfev",
                                    "njev",
                                    "nlu",
                                    "final_time_s",
                                    "evaluation_count",
                                    "maximum_conservation_drift_mol",
                                    "minimum_raw_amount_mol",
                                    "nonnegative_passed",
                                }
                            },
                        }
                        for case in item["conditions"]
                    ],
                }
                for item in state["history"]
                if include_rounds
            ],
            "pending": None
            if state["pending"] is None
            else {
                "round": state["pending"]["round"],
                "core_ids": state["pending"]["core_ids"],
                "completed_conditions": [c["condition_id"] for c in state["pending"]["conditions"]],
                "proposed_decisions": state["pending"].get("proposed_decisions"),
            },
            "last_generation_complete": (
                state["last_generation_attempt"]["complete"]
                if state["last_generation_attempt"]
                else None
            ),
            "dormant_edge_keys": state["dormant_edge_keys"],
        }

    units = []
    for unit in report["units"]:
        checks = unit["comparisons"]
        compact = {
            "unit": unit["unit"],
            "comparison_count": len(checks),
            "passed_comparisons": sum(bool(c["passed"]) for c in checks),
            "failures": [c for c in checks if not c["passed"]],
        }
        numeric = [
            abs(c["actual"] - c["expected"])
            for c in checks
            if type(c["actual"]) in (int, float) and type(c["expected"]) in (int, float)
        ]
        compact["maximum_absolute_numeric_discrepancy"] = max(numeric, default=0)
        compact["nonnumeric_checks"] = [
            {
                "label": c["label"],
                "passed": c["passed"],
                **(
                    {"actual": c["actual"], "expected": c["expected"]}
                    if not isinstance(c["actual"], (dict, list))
                    else {"comparison": "complete structured equality; values in external evidence"}
                ),
            }
            for c in checks
            if type(c["actual"]) not in (int, float)
        ]
        compact["evidence"] = {
            key: growth_summary(value, include_rounds=key == "growth")
            if key in {"growth", "stopped", "resumed"}
            else value
            for key, value in unit["evidence"].items()
        }
        units.append(compact)
    return {
        "model": "native-growth38-independent-summary-v1",
        "evidence_kind": "summary; full comparisons and checkpoints are in the external artifact",
        "full_artifact_path": str(artifact_path.resolve()),
        **{k: v for k, v in report.items() if k != "units"},
        "fixed_input_declarations": first_growth["checkpoint"]["design"],
        "other_grid_spacing_s": 0.125,
        "promotion_relative_thresholds": [0.001, 0.1, 10],
        "units": units,
    }


def growth_case(record, threshold, spacing):
    library, families, conditions = example_growth_inputs(sample_spacing_s=spacing)
    policy = CoreEdgeGrowthPolicy(
        promotion_relative_threshold=threshold, keep_relative_threshold=threshold / 2
    )
    result = grow_mechanism(library, families, ("ethyl",), conditions, policy=policy)
    record.evidence["growth"] = result.to_dict()
    record.check("complete", result.complete, True)
    expected_species = set(SPECIES) if threshold < 1 else {"ethyl", "H", "ethene", "butane"}
    expected_keys = set(KEYS) if threshold < 1 else set(KEYS[:2])
    record.check("final_species", sorted(result.network.species_ids), sorted(expected_species))
    record.check("final_reactions", sorted(result.network.reaction_ids), sorted(expected_keys))
    record.check("round_count", len(result.checkpoint.state["history"]), 3 if threshold < 1 else 2)
    condition_lookup = {c.condition_id: c for c in conditions}
    for step in result.checkpoint.state["history"]:
        label = f"round{step['round']}"
        record.check(
            label + "/all_conditions",
            [c["condition_id"] for c in step["conditions"]],
            sorted(condition_lookup),
        )
        core_names = set(step["core_ids"])
        core_keys = {r["reaction_id"] for r in step["core_network"]["reactions"]}
        edge_keys = {r["reaction_id"] for r in step["edge_network"]["reactions"]}
        for case in step["conditions"]:
            condition = condition_lookup[case["condition_id"]]
            independent = independent_trajectory(condition, core_keys)
            record.check(
                label + "/" + case["condition_id"] + "/all_samples",
                len(case["samples"]),
                len(condition.sample_times_s),
            )
            for i, (sample, expected) in enumerate(zip(case["samples"], independent, strict=True)):
                prefix = f"{label}/{condition.condition_id}/sample{i}"
                for species in sorted(core_names):
                    record.check(
                        prefix + "/amount/" + species,
                        sample["core_amounts_mol"][species],
                        expected[species],
                        numeric=True,
                    )
                rates = scalar_rates(expected, condition.temperature_K, condition.volume_L)
                derivative = scalar_derivatives(
                    expected, condition.temperature_K, condition.volume_L, core_keys
                )
                for species in sorted(core_names):
                    record.check(
                        prefix + "/core_production/" + species,
                        sample["core_net_production_mol_L_s"][species],
                        derivative[species],
                        numeric=True,
                    )
                record.check(
                    prefix + "/characteristic",
                    sample["characteristic_rate_mol_L_s"],
                    sqrt(sum(derivative[s] ** 2 for s in core_names)),
                    numeric=True,
                )
                for key in sorted(edge_keys):
                    record.check(
                        prefix + "/candidate/" + key,
                        sample["candidate_rates_mol_L_s"][key],
                        rates[key],
                        numeric=True,
                    )
                for species, actual in sample["edge_production_mol_L_s"].items():
                    expected_flux = sum(
                        right.get(species, 0) * rates[key]
                        for key, (_, left, right) in zip(KEYS, ROWS, strict=True)
                        if key in edge_keys and species not in left
                    )
                    record.check(
                        prefix + "/edge_production/" + species, actual, expected_flux, numeric=True
                    )
                n = sample["core_amounts_mol"]
                carbons = 2 * sum(n.get(s, 0) for s in ("ethyl", "ethene", "ethane")) + 4 * n.get(
                    "butane", 0
                )
                hydrogens = (
                    5 * n.get("ethyl", 0)
                    + 4 * n.get("ethene", 0)
                    + n.get("H", 0)
                    + 2 * n.get("H2", 0)
                    + 6 * n.get("ethane", 0)
                    + 10 * n.get("butane", 0)
                )
                initial = condition.initial_amounts_mol["ethyl"]
                record.check(prefix + "/carbon_conservation", carbons, 2 * initial, numeric=True)
                record.check(
                    prefix + "/hydrogen_conservation", hydrogens, 5 * initial, numeric=True
                )
    errors = {}
    for case in result.checkpoint.state["history"][-1]["conditions"]:
        condition = condition_lookup[case["condition_id"]]
        full = independent_trajectory(condition, set(KEYS))
        errors[condition.condition_id] = max(
            abs(sample["core_amounts_mol"].get(s, 0) - expected[s])
            for sample, expected in zip(case["samples"], full, strict=True)
            for s in SPECIES
        )
    record.evidence["full_mechanism_maximum_amount_error_mol"] = errors
    if threshold < 1:
        for name, error in errors.items():
            record.check("complete_mechanism_trajectory/" + name, error <= 2e-10, True)
    else:
        record.check(
            "subthreshold_edge_pruning",
            len(result.checkpoint.state["history"][-1]["decisions"]["pruned_edge_reactions"]),
            2,
        )
    return result, errors


def run_block(output):
    if output.exists():
        raise ValueError("preserve existing evidence; choose a fresh output file")
    records, outcomes, errors = [], {}, {}
    started = time.perf_counter()

    def unit(name, operation):
        record = Record(name)
        try:
            returned = operation(record)
        except Exception as error:
            record.evidence["exception"] = {"type": type(error).__name__, "message": str(error)}
            record.check("execution_completed", False, True)
            returned = None
        records.append(record)
        elapsed = time.perf_counter() - started
        print(
            f"stage=reference units={len(records)}/13 throughput={len(records) / elapsed:.3f}/s "
            f"eta_s={(13 - len(records)) * elapsed / len(records):.1f} unit={name}",
            flush=True,
        )
        return returned

    for spacing in (0.25, 0.125):
        for threshold in (0.001, 0.1, 10):
            returned = unit(
                f"growth-grid{spacing}-threshold{threshold}",
                lambda r, t=threshold, s=spacing: growth_case(r, t, s),
            )
            if returned is not None:
                outcomes[spacing, threshold], errors[spacing, threshold] = returned
    library, families, conditions = example_growth_inputs()

    def restart(record, limits):
        first = grow_mechanism(library, families, ("ethyl",), conditions, resources=limits)
        record.evidence["stopped"] = first.to_dict()
        record.check("resource_status", first.status, "resource_limit")
        restored = MechanismGrowthCheckpoint.from_dict(
            json.loads(json.dumps(first.checkpoint.to_dict()))
        )
        second = grow_mechanism(library, families, ("ethyl",), conditions, checkpoint=restored)
        record.evidence["resumed"] = second.to_dict()
        baseline = outcomes[0.25, 0.001]
        record.check("final_network", second.network.to_dict(), baseline.network.to_dict())
        record.check(
            "all_decisions_and_trajectories",
            second.checkpoint.state["history"] == baseline.checkpoint.state["history"],
            True,
        )
        record.check(
            "condition_solves_not_repeated", second.checkpoint.state["condition_solves"], 12
        )
        record.check("resource_stop_retained", len(second.checkpoint.state["stops"]), 1)
        record.check("extension_retained", len(second.checkpoint.state["resource_history"]), 2)

    for name, limits in (
        ("condition", GrowthResourceLimits(maximum_condition_solves=1)),
        ("round", GrowthResourceLimits(maximum_rounds=1)),
        ("core", GrowthResourceLimits(maximum_core_species=1)),
    ):
        unit("resume-" + name, lambda r, lim=limits: restart(r, lim))

    def missing(record, species):
        if species:
            changed = replace(
                library,
                species=tuple(b for b in library.species if b.species.species_id != "butane"),
                reactions=tuple(r for r in library.reactions if "butane" not in r.products),
            )
        else:
            changed = replace(library, reactions=library.reactions[1:])
        result = grow_mechanism(changed, families, ("ethyl",), conditions)
        record.evidence["stopped"] = result.to_dict()
        record.check("missing_data_status", result.status, "missing_data")
        record.check("unknown_not_zero", result.checkpoint.state["condition_solves"], 0)
        record.check(
            "candidate_retained",
            bool(result.checkpoint.state["last_generation_attempt"]["reactions"]),
            True,
        )
        record.check(
            "explicit_reason",
            any(
                r["kind"] == ("species_binding" if species else "kinetics_binding")
                for r in result.reasons
            ),
            True,
        )

    unit("missing-species", lambda r: missing(r, True))
    unit("missing-kinetics", lambda r: missing(r, False))

    def truncated(record):
        result = grow_mechanism(
            library,
            families,
            ("ethyl",),
            conditions,
            policy=CoreEdgeGrowthPolicy(generation=GenerationPolicy(maximum_matches=1)),
        )
        record.evidence["stopped"] = result.to_dict()
        record.check("truncated_status", result.status, "generation_truncated")
        record.check("not_completed", result.complete, False)
        record.check("reason_retained", bool(result.reasons), True)
        record.check(
            "no_trajectory_on_incomplete_generation", result.checkpoint.state["condition_solves"], 0
        )

    unit("enumeration-truncated", truncated)

    def world(record):
        result = run_example()
        record.evidence["consumer"] = {k: v for k, v in result.items() if k != "growth"}
        record.check("time", result["world_time_s"], 1.0, numeric=True)
        record.check("snapshot_replay", result["snapshot_replay_equal"], True)
        c = replace(conditions[0], sample_times_s=(0, 1))
        expected = independent_trajectory(c, set(KEYS))[-1]
        for s in SPECIES:
            record.check(
                "world_amount/" + s, result["world_amounts_mol"][s], expected[s], numeric=True
            )

    unit("original-world-consumer", world)
    # Predetermined cross-grid and threshold comparisons belong to their original
    # six units; missing runs are failures, never removed from the denominator.
    for i, spacing in enumerate((0.25, 0.125)):
        for j, threshold in enumerate((0.001, 0.1, 10)):
            record = records[3 * i + j]
            if (spacing, threshold) not in outcomes:
                continue
            if spacing == 0.125:
                record.check(
                    "grid_membership_unchanged",
                    sorted(outcomes[spacing, threshold].network.species_ids),
                    sorted(outcomes[0.25, threshold].network.species_ids),
                )
                record.check(
                    "grid_reactions_unchanged",
                    sorted(outcomes[spacing, threshold].network.reaction_ids),
                    sorted(outcomes[0.25, threshold].network.reaction_ids),
                )
            if threshold < 1 and (spacing, 10) in errors:
                record.check(
                    "error_does_not_increase_with_stricter_threshold",
                    all(
                        v <= errors[spacing, 10][k] + 2e-10
                        for k, v in errors[spacing, threshold].items()
                    ),
                    True,
                )
    comparisons = [c for r in records for c in r.comparisons]
    failures = [{"unit": r.unit, **c} for r in records for c in r.comparisons if not c["passed"]]
    report = {
        "mode": "development",
        "design": "MECHANISM_GROWTH_EXPERIMENT.md",
        "unit_count": 13,
        "completed_units": len(records),
        "passed_units": sum(all(c["passed"] for c in r.comparisons) for r in records),
        "comparison_count": len(comparisons),
        "passed_comparisons": sum(c["passed"] for c in comparisons),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "units": [
            {"unit": r.unit, "comparisons": r.comparisons, "evidence": r.evidence} for r in records
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"units", "failures"}}), flush=True
    )
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summarize",
        type=Path,
        help="read an existing complete block; write compact summary without rerunning",
    )
    args = parser.parse_args()
    if args.summarize is not None:
        summary = summarize_report(
            json.loads(args.summarize.read_text(encoding="utf-8")), args.summarize
        )
        encoded = json.dumps(summary, indent=2, allow_nan=False)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in ("unit_count", "comparison_count", "passed_comparisons", "failures")
                }
            )
        )
    else:
        raise SystemExit(run_block(args.output))
