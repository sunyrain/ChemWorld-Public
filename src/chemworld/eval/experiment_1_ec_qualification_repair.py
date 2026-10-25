"""Frozen helpers for the Experiment 1 EC qualification repair v1.0.2."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from chemworld.eval import work_ii_structural_candidate_qualification as structural
from chemworld.eval.experiment_1_ec_qualification import (
    EXPECTED_COMMON_GATES,
    Experiment1ECQualificationError,
    analyze_entity_world,
    analyze_structural_world,
    entity_prior_audit,
    private_world_audit,
)
from chemworld.eval.experiment_1_ec_qualification import _validate_binding as _validate_binding
from chemworld.eval.provenance import canonical_json_sha256, file_sha256

REPAIR_CONTRACT_VERSION = "chemworld-experiment-1-ec-qualification-repair-contract-1.0.2"
ENTITY_REPORT_VERSION = "chemworld-experiment-1-ec-entity-world-report-1.0.2"
STRUCTURAL_REPORT_VERSION = "chemworld-experiment-1-ec-structural-world-report-1.0.2"
EXPECTED_WORLD_IDS = tuple(f"EC-W0{index}" for index in range(1, 6))
EXPECTED_WORLD_SEEDS = tuple(range(5))
EXPECTED_ENTITY_PERMUTATIONS = {
    "EC-W01": (3, 1, 2, 0),
    "EC-W02": (2, 1, 0, 3),
    "EC-W03": (0, 3, 2, 1),
    "EC-W04": (0, 1, 3, 2),
    "EC-W05": (0, 2, 1, 3),
}
EXPECTED_STRUCTURAL_GROUPS = ((0, 0), (0, 1), (0, 2))


def load_repair_contract(root: Path, path: Path) -> dict[str, Any]:
    """Load and validate the frozen repair contract without requiring ignored evidence."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Experiment1ECQualificationError("EC repair contract must be an object")
    errors = validate_repair_contract(root, value)
    if errors:
        raise Experiment1ECQualificationError("; ".join(errors))
    return value


def validate_repair_contract(root: Path, contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != REPAIR_CONTRACT_VERSION:
        errors.append("repair contract schema version changed")
    if contract.get("status") != "development_frozen_before_execution":
        errors.append("repair contract is not frozen before execution")
    if contract.get("development_only") is not True:
        errors.append("repair contract must remain development-only")
    if contract.get("participant_provider_calls") != 0:
        errors.append("repair qualification must make zero provider calls")
    if contract.get("participant_execution_authorized") is not False:
        errors.append("participant execution must remain disabled")
    if contract.get("formal_benchmark_execution_authorized") is not False:
        errors.append("formal benchmark execution must remain disabled")
    if tuple(contract.get("common_gates", ())) != EXPECTED_COMMON_GATES:
        errors.append("Q1-Q8 gate registry changed")

    for key in ("parent_contract", "specification", "repair_note"):
        _validate_binding(root, contract.get(key), key, errors)
    task = contract.get("task")
    if not isinstance(task, Mapping):
        errors.append("task contract is missing")
    else:
        _validate_binding(root, task.get("campaign_config"), "campaign_config", errors)
        expected_task = {
            "system_id": "EC",
            "task_id": "electrochemical-conversion",
            "world_split": "public-test",
            "objective": "balanced",
            "electrochemical_material_family_id": "nominal-prior-latent-v2",
            "electrochemical_workflow_mode": "autonomous_open_v1",
            "scoring_contract_id": "electrochemical-s0-balanced-efficiency-v2",
        }
        for key, expected in expected_task.items():
            if task.get(key) != expected:
                errors.append(f"task.{key} changed")

    source_assets = contract.get("source_assets")
    if not isinstance(source_assets, Mapping):
        errors.append("source asset bindings are missing")
    else:
        for key in ("parametric_reference_summary", "parametric_noise_summary"):
            _validate_binding(root, source_assets.get(key), key, errors)

    compatibility = contract.get("carry_forward_compatibility")
    if not isinstance(compatibility, Mapping):
        errors.append("parametric carry-forward compatibility contract is missing")
    else:
        if compatibility.get("locus") != "parametric":
            errors.append("carry-forward locus changed")
        if compatibility.get("required_truth_match_with_repair_worlds") is not True:
            errors.append("carry-forward truth-match requirement changed")
        code_bindings = compatibility.get("source_code_bindings")
        if not isinstance(code_bindings, list) or not code_bindings:
            errors.append("carry-forward source-code bindings are missing")
        else:
            for index, binding in enumerate(code_bindings):
                _validate_binding(root, binding, f"carry_forward_source_code[{index}]", errors)

    worlds = contract.get("worlds")
    rows = worlds.get("qualification") if isinstance(worlds, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 5:
        errors.append("exactly five repair qualification worlds are required")
    else:
        if tuple(row.get("world_id") for row in rows) != EXPECTED_WORLD_IDS:
            errors.append("repair world IDs changed")
        if tuple(row.get("world_seed") for row in rows) != EXPECTED_WORLD_SEEDS:
            errors.append("repair world seeds changed")
    canary = worlds.get("canary") if isinstance(worlds, Mapping) else None
    if not isinstance(canary, Mapping) or canary.get("world_id") != "EC-W00":
        errors.append("EC-W00 repair canary is missing")
    elif canary.get("world_seed") != 900000 or canary.get("formal_denominator") is not False:
        errors.append("EC-W00 repair canary semantics changed")

    loci = contract.get("loci")
    entity = loci.get("entity") if isinstance(loci, Mapping) else None
    structural_locus = loci.get("structural") if isinstance(loci, Mapping) else None
    if not isinstance(entity, Mapping):
        errors.append("entity repair locus is missing")
    else:
        permutations = entity.get("descriptor_permutation_by_world")
        if not isinstance(permutations, Mapping):
            errors.append("world-balanced entity permutations are missing")
        else:
            actual = {
                str(world_id): tuple(int(value) for value in permutation)
                for world_id, permutation in permutations.items()
                if isinstance(permutation, Sequence)
            }
            if actual != EXPECTED_ENTITY_PERMUTATIONS:
                errors.append("world-balanced entity permutations changed")
            for world_id, permutation in actual.items():
                if not _is_transposition(permutation):
                    errors.append(f"{world_id} entity mapping is not one transposition")
        if entity.get("independent_replicates") != 3:
            errors.append("entity repeat count changed")
        if entity.get("minimum_absolute_separation") != 0.05:
            errors.append("entity absolute separation gate changed")
        if entity.get("minimum_signal_to_noise_ratio") != 2.0:
            errors.append("entity SNR gate changed")
        if entity.get("participant_unique_experiment_budget") != 4:
            errors.append("entity participant budget changed")
        if entity.get("observation_noise_namespace") != ("experiment-1-v1.0.2-ec-entity-repair"):
            errors.append("entity repair noise namespace changed")
    if not isinstance(structural_locus, Mapping):
        errors.append("structural repair locus is missing")
    else:
        groups = structural_locus.get("validation_groups")
        actual_groups = (
            tuple(tuple(int(value) for value in row) for row in groups)
            if isinstance(groups, Sequence)
            else ()
        )
        if actual_groups != EXPECTED_STRUCTURAL_GROUPS:
            errors.append("structural repair validation groups changed")
        if structural_locus.get("effect_floor") != 0.03:
            errors.append("structural effect floor changed")
        if structural_locus.get("noise_multiplier") != 6.0:
            errors.append("structural noise multiplier changed")
        if structural_locus.get("minimum_disagreement_fraction") != 0.4:
            errors.append("structural disagreement gate changed")
        if structural_locus.get("participant_unique_experiment_budget") != 4:
            errors.append("structural participant budget changed")

    scope = contract.get("repair_scope")
    if not isinstance(scope, Mapping):
        errors.append("repair scope is missing")
    elif (
        scope.get("rerun_loci") != ["entity", "structural"]
        or scope.get("carry_forward_loci") != ["parametric"]
        or scope.get("world_truth_change") is not False
        or scope.get("gate_change") is not False
        or scope.get("participant_budget_change") is not False
    ):
        errors.append("repair scope changed")
    execution = contract.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("repair execution policy is missing")
    elif (
        execution.get("exact_replay_required") is not True
        or execution.get("overwrite_forbidden") is not True
        or execution.get("post_failure_redesign_in_same_campaign_forbidden") is not True
        or execution.get("participant_execution_authorized") is not False
        or execution.get("formal_benchmark_execution_authorized") is not False
    ):
        errors.append("repair execution safety policy changed")
    return errors


def entity_contract_view(contract: Mapping[str, Any], *, world_id: str) -> dict[str, Any]:
    """Adapt the repair contract to the immutable v1.0.1 entity helpers."""
    value = copy.deepcopy(dict(contract))
    entity = value["loci"]["entity"]
    if world_id == "EC-W00":
        permutation = entity["canary_descriptor_permutation"]
    else:
        permutation = entity["descriptor_permutation_by_world"][world_id]
    entity["descriptor_permutation"] = list(permutation)
    return value


def entity_prior_audit_repair(contract: Mapping[str, Any], *, world_id: str) -> dict[str, Any]:
    return entity_prior_audit(entity_contract_view(contract, world_id=world_id))


def private_world_audit_repair(
    contract: Mapping[str, Any], *, world_id: str, world_seed: int
) -> dict[str, Any]:
    return private_world_audit(
        entity_contract_view(contract, world_id=world_id), world_seed=world_seed
    )


def analyze_entity_repair_world(
    contract: Mapping[str, Any],
    *,
    world_id: str,
    world_seed: int,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = analyze_entity_world(
        entity_contract_view(contract, world_id=world_id),
        world_id=world_id,
        world_seed=world_seed,
        receipts=receipts,
    )
    report["schema_version"] = ENTITY_REPORT_VERSION
    report["repair_contract_sha256"] = canonical_json_sha256(contract)
    report["descriptor_permutation"] = list(
        contract["loci"]["entity"]["descriptor_permutation_by_world"][world_id]
    )
    report["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report


def structural_repair_queries(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    locus = contract["loci"]["structural"]
    candidate_id = str(locus["candidate_id"])
    spec = structural.candidate_specs()[candidate_id]
    axis_a, axis_b = spec["axis_names"]
    levels_a, levels_b = spec["axis_levels"]
    rows: list[dict[str, Any]] = []
    for axis_a_index, axis_a_value in enumerate(levels_a):
        for axis_b_index, axis_b_value in enumerate(levels_b):
            rows.append(
                {
                    "query_id": f"a{axis_a_index}-b{axis_b_index}",
                    "phase": "main_grid",
                    "axis_a_index": axis_a_index,
                    "axis_b_index": axis_b_index,
                    "feature_values": {
                        **spec["fixed_context"],
                        axis_a: axis_a_value,
                        axis_b: axis_b_value,
                    },
                    "metric_ids": list(spec["metrics"]),
                }
            )
    groups = _structural_groups(contract)
    replicates = int(locus["validation_replicates"])
    for group_index, (axis_a_index, axis_b_index) in enumerate(groups):
        for replicate in range(1, replicates + 1):
            rows.append(
                {
                    "query_id": f"validation-g{group_index}-r{replicate}",
                    "phase": "noisy_validation",
                    "validation_group": group_index,
                    "replicate": replicate,
                    "axis_a_index": axis_a_index,
                    "axis_b_index": axis_b_index,
                    "feature_values": {
                        **spec["fixed_context"],
                        axis_a: levels_a[axis_a_index],
                        axis_b: levels_b[axis_b_index],
                    },
                    "metric_ids": list(spec["metrics"]),
                }
            )
    return rows


def analyze_structural_candidate_repair(
    contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    locus = contract["loci"]["structural"]
    candidate_id = str(locus["candidate_id"])
    spec = structural.candidate_specs()[candidate_id]
    main = [row for row in rows if row.get("phase") == "main_grid"]
    validation = [row for row in rows if row.get("phase") == "noisy_validation"]
    groups = _structural_groups(contract)
    expected = 9 + len(groups) * int(locus["validation_replicates"])
    checks: dict[str, bool] = {
        "fixed_query_count": len(rows) == expected,
        "main_grid_count": len(main) == 9,
        "validation_count": len(validation) == len(groups) * int(locus["validation_replicates"]),
        "all_outcomes_classified": all(
            row.get("status") in {"completed", "physical_failure", "platform_failure"}
            for row in rows
        ),
        "zero_platform_failures": not any(row.get("status") == "platform_failure" for row in rows),
        "all_exact_replay": all(row.get("exact_replay") is True for row in rows),
    }
    completed_main = [row for row in main if row.get("status") == "completed"]
    completed_validation = [row for row in validation if row.get("status") == "completed"]
    checks["complete_main_surface"] = len(completed_main) == 9
    checks["complete_validation_surface"] = len(completed_validation) == len(groups) * int(
        locus["validation_replicates"]
    )
    if not all(checks.values()):
        result = structural._early_result(candidate_id, rows, checks)
        result["validation_groups"] = [list(group) for group in groups]
        return result

    sigma = structural._validation_sigma(completed_validation, spec["metrics"], groups=groups)
    effects = structural._electrochemical_effects(completed_main, sigma)
    model = _structural_model_qualification_repair(
        completed_main,
        completed_validation,
        sigma=sigma,
        metrics=spec["model_metrics"],
        aligned_features=structural._electrochemical_aligned_features,
        misspecified_features=structural._electrochemical_misspecified_features,
        validation_groups=groups,
        target_axis="b",
        candidate_id=candidate_id,
        effect_floor=float(locus["effect_floor"]),
        noise_multiplier=float(locus["noise_multiplier"]),
        minimum_disagreement_fraction=float(locus["minimum_disagreement_fraction"]),
    )
    checks.update(
        {
            "axis_a_effect": bool(effects["axis_a"]["passed"]),
            "axis_b_effect": bool(effects["axis_b"]["passed"]),
            "topology_signature": bool(effects["topology_signature"]["passed"]),
            "baseline_error_matched": bool(model["checks"]["baseline_error_matched"]),
            "held_out_disagreement": bool(model["checks"]["held_out_disagreement"]),
            "low_counterexample_region": bool(model["checks"]["low_counterexample_region"]),
            "high_counterexample_region": bool(model["checks"]["high_counterexample_region"]),
            "blind_identification": bool(model["checks"]["blind_identification"]),
            "prior_schema_matched": bool(model["checks"]["prior_schema_matched"]),
            "prior_word_count_matched": bool(model["checks"]["prior_word_count_matched"]),
        }
    )
    return {
        "candidate_id": candidate_id,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, passed in checks.items() if not passed),
        "denominators": structural._denominators(rows),
        "validation_groups": [list(group) for group in groups],
        "validation_sigma": sigma,
        "effects": effects,
        "model_qualification": model,
        "prior_arms": structural.build_prior_arms(candidate_id),
    }


def analyze_structural_repair_world(
    contract: Mapping[str, Any],
    *,
    world_id: str,
    world_seed: int,
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    truth_sha256: str,
) -> dict[str, Any]:
    report = analyze_structural_world(
        contract,
        world_id=world_id,
        world_seed=world_seed,
        rows=rows,
        analysis=analysis,
        truth_sha256=truth_sha256,
    )
    report["schema_version"] = STRUCTURAL_REPORT_VERSION
    report["repair_contract_sha256"] = canonical_json_sha256(contract)
    report["validation_groups"] = [list(group) for group in _structural_groups(contract)]
    report["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report


def validate_parent_parametric_evidence(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable v1.0.1 EC-P evidence before composite carry-forward."""
    parent = contract["parent_result"]
    registry_path = root / str(parent["registry"]["path"])
    summary_path = root / str(parent["summary"]["path"])
    checks = {
        "registry_exists": registry_path.is_file(),
        "summary_exists": summary_path.is_file(),
    }
    registry: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    if checks["registry_exists"]:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        checks["registry_file_hash"] = file_sha256(registry_path) == parent["registry"]["sha256"]
        checks["registry_self_hash"] = (
            registry.get("registry_sha256")
            == parent["registry"]["self_sha256"]
            == canonical_json_sha256(
                {key: value for key, value in registry.items() if key != "registry_sha256"}
            )
        )
    else:
        checks["registry_file_hash"] = False
        checks["registry_self_hash"] = False
    if checks["summary_exists"]:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checks["summary_file_hash"] = file_sha256(summary_path) == parent["summary"]["sha256"]
        checks["summary_self_hash"] = (
            summary.get("summary_sha256")
            == parent["summary"]["self_sha256"]
            == canonical_json_sha256(
                {key: value for key, value in summary.items() if key != "summary_sha256"}
            )
        )
    else:
        checks["summary_file_hash"] = False
        checks["summary_self_hash"] = False
    parametric_rows = [
        row for row in registry.get("rows", []) if row.get("prior_locus") == "parametric"
    ]
    checks["five_parametric_rows"] = len(parametric_rows) == 5
    checks["all_parametric_qualified"] = bool(parametric_rows) and all(
        row.get("status") == "qualified" for row in parametric_rows
    )
    checks["world_ids_unchanged"] = (
        tuple(row.get("world_id") for row in parametric_rows) == EXPECTED_WORLD_IDS
    )
    code_bindings = contract["carry_forward_compatibility"]["source_code_bindings"]
    checks["source_code_bindings_unchanged"] = all(
        file_sha256(root / str(binding["path"])) == binding["sha256"] for binding in code_bindings
    )
    report_checks: list[bool] = []
    for row in parametric_rows:
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("path"), str):
            report_checks.append(False)
            continue
        report_path = root / str(evidence["path"])
        if not report_path.is_file() or file_sha256(report_path) != evidence.get("sha256"):
            report_checks.append(False)
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_checks.append(
            isinstance(report, Mapping)
            and report.get("report_sha256") == evidence.get("report_sha256")
            and report.get("report_sha256")
            == canonical_json_sha256(
                {key: value for key, value in report.items() if key != "report_sha256"}
            )
        )
    checks["all_parametric_report_evidence_valid"] = len(report_checks) == 5 and all(report_checks)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "registry_path": parent["registry"]["path"],
        "summary_path": parent["summary"]["path"],
        "parametric_rows": parametric_rows,
        "parent_summary": summary,
    }


def _structural_model_qualification_repair(
    main: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    sigma: Mapping[str, float],
    metrics: Sequence[str],
    aligned_features: Any,
    misspecified_features: Any,
    validation_groups: Sequence[tuple[int, int]],
    target_axis: str,
    candidate_id: str,
    effect_floor: float,
    noise_multiplier: float,
    minimum_disagreement_fraction: float,
) -> dict[str, Any]:
    """Evaluate the frozen low-potential design without requiring a validation baseline row."""
    aligned_models = {
        metric: structural._fit_model(main, metric, aligned_features) for metric in metrics
    }
    misspecified_models = {
        metric: structural._fit_model(main, metric, misspecified_features) for metric in metrics
    }
    baseline = (1, 1)
    for metric in metrics:
        aligned_baseline = structural._predict(
            aligned_models[metric], metric, baseline, aligned_features
        )
        misspecified_baseline = structural._predict(
            misspecified_models[metric], metric, baseline, misspecified_features
        )
        misspecified_models[metric]["baseline_offset"] = aligned_baseline - misspecified_baseline
    baseline_prediction_gap = max(
        abs(
            structural._predict(aligned_models[metric], metric, baseline, aligned_features)
            - structural._predict(
                misspecified_models[metric], metric, baseline, misspecified_features
            )
        )
        for metric in metrics
    )

    group_means = structural._validation_group_means(
        validation,
        metrics,
        groups=validation_groups,
    )
    comparisons: list[dict[str, Any]] = []
    aligned_errors: list[float] = []
    misspecified_errors: list[float] = []
    for (axis_a, axis_b), observed in group_means.items():
        for metric in metrics:
            aligned_prediction = structural._predict(
                aligned_models[metric], metric, (axis_a, axis_b), aligned_features
            )
            misspecified_prediction = structural._predict(
                misspecified_models[metric],
                metric,
                (axis_a, axis_b),
                misspecified_features,
            )
            gate = max(effect_floor, noise_multiplier * float(sigma[metric]))
            aligned_error = abs(aligned_prediction - float(observed[metric]))
            misspecified_error = abs(misspecified_prediction - float(observed[metric]))
            aligned_errors.append(aligned_error)
            misspecified_errors.append(misspecified_error)
            comparisons.append(
                {
                    "axis_a_index": axis_a,
                    "axis_b_index": axis_b,
                    "metric": metric,
                    "observed_validation_mean": float(observed[metric]),
                    "aligned_prediction": aligned_prediction,
                    "misspecified_prediction": misspecified_prediction,
                    "prediction_difference": abs(aligned_prediction - misspecified_prediction),
                    "disagreement_gate": gate,
                    "disagrees": abs(aligned_prediction - misspecified_prediction) >= gate,
                    "aligned_error": aligned_error,
                    "misspecified_error": misspecified_error,
                }
            )
    disagreement = [row for row in comparisons if row["disagrees"]]
    disagreement_fraction = len(disagreement) / len(comparisons)
    target_index = "axis_a_index" if target_axis == "a" else "axis_b_index"
    low_support = sum(row[target_index] == 0 for row in disagreement)
    high_support = sum(row[target_index] == 2 for row in disagreement)
    aligned_mae = float(np.mean(aligned_errors))
    misspecified_mae = float(np.mean(misspecified_errors))
    priors = structural.build_prior_arms(candidate_id)
    aligned_prior = priors["aligned_nominal"]
    misspecified_prior = priors["misindexed_nominal"]
    schema_matched = set(aligned_prior) == set(misspecified_prior)
    word_counts = {
        "aligned": len(str(aligned_prior["claim"]).split()),
        "misspecified": len(str(misspecified_prior["claim"]).split()),
    }
    checks = {
        "baseline_error_matched": baseline_prediction_gap <= 1.0e-12,
        "held_out_disagreement": (disagreement_fraction >= minimum_disagreement_fraction),
        "low_counterexample_region": low_support > 0,
        "high_counterexample_region": high_support > 0,
        "blind_identification": aligned_mae < misspecified_mae,
        "prior_schema_matched": schema_matched,
        "prior_word_count_matched": (
            abs(word_counts["aligned"] - word_counts["misspecified"]) <= 2
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_error_evaluation": "direct_prediction_match_at_frozen_center",
        "baseline_error_gap": baseline_prediction_gap,
        "comparison_count": len(comparisons),
        "disagreement_count": len(disagreement),
        "disagreement_fraction": disagreement_fraction,
        "low_counterexample_support": low_support,
        "high_counterexample_support": high_support,
        "aligned_validation_mae": aligned_mae,
        "misspecified_validation_mae": misspecified_mae,
        "blind_identified_aligned_model": aligned_mae < misspecified_mae,
        "prior_word_counts": word_counts,
        "comparisons": comparisons,
        "model_coefficients": {
            "aligned": aligned_models,
            "misspecified": misspecified_models,
        },
    }


def _is_transposition(permutation: Sequence[int]) -> bool:
    if len(permutation) != 4:
        return False
    moved = [index for index, source in enumerate(permutation) if index != source]
    return bool(
        len(moved) == 2 and permutation[moved[0]] == moved[1] and permutation[moved[1]] == moved[0]
    )


def _structural_groups(contract: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(int(value) for value in row)
        for row in contract["loci"]["structural"]["validation_groups"]
    )
