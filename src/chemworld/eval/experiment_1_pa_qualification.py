"""Frozen helpers for Experiment 1 PA qualification v1.0.1."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, variance
from typing import Any

import numpy as np

from chemworld.eval.experiment_1_contracts import load_contract_document
from chemworld.eval.experiment_1_ec_qualification import EXPECTED_COMMON_GATES
from chemworld.eval.experiment_1_ec_qualification import _validate_binding as _validate_binding
from chemworld.eval.provenance import canonical_json_sha256
from chemworld.materials import (
    STATIC_MATERIAL_INFORMATION_MISINDEXED,
    STATIC_MATERIAL_INFORMATION_NOMINAL,
    static_material_information_dossier,
)
from chemworld.world.phase_kernel import (
    INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1,
    partition_split,
)
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario

CONTRACT_VERSION = "chemworld-experiment-1-pa-qualification-contract-1.0.1"
REPAIR_CONTRACT_VERSION = "chemworld-experiment-1-pa-qualification-contract-1.0.2"
CONTRACT_VERSIONS = (CONTRACT_VERSION, REPAIR_CONTRACT_VERSION)
ENTITY_REPORT_VERSION = "chemworld-experiment-1-pa-entity-world-report-1.0.1"
PARAMETRIC_REPORT_VERSION = "chemworld-experiment-1-pa-parametric-world-report-1.0.1"
STRUCTURAL_REPORT_VERSION = "chemworld-experiment-1-pa-structural-world-report-1.0.1"
EXPECTED_WORLD_IDS = tuple(f"PA-W0{index}" for index in range(1, 6))
EXPECTED_WORLD_SEEDS = tuple(range(5))
FORBIDDEN_PUBLIC_TOKENS = (
    "aligned",
    "hidden mechanism",
    "hidden_state",
    "misindexed",
    "misspecified",
    "oracle",
    "world seed",
    "world_seed",
)


class Experiment1PAQualificationError(ValueError):
    """Raised when the frozen PA contract or evidence is malformed."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    value = load_contract_document(root, path)
    errors = validate_contract(root, value)
    if errors:
        raise Experiment1PAQualificationError("; ".join(errors))
    return value


def validate_contract(root: Path, contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") not in CONTRACT_VERSIONS:
        errors.append("contract schema version changed")
    if contract.get("status") != "development_frozen_before_execution":
        errors.append("contract is not frozen before execution")
    if contract.get("development_only") is not True:
        errors.append("contract must remain development-only")
    if contract.get("participant_provider_calls") != 0:
        errors.append("qualification must make zero participant provider calls")
    if contract.get("participant_execution_authorized") is not False:
        errors.append("participant execution must remain disabled")
    if contract.get("formal_benchmark_execution_authorized") is not False:
        errors.append("formal benchmark execution must remain disabled")
    _validate_binding(root, contract.get("specification"), "minimum specification", errors)
    _validate_binding(root, contract.get("system_specification"), "PA specification", errors)
    task = _mapping(contract.get("task"))
    if task.get("system_id") != "PA" or task.get("task_id") != "partition-discovery":
        errors.append("PA task binding changed")
    if task.get("scoring_contract_id") != "partition-s0-extraction-efficiency-v3":
        errors.append("PA must bind the corrected v3 two-phase scoring contract")
    _validate_binding(root, task.get("campaign_config"), "campaign config", errors)
    worlds = _mapping(contract.get("worlds"))
    rows = worlds.get("qualification")
    if not isinstance(rows, list) or len(rows) != 5:
        errors.append("PA must freeze five qualification Worlds")
    else:
        if tuple(str(row.get("world_id")) for row in rows) != EXPECTED_WORLD_IDS:
            errors.append("PA World IDs changed")
        if tuple(int(row.get("world_seed", -1)) for row in rows) != EXPECTED_WORLD_SEEDS:
            errors.append("PA World seeds changed")
        intervention_hashes = []
        for row in rows:
            interventions = row.get("world_interventions")
            if not isinstance(interventions, list):
                errors.append(f"{row.get('world_id')} interventions are malformed")
                continue
            intervention_hashes.append(canonical_json_sha256(interventions))
        if len(set(intervention_hashes)) != 5:
            errors.append("PA Worlds are not five distinct executable intervention manifests")
    if tuple(contract.get("common_gates", ())) != EXPECTED_COMMON_GATES:
        errors.append("Q1-Q8 registry changed")
    loci = _mapping(contract.get("loci"))
    if set(loci) != {"entity", "parametric", "structural"}:
        errors.append("PA prior loci changed")
    else:
        entity = _mapping(loci["entity"])
        if entity.get("descriptor_permutation") != [3, 1, 2, 0]:
            errors.append("PA entity permutation changed")
        if entity.get("solvent_anchors") != [0, 2]:
            errors.append("PA entity anchors changed")
        if entity.get("extractant_targets") != [0, 3]:
            errors.append("PA entity targets changed")
        parametric = _mapping(loci["parametric"])
        design = parametric.get("phase_design")
        if not isinstance(design, list) or len(design) != 5:
            errors.append("PA parametric phase design must contain five points")
        structural = _mapping(loci["structural"])
        if structural.get("pair_count") != 16 or structural.get("law_count") != 2:
            errors.append("PA structural denominator changed")
    execution = _mapping(contract.get("execution"))
    if execution.get("exact_replay_required") is not True:
        errors.append("exact replay must remain required")
    if execution.get("overwrite_forbidden") is not True:
        errors.append("write-once execution changed")
    return errors


def world_truth_audit(contract: Mapping[str, Any], world: Mapping[str, Any]) -> dict[str, Any]:
    interventions = tuple(dict(item) for item in world["world_interventions"])
    scenario = DefaultScenarioGenerator().generate(
        get_scenario("partition-discovery"),
        int(world["world_seed"]),
        interventions,
    )
    repeated = DefaultScenarioGenerator().generate(
        get_scenario("partition-discovery"),
        int(world["world_seed"]),
        interventions,
    )
    domain = dict(scenario.parameters.domain_parameters)
    truth = {
        "system_id": "PA",
        "world_id": str(world["world_id"]),
        "world_seed": int(world["world_seed"]),
        "task_id": "partition-discovery",
        "world_interventions": list(interventions),
        "world_family_intervention_hash": scenario.initial_state.metadata.get(
            "world_family_intervention_hash"
        ),
        "partition_coefficient_multiplier": float(domain["partition_coefficient_multiplier"]),
        "partition_coefficient_exponent": float(domain["partition_coefficient_exponent"]),
        "partition_phase_volume_multiplier": float(domain["partition_phase_volume_multiplier"]),
        "mechanism_hash": scenario.compiled_mechanism.mechanism_hash,
    }
    truth["truth_sha256"] = canonical_json_sha256(truth)
    truth["deterministic"] = bool(
        scenario.parameters.world_id == repeated.parameters.world_id
        and scenario.initial_state.metadata.get("world_family_intervention_hash")
        == repeated.initial_state.metadata.get("world_family_intervention_hash")
    )
    return truth


def entity_prior_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    permutation = list(contract["loci"]["entity"]["descriptor_permutation"])
    aligned = static_material_information_dossier(
        {"mode": STATIC_MATERIAL_INFORMATION_NOMINAL}, task_id="partition-discovery"
    )
    misspecified = static_material_information_dossier(
        {
            "mode": STATIC_MATERIAL_INFORMATION_MISINDEXED,
            "target_field": "extractant",
            "descriptor_permutation": permutation,
        },
        task_id="partition-discovery",
    )
    if not isinstance(aligned, Mapping) or not isinstance(misspecified, Mapping):
        raise Experiment1PAQualificationError("PA entity dossiers are unavailable")
    aligned_rows = list(aligned["choices"]["extractant"])
    misspecified_rows = list(misspecified["choices"]["extractant"])
    aligned_properties = [row["nominal_properties"] for row in aligned_rows]
    misspecified_properties = [row["nominal_properties"] for row in misspecified_rows]
    rendered = json.dumps(misspecified, sort_keys=True).lower()
    leakage = sorted(token for token in FORBIDDEN_PUBLIC_TOKENS if token in rendered)
    checks = {
        "aligned_and_misspecified_present": True,
        "schema_matched": _public_shape(aligned) == _public_shape(misspecified),
        "same_descriptor_multiset": sorted(map(canonical_json_sha256, aligned_properties))
        == sorted(map(canonical_json_sha256, misspecified_properties)),
        "single_target_transposition": permutation == [3, 1, 2, 0],
        "solvent_dossier_unchanged": aligned["choices"]["solvent"]
        == misspecified["choices"]["solvent"],
        "pair_table_withheld": "product_distribution_coefficients"
        not in json.dumps(aligned, sort_keys=True),
        "public_leakage_free": not leakage,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "leakage_tokens": leakage,
        "aligned": aligned,
        "misspecified": misspecified,
        "aligned_sha256": canonical_json_sha256(aligned),
        "misspecified_sha256": canonical_json_sha256(misspecified),
    }


def analyze_entity_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    locus = contract["loci"]["entity"]
    truth = world_truth_audit(contract, world)
    prior = entity_prior_audit(contract)
    expected = (
        len(locus["solvent_anchors"])
        * len(locus["extractant_targets"])
        * int(locus["independent_replicates"])
    )
    completed = [row for row in receipts if row.get("status") == "completed"]
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(int(row["solvent"]), int(row["extractant"]))].append(row)
    comparisons = []
    resolving = 0
    robust = 0
    expected_sign = _entity_expected_sign(prior["aligned"], tuple(locus["extractant_targets"]))
    for solvent in locus["solvent_anchors"]:
        left, right = (int(value) for value in locus["extractant_targets"])
        left_rows = grouped.get((int(solvent), left), [])
        right_rows = grouped.get((int(solvent), right), [])
        left_values = [float(row["measurement"]["product_in_organic"]) for row in left_rows]
        right_values = [float(row["measurement"]["product_in_organic"]) for row in right_rows]
        if len(left_values) != int(locus["independent_replicates"]) or len(right_values) != int(
            locus["independent_replicates"]
        ):
            comparisons.append({"solvent": solvent, "complete": False})
            continue
        signed_gap = fmean(right_values) - fmean(left_values)
        separation = abs(signed_gap)
        standard_error = math.sqrt(
            variance(left_values) / len(left_values) + variance(right_values) / len(right_values)
        )
        snr = separation / max(standard_error, 1.0e-12)
        mapping_consistent = signed_gap * expected_sign > 0.0
        passed = bool(mapping_consistent and separation >= float(locus["minimum_mean_separation"]))
        if passed:
            resolving += 1
        if passed and snr >= float(locus["minimum_signal_to_noise_ratio"]):
            robust += 1
        comparisons.append(
            {
                "solvent": solvent,
                "complete": True,
                "left_extractant": left,
                "right_extractant": right,
                "left_mean_product_in_organic": fmean(left_values),
                "right_mean_product_in_organic": fmean(right_values),
                "signed_gap": signed_gap,
                "absolute_gap": separation,
                "standard_error": standard_error,
                "signal_to_noise_ratio": snr,
                "aligned_mapping_consistent": mapping_consistent,
                "resolved": passed,
            }
        )
    denominators = _denominators(receipts, expected)
    common = _common_receipt_checks(receipts, expected)
    gates = {
        "Q1_world_integrity": bool(
            truth["deterministic"] and common["complete_denominator"] and common["all_exact_replay"]
        ),
        "Q2_task_accessibility": bool(common["all_measurements_public_and_finite"]),
        "Q3_public_contract_invariance": bool(
            common["public_contract_invariant"] and common["leakage_free"]
        ),
        "Q4_prior_symmetry": bool(prior["passed"]),
        "Q5_identifiability": resolving >= int(locus["minimum_resolving_anchors"]),
        "Q6_budgeted_falsifiability": bool(
            int(locus["participant_unique_experiment_budget"]) >= 4 and resolving >= 1
        ),
        "Q7_behavioral_relevance": any(
            row.get("absolute_gap", 0.0) >= float(locus["minimum_observed_endpoint_consequence"])
            for row in comparisons
        ),
        "Q8_noise_robustness": robust >= int(locus["minimum_robust_anchors"]),
    }
    return _world_report(
        schema_version=ENTITY_REPORT_VERSION,
        world=world,
        locus="entity",
        truth_sha256=str(truth["truth_sha256"]),
        denominators=denominators,
        gates=gates,
        extra={
            "truth_audit": truth,
            "prior_audit": {k: v for k, v in prior.items() if k not in {"aligned", "misspecified"}},
            "comparisons": comparisons,
        },
    )


def true_parametric_k_star(contract: Mapping[str, Any], world: Mapping[str, Any]) -> dict[str, Any]:
    truth = world_truth_audit(contract, world)
    locus = contract["loci"]["parametric"]
    observations = []
    for point in locus["phase_design"]:
        aqueous_total = float(point["solvent_volume_L"]) + float(point["aqueous_volume_L"])
        organic = float(point["extractant_volume_L"])
        split = partition_split(
            product_mol=1.0,
            impurity_mol=0.1,
            solvent=int(locus["reference_pair"]["solvent"]),
            extractant=int(locus["reference_pair"]["extractant"]),
            nominal_pair_contract=INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1,
            temperature_K=298.15,
            duration_s=float(locus["mix_duration_s"]),
            stirring_speed_rpm=float(locus["stirring_speed_rpm"]),
            organic_volume_L=organic,
            aqueous_volume_L=aqueous_total,
            coefficient_multiplier=float(truth["partition_coefficient_multiplier"]),
            coefficient_exponent=float(truth["partition_coefficient_exponent"]),
            phase_volume_multiplier=float(truth["partition_phase_volume_multiplier"]),
        )
        observations.append(
            {
                "organic_fraction": float(split["organic_product_mol"]),
                "aqueous_fraction": float(split["aqueous_product_mol"]),
                "organic_volume_L": organic,
                "aqueous_volume_L": aqueous_total,
            }
        )
    fitted = fit_effective_k(observations)
    return {
        "k_star": fitted["k_star"],
        "fit_rmse": fitted["rmse"],
        "truth_sha256": truth["truth_sha256"],
    }


def fit_effective_k(observations: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    log_grid = np.linspace(math.log(0.05), math.log(50.0), 6001)
    candidates = np.exp(log_grid)
    errors = np.zeros_like(candidates)
    valid = 0
    for row in observations:
        organic = float(row["organic_fraction"])
        aqueous = float(row["aqueous_fraction"])
        total = organic + aqueous
        if total <= 0.0:
            continue
        observed = organic / total
        v_org = float(row["organic_volume_L"])
        v_aq = float(row["aqueous_volume_L"])
        predicted = candidates * v_org / (candidates * v_org + v_aq)
        errors += (predicted - observed) ** 2
        valid += 1
    if valid == 0:
        raise Experiment1PAQualificationError("K* fit has no positive public allocations")
    index = int(np.argmin(errors))
    return {
        "k_star": float(candidates[index]),
        "rmse": float(math.sqrt(float(errors[index]) / valid)),
    }


def analyze_parametric_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    locus = contract["loci"]["parametric"]
    replicates = int(locus["independent_replicates"])
    expected = len(locus["phase_design"]) * replicates
    truth = world_truth_audit(contract, world)
    private = true_parametric_k_star(contract, world)
    completed = [row for row in receipts if row.get("status") == "completed"]
    observations = []
    by_point: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        by_point[str(row["point_id"])].append(row)
        observations.append(
            {
                "organic_fraction": float(row["measurement"]["product_in_organic"]),
                "aqueous_fraction": float(row["measurement"]["product_in_aqueous"]),
                "organic_volume_L": float(row["extractant_volume_L"]),
                "aqueous_volume_L": float(row["solvent_volume_L"]) + float(row["aqueous_volume_L"]),
            }
        )
    fit = fit_effective_k(observations) if observations else {"k_star": math.nan, "rmse": math.inf}
    true_k = float(private["k_star"])
    false_factor = float(locus["false_center_factors"][int(world["world_seed"])])
    false_k = true_k * false_factor
    relative_half_width = float(locus["relative_band_half_width"])
    aligned_band = [true_k * (1.0 - relative_half_width), true_k * (1.0 + relative_half_width)]
    false_band = [false_k * (1.0 - relative_half_width), false_k * (1.0 + relative_half_width)]
    fitted_k = float(fit["k_star"])
    aligned_contains_fit = aligned_band[0] <= fitted_k <= aligned_band[1]
    false_contains_fit = false_band[0] <= fitted_k <= false_band[1]
    point_reports = []
    counterexamples = 0
    robust_counterexamples = 0
    for point in locus["phase_design"]:
        rows = by_point.get(str(point["point_id"]), [])
        values = [
            float(row["measurement"]["product_in_organic"])
            / max(
                float(row["measurement"]["product_in_organic"])
                + float(row["measurement"]["product_in_aqueous"]),
                1.0e-12,
            )
            for row in rows
        ]
        v_org = float(point["extractant_volume_L"])
        v_aq = float(point["solvent_volume_L"]) + float(point["aqueous_volume_L"])
        aligned_prediction = true_k * v_org / (true_k * v_org + v_aq)
        false_prediction = false_k * v_org / (false_k * v_org + v_aq)
        consequence = abs(aligned_prediction - false_prediction)
        standard_error = math.sqrt(variance(values) / len(values)) if len(values) >= 2 else math.inf
        is_counterexample = consequence >= float(locus["minimum_prediction_gap"])
        is_robust = is_counterexample and consequence / max(standard_error, 1.0e-12) >= float(
            locus["minimum_signal_to_noise_ratio"]
        )
        counterexamples += int(is_counterexample)
        robust_counterexamples += int(is_robust)
        point_reports.append(
            {
                "point_id": point["point_id"],
                "replicate_count": len(values),
                "observed_mean_organic_allocation": fmean(values) if values else None,
                "standard_error": standard_error,
                "aligned_prediction": aligned_prediction,
                "misspecified_prediction": false_prediction,
                "prediction_gap": consequence,
                "counterexample": is_counterexample,
                "noise_robust_counterexample": is_robust,
            }
        )
    denominators = _denominators(receipts, expected)
    common = _common_receipt_checks(receipts, expected)
    participant_point_ids = locus.get("participant_point_ids")
    budget_point_reports = (
        [row for row in point_reports if row["point_id"] in set(participant_point_ids)]
        if isinstance(participant_point_ids, list)
        else []
    )
    budget_counterexamples = sum(bool(row["counterexample"]) for row in budget_point_reports)
    budget_robust_counterexamples = sum(
        bool(row["noise_robust_counterexample"]) for row in budget_point_reports
    )
    end_counterexamples = (
        sum(bool(row["counterexample"]) for row in (point_reports[0], point_reports[-1]))
        if len(point_reports) == 5
        else 0
    )
    if budget_point_reports:
        budgeted_falsifiability = bool(
            len(budget_point_reports) == int(locus["participant_unique_experiment_budget"])
            and budget_counterexamples >= int(locus.get("minimum_budget_counterexample_points", 2))
            and budget_robust_counterexamples
            >= int(locus.get("minimum_budget_robust_counterexample_points", 2))
        )
    else:
        budgeted_falsifiability = bool(
            int(locus["participant_unique_experiment_budget"]) >= 3
            and counterexamples >= int(locus["minimum_counterexample_points"])
            and end_counterexamples == 2
        )
    gates = {
        "Q1_world_integrity": bool(
            truth["deterministic"] and common["complete_denominator"] and common["all_exact_replay"]
        ),
        "Q2_task_accessibility": bool(common["all_measurements_public_and_finite"]),
        "Q3_public_contract_invariance": bool(
            common["public_contract_invariant"] and common["leakage_free"]
        ),
        "Q4_prior_symmetry": bool(relative_half_width == float(locus["relative_band_half_width"])),
        "Q5_identifiability": bool(aligned_contains_fit and not false_contains_fit),
        "Q6_budgeted_falsifiability": budgeted_falsifiability,
        "Q7_behavioral_relevance": counterexamples >= int(locus["minimum_counterexample_points"]),
        "Q8_noise_robustness": robust_counterexamples
        >= int(locus["minimum_robust_counterexample_points"]),
    }
    return _world_report(
        schema_version=PARAMETRIC_REPORT_VERSION,
        world=world,
        locus="parametric",
        truth_sha256=str(truth["truth_sha256"]),
        denominators=denominators,
        gates=gates,
        extra={
            "truth_audit": truth,
            "private_authoring_target": private,
            "aligned_prior": {
                "k_star_band": aligned_band,
                "relative_half_width": relative_half_width,
            },
            "misspecified_prior": {
                "k_star_band": false_band,
                "relative_half_width": relative_half_width,
            },
            "public_fit": fit,
            "aligned_contains_fit": aligned_contains_fit,
            "misspecified_contains_fit": false_contains_fit,
            "phase_point_reports": point_reports,
            "participant_budget_point_reports": budget_point_reports,
        },
    )


def structural_prior_arms() -> dict[str, Any]:
    def arm(family: str) -> dict[str, Any]:
        return {
            "statement": "The partition response follows the registered {family} family.",
            "family": family,
            "confidence": "bounded qualitative prior",
        }

    aligned = arm("linear_response")
    misspecified = arm("power_response")
    return {
        "aligned": aligned,
        "misspecified": misspecified,
        "schema_matched": _public_shape(aligned) == _public_shape(misspecified),
        "text_template_matched": aligned["statement"].replace("linear_response", "{family}")
        == misspecified["statement"].replace("power_response", "{family}"),
        "leakage_tokens": sorted(
            token
            for token in FORBIDDEN_PUBLIC_TOKENS
            if token in json.dumps([aligned, misspecified], sort_keys=True).lower()
        ),
    }


def analyze_structural_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    locus = contract["loci"]["structural"]
    expected = int(locus["pair_count"]) * int(locus["law_count"])
    truth = world_truth_audit(contract, world)
    prior = structural_prior_arms()
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in receipts:
        pairs[str(row["pair_id"])][str(row["law_id"])] = row
    metric_reports = {}
    support_by_metric: dict[str, int] = {}
    effect_gate = float(locus["minimum_paired_endpoint_gap"])
    for metric in ("product_in_organic", "product_in_aqueous"):
        pair_rows = []
        for pair_id, laws in sorted(pairs.items()):
            if set(laws) != {"linear_response", "power_response"}:
                continue
            left = laws["linear_response"]
            right = laws["power_response"]
            if left.get("status") != "completed" or right.get("status") != "completed":
                continue
            gap = float(right["measurement"][metric]) - float(left["measurement"][metric])
            pair_rows.append({"pair_id": pair_id, "signed_gap": gap, "absolute_gap": abs(gap)})
        supporting = sum(row["absolute_gap"] >= effect_gate for row in pair_rows)
        support_by_metric[metric] = supporting
        metric_reports[metric] = {
            "effect_gate": effect_gate,
            "supporting_pair_count": supporting,
            "maximum_absolute_gap": max((row["absolute_gap"] for row in pair_rows), default=0.0),
            "pairs": pair_rows,
        }
    slope_report = _structural_slope_report(pairs, float(locus["minimum_slope_deviation"]))
    budget_pairs = {str(value) for value in locus["participant_pair_ids"]}
    budget_support = sum(
        any(
            row["pair_id"] in budget_pairs and row["absolute_gap"] >= effect_gate
            for row in metric_reports[metric]["pairs"]
        )
        for metric in metric_reports
    )
    denominators = _denominators(receipts, expected)
    common = _common_receipt_checks(receipts, expected)
    paired_actions = all(
        len(laws) == 2
        and len({row.get("action_plan_sha256") for row in laws.values()}) == 1
        and len({row.get("observation_seed") for row in laws.values()}) == 1
        for laws in pairs.values()
    )
    structural_binding = all(row.get("law_binding_verified") is True for row in receipts)
    resolving_metrics = sum(
        count >= int(locus["minimum_supporting_pair_count_per_metric"])
        for count in support_by_metric.values()
    )
    gates = {
        "Q1_world_integrity": bool(
            truth["deterministic"]
            and common["complete_denominator"]
            and common["all_exact_replay"]
            and structural_binding
        ),
        "Q2_task_accessibility": bool(common["all_measurements_public_and_finite"]),
        "Q3_public_contract_invariance": bool(
            common["public_contract_invariant"] and common["leakage_free"]
        ),
        "Q4_prior_symmetry": bool(
            prior["schema_matched"]
            and prior["text_template_matched"]
            and not prior["leakage_tokens"]
        ),
        "Q5_identifiability": bool(
            resolving_metrics >= int(locus["minimum_resolving_metric_count"])
            and slope_report["slope_signature_passed"]
        ),
        "Q6_budgeted_falsifiability": bool(
            len(budget_pairs) <= int(locus["participant_unique_experiment_budget"])
            and budget_support >= int(locus["minimum_budget_supporting_metric_count"])
        ),
        "Q7_behavioral_relevance": resolving_metrics
        >= int(locus["minimum_resolving_metric_count"]),
        "Q8_noise_robustness": bool(
            common["all_exact_replay"] and paired_actions and slope_report["slope_signature_passed"]
        ),
    }
    return _world_report(
        schema_version=STRUCTURAL_REPORT_VERSION,
        world=world,
        locus="structural",
        truth_sha256=str(truth["truth_sha256"]),
        denominators=denominators,
        gates=gates,
        extra={
            "truth_audit": truth,
            "prior_arms": prior,
            "metric_reports": metric_reports,
            "public_log_ratio_slope": slope_report,
            "paired_actions_and_noise": paired_actions,
            "law_binding_verified": structural_binding,
            "budget_supporting_metric_count": budget_support,
        },
    )


def _structural_slope_report(
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]], minimum_deviation: float
) -> dict[str, Any]:
    baseline = []
    power = []
    for laws in pairs.values():
        if set(laws) != {"linear_response", "power_response"}:
            continue
        values = []
        for law_id in ("linear_response", "power_response"):
            measurement = laws[law_id].get("measurement")
            if not isinstance(measurement, Mapping):
                values = []
                break
            organic = float(measurement["product_in_organic"])
            aqueous = float(measurement["product_in_aqueous"])
            if min(organic, aqueous) <= 0.0:
                values = []
                break
            values.append(math.log(organic / aqueous))
        if values:
            baseline.append(values[0])
            power.append(values[1])
    if len(baseline) < 4 or float(np.var(baseline)) <= 0.0:
        return {
            "pair_count": len(baseline),
            "slope": None,
            "absolute_slope_deviation_from_one": 0.0,
            "minimum_deviation": minimum_deviation,
            "slope_signature_passed": False,
        }
    slope = float(np.cov(baseline, power, ddof=0)[0, 1] / np.var(baseline))
    deviation = abs(slope - 1.0)
    return {
        "pair_count": len(baseline),
        "slope": slope,
        "absolute_slope_deviation_from_one": deviation,
        "minimum_deviation": minimum_deviation,
        "slope_signature_passed": deviation >= minimum_deviation,
    }


def _common_receipt_checks(receipts: Sequence[Mapping[str, Any]], expected: int) -> dict[str, bool]:
    completed = [row for row in receipts if row.get("status") == "completed"]
    contract_hashes = {row.get("task_contract_hash") for row in completed}
    return {
        "complete_denominator": len(receipts) == expected
        and len(completed) == expected
        and not any(row.get("status") == "platform_failure" for row in receipts),
        "all_exact_replay": len(receipts) == expected
        and all(row.get("exact_replay") is True for row in receipts),
        "all_measurements_public_and_finite": len(completed) == expected
        and all(
            isinstance(row.get("measurement"), Mapping)
            and all(
                row.get("observed_mask", {}).get(metric) is True
                and math.isfinite(float(row["measurement"][metric]))
                for metric in ("product_in_organic", "product_in_aqueous", "phase_ratio")
            )
            for row in completed
        ),
        "public_contract_invariant": len(contract_hashes) == 1 and None not in contract_hashes,
        "leakage_free": not any(row.get("participant_visible_leakage_matches") for row in receipts),
    }


def _denominators(receipts: Sequence[Mapping[str, Any]], expected: int) -> dict[str, int]:
    return {
        "planned": expected,
        "attempted": len(receipts),
        "completed": sum(row.get("status") == "completed" for row in receipts),
        "classified": sum(
            row.get("status") in {"completed", "physical_failure", "platform_failure"}
            for row in receipts
        ),
        "exact_replay": sum(row.get("exact_replay") is True for row in receipts),
        "physical_failures": sum(row.get("status") == "physical_failure" for row in receipts),
        "platform_failures": sum(row.get("status") == "platform_failure" for row in receipts),
    }


def _world_report(
    *,
    schema_version: str,
    world: Mapping[str, Any],
    locus: str,
    truth_sha256: str,
    denominators: Mapping[str, Any],
    gates: Mapping[str, bool],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    failures = [gate for gate in EXPECTED_COMMON_GATES if not gates[gate]]
    report: dict[str, Any] = {
        "schema_version": schema_version,
        "formal_result": False,
        "provider_call_count": 0,
        "system_id": "PA",
        "world_id": str(world["world_id"]),
        "world_seed": int(world["world_seed"]),
        "prior_locus": locus,
        "truth_sha256": truth_sha256,
        "denominators": dict(denominators),
        "gates": dict(gates),
        "failures": failures,
        "status": "qualified" if not failures else "failed",
        **dict(extra),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def _entity_expected_sign(dossier: Mapping[str, Any], targets: tuple[int, int]) -> float:
    choices = dossier["choices"]["extractant"]
    values = {
        int(row["action_value"]): float(
            row["nominal_properties"]["partner_panel_product_distribution_geomean"]
        )
        for row in choices
    }
    difference = values[targets[1]] - values[targets[0]]
    if difference == 0.0:
        raise Experiment1PAQualificationError("entity target dossiers are tied")
    return math.copysign(1.0, difference)


def _public_shape(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _public_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_shape(item) for item in value]
    return type(value).__name__


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
