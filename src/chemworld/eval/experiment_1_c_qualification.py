"""Frozen helpers for Experiment 1 crystallization qualification v1.0.1."""

from __future__ import annotations

import itertools
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
from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_structural_candidate_qualification import (
    analyze_candidate_world,
    build_prior_arms,
)
from chemworld.materials import static_material_information_dossier
from chemworld.world.crystallization_material_family import (
    REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
    apply_crystallization_material_family,
    crystallization_material_family,
)
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario

CONTRACT_VERSION = "chemworld-experiment-1-c-qualification-contract-1.0.1"
REPAIR_CONTRACT_VERSION = "chemworld-experiment-1-c-qualification-contract-1.0.2"
CONTRACT_VERSIONS = (CONTRACT_VERSION, REPAIR_CONTRACT_VERSION)
ENTITY_REPORT_VERSION = "chemworld-experiment-1-c-entity-world-report-1.0.1"
PARAMETRIC_REPORT_VERSION = "chemworld-experiment-1-c-parametric-world-report-1.0.1"
STRUCTURAL_REPORT_VERSION = "chemworld-experiment-1-c-structural-world-report-1.0.1"
EXPECTED_WORLD_IDS = tuple(f"C-W0{index}" for index in range(1, 6))
EXPECTED_WORLD_SEEDS = tuple(range(5))
EXPECTED_ENTITY_PERMUTATION = (0, 3, 2, 1)
FORBIDDEN_PUBLIC_TOKENS = (
    "aligned",
    "hidden_state",
    "misindexed",
    "misspecified",
    "oracle",
    "world seed",
    "world_seed",
)


class Experiment1CQualificationError(ValueError):
    """Raised when the frozen C contract or evidence is malformed."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    value = load_contract_document(root, path)
    errors = validate_contract(root, value)
    if errors:
        raise Experiment1CQualificationError("; ".join(errors))
    return value


def validate_contract(root: Path, contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_flags = {
        "status": "development_frozen_before_execution",
        "development_only": True,
        "participant_provider_calls": 0,
        "participant_execution_authorized": False,
        "formal_benchmark_execution_authorized": False,
    }
    if contract.get("schema_version") not in CONTRACT_VERSIONS:
        errors.append("contract.schema_version changed")
    for key, expected in expected_flags.items():
        if contract.get(key) != expected:
            errors.append(f"contract.{key} changed")
    _validate_binding(root, contract.get("specification"), "minimum specification", errors)
    _validate_binding(root, contract.get("system_specification"), "C specification", errors)
    task = _mapping(contract.get("task"))
    expected_task = {
        "system_id": "C",
        "task_id": "reaction-to-crystallization",
        "world_split": "public-test",
        "objective": "balanced",
        "truth_family": "seed_mediated_population_balance_parent",
        "crystallization_material_family_id": REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            errors.append(f"task.{key} changed")
    _validate_binding(root, task.get("campaign_config"), "campaign config", errors)
    worlds = _mapping(contract.get("worlds"))
    rows = worlds.get("qualification")
    if not isinstance(rows, list) or len(rows) != 5:
        errors.append("C must freeze five qualification Worlds")
    else:
        if tuple(str(row.get("world_id")) for row in rows) != EXPECTED_WORLD_IDS:
            errors.append("C World IDs changed")
        if tuple(int(row.get("world_seed", -1)) for row in rows) != EXPECTED_WORLD_SEEDS:
            errors.append("C World seeds changed")
        manifests = [canonical_json_sha256(row.get("world_interventions", [])) for row in rows]
        if len(set(manifests)) != 5:
            errors.append("C Worlds are not five distinct intervention manifests")
    canary = _mapping(worlds.get("canary"))
    if (
        canary.get("world_id") != "C-W00"
        or canary.get("world_seed") != 900004
        or canary.get("formal_denominator") is not False
    ):
        errors.append("C-W00 canary changed")
    if tuple(contract.get("common_gates", ())) != EXPECTED_COMMON_GATES:
        errors.append("Q1-Q8 gate registry changed")
    loci = _mapping(contract.get("loci"))
    if set(loci) != {"entity", "parametric", "structural"}:
        errors.append("C prior loci changed")
    else:
        entity = _mapping(loci["entity"])
        if (
            tuple(entity.get("descriptor_permutation", ())) != EXPECTED_ENTITY_PERMUTATION
            or entity.get("solvent_targets") != [1, 3]
            or entity.get("cooling_anchors_K") != [290.0, 270.0]
            or entity.get("independent_replicates") != 3
        ):
            errors.append("C entity design changed")
        parametric = _mapping(loci["parametric"])
        if (
            parametric.get("temperature_levels_K") != [310.0, 290.0, 270.0]
            or parametric.get("independent_replicates") != 3
        ):
            errors.append("C parametric design changed")
        structural = _mapping(loci["structural"])
        if (
            structural.get("candidate_id") != "crystallization_nucleation_growth"
            or structural.get("main_grid_cells") != 9
            or structural.get("validation_groups") != 3
            or structural.get("validation_replicates") != 3
        ):
            errors.append("C structural design changed")
    execution = _mapping(contract.get("execution"))
    if (
        execution.get("exact_replay_required") is not True
        or execution.get("overwrite_forbidden") is not True
        or execution.get("post_failure_redesign_in_same_campaign_forbidden") is not True
    ):
        errors.append("C execution policy changed")
    return errors


def world_truth_audit(contract: Mapping[str, Any], world: Mapping[str, Any]) -> dict[str, Any]:
    interventions = tuple(dict(item) for item in world["world_interventions"])
    generator = DefaultScenarioGenerator()
    scenario = apply_crystallization_material_family(
        generator.generate(
            get_scenario("reaction-to-crystallization"),
            int(world["world_seed"]),
            interventions,
        ),
        REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
    )
    repeated = apply_crystallization_material_family(
        generator.generate(
            get_scenario("reaction-to-crystallization"),
            int(world["world_seed"]),
            interventions,
        ),
        REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
    )
    parameters = scenario.parameters
    family = crystallization_material_family(REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY)
    realized = []
    nominal = []
    for index, row in enumerate(family.solvent_profiles):
        realized.append(
            np.concatenate(
                [
                    np.asarray(parameters.crystallization_solvent_effects[index], dtype=float),
                    np.asarray(
                        [
                            parameters.crystallization_solvent_solubility_multipliers[index],
                            parameters.crystallization_solvent_nucleation_multipliers[index],
                            parameters.crystallization_solvent_growth_multipliers[index],
                            parameters.crystallization_solvent_occlusion_multipliers[index],
                        ],
                        dtype=float,
                    ),
                ]
            )
        )
        nominal.append(
            np.asarray(
                [
                    *row["reaction_multipliers"],
                    row["solubility_multiplier"],
                    row["nucleation_multiplier"],
                    row["growth_multiplier"],
                    row["impurity_occlusion_multiplier"],
                ],
                dtype=float,
            )
        )
    nominal_array = np.asarray(nominal)
    scale = np.std(nominal_array, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    mapping_rows = []
    for action_index, swapped_index in ((1, 3), (3, 1)):
        own_distance = float(
            np.linalg.norm((realized[action_index] - nominal_array[action_index]) / scale)
        )
        swapped_distance = float(
            np.linalg.norm((realized[action_index] - nominal_array[swapped_index]) / scale)
        )
        mapping_rows.append(
            {
                "action_index": action_index,
                "own_nominal_distance": own_distance,
                "swapped_nominal_distance": swapped_distance,
                "own_mapping_closer": own_distance < swapped_distance,
            }
        )
    domain = parameters.domain_parameters
    solubility_multiplier = float(domain["crystallization_solubility_multiplier"])
    threshold_center = min(max(290.0 + 20.0 * math.log(solubility_multiplier), 275.0), 305.0)
    payload = {
        "system_id": "C",
        "world_id": str(world["world_id"]),
        "world_seed": int(world["world_seed"]),
        "task_id": "reaction-to-crystallization",
        "world_interventions": list(interventions),
        "runtime_world_id": parameters.world_id,
        "mechanism_hash": scenario.compiled_mechanism.mechanism_hash,
        "world_family_intervention_hash": scenario.initial_state.metadata.get(
            "world_family_intervention_hash"
        ),
        "crystallization_material_instance_sha256": scenario.initial_state.metadata.get(
            "crystallization_material_instance_sha256"
        ),
        "crystallization_nucleation_multiplier": float(
            domain["crystallization_nucleation_multiplier"]
        ),
        "crystallization_solubility_multiplier": solubility_multiplier,
        "aligned_threshold_center_K": threshold_center,
        "mapping_rows": mapping_rows,
    }
    payload["truth_sha256"] = canonical_json_sha256(payload)
    payload["deterministic"] = bool(
        scenario.parameters.world_id == repeated.parameters.world_id
        and scenario.initial_state.metadata.get("crystallization_material_instance_sha256")
        == repeated.initial_state.metadata.get("crystallization_material_instance_sha256")
    )
    payload["aligned_mapping_not_reversed"] = all(row["own_mapping_closer"] for row in mapping_rows)
    return payload


def entity_prior_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    locus = contract["loci"]["entity"]
    family_id = contract["task"]["crystallization_material_family_id"]
    aligned = static_material_information_dossier(
        {"mode": "anonymous_nominal_properties"},
        task_id="reaction-to-crystallization",
        material_family_id=family_id,
    )
    misspecified = static_material_information_dossier(
        {
            "mode": "anonymous_misindexed_properties",
            "target_field": "solvent",
            "descriptor_permutation": list(locus["descriptor_permutation"]),
        },
        task_id="reaction-to-crystallization",
        material_family_id=family_id,
    )
    joined = json.dumps([aligned, misspecified], ensure_ascii=False, sort_keys=True).lower()
    leakage = sorted(token for token in FORBIDDEN_PUBLIC_TOKENS if token in joined)
    checks = {
        "aligned_and_misspecified_present": aligned is not None and misspecified is not None,
        "schema_matched": _public_shape(aligned) == _public_shape(misspecified),
        "text_template_matched": _public_text(aligned) == _public_text(misspecified),
        "solvent_only_transposition": tuple(locus["descriptor_permutation"])
        == EXPECTED_ENTITY_PERMUTATION,
        "public_leakage_free": not leakage,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "leakage_tokens": leakage,
        "aligned_sha256": canonical_json_sha256(aligned),
        "misspecified_sha256": canonical_json_sha256(misspecified),
        "aligned": aligned,
        "misspecified": misspecified,
    }


def parametric_prior_arms(contract: Mapping[str, Any], world: Mapping[str, Any]) -> dict[str, Any]:
    locus = contract["loci"]["parametric"]
    if locus.get("question_id") == "cooling_response_gain_310_to_270_K":
        center = float(locus["aligned_effect_centers_by_world"][str(world["world_id"])])
        false_center = center * float(locus["misspecified_center_factor"])
        half_width = float(locus["relative_band_half_width"])
        common = {
            "target": "cooling_response_gain_310_to_270_K",
            "metric": locus["threshold_metric"],
            "seed_mass_g": locus["seed_mass_g"],
            "scope": "fixed upstream composition and solvent context",
        }
        arms = {
            "aligned": {
                **common,
                "effect_band": [center * (1.0 - half_width), center * (1.0 + half_width)],
            },
            "misspecified": {
                **common,
                "effect_band": [
                    false_center * (1.0 - half_width),
                    false_center * (1.0 + half_width),
                ],
            },
            "opaque": {**common, "effect_band": ["withheld", "withheld"]},
        }
        return _prior_arm_audit(arms)
    center = float(world_truth_audit(contract, world)["aligned_threshold_center_K"])
    half_width = float(locus["aligned_band_half_width_K"])
    shift = float(locus["misspecified_shift_K"])
    common = {
        "target": "effective_cooling_threshold_band",
        "metric": locus["threshold_metric"],
        "seed_mass_g": locus["seed_mass_g"],
        "scope": "fixed upstream composition and solvent context",
    }
    arms = {
        "aligned": {**common, "band_K": [center - half_width, center + half_width]},
        "misspecified": {
            **common,
            "band_K": [center + shift - half_width, center + shift + half_width],
        },
        "opaque": {**common, "band_K": ["withheld", "withheld"]},
    }
    return _prior_arm_audit(arms)


def structural_prior_audit() -> dict[str, Any]:
    arms = build_prior_arms("crystallization_nucleation_growth")
    values = list(arms.values())
    checks = {
        "three_arms_present": set(arms) == {"opaque", "aligned_nominal", "misindexed_nominal"},
        "same_public_keys": len({_public_shape(value) for value in values}) == 1,
        "target_is_seed_mediated": all(
            value.get("target") == "seed_mediated_nucleation_growth" for value in values
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "arm_sha256": {key: canonical_json_sha256(value) for key, value in arms.items()},
    }


def analyze_entity_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    locus = contract["loci"]["entity"]
    expected = (
        len(locus["cooling_anchors_K"])
        * len(locus["solvent_targets"])
        * int(locus["independent_replicates"])
    )
    completed = _completed(receipts)
    grouped: dict[tuple[float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(float(row["temperature_K"]), int(row["solvent"]))].append(row)
    anchors = []
    for temperature in locus["cooling_anchors_K"]:
        left = grouped[(float(temperature), 1)]
        right = grouped[(float(temperature), 3)]
        metric_rows = _group_metric_contrasts(left, right, locus["direct_metrics"])
        complete = len(left) == len(right) == int(locus["independent_replicates"])
        mean_gap = fmean(row["absolute_separation"] for row in metric_rows)
        max_gap = max(row["absolute_separation"] for row in metric_rows)
        rms_se = math.sqrt(fmean(float(row["standard_error"] or 0.0) ** 2 for row in metric_rows))
        snr = mean_gap / max(rms_se, 1.0e-12)
        anchors.append(
            {
                "temperature_K": temperature,
                "metric_results": metric_rows,
                "mean_support_separation": mean_gap,
                "maximum_support_separation": max_gap,
                "support_signal_to_noise_ratio": snr,
                "passed": bool(
                    complete
                    and mean_gap >= float(locus["minimum_mean_support_separation"])
                    and max_gap >= float(locus["minimum_single_support_separation"])
                    and snr >= float(locus["minimum_support_signal_to_noise_ratio"])
                ),
            }
        )
    prior = entity_prior_audit(contract)
    truth = world_truth_audit(contract, world)
    gates = {
        "Q1_world_integrity": _integrity(receipts, expected) and truth["deterministic"],
        "Q2_task_accessibility": len({row.get("action_plan_sha256") for row in completed}) == 4,
        "Q3_public_contract_invariance": _public_receipts_ok(completed, locus["direct_metrics"]),
        "Q4_prior_symmetry": prior["passed"],
        "Q5_identifiability": all(anchor["passed"] for anchor in anchors),
        "Q6_budgeted_falsifiability": int(locus["participant_unique_experiment_budget"]) == 4,
        "Q7_behavioral_relevance": bool(
            truth["aligned_mapping_not_reversed"]
            and max(anchor["mean_support_separation"] for anchor in anchors)
            >= float(locus["minimum_observed_endpoint_consequence"])
        ),
        "Q8_noise_robustness": all(
            anchor["support_signal_to_noise_ratio"]
            >= float(locus["minimum_support_signal_to_noise_ratio"])
            for anchor in anchors
        ),
    }
    return _world_report(
        ENTITY_REPORT_VERSION,
        world,
        "entity",
        receipts,
        expected,
        gates,
        {
            "private_world_audit": truth,
            "prior_audit": {k: v for k, v in prior.items() if k not in {"aligned", "misspecified"}},
            "anchor_results": anchors,
        },
    )


def analyze_parametric_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    locus = contract["loci"]["parametric"]
    expected = len(locus["temperature_levels_K"]) * int(locus["independent_replicates"])
    completed = _completed(receipts)
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[float(row["temperature_K"])].append(row)
    metric = str(locus["threshold_metric"])
    level_rows = []
    for temperature in locus["temperature_levels_K"]:
        values = [float(row["metrics"][metric]) for row in grouped[float(temperature)]]
        level_rows.append(
            {
                "temperature_K": float(temperature),
                "mean": fmean(values) if values else None,
                "standard_error": (
                    math.sqrt(variance(values) / len(values)) if len(values) > 1 else None
                ),
            }
        )
    endpoint_effect = abs(
        float(level_rows[-1]["mean"] or 0.0) - float(level_rows[0]["mean"] or 0.0)
    )
    endpoint_se = math.sqrt(
        float(level_rows[-1]["standard_error"] or 0.0) ** 2
        + float(level_rows[0]["standard_error"] or 0.0) ** 2
    )
    endpoint_snr = endpoint_effect / max(endpoint_se, 1.0e-12)
    crossing = any(
        (float(left["mean"] or 0.0) - float(locus["crossing_level"]))
        * (float(right["mean"] or 0.0) - float(locus["crossing_level"]))
        <= 0.0
        for left, right in itertools.pairwise(level_rows)
    )
    prior = parametric_prior_arms(contract, world)
    truth = world_truth_audit(contract, world)
    fixed_context_hashes = {row.get("fixed_context_sha256") for row in completed}
    response_gain_question = locus.get("question_id") == "cooling_response_gain_310_to_270_K"
    aligned_band: list[float] | None = None
    misspecified_band: list[float] | None = None
    if response_gain_question:
        aligned_center = float(locus["aligned_effect_centers_by_world"][str(world["world_id"])])
        false_center = aligned_center * float(locus["misspecified_center_factor"])
        relative_half_width = float(locus["relative_band_half_width"])
        aligned_band = [
            aligned_center * (1.0 - relative_half_width),
            aligned_center * (1.0 + relative_half_width),
        ]
        misspecified_band = [
            false_center * (1.0 - relative_half_width),
            false_center * (1.0 + relative_half_width),
        ]
        identifiability = bool(
            aligned_band[0] <= endpoint_effect <= aligned_band[1]
            and not misspecified_band[0] <= endpoint_effect <= misspecified_band[1]
            and abs(aligned_center - false_center) >= float(locus["minimum_prior_center_gap"])
        )
    else:
        identifiability = bool(
            crossing and endpoint_effect >= float(locus["minimum_endpoint_effect"])
        )
    gates = {
        "Q1_world_integrity": _integrity(receipts, expected) and truth["deterministic"],
        "Q2_task_accessibility": len({row.get("action_plan_sha256") for row in completed}) == 3,
        "Q3_public_contract_invariance": _public_receipts_ok(completed, locus["direct_metrics"]),
        "Q4_prior_symmetry": prior["passed"],
        "Q5_identifiability": identifiability,
        "Q6_budgeted_falsifiability": int(locus["participant_unique_experiment_budget"]) == 3,
        "Q7_behavioral_relevance": bool(endpoint_effect >= float(locus["minimum_endpoint_effect"])),
        "Q8_noise_robustness": bool(
            len(fixed_context_hashes) == 1
            and endpoint_snr >= float(locus["minimum_signal_to_noise_ratio"])
        ),
    }
    return _world_report(
        PARAMETRIC_REPORT_VERSION,
        world,
        "parametric",
        receipts,
        expected,
        gates,
        {
            "private_world_audit": truth,
            "prior_audit": prior,
            "temperature_results": level_rows,
            "endpoint_effect": endpoint_effect,
            "endpoint_signal_to_noise_ratio": endpoint_snr,
            "crossing_detected": crossing,
            "aligned_effect_band": aligned_band,
            "misspecified_effect_band": misspecified_band,
        },
    )


def analyze_structural_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = 18
    analysis = analyze_candidate_world("crystallization_nucleation_growth", receipts)
    checks = analysis["checks"]
    prior = structural_prior_audit()
    gates = {
        "Q1_world_integrity": bool(
            len(receipts) == expected
            and checks["all_outcomes_classified"]
            and checks["zero_platform_failures"]
            and checks["all_exact_replay"]
        ),
        "Q2_task_accessibility": bool(
            checks["fixed_query_count"] and checks["main_grid_count"] and checks["validation_count"]
        ),
        "Q3_public_contract_invariance": bool(
            checks["complete_main_surface"] and checks["complete_validation_surface"]
        ),
        "Q4_prior_symmetry": bool(
            prior["passed"]
            and checks.get("prior_schema_matched")
            and checks.get("prior_word_count_matched")
        ),
        "Q5_identifiability": bool(
            checks.get("axis_a_effect")
            and checks.get("axis_b_effect")
            and checks.get("topology_signature")
        ),
        "Q6_budgeted_falsifiability": int(
            contract["loci"]["structural"]["participant_unique_experiment_budget"]
        )
        == 4,
        "Q7_behavioral_relevance": bool(
            checks.get("held_out_disagreement")
            and checks.get("low_counterexample_region")
            and checks.get("high_counterexample_region")
            and checks.get("blind_identification")
        ),
        "Q8_noise_robustness": bool(
            checks.get("baseline_error_matched") and checks["complete_validation_surface"]
        ),
    }
    return _world_report(
        STRUCTURAL_REPORT_VERSION,
        world,
        "structural",
        receipts,
        expected,
        gates,
        {"prior_audit": prior, "structural_candidate_analysis": analysis},
    )


def _integrity(receipts: Sequence[Mapping[str, Any]], expected: int) -> bool:
    return bool(
        len(receipts) == expected
        and all(row.get("status") == "completed" for row in receipts)
        and all(row.get("exact_replay") is True for row in receipts)
        and all(row.get("truth_binding_verified") is True for row in receipts)
    )


def _completed(receipts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in receipts if row.get("status") == "completed"]


def _public_receipts_ok(receipts: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> bool:
    return bool(
        receipts
        and all(not row.get("participant_visible_leakage_matches") for row in receipts)
        and all(
            isinstance(row.get("metrics"), Mapping)
            and all(metric in row["metrics"] for metric in metrics)
            for row in receipts
        )
    )


def _group_metric_contrasts(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    result = []
    for metric in metrics:
        left_values = [float(row["metrics"][metric]) for row in left]
        right_values = [float(row["metrics"][metric]) for row in right]
        complete = len(left_values) > 1 and len(right_values) > 1
        standard_error = (
            math.sqrt(
                variance(left_values) / len(left_values)
                + variance(right_values) / len(right_values)
            )
            if complete
            else None
        )
        result.append(
            {
                "metric": metric,
                "left_mean": fmean(left_values) if left_values else None,
                "right_mean": fmean(right_values) if right_values else None,
                "absolute_separation": (
                    abs(fmean(right_values) - fmean(left_values))
                    if left_values and right_values
                    else 0.0
                ),
                "standard_error": standard_error,
            }
        )
    return result


def _prior_arm_audit(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = list(arms.values())
    rendered = json.dumps(values, ensure_ascii=False, sort_keys=True).lower()
    leakage = sorted(token for token in FORBIDDEN_PUBLIC_TOKENS if token in rendered)
    checks = {
        "three_arms_present": set(arms) == {"aligned", "misspecified", "opaque"},
        "schema_matched": len({_public_shape(value) for value in values}) == 1,
        "public_leakage_free": not leakage,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "leakage_tokens": leakage,
        "arm_sha256": {key: canonical_json_sha256(value) for key, value in arms.items()},
    }


def _world_report(
    schema_version: str,
    world: Mapping[str, Any],
    locus: str,
    receipts: Sequence[Mapping[str, Any]],
    expected: int,
    gates: Mapping[str, bool],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    failures = sorted(key for key, value in gates.items() if not value)
    report: dict[str, Any] = {
        "schema_version": schema_version,
        "formal_result": False,
        "provider_call_count": 0,
        "system_id": "C",
        "task_id": "reaction-to-crystallization",
        "world_id": str(world["world_id"]),
        "world_seed": int(world["world_seed"]),
        "prior_locus": locus,
        "truth_sha256": world_truth_audit({}, world)["truth_sha256"],
        "status": "qualified" if not failures else "failed",
        "failures": failures,
        "gates": dict(gates),
        "denominators": {
            "planned": expected,
            "attempted": len(receipts),
            "completed": sum(row.get("status") == "completed" for row in receipts),
            "exact_replay": sum(row.get("exact_replay") is True for row in receipts),
            "physical_failures": sum(row.get("status") == "physical_failure" for row in receipts),
            "platform_failures": sum(row.get("status") == "platform_failure" for row in receipts),
        },
        **dict(extra),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def _validate_binding(root: Path, payload: object, label: str, errors: list[str]) -> None:
    if not isinstance(payload, Mapping):
        errors.append(f"{label} binding is missing")
        return
    path = root / str(payload.get("path", ""))
    if not path.is_file():
        errors.append(f"{label} path is missing")
    elif payload.get("sha256") != file_sha256(path):
        errors.append(f"{label} hash changed")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _public_shape(value: Any) -> str:
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(f"{key}:{_public_shape(item)}" for key, item in sorted(value.items()))
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_public_shape(item) for item in value) + "]"
    return "scalar"


def _public_text(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            item
            for key, child in sorted(value.items())
            if key != "nominal_properties"
            for item in _public_text(child)
        ]
    if isinstance(value, list):
        return [item for child in value for item in _public_text(child)]
    return [value] if isinstance(value, str) else []


__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_VERSIONS",
    "EXPECTED_WORLD_IDS",
    "REPAIR_CONTRACT_VERSION",
    "Experiment1CQualificationError",
    "analyze_entity_world",
    "analyze_parametric_world",
    "analyze_structural_world",
    "entity_prior_audit",
    "load_contract",
    "parametric_prior_arms",
    "structural_prior_audit",
    "validate_contract",
    "world_truth_audit",
]
