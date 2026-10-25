"""Frozen helpers for Experiment 1 FL qualification v1.0.1 and v1.1.0."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, variance
from typing import Any

import numpy as np

from chemworld.eval.experiment_1_c_qualification import _public_text as _public_text
from chemworld.eval.experiment_1_c_qualification import _validate_binding as _validate_binding
from chemworld.eval.experiment_1_ec_qualification import EXPECTED_COMMON_GATES
from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_static_topology_q0 import analyze_task
from chemworld.materials import static_material_information_dossier
from chemworld.world.continuous_flow import (
    FIXED_FLOW_REACTOR_INNER_DIAMETER_M,
    FIXED_FLOW_REACTOR_VOLUME_L,
)
from chemworld.world.parameters import REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario

CONTRACT_VERSION = "chemworld-experiment-1-fl-qualification-contract-1.0.1"
CONTRACT_VERSION_V110 = "chemworld-experiment-1-fl-qualification-contract-1.1.0"
CONTRACT_VERSIONS = (CONTRACT_VERSION, CONTRACT_VERSION_V110)
ENTITY_REPORT_VERSION = "chemworld-experiment-1-fl-entity-world-report-1.0.1"
PARAMETRIC_REPORT_VERSION = "chemworld-experiment-1-fl-parametric-world-report-1.0.1"
STRUCTURAL_REPORT_VERSION = "chemworld-experiment-1-fl-structural-world-report-1.0.1"
ENTITY_REPORT_VERSION_V110 = "chemworld-experiment-1-fl-entity-world-report-1.1.0"
PARAMETRIC_REPORT_VERSION_V110 = "chemworld-experiment-1-fl-parametric-world-report-1.1.0"
STRUCTURAL_REPORT_VERSION_V110 = "chemworld-experiment-1-fl-structural-world-report-1.1.0"
EXPECTED_WORLD_IDS = tuple(f"FL-W0{index}" for index in range(1, 6))
EXPECTED_WORLD_SEEDS = tuple(range(5))
EXPECTED_ENTITY_PERMUTATION = (0, 2, 1, 3)
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


class Experiment1FLQualificationError(ValueError):
    """Raised when the frozen FL contract or evidence is malformed."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Experiment1FLQualificationError("FL qualification contract must be an object")
    errors = validate_contract(root, value)
    if errors:
        raise Experiment1FLQualificationError("; ".join(errors))
    return value


def validate_contract(root: Path, contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    version = contract.get("schema_version")
    if version not in CONTRACT_VERSIONS:
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
    _validate_binding(root, contract.get("system_specification"), "FL specification", errors)
    task = _mapping(contract.get("task"))
    expected_task = {
        "system_id": "FL",
        "task_id": "flow-reaction-optimization",
        "world_split": "public-test",
        "objective": "balanced",
        "truth_family": (
            "fixed_geometry_pfr_parent"
            if version == CONTRACT_VERSION_V110
            else "geometry_resolved_pfr_parent"
        ),
    }
    for key, value in expected_task.items():
        if task.get(key) != value:
            errors.append(f"task.{key} changed")
    _validate_binding(root, task.get("campaign_config"), "campaign config", errors)
    worlds = _mapping(contract.get("worlds"))
    rows = worlds.get("qualification")
    if not isinstance(rows, list) or len(rows) != 5:
        errors.append("FL must freeze five qualification Worlds")
    else:
        if tuple(str(row.get("world_id")) for row in rows) != EXPECTED_WORLD_IDS:
            errors.append("FL World IDs changed")
        if tuple(int(row.get("world_seed", -1)) for row in rows) != EXPECTED_WORLD_SEEDS:
            errors.append("FL World seeds changed")
        manifests = [canonical_json_sha256(row.get("world_interventions", [])) for row in rows]
        if len(set(manifests)) != 5:
            errors.append("FL Worlds are not five distinct intervention manifests")
    canary = _mapping(worlds.get("canary"))
    if (
        canary.get("world_id") != "FL-W00"
        or canary.get("world_seed") != 900003
        or canary.get("formal_denominator") is not False
    ):
        errors.append("FL-W00 canary changed")
    if tuple(contract.get("common_gates", ())) != EXPECTED_COMMON_GATES:
        errors.append("Q1-Q8 gate registry changed")
    if version == CONTRACT_VERSION_V110:
        hardware = _mapping(contract.get("hardware"))
        if (
            hardware.get("reactor_volume_L") != FIXED_FLOW_REACTOR_VOLUME_L
            or hardware.get("internal_diameter_m") != FIXED_FLOW_REACTOR_INNER_DIAMETER_M
            or hardware.get("residence_derivation") != "tau_s=1080/Q_mL_min"
        ):
            errors.append("FL fixed-hardware contract changed")
    loci = _mapping(contract.get("loci"))
    if set(loci) != {"entity", "parametric", "structural"}:
        errors.append("FL prior loci changed")
    else:
        entity = _mapping(loci["entity"])
        expected_entity_support = (
            entity.get("flow_rate_anchors_mL_min") == [3.6, 1.2]
            and entity.get("derived_residence_anchors_s") == [300.0, 900.0]
            if version == CONTRACT_VERSION_V110
            else entity.get("residence_anchors_s") == [300.0, 900.0]
        )
        if (
            tuple(entity.get("descriptor_permutation", ())) != EXPECTED_ENTITY_PERMUTATION
            or entity.get("catalyst_targets") != [1, 2]
            or not expected_entity_support
            or entity.get("independent_replicates") != 3
        ):
            errors.append("FL entity design changed")
        parametric = _mapping(loci["parametric"])
        expected_parametric_support = (
            parametric.get("flow_rate_levels_mL_min") == [2.4, 0.72]
            and parametric.get("derived_residence_levels_s") == [450.0, 1500.0]
            if version == CONTRACT_VERSION_V110
            else parametric.get("residence_levels_s") == [450.0, 1500.0]
        )
        if (
            parametric.get("temperature_levels_K") != [370.0, 410.0]
            or not expected_parametric_support
            or parametric.get("independent_replicates") != 3
        ):
            errors.append("FL parametric design changed")
        structural = _mapping(loci["structural"])
        expected_structural_support = (
            structural.get("flow_rate_levels_mL_min") == [3.6, 1.2, 0.6]
            and structural.get("derived_residence_levels_s") == [300.0, 900.0, 1800.0]
            if version == CONTRACT_VERSION_V110
            else structural.get("residence_levels_s") == [300.0, 900.0, 1800.0]
        )
        if (
            structural.get("temperature_levels_K") != [350.0, 390.0, 425.0]
            or not expected_structural_support
            or structural.get("grid_cells") != 9
            or structural.get("law_count") != 2
        ):
            errors.append("FL structural design changed")
    execution = _mapping(contract.get("execution"))
    if (
        execution.get("exact_replay_required") is not True
        or execution.get("overwrite_forbidden") is not True
        or execution.get("post_failure_redesign_in_same_campaign_forbidden") is not True
    ):
        errors.append("FL execution policy changed")
    return errors


def world_truth_audit(contract: Mapping[str, Any], world: Mapping[str, Any]) -> dict[str, Any]:
    interventions = tuple(dict(item) for item in world["world_interventions"])
    scenario = DefaultScenarioGenerator().generate(
        get_scenario("flow-reaction-optimization"), int(world["world_seed"]), interventions
    )
    repeated = DefaultScenarioGenerator().generate(
        get_scenario("flow-reaction-optimization"), int(world["world_seed"]), interventions
    )
    parameters = scenario.parameters
    nominal = np.asarray(REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES, dtype=float)
    realized = np.asarray(parameters.catalyst_effects, dtype=float)
    scale = np.std(nominal, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    mapping_rows = []
    for action_index in (1, 2):
        swapped_index = 2 if action_index == 1 else 1
        own_distance = float(
            np.linalg.norm((realized[action_index] - nominal[action_index]) / scale)
        )
        swapped_distance = float(
            np.linalg.norm((realized[action_index] - nominal[swapped_index]) / scale)
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
    payload = {
        "system_id": "FL",
        "world_id": str(world["world_id"]),
        "world_seed": int(world["world_seed"]),
        "task_id": "flow-reaction-optimization",
        "world_interventions": list(interventions),
        "runtime_world_id": parameters.world_id,
        "mechanism_hash": scenario.compiled_mechanism.mechanism_hash,
        "world_family_intervention_hash": scenario.initial_state.metadata.get(
            "world_family_intervention_hash"
        ),
        "flow_rate_multiplier": float(domain["flow_rate_multiplier"]),
        "flow_residence_multiplier": float(domain["flow_residence_multiplier"]),
        "legacy_unused_flow_residence_multiplier": float(domain["flow_residence_multiplier"]),
        "flow_boundary_ua_multiplier": float(domain["flow_boundary_ua_multiplier"]),
        "fixed_reactor_volume_L": FIXED_FLOW_REACTOR_VOLUME_L,
        "fixed_reactor_inner_diameter_m": FIXED_FLOW_REACTOR_INNER_DIAMETER_M,
        "mapping_rows": mapping_rows,
    }
    payload["truth_sha256"] = canonical_json_sha256(payload)
    payload["deterministic"] = bool(
        scenario.parameters.world_id == repeated.parameters.world_id
        and scenario.compiled_mechanism.mechanism_hash == repeated.compiled_mechanism.mechanism_hash
    )
    payload["aligned_mapping_not_reversed"] = all(row["own_mapping_closer"] for row in mapping_rows)
    return payload


def entity_prior_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    locus = contract["loci"]["entity"]
    aligned = static_material_information_dossier(
        {"mode": "anonymous_nominal_properties"},
        task_id="flow-reaction-optimization",
    )
    misspecified = static_material_information_dossier(
        {
            "mode": "anonymous_misindexed_properties",
            "target_field": "catalyst",
            "descriptor_permutation": list(locus["descriptor_permutation"]),
        },
        task_id="flow-reaction-optimization",
    )
    joined = json.dumps([aligned, misspecified], ensure_ascii=False, sort_keys=True).lower()
    leakage = sorted(token for token in FORBIDDEN_PUBLIC_TOKENS if token in joined)
    checks = {
        "aligned_and_misspecified_present": aligned is not None and misspecified is not None,
        "schema_matched": _public_shape(aligned) == _public_shape(misspecified),
        "text_template_matched": _public_text(aligned) == _public_text(misspecified),
        "catalyst_only_transposition": tuple(locus["descriptor_permutation"])
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


def parametric_prior_arms(contract: Mapping[str, Any]) -> dict[str, Any]:
    locus = contract["loci"]["parametric"]
    fixed_hardware = contract.get("schema_version") == CONTRACT_VERSION_V110
    common = {
        "target": (
            "local_flow_derived_residence_effect_on_public_product_metrics"
            if fixed_hardware
            else "local_residence_effect_on_public_product_metrics"
        ),
        "temperature_domain_K": list(locus["temperature_levels_K"]),
        "residence_domain_s": list(
            locus["derived_residence_levels_s" if fixed_hardware else "residence_levels_s"]
        ),
        "scope": (
            "fixed reactor hardware and material context"
            if fixed_hardware
            else "fixed public flow-rate and material context"
        ),
    }
    if fixed_hardware:
        common["flow_rate_domain_mL_min"] = list(locus["flow_rate_levels_mL_min"])
    arms = {
        "aligned": {**common, "relation": locus["aligned_relation"]},
        "misspecified": {**common, "relation": locus["misspecified_relation"]},
        "opaque": {**common, "relation": "withheld"},
    }
    return _prior_arm_audit(arms)


def structural_prior_arms(contract: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        "target": "target_pathway_topology",
        "support": "frozen temperature-by-residence grid",
        "public_metrics": list(contract["loci"]["structural"]["direct_metrics"]),
    }
    arms = {
        "aligned": {**common, "law_family": "irreversible_target_pathway"},
        "misspecified": {**common, "law_family": "reversible_target_pathway"},
        "opaque": {**common, "law_family": "withheld"},
    }
    return _prior_arm_audit(arms)


def analyze_entity_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    locus = contract["loci"]["entity"]
    fixed_hardware = contract.get("schema_version") == CONTRACT_VERSION_V110
    support_values = locus["flow_rate_anchors_mL_min" if fixed_hardware else "residence_anchors_s"]
    expected = (
        len(support_values) * len(locus["catalyst_targets"]) * int(locus["independent_replicates"])
    )
    completed = _completed(receipts)
    grouped: dict[tuple[float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        support = row["flow_rate_mL_min"] if fixed_hardware else row["residence_time_s"]
        grouped[(float(support), int(row["catalyst"]))].append(row)
    anchors = []
    for support_index, support in enumerate(support_values):
        left = grouped[(float(support), 1)]
        right = grouped[(float(support), 2)]
        metric_rows = _group_metric_contrasts(left, right, locus["direct_metrics"])
        complete = len(left) == len(right) == int(locus["independent_replicates"])
        mean_gap = fmean(row["absolute_separation"] for row in metric_rows)
        max_gap = max(row["absolute_separation"] for row in metric_rows)
        rms_se = math.sqrt(fmean(float(row["standard_error"] or 0.0) ** 2 for row in metric_rows))
        snr = mean_gap / max(rms_se, 1.0e-12)
        anchors.append(
            {
                "residence_time_s": (
                    locus["derived_residence_anchors_s"][support_index]
                    if fixed_hardware
                    else support
                ),
                "flow_rate_mL_min": float(support) if fixed_hardware else locus["flow_rate_mL_min"],
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
        "Q6_budgeted_falsifiability": int(locus["participant_unique_experiment_budget"]) >= 4,
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
        ENTITY_REPORT_VERSION_V110 if fixed_hardware else ENTITY_REPORT_VERSION,
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
    fixed_hardware = contract.get("schema_version") == CONTRACT_VERSION_V110
    support_values = locus["flow_rate_levels_mL_min" if fixed_hardware else "residence_levels_s"]
    expected = (
        len(locus["temperature_levels_K"])
        * len(support_values)
        * int(locus["independent_replicates"])
    )
    completed = _completed(receipts)
    grouped: dict[tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        support = row["flow_rate_mL_min"] if fixed_hardware else row["residence_time_s"]
        grouped[(float(row["temperature_K"]), float(support))].append(row)
    left_support, right_support = map(float, support_values)
    temperature_reports = []
    for temperature in locus["temperature_levels_K"]:
        low = grouped[(float(temperature), left_support)]
        high = grouped[(float(temperature), right_support)]
        metrics = _signed_group_metric_contrasts(low, high, locus["product_metrics"])
        passing = [
            row
            for row in metrics
            if row["signed_effect"] >= float(locus["minimum_residence_effect"])
            and row["signal_to_noise_ratio"] >= float(locus["minimum_signal_to_noise_ratio"])
        ]
        temperature_reports.append(
            {
                "temperature_K": temperature,
                "metric_results": metrics,
                "passed": bool(
                    len(low) == len(high) == int(locus["independent_replicates"]) and passing
                ),
            }
        )
    prior = parametric_prior_arms(contract)
    truth = world_truth_audit(contract, world)
    geometry = [row.get("flow_configuration") for row in completed]
    volumes = {
        float(item["reactor_volume_L"])
        for item in geometry
        if isinstance(item, Mapping) and "reactor_volume_L" in item
    }
    lengths = {
        float(item["geometry_length_m"])
        for item in geometry
        if isinstance(item, Mapping) and "geometry_length_m" in item
    }
    geometry_bound = bool(
        geometry
        and all(
            isinstance(item, Mapping)
            and (
                math.isclose(
                    float(item["configured_flow_rate_mL_min"]),
                    float(row["flow_rate_mL_min"]),
                )
                if fixed_hardware
                else float(item["configured_flow_rate_mL_min"]) == float(locus["flow_rate_mL_min"])
            )
            and float(item["reactor_volume_L"]) > 0.0
            and float(item["geometry_length_m"]) > 0.0
            for row, item in zip(completed, geometry, strict=True)
        )
        and (not fixed_hardware or volumes == {FIXED_FLOW_REACTOR_VOLUME_L})
        and (not fixed_hardware or len(lengths) == 1)
    )
    gates = {
        "Q1_world_integrity": _integrity(receipts, expected) and truth["deterministic"],
        "Q2_task_accessibility": len({row.get("action_plan_sha256") for row in completed}) == 4,
        "Q3_public_contract_invariance": _public_receipts_ok(completed, locus["direct_metrics"]),
        "Q4_prior_symmetry": prior["passed"],
        "Q5_identifiability": all(row["passed"] for row in temperature_reports),
        "Q6_budgeted_falsifiability": int(locus["participant_unique_experiment_budget"]) == 4,
        "Q7_behavioral_relevance": geometry_bound
        and any(
            any(
                metric["signed_effect"] >= float(locus["minimum_residence_effect"])
                for metric in row["metric_results"]
            )
            for row in temperature_reports
        ),
        "Q8_noise_robustness": all(
            any(
                metric["signal_to_noise_ratio"] >= float(locus["minimum_signal_to_noise_ratio"])
                for metric in row["metric_results"]
            )
            for row in temperature_reports
        ),
    }
    return _world_report(
        PARAMETRIC_REPORT_VERSION_V110 if fixed_hardware else PARAMETRIC_REPORT_VERSION,
        world,
        "parametric",
        receipts,
        expected,
        gates,
        {
            "private_world_audit": truth,
            "prior_audit": prior,
            "temperature_reports": temperature_reports,
            "geometry_binding_verified": geometry_bound,
        },
    )


def analyze_structural_world(
    contract: Mapping[str, Any],
    *,
    world: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    mechanism_audit: Mapping[str, Any],
) -> dict[str, Any]:
    expected = 18
    analysis = analyze_task("flow-reaction-optimization", receipts, mechanism_audit)
    checks = analysis["checks"]
    prior = structural_prior_arms(contract)
    gates = {
        "Q1_world_integrity": bool(
            len(receipts) == expected
            and checks["all_completed"]
            and checks["all_exact_replay"]
            and checks["zero_platform_failures"]
        ),
        "Q2_task_accessibility": checks["fixed_execution_denominator"],
        "Q3_public_contract_invariance": bool(
            checks.get("all_direct_metrics_publicly_observed")
            and checks.get("participant_visible_leakage_free")
        ),
        "Q4_prior_symmetry": bool(
            prior["passed"] and checks["paired_action_plans"] and checks["paired_observation_noise"]
        ),
        "Q5_identifiability": bool(
            checks.get("at_least_two_direct_metrics_resolve_topology")
            and checks.get("two_separated_supporting_cells")
        ),
        "Q6_budgeted_falsifiability": int(
            contract["loci"]["structural"]["participant_unique_experiment_budget"]
        )
        == 4,
        "Q7_behavioral_relevance": bool(
            checks.get("duration_accumulation_signature")
            and checks["mechanism_adds_one_reverse_reaction"]
            and checks["mechanism_hash_changes"]
            and checks["execution_mechanism_binding_matches"]
        ),
        "Q8_noise_robustness": bool(
            checks["paired_observation_noise"] and checks.get("all_direct_metrics_finite")
        ),
    }
    return _world_report(
        (
            STRUCTURAL_REPORT_VERSION_V110
            if contract.get("schema_version") == CONTRACT_VERSION_V110
            else STRUCTURAL_REPORT_VERSION
        ),
        world,
        "structural",
        receipts,
        expected,
        gates,
        {"prior_audit": prior, "static_topology_analysis": analysis},
    )


def _integrity(receipts: Sequence[Mapping[str, Any]], expected: int) -> bool:
    return bool(
        len(receipts) == expected
        and all(row.get("status") == "completed" for row in receipts)
        and all(row.get("exact_replay") is True for row in receipts)
        and all(row.get("law_binding_verified") is True for row in receipts)
    )


def _completed(receipts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in receipts if row.get("status") == "completed"]


def _public_receipts_ok(receipts: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> bool:
    return bool(
        receipts
        and all(not row.get("participant_visible_leakage_matches") for row in receipts)
        and all(
            isinstance(row.get("measurement"), Mapping)
            and isinstance(row.get("observed_mask"), Mapping)
            and all(row["observed_mask"].get(metric) is True for metric in metrics)
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
        left_values = [float(row["measurement"][metric]) for row in left]
        right_values = [float(row["measurement"][metric]) for row in right]
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


def _signed_group_metric_contrasts(
    low: Sequence[Mapping[str, Any]],
    high: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows = _group_metric_contrasts(low, high, metrics)
    for row in rows:
        signed = float(row["right_mean"] or 0.0) - float(row["left_mean"] or 0.0)
        row["signed_effect"] = signed
        row["signal_to_noise_ratio"] = signed / max(float(row["standard_error"] or 0.0), 1.0e-12)
    return rows


def _prior_arm_audit(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    shapes = {_public_shape(value) for value in arms.values()}
    rendered = json.dumps(list(arms.values()), ensure_ascii=False, sort_keys=True).lower()
    leakage = sorted(token for token in FORBIDDEN_PUBLIC_TOKENS if token in rendered)
    checks = {
        "three_arms_present": set(arms) == {"aligned", "misspecified", "opaque"},
        "schema_matched": len(shapes) == 1,
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
        "system_id": "FL",
        "task_id": "flow-reaction-optimization",
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
    return type(value).__name__


__all__ = [
    "CONTRACT_VERSION",
    "EXPECTED_WORLD_IDS",
    "Experiment1FLQualificationError",
    "analyze_entity_world",
    "analyze_parametric_world",
    "analyze_structural_world",
    "entity_prior_audit",
    "load_contract",
    "parametric_prior_arms",
    "structural_prior_arms",
    "validate_contract",
    "world_truth_audit",
]
