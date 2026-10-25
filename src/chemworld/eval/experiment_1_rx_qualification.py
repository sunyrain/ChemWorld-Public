"""Frozen helpers for Experiment 1 RX qualification v1.0.1 and repairs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, variance
from typing import Any

import numpy as np

from chemworld.eval.experiment_1_ec_qualification import EXPECTED_COMMON_GATES
from chemworld.eval.experiment_1_ec_qualification import _validate_binding as _validate_binding
from chemworld.eval.provenance import canonical_json_sha256
from chemworld.materials import static_material_information_dossier
from chemworld.world.parameters import REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario

CONTRACT_VERSION = "chemworld-experiment-1-rx-qualification-contract-1.0.1"
CONTRACT_VERSION_V110 = "chemworld-experiment-1-rx-qualification-contract-1.1.0"
CONTRACT_VERSIONS = (CONTRACT_VERSION, CONTRACT_VERSION_V110)
ENTITY_REPORT_VERSION = "chemworld-experiment-1-rx-entity-world-report-1.0.1"
PARAMETRIC_REPORT_VERSION = "chemworld-experiment-1-rx-parametric-world-report-1.0.1"
STRUCTURAL_REPORT_VERSION = "chemworld-experiment-1-rx-structural-world-report-1.0.1"
STRUCTURAL_REPORT_VERSION_V110 = "chemworld-experiment-1-rx-structural-world-report-1.1.0"
EXPECTED_WORLD_IDS = tuple(f"RX-W0{index}" for index in range(1, 6))
EXPECTED_WORLD_SEEDS = tuple(range(5))
EXPECTED_ENTITY_PERMUTATION = (0, 2, 1, 3)
FORBIDDEN_PUBLIC_TOKENS = (
    "aligned",
    "hidden mechanism",
    "misindexed",
    "misspecified",
    "oracle",
    "world seed",
    "world_seed",
)


class Experiment1RXQualificationError(ValueError):
    """Raised when the frozen RX contract or evidence is malformed."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Experiment1RXQualificationError("RX qualification contract must be an object")
    errors = validate_contract(root, value)
    if errors:
        raise Experiment1RXQualificationError("; ".join(errors))
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
    if tuple(contract.get("common_gates", ())) != EXPECTED_COMMON_GATES:
        errors.append("Q1-Q8 gate registry changed")
    for key in ("specification", "system_specification"):
        _validate_binding(root, contract.get(key), key, errors)
    task = contract.get("task")
    if not isinstance(task, Mapping):
        errors.append("task contract is missing")
    else:
        _validate_binding(root, task.get("campaign_config"), "campaign_config", errors)
        expected = {
            "system_id": "RX",
            "task_id": "reaction-safety-constrained",
            "world_split": "public-test",
            "objective": "safe",
            "truth_family": "deactivating_baseline",
        }
        for key, value in expected.items():
            if task.get(key) != value:
                errors.append(f"task.{key} changed")
    sources = contract.get("source_assets")
    if not isinstance(sources, Mapping):
        errors.append("source asset bindings are missing")
    else:
        for key in ("parametric_reference_summary", "parametric_noise_summary"):
            _validate_binding(root, sources.get(key), key, errors)
    worlds = contract.get("worlds")
    rows = worlds.get("qualification") if isinstance(worlds, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 5:
        errors.append("exactly five RX worlds are required")
    else:
        if tuple(row.get("world_id") for row in rows) != EXPECTED_WORLD_IDS:
            errors.append("RX world IDs changed")
        if tuple(row.get("world_seed") for row in rows) != EXPECTED_WORLD_SEEDS:
            errors.append("RX world seeds changed")
        if any(row.get("truth_family") != "deactivating_baseline" for row in rows):
            errors.append("RX parent truth family changed")
    canary = worlds.get("canary") if isinstance(worlds, Mapping) else None
    if not isinstance(canary, Mapping) or canary.get("world_id") != "RX-W00":
        errors.append("RX-W00 canary is missing")
    elif canary.get("world_seed") != 900001 or canary.get("formal_denominator") is not False:
        errors.append("RX-W00 semantics changed")
    loci = contract.get("loci")
    if not isinstance(loci, Mapping):
        errors.append("locus contracts are missing")
    else:
        entity = loci.get("entity")
        if not isinstance(entity, Mapping):
            errors.append("entity locus is missing")
        elif (
            tuple(entity.get("descriptor_permutation", ())) != EXPECTED_ENTITY_PERMUTATION
            or entity.get("independent_replicates") != 3
            or entity.get("participant_unique_experiment_budget") != 4
        ):
            errors.append("entity frozen design changed")
        parametric = loci.get("parametric")
        if not isinstance(parametric, Mapping):
            errors.append("parametric locus is missing")
        elif (
            len(parametric.get("temperature_grid_K", ())) != 11
            or len(parametric.get("duration_grid_s", ())) != 11
            or parametric.get("participant_unique_experiment_budget") != 4
        ):
            errors.append("parametric frozen design changed")
        structural = loci.get("structural")
        expected_child = (
            "reversible_target_pathway" if version == CONTRACT_VERSION_V110 else "stable_catalyst"
        )
        if not isinstance(structural, Mapping):
            errors.append("structural locus is missing")
        elif (
            structural.get("parent_law_id") != "deactivating_baseline"
            or structural.get("child_law_id") != expected_child
            or structural.get("grid_cells") != 27
            or structural.get("law_count") != 2
            or structural.get("participant_unique_experiment_budget") != 4
        ):
            errors.append("structural frozen design changed")
        elif version == CONTRACT_VERSION_V110 and (
            structural.get("topology_transform_id") != "reversible_target_pathway_stress_v1"
            or structural.get("topology_severity") != 0.8
            or structural.get("reverse_rate_constant_s_inv_at_full_severity") != 0.000625
            or structural.get("observation_noise_namespace") != "experiment-1-v1.1.0-rx-structural"
            or structural.get("calibration_summary_sha256")
            != "b404f7416218f3a20b592c2e16eb411a346fc96de18d706d0b14222a8689bf63"
        ):
            errors.append("RX-S v1.1.0 repair binding changed")
    execution = contract.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("execution policy is missing")
    elif (
        execution.get("exact_replay_required") is not True
        or execution.get("overwrite_forbidden") is not True
        or execution.get("post_failure_redesign_in_same_campaign_forbidden") is not True
        or execution.get("participant_execution_authorized") is not False
        or execution.get("formal_benchmark_execution_authorized") is not False
    ):
        errors.append("execution safety policy changed")
    return errors


def world_truth_audit(contract: Mapping[str, Any], *, world_seed: int) -> dict[str, Any]:
    instance = DefaultScenarioGenerator().generate(get_scenario("reaction-safety"), world_seed)
    parameters = instance.parameters
    payload = {
        "system_id": "RX",
        "task_id": contract["task"]["task_id"],
        "world_seed": world_seed,
        "runtime_world_id": parameters.world_id,
        "world_split": parameters.split,
        "family_version": parameters.family_version,
        "truth_family": contract["task"]["truth_family"],
        "mechanism_hash": instance.compiled_mechanism.mechanism_hash,
        "pre_exponential": np.asarray(parameters.pre_exponential).tolist(),
        "activation_energy": np.asarray(parameters.activation_energy).tolist(),
        "catalyst_effects": np.asarray(parameters.catalyst_effects).tolist(),
        "solvent_effects": np.asarray(parameters.solvent_effects).tolist(),
        "solvent_risks": np.asarray(parameters.solvent_risks).tolist(),
        "catalyst_costs": np.asarray(parameters.catalyst_costs).tolist(),
        "delta_h_J_per_mol": np.asarray(parameters.delta_h_J_per_mol).tolist(),
        "ua_W_per_K": parameters.ua_W_per_K,
        "rho_cp_J_per_L_K": parameters.rho_cp_J_per_L_K,
        "environment_temperature_K": parameters.environment_temperature_K,
    }
    nominal = np.asarray(REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES, dtype=float)
    realized = np.asarray(parameters.catalyst_effects, dtype=float)
    scale = np.std(nominal, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    pair = _moved_pair(EXPECTED_ENTITY_PERMUTATION)
    mapping_rows = []
    for action_index in pair:
        swapped_index = pair[1] if action_index == pair[0] else pair[0]
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
    return {
        "world_seed": world_seed,
        "runtime_world_id": parameters.world_id,
        "truth_family": contract["task"]["truth_family"],
        "mechanism_hash": instance.compiled_mechanism.mechanism_hash,
        "truth_sha256": canonical_json_sha256(payload),
        "mapping_rows": mapping_rows,
        "aligned_mapping_not_reversed": all(row["own_mapping_closer"] for row in mapping_rows),
    }


def entity_prior_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    entity = contract["loci"]["entity"]
    task_id = contract["task"]["task_id"]
    aligned = static_material_information_dossier(
        {"mode": "anonymous_nominal_properties"}, task_id=task_id
    )
    misspecified = static_material_information_dossier(
        {
            "mode": "anonymous_misindexed_properties",
            "target_field": entity["target_field"],
            "descriptor_permutation": entity["descriptor_permutation"],
        },
        task_id=task_id,
    )
    aligned_text = _public_text(aligned)
    misspecified_text = _public_text(misspecified)
    joined = " ".join((*aligned_text, *misspecified_text)).lower()
    leakage = [token for token in FORBIDDEN_PUBLIC_TOKENS if token in joined]
    checks = {
        "aligned_and_misspecified_present": aligned is not None and misspecified is not None,
        "schema_matched": _public_shape(aligned) == _public_shape(misspecified),
        "text_template_matched": aligned_text == misspecified_text,
        "public_leakage_free": not leakage,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "leakage_tokens": leakage,
        "aligned_sha256": canonical_json_sha256(aligned),
        "misspecified_sha256": canonical_json_sha256(misspecified),
        "opaque": None,
        "aligned": aligned,
        "misspecified": misspecified,
    }


def analyze_entity_world(
    contract: Mapping[str, Any],
    *,
    world_id: str,
    world_seed: int,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entity = contract["loci"]["entity"]
    expected = (
        int(entity["nuisance_anchor_count"])
        * int(entity["categories_per_anchor"])
        * int(entity["independent_replicates"])
    )
    completed = [row for row in receipts if row.get("status") == "completed"]
    replayed = [
        row
        for row in completed
        if isinstance(row.get("exact_replay"), Mapping)
        and row["exact_replay"].get("verified") is True
    ]
    pair = _moved_pair(entity["descriptor_permutation"])
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(int(row["nuisance_anchor"]), int(row["target_category"]))].append(row)
    anchors = []
    metrics = (
        str(entity["gate_endpoint_id"]),
        *(str(value) for value in entity["support_endpoint_ids"]),
    )
    for anchor in range(int(entity["nuisance_anchor_count"])):
        left = grouped[(anchor, pair[0])]
        right = grouped[(anchor, pair[1])]
        complete = len(left) == len(right) == int(entity["independent_replicates"])
        metric_rows = []
        for metric in metrics:
            left_values = [float(row["allowed_metrics"][metric]) for row in left]
            right_values = [float(row["allowed_metrics"][metric]) for row in right]
            separation = abs(fmean(right_values) - fmean(left_values)) if complete else 0.0
            standard_error = (
                math.sqrt(
                    variance(left_values) / len(left_values)
                    + variance(right_values) / len(right_values)
                )
                if complete
                else None
            )
            metric_rows.append(
                {
                    "metric": metric,
                    "absolute_separation": separation,
                    "welch_standard_error": standard_error,
                }
            )
        mean_separation = fmean(row["absolute_separation"] for row in metric_rows)
        maximum_separation = max(row["absolute_separation"] for row in metric_rows)
        rms_standard_error = math.sqrt(
            fmean(float(row["welch_standard_error"] or 0.0) ** 2 for row in metric_rows)
        )
        snr = mean_separation / max(rms_standard_error, 1.0e-12)
        passed = bool(
            complete
            and mean_separation >= float(entity["minimum_mean_support_separation"])
            and maximum_separation >= float(entity["minimum_single_support_separation"])
            and snr >= float(entity["minimum_support_signal_to_noise_ratio"])
        )
        anchors.append(
            {
                "anchor": anchor,
                "left_category": pair[0],
                "right_category": pair[1],
                "metric_results": metric_rows,
                "mean_support_separation": mean_separation,
                "maximum_support_separation": maximum_separation,
                "support_rms_standard_error": rms_standard_error,
                "support_signal_to_noise_ratio": snr,
                "passed": passed,
            }
        )
    prior = entity_prior_audit(contract)
    truth = world_truth_audit(contract, world_seed=world_seed)
    noise_coordinates = {
        (row.get("nuisance_anchor"), row.get("target_category"), row.get("replicate"))
        for row in receipts
    }
    gates = {
        "Q1_world_integrity": len(receipts) == len(completed) == len(replayed) == expected,
        "Q2_task_accessibility": len({row.get("recipe_id") for row in completed}) == 8,
        "Q3_public_contract_invariance": bool(prior["checks"]["public_leakage_free"]),
        "Q4_prior_symmetry": bool(
            prior["checks"]["schema_matched"] and prior["checks"]["text_template_matched"]
        ),
        "Q5_identifiability": all(row["passed"] for row in anchors),
        "Q6_budgeted_falsifiability": int(entity["participant_unique_experiment_budget"]) >= 4,
        "Q7_behavioral_relevance": bool(
            truth["aligned_mapping_not_reversed"]
            and max(row["mean_support_separation"] for row in anchors)
            >= float(entity["minimum_observed_endpoint_consequence"])
        ),
        "Q8_noise_robustness": bool(
            len(noise_coordinates) == expected
            and all(
                row["support_signal_to_noise_ratio"]
                >= float(entity["minimum_support_signal_to_noise_ratio"])
                for row in anchors
            )
        ),
    }
    return _world_report(
        schema_version=ENTITY_REPORT_VERSION,
        world_id=world_id,
        world_seed=world_seed,
        locus="entity",
        truth_sha256=str(truth["truth_sha256"]),
        denominators={
            "planned": expected,
            "attempted": len(receipts),
            "completed": len(completed),
            "exact_replay": len(replayed),
            "failures": len(receipts) - len(completed),
        },
        gates=gates,
        extra={
            "prior_audit": {
                key: value for key, value in prior.items() if key not in {"aligned", "misspecified"}
            },
            "private_world_audit": truth,
            "anchor_results": anchors,
        },
    )


def analyze_parametric_world(
    contract: Mapping[str, Any],
    *,
    world_id: str,
    world_seed: int,
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    locus = contract["loci"]["parametric"]
    checks = _mapping(analysis.get("checks"))
    selected = _mapping(analysis.get("selected_reflection"))
    reflection_checks = _mapping(selected.get("checks"))
    prior_matching = _mapping(analysis.get("prior_matching"))
    leakage = _mapping(analysis.get("leakage_audit"))
    blind = _mapping(analysis.get("blind_identification"))
    expected = len(locus["temperature_grid_K"]) * len(locus["duration_grid_s"])
    replayed = sum(
        isinstance(row.get("exact_replay"), Mapping) and row["exact_replay"].get("verified") is True
        for row in rows
    )
    classified = sum(row.get("status") in {"completed", "physical_failure"} for row in rows)
    truth = world_truth_audit(contract, world_seed=world_seed)
    gates = {
        "Q1_world_integrity": bool(
            len(rows) == expected
            and classified == expected
            and replayed == expected
            and int(analysis.get("platform_failure_count", -1)) == 0
        ),
        "Q2_task_accessibility": bool(
            checks.get("safe_fit_count") and checks.get("safe_held_out_count")
        ),
        "Q3_public_contract_invariance": bool(leakage.get("passed")),
        "Q4_prior_symmetry": bool(prior_matching.get("passed")),
        "Q5_identifiability": bool(
            checks.get("aligned_score_normalized_mae")
            and checks.get("qualified_reflection_exists")
            and reflection_checks.get("held_out_disagreement")
            and reflection_checks.get("blind_identification_margin")
        ),
        "Q6_budgeted_falsifiability": bool(
            int(locus["participant_unique_experiment_budget"]) >= 4
            and reflection_checks.get("low_side_falsification_region")
            and reflection_checks.get("high_side_falsification_region")
            and reflection_checks.get("representatives_separated")
        ),
        "Q7_behavioral_relevance": bool(
            blind.get("identified_aligned_law")
            and float(selected.get("blind_error_margin", 0.0))
            >= float(locus["minimum_blind_error_margin"])
        ),
        "Q8_noise_robustness": bool(
            reflection_checks.get("held_out_disagreement")
            and float(selected.get("disagreement_fraction", 0.0))
            >= float(locus["minimum_disagreement_fraction"])
        ),
    }
    return _world_report(
        schema_version=PARAMETRIC_REPORT_VERSION,
        world_id=world_id,
        world_seed=world_seed,
        locus="parametric",
        truth_sha256=str(truth["truth_sha256"]),
        denominators={
            "planned": expected,
            "attempted": len(rows),
            "classified": classified,
            "exact_replay": replayed,
            "platform_failures": int(analysis.get("platform_failure_count", 0)),
            "physical_failures": int(analysis.get("physical_failure_count", 0)),
        },
        gates=gates,
        extra={"legacy_analysis": dict(analysis)},
    )


def structural_prior_arms(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reversible = bool(
        contract is not None and contract.get("schema_version") == CONTRACT_VERSION_V110
    )
    schema_version = (
        "chemworld-rx-structural-prior-1.1.0"
        if reversible
        else "chemworld-rx-structural-prior-1.0.1"
    )
    return {
        "opaque": None,
        "aligned": {
            "schema_version": schema_version,
            "confidence": "moderate",
            "claim": (
                "The target pathway may be effectively irreversible on the public support."
                if reversible
                else "Catalyst activity may decline cumulatively with temperature and duration."
            ),
        },
        "misspecified": {
            "schema_version": schema_version,
            "confidence": "moderate",
            "claim": (
                "The target pathway may have an appreciable reverse channel on the public support."
                if reversible
                else (
                    "Catalyst activity may remain stable cumulatively with temperature "
                    "and duration."
                )
            ),
        },
    }


def analyze_structural_world(
    contract: Mapping[str, Any],
    *,
    world_id: str,
    world_seed: int,
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    locus = contract["loci"]["structural"]
    checks = _mapping(analysis.get("checks"))
    mechanism = _mapping(analysis.get("mechanism_audit"))
    reversible = contract.get("schema_version") == CONTRACT_VERSION_V110
    priors = structural_prior_arms(contract)
    aligned = _mapping(priors["aligned"])
    misspecified = _mapping(priors["misspecified"])
    leakage_text = " ".join((*_public_text(aligned), *_public_text(misspecified))).lower()
    leakage = [token for token in FORBIDDEN_PUBLIC_TOKENS if token in leakage_text]
    truth = world_truth_audit(contract, world_seed=world_seed)
    expected = int(locus["grid_cells"]) * int(locus["law_count"])
    denominators = _mapping(analysis.get("denominators"))
    gates = {
        "Q1_world_integrity": bool(
            len(rows) == expected
            and denominators.get("attempted") == expected
            and denominators.get("exact_replay") == expected
            and denominators.get("platform_failures") == 0
            and mechanism.get("baseline_mechanism_hash") == truth["mechanism_hash"]
        ),
        "Q2_task_accessibility": bool(
            checks.get("all_outcomes_classified")
            and checks.get("all_direct_metrics_publicly_observed")
        ),
        "Q3_public_contract_invariance": bool(
            not leakage and checks.get("participant_visible_leakage_free")
        ),
        "Q4_prior_symmetry": bool(
            set(aligned) == set(misspecified)
            and checks.get("paired_action_plans")
            and checks.get("paired_observation_noise")
        ),
        "Q5_identifiability": bool(
            checks.get("at_least_two_direct_metrics_resolve_topology")
            and checks.get("duration_accumulation_signature")
        ),
        "Q6_budgeted_falsifiability": bool(
            int(locus["participant_unique_experiment_budget"]) >= 4
            and checks.get("two_separated_safe_supporting_cells")
            and checks.get("support_spans_two_catalyst_doses")
        ),
        "Q7_behavioral_relevance": bool(
            (
                checks.get("mechanism_adds_one_reverse_reaction")
                and checks.get("opposite_stoichiometric_channel")
                and checks.get("execution_mechanism_binding_matches")
                and checks.get("stopping_decision_changed")
                if reversible
                else checks.get("mechanism_removes_one_deactivation_reaction")
            )
            and checks.get("mechanism_hash_changes")
            and checks.get("at_least_two_direct_metrics_resolve_topology")
        ),
        "Q8_noise_robustness": bool(
            checks.get("all_exact_replay") and checks.get("paired_observation_noise")
        ),
    }
    return _world_report(
        schema_version=(
            STRUCTURAL_REPORT_VERSION_V110 if reversible else STRUCTURAL_REPORT_VERSION
        ),
        world_id=world_id,
        world_seed=world_seed,
        locus="structural",
        truth_sha256=str(truth["truth_sha256"]),
        denominators=dict(denominators),
        gates=gates,
        extra={
            "prior_arms": priors,
            "prior_leakage_tokens": leakage,
            "legacy_analysis": dict(analysis),
        },
    )


def _world_report(
    *,
    schema_version: str,
    world_id: str,
    world_seed: int,
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
        "system_id": "RX",
        "world_id": world_id,
        "world_seed": world_seed,
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


def _moved_pair(permutation: Sequence[Any]) -> tuple[int, int]:
    values = [int(value) for value in permutation]
    moved = [index for index, source in enumerate(values) if index != source]
    if len(moved) != 2 or values[moved[0]] != moved[1] or values[moved[1]] != moved[0]:
        raise ValueError("descriptor permutation must be one transposition")
    return moved[0], moved[1]


def _public_shape(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _public_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_shape(item) for item in value]
    return type(value).__name__


def _public_text(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(text for item in value.values() for text in _public_text(item))
    if isinstance(value, list):
        return tuple(text for item in value for text in _public_text(item))
    return (value,) if isinstance(value, str) else ()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
