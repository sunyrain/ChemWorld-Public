"""Frozen helpers for Experiment 1 EC qualification v1.0.1."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, variance
from typing import Any

import numpy as np

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.materials import static_material_information_dossier
from chemworld.world.electrochemical_material_family import (
    electrochemical_material_family,
    electrochemical_material_instance_sha256,
)
from chemworld.world.parameters import load_chemworld_parameters

CONTRACT_VERSION = "chemworld-experiment-1-ec-qualification-contract-1.0.1"
REPORT_VERSION = "chemworld-experiment-1-ec-entity-world-report-1.0.1"
PARAMETRIC_REPORT_VERSION = (
    "chemworld-experiment-1-ec-parametric-world-report-1.0.1"
)
STRUCTURAL_REPORT_VERSION = (
    "chemworld-experiment-1-ec-structural-world-report-1.0.1"
)
EXPECTED_WORLD_IDS = tuple(f"EC-W0{index}" for index in range(1, 6))
EXPECTED_WORLD_SEEDS = tuple(range(5))
EXPECTED_COMMON_GATES = (
    "Q1_world_integrity",
    "Q2_task_accessibility",
    "Q3_public_contract_invariance",
    "Q4_prior_symmetry",
    "Q5_identifiability",
    "Q6_budgeted_falsifiability",
    "Q7_behavioral_relevance",
    "Q8_noise_robustness",
)
FORBIDDEN_PUBLIC_TOKENS = (
    "aligned",
    "hidden mechanism",
    "misindexed",
    "misspecified",
    "oracle",
    "world seed",
    "world_seed",
)


class Experiment1ECQualificationError(ValueError):
    """Raised when the frozen contract or evidence is malformed."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Experiment1ECQualificationError("EC qualification contract must be an object")
    errors = validate_contract(root, value)
    if errors:
        raise Experiment1ECQualificationError("; ".join(errors))
    return value


def validate_contract(root: Path, contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != CONTRACT_VERSION:
        errors.append("contract schema version changed")
    if contract.get("status") != "development_frozen_before_execution":
        errors.append("contract is not frozen before execution")
    if contract.get("development_only") is not True:
        errors.append("contract must remain development-only")
    if contract.get("participant_provider_calls") != 0:
        errors.append("qualification must make zero participant provider calls")
    if tuple(contract.get("common_gates", ())) != EXPECTED_COMMON_GATES:
        errors.append("Q1-Q8 gate registry changed")
    _validate_binding(root, contract.get("specification"), "specification", errors)
    _validate_binding(root, contract.get("experiment_note"), "experiment_note", errors)
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
        errors.append("source_assets bindings are missing")
    else:
        _validate_binding(
            root,
            source_assets.get("parametric_reference_summary"),
            "parametric_reference_summary",
            errors,
        )
        _validate_binding(
            root,
            source_assets.get("parametric_noise_summary"),
            "parametric_noise_summary",
            errors,
        )
    worlds = contract.get("worlds")
    rows = worlds.get("qualification") if isinstance(worlds, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 5:
        errors.append("exactly five qualification worlds are required")
    else:
        if tuple(row.get("world_id") for row in rows) != EXPECTED_WORLD_IDS:
            errors.append("qualification world IDs changed")
        if tuple(row.get("world_seed") for row in rows) != EXPECTED_WORLD_SEEDS:
            errors.append("qualification world seeds changed")
    canary = worlds.get("canary") if isinstance(worlds, Mapping) else None
    if not isinstance(canary, Mapping) or canary.get("world_id") != "EC-W00":
        errors.append("EC-W00 canary is missing")
    elif canary.get("world_seed") != 900000 or canary.get("formal_denominator") is not False:
        errors.append("EC-W00 semantics changed")
    entity = _locus(contract, "entity", errors)
    if entity:
        if entity.get("descriptor_permutation") != [2, 1, 0, 3]:
            errors.append("entity transposition changed")
        if entity.get("independent_replicates") != 3:
            errors.append("entity replicate count changed")
        if entity.get("participant_unique_experiment_budget") != 4:
            errors.append("entity participant budget changed")
    execution = contract.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("execution policy is missing")
    elif (
        execution.get("exact_replay_required") is not True
        or execution.get("overwrite_forbidden") is not True
        or execution.get("participant_execution_authorized") is not False
        or execution.get("formal_benchmark_execution_authorized") is not False
    ):
        errors.append("execution safety policy changed")
    return errors


def _validate_binding(
    root: Path,
    value: object,
    name: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{name} binding is missing")
        return
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        errors.append(f"{name} binding is malformed")
        return
    path = (root / path_value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        errors.append(f"{name} binding path is invalid")
    elif file_sha256(path) != digest:
        errors.append(f"{name} binding hash is stale")


def _locus(
    contract: Mapping[str, Any],
    locus_id: str,
    errors: list[str] | None = None,
) -> Mapping[str, Any]:
    loci = contract.get("loci")
    value = loci.get(locus_id) if isinstance(loci, Mapping) else None
    if not isinstance(value, Mapping):
        if errors is not None:
            errors.append(f"{locus_id} locus is missing")
        return {}
    return value


def entity_prior_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    task = contract["task"]
    entity = contract["loci"]["entity"]
    family_id = task["electrochemical_material_family_id"]
    aligned = static_material_information_dossier(
        {"mode": "anonymous_nominal_properties"},
        task_id=task["task_id"],
        material_family_id=family_id,
    )
    misspecified = static_material_information_dossier(
        {
            "mode": "anonymous_misindexed_properties",
            "target_field": entity["target_field"],
            "descriptor_permutation": entity["descriptor_permutation"],
        },
        task_id=task["task_id"],
        material_family_id=family_id,
    )
    aligned_shape = _public_shape(aligned)
    misspecified_shape = _public_shape(misspecified)
    aligned_text = _public_text(aligned)
    misspecified_text = _public_text(misspecified)
    joined = " ".join((*aligned_text, *misspecified_text)).lower()
    leakage = [token for token in FORBIDDEN_PUBLIC_TOKENS if token in joined]
    checks = {
        "aligned_and_misspecified_present": aligned is not None and misspecified is not None,
        "schema_matched": aligned_shape == misspecified_shape,
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


def private_world_audit(contract: Mapping[str, Any], *, world_seed: int) -> dict[str, Any]:
    task = contract["task"]
    family = electrochemical_material_family(task["electrochemical_material_family_id"])
    parameters = load_chemworld_parameters(task["world_split"], world_seed)
    pair = _moved_pair(contract["loci"]["entity"]["descriptor_permutation"])
    nominal = np.asarray([_nominal_vector(row) for row in family.electrolyte_profiles])
    scale = np.std(nominal, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    realized = np.asarray(
        [
            _realized_vector(
                family.electrolyte_profiles[index],
                parameters.electrochemical_electrolyte_effects[index],
                parameters.electrochemical_electrolyte_potential_residual_V[index],
            )
            for index in range(4)
        ]
    )
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
    truth_sha256 = electrochemical_material_instance_sha256(parameters, family)
    return {
        "world_seed": world_seed,
        "runtime_world_id": parameters.world_id,
        "truth_sha256": truth_sha256,
        "family_sha256": family.family_sha256,
        "mapping_rows": mapping_rows,
        "aligned_mapping_not_reversed": all(row["own_mapping_closer"] for row in mapping_rows),
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
    noise_coordinates = {
        (row.get("nuisance_anchor"), row.get("target_category"), row.get("replicate"))
        for row in receipts
    }
    pair = _moved_pair(entity["descriptor_permutation"])
    primary = str(entity["gate_endpoint_id"])
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(int(row["nuisance_anchor"]), int(row["target_category"]))].append(row)
    anchors = []
    for anchor in range(int(entity["nuisance_anchor_count"])):
        left = grouped[(anchor, pair[0])]
        right = grouped[(anchor, pair[1])]
        complete = len(left) == len(right) == int(entity["independent_replicates"])
        left_values = [float(row["allowed_metrics"][primary]) for row in left]
        right_values = [float(row["allowed_metrics"][primary]) for row in right]
        separation = abs(fmean(right_values) - fmean(left_values)) if complete else 0.0
        standard_error = (
            math.sqrt(
                variance(left_values) / len(left_values)
                + variance(right_values) / len(right_values)
            )
            if complete
            else None
        )
        snr = separation / max(float(standard_error or 0.0), 1.0e-12) if complete else 0.0
        passed = (
            complete
            and separation >= float(entity["minimum_absolute_separation"])
            and snr >= float(entity["minimum_signal_to_noise_ratio"])
        )
        anchors.append(
            {
                "anchor": anchor,
                "left_category": pair[0],
                "right_category": pair[1],
                "left_mean": fmean(left_values) if left_values else None,
                "right_mean": fmean(right_values) if right_values else None,
                "absolute_separation": separation,
                "welch_standard_error": standard_error,
                "signal_to_noise_ratio": snr,
                "passed": passed,
            }
        )
    prior = entity_prior_audit(contract)
    private = private_world_audit(contract, world_seed=world_seed)
    policy = entity["behavioral_policy"]
    behavioral_gap = max((row["absolute_separation"] for row in anchors), default=0.0)
    gates = {
        "Q1_world_integrity": len(receipts) == len(completed) == len(replayed) == expected,
        "Q2_task_accessibility": len({row.get("recipe_id") for row in completed}) == 8,
        "Q3_public_contract_invariance": bool(prior["checks"]["public_leakage_free"]),
        "Q4_prior_symmetry": bool(
            prior["checks"]["schema_matched"] and prior["checks"]["text_template_matched"]
        ),
        "Q5_identifiability": all(row["passed"] for row in anchors),
        "Q6_budgeted_falsifiability": int(entity["participant_unique_experiment_budget"]) >= 4,
        "Q7_behavioral_relevance": (
            private["aligned_mapping_not_reversed"]
            and behavioral_gap >= float(policy["minimum_observed_endpoint_consequence"])
        ),
        "Q8_noise_robustness": (
            len(noise_coordinates) == expected
            and all(
                row["signal_to_noise_ratio"]
                >= float(entity["minimum_signal_to_noise_ratio"])
                for row in anchors
            )
        ),
    }
    failures = [gate for gate in EXPECTED_COMMON_GATES if not gates[gate]]
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "formal_result": False,
        "provider_call_count": 0,
        "system_id": "EC",
        "world_id": world_id,
        "world_seed": world_seed,
        "prior_locus": "entity",
        "denominators": {
            "planned": expected,
            "attempted": len(receipts),
            "completed": len(completed),
            "exact_replay": len(replayed),
            "failures": len(receipts) - len(completed),
        },
        "prior_audit": {
            key: value
            for key, value in prior.items()
            if key not in {"aligned", "misspecified"}
        },
        "private_world_audit": private,
        "anchor_results": anchors,
        "gates": gates,
        "failures": failures,
        "status": "qualified" if not failures else "failed",
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def analyze_parametric_world(
    contract: Mapping[str, Any],
    *,
    world_id: str,
    world_seed: int,
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    source_truth_sha256: str,
) -> dict[str, Any]:
    """Map the frozen EC matched-prior surface into the v1.0.1 Q1-Q8 registry."""
    locus = contract["loci"]["parametric"]
    selected = analysis.get("selected_reflection")
    reflection = selected if isinstance(selected, Mapping) else {}
    reflection_checks = reflection.get("checks")
    reflection_checks = reflection_checks if isinstance(reflection_checks, Mapping) else {}
    checks = analysis.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    prior_matching = analysis.get("prior_matching")
    prior_matching = prior_matching if isinstance(prior_matching, Mapping) else {}
    leakage = analysis.get("leakage_audit")
    leakage = leakage if isinstance(leakage, Mapping) else {}
    blind = analysis.get("blind_identification")
    blind = blind if isinstance(blind, Mapping) else {}
    replayed = sum(
        isinstance(row.get("exact_replay"), Mapping)
        and row["exact_replay"].get("verified") is True
        for row in rows
    )
    classified = sum(
        row.get("status") in {"completed", "physical_failure"} for row in rows
    )
    expected = len(locus["grid_coordinates"]) ** 2
    gates = {
        "Q1_world_integrity": (
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
            and float(reflection.get("blind_error_margin", 0.0))
            >= float(locus["minimum_blind_error_margin"])
        ),
        "Q8_noise_robustness": bool(
            reflection_checks.get("held_out_disagreement")
            and float(reflection.get("disagreement_fraction", 0.0))
            >= float(locus["minimum_disagreement_fraction"])
        ),
    }
    failures = [gate for gate in EXPECTED_COMMON_GATES if not gates[gate]]
    report: dict[str, Any] = {
        "schema_version": PARAMETRIC_REPORT_VERSION,
        "formal_result": False,
        "provider_call_count": 0,
        "system_id": "EC",
        "world_id": world_id,
        "world_seed": world_seed,
        "prior_locus": "parametric",
        "source_truth_sha256": source_truth_sha256,
        "denominators": {
            "planned": expected,
            "attempted": len(rows),
            "classified": classified,
            "exact_replay": replayed,
            "platform_failures": int(analysis.get("platform_failure_count", 0)),
            "physical_failures": int(analysis.get("physical_failure_count", 0)),
        },
        "gates": gates,
        "failures": failures,
        "status": "qualified" if not failures else "failed",
        "legacy_analysis": dict(analysis),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def analyze_structural_world(
    contract: Mapping[str, Any],
    *,
    world_id: str,
    world_seed: int,
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    truth_sha256: str,
) -> dict[str, Any]:
    """Map the frozen EC transport candidate into the v1.0.1 Q1-Q8 registry."""
    locus = contract["loci"]["structural"]
    checks = analysis.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    model = analysis.get("model_qualification")
    model = model if isinstance(model, Mapping) else {}
    model_checks = model.get("checks")
    model_checks = model_checks if isinstance(model_checks, Mapping) else {}
    effects = analysis.get("effects")
    effects = effects if isinstance(effects, Mapping) else {}
    topology = effects.get("topology_signature")
    topology = topology if isinstance(topology, Mapping) else {}
    priors = analysis.get("prior_arms")
    priors = priors if isinstance(priors, Mapping) else {}
    aligned = priors.get("aligned_nominal")
    misspecified = priors.get("misindexed_nominal")
    prior_text = " ".join(
        (*_public_text(aligned), *_public_text(misspecified))
    ).lower()
    leakage = [token for token in FORBIDDEN_PUBLIC_TOKENS if token in prior_text]
    expected = int(locus["grid_levels"]) ** 2 + 3 * int(locus["validation_replicates"])
    replayed = sum(row.get("exact_replay") is True for row in rows)
    classified = sum(
        row.get("status") in {"completed", "physical_failure"} for row in rows
    )
    gates = {
        "Q1_world_integrity": bool(
            len(rows) == expected
            and classified == expected
            and replayed == expected
            and checks.get("zero_platform_failures")
        ),
        "Q2_task_accessibility": bool(
            checks.get("main_grid_count") and checks.get("complete_main_surface")
        ),
        "Q3_public_contract_invariance": not leakage,
        "Q4_prior_symmetry": bool(
            model_checks.get("prior_schema_matched")
            and model_checks.get("prior_word_count_matched")
        ),
        "Q5_identifiability": bool(
            model_checks.get("held_out_disagreement")
            and model_checks.get("blind_identification")
        ),
        "Q6_budgeted_falsifiability": bool(
            int(locus["participant_unique_experiment_budget"]) >= 4
            and model_checks.get("low_counterexample_region")
            and model_checks.get("high_counterexample_region")
        ),
        "Q7_behavioral_relevance": bool(
            checks.get("axis_b_effect") and topology.get("passed")
        ),
        "Q8_noise_robustness": bool(
            checks.get("complete_validation_surface")
            and topology.get("passed")
            and float(topology.get("value", 0.0))
            >= max(
                float(locus["effect_floor"]),
                float(locus["noise_multiplier"])
                * float(topology.get("sigma_observed", math.inf)),
            )
        ),
    }
    failures = [gate for gate in EXPECTED_COMMON_GATES if not gates[gate]]
    report: dict[str, Any] = {
        "schema_version": STRUCTURAL_REPORT_VERSION,
        "formal_result": False,
        "provider_call_count": 0,
        "system_id": "EC",
        "world_id": world_id,
        "world_seed": world_seed,
        "prior_locus": "structural",
        "truth_sha256": truth_sha256,
        "prior_leakage_tokens": leakage,
        "denominators": {
            "planned": expected,
            "attempted": len(rows),
            "classified": classified,
            "exact_replay": replayed,
            "platform_failures": sum(row.get("status") == "platform_failure" for row in rows),
            "physical_failures": sum(row.get("status") == "physical_failure" for row in rows),
        },
        "gates": gates,
        "failures": failures,
        "status": "qualified" if not failures else "failed",
        "legacy_analysis": dict(analysis),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def _moved_pair(permutation: Sequence[Any]) -> tuple[int, int]:
    values = [int(value) for value in permutation]
    moved = [index for index, source in enumerate(values) if index != source]
    if (
        len(values) != 4
        or len(moved) != 2
        or values[moved[0]] != moved[1]
        or values[moved[1]] != moved[0]
    ):
        raise Experiment1ECQualificationError("entity permutation must be one transposition")
    return moved[0], moved[1]


def _public_shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _public_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_public_shape(item) for item in value]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if value is None:
        return "null"
    return value


def _public_text(value: Any) -> tuple[str, ...]:
    rows: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            rows.extend(_public_text(value[key]))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_public_text(item))
    elif isinstance(value, str):
        rows.append(value)
    return tuple(rows)


def _nominal_vector(row: Mapping[str, float]) -> np.ndarray:
    return np.asarray(
        [
            math.log(float(row["electrolyte_conductivity_S_m"])),
            math.log(float(row["diffusivity_m2_s"])),
            math.log(float(row["diffusion_layer_thickness_m"])),
            math.log(float(row["double_layer_capacitance_F_m2"])),
            float(row["electrolyte_acid_pka"]),
            math.log(float(row["electrolyte_ksp"])),
            float(row["standard_potential_shift_V"]),
        ],
        dtype=float,
    )


def _realized_vector(
    row: Mapping[str, float],
    effects: Sequence[float],
    potential_residual: float,
) -> np.ndarray:
    values = _nominal_vector(row)
    effect = np.asarray(effects, dtype=float)
    values[0] += math.log(float(effect[0]))
    values[1] += math.log(float(effect[1]))
    values[2] -= 0.5 * math.log(float(effect[1]))
    values[3] += math.log(float(effect[2]))
    values[4] -= math.log10(float(effect[3]))
    values[5] += math.log(float(effect[4]))
    values[6] += float(potential_residual)
    return values


__all__ = [
    "CONTRACT_VERSION",
    "EXPECTED_COMMON_GATES",
    "EXPECTED_WORLD_IDS",
    "EXPECTED_WORLD_SEEDS",
    "Experiment1ECQualificationError",
    "analyze_entity_world",
    "entity_prior_audit",
    "load_contract",
    "private_world_audit",
    "validate_contract",
]
