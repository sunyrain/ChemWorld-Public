"""Development-only scientific gate for the W2-27 qualification triplet.

This module deliberately does not build or validate the formal/C2 release
preflight.  W2-27 needs the participant, provider, campaign, resource-card,
and qualification contracts; formal schedule completion and release
provenance belong to the later release freeze.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_formal import (
    EXPECTED_METHOD_QUALIFICATION_CONTRACT,
    EXPECTED_PARTICIPANT_EXECUTION_CONTRACT,
    FORMAL_ARMS,
    FORMAL_CHECKPOINT_EXPERIMENTS,
    FORMAL_SNAPSHOT_STAGES,
)
from chemworld.eval.work_ii_resource_calibration_v02 import (
    RESOURCE_CALIBRATION_ARMS,
)
from chemworld.eval.work_ii_resource_calibration_v02 import (
    build_summary as build_resource_calibration_summary,
)
from chemworld.eval.work_ii_resource_calibration_v02 import (
    pattern_key as resource_calibration_pattern_key,
)
from chemworld.eval.work_ii_resource_calibration_v02 import (
    validate_manifest as validate_resource_calibration_manifest,
)
from chemworld.eval.work_ii_task_resources import (
    materialize_task_resource_caps,
    resolve_task_resource_card,
)

LOCAL_MANIFEST_VERSION = "chemworld-work-ii-method-qualification-local-manifest-0.1"
SELECTED_CARD_RECEIPT_VERSION = (
    "chemworld-work-ii-method-qualification-selected-resource-card-0.1"
)
W2_27_RESOURCE_CARD_IDENTITY = {
    "rounds": 8,
    "locus": "A_E",
    "task_id": "electrochemical-conversion",
    "world_seed": 0,
}
_PROVIDER_FIELDS = (
    "id",
    "name",
    "base_url",
    "wire_api",
    "model",
    "reasoning_effort",
    "request_timeout_s",
    "finalization_timeout_s",
)
_EXPECTED_DATA_GENERATION_CONTRACT = {
    "world_split": "public-test",
    "objective": "balanced",
    "electrochemical_material_family_id": "nominal-prior-latent-v2",
    "electrochemical_workflow_mode": "autonomous_open_v1",
    "scoring_contract_id": "electrochemical-s0-balanced-efficiency-v2",
    "observation_noise_mode": "keyed",
    "observation_noise_namespace": "work-ii-electrochemical-prior-campaign",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("W2-27 inputs must remain inside the repository") from error


def _self_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def _selected_card_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "selected_card_receipt_sha256"
        }
    )


def build_w2_27_selected_resource_card_receipt(
    root: Path,
    resource_calibration_manifest_path: Path,
    resource_calibration_terminal_report_path: Path,
) -> dict[str, Any]:
    """Derive the one W2-27 card from immutable W2-26 terminal evidence."""

    root = root.resolve()
    manifest_path = resource_calibration_manifest_path.resolve()
    terminal_path = resource_calibration_terminal_report_path.resolve()
    manifest = _load_object(manifest_path)
    manifest_errors = validate_resource_calibration_manifest(root, manifest)
    if manifest_errors:
        raise ValueError(
            "W2-26 execution manifest is invalid: " + "; ".join(manifest_errors)
        )
    expected_key = (
        str(W2_27_RESOURCE_CARD_IDENTITY["locus"]),
        str(W2_27_RESOURCE_CARD_IDENTITY["task_id"]),
        int(W2_27_RESOURCE_CARD_IDENTITY["rounds"]),
    )
    patterns = manifest.get("patterns")
    patterns = patterns if isinstance(patterns, list) else []
    matching_patterns = [
        pattern
        for pattern in patterns
        if isinstance(pattern, Mapping)
        and resource_calibration_pattern_key(pattern) == expected_key
    ]
    if len(matching_patterns) != 1:
        raise ValueError("W2-26 manifest lacks exactly one W2-27 resource pattern")
    pattern = matching_patterns[0]
    report = _load_object(terminal_path)
    report_hash = report.get("triplet_report_sha256")
    report_payload = {
        key: value for key, value in report.items() if key != "triplet_report_sha256"
    }
    binding = pattern.get("campaign_config_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    rows = report.get("results")
    rows = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    if (
        report.get("schema_version")
        != "chemworld-work-ii-resource-calibration-triplet-0.2"
        or report_hash != canonical_json_sha256(report_payload)
        or resource_calibration_pattern_key(report) != expected_key
        or report.get("world_seed") != W2_27_RESOURCE_CARD_IDENTITY["world_seed"]
        or report.get("config_file_sha256") != binding.get("sha256")
        or report.get("manifest_sha256") != canonical_json_sha256(manifest)
        or len(rows) != 3
        or {row.get("arm") for row in rows} != set(RESOURCE_CALIBRATION_ARMS)
    ):
        raise ValueError("W2-26 W2-27 terminal triplet binding is invalid")
    partial = build_resource_calibration_summary(
        manifest,
        [report],
        source_commit=str(report.get("development_runtime_commit_observed", "")),
    )
    matching_patterns = [
        row
        for row in partial.get("pattern_summaries", [])
        if isinstance(row, Mapping)
        and resource_calibration_pattern_key(row) == expected_key
    ]
    matching_cells = [
        row
        for row in partial.get("cell_summaries", [])
        if isinstance(row, Mapping)
        and resource_calibration_pattern_key(row) == expected_key
    ]
    matching_cards = [
        row
        for row in partial.get("resource_card_proposals", [])
        if isinstance(row, Mapping)
        and resource_calibration_pattern_key(row.get("card_identity", {}))
        == expected_key
    ]
    if (
        len(matching_patterns) != 1
        or matching_patterns[0].get("triplet_passed") is not True
        or matching_patterns[0].get("platform_defect_detected") is not False
        or len(matching_cells) != 3
        or {row.get("arm") for row in matching_cells}
        != set(RESOURCE_CALIBRATION_ARMS)
        or any(
            row.get("terminal") is not True
            or row.get("calibration_passed") is not True
            or row.get("exact_replay_verified") is not True
            for row in matching_cells
        )
        or len(matching_cards) != 1
    ):
        raise ValueError("W2-26 W2-27 resource triplet is not evidence-terminal")
    card = dict(matching_cards[0])
    receipt: dict[str, Any] = {
        "schema_version": SELECTED_CARD_RECEIPT_VERSION,
        "status": "selected_card_passed",
        "formal_result": False,
        "whole_w2_26_status": partial.get("status"),
        "whole_w2_26_calibration_passed": False,
        "method_qualification_may_be_authorized_from_whole_w2_26": False,
        "resource_calibration_manifest_binding": {
            "path": _relative(root, manifest_path),
            "file_sha256": file_sha256(manifest_path),
            "canonical_json_sha256": canonical_json_sha256(manifest),
        },
        "terminal_triplet_binding": {
            "path": _relative(root, terminal_path),
            "file_sha256": file_sha256(terminal_path),
            "triplet_report_sha256": report_hash,
        },
        "selected_pattern_summary": dict(matching_patterns[0]),
        "selected_cell_summaries": [dict(row) for row in matching_cells],
        "selected_resource_card": card,
        "selected_resource_card_sha256": card.get("card_sha256"),
        "retained_method_findings": [
            json.loads(json.dumps(dict(row)))
            for row in partial.get("retained_method_findings", [])
            if isinstance(row, Mapping)
            and tuple(row.get("pattern", ())) == expected_key
        ],
    }
    receipt["selected_card_receipt_sha256"] = _selected_card_receipt_sha256(
        receipt
    )
    return receipt


def validate_w2_27_selected_resource_card_receipt(
    root: Path,
    receipt: Mapping[str, Any],
) -> list[str]:
    """Validate the selected-card receipt from its bound terminal evidence."""

    errors: list[str] = []
    if receipt.get("schema_version") != SELECTED_CARD_RECEIPT_VERSION:
        errors.append("unexpected W2-27 selected-card receipt schema")
    if receipt.get("selected_card_receipt_sha256") != _selected_card_receipt_sha256(
        receipt
    ):
        errors.append("W2-27 selected-card receipt self-hash mismatch")
    manifest_binding = receipt.get("resource_calibration_manifest_binding")
    manifest_binding = (
        manifest_binding if isinstance(manifest_binding, Mapping) else {}
    )
    terminal_binding = receipt.get("terminal_triplet_binding")
    terminal_binding = terminal_binding if isinstance(terminal_binding, Mapping) else {}
    manifest_path = root.resolve() / str(manifest_binding.get("path", ""))
    terminal_path = root.resolve() / str(terminal_binding.get("path", ""))
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != manifest_binding.get("file_sha256")
        or not terminal_path.is_file()
        or file_sha256(terminal_path) != terminal_binding.get("file_sha256")
    ):
        errors.append("W2-27 selected-card evidence binding is missing or stale")
        return errors
    try:
        rebuilt = build_w2_27_selected_resource_card_receipt(
            root, manifest_path, terminal_path
        )
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"W2-27 selected-card evidence is invalid: {error}")
        return errors
    if dict(receipt) != rebuilt:
        errors.append("W2-27 selected-card receipt differs from immutable evidence")
    return errors


def _qualification_task(design: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    qualification = design.get("method_qualification_contract")
    qualification = (
        dict(qualification) if isinstance(qualification, Mapping) else {}
    )
    task_id = str(qualification.get("qualification_task_id", ""))
    tasks = design.get("tasks")
    tasks = tasks if isinstance(tasks, list) else []
    matches = [
        dict(item)
        for item in tasks
        if isinstance(item, Mapping) and item.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise ValueError("W2-27 design lacks exactly one qualification task")
    return qualification, matches[0]


def _design_slice(design: Mapping[str, Any]) -> dict[str, Any]:
    """Return the W2-27-owned design slice, excluding unrelated C2 state."""

    _qualification, task = _qualification_task(design)
    return {
        "prior_arms": list(design.get("prior_arms", [])),
        "participant_execution_contract": dict(
            design.get("participant_execution_contract", {})
        ),
        "method_qualification_contract": dict(
            design.get("method_qualification_contract", {})
        ),
        "provider_attempt_contract": dict(design.get("provider_attempt_contract", {})),
        "qualification_task": task,
    }


def build_w2_27_runtime_config(
    root: Path,
    design_path: Path,
    resource_calibration_summary_path: Path,
) -> dict[str, Any]:
    """Materialize W2-27 caps from the exact W2-26 A_E/electrochemical/r8 card."""

    root = root.resolve()
    design = _load_object(design_path.resolve())
    qualification, task = _qualification_task(design)
    if task.get("task_id") != W2_27_RESOURCE_CARD_IDENTITY["task_id"]:
        raise ValueError("W2-27 qualification task differs from its resource-card identity")
    relative_config = task.get("campaign_config")
    source_path = (root / str(relative_config or "")).resolve()
    if not isinstance(relative_config, str) or not source_path.is_file():
        raise ValueError("W2-27 qualification campaign config is missing")
    source = _load_object(source_path)
    selected_receipt = _load_object(resource_calibration_summary_path.resolve())
    receipt_errors = validate_w2_27_selected_resource_card_receipt(
        root, selected_receipt
    )
    if receipt_errors:
        raise ValueError("W2-27 selected-card receipt failed: " + "; ".join(receipt_errors))
    summary = {
        "status": "passed",
        "calibration_passed": True,
        "resource_card_proposals": [selected_receipt["selected_resource_card"]],
    }
    source_binding = {
        "path": _relative(root, source_path),
        "sha256": file_sha256(source_path),
        "config_canonical_json_sha256": canonical_json_sha256(source),
    }
    card = resolve_task_resource_card(
        summary,
        rounds=int(W2_27_RESOURCE_CARD_IDENTITY["rounds"]),
        locus=str(W2_27_RESOURCE_CARD_IDENTITY["locus"]),
        task_id=str(W2_27_RESOURCE_CARD_IDENTITY["task_id"]),
        formal_source_config=source,
        formal_source_binding=source_binding,
    )
    identity = card.get("card_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    if identity.get("world_seed") != qualification.get("qualification_world_seed"):
        raise ValueError("W2-27 resource card uses a different qualification world seed")
    return materialize_task_resource_caps(source, card)


def build_method_qualification_local_manifest(
    root: Path,
    design_path: Path,
    runtime_config_path: Path,
) -> dict[str, Any]:
    """Build only the execution-semantic contract needed by W2-27."""

    root = root.resolve()
    design_path = design_path.resolve()
    design = _load_object(design_path)
    errors: list[str] = []

    arms = tuple(design.get("prior_arms", ()))
    if arms != FORMAL_ARMS:
        errors.append("W2-27 prior-arm triplet differs from the frozen method")

    participant = design.get("participant_execution_contract")
    participant = dict(participant) if isinstance(participant, Mapping) else {}
    if participant != EXPECTED_PARTICIPANT_EXECUTION_CONTRACT:
        errors.append("W2-27 participant execution contract drifted")

    qualification = design.get("method_qualification_contract")
    qualification = dict(qualification) if isinstance(qualification, Mapping) else {}
    if qualification != EXPECTED_METHOD_QUALIFICATION_CONTRACT:
        errors.append("W2-27 qualification contract drifted")

    attempt = design.get("provider_attempt_contract")
    attempt = dict(attempt) if isinstance(attempt, Mapping) else {}
    expected_attempt = {
        "attempt_unit": "host_codex_process_launch",
        "initial_attempts_per_cell": 1,
        "maximum_infrastructure_resume_attempts_per_cell": 1,
        "maximum_total_provider_attempts_per_cell": 2,
        "pre_action_restart_limit_within_attempt": 0,
        "any_persisted_trajectory_forbids_replacement": True,
        "retry_after_scientific_operation_forbidden": True,
    }
    if any(attempt.get(key) != value for key, value in expected_attempt.items()):
        errors.append("W2-27 provider-attempt contract drifted")

    task_id = str(qualification.get("qualification_task_id", ""))
    try:
        _qualification, task = _qualification_task(design)
    except ValueError as error:
        task = {}
        errors.append(str(error))
    relative_config = task.get("campaign_config")
    source_config_path = root / str(relative_config or "")
    config_path = runtime_config_path.resolve()
    config: dict[str, Any] = {}
    if not isinstance(relative_config, str) or not source_config_path.is_file():
        errors.append("W2-27 source qualification campaign config is missing")
    if not config_path.is_file():
        errors.append("W2-27 materialized runtime config is missing")
    else:
        config = _load_object(config_path)

    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    resources = config.get("method_resources")
    resources = resources if isinstance(resources, Mapping) else {}
    config_qualification = config.get("qualification")
    config_qualification = (
        config_qualification if isinstance(config_qualification, Mapping) else {}
    )
    execution = config.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    provider = config.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    reduced_provider = {field: provider.get(field) for field in _PROVIDER_FIELDS}

    if config.get("task_id") != task_id or config.get("episode_mode") != "campaign":
        errors.append("W2-27 campaign config has the wrong task or episode mode")
    configured_arms = config.get("prior_arms")
    configured_arms = configured_arms if isinstance(configured_arms, Mapping) else {}
    if set(configured_arms) != set(FORMAL_ARMS):
        errors.append("W2-27 campaign config does not preserve the arm triplet")
    expected_arms = {
        "opaque": {"mode": "opaque_codes"},
        "aligned_nominal": {"mode": "anonymous_nominal_properties"},
        "misindexed_nominal": {
            "mode": "anonymous_misindexed_properties",
            "target_field": task.get("target_field"),
            "descriptor_permutation": task.get("descriptor_permutation"),
        },
    }
    if dict(configured_arms) != expected_arms:
        errors.append("W2-27 participant-visible prior-arm payload drifted")
    if config.get("world_seed") != qualification.get("qualification_world_seed"):
        errors.append("W2-27 qualification world seed drifted")
    if any(
        config.get(field) != expected
        for field, expected in _EXPECTED_DATA_GENERATION_CONTRACT.items()
    ):
        errors.append("W2-27 data-generation contract drifted")
    if tuple(config.get("snapshot_stages", ())) != FORMAL_SNAPSHOT_STAGES:
        errors.append("W2-27 typed checkpoint stages drifted")
    if (
        campaign.get("complete_experiments") != 8
        or tuple(campaign.get("checkpoint_complete_experiments", ()))
        != FORMAL_CHECKPOINT_EXPERIMENTS
        or resources.get("complete_experiment_limit") != 8
        or tuple(resources.get("checkpoint_complete_experiments", ())) != (2, 4, 6, 8)
    ):
        errors.append("W2-27 experiment/checkpoint denominators drifted")
    if (
        config_qualification.get("minimum_unique_recipes") != 6
        or config_qualification.get("maximum_exact_repeats") != 2
    ):
        errors.append("W2-27 recipe-diversity qualification thresholds drifted")
    if (
        execution.get("max_concurrency") != 3
        or execution.get("within_cell_concurrency") != 1
        or execution.get("parallelization_unit") != "same_seed_prior_arm_triplet"
        or execution.get("failure_semantics")
        != "finish the in-flight seed triplet, then stop before the next world seed"
    ):
        errors.append("W2-27 triplet execution/failure semantics drifted")
    sampling = participant.get("sampling_contract")
    sampling = sampling if isinstance(sampling, Mapping) else {}
    timeouts = participant.get("timeout_contract_s")
    timeouts = timeouts if isinstance(timeouts, Mapping) else {}
    if (
        provider.get("reasoning_effort") != sampling.get("reasoning_effort")
        or provider.get("request_timeout_s") != timeouts.get("request")
        or provider.get("finalization_timeout_s") != timeouts.get("finalization")
    ):
        errors.append("W2-27 provider sampling or timeout contract drifted")
    if any(
        not isinstance(resources.get(field), int) or int(resources[field]) <= 0
        for field in (
            "operation_limit",
            "input_token_limit",
            "uncached_input_token_limit",
            "output_token_limit",
        )
    ) or not isinstance(resources.get("wall_time_limit_s"), int | float) or float(
        resources.get("wall_time_limit_s", 0.0)
    ) <= 0:
        errors.append("W2-27 participant resource caps are missing or invalid")

    config_binding = {
        "path": _relative(root, config_path) if config_path.is_file() else str(config_path),
        "sha256": file_sha256(config_path) if config_path.is_file() else None,
        "config_canonical_json_sha256": (
            canonical_json_sha256(config) if config_path.is_file() else None
        ),
        "hash_kind": "file_sha256",
    }
    card_binding = config.get("resource_calibration_card_binding")
    card_binding = card_binding if isinstance(card_binding, Mapping) else {}
    card_identity = card_binding.get("card_identity")
    card_identity = card_identity if isinstance(card_identity, Mapping) else {}
    if any(
        card_identity.get(field) != expected
        for field, expected in W2_27_RESOURCE_CARD_IDENTITY.items()
    ) or not isinstance(card_binding.get("card_sha256"), str):
        errors.append("W2-27 runtime config lacks its exact A_E resource-card binding")
    design_slice = _design_slice(design) if task else {}
    manifest: dict[str, Any] = {
        "schema_version": LOCAL_MANIFEST_VERSION,
        "status": "passed" if not errors else "failed",
        "formal_result": False,
        "formal_execution_authorized": False,
        "design_slice_binding": {
            "path": _relative(root, design_path),
            "sha256": canonical_json_sha256(design_slice),
            "hash_kind": "canonical_json_sha256:w2_27_slice",
        },
        "provider_contract": reduced_provider,
        "participant_execution_contract": participant,
        "participant_execution_contract_sha256": canonical_json_sha256(participant),
        "method_qualification_contract": qualification,
        "method_qualification_contract_sha256": canonical_json_sha256(qualification),
        "provider_attempt_contract": attempt,
        "resource_calibration_card_binding": dict(card_binding),
        "task_bindings": [
            {
                "task_id": task_id,
                "campaign_config": config_binding,
            }
        ],
        "expected_counts": {
            "participant_cells": 3,
            "provider_sessions": 3,
            "provider_attempts_initial_planned": 3,
            "provider_attempts_hard_cap": 6,
            "complete_experiments": 24,
            "belief_checkpoints": 15,
        },
        "errors": errors,
    }
    manifest["manifest_sha256"] = _self_hash(manifest)
    return manifest


def validate_method_qualification_local_manifest(
    root: Path,
    manifest: Mapping[str, Any],
) -> list[str]:
    """Validate W2-27 without consulting formal/C2 release state."""

    errors: list[str] = []
    if manifest.get("schema_version") != LOCAL_MANIFEST_VERSION:
        errors.append("unexpected W2-27 local manifest schema")
    if manifest.get("manifest_sha256") != _self_hash(manifest):
        errors.append("W2-27 local manifest self-hash mismatch")
    internal = manifest.get("errors")
    if not isinstance(internal, list):
        errors.append("W2-27 local manifest errors are malformed")
    else:
        errors.extend(str(error) for error in internal)
    if manifest.get("status") != ("failed" if internal else "passed"):
        errors.append("W2-27 local manifest status differs from its errors")
    if (
        manifest.get("formal_result") is not False
        or manifest.get("formal_execution_authorized") is not False
    ):
        errors.append("W2-27 local manifest crossed the formal boundary")
    design_binding = manifest.get("design_slice_binding")
    design_binding = design_binding if isinstance(design_binding, Mapping) else {}
    design_relative = design_binding.get("path")
    design_path = root.resolve() / str(design_relative or "")
    if (
        design_binding.get("hash_kind") != "canonical_json_sha256:w2_27_slice"
        or not isinstance(design_relative, str)
        or not design_path.is_file()
        or design_binding.get("sha256")
        != canonical_json_sha256(_design_slice(_load_object(design_path)))
    ):
        errors.append("W2-27 local design-slice binding is missing or stale")
    for field, expected in (
        ("participant_execution_contract", EXPECTED_PARTICIPANT_EXECUTION_CONTRACT),
        ("method_qualification_contract", EXPECTED_METHOD_QUALIFICATION_CONTRACT),
    ):
        if manifest.get(field) != expected:
            errors.append(f"W2-27 local manifest has a mismatched {field}")
    bindings = manifest.get("task_bindings")
    bindings = bindings if isinstance(bindings, list) else []
    campaign = bindings[0].get("campaign_config") if len(bindings) == 1 else None
    if not isinstance(campaign, Mapping):
        errors.append("W2-27 local manifest lacks its campaign config binding")
    else:
        relative = campaign.get("path")
        digest = campaign.get("sha256")
        path = root.resolve() / str(relative or "")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not path.is_file()
            or file_sha256(path) != digest
        ):
            errors.append("W2-27 local campaign config binding is missing or stale")
        elif campaign.get("config_canonical_json_sha256") != canonical_json_sha256(
            _load_object(path)
        ):
            errors.append("W2-27 local campaign canonical binding is stale")
    card_binding = manifest.get("resource_calibration_card_binding")
    card_binding = card_binding if isinstance(card_binding, Mapping) else {}
    card_identity = card_binding.get("card_identity")
    card_identity = card_identity if isinstance(card_identity, Mapping) else {}
    if any(
        card_identity.get(field) != expected
        for field, expected in W2_27_RESOURCE_CARD_IDENTITY.items()
    ) or not isinstance(card_binding.get("card_sha256"), str):
        errors.append("W2-27 local manifest lacks its exact resource-card binding")
    return errors


__all__ = [
    "LOCAL_MANIFEST_VERSION",
    "SELECTED_CARD_RECEIPT_VERSION",
    "W2_27_RESOURCE_CARD_IDENTITY",
    "build_method_qualification_local_manifest",
    "build_w2_27_runtime_config",
    "build_w2_27_selected_resource_card_receipt",
    "validate_method_qualification_local_manifest",
    "validate_w2_27_selected_resource_card_receipt",
]
