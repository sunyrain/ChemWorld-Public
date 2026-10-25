"""Full task-specific W2-26 resource-calibration contracts.

Version 0.2 replaces the retired three-representative design with one triplet for
every task/locus contract used by C2.  It is intentionally development evidence:
provider calls remain separately authorized and no scientific outcome is selected.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from chemworld.agents.interactive_codex_experiment import (
    validated_mcp_tool_failure_budget,
)
from chemworld.campaign_resources import (
    DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS,
    PROTECTED_CLOSEOUT_POLICY,
)
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    git_source_commit,
    write_json_atomic,
)
from chemworld.eval.resource_accounting import MethodResourceLimits
from chemworld.eval.work_ii_task_resources import (
    TASK_RESOURCE_CALIBRATED_CAP_FIELDS,
    build_task_resource_formula_binding,
    validate_task_resource_card,
)

MANIFEST_VERSION = "chemworld-work-ii-resource-calibration-manifest-0.2"
SUMMARY_VERSION = "chemworld-work-ii-resource-calibration-summary-0.2"
AUTHORIZATION_VERSION = "chemworld-work-ii-resource-calibration-authorization-0.2"
RUNTIME_CONFIG_ROOT = Path(
    "workstreams/flagship_tasks/reports/work-ii-w2-26-runtime-configs-v0.6"
)
DEEPSEEK_RUNTIME_CONFIG_ROOT = Path(
    "workstreams/flagship_tasks/reports/"
    "work-ii-w2-26-deepseek-runtime-configs-v0.1"
)
DEEPSEEK_PROVIDER_CONTRACT = {
    "id": "deepseek",
    "wire_api": "responses",
    "model": "deepseek-v4-flash",
    "reasoning_effort": "high",
}
DEEPSEEK_PROVIDER_RUNTIME = {
    "id": "deepseek",
    "name": "DeepSeek",
    "base_url": "https://api.deepseek.com/",
    "wire_api": "responses",
    "auth_mode": "experimental_bearer_token",
    "api_key_file": "api.md",
    "model_catalog_json": "configs/providers/deepseek_v4_flash_models.json",
    "preferred_auth_method": "apikey",
    "forced_login_method": "api",
    "model": "deepseek-v4-flash",
    "reasoning_effort": "high",
    "request_timeout_s": 1200.0,
    "finalization_timeout_s": 600.0,
    "progress_interval_s": 30.0,
    "max_recovered_mcp_tool_failures": 3,
    "max_consecutive_mcp_tool_failures": 1,
    "max_provider_error_events": 1,
}
# W2-26 measures resource use; these provider-specific values are permissive
# planning ceilings, not observed results or proposed formal caps.  They scale
# the retained 10-experiment DeepSeek production envelope (36M/600k/160k) by
# the unchanged 8/10/12 experiment denominators.
DEEPSEEK_PLANNING_RESOURCE_LIMITS = {
    8: {
        "input_token_limit": 28_800_000,
        "uncached_input_token_limit": 480_000,
        "output_token_limit": 128_000,
    },
    10: {
        "input_token_limit": 36_000_000,
        "uncached_input_token_limit": 600_000,
        "output_token_limit": 160_000,
    },
    12: {
        "input_token_limit": 43_200_000,
        "uncached_input_token_limit": 720_000,
        "output_token_limit": 192_000,
    },
}
RESOURCE_CALIBRATION_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
METHOD_RESOURCE_LIMIT_FIELDS = frozenset(
    field.name for field in fields(MethodResourceLimits)
)
CHECKPOINTS = {
    8: (0, 2, 4, 6, 8),
    10: (0, 2, 4, 7, 10),
    12: (0, 3, 6, 9, 12),
}
AE_CONFIGS = (
    ("electrochemical-conversion", "configs/benchmark/work_ii_campaign_pilot.json"),
    (
        "reaction-to-crystallization",
        "configs/benchmark/work_ii_crystallization_campaign.json",
    ),
    (
        "reaction-to-distillation",
        "configs/benchmark/work_ii_distillation_campaign.json",
    ),
    ("partition-discovery", "configs/benchmark/work_ii_partition_campaign.json"),
    ("reaction-safety-constrained", "configs/benchmark/work_ii_safety_campaign.json"),
)
AP_Q2 = {
    "reaction-safety-constrained": (
        "workstreams/flagship_tasks/reports/"
        "work-ii-reaction-safety-matched-prior-qualification-20260811.json"
    ),
    "electrochemical-conversion": (
        "workstreams/flagship_tasks/reports/"
        "work-ii-electrochemical-matched-prior-qualification-20260811.json"
    ),
}
AS_Q2 = (
    "workstreams/flagship_tasks/reports/"
    "work-ii-as-paired-law-q1-q2-five-world-20260812.json"
)
AS_CANDIDATES = {
    "partition-discovery": "partition_power_response",
    "reaction-to-crystallization": "crystallization_reversible_topology",
}
EXPECTED_PATTERN_KEYS = (
    *(("A_E", task_id, 8) for task_id, _ in AE_CONFIGS),
    *(("A_P", task_id, 10) for task_id in AP_Q2),
    *(("A_S", task_id, 12) for task_id in AS_CANDIDATES),
)
EXPECTED_DENOMINATORS = {
    "task_triplets": 9,
    "cells": 27,
    "complete_experiments": 252,
    "belief_checkpoints": 135,
    "accepted_provider_sessions": 27,
    "accepted_participant_model_calls_minimum": 27,
    "accepted_participant_model_calls_maximum": 54,
}
EXPECTED_OBSERVED_DENOMINATORS = {
    "task_triplets_started": 9,
    "task_triplets_terminal": 9,
    "cells_started": 27,
    "cells_terminal": 27,
    "complete_experiments": 252,
    "belief_checkpoints": 135,
    "provider_sessions": 27,
}
PROTECTED_RESERVE_FRACTIONS = {"A_E": 0.15, "A_P": 0.15, "A_S": 0.20}
MINIMUM_UNIQUE_RECIPES = {"A_E": 6, "A_P": 8, "A_S": 10}
AGENT_INVALID_ENFORCEMENT_POLICY = "measure_only"
PROVIDER_ERROR_ENFORCEMENT_POLICY = "measure_only"
AGENT_INVALID_CAP_HEADROOM_MULTIPLIER = 1.2


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _binding(root: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"cannot bind repository artifact: {resolved}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(resolved),
        "hash_kind": "file_sha256",
    }


def _write_runtime_config(
    root: Path,
    source: Mapping[str, Any],
    *,
    locus: str,
    task_id: str,
    rounds: int,
    runtime_config_root: Path = RUNTIME_CONFIG_ROOT,
    provider_override: Mapping[str, Any] | None = None,
    planning_resource_override: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = _materialize_runtime_config(
        source,
        locus=locus,
        task_id=task_id,
        rounds=rounds,
        provider_override=provider_override,
        planning_resource_override=planning_resource_override,
    )
    target = root / runtime_config_root / f"{locus.lower()}--{task_id}--r{rounds}.json"
    if target.is_file():
        if _load(target) != config:
            raise ValueError(f"{locus}/{task_id} runtime config already differs")
    else:
        write_json_atomic(target, config)
    return target, config


def pattern_key(pattern: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(pattern.get("locus")),
        str(pattern.get("task_id")),
        int(pattern.get("rounds", -1)),
    )


def pattern_slug(pattern: Mapping[str, Any]) -> str:
    locus, task_id, rounds = pattern_key(pattern)
    return f"{locus.lower()}--{task_id}--r{rounds}"


def _materialize_runtime_config(
    source: Mapping[str, Any],
    *,
    locus: str,
    task_id: str,
    rounds: int,
    provider_override: Mapping[str, Any] | None = None,
    planning_resource_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Activate the frozen task envelope without changing participant coverage."""

    config = copy.deepcopy(dict(source))
    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, dict) else {}
    process = campaign.get("process_time_policy")
    process = process if isinstance(process, dict) else {}
    closeout = campaign.get("closeout_policy")
    closeout = closeout if isinstance(closeout, dict) else {}
    fraction = PROTECTED_RESERVE_FRACTIONS[locus]
    required = float(process.get("required_stage_max_s", 0.0))
    repeats = float(process.get("repeat_allowance_s", 0.0))
    implicit = float(process.get("implicit_stage_reserve_s", 0.0))
    extra = float(process.get("quench_transfer_allowance_s", 0.0))
    protected = (required + repeats) * fraction
    process.update(
        {
            "protected_reserve_s": protected,
            "protected_reserve_fraction": fraction,
            "resource_status": "w2_26_runtime_envelope",
        }
    )
    campaign["process_time_limit_s"] = required + repeats + implicit + extra + protected
    closeout.update(
        {
            "policy": PROTECTED_CLOSEOUT_POLICY,
            "allowed_operation_classes": sorted(
                DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS
            ),
            "resource_status": "w2_26_runtime_envelope",
        }
    )
    campaign["process_time_policy"] = process
    campaign["closeout_policy"] = closeout
    config["campaign"] = campaign
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, dict) else {}
    qualification.update(
        {
            "minimum_unique_recipes": MINIMUM_UNIQUE_RECIPES[locus],
            "maximum_exact_repeats": 2,
            # W2-26 calibrates participant-owned multi-task operation sequences.
            # The legacy campaign runner otherwise falls back to the original
            # four-experiment electrochemistry-only [4, 5] interval, which is
            # neither part of the nine-task note nor valid for 8/10/12 rounds.
            "required_operation_counts": {},
            "resource_calibration_status": "w2_26_runtime_envelope",
            "execution_authorized": False,
            "formal_r5_authorized": False,
        }
    )
    config["qualification"] = qualification
    if provider_override is None:
        provider = config.get("provider")
        provider = provider if isinstance(provider, dict) else {}
    else:
        provider = copy.deepcopy(dict(provider_override))
    # W2-26 counts a strictly zero-action infrastructure predecessor as a
    # provider-process attempt, not as an accepted participant session.  One
    # bounded in-cell restart prevents a transient startup/IPC exit from
    # replacing three already-started calibration arms.
    provider["pre_action_restart_limit"] = 1
    # A clean Codex turn may finish after accepted operations before the
    # campaign lifecycle is terminal.  Preserve the same provider thread and
    # append at most one continuation turn.  The shared process cap permits
    # both the zero-action predecessor and that continuation when needed.
    provider["accepted_turn_continuation_limit"] = 1
    provider["provider_process_attempt_limit"] = 3
    config["provider"] = provider
    resources = config.get("method_resources")
    resources = resources if isinstance(resources, dict) else {}
    if planning_resource_override is not None:
        resources.update(copy.deepcopy(dict(planning_resource_override)))
    # One accepted campaign thread may contain the initial model turn plus one
    # bounded same-thread continuation.  The method ledger must admit both
    # turns or it would reject the first post-resume action before execution.
    resources["model_call_limit"] = 2
    # The source-level planning label is metadata, while the production runner
    # passes this mapping directly into MethodResourceLimits.
    resources.pop("resource_status", None)
    config["method_resources"] = resources
    config["w2_26_runtime_identity"] = {
        "locus": locus,
        "task_id": task_id,
        "rounds": rounds,
        "protected_closeout_reserve_fraction": fraction,
        "agent_invalid_enforcement": AGENT_INVALID_ENFORCEMENT_POLICY,
        "provider_error_enforcement": PROVIDER_ERROR_ENFORCEMENT_POLICY,
    }
    return config


def _config_errors(
    config: Mapping[str, Any], *, locus: str, task_id: str, rounds: int
) -> list[str]:
    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    resources = config.get("method_resources")
    resources = resources if isinstance(resources, Mapping) else {}
    errors: list[str] = []
    unsupported_resource_fields = sorted(
        set(resources).difference(METHOD_RESOURCE_LIMIT_FIELDS)
    )
    if unsupported_resource_fields:
        errors.append(
            f"{locus}/{task_id} method_resources has unsupported "
            "MethodResourceLimits fields: "
            + ", ".join(unsupported_resource_fields)
        )
    if config.get("task_id") != task_id:
        errors.append(f"{locus}/{task_id} config task differs")
    if isinstance(config.get("world_seed"), bool) or not isinstance(
        config.get("world_seed"), int
    ):
        errors.append(f"{locus}/{task_id} config lacks integer world_seed")
    if not isinstance(config.get("prior_arms"), Mapping) or set(
        config.get("prior_arms", {})
    ) != set(RESOURCE_CALIBRATION_ARMS):
        errors.append(f"{locus}/{task_id} config lacks the exact arm triplet")
    if (
        campaign.get("complete_experiments") != rounds
        or tuple(campaign.get("checkpoint_complete_experiments", []))
        != CHECKPOINTS[rounds]
        or resources.get("complete_experiment_limit") != rounds
    ):
        errors.append(f"{locus}/{task_id} config differs from its lifecycle")
    process = campaign.get("process_time_policy")
    process = process if isinstance(process, Mapping) else {}
    closeout = campaign.get("closeout_policy")
    closeout = closeout if isinstance(closeout, Mapping) else {}
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    provider = config.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    runtime_identity = config.get("w2_26_runtime_identity")
    runtime_identity = (
        runtime_identity if isinstance(runtime_identity, Mapping) else {}
    )
    if (
        closeout.get("policy") != PROTECTED_CLOSEOUT_POLICY
        or set(closeout.get("allowed_operation_classes", []))
        != set(DEFAULT_PROTECTED_CLOSEOUT_OPERATIONS)
        or process.get("protected_reserve_fraction")
        != PROTECTED_RESERVE_FRACTIONS[locus]
        or qualification.get("minimum_unique_recipes")
        != MINIMUM_UNIQUE_RECIPES[locus]
        or qualification.get("maximum_exact_repeats") != 2
        or qualification.get("required_operation_counts") != {}
        or provider.get("pre_action_restart_limit") != 1
        or provider.get("accepted_turn_continuation_limit") != 1
        or provider.get("provider_process_attempt_limit") != 3
        or resources.get("model_call_limit") != 2
        or runtime_identity.get("locus") != locus
        or runtime_identity.get("task_id") != task_id
        or runtime_identity.get("rounds") != rounds
        or runtime_identity.get("agent_invalid_enforcement")
        != AGENT_INVALID_ENFORCEMENT_POLICY
        or runtime_identity.get("provider_error_enforcement")
        != PROVIDER_ERROR_ENFORCEMENT_POLICY
    ):
        errors.append(f"{locus}/{task_id} lacks its enforced closeout design")
    return errors


def _selection_tasks(root: Path, locus: str) -> tuple[set[str], dict[str, str]]:
    filename = (
        "work_ii_c2_ap_selection_protocol_v0.1.json"
        if locus == "A_P"
        else "work_ii_c2_as_selection_protocol_v0.1.json"
    )
    path = root / "configs/benchmark" / filename
    protocol = _load(path)
    protocol_digest = protocol.get("protocol_sha256")
    protocol_payload = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    roster = protocol.get("candidate_roster")
    roster = roster if isinstance(roster, list) else []
    tasks = {
        str(row.get("task_id"))
        for row in roster
        if isinstance(row, Mapping) and isinstance(row.get("task_id"), str)
    }
    if (
        protocol.get("locus") != locus
        or len(tasks) != 2
        or protocol_digest != canonical_json_sha256(protocol_payload)
        or protocol.get("formal_participant_outcomes_observed_at_freeze") != 0
        or protocol.get("eligible_terminal_task_receipts_observed_at_freeze") != 0
    ):
        raise ValueError(f"{locus} protected selection protocol is invalid")
    return tasks, _binding(root, path)


def _ap_config(root: Path, task_id: str) -> tuple[Path, dict[str, Any], dict[str, str]]:
    path = root / AP_Q2[task_id]
    summary = _load(path)
    worlds = summary.get("worlds")
    worlds = worlds if isinstance(worlds, list) else []
    denominators = summary.get("denominators")
    denominators = denominators if isinstance(denominators, Mapping) else {}
    if (
        summary.get("formal_result") is not False
        or summary.get("provider_call_count") != 0
        or summary.get("task_id") != task_id
        or summary.get("qualification_passed") is not True
        or summary.get("failure_count") != 0
        or summary.get("failures") != []
        or summary.get("world_seeds") != [0, 1, 2, 3, 4]
        or len(worlds) != 5
        or any(
            not isinstance(world, Mapping)
            or world.get("world_seed") != seed
            or world.get("passed") is not True
            for seed, world in enumerate(worlds)
        )
        or denominators.get("world_count") != 5
        or denominators.get("passed_world_count") != 5
        or denominators.get("classified_surface_query_count") != 605
        or denominators.get("platform_failure_count") != 0
        or summary.get("summary_sha256")
        != canonical_json_sha256(
            {key: value for key, value in summary.items() if key != "summary_sha256"}
        )
    ):
        raise ValueError(f"{task_id} A-P Q2 summary is not fully passed")
    generated = summary.get("generated_d1_config")
    generated = generated if isinstance(generated, Mapping) else {}
    relative = generated.get("path")
    digest = generated.get("sha256")
    config_path = (root / str(relative)).resolve()
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or not config_path.is_file()
        or file_sha256(config_path) != digest
    ):
        raise ValueError(f"{task_id} A-P Q2 config binding is stale")
    return config_path, _load(config_path), _binding(root, path)


def _as_configs(
    root: Path,
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], dict[str, str]]:
    from chemworld.eval.work_ii_constitutive_structural_qualification import (
        validate_summary,
    )

    path = root / AS_Q2
    summary = _load(path)
    # The canonical A-S summary is published only by the integration bridge
    # after the source closeout has deeply validated every receipt.  Recheck
    # all canonical bindings, roster, denominators, aggregates, and pass
    # decision here without replaying the same 20,480 executions again.
    errors = validate_summary(root, summary, deep_validate_world_reports=False)
    if errors or summary.get("all_candidates_passed") is not True:
        raise ValueError("A-S Q2 summary is absent, invalid, or not fully passed")
    generated = summary.get("participant_d1_configs_generated")
    generated = generated if isinstance(generated, Mapping) else {}
    resolved: dict[str, tuple[Path, dict[str, Any]]] = {}
    for task_id, candidate_id in AS_CANDIDATES.items():
        binding = generated.get(candidate_id)
        binding = binding if isinstance(binding, Mapping) else {}
        relative = binding.get("path")
        digest = binding.get("sha256")
        config_path = (root / str(relative)).resolve()
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not config_path.is_file()
            or file_sha256(config_path) != digest
        ):
            raise ValueError(f"{task_id} A-S Q2 config binding is stale")
        resolved[task_id] = (config_path, _load(config_path))
    return resolved, _binding(root, path)


def build_execution_manifest(
    root: Path,
    protocol: Mapping[str, Any],
    *,
    runtime_config_root: Path = RUNTIME_CONFIG_ROOT,
    provider_override: Mapping[str, Any] | None = None,
    planning_resource_overrides: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve all nine task contracts without making provider calls."""

    root = root.resolve()
    errors = validate_manifest(root, protocol, allow_pending=True)
    if errors:
        raise ValueError("W2-26 protocol is invalid: " + "; ".join(errors))
    result = copy.deepcopy(dict(protocol))
    blockers: list[str] = []
    ap_tasks, ap_protocol = _selection_tasks(root, "A_P")
    as_tasks, as_protocol = _selection_tasks(root, "A_S")
    as_resolved: dict[str, tuple[Path, dict[str, Any]]] = {}
    as_q2_binding: dict[str, str] | None = None
    try:
        as_resolved, as_q2_binding = _as_configs(root)
    except (OSError, TypeError, ValueError) as error:
        blockers.append(str(error))
    for pattern in result["patterns"]:
        locus, task_id, rounds = pattern_key(pattern)
        try:
            if locus == "A_E":
                relative = dict(AE_CONFIGS)[task_id]
                config_path = root / relative
                config = _load(config_path)
                evidence = {
                    "selection_source": "formal_design_v0.2_all_tasks",
                    "formal_design_binding": _binding(
                        root, root / "configs/benchmark/work_ii_formal_design_v0.2.json"
                    ),
                }
            elif locus == "A_P":
                if task_id not in ap_tasks:
                    raise ValueError(f"{task_id} is absent from protected A-P roster")
                config_path, config, q2_binding = _ap_config(root, task_id)
                evidence = {
                    "selection_source": "protected_ap_full_roster",
                    "selection_protocol_binding": ap_protocol,
                    "q2_summary_binding": q2_binding,
                }
            else:
                if task_id not in as_tasks:
                    raise ValueError(f"{task_id} is absent from protected A-S roster")
                if task_id not in as_resolved or as_q2_binding is None:
                    raise ValueError(f"{task_id} A-S Q2 config is not yet available")
                config_path, config = as_resolved[task_id]
                evidence = {
                    "selection_source": "protected_as_full_roster",
                    "selection_protocol_binding": as_protocol,
                    "q2_summary_binding": as_q2_binding,
                }
            config_errors = _config_errors(
                _materialize_runtime_config(
                    config,
                    locus=locus,
                    task_id=task_id,
                    rounds=rounds,
                    provider_override=provider_override,
                    planning_resource_override=(
                        None
                        if planning_resource_overrides is None
                        else planning_resource_overrides[rounds]
                    ),
                ),
                locus=locus,
                task_id=task_id,
                rounds=rounds,
            )
            if config_errors:
                raise ValueError("; ".join(config_errors))
            source_config_binding = _binding(root, config_path)
            config_path, config = _write_runtime_config(
                root,
                config,
                locus=locus,
                task_id=task_id,
                rounds=rounds,
                runtime_config_root=runtime_config_root,
                provider_override=provider_override,
                planning_resource_override=(
                    None
                    if planning_resource_overrides is None
                    else planning_resource_overrides[rounds]
                ),
            )
            evidence["source_campaign_config_binding"] = source_config_binding
        except (OSError, TypeError, ValueError) as error:
            pattern["status"] = "pending_fail_closed"
            pattern["campaign_config_binding"] = None
            pattern["world_seed"] = None
            pattern["resource_formula_binding"] = None
            pattern["evidence"] = None
            blockers.append(f"{locus}/{task_id}: {error}")
            continue
        pattern.update(
            {
                "status": "resolved_authorization_blocked",
                "campaign_config_binding": _binding(root, config_path),
                "world_seed": config["world_seed"],
                "resource_formula_binding": build_task_resource_formula_binding(config),
                "evidence": evidence,
            }
        )
    result["status"] = (
        "ready_authorization_blocked" if not blockers else "not_ready_fail_closed"
    )
    result["blocking_requirements"] = blockers
    result["provider_execution_allowed"] = False
    result["provider_calls_executed"] = 0
    errors = validate_manifest(root, result, allow_pending=bool(blockers))
    if errors:
        raise ValueError("W2-26 execution manifest failed: " + "; ".join(errors))
    return result


def validate_manifest(
    root: Path, manifest: Mapping[str, Any], *, allow_pending: bool = False
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_VERSION:
        errors.append("unexpected W2-26 v0.2 manifest schema")
    if (
        manifest.get("formal_result") is not False
        or manifest.get("provider_execution_allowed") is not False
        or manifest.get("provider_calls_executed") != 0
    ):
        errors.append("W2-26 manifest crossed its execution boundary")
    patterns = manifest.get("patterns")
    patterns = patterns if isinstance(patterns, list) else []
    keys = tuple(pattern_key(row) for row in patterns if isinstance(row, Mapping))
    if keys != EXPECTED_PATTERN_KEYS:
        errors.append("W2-26 manifest lacks the exact nine task contracts")
        return errors
    resolved_count = 0
    for pattern in patterns:
        locus, task_id, rounds = pattern_key(pattern)
        if tuple(pattern.get("checkpoint_complete_experiments", [])) != CHECKPOINTS[rounds]:
            errors.append(f"{locus}/{task_id} checkpoints differ")
        binding = pattern.get("campaign_config_binding")
        if not isinstance(binding, Mapping):
            if not allow_pending:
                errors.append(f"{locus}/{task_id} config remains unresolved")
            continue
        try:
            config_path = root / str(binding["path"])
            actual_digest = (
                file_sha256(config_path)
                if binding.get("hash_kind") == "file_sha256"
                else canonical_json_sha256(_load(config_path))
                if binding.get("hash_kind") == "canonical_json_sha256"
                else None
            )
            if actual_digest != binding.get("sha256"):
                raise ValueError("stale binding")
            config = _load(config_path)
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"{locus}/{task_id} config binding invalid: {error}")
            continue
        errors.extend(_config_errors(config, locus=locus, task_id=task_id, rounds=rounds))
        provider = config.get("provider")
        provider = provider if isinstance(provider, Mapping) else {}
        manifest_provider = manifest.get("provider_contract")
        manifest_provider = (
            manifest_provider if isinstance(manifest_provider, Mapping) else {}
        )
        reduced_provider = {
            field: provider.get(field)
            for field in ("id", "wire_api", "model", "reasoning_effort")
        }
        if reduced_provider != dict(manifest_provider):
            errors.append(f"{locus}/{task_id} provider contract differs")
        if pattern.get("world_seed") != config.get("world_seed"):
            errors.append(f"{locus}/{task_id} world seed differs")
        if pattern.get("resource_formula_binding") != build_task_resource_formula_binding(
            config
        ):
            errors.append(f"{locus}/{task_id} resource formula binding differs")
        resolved_count += 1
    expected_status = (
        "ready_authorization_blocked"
        if resolved_count == len(EXPECTED_PATTERN_KEYS)
        else "not_ready_fail_closed"
    )
    if manifest.get("status") != expected_status:
        errors.append("W2-26 manifest status differs from resolved coverage")
    if not allow_pending and resolved_count != len(EXPECTED_PATTERN_KEYS):
        errors.append("W2-26 full task matrix is incomplete")
    if manifest.get("expected_denominators") != EXPECTED_DENOMINATORS:
        errors.append("W2-26 expected denominators differ")
    return errors


def authorization_sha256(authorization: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            key: value
            for key, value in authorization.items()
            if key != "authorization_sha256"
        }
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= 0.0
        and float(value) != float("inf")
    )


def build_authorization(
    root: Path,
    manifest_path: Path,
    *,
    currency_ceiling_usd: float | None,
    approved_at: str,
    pricing_source: str | None,
    pricing_observed_at: str | None,
    cache_hit_input_usd_per_million: float | None,
    cache_miss_input_usd_per_million: float | None,
    output_usd_per_million: float | None,
    unlimited_spend_authorized: bool = False,
) -> dict[str, Any]:
    """Authorize the exact nine-task development calibration once."""

    root = root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _load(manifest_path)
    errors = validate_manifest(root, manifest)
    if errors:
        raise ValueError("W2-26 manifest failed: " + "; ".join(errors))
    rates = (
        cache_hit_input_usd_per_million,
        cache_miss_input_usd_per_million,
        output_usd_per_million,
    )
    if unlimited_spend_authorized:
        if currency_ceiling_usd is not None or any(rate is not None for rate in rates):
            raise ValueError("W2-26 unlimited authorization cannot invent pricing or a ceiling")
    elif (
        not _finite_nonnegative(currency_ceiling_usd)
        or float(currency_ceiling_usd) <= 0.0
        or any(not _finite_nonnegative(rate) for rate in rates)
        or not any(float(rate) > 0.0 for rate in rates)
    ):
        raise ValueError("W2-26 pricing and currency ceiling must be finite and positive")
    if not approved_at.strip():
        raise ValueError("W2-26 authorization metadata is incomplete")
    if unlimited_spend_authorized:
        pricing_source = pricing_source or "provider_pricing_unavailable"
        pricing_observed_at = pricing_observed_at or approved_at
    elif (
        not isinstance(pricing_source, str)
        or not pricing_source.strip()
        or not isinstance(pricing_observed_at, str)
        or not pricing_observed_at.strip()
    ):
        raise ValueError("W2-26 authorization metadata is incomplete")

    contracts: list[dict[str, Any]] = []
    initial_cost = 0.0
    hard_cost = 0.0
    for pattern in manifest["patterns"]:
        binding = pattern["campaign_config_binding"]
        config = _load(root / str(binding["path"]))
        resources = config.get("method_resources")
        resources = resources if isinstance(resources, Mapping) else {}
        input_tokens = int(resources["input_token_limit"])
        uncached_tokens = int(resources["uncached_input_token_limit"])
        output_tokens = int(resources["output_token_limit"])
        if uncached_tokens > input_tokens:
            raise ValueError("W2-26 uncached-input cap exceeds the input cap")
        per_cell = (
            None
            if unlimited_spend_authorized
            else round(
                (
                    (input_tokens - uncached_tokens)
                    * float(cache_hit_input_usd_per_million)
                    + uncached_tokens * float(cache_miss_input_usd_per_million)
                    + output_tokens * float(output_usd_per_million)
                )
                / 1_000_000,
                12,
            )
        )
        initial_triplet = (
            None
            if per_cell is None
            else round(per_cell * len(RESOURCE_CALIBRATION_ARMS), 12)
        )
        resumed_triplet = (
            None if initial_triplet is None else round(initial_triplet * 2, 12)
        )
        if initial_triplet is not None and resumed_triplet is not None:
            initial_cost = round(initial_cost + initial_triplet, 12)
            hard_cost = round(hard_cost + resumed_triplet, 12)
        contracts.append(
            {
                "locus": pattern["locus"],
                "task_id": pattern["task_id"],
                "rounds": pattern["rounds"],
                "world_seed": pattern["world_seed"],
                "campaign_config_binding": copy.deepcopy(binding),
                "per_cell_token_caps": {
                    "input_tokens": input_tokens,
                    "uncached_input_tokens": uncached_tokens,
                    "output_tokens": output_tokens,
                },
                "per_cell_attempt_cost_cap_usd": per_cell,
                "initial_triplet_cost_cap_usd": initial_triplet,
                "triplet_with_infrastructure_resume_cost_cap_usd": resumed_triplet,
            }
        )
    if not unlimited_spend_authorized and float(currency_ceiling_usd) < hard_cost:
        raise ValueError(
            "W2-26 currency ceiling is below the all-attempt cap "
            f"({currency_ceiling_usd} < {hard_cost})"
        )

    authorization: dict[str, Any] = {
        "schema_version": AUTHORIZATION_VERSION,
        "status": "authorized_calibration_only",
        "formal_result": False,
        "provider_execution_allowed": True,
        "formal_execution_authorized": False,
        "manifest_binding": {
            "path": manifest_path.relative_to(root).as_posix(),
            "file_sha256": file_sha256(manifest_path),
            "canonical_json_sha256": canonical_json_sha256(manifest),
        },
        "development_runtime_commit_observed": git_source_commit(root),
        "clean_worktree_required": False,
        "whole_tree_hash_required": False,
        "approved_at": approved_at,
        "provider_contract": copy.deepcopy(manifest["provider_contract"]),
        "provider_contract_confirmed_by_user": True,
        "credential_rotation_confirmed_by_user": True,
        "pricing": {
            "source": pricing_source,
            "observed_at": pricing_observed_at,
            "unit": "usd_per_million_tokens",
            "cache_hit_input": (
                None
                if cache_hit_input_usd_per_million is None
                else float(cache_hit_input_usd_per_million)
            ),
            "cache_miss_input": (
                None
                if cache_miss_input_usd_per_million is None
                else float(cache_miss_input_usd_per_million)
            ),
            "output": (
                None
                if output_usd_per_million is None
                else float(output_usd_per_million)
            ),
            "pricing_available": not unlimited_spend_authorized,
            "pricing_unavailable_reason": (
                "provider contract exposes no attributable per-run USD price"
                if unlimited_spend_authorized
                else None
            ),
        },
        "pattern_attempt_contracts": contracts,
        "initial_schedule": {
            "provider_process_attempts": 81,
            "cost_cap_usd": None if unlimited_spend_authorized else initial_cost,
        },
        "all_infrastructure_resumes": {
            "provider_process_attempts": 162,
            "cost_cap_usd": None if unlimited_spend_authorized else hard_cost,
        },
        "currency_ceiling_usd": (
            None if currency_ceiling_usd is None else float(currency_ceiling_usd)
        ),
        "unlimited_spend_authorized": unlimited_spend_authorized,
        "runtime_enforcement": {
            "agent_invalid_enforcement": AGENT_INVALID_ENFORCEMENT_POLICY,
            "provider_error_enforcement": PROVIDER_ERROR_ENFORCEMENT_POLICY,
            "participant_payload_auto_repair": False,
            "raw_mcp_failures_retained": True,
            "per_cell_provider_process_attempt_hard_cap": 3,
            "per_triplet_infrastructure_attempt_hard_cap": 2,
            "reserve_full_token_cost_before_launch": True,
            "missing_infrastructure_only_resume": True,
            "participant_scientific_or_method_failure_retained": True,
            "platform_defect_invalidates_affected_triplet": True,
            "affected_triplet_restarts_from_first_cell": True,
        },
    }
    authorization["authorization_sha256"] = authorization_sha256(authorization)
    return authorization


def validate_authorization(
    root: Path,
    authorization: Mapping[str, Any],
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _load(manifest_path)
    errors.extend(validate_manifest(root, manifest))
    if authorization.get("schema_version") != AUTHORIZATION_VERSION:
        errors.append("unexpected W2-26 v0.2 authorization schema")
    if authorization.get("authorization_sha256") != authorization_sha256(authorization):
        errors.append("W2-26 authorization self-hash mismatch")
    binding = authorization.get("manifest_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    if (
        binding.get("path") != manifest_path.relative_to(root).as_posix()
        or binding.get("file_sha256") != file_sha256(manifest_path)
        or binding.get("canonical_json_sha256") != canonical_json_sha256(manifest)
    ):
        errors.append("W2-26 authorization manifest binding is stale")
    if (
        authorization.get("status") != "authorized_calibration_only"
        or authorization.get("formal_result") is not False
        or authorization.get("provider_execution_allowed") is not True
        or authorization.get("formal_execution_authorized") is not False
        or authorization.get("clean_worktree_required") is not False
        or authorization.get("whole_tree_hash_required") is not False
    ):
        errors.append("W2-26 authorization boundary is invalid")
    if (
        authorization.get("provider_contract") != manifest.get("provider_contract")
        or authorization.get("provider_contract_confirmed_by_user") is not True
        or authorization.get("credential_rotation_confirmed_by_user") is not True
    ):
        errors.append("W2-26 authorization provider contract is invalid")
    contracts = authorization.get("pattern_attempt_contracts")
    contracts = contracts if isinstance(contracts, list) else []
    contract_keys = tuple(
        pattern_key(row) for row in contracts if isinstance(row, Mapping)
    )
    if contract_keys != EXPECTED_PATTERN_KEYS:
        errors.append("W2-26 authorization lacks the exact nine task contracts")
    initial = authorization.get("initial_schedule")
    initial = initial if isinstance(initial, Mapping) else {}
    hard = authorization.get("all_infrastructure_resumes")
    hard = hard if isinstance(hard, Mapping) else {}
    ceiling = authorization.get("currency_ceiling_usd")
    unlimited = authorization.get("unlimited_spend_authorized") is True
    finite_cost_contract_valid = (
        _finite_nonnegative(initial.get("cost_cap_usd"))
        and _finite_nonnegative(hard.get("cost_cap_usd"))
        and _finite_nonnegative(ceiling)
        and float(ceiling) >= float(hard.get("cost_cap_usd", float("inf")))
    )
    unlimited_cost_contract_valid = (
        unlimited
        and initial.get("cost_cap_usd") is None
        and hard.get("cost_cap_usd") is None
        and ceiling is None
        and isinstance(authorization.get("pricing"), Mapping)
        and authorization["pricing"].get("pricing_available") is False
        and authorization["pricing"].get("cache_hit_input") is None
        and authorization["pricing"].get("cache_miss_input") is None
        and authorization["pricing"].get("output") is None
        and all(
            isinstance(row, Mapping)
            and row.get("per_cell_attempt_cost_cap_usd") is None
            and row.get("initial_triplet_cost_cap_usd") is None
            and row.get("triplet_with_infrastructure_resume_cost_cap_usd") is None
            for row in contracts
        )
    )
    if (
        initial.get("provider_process_attempts") != 81
        or hard.get("provider_process_attempts") != 162
        or not (finite_cost_contract_valid or unlimited_cost_contract_valid)
    ):
        errors.append("W2-26 authorization cost contract is invalid")
    runtime = authorization.get("runtime_enforcement")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    if (
        runtime.get("agent_invalid_enforcement")
        != AGENT_INVALID_ENFORCEMENT_POLICY
        or runtime.get("provider_error_enforcement")
        != PROVIDER_ERROR_ENFORCEMENT_POLICY
        or runtime.get("participant_payload_auto_repair") is not False
        or runtime.get("raw_mcp_failures_retained") is not True
        or runtime.get("per_cell_provider_process_attempt_hard_cap") != 3
        or runtime.get("per_triplet_infrastructure_attempt_hard_cap") != 2
        or any(
        runtime.get(field) is not True
        for field in (
            "reserve_full_token_cost_before_launch",
            "missing_infrastructure_only_resume",
            "participant_scientific_or_method_failure_retained",
            "platform_defect_invalidates_affected_triplet",
            "affected_triplet_restarts_from_first_cell",
        )
        )
    ):
        errors.append("W2-26 authorization runtime contract is incomplete")
    return errors


def summary_sha256(summary: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )


def empty_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "status": "not_executed",
        "formal_result": False,
        "manifest_sha256": canonical_json_sha256(manifest),
        "provider_calls_executed": 0,
        "provider_process_attempts_executed": 0,
        "expected_denominators": EXPECTED_DENOMINATORS,
        "observed_denominators": dict.fromkeys(EXPECTED_OBSERVED_DENOMINATORS, 0),
        "pattern_summaries": [],
        "cell_summaries": [],
        "all_failures": [],
        "resource_card_proposals": [],
        "calibration_passed": False,
        "method_qualification_may_be_authorized": False,
    }
    summary["summary_sha256"] = summary_sha256(summary)
    return summary


def _positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def _agent_invalid_recovery_observation(
    receipt: Mapping[str, Any],
) -> dict[str, int]:
    """Validate raw participant-invalid calls and deterministic recovery episodes."""

    raw = validated_mcp_tool_failure_budget(receipt)
    taxonomy = receipt.get("mcp_tool_failure_taxonomy")
    taxonomy = taxonomy if isinstance(taxonomy, Mapping) else {}
    if not isinstance(taxonomy.get("recovery_episode_taxonomy"), Mapping):
        raise ValueError("W2-26 receipt lacks recovery episode taxonomy")
    episode_count = int(raw["scientific_episode_count"])
    episode_maximum = int(raw["scientific_episode_maximum"])
    raw_count = int(raw["scientific_count"])
    raw_maximum = int(raw["scientific_maximum"])
    if (
        episode_count > raw_count
        or episode_maximum > episode_count
        or (raw_count == 0) != (episode_count == 0)
        or (episode_count == 0) != (episode_maximum == 0)
    ):
        raise ValueError("participant-invalid recovery episodes disagree with raw calls")
    return {
        "raw_count": raw_count,
        "raw_maximum": raw_maximum,
        "episode_count": episode_count,
        "episode_maximum": episode_maximum,
        "transport_count": int(raw["transport_count"]),
        "provider_network_count": int(raw["provider_network_count"]),
        "unclassified_count": int(raw["unclassified_count"]),
    }


def _terminal_provider_receipt(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    receipts = row.get("provider_receipts")
    receipts = (
        [item for item in receipts if isinstance(item, Mapping)]
        if isinstance(receipts, list)
        else []
    )
    terminal_receipts = [
        item
        for item in receipts
        if item.get("pre_action_retry_classification") == "terminal_accepted"
    ]
    if not terminal_receipts and len(receipts) == 1:
        # Historical single-session rows predate the explicit classification.
        terminal_receipts = receipts
    predecessor_receipts = [
        item
        for item in receipts
        if item.get("pre_action_retry_classification")
        == "eligible_zero_action_infrastructure_predecessor"
    ]
    if (
        len(terminal_receipts) != 1
        or len(predecessor_receipts) > 1
        or len(receipts) != len(terminal_receipts) + len(predecessor_receipts)
    ):
        return None
    return terminal_receipts[0]


def cell_has_platform_defect(row: Mapping[str, Any]) -> bool:
    """Return whether a W2-26 cell lacks intact execution evidence."""

    receipt = _terminal_provider_receipt(row)
    if receipt is None:
        return True
    method = row.get("method_resources")
    method = method if isinstance(method, Mapping) else {}
    qualification = row.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    checks = qualification.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    analysis = row.get("analysis")
    analysis = analysis if isinstance(analysis, Mapping) else {}
    try:
        mcp_failure_observation = _agent_invalid_recovery_observation(receipt)
    except (KeyError, TypeError, ValueError):
        return True
    return (
        method.get("provider_usage_pending") is not False
        or method.get("provider_usage_accounting_complete") is not True
        or method.get("in_flight_model_call_count") != 0
        or analysis.get("nonterminal_no_legal_actions") is True
        or int(mcp_failure_observation["transport_count"]) != 0
        or int(mcp_failure_observation["unclassified_count"]) != 0
        or any(
            checks.get(name) is not True
            for name in (
                "one_campaign_session",
                "tool_integrity",
                "exact_replay",
                "execution_audit",
            )
        )
    )


def validate_summary(
    summary: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != SUMMARY_VERSION:
        errors.append("unexpected W2-26 v0.2 summary schema")
    if summary.get("summary_sha256") != summary_sha256(summary):
        errors.append("W2-26 summary self-hash mismatch")
    if summary.get("manifest_sha256") != canonical_json_sha256(manifest):
        errors.append("W2-26 summary manifest binding differs")
    status = summary.get("status")
    if status == "not_executed":
        if (
            summary.get("provider_calls_executed") != 0
            or summary.get("cell_summaries") != []
            or summary.get("resource_card_proposals") != []
            or summary.get("calibration_passed") is not False
        ):
            errors.append("unexecuted W2-26 summary claims results")
        return errors
    if status not in {"passed", "failed", "invalidated_platform_defect"}:
        errors.append("W2-26 summary status is invalid")
        return errors
    if status != "passed":
        if (
            summary.get("calibration_passed") is not False
            or summary.get("method_qualification_may_be_authorized") is not False
            or not summary.get("all_failures")
        ):
            errors.append("failed W2-26 summary is not fail closed")
        return errors
    observed = summary.get("observed_denominators")
    observed = observed if isinstance(observed, Mapping) else {}
    participant_model_calls = observed.get("participant_model_calls")
    participant_model_calls_valid = (
        isinstance(participant_model_calls, int)
        and not isinstance(participant_model_calls, bool)
        and EXPECTED_DENOMINATORS["accepted_participant_model_calls_minimum"]
        <= participant_model_calls
        <= EXPECTED_DENOMINATORS["accepted_participant_model_calls_maximum"]
    )
    if (
        summary.get("formal_result") is not False
        or summary.get("expected_denominators") != EXPECTED_DENOMINATORS
        or any(
            observed.get(key) != value
            for key, value in EXPECTED_OBSERVED_DENOMINATORS.items()
        )
        or not participant_model_calls_valid
        or summary.get("provider_calls_executed") != 27
        or summary.get("calibration_passed") is not True
        or summary.get("method_qualification_may_be_authorized") is not True
        or summary.get("all_failures") != []
    ):
        errors.append("passed W2-26 summary denominators or boundary differ")
    patterns = summary.get("pattern_summaries")
    cells = summary.get("cell_summaries")
    cards = summary.get("resource_card_proposals")
    patterns = patterns if isinstance(patterns, list) else []
    cells = cells if isinstance(cells, list) else []
    cards = cards if isinstance(cards, list) else []
    if len(patterns) != 9 or len(cells) != 27 or len(cards) != 9:
        errors.append("passed W2-26 summary lacks exact rows")
    manifest_patterns = manifest.get("patterns")
    manifest_patterns = manifest_patterns if isinstance(manifest_patterns, list) else []
    for manifest_pattern in manifest_patterns:
        key = pattern_key(manifest_pattern)
        matching_patterns = [row for row in patterns if pattern_key(row) == key]
        matching_cards = [
            row
            for row in cards
            if isinstance(row, Mapping)
            and pattern_key(row.get("card_identity", {})) == key
        ]
        matching_cells = [
            row
            for row in cells
            if isinstance(row, Mapping) and pattern_key(row) == key
        ]
        if (
            len(matching_patterns) != 1
            or matching_patterns[0].get("triplet_passed") is not True
            or len(matching_cells) != 3
            or {row.get("arm") for row in matching_cells}
            != set(RESOURCE_CALIBRATION_ARMS)
            or any(row.get("calibration_passed") is not True for row in matching_cells)
            or len(matching_cards) != 1
        ):
            errors.append(f"W2-26 task triplet is incomplete: {key}")
            continue
        currency_accounting = matching_cards[0].get("currency_accounting")
        currency_accounting = (
            currency_accounting if isinstance(currency_accounting, Mapping) else {}
        )
        currency_unavailable = (
            currency_accounting.get("status") == "unavailable_provider_pricing"
            and currency_accounting.get("formal_currency_contract_required") is True
        )
        card_errors = validate_task_resource_card(matching_cards[0])
        if currency_unavailable:
            card_errors = [
                error
                for error in card_errors
                if error
                != "resource card lacks positive cap currency_ceiling_usd"
            ]
        if card_errors:
            errors.extend(f"{key}: {error}" for error in card_errors)
        caps = matching_cards[0].get("proposed_hard_caps")
        caps = caps if isinstance(caps, Mapping) else {}
        if any(
            not _positive(caps.get(field))
            for field in TASK_RESOURCE_CALIBRATED_CAP_FIELDS
            if field != "currency_ceiling_usd"
        ):
            errors.append(f"W2-26 task card lacks positive caps: {key}")
        if currency_unavailable:
            if caps.get("currency_ceiling_usd") is not None:
                errors.append(f"W2-26 unavailable currency was represented as a cap: {key}")
        elif not _positive(caps.get("currency_ceiling_usd")):
            errors.append(f"W2-26 task card lacks a positive currency cap: {key}")
    return errors


def build_summary(
    manifest: Mapping[str, Any],
    reports: list[Mapping[str, Any]],
    *,
    source_commit: str,
    observed_currency_usd_by_cell: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate nine immutable triplet reports into task-specific cards."""

    currency = observed_currency_usd_by_cell or {}
    pattern_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    method_findings: list[dict[str, Any]] = []
    provider_attempts = 0
    provider_sessions = 0
    for pattern in manifest.get("patterns", []):
        if not isinstance(pattern, Mapping):
            continue
        key = pattern_key(pattern)
        binding = pattern.get("campaign_config_binding")
        binding = binding if isinstance(binding, Mapping) else {}
        matches = [
            report
            for report in reports
            if pattern_key(report) == key
            and report.get("config_file_sha256") == binding.get("sha256")
        ]
        report = matches[0] if len(matches) == 1 else {}
        raw_rows = report.get("results")
        raw_rows = raw_rows if isinstance(raw_rows, list) else []
        platform_defect = len(matches) != 1 or len(raw_rows) != 3
        maxima: dict[str, float | None] = {
            "operation_attempts": 0.0,
            "exact_repeat_count": 0.0,
            "process_time_used_s": 0.0,
            "input_tokens": 0.0,
            "uncached_input_tokens": 0.0,
            "output_tokens": 0.0,
            "provider_elapsed_s": 0.0,
            "observed_currency_usd": None,
            "agent_invalid_recovered_count": 0.0,
            "agent_invalid_maximum_consecutive_count": 0.0,
            "agent_invalid_recovery_episode_count": 0.0,
            "agent_invalid_maximum_consecutive_recovery_episode_count": 0.0,
        }
        campaign_contract = report.get("calibration_campaign_contract")
        campaign_contract = (
            campaign_contract if isinstance(campaign_contract, Mapping) else {}
        )
        if not campaign_contract and raw_rows:
            first_contract = raw_rows[0].get("calibration_campaign_contract")
            campaign_contract = (
                first_contract if isinstance(first_contract, Mapping) else {}
            )
        process_policy = campaign_contract.get("process_time_policy")
        process_policy = process_policy if isinstance(process_policy, Mapping) else {}
        closeout_policy = campaign_contract.get("closeout_policy")
        closeout_policy = closeout_policy if isinstance(closeout_policy, Mapping) else {}
        task_cells: list[dict[str, Any]] = []
        for arm in RESOURCE_CALIBRATION_ARMS:
            arm_rows = [
                row
                for row in raw_rows
                if isinstance(row, Mapping) and row.get("arm") == arm
            ]
            if len(arm_rows) != 1:
                platform_defect = True
                failures.append({"pattern": key, "arm": arm, "class": "platform_execution_failure"})
                continue
            row = arm_rows[0]
            analysis = row.get("analysis")
            analysis = analysis if isinstance(analysis, Mapping) else {}
            qualification = row.get("qualification")
            qualification = qualification if isinstance(qualification, Mapping) else {}
            checks = qualification.get("checks")
            checks = checks if isinstance(checks, Mapping) else {}
            replay = row.get("exact_replay")
            replay = replay if isinstance(replay, Mapping) else {}
            method = row.get("method_resources")
            method = method if isinstance(method, Mapping) else {}
            receipts = row.get("provider_receipts")
            receipts = (
                [item for item in receipts if isinstance(item, Mapping)]
                if isinstance(receipts, list)
                else []
            )
            terminal_receipt = _terminal_provider_receipt(row)
            receipt = terminal_receipt or {}
            try:
                mcp_failure_observation = _agent_invalid_recovery_observation(receipt)
            except (KeyError, TypeError, ValueError):
                mcp_failure_observation = None
            final_resources = analysis.get("final_campaign_resources")
            final_resources = final_resources if isinstance(final_resources, Mapping) else {}
            state = final_resources.get("state")
            state = state if isinstance(state, Mapping) else {}
            report_only = state.get("report_only")
            report_only = report_only if isinstance(report_only, Mapping) else {}
            process_profile = analysis.get("process_profile")
            process_profile = process_profile if isinstance(process_profile, Mapping) else {}
            counts = process_profile.get("counts")
            counts = counts if isinstance(counts, Mapping) else {}
            platform_failure = cell_has_platform_defect(row)
            complete = int(analysis.get("complete_experiment_count", 0))
            resource_checks = (
                "planned_complete_experiments",
                "typed_belief_checkpoints_complete",
                "one_campaign_session",
                "final_recommendation_committed",
                "tool_integrity",
                "campaign_terminal",
                "process_time_reconciled",
                "task_required_operations_reconciled",
                "exact_replay",
                "execution_audit",
            )
            resource_calibrated = (
                not platform_failure
                and (
                    qualification.get("passed") is True
                    or all(checks.get(name) is True for name in resource_checks)
                )
                and replay.get("verified") is True
                and complete == key[2]
            )
            method_qualification_passed = (
                row.get("completed") is True
                and qualification.get("passed") is True
                and receipt.get("provider_error_event_count", 0) == 0
            )
            provider_attempt_count = int(
                method.get("provider_process_attempt_count", len(receipts))
            )
            accepted_provider_session_count = int(
                method.get("accepted_provider_session_count", 0)
            )
            accepted_model_call_count = int(
                method.get("accepted_participant_model_call_count", 0)
            )
            provider_attempts += provider_attempt_count
            provider_sessions += accepted_provider_session_count
            cell_id = f"{key[0]}:{key[1]}:{key[2]}:{pattern.get('world_seed')}:{arm}"
            observed_currency = currency.get(cell_id)
            cell = {
                "locus": key[0],
                "task_id": key[1],
                "rounds": key[2],
                "world_seed": pattern.get("world_seed"),
                "arm": arm,
                "status": (
                    "passed"
                    if resource_calibrated and method_qualification_passed
                    else "resource_calibrated_method_failure_retained"
                    if resource_calibrated
                    else "platform_defect"
                    if platform_failure
                    else "resource_calibration_incomplete"
                ),
                "terminal": not platform_failure,
                "calibration_passed": resource_calibrated,
                "method_qualification_passed": method_qualification_passed,
                "method_qualification_failed_checks": list(
                    qualification.get("failed_checks", [])
                ),
                "complete_experiments": complete,
                "checkpoint_count": len(analysis.get("belief_snapshots", [])),
                "exact_replay_verified": replay.get("verified") is True,
                "operation_attempts": int(
                    analysis.get(
                        "operation_attempt_count",
                        counts.get("participant_operation_attempt_count", 0),
                    )
                ),
                "exact_repeat_count": int(analysis.get("exact_repeat_count", 0)),
                "process_time_used_s": float(report_only.get("process_time_s", 0.0)),
                "input_tokens": int(method.get("input_token_count", 0)),
                "uncached_input_tokens": int(method.get("uncached_input_token_count", 0)),
                "output_tokens": int(method.get("output_token_count", 0)),
                "provider_elapsed_s": float(
                    method.get(
                        "session_elapsed_s",
                        sum(float(item.get("session_elapsed_s", 0.0)) for item in receipts),
                    )
                ),
                "provider_attempts": provider_attempt_count,
                "accepted_provider_sessions": accepted_provider_session_count,
                "accepted_model_calls": accepted_model_call_count,
                "agent_invalid_recovered_count": (
                    0
                    if mcp_failure_observation is None
                    else int(mcp_failure_observation["raw_count"])
                ),
                "agent_invalid_maximum_consecutive_count": (
                    0
                    if mcp_failure_observation is None
                    else int(mcp_failure_observation["raw_maximum"])
                ),
                "agent_invalid_recovery_episode_count": (
                    0
                    if mcp_failure_observation is None
                    else int(mcp_failure_observation["episode_count"])
                ),
                "agent_invalid_maximum_consecutive_recovery_episode_count": (
                    0
                    if mcp_failure_observation is None
                    else int(mcp_failure_observation["episode_maximum"])
                ),
                "observed_currency_usd": (
                    None if observed_currency is None else float(observed_currency)
                ),
            }
            task_cells.append(cell)
            cell_rows.append(cell)
            for field in maxima:
                if field == "observed_currency_usd":
                    continue
                maxima[field] = max(float(maxima[field] or 0.0), float(cell[field]))
            if not resource_calibrated:
                platform_defect = platform_defect or platform_failure
                failures.append(
                    {
                        "pattern": key,
                        "arm": arm,
                        "class": (
                            "platform_execution_failure"
                            if platform_failure
                            else "resource_calibration_incomplete"
                        ),
                    }
                )
            elif not method_qualification_passed:
                method_findings.append(
                    {
                        "pattern": key,
                        "arm": arm,
                        "class": "participant_method_failure_retained",
                        "failed_checks": list(qualification.get("failed_checks", [])),
                    }
                )
        triplet_passed = (
            not platform_defect
            and len(task_cells) == 3
            and all(cell["calibration_passed"] for cell in task_cells)
        )
        pattern_rows.append(
            {
                "locus": key[0],
                "task_id": key[1],
                "rounds": key[2],
                "world_seed": pattern.get("world_seed"),
                "cell_count": len(task_cells),
                "triplet_passed": triplet_passed,
                "platform_defect_detected": platform_defect,
            }
        )
        task_currency = [
            float(cell["observed_currency_usd"])
            for cell in task_cells
            if cell["observed_currency_usd"] is not None
        ]
        currency_available = len(task_currency) == len(task_cells) == 3
        maxima["observed_currency_usd"] = (
            max(task_currency) if currency_available else None
        )
        caps = {
            "operation_attempt_limit": int(maxima["operation_attempts"]),
            "protected_closeout_operation_reserve": int(
                closeout_policy.get("final_assay_path_total_operation_reserve", 0)
            ),
            "process_time_limit_s": maxima["process_time_used_s"]
            + float(process_policy.get("protected_reserve_s", 0.0)),
            "protected_closeout_reserve_s": float(
                process_policy.get("protected_reserve_s", 0.0)
            ),
            "input_token_limit": int(maxima["input_tokens"]),
            "uncached_input_token_limit": int(maxima["uncached_input_tokens"]),
            "output_token_limit": int(maxima["output_tokens"]),
            "provider_wall_time_limit_s": maxima["provider_elapsed_s"],
            "currency_ceiling_usd": maxima["observed_currency_usd"],
            "max_recovered_mcp_tool_failures": max(
                1,
                math.ceil(
                    float(maxima["agent_invalid_recovery_episode_count"])
                    * AGENT_INVALID_CAP_HEADROOM_MULTIPLIER
                ),
            ),
            "max_consecutive_mcp_tool_failures": max(
                1,
                math.ceil(
                    float(
                        maxima[
                            "agent_invalid_maximum_consecutive_recovery_episode_count"
                        ]
                    )
                    * AGENT_INVALID_CAP_HEADROOM_MULTIPLIER
                ),
            ),
        }
        card: dict[str, Any] = {
            "card_identity": {
                "locus": key[0],
                "task_id": key[1],
                "rounds": key[2],
                "world_seed": pattern.get("world_seed"),
                "calibration_campaign_binding": copy.deepcopy(binding),
                "resource_formula_binding": copy.deepcopy(
                    pattern.get("resource_formula_binding")
                ),
            },
            "observed_maxima": maxima,
            "currency_accounting": {
                "status": (
                    "observed_attributable_usd"
                    if currency_available
                    else "unavailable_provider_pricing"
                ),
                "observed_cell_count": len(task_currency),
                "expected_cell_count": 3,
                "formal_currency_contract_required": not currency_available,
            },
            "protected_closeout_reserve_enforced": True,
            "proposed_hard_caps": caps,
        }
        card["card_sha256"] = canonical_json_sha256(card)
        if triplet_passed:
            cards.append(card)
    platform_defect = any(row["platform_defect_detected"] for row in pattern_rows)
    passed = (
        len(pattern_rows) == 9
        and len(cell_rows) == 27
        and not platform_defect
        and all(row["triplet_passed"] for row in pattern_rows)
    )
    observed = {
        "task_triplets_started": len(pattern_rows),
        "task_triplets_terminal": sum(not row["platform_defect_detected"] for row in pattern_rows),
        "cells_started": len(cell_rows),
        "cells_terminal": sum(row["terminal"] for row in cell_rows),
        "complete_experiments": sum(row["complete_experiments"] for row in cell_rows),
        "belief_checkpoints": sum(row["checkpoint_count"] for row in cell_rows),
        "provider_sessions": sum(row["accepted_provider_sessions"] for row in cell_rows),
        "participant_model_calls": sum(row["accepted_model_calls"] for row in cell_rows),
    }
    currency_cells_observed = sum(
        row.get("observed_currency_usd") is not None for row in cell_rows
    )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "status": (
            "invalidated_platform_defect"
            if platform_defect
            else "passed"
            if passed
            else "failed"
        ),
        "formal_result": False,
        "manifest_sha256": canonical_json_sha256(manifest),
        "source_commit": source_commit,
        "provider_calls_executed": provider_sessions,
        "provider_process_attempts_executed": provider_attempts,
        "currency_accounting": {
            "status": (
                "observed_attributable_usd"
                if currency_cells_observed == len(cell_rows) == 27
                else "unavailable_provider_pricing"
            ),
            "observed_cell_count": currency_cells_observed,
            "expected_cell_count": 27,
            "formal_currency_contract_required": currency_cells_observed != 27,
        },
        "expected_denominators": EXPECTED_DENOMINATORS,
        "observed_denominators": observed,
        "pattern_summaries": pattern_rows,
        "cell_summaries": cell_rows,
        "all_failures": failures,
        "retained_method_findings": method_findings,
        "resource_card_proposals": cards,
        "calibration_passed": passed,
        "method_qualification_may_be_authorized": passed,
    }
    summary["summary_sha256"] = summary_sha256(summary)
    return summary


__all__ = [
    "AE_CONFIGS",
    "AGENT_INVALID_CAP_HEADROOM_MULTIPLIER",
    "AGENT_INVALID_ENFORCEMENT_POLICY",
    "AP_Q2",
    "AS_CANDIDATES",
    "AS_Q2",
    "AUTHORIZATION_VERSION",
    "CHECKPOINTS",
    "EXPECTED_DENOMINATORS",
    "EXPECTED_OBSERVED_DENOMINATORS",
    "EXPECTED_PATTERN_KEYS",
    "MANIFEST_VERSION",
    "PROVIDER_ERROR_ENFORCEMENT_POLICY",
    "RESOURCE_CALIBRATION_ARMS",
    "SUMMARY_VERSION",
    "authorization_sha256",
    "build_authorization",
    "build_execution_manifest",
    "build_summary",
    "cell_has_platform_defect",
    "empty_summary",
    "pattern_key",
    "pattern_slug",
    "summary_sha256",
    "validate_authorization",
    "validate_manifest",
    "validate_summary",
]
