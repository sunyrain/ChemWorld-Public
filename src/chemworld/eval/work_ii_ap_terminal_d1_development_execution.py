"""Build provider-blocked development execution configs for terminal A-P D1."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.eval.work_ii_ap_terminal_d1_readiness import (
    build_independent_ap_d1_readiness,
)
from chemworld.eval.work_ii_d1_execution import D1_EXECUTION_CONTRACT

AP_D1_READINESS_VERSION = "chemworld-work-ii-ap-independent-d1-readiness-0.1"
AP_D1_READY_STATUS = "ready_static_config_provider_execution_blocked"
AP_D1_WORLD_SEED = 2
AP_D1_EXPERIMENTS = 10
AP_D1_CHECKPOINTS = [0, 2, 4, 7, 10]
AP_D1_PLAN = Path(
    "configs/benchmark/work_ii_ap_terminal_d1_independent_plan_v0.1.json"
)
AP_D1_TASK_SPECS = {
    "reaction-safety-constrained": {
        "source": (
            "configs/benchmark/"
            "work_ii_reaction_safety_independent_terminal_d1_seed2.json"
        ),
        "output": (
            "configs/benchmark/"
            "work_ii_reaction_safety_independent_terminal_d1_execution_seed2.json"
        ),
        "pilot_id": (
            "work-ii-reaction-safety-independent-terminal-d1-execution-seed2"
        ),
    },
    "electrochemical-conversion": {
        "source": (
            "configs/benchmark/"
            "work_ii_electrochemical_independent_terminal_d1_seed2.json"
        ),
        "output": (
            "configs/benchmark/"
            "work_ii_electrochemical_independent_terminal_d1_execution_seed2.json"
        ),
        "pilot_id": (
            "work-ii-electrochemical-independent-terminal-d1-execution-seed2"
        ),
    },
}
AP_D1_PROVIDER_SPECS = {
    "wellau": {
        "outputs": {
            task_id: spec["output"] for task_id, spec in AP_D1_TASK_SPECS.items()
        },
        "pilot_suffix": "",
        "provider": None,
        "method_resources": {
            # v0.11 stages every required belief checkpoint into seven small pages.
            # The first real six-cell qualification observed 1,983,667 cumulative
            # input tokens and 226,483 uncached tokens in a fully completed cell.
            # Keep the frozen 20% development headroom above those prospective
            # maxima; this changes only the method-resource envelope, never the
            # scientific campaign/checkpoint denominator.
            "input_token_limit": 2_380_401,
            "uncached_input_token_limit": 271_780,
            "output_token_limit": 20_000,
            "wall_time_limit_s": 7_200.0,
        },
    },
    "deepseek": {
        "outputs": {
            "reaction-safety-constrained": (
                "configs/benchmark/"
                "work_ii_reaction_safety_independent_terminal_d1_execution_seed2_"
                "deepseek_v4_flash.json"
            ),
            "electrochemical-conversion": (
                "configs/benchmark/"
                "work_ii_electrochemical_independent_terminal_d1_execution_seed2_"
                "deepseek_v4_flash.json"
            ),
        },
        "pilot_suffix": "-deepseek-v4-flash",
        "provider": {
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
            "request_timeout_s": 1_200.0,
            "finalization_timeout_s": 600.0,
            "session_wall_time_limit_s": 6_600.0,
            "max_recovered_mcp_tool_failures": 3,
            "max_consecutive_mcp_tool_failures": 1,
            "max_provider_error_events": 1,
            "progress_interval_s": 30.0,
            "pre_action_restart_limit": 0,
        },
        "method_resources": {
            "input_token_limit": 36_000_000,
            "uncached_input_token_limit": 600_000,
            "output_token_limit": 160_000,
            "wall_time_limit_s": 7_200.0,
        },
    },
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _inside(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = (root.resolve() / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} path escapes the repository")
    return path


def _ready_rows(readiness: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tasks = readiness.get("tasks")
    if (
        readiness.get("schema_version") != AP_D1_READINESS_VERSION
        or readiness.get("formal_result") is not False
        or readiness.get("provider_call_count") != 0
        or readiness.get("provider_execution_authorized") is not False
        or readiness.get("status") != "ready"
        or not isinstance(tasks, list)
    ):
        raise ValueError("W2-38 readiness is not a provider-free ready result")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in tasks:
        if not isinstance(row, Mapping):
            raise ValueError("W2-38 readiness contains a malformed task row")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id in rows:
            raise ValueError("W2-38 readiness task rows are missing or duplicated")
        rows[task_id] = row
    if set(rows) != set(AP_D1_TASK_SPECS):
        raise ValueError("W2-38 readiness must contain exactly the two A-P tasks")
    return rows


def _validate_ready_row(task_id: str, row: Mapping[str, Any]) -> None:
    expected_source = AP_D1_TASK_SPECS[task_id]["source"]
    q2_passed = row.get("q2_passed_world_seeds")
    eligible = row.get("eligible_unexposed_q2_passed_world_seeds")
    exposed = row.get("historical_participant_exposed_world_seeds")
    blockers = row.get("blockers")
    if (
        row.get("status") != AP_D1_READY_STATUS
        or row.get("selected_world_seed") != AP_D1_WORLD_SEED
        or row.get("selection_rule_satisfied") is not True
        or row.get("provider_execution_authorized") is not False
        or row.get("historical_participant_results_replaced") is not False
        or row.get("output_config") != expected_source
        or not isinstance(q2_passed, list)
        or AP_D1_WORLD_SEED not in q2_passed
        or not isinstance(eligible, list)
        or not eligible
        or min(eligible) != AP_D1_WORLD_SEED
        or not isinstance(exposed, list)
        or AP_D1_WORLD_SEED in exposed
        or blockers != []
    ):
        raise ValueError(f"W2-38 readiness row is not the exact ready seed2 row: {task_id}")


def _validate_static_config(
    config: Mapping[str, Any],
    *,
    task_id: str,
    source_path: str,
) -> None:
    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    independent = config.get("independent_terminal_d1")
    independent = independent if isinstance(independent, Mapping) else {}
    if (
        config.get("task_id") != task_id
        or config.get("world_seed") != AP_D1_WORLD_SEED
        or config.get("formal_result") is not False
        or campaign.get("complete_experiments") != AP_D1_EXPERIMENTS
        or campaign.get("checkpoint_complete_experiments") != AP_D1_CHECKPOINTS
        or qualification.get("q2_passed") is not True
        or qualification.get("execution_authorized") is not False
        or qualification.get("formal_r5_authorized") is not False
        or independent.get("readiness_only") is not True
        or independent.get("provider_execution_authorized") is not False
        or independent.get("historical_participant_results_replaced") is not False
    ):
        raise ValueError(
            "static A-P D1 config is not an eligible seed2 readiness config: "
            + source_path
        )
    forbidden_release_fields = {
        "execution_context",
        "legacy_source_evidence",
        "qualification_evidence",
        "release_manifest",
    }
    if forbidden_release_fields.intersection(config):
        raise ValueError(f"static A-P D1 config crosses the development boundary: {source_path}")


def _build_one(
    source: Mapping[str, Any],
    *,
    task_id: str,
    source_path: str,
    provider_id: str,
) -> dict[str, Any]:
    _validate_static_config(source, task_id=task_id, source_path=source_path)
    if provider_id not in AP_D1_PROVIDER_SPECS:
        raise ValueError(f"unsupported A-P D1 development provider: {provider_id}")
    provider_spec = AP_D1_PROVIDER_SPECS[provider_id]
    config = copy.deepcopy(dict(source))
    pilot_id = str(AP_D1_TASK_SPECS[task_id]["pilot_id"]) + str(
        provider_spec["pilot_suffix"]
    )
    config["pilot_id"] = pilot_id
    config["observation_noise_namespace"] = pilot_id
    config["execution"].update(
        {
            "failure_semantics": (
                "retain cell failures and continue every scheduled seed triplet"
            ),
            "systemic_failure_semantics": (
                "stop only when all three arms fail before the first committed operation"
            ),
            "pilot_expansion_headroom_fraction": 0.20,
            "d1_execution_contract": copy.deepcopy(D1_EXECUTION_CONTRACT),
        }
    )
    provider = provider_spec["provider"]
    if provider is None:
        config["provider"].update(
            {
                "session_wall_time_limit_s": 6_600.0,
                # The first staged WellAU block observed three independent
                # feedback/recovery episodes and at most two consecutively.
                # Preserve the raw eleven invalid calls in receipts, but size
                # the provisional qualification caps with the frozen 20%
                # episode headroom rule used by W2-26.
                "max_recovered_mcp_tool_failures": 4,
                "max_consecutive_mcp_tool_failures": 3,
                "max_provider_error_events": 1,
                "progress_interval_s": 30.0,
                "pre_action_restart_limit": 0,
            }
        )
    else:
        config["provider"] = copy.deepcopy(provider)
    config["method_resources"].pop("resource_status", None)
    config["method_resources"].update(copy.deepcopy(provider_spec["method_resources"]))
    config["qualification"].update(
        {
            "max_resource_rejections": 1,
            "execution_authorized": False,
            "formal_r5_authorized": False,
        }
    )
    config["independent_terminal_d1"].update(
        {
            "source_static_config_path": source_path,
            "readiness_status": AP_D1_READY_STATUS,
        }
    )
    errors = validate_development_execution_config(
        config,
        task_id=task_id,
        source_path=source_path,
        provider_id=provider_id,
    )
    if errors:
        raise ValueError("invalid A-P D1 development execution config: " + "; ".join(errors))
    return config


def validate_development_execution_config(
    config: Mapping[str, Any],
    *,
    task_id: str,
    source_path: str,
    provider_id: str | None = None,
) -> list[str]:
    """Validate the execution-only additions without requiring release evidence."""

    errors: list[str] = []
    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    execution = config.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    provider = config.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    resources = config.get("method_resources")
    resources = resources if isinstance(resources, Mapping) else {}
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    independent = config.get("independent_terminal_d1")
    independent = independent if isinstance(independent, Mapping) else {}
    provider_id = provider_id or str(provider.get("id", ""))
    provider_spec = AP_D1_PROVIDER_SPECS.get(provider_id)
    if provider_spec is None:
        errors.append("development config uses an unsupported provider")
        provider_spec = AP_D1_PROVIDER_SPECS["wellau"]
    expected_pilot_id = str(AP_D1_TASK_SPECS[task_id]["pilot_id"]) + str(
        provider_spec["pilot_suffix"]
    )
    if (
        config.get("task_id") != task_id
        or config.get("world_seed") != AP_D1_WORLD_SEED
        or config.get("pilot_id") != expected_pilot_id
        or config.get("observation_noise_namespace") != expected_pilot_id
    ):
        errors.append("task, seed, or execution namespace mismatch")
    if config.get("formal_result") is not False:
        errors.append("development config crossed the formal-result boundary")
    if (
        campaign.get("complete_experiments") != AP_D1_EXPERIMENTS
        or campaign.get("checkpoint_complete_experiments") != AP_D1_CHECKPOINTS
    ):
        errors.append("development config does not preserve the ten-round campaign")
    if (
        execution.get("failure_semantics")
        != "retain cell failures and continue every scheduled seed triplet"
        or execution.get("systemic_failure_semantics")
        != "stop only when all three arms fail before the first committed operation"
        or execution.get("pilot_expansion_headroom_fraction") != 0.20
        or execution.get("d1_execution_contract") != D1_EXECUTION_CONTRACT
    ):
        errors.append("D1 failure, headroom, or immutable execution contract is incomplete")
    expected_provider = {
        "session_wall_time_limit_s": 6_600.0,
        "max_recovered_mcp_tool_failures": 4 if provider_id == "wellau" else 3,
        "max_consecutive_mcp_tool_failures": 3 if provider_id == "wellau" else 1,
        "max_provider_error_events": 1,
        "progress_interval_s": 30.0,
        "pre_action_restart_limit": 0,
    }
    if any(provider.get(key) != value for key, value in expected_provider.items()):
        errors.append("provider recovery or progress limits are incomplete")
    frozen_provider = provider_spec["provider"]
    if frozen_provider is not None and dict(provider) != frozen_provider:
        errors.append("development config differs from its frozen provider contract")
    expected_resources = {
        "complete_experiment_limit": AP_D1_EXPERIMENTS,
        **dict(provider_spec["method_resources"]),
    }
    if (
        any(resources.get(key) != value for key, value in expected_resources.items())
        or "resource_status" in resources
    ):
        errors.append("ten-round resource limits are incomplete or retain resource_status")
    if (
        qualification.get("q2_passed") is not True
        or qualification.get("max_resource_rejections") != 1
        or qualification.get("execution_authorized") is not False
        or qualification.get("formal_r5_authorized") is not False
    ):
        errors.append("development execution remains unauthorized or qualification is incomplete")
    if (
        independent.get("readiness_only") is not True
        or independent.get("provider_execution_authorized") is not False
        or independent.get("historical_participant_results_replaced") is not False
        or independent.get("source_static_config_path") != source_path
        or independent.get("readiness_status") != AP_D1_READY_STATUS
    ):
        errors.append("independent terminal readiness metadata is incomplete")
    if any(
        field in config
        for field in (
            "execution_context",
            "legacy_source_evidence",
            "qualification_evidence",
            "release_manifest",
            "development_resource_basis",
        )
    ):
        errors.append("development config contains release or audit-chain fields")
    return errors


def build_ap_d1_development_execution_configs(
    root: Path,
    readiness_path: Path,
    *,
    provider_id: str = "wellau",
) -> dict[str, dict[str, Any]]:
    """Build both exact seed2 configs for one provider from W2-38 readiness."""

    root = root.resolve()
    readiness = _load(readiness_path.resolve())
    if provider_id not in AP_D1_PROVIDER_SPECS:
        raise ValueError(f"unsupported A-P D1 development provider: {provider_id}")
    rows = _ready_rows(readiness)
    for task_id, row in rows.items():
        _validate_ready_row(task_id, row)
    expected_readiness, expected_static_configs = build_independent_ap_d1_readiness(
        root, root / AP_D1_PLAN
    )
    if readiness != expected_readiness:
        raise ValueError("W2-38 readiness differs from its deterministic rebuild")
    configs: dict[str, dict[str, Any]] = {}
    for task_id, spec in AP_D1_TASK_SPECS.items():
        row = rows[task_id]
        source_path = str(spec["source"])
        source = _load(_inside(root, source_path, label="static A-P D1 config"))
        if source != expected_static_configs.get(task_id):
            raise ValueError(
                "static A-P D1 scientific config differs from its deterministic "
                f"readiness rebuild: {task_id}"
            )
        configs[task_id] = _build_one(
            source,
            task_id=task_id,
            source_path=source_path,
            provider_id=provider_id,
        )
    return configs


def build_all_ap_d1_development_execution_configs(
    root: Path,
    readiness_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the complete task-by-provider development execution matrix."""

    return {
        provider_id: build_ap_d1_development_execution_configs(
            root,
            readiness_path,
            provider_id=provider_id,
        )
        for provider_id in AP_D1_PROVIDER_SPECS
    }


def validate_ap_d1_development_execution_configs(
    root: Path,
    readiness_path: Path,
    configs: Mapping[str, Mapping[str, Any]],
    *,
    provider_id: str = "wellau",
) -> list[str]:
    """Compare candidate configs with a deterministic readiness-based rebuild."""

    expected = build_ap_d1_development_execution_configs(
        root, readiness_path, provider_id=provider_id
    )
    observed = {task_id: dict(config) for task_id, config in configs.items()}
    if observed != expected:
        return ["A-P D1 development execution configs differ from deterministic rebuild"]
    return []


__all__ = [
    "AP_D1_PROVIDER_SPECS",
    "AP_D1_TASK_SPECS",
    "build_all_ap_d1_development_execution_configs",
    "build_ap_d1_development_execution_configs",
    "validate_ap_d1_development_execution_configs",
    "validate_development_execution_config",
]
