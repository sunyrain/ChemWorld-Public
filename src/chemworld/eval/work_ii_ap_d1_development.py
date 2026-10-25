"""Lightweight, explicit authorization for independent A-P development D1 triplets.

This module deliberately does not create a release freeze.  It validates the
scientific/runtime contract that is needed to run one development triplet and
requires a user-authored spending authorization before any provider process is
started.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import file_sha256
from chemworld.eval.work_ii_ap_terminal_d1_development_execution import (
    AP_D1_PROVIDER_SPECS,
    AP_D1_TASK_SPECS,
    build_ap_d1_development_execution_configs,
)
from chemworld.eval.work_ii_ap_terminal_d1_readiness import (
    AP_D1_ARMS,
    AP_D1_CHECKPOINTS,
    AP_D1_EXPERIMENTS,
    validate_independent_ap_d1_readiness,
)
from chemworld.eval.work_ii_cost import _cost_usd
from chemworld.eval.work_ii_d1_execution import D1_EXECUTION_CONTRACT, D1CellStore

AP_D1_DEVELOPMENT_AUTHORIZATION_VERSION = "chemworld-work-ii-ap-d1-development-authorization-0.1"
AP_D1_REQUALIFICATION_AUTHORIZATION_VERSION = (
    "chemworld-work-ii-ap-d1-platform-requalification-authorization-0.2"
)
DEFAULT_AP_D1_PLAN = Path("configs/benchmark/work_ii_ap_terminal_d1_independent_plan_v0.1.json")
DEFAULT_AP_D1_READINESS = Path(
    "workstreams/flagship_tasks/reports/work-ii-ap-independent-terminal-d1-readiness-v0.1.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _inside(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must remain inside the repository")
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _readiness_configs(root: Path, readiness: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for row in readiness.get("tasks", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str):
            continue
        output = row.get("output_config")
        if isinstance(output, str) and output:
            configs[str(row["task_id"])] = _load(root / output)
    return configs


def validate_ap_d1_development_config(
    root: Path,
    config_path: Path,
    *,
    readiness_path: Path | None = None,
    plan_path: Path | None = None,
) -> list[str]:
    """Validate a seed-2 execution config against current provider-free readiness."""

    root = root.resolve()
    errors: list[str] = []
    try:
        config_path = _inside(root, config_path, label="A-P D1 config")
        readiness_path = _inside(
            root,
            readiness_path or root / DEFAULT_AP_D1_READINESS,
            label="A-P D1 readiness",
        )
        plan_path = _inside(
            root,
            plan_path or root / DEFAULT_AP_D1_PLAN,
            label="A-P D1 plan",
        )
        config = _load(config_path)
        readiness = _load(readiness_path)
        static_configs = _readiness_configs(root, readiness)
        errors.extend(
            validate_independent_ap_d1_readiness(root, plan_path, readiness, static_configs)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"A-P development D1 inputs cannot be read: {error}"]

    task_id = config.get("task_id")
    try:
        config_provider = config.get("provider")
        config_provider = config_provider if isinstance(config_provider, Mapping) else {}
        provider_id = str(config_provider.get("id", ""))
        expected_configs = build_ap_d1_development_execution_configs(
            root, readiness_path, provider_id=provider_id
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"A-P development execution config cannot be rebuilt: {error}")
        expected_configs = {}
    if config != expected_configs.get(str(task_id)):
        errors.append("A-P development execution config differs from its deterministic rebuild")
    rows = [
        row
        for row in readiness.get("tasks", [])
        if isinstance(row, Mapping) and row.get("task_id") == task_id
    ]
    if len(rows) != 1:
        errors.append("A-P development D1 task has no unique readiness row")
        return errors
    row = rows[0]
    static = static_configs.get(str(task_id), {})
    if (
        readiness.get("status") != "ready"
        or readiness.get("provider_call_count") != 0
        or row.get("status") != "ready_static_config_provider_execution_blocked"
        or row.get("selected_world_seed") != config.get("world_seed")
        or config.get("world_seed") != 2
    ):
        errors.append("A-P development D1 is not selected by current readiness")

    independent = config.get("independent_terminal_d1")
    independent = independent if isinstance(independent, Mapping) else {}
    if independent.get("source_static_config_path") != row.get("output_config"):
        errors.append("A-P development D1 does not identify its readiness config")
    if independent.get("readiness_status") != row.get("status"):
        errors.append("A-P development D1 readiness status differs from its source row")
    if (
        independent.get("historical_participant_results_replaced") is not False
        or independent.get("scientific_outcome_is_admission_rule") is not False
        or independent.get("provider_execution_authorized") is not False
    ):
        errors.append("A-P development D1 outcome-blind boundary is missing")

    for field in (
        "task_id",
        "world_seed",
        "world_split",
        "prior_arms",
        "belief_checkpoint",
        "intervention",
        "snapshot_stages",
    ):
        if config.get(field) != static.get(field):
            errors.append(f"A-P development D1 changed frozen scientific field: {field}")

    campaign = config.get("campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    method = config.get("method_resources")
    method = method if isinstance(method, Mapping) else {}
    execution = config.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    provider = config.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    if (
        set(config.get("prior_arms", {})) != AP_D1_ARMS
        or campaign.get("complete_experiments") != AP_D1_EXPERIMENTS
        or campaign.get("checkpoint_complete_experiments") != AP_D1_CHECKPOINTS
        or method.get("complete_experiment_limit") != AP_D1_EXPERIMENTS
        or method.get("checkpoint_complete_experiments") != AP_D1_CHECKPOINTS[1:]
        or "resource_status" in method
    ):
        errors.append("A-P development D1 has an invalid 3-arm/10-round resource pattern")
    if (
        execution.get("max_concurrency") != 3
        or execution.get("within_cell_concurrency") != 1
        or execution.get("parallelization_unit") != "same_seed_prior_arm_triplet"
        or execution.get("failure_semantics")
        != "retain cell failures and continue every scheduled seed triplet"
        or execution.get("systemic_failure_semantics")
        != "stop only when all three arms fail before the first committed operation"
        or execution.get("pilot_expansion_headroom_fraction") != 0.2
        or execution.get("d1_execution_contract") != D1_EXECUTION_CONTRACT
    ):
        errors.append("A-P development D1 lifecycle contract is incomplete")
    numeric_positive = (
        "input_token_limit",
        "uncached_input_token_limit",
        "output_token_limit",
        "wall_time_limit_s",
    )
    if any(float(method.get(field, 0)) <= 0 for field in numeric_positive):
        errors.append("A-P development D1 method resource envelope is incomplete")
    if (
        not isinstance(provider.get("session_wall_time_limit_s"), int | float)
        or float(provider.get("session_wall_time_limit_s", 0)) <= 0
        or float(provider.get("progress_interval_s", 0)) <= 0
        or float(provider.get("progress_interval_s", 0)) > 60
        or any(
            not isinstance(provider.get(field), int) or int(provider.get(field, -1)) < 0
            for field in (
                "max_recovered_mcp_tool_failures",
                "max_consecutive_mcp_tool_failures",
                "max_provider_error_events",
            )
        )
    ):
        errors.append("A-P development D1 provider recovery envelope is incomplete")
    if (
        config.get("formal_result") is not False
        or qualification.get("q2_passed") is not True
        or qualification.get("execution_authorized") is not False
        or qualification.get("formal_r5_authorized") is not False
        or not isinstance(qualification.get("max_resource_rejections"), int)
        or int(qualification.get("max_resource_rejections", -1)) < 0
    ):
        errors.append("A-P development D1 crossed or lacks its authorization boundary")
    return errors


def build_ap_d1_development_cost_budget(
    root: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the worst-case provider reservation from exact task configs."""

    pricing = authorization.get("pricing")
    pricing = pricing if isinstance(pricing, Mapping) else {}
    rates: list[float] = []
    for field in ("cache_hit_input", "cache_miss_input", "output"):
        value = pricing.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("A-P development pricing rates must be finite numbers")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("A-P development pricing rates must be finite and non-negative")
        rates.append(parsed)
    if not any(rate > 0 for rate in rates):
        raise ValueError("A-P development pricing rates cannot all be zero")
    if (
        pricing.get("unit") != "usd_per_million_tokens"
        or not isinstance(pricing.get("source"), str)
        or not str(pricing.get("source")).strip()
        or not isinstance(pricing.get("observed_at"), str)
        or not str(pricing.get("observed_at")).strip()
    ):
        raise ValueError("A-P development pricing source/timestamp is incomplete")

    blocks = authorization.get("task_blocks")
    if not isinstance(blocks, list):
        raise ValueError("A-P development authorization lacks task blocks")
    per_task: dict[str, dict[str, Any]] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("A-P development authorization has a malformed task block")
        task_id = str(block.get("task_id", ""))
        config_relative = block.get("campaign_config")
        if not isinstance(config_relative, str):
            raise ValueError("A-P development task block lacks its config")
        config_path = _inside(root, root / config_relative, label="A-P D1 config")
        config = _load(config_path)
        resources = config.get("method_resources")
        resources = resources if isinstance(resources, Mapping) else {}
        input_cap = int(resources.get("input_token_limit", -1))
        uncached_cap = int(resources.get("uncached_input_token_limit", -1))
        output_cap = int(resources.get("output_token_limit", -1))
        per_attempt = _cost_usd(
            input_tokens=input_cap,
            uncached_input_tokens=uncached_cap,
            output_tokens=output_cap,
            cache_hit_input_usd_per_million=rates[0],
            cache_miss_input_usd_per_million=rates[1],
            output_usd_per_million=rates[2],
        )
        per_task[task_id] = {
            "per_attempt_cost_cap_usd": per_attempt,
            "initial_triplet_cost_cap_usd": round(per_attempt * 3, 12),
            "all_attempts_cost_cap_usd": round(per_attempt * 6, 12),
        }
    initial = round(sum(row["initial_triplet_cost_cap_usd"] for row in per_task.values()), 12)
    hard = round(sum(row["all_attempts_cost_cap_usd"] for row in per_task.values()), 12)
    return {
        "pricing": dict(pricing),
        "per_task": per_task,
        "initial_schedule_cost_cap_usd": initial,
        "all_infrastructure_resumes_cost_cap_usd": hard,
    }


def validate_ap_d1_development_authorization(
    root: Path,
    authorization_path: Path,
    *,
    config_path: Path,
    output_root: Path,
    readiness_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate explicit user scope without release hashes or a clean-tree gate."""

    root = root.resolve()
    errors = validate_ap_d1_development_config(root, config_path, readiness_path=readiness_path)
    try:
        authorization = _load(authorization_path.resolve())
        config_path = _inside(root, config_path, label="A-P D1 config")
        output_root = _inside(root, output_root, label="A-P D1 output")
        output_relative = _relative(root, output_root)
        config = _load(config_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {}, [*errors, f"A-P development authorization cannot be read: {error}"]

    if not output_relative.startswith("runs/development/"):
        errors.append("A-P development D1 output must remain under runs/development")
    authorization_version = authorization.get("schema_version")
    original_outcome_blind_authorization = (
        authorization_version == AP_D1_DEVELOPMENT_AUTHORIZATION_VERSION
        and authorization.get("participant_outcomes_observed_before_authorization") == 0
    )
    retained_platform_requalification = (
        authorization_version == AP_D1_REQUALIFICATION_AUTHORIZATION_VERSION
        and authorization.get("authorization_scope")
        == "platform_requalification_after_retained_development_diagnosis"
        and authorization.get("current_requalification_outcomes_observed_before_authorization")
        == 0
        and isinstance(
            authorization.get("historical_development_cells_observed_before_requalification"),
            int,
        )
        and not isinstance(
            authorization.get("historical_development_cells_observed_before_requalification"),
            bool,
        )
        and authorization.get("historical_development_cells_observed_before_requalification", 0)
        > 0
        and authorization.get("historical_development_outputs_retained") is True
        and authorization.get("fresh_output_roots") is True
        and authorization.get("platform_requalification_only") is True
        and authorization.get("scientific_design_changed_after_historical_outcomes") is False
        and authorization.get("resource_envelope_calibration_basis")
        == "provider_usage_only_without_scientific_effect_selection"
    )
    if (
        not (original_outcome_blind_authorization or retained_platform_requalification)
        or authorization.get("status") != "authorized"
        or authorization.get("authorized_by") != "user"
        or not isinstance(authorization.get("approved_at"), str)
        or not authorization.get("approved_at")
        or authorization.get("provider_execution_allowed") is not True
        or authorization.get("formal_result") is not False
        or authorization.get("formal_r5_authorized") is not False
    ):
        errors.append(
            "A-P development D1 lacks explicit original or retained-platform-requalification "
            "user authorization"
        )
    provider = authorization.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    config_provider = config.get("provider")
    config_provider = config_provider if isinstance(config_provider, Mapping) else {}
    if provider != {
        "provider_id": config_provider.get("id"),
        "model": config_provider.get("model"),
    }:
        errors.append("A-P development authorization provider/model mismatch")

    blocks = authorization.get("task_blocks")
    blocks = blocks if isinstance(blocks, list) else []
    expected_task_ids = set(AP_D1_TASK_SPECS)
    configured_provider_id = str(config_provider.get("id", ""))
    provider_spec = AP_D1_PROVIDER_SPECS.get(configured_provider_id, {})
    provider_outputs = provider_spec.get("outputs")
    provider_outputs = provider_outputs if isinstance(provider_outputs, Mapping) else {}
    block_task_ids = {str(block.get("task_id")) for block in blocks if isinstance(block, Mapping)}
    if len(blocks) != len(expected_task_ids) or block_task_ids != expected_task_ids:
        errors.append("A-P development authorization must name exactly both task blocks")
    seen_outputs: set[str] = set()
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        task = str(block.get("task_id", ""))
        expected_config_relative = provider_outputs.get(task)
        output_value = block.get("output_root")
        config_relative = expected_config_relative
        try:
            canonical_output = (
                _relative(root, root / output_value) if isinstance(output_value, str) else ""
            )
        except ValueError:
            canonical_output = ""
        if (
            block.get("world_seed") != 2
            or not isinstance(config_relative, str)
            or block.get("campaign_config") != config_relative
            or block.get("campaign_config_sha256")
            != (file_sha256(root / config_relative) if isinstance(config_relative, str) else None)
            or not isinstance(output_value, str)
            or not output_value.startswith("runs/development/")
            or canonical_output != output_value
            or output_value in seen_outputs
        ):
            errors.append(f"A-P development authorization task block drifted: {task}")
        if isinstance(output_value, str):
            seen_outputs.add(output_value)
    matching = [
        block
        for block in blocks
        if isinstance(block, Mapping)
        and block.get("task_id") == config.get("task_id")
        and block.get("world_seed") == config.get("world_seed")
        and block.get("campaign_config") == _relative(root, config_path)
        and block.get("output_root") == output_relative
    ]
    if len(matching) != 1:
        errors.append("A-P development authorization does not name this exact task/output")
    expected_sessions = len(blocks) * 3
    expected_experiments = len(blocks) * 3 * AP_D1_EXPERIMENTS
    if (
        not blocks
        or authorization.get("provider_sessions_initial") != expected_sessions
        or authorization.get("provider_process_attempts_hard_cap") != expected_sessions * 2
        or authorization.get("complete_experiments_total") != expected_experiments
    ):
        errors.append("A-P development authorization denominator mismatch")
    unlimited_spend = authorization.get("spending_limit") == "unlimited"
    finite_spend = (
        authorization.get("spending_limit") == "finite_ceiling"
        and not isinstance(authorization.get("currency_ceiling_usd"), bool)
        and isinstance(authorization.get("currency_ceiling_usd"), int | float)
        and math.isfinite(float(authorization.get("currency_ceiling_usd", 0)))
        and float(authorization.get("currency_ceiling_usd", 0)) > 0
    )
    if (
        authorization.get("currency") != "USD"
        or not (unlimited_spend or finite_spend)
        or not (
            authorization.get("credential_rotation_confirmed") is True
            or authorization.get("credential_use_authorized") is True
        )
    ):
        errors.append("A-P development authorization lacks provider/currency confirmation")
    if unlimited_spend:
        if authorization.get("currency_ceiling_usd") is not None:
            errors.append("unlimited A-P development spend must not declare a ceiling")
    else:
        try:
            budget = build_ap_d1_development_cost_budget(root, authorization)
            ceiling = float(authorization.get("currency_ceiling_usd", 0))
            hard = float(budget["all_infrastructure_resumes_cost_cap_usd"])
            if (
                authorization.get("initial_schedule_cost_cap_usd")
                != budget["initial_schedule_cost_cap_usd"]
                or authorization.get("all_infrastructure_resumes_cost_cap_usd") != hard
                or ceiling < hard
            ):
                errors.append("A-P development currency ceiling does not cover all attempts")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"A-P development currency budget cannot be rebuilt: {error}")
    return authorization, errors


def validate_and_claim_ap_d1_development_attempt(
    root: Path,
    *,
    config_path: Path,
    output_root: Path,
    attempt_output: Path,
    attempt_receipt_path: Path,
    cost_ledger_path: Path,
    world_seed: int,
    arm: str,
    authorization_path: Path,
    readiness_path: Path,
) -> dict[str, Any]:
    """Require and atomically claim one fully parent-authorized D1 provider attempt."""

    config = _load(config_path)
    if world_seed != config.get("world_seed") or arm not in config.get("prior_arms", {}):
        raise ValueError("A-P development child seed/arm differs from its config")
    store = D1CellStore(
        output_root / "store",
        config_path=config_path,
        task_id=str(config["task_id"]),
        world_seeds=[world_seed],
        arms=list(config["prior_arms"]),
    )
    key = store.key(world_seed, arm)
    receipt = _load(attempt_receipt_path)
    attempt_id = str(receipt.get("attempt_id", ""))
    expected_receipt = store.provider_attempts / key / f"{attempt_id}.json"
    expected_output = output_root / "attempts" / key / attempt_id
    ledger = _load(cost_ledger_path)
    expected_ledger = output_root / "cost_ledgers" / key / f"{attempt_id}.json"
    if (
        attempt_receipt_path.resolve() != expected_receipt.resolve()
        or attempt_output.resolve() != expected_output.resolve()
        or cost_ledger_path.resolve() != expected_ledger.resolve()
        or receipt.get("state") != "provider_process_launch_authorized"
        or receipt.get("cell_key_sha256") != key
        or ledger.get("state")
        not in {
            "full_token_cap_reserved_before_provider_launch",
            "user_authorized_unlimited_spend_before_provider_launch",
        }
        or ledger.get("cell_key_sha256") != key
        or ledger.get("attempt_id") != attempt_id
        or ledger.get("task_id") != config.get("task_id")
        or ledger.get("authorization_sha256") != file_sha256(authorization_path)
        or ledger.get("readiness_sha256") != file_sha256(readiness_path)
        or ledger.get("within_authorized_ceiling") is not True
        or not attempt_id
        or store.audit().get("invalid_receipts")
    ):
        raise ValueError("A-P development child lacks an exact parent attempt receipt")
    authorization = _load(authorization_path)
    provider = config.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    if (
        authorization.get("status") != "authorized"
        or authorization.get("provider_execution_allowed") is not True
        or authorization.get("formal_result") is not False
        or authorization.get("formal_r5_authorized") is not False
        or authorization.get("provider")
        != {"provider_id": provider.get("id"), "model": provider.get("model")}
    ):
        raise ValueError("A-P development child authorization binding is invalid")
    claim = store.root / "attempt_claims" / key / f"{attempt_id}.json"
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        with claim.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "state": "claimed_by_child",
                    "cell_key_sha256": key,
                    "attempt_id": attempt_id,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise ValueError("A-P development attempt receipt was already claimed") from error
    return {"attempt_id": attempt_id, "cell_key_sha256": key, "claim": str(claim)}


__all__ = [
    "AP_D1_DEVELOPMENT_AUTHORIZATION_VERSION",
    "AP_D1_REQUALIFICATION_AUTHORIZATION_VERSION",
    "DEFAULT_AP_D1_PLAN",
    "DEFAULT_AP_D1_READINESS",
    "build_ap_d1_development_cost_budget",
    "validate_and_claim_ap_d1_development_attempt",
    "validate_ap_d1_development_authorization",
    "validate_ap_d1_development_config",
]
