"""Trajectory logging utilities."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from chemworld import __version__
from chemworld.data.schema import (
    TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION,
    TRAJECTORY_COMPATIBILITY_ALIASES,
    TRAJECTORY_SCHEMA_VERSION,
)
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID
from chemworld.world.thermal_control import REACTION_THERMAL_CONTROL_ID


def to_builtin(value: Any) -> Any:
    """Convert numpy-heavy values to JSON-safe Python objects."""

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return to_builtin(float(value.reshape(-1)[0]))
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_builtin(item) for item in value]
    return value


def observation_to_json(observation: dict[str, Any]) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    for key, value in observation.items():
        converted = to_builtin(value)
        payload[key] = None if converted is None else float(converted)
    return payload


def action_payload(action: dict[str, Any]) -> dict[str, Any]:
    return to_builtin(action)


class TrajectoryLogger:
    """Append-only JSONL logger for benchmark trajectories."""

    compatibility_aliases = TRAJECTORY_COMPATIBILITY_ALIASES
    compatibility_alias_write_removal_version = TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> TrajectoryLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def log(
        self,
        *,
        task_info: dict[str, Any],
        step: int,
        action: dict[str, Any],
        observation: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        agent_metadata: dict[str, Any],
        explanation: dict[str, Any] | None = None,
        agent_view: dict[str, Any] | None = None,
        agent_trace: list[dict[str, Any]] | None = None,
        method_resources: dict[str, Any] | None = None,
        environment_outcome: dict[str, Any] | None = None,
        agent_visible_observation: dict[str, Any] | None = None,
        evaluation_outcome: dict[str, Any] | None = None,
    ) -> None:
        observation_payload = observation_to_json(observation)
        stable_task_id = str(task_info.get("task_id") or task_info["env_id"])
        run_id = (
            f"{task_info['env_id']}:{task_info['world_split']}:"
            f"{task_info['objective']}:seed-{task_info['seed']}"
        )
        default_environment_outcome = {
            "observation": observation_payload,
            "transaction_status": info.get("transaction_status"),
            "operation_outcome": info.get("operation_outcome"),
            "actual_duration_s": info.get("actual_duration_s"),
            "outcome_diagnostics": to_builtin(info.get("outcome_diagnostics", {})),
            "measurement_status": info.get("measurement_status"),
            "terminal_assay": info.get("terminal_assay", False),
            "operation_type": info.get("operation_type"),
            "constraint_flags": to_builtin(info.get("constraint_flags", {})),
            "world_events": to_builtin(info.get("world_events", [])),
            "state_delta_summary": to_builtin(info.get("state_delta_summary", {})),
            "state_patches_summary": to_builtin(info.get("state_patches_summary", [])),
            "raw_signal": to_builtin(info.get("raw_signal", {})),
            "native_observation": to_builtin(info.get("native_observation")),
            "processed_estimate": to_builtin(info.get("processed_estimate", {})),
            "uncertainty": to_builtin(info.get("uncertainty", {})),
            "measurement_cost": float(info.get("measurement_cost", 0.0)),
            "sample_consumed": float(info.get("sample_consumed", 0.0)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "last_terminal_summary": to_builtin(info.get("last_terminal_summary")),
            "batch_discarded": bool(info.get("batch_discarded", False)),
            "right_censored_open_batch": bool(info.get("right_censored_open_batch", False)),
            "sample_inventory": to_builtin(info.get("sample_inventory", [])),
            "material_routing": to_builtin(info.get("material_routing", {})),
        }
        default_agent_visible_observation = {
            "observation": observation_payload,
            "views": to_builtin(agent_view or {}),
            "observed_reward": float(info.get("observed_reward", reward)),
        }
        default_evaluation_outcome = {
            "online_transition_reward": float(reward),
            "leaderboard_score": to_builtin(info.get("leaderboard_score")),
            "reward_source": info.get("reward_source"),
            "scoring_contract_hash": info.get(
                "scoring_contract_hash",
                task_info.get("scoring_contract_hash"),
            ),
        }
        resolved_environment_outcome = {
            **default_environment_outcome,
            **to_builtin(environment_outcome or {}),
        }
        if "batch_resource_preflight" in info:
            for field in (
                "batch_resource_preflight", "batch_resource_state", "batch_budget_exhausted",
            ):
                resolved_environment_outcome[field] = to_builtin(info.get(field))
        resolved_agent_visible_observation = {
            **default_agent_visible_observation,
            **to_builtin(agent_visible_observation or {}),
        }
        resolved_evaluation_outcome = {
            **default_evaluation_outcome,
            **to_builtin(evaluation_outcome or {}),
        }
        raw_material_information = task_info.get("material_information")
        material_information_mode = (
            raw_material_information.get("mode")
            if isinstance(raw_material_information, dict)
            else None
        )
        raw_material_information_config = task_info.get(
            "material_information_config"
        )
        material_information_config = (
            to_builtin(raw_material_information_config)
            if isinstance(raw_material_information_config, dict)
            else (
                {"mode": raw_material_information.get("mode")}
                if isinstance(raw_material_information, dict)
                else None
            )
        )
        raw_campaign_resources = task_info.get("campaign_resources")
        campaign_resource_card = (
            raw_campaign_resources.get("card")
            if isinstance(raw_campaign_resources, dict)
            else None
        )
        scoring_contract = task_info.get("scoring_contract")
        payload = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "env_version": info.get("env_version", __version__),
            "world_family_version": info.get(
                "world_family_version",
                task_info.get("world_family_version"),
            ),
            "world_family_intervention_version": info.get(
                "world_family_intervention_version",
                task_info.get("world_family_intervention_version"),
            ),
            "world_family_intervention_hash": info.get(
                "world_family_intervention_hash",
                task_info.get("world_family_intervention_hash"),
            ),
            "mechanism_family_intervention_version": info.get(
                "mechanism_family_intervention_version",
                task_info.get("mechanism_family_intervention_version"),
            ),
            "mechanism_family_intervention_hash": info.get(
                "mechanism_family_intervention_hash",
                task_info.get("mechanism_family_intervention_hash"),
            ),
            "electrochemical_workflow_mode": task_info.get(
                "electrochemical_workflow_mode"
            ),
            "material_information": (
                {"mode": material_information_mode}
                if material_information_mode is not None
                else None
            ),
            # Keep the public material_information field stable while storing
            # the evaluator-side configuration needed for exact replay of
            # misindexed conditions (target_field + descriptor_permutation).
            "material_information_config": material_information_config,
            "scoring_contract_id": (
                scoring_contract.get("contract_id")
                if isinstance(scoring_contract, dict)
                else None
            ),
            "observation_seed": task_info.get("observation_seed"),
            "observation_noise_mode": task_info.get(
                "observation_noise_mode"
            ),
            "full_process_contract_id": task_info.get("full_process_contract_id"),
            "runtime_semantics_id": task_info.get("runtime_semantics_id", RUNTIME_SEMANTICS_ID),
            "risk_signal_contract": to_builtin(task_info.get("risk_signal_contract")),
            "reaction_thermal_control_id": task_info.get(
                "reaction_thermal_control_id", REACTION_THERMAL_CONTROL_ID
            ),
            "research_brief": {
                key: to_builtin(task_info["research_brief"][key])
                for key in ("schema_version", "card", "prior_record")
            } if task_info.get("research_brief") is not None else None,
            "observation_noise_namespace": task_info.get(
                "observation_noise_namespace"
            ),
            "campaign_resource_card_sha256": task_info.get(
                "campaign_resource_card_sha256"
            ),
            "campaign_resource_card": to_builtin(campaign_resource_card),
            "electrochemical_material_family_id": task_info.get(
                "electrochemical_material_family_id"
            ),
            "electrochemical_material_family_sha256": task_info.get(
                "electrochemical_material_family_sha256"
            ),
            "electrochemical_material_instance_sha256": task_info.get(
                "electrochemical_material_instance_sha256"
            ),
            "crystallization_material_family_id": task_info.get(
                "crystallization_material_family_id"
            ),
            "crystallization_material_family_sha256": task_info.get(
                "crystallization_material_family_sha256"
            ),
            "crystallization_material_instance_sha256": task_info.get(
                "crystallization_material_instance_sha256"
            ),
            "task_id": stable_task_id,
            "env_id": task_info["env_id"],
            "world_split": task_info["world_split"],
            "world_provider": task_info.get("world_provider"),
            "objective": task_info["objective"],
            "budget": int(task_info["budget"]),
            "official_budget": int(task_info.get("official_budget", task_info["budget"])),
            "episode_mode": task_info.get("episode_mode", "single_experiment"),
            "contract_profile": task_info.get("contract_profile", "official"),
            "safety_limit": float(task_info.get("safety_limit", 0.65)),
            "task_contract_hash": info.get(
                "task_contract_hash",
                task_info.get("task_contract_hash"),
            ),
            "runtime_profile_hash": info.get(
                "runtime_profile_hash",
                task_info.get("runtime_profile_hash"),
            ),
            "mechanism_id": info.get("mechanism_id", task_info.get("mechanism_id")),
            "mechanism_hash": info.get("mechanism_hash", task_info.get("mechanism_hash")),
            "scoring_contract_hash": info.get(
                "scoring_contract_hash",
                task_info.get("scoring_contract_hash"),
            ),
            "observation_contract_hash": info.get(
                "observation_contract_hash",
                task_info.get("observation_contract_hash"),
            ),
            "kernel_id": info.get("kernel_id"),
            "kernel_version": info.get("kernel_version"),
            "affected_ledgers": to_builtin(info.get("affected_ledgers", [])),
            "world_events": to_builtin(info.get("world_events", [])),
            "state_patches_summary": to_builtin(info.get("state_patches_summary", [])),
            "transaction_status": info.get("transaction_status"),
            "rollback_reason": info.get("rollback_reason"),
            "operation_outcome": info.get("operation_outcome"),
            "actual_duration_s": info.get("actual_duration_s"),
            "outcome_diagnostics": to_builtin(info.get("outcome_diagnostics", {})),
            "measurement_status": info.get("measurement_status"),
            "kernel_maturity": to_builtin(task_info.get("kernel_maturity", {})),
            "physics_maturity": task_info.get("physics_maturity"),
            "proxy_allowed": bool(task_info.get("proxy_allowed", False)),
            "composition_request": to_builtin(task_info.get("composition_request")),
            "world_id": task_info["world_id"],
            "seed": int(task_info["seed"]),
            "step": step,
            "run_id": run_id,
            "campaign_id": info.get(
                "campaign_id",
                f"{task_info.get('task_id') or 'adhoc'}:{task_info.get('seed')}",
            ),
            "experiment_index": int(info.get("experiment_index", 0)),
            "operation_id": int(info.get("operation_id", step)),
            "scenario_id": info.get("scenario_id", task_info.get("scenario_id")),
            "initial_state_id": info.get("initial_state_id", task_info.get("initial_state_id")),
            "action": action_payload(action),
            "environment_outcome": resolved_environment_outcome,
            "agent_visible_observation": resolved_agent_visible_observation,
            "evaluation_outcome": resolved_evaluation_outcome,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "constraint_flags": to_builtin(info.get("constraint_flags", {})),
            "operation_type": info.get("operation_type"),
            "preconditions": to_builtin(info.get("preconditions", {})),
            "state_delta_summary": to_builtin(info.get("state_delta_summary", {})),
            "constitution_checks": to_builtin(info.get("constitution_checks", [])),
            "material_balance": to_builtin(info.get("material_balance", {})),
            "instrument": info.get("instrument"),
            "instrument_source": info.get("instrument_source"),
            "terminal_assay": info.get("terminal_assay", False),
            "observed_keys": to_builtin(info.get("observed_keys", [])),
            "observed_mask": to_builtin(info.get("observed_mask", {})),
            "raw_signal": to_builtin(info.get("raw_signal", {})),
            "native_observation": to_builtin(info.get("native_observation")),
            "processed_estimate": to_builtin(info.get("processed_estimate", {})),
            "uncertainty": to_builtin(info.get("uncertainty", {})),
            "measurement_cost": float(info.get("measurement_cost", 0.0)),
            "sample_consumed": float(info.get("sample_consumed", 0.0)),
            "observed_reward": float(info.get("observed_reward", reward)),
            "reward_source": info.get("reward_source"),
            "agent_metadata": to_builtin(agent_metadata),
            "agent_trace": to_builtin(agent_trace or []),
            "method_resources": to_builtin(method_resources or {}),
            "timestamp": datetime.now(UTC).isoformat(),
            "explanation": explanation or {},
        }
        # The v0.2 writer materializes these five v0.1 aliases only through this
        # bounded compatibility block. New readers must use the three outcome
        # layers; alias writes end at TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION.
        compatibility_aliases = {
            "benchmark_task_id": task_info.get("task_id"),
            "observation": observation_payload,
            "reward": float(reward),
            "agent_view": to_builtin(agent_view or {}),
            "leaderboard_score": to_builtin(info.get("leaderboard_score")),
        }
        if tuple(compatibility_aliases) != self.compatibility_aliases:
            raise RuntimeError("trajectory compatibility alias contract drifted")
        payload.update(compatibility_aliases)
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = []
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: trajectory record must be a JSON object")
            records.append(record)
        return records
