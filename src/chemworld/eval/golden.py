"""Golden trajectory summaries for core benchmark contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from chemworld.data.records import trajectory_value
from chemworld.eval.metrics import evaluate_records
from chemworld.tasks import CORE_TASK_IDS, TASK_REGISTRY

GOLDEN_SUMMARY_SCHEMA_VERSION = "chemworld-golden-summary-0.1"
CORE_GOLDEN_TASKS = CORE_TASK_IDS

FLOAT_DIGITS = 12
FINAL_METRIC_KEYS: tuple[str, ...] = (
    "steps",
    "final_assay_count",
    "final_best_score",
    "best_valid_score",
    "best_valid_yield",
    "area_under_best_score",
    "safety_violations",
    "high_cost_violations",
    "invalid_action_count",
    "precondition_failure_count",
    "mean_cost",
    "mean_safety_risk",
    "mean_purity",
    "mean_recovery",
    "purification_score",
    "safety_cost",
    "safety_aware_score",
    "cost_aware_score",
    "total_score",
)


@dataclass(frozen=True)
class GoldenTrajectoryTarget:
    task_id: str
    seed: int
    agent_name: str = "scripted_chemistry"


def core_golden_targets() -> tuple[GoldenTrajectoryTarget, ...]:
    """Return canonical scripted trajectory targets for the frozen core tasks."""

    return tuple(
        GoldenTrajectoryTarget(task_id=task_id, seed=TASK_REGISTRY[task_id].seeds[0])
        for task_id in CORE_GOLDEN_TASKS
    )


def round_public_value(value: Any) -> Any:
    """Round JSON-friendly public values while preserving booleans and nulls."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS)
    if isinstance(value, Mapping):
        return {str(key): round_public_value(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [round_public_value(item) for item in value]
    return value


def _observation_summary(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: round_public_value(value)
        for key, value in sorted(observation.items())
        if value is not None
    }


def _event_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_type": event.get("event_type"),
            "operation_type": event.get("operation_type"),
            "kernel_id": event.get("kernel_id"),
            "service_id": event.get("service_id"),
            "payload": round_public_value(event.get("payload", {})),
        }
        for event in events
    ]


def _patch_summary(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "patch_type": patch.get("patch_type"),
            "affected_ledgers": list(patch.get("affected_ledgers", [])),
            "summary": round_public_value(patch.get("summary", {})),
        }
        for patch in patches
    ]


def summarize_golden_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic comparison summary from public trajectory records."""

    if not records:
        raise ValueError("Cannot summarize an empty golden trajectory")
    first = records[0]
    task_id = str(trajectory_value(first, 'benchmark_task_id', None) or first["task_id"])
    task = TASK_REGISTRY[task_id]
    metrics = evaluate_records(records, threshold=task.threshold).to_dict()
    final_record = records[-1]

    return {
        "summary_schema_version": GOLDEN_SUMMARY_SCHEMA_VERSION,
        "task_id": task_id,
        "agent_name": str(first.get("agent_metadata", {}).get("agent_name", "")),
        "seed": int(first["seed"]),
        "world_split": str(first["world_split"]),
        "scenario_id": first.get("scenario_id"),
        "mechanism_id": first.get("mechanism_id"),
        "mechanism_hash": first.get("mechanism_hash"),
        "task_contract_hash": first.get("task_contract_hash"),
        "runtime_profile_hash": first.get("runtime_profile_hash"),
        "scoring_contract_hash": first.get("scoring_contract_hash"),
        "observation_contract_hash": first.get("observation_contract_hash"),
        "budget": int(first["budget"]),
        "episode_mode": str(first.get("episode_mode", "")),
        "step_count": len(records),
        "final": {
            "terminated": bool(final_record["terminated"]),
            "truncated": bool(final_record["truncated"]),
            "leaderboard_score": round_public_value(
                trajectory_value(final_record, 'leaderboard_score', None)
            ),
            "reward": round_public_value(trajectory_value(final_record, 'reward', None)),
            "observation": _observation_summary(trajectory_value(final_record, 'observation', {})),
            "processed_estimate": round_public_value(
                final_record.get("processed_estimate", {})
            ),
            "raw_signal_keys": sorted(final_record.get("raw_signal", {}).keys()),
        },
        "final_metrics": {
            key: round_public_value(metrics[key]) for key in FINAL_METRIC_KEYS
        },
        "steps": [
            {
                "step": int(record["step"]),
                "operation_type": record.get("operation_type"),
                "action": round_public_value(record["action"]),
                "reward": round_public_value(trajectory_value(record, 'reward')),
                "terminated": bool(record["terminated"]),
                "truncated": bool(record["truncated"]),
                "transaction_status": record.get("transaction_status"),
                "rollback_reason": record.get("rollback_reason"),
                "kernel_id": record.get("kernel_id"),
                "kernel_version": record.get("kernel_version"),
                "instrument": record.get("instrument"),
                "affected_ledgers": list(record.get("affected_ledgers", [])),
                "world_events": _event_summary(record.get("world_events", [])),
                "state_patches_summary": _patch_summary(
                    record.get("state_patches_summary", [])
                ),
                "constraint_flags": round_public_value(
                    record.get("constraint_flags", {})
                ),
                "preconditions": round_public_value(record.get("preconditions", {})),
                "observed_keys": list(record.get("observed_keys", [])),
                "observed_mask": round_public_value(record.get("observed_mask", {})),
                "observation": _observation_summary(trajectory_value(record, 'observation', {})),
                "processed_estimate": round_public_value(
                    record.get("processed_estimate", {})
                ),
                "leaderboard_score": round_public_value(
                    trajectory_value(record, 'leaderboard_score', None)
                ),
                "measurement_cost": round_public_value(
                    record.get("measurement_cost", 0.0)
                ),
                "sample_consumed": round_public_value(
                    record.get("sample_consumed", 0.0)
                ),
            }
            for record in records
        ],
    }
