"""Trajectory schema validation without external JSON-schema dependencies."""

from __future__ import annotations

from typing import Any

from chemworld.data.schema import (
    LEGACY_TRAJECTORY_SCHEMA_VERSIONS,
    OUTCOME_LAYER_FIELDS,
    SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS,
    TRAJECTORY_SCHEMA_VERSION,
)
from chemworld.schemas.validation import TRAJECTORY_REQUIRED_KEYS
from chemworld.world.operations import PUBLIC_OBSERVATION_KEYS

REQUIRED_RECORD_KEYS = TRAJECTORY_REQUIRED_KEYS
LEGACY_RECORD_KEYS = REQUIRED_RECORD_KEYS - {"run_id", *OUTCOME_LAYER_FIELDS}

EVENT_ACTION_KEYS = {"operation"}

REQUIRED_OBSERVATION_KEYS = set(PUBLIC_OBSERVATION_KEYS)

MATURITY_LEVELS = {
    "proxy",
    "lite",
    "reference_validated",
    "professional_candidate",
    "professional",
}


def validate_record(record: dict[str, Any]) -> None:
    if "schema_version" not in record:
        raise ValueError("Trajectory record is missing keys: ['schema_version']")
    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version={schema_version!r}; expected one of "
            f"{SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS!r}"
        )

    required_keys = (
        LEGACY_RECORD_KEYS
        if schema_version in LEGACY_TRAJECTORY_SCHEMA_VERSIONS
        else REQUIRED_RECORD_KEYS
    )
    missing = required_keys - record.keys()
    if missing:
        raise ValueError(f"Trajectory record is missing keys: {sorted(missing)}")

    if schema_version == TRAJECTORY_SCHEMA_VERSION:
        for field_name in OUTCOME_LAYER_FIELDS:
            if not isinstance(record[field_name], dict):
                raise ValueError(f"{field_name} must be an object")
        if not isinstance(record["run_id"], str) or not record["run_id"]:
            raise ValueError("run_id must be a non-empty string")
        benchmark_task_id = record.get("benchmark_task_id")
        if benchmark_task_id is not None and record["task_id"] != benchmark_task_id:
            raise ValueError("trajectory v0.2 task_id must identify the stable task contract")
        if record["environment_outcome"].get("observation") != record["observation"]:
            raise ValueError(
                "environment_outcome.observation must match the v0.1 observation alias"
            )
        evaluation_outcome = record["evaluation_outcome"]
        if evaluation_outcome.get("online_transition_reward") != record["reward"]:
            raise ValueError(
                "evaluation_outcome.online_transition_reward must match the reward alias"
            )
        if evaluation_outcome.get("leaderboard_score") != record["leaderboard_score"]:
            raise ValueError(
                "evaluation_outcome.leaderboard_score must match the leaderboard_score alias"
            )

    if not isinstance(record["action"], dict):
        raise ValueError("Trajectory action must be a JSON object")
    if not isinstance(record["observation"], dict):
        raise ValueError("Trajectory observation must be a JSON object")
    if not isinstance(record.get("observed_mask", {}), dict):
        raise ValueError("Trajectory observed_mask must be a JSON object")
    action_missing = EVENT_ACTION_KEYS - record["action"].keys()
    if action_missing:
        raise ValueError(f"Action is missing keys: {sorted(action_missing)}")

    observation_missing = REQUIRED_OBSERVATION_KEYS - record["observation"].keys()
    if observation_missing:
        raise ValueError(f"Observation is missing keys: {sorted(observation_missing)}")

    observed_mask = record.get("observed_mask", {})
    for key in REQUIRED_OBSERVATION_KEYS:
        raw_value = record["observation"][key]
        if raw_value is None:
            if observed_mask.get(key, False):
                raise ValueError(f"Observed key {key} cannot be null")
            continue
        value = float(raw_value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Observation {key}={value} is outside [0, 1]")

    physics_maturity = str(record["physics_maturity"])
    if physics_maturity not in MATURITY_LEVELS:
        raise ValueError(f"Unsupported physics_maturity={physics_maturity!r}")
    kernel_maturity = record.get("kernel_maturity", {})
    if not isinstance(kernel_maturity, dict):
        raise ValueError("kernel_maturity must be an object")
    if kernel_maturity.get("lowest_level") not in {None, physics_maturity}:
        raise ValueError("kernel_maturity.lowest_level must match physics_maturity")
    if physics_maturity == "proxy" and not bool(record["proxy_allowed"]):
        raise ValueError("proxy physics_maturity requires proxy_allowed=True")


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Trajectory is empty")

    validate_record(records[0])
    first_task = records[0]["task_id"]
    previous_step = 0
    for record in records:
        validate_record(record)
        if record["task_id"] != first_task:
            raise ValueError("A trajectory file must contain exactly one task_id")
        step = int(record["step"])
        if step != previous_step + 1:
            raise ValueError(f"Steps must be contiguous; expected {previous_step + 1}, got {step}")
        previous_step = step
