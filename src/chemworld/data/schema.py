"""Versioned trajectory schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TRAJECTORY_SCHEMA_VERSION = "chemworld-trajectory-0.3"
LEGACY_TRAJECTORY_SCHEMA_VERSIONS = (
    "chemworld-trajectory-0.1", "chemworld-trajectory-0.2",
)
# Historical read aliases, never materialized by the current writer.
TRAJECTORY_COMPATIBILITY_ALIASES = (
    "benchmark_task_id",
    "observation",
    "reward",
    "agent_view",
    "leaderboard_score",
)
TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION = "chemworld-trajectory-0.3"
SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS = (
    *LEGACY_TRAJECTORY_SCHEMA_VERSIONS,
    TRAJECTORY_SCHEMA_VERSION,
)
OUTCOME_LAYER_FIELDS = (
    "environment_outcome",
    "agent_visible_observation",
    "evaluation_outcome",
)


@dataclass(frozen=True)
class TrajectoryRecordPayload:
    schema_version: str
    env_version: str
    world_family_version: str
    task_id: str
    env_id: str
    world_split: str
    objective: str
    budget: int
    episode_mode: str
    safety_limit: float
    task_contract_hash: str | None
    runtime_profile_hash: str | None
    mechanism_id: str | None
    mechanism_hash: str | None
    scoring_contract_hash: str | None
    observation_contract_hash: str | None
    kernel_id: str | None
    kernel_version: str | None
    affected_ledgers: list[str]
    world_events: list[dict[str, Any]]
    state_patches_summary: list[dict[str, Any]]
    transaction_status: str | None
    rollback_reason: str | None
    world_id: str
    seed: int
    step: int
    run_id: str
    campaign_id: str
    experiment_index: int
    operation_id: int
    scenario_id: str | None
    initial_state_id: str | None
    action: dict[str, Any]
    environment_outcome: dict[str, Any]
    agent_visible_observation: dict[str, Any]
    evaluation_outcome: dict[str, Any]
    terminated: bool
    truncated: bool
    constraint_flags: dict[str, bool]
    agent_metadata: dict[str, Any]
    timestamp: str
    operation_type: str | None = None
    preconditions: dict[str, bool] = field(default_factory=dict)
    state_delta_summary: dict[str, float] = field(default_factory=dict)
    constitution_checks: list[dict[str, Any]] = field(default_factory=list)
    instrument: str | None = None
    instrument_source: str | None = None
    observed_keys: list[str] = field(default_factory=list)
    observed_mask: dict[str, bool] = field(default_factory=dict)
    raw_signal: dict[str, Any] = field(default_factory=dict)
    processed_estimate: dict[str, float | None] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    measurement_cost: float = 0.0
    sample_consumed: float = 0.0
    observed_reward: float = 0.0
    reward_source: str | None = None
    explanation: dict[str, Any] = field(default_factory=dict)
