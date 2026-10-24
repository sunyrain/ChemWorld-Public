from __future__ import annotations

import json

import pytest

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.data.schema import (
    TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION,
    TRAJECTORY_COMPATIBILITY_ALIASES,
)
from chemworld.data.validation import validate_records
from chemworld.eval.runner import make_agent, run_agent


def test_validate_records_rejects_missing_schema(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("random"),
        world_split="public-dev",
        budget=2,
        objective="balanced",
        seed=1,
        output_path=path,
    )
    records = load_jsonl(path)
    validate_records(records)
    records[0].pop("schema_version")
    with pytest.raises(ValueError, match="missing keys"):
        validate_records(records)


def test_trajectory_records_include_instrument_signal_layers(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=2,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    validate_records(records)
    assert records[0]["schema_version"] == "chemworld-trajectory-0.3"
    assert TRAJECTORY_COMPATIBILITY_ALIASES == (
        "benchmark_task_id",
        "observation",
        "reward",
        "agent_view",
        "leaderboard_score",
    )
    assert TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION == "chemworld-trajectory-0.3"
    assert not set(TRAJECTORY_COMPATIBILITY_ALIASES).intersection(records[0])
    assert records[0]["task_id"] == "reaction-to-assay"
    assert records[0]["run_id"].endswith(":seed-2")
    assert isinstance(records[0]["environment_outcome"]["observation"], dict)
    assert isinstance(records[0]["agent_visible_observation"]["observation"], dict)
    assert isinstance(records[0]["evaluation_outcome"]["online_transition_reward"], float)
    measured = [record for record in records if record["instrument"] is not None]
    assert measured
    assert records[0]["task_contract_hash"]
    assert records[0]["runtime_profile_hash"]
    assert records[0]["scoring_contract_hash"]
    assert records[0]["observation_contract_hash"]
    assert len(records[0]["task_contract_hash"]) == 64
    assert len(records[0]["runtime_profile_hash"]) == 64
    assert len(records[0]["scoring_contract_hash"]) == 64
    assert len(records[0]["observation_contract_hash"]) == 64
    assert "raw_signal" in measured[-1]
    assert "processed_estimate" in measured[-1]
    assert "uncertainty" in measured[-1]


def test_trajectory_validation_accepts_v01_compatibility_aliases(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("random"),
        world_split="public-dev",
        budget=2,
        objective="balanced",
        seed=3,
        output_path=path,
    )
    legacy = json.loads(json.dumps(load_jsonl(path)[0]))
    legacy.update({key: trajectory_value(legacy, key) for key in TRAJECTORY_COMPATIBILITY_ALIASES})
    legacy["schema_version"] = "chemworld-trajectory-0.1"
    for field_name in (
        "run_id",
        "environment_outcome",
        "agent_visible_observation",
        "evaluation_outcome",
    ):
        legacy.pop(field_name)
    validate_records([legacy])
