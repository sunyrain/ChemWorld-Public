from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
from scripts.check_score_replay import (
    audit_score_replay_protocol,
    load_score_replay_protocol,
)

from chemworld.data.logging import load_jsonl
from chemworld.eval.metrics import evaluate_records
from chemworld.eval.result_artifacts import (
    EVALUATION_RESULT_SCHEMA_VERSION,
    SCORE_REPLAY_BINDING_VERSION,
    build_verified_evaluation_result,
    validate_verified_evaluation_result,
)
from chemworld.eval.runner import run_agent
from chemworld.legacy.agent_registry import make_agent
from chemworld.legacy.leaderboard import aggregate_leaderboard


@pytest.fixture
def verified_result(tmp_path: Path) -> tuple[list[dict], Path, dict]:
    trajectory = tmp_path / "trajectory.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=trajectory,
    )
    records = load_jsonl(trajectory)
    result = build_verified_evaluation_result(records, trajectory_path=trajectory)
    return records, trajectory, result


def test_result_binds_exact_source_records(
    verified_result: tuple[list[dict], Path, dict],
) -> None:
    records, trajectory, result = verified_result
    assert result["result_schema_version"] == EVALUATION_RESULT_SCHEMA_VERSION
    assert result["score_replay"]["schema_version"] == SCORE_REPLAY_BINDING_VERSION
    assert result["score_replay"]["metric_source"] == "exact_trajectory_replay"

    changed = copy.deepcopy(records)
    changed_reward = float(changed[0]["evaluation_outcome"]["online_transition_reward"]) + 0.01
    changed[0]["evaluation_outcome"]["online_transition_reward"] = changed_reward
    with pytest.raises(ValueError, match="bound trajectory source"):
        build_verified_evaluation_result(changed, trajectory_path=trajectory)


def test_result_rejects_metric_binding_and_source_byte_tampering(
    verified_result: tuple[list[dict], Path, dict],
) -> None:
    _, trajectory, result = verified_result
    metric_tamper = copy.deepcopy(result)
    metric_tamper["total_score"] = 1.0 - float(metric_tamper["total_score"])
    with pytest.raises(ValueError, match="does not match trajectory evaluation"):
        validate_verified_evaluation_result(metric_tamper, replay=True)

    binding_tamper = copy.deepcopy(result)
    binding_tamper["score_replay"]["trajectory_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding does not match"):
        validate_verified_evaluation_result(binding_tamper, replay=False)

    trajectory.write_bytes(trajectory.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest"):
        validate_verified_evaluation_result(result, replay=True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_metrics_reject_nonfinite_leaderboard_values(value: float) -> None:
    record = {
        "agent_metadata": {"agent_name": "probe"},
        "env_id": "ChemWorld",
        "world_split": "public-test",
        "seed": 0,
        "constraint_flags": {},
        "leaderboard_score": value,
        "observation": {"cost": 0.1, "safety_risk": 0.1},
    }
    with pytest.raises(ValueError, match="leaderboard_score must be finite"):
        evaluate_records([record])


def test_leaderboard_rejects_nonfinite_scores_even_after_validation(
    verified_result: tuple[list[dict], Path, dict],
) -> None:
    _, _, result = verified_result
    changed = copy.deepcopy(result)
    changed["total_score"] = math.nan
    with pytest.raises(ValueError, match="does not match trajectory evaluation"):
        aggregate_leaderboard([changed])


def test_score_replay_audit(tmp_path: Path) -> None:
    report = audit_score_replay_protocol(
        load_score_replay_protocol(),
        workspace=tmp_path / "audit",
    )
    assert report["controls_ready"] is True
    assert report["status"] == "passed"
    assert all(report["adversarial_probes"].values())
