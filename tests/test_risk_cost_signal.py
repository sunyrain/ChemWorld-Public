from __future__ import annotations

from pathlib import Path

import pytest

from chemworld.agents.random import RandomRecipeAgent
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.risk_policy import (
    RiskCostTaskPolicy,
    load_risk_cost_protocol,
)
from chemworld.eval.runner import run_agent
from chemworld.tasks import SERIOUS_TASK_IDS, get_task


def test_risk_cost_protocol_has_task_specific_limits() -> None:
    protocol = load_risk_cost_protocol()
    assert tuple(protocol["tasks"]) == SERIOUS_TASK_IDS
    policies = [RiskCostTaskPolicy.from_protocol(task_id, protocol) for task_id in SERIOUS_TASK_IDS]
    assert len({policy.risk_limit for policy in policies}) == len(SERIOUS_TASK_IDS)
    assert len({policy.process_cost_limit for policy in policies}) == len(SERIOUS_TASK_IDS)


def test_vnext_task_info_overlay_is_explicit_and_non_physical() -> None:
    protocol = load_risk_cost_protocol()
    policy = RiskCostTaskPolicy.from_protocol("flow-reaction-optimization", protocol)
    overlay = policy.task_info_overlay()
    assert overlay["safety_limit"] == pytest.approx(policy.risk_limit)
    assert overlay["risk_aggregation"] == "max_operation_risk_per_experiment"
    assert "not_real_world_safety" in overlay["risk_limit_semantics"]
    assert len(policy.policy_hash) == 64


def test_official_runner_binds_vnext_policy_to_agent_environment_and_trajectory(
    tmp_path: Path,
) -> None:
    task_id = "partition-discovery"
    policy = RiskCostTaskPolicy.from_protocol(task_id, load_risk_cost_protocol())
    agent = RandomRecipeAgent()
    trajectory = tmp_path / "risk-bound.jsonl"
    history = run_agent(
        env_id="ChemWorld",
        agent=agent,
        world_split="public-dev",
        budget=get_task(task_id).budget,
        objective="balanced",
        seed=3,
        task_id=task_id,
        output_path=trajectory,
        evaluation_policy="vnext_risk_cost",
    )
    records = load_jsonl(trajectory)
    assert history
    assert records
    assert agent.task_info["safety_limit"] == pytest.approx(policy.risk_limit)
    assert agent.task_info["risk_limit_semantics"] == policy.risk_semantics
    assert records[0]["safety_limit"] == pytest.approx(policy.risk_limit)
    assert records[0]["agent_metadata"]["evaluation_policy"] == "vnext_risk_cost"
    assert records[0]["agent_metadata"]["risk_policy_hash"] == policy.policy_hash
    for record in records:
        risk = trajectory_value(record, 'observation')["safety_risk"]
        if risk is not None:
            assert record["constraint_flags"]["unsafe"] is (risk >= policy.risk_limit)


def test_vnext_policy_requires_a_registered_serious_task() -> None:
    with pytest.raises(ValueError, match="registered serious task_id"):
        run_agent(
            env_id="ChemWorld",
            agent=RandomRecipeAgent(),
            world_split="public-dev",
            budget=1,
            objective="balanced",
            seed=0,
            evaluation_policy="vnext_risk_cost",
        )
