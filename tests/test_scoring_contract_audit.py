from __future__ import annotations

from pathlib import Path

import pytest

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.golden import core_golden_targets
from chemworld.eval.runner import make_agent, run_agent
from chemworld.eval.scoring_audit import audit_scoring_contract
from chemworld.tasks import TASK_REGISTRY


@pytest.mark.parametrize(
    "target",
    core_golden_targets(),
    ids=lambda target: target.task_id,
)
def test_core_scoring_contracts_recompute(target, tmp_path: Path) -> None:
    task = TASK_REGISTRY[target.task_id]
    path = tmp_path / f"{target.task_id}_seed{target.seed}.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent(target.agent_name),
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=target.seed,
        task_id=target.task_id,
        output_path=path,
    )

    result = audit_scoring_contract(load_jsonl(path))

    assert result.passed, [failure.to_dict() for failure in result.failures]
    assert result.checked_steps > 0
    assert result.final_assay_count == sum(
        1
        for record in load_jsonl(path)
        if trajectory_value(record, "leaderboard_score", None) is not None
    )
    assert result.final_assay_count >= 1
    assert result.max_score_error <= 1.0e-6
    assert result.max_leaderboard_error <= 1.0e-6
    assert result.max_processed_metric_error <= 1.0e-6


def test_scoring_contract_audit_catches_tampered_score(tmp_path: Path) -> None:
    task = TASK_REGISTRY["reaction-to-assay"]
    path = tmp_path / "reaction_to_assay.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=task.seeds[0],
        task_id=task.task_id,
        output_path=path,
    )
    records = load_jsonl(path)
    trajectory_value(records[-1], 'observation')["score"] = 0.0

    result = audit_scoring_contract(records)

    assert not result.passed
    assert any(failure.field == "observation.score" for failure in result.failures)


def test_scoring_contract_audit_catches_non_final_leaderboard_score(
    tmp_path: Path,
) -> None:
    task = TASK_REGISTRY["reaction-to-assay"]
    path = tmp_path / "reaction_to_assay_nonfinal.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=task.seeds[0],
        task_id=task.task_id,
        output_path=path,
    )
    records = load_jsonl(path)
    records[0]["evaluation_outcome"]["leaderboard_score"] = 0.5

    result = audit_scoring_contract(records)

    assert not result.passed
    assert any(failure.field == "leaderboard_score" for failure in result.failures)
