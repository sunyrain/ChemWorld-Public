from __future__ import annotations

import json
from pathlib import Path

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.metrics import evaluate_records
from chemworld.eval.runner import run_agent
from chemworld.legacy.agents.llm import LLMCompletionReplayAgent, LLMReplayAgent

PUBLIC_REPLAY_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "fixtures"
    / "llm_replay"
    / "reaction_to_assay_public_trace.jsonl"
)


def test_completion_replay_agent_executes_event_sequence(tmp_path) -> None:
    replay_path = tmp_path / "llm_replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "temperature": 115.0,
                "time": 1.2,
                "initial_concentration": 0.7,
                "stirring_speed": 680.0,
                "catalyst": 1,
                "solvent": 2,
                "hypothesis": "moderate temperature should limit degradation",
                "rationale": "start with a balanced condition",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    history = run_agent(
        env_id="ChemWorld",
        agent=LLMCompletionReplayAgent(replay_path),
        world_split="public-dev",
        budget=6,
        objective="balanced",
        seed=21,
    )

    assert len(history) == 6
    assert history[0].action["operation"] == "add_solvent"
    assert history[-1].action == {"operation": "measure", "instrument": "final_assay"}
    assert LLMCompletionReplayAgent(replay_path).name == "llm_completion_replay"
    assert LLMReplayAgent(PUBLIC_REPLAY_FIXTURE).name == "llm_replay"


def test_public_llm_replay_fixture_is_deterministic_baseline(tmp_path) -> None:
    outputs = [tmp_path / "run_a.jsonl", tmp_path / "run_b.jsonl"]
    histories = []
    records_by_run = []
    for output in outputs:
        histories.append(
            run_agent(
                env_id="ChemWorld",
                agent=LLMReplayAgent(PUBLIC_REPLAY_FIXTURE),
                world_split="public-dev",
                budget=12,
                objective="balanced",
                seed=0,
                task_id="reaction-to-assay",
                output_path=output,
            )
        )
        records_by_run.append(load_jsonl(output))

    actions_a = [record.action for record in histories[0]]
    actions_b = [record.action for record in histories[1]]
    assert actions_a == actions_b
    assert actions_a[-1] == {"operation": "measure", "instrument": "final_assay"}

    scores_a = [trajectory_value(record, 'leaderboard_score', None) for record in records_by_run[0]]
    scores_b = [trajectory_value(record, 'leaderboard_score', None) for record in records_by_run[1]]
    assert scores_a == scores_b
    assert any(score is not None for score in scores_a)

    trace = records_by_run[0][-1]["agent_trace"]
    assert trace[-1]["reasoning_summary"]
    assert trace[-1]["hypothesis_note"]
    assert trace[-1]["memory_note"]
    assert trace[-1]["validator_result"]
    assert trace[-1]["observation_summary"]

    metrics_a = evaluate_records(records_by_run[0]).to_dict()
    metrics_b = evaluate_records(records_by_run[1]).to_dict()
    for key in (
        "final_best_score",
        "area_under_best_score",
        "invalid_action_count",
        "final_assay_count",
        "total_score",
    ):
        assert metrics_a[key] == metrics_b[key]
