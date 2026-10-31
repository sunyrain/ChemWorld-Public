"""Read the frozen display fixture; replay current scripts against their own version.

The July fixture remains immutable historical evidence. It is not a numerical
oracle for later physical models. Current numerical expectations live in
test_runtime_golden_characterization.py and the domain-specific physical tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.golden import (
    GOLDEN_SUMMARY_SCHEMA_VERSION,
    GoldenTrajectoryTarget,
    core_golden_targets,
)
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.legacy.agent_registry import make_agent
from chemworld.tasks import TASK_REGISTRY

FIXTURE_PATH = Path(__file__).parent / "fixtures/golden/core_scripted_trajectories.json"


def test_historical_core_fixture_remains_readable() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    assert payload["fixture_schema_version"] == "chemworld-golden-fixture-0.1"
    assert {summary["task_id"] for summary in summaries} == {
        "reaction-to-assay", "reaction-to-purification", "partition-discovery",
    }
    assert {summary["summary_schema_version"] for summary in summaries} == {
        GOLDEN_SUMMARY_SCHEMA_VERSION,
    }


@pytest.mark.parametrize("target", core_golden_targets(), ids=lambda target: target.task_id)
def test_current_core_scripts_replay_exactly(
    target: GoldenTrajectoryTarget, tmp_path: Path,
) -> None:
    task = TASK_REGISTRY[target.task_id]
    trajectory_path = tmp_path / f"{target.task_id}_seed{target.seed}.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent(target.agent_name),
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=target.seed,
        task_id=target.task_id,
        output_path=trajectory_path,
    )
    records = load_jsonl(trajectory_path)
    assert records
    assert any(
        trajectory_value(record, "leaderboard_score", None) is not None for record in records
    )
    replay = verify_records(records, tolerance=0.0)
    assert replay.verified, replay.mismatches
    assert replay.checked_steps == len(records)
    assert replay.max_abs_error == 0.0
