from __future__ import annotations

import json

import pytest
from examples import demo_offline_research as demo

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value


def test_complete_research_retains_failures_seals_forecasts_and_replays(tmp_path, monkeypatch):
    output = tmp_path / "research"
    original = demo.run_unit
    sealed = []

    def observe_order(*args, **kwargs):
        if args[2].startswith("test-"):
            sealed.append((output / "predictions.json").read_bytes())
            assert len(json.loads(sealed[-1])) == 2
        return original(*args, **kwargs)

    monkeypatch.setattr(demo, "run_unit", observe_order)
    report = demo.run(output)
    assert report["accepted"] and not report["formal_result"]
    assert sealed == [(output / "predictions.json").read_bytes()] * 2
    assert report["outcome_counts"]["experiment_count"] == 7
    assert report["outcome_counts"]["operation_count"] == 45
    assert report["outcome_counts"]["rejected_action_count"] == 1
    assert len(report["new_process_replay"]) == 7
    assert all(
        r["verified"] and r["max_abs_error"] == 0 for r in report["new_process_replay"].values()
    )
    assert all(r["yield"] is None and r["final_score"] is None for r in report["episodes"][-2:])
    exported = load_jsonl(output / "dataset.jsonl")
    assert len(exported) == 45
    assert sum(r["transaction_status"] != "committed" for r in exported) == 1
    assert any(
        value is None for r in exported for value in trajectory_value(r, "observation").values()
    )
    for prediction in report["prediction_errors"]:
        assert prediction["absolute_error"] == abs(
            prediction["yield_prediction"] - prediction["observed_yield"]
        )
    notebook = output / report["documents"]["model_notebook"]["relative_path"]
    assert "competing explanations" in notebook.read_text()
    with pytest.raises(FileExistsError):
        demo.run(output)


def test_missing_observation_cannot_become_a_forecast():
    with pytest.raises(ValueError, match="cannot fabricate"):
        demo.forecast(360, [{"setpoint_K": t, "yield": None} for t in (330, 350, 370)])
