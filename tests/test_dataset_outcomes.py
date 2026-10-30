from __future__ import annotations

import json

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.datasets import (
    export_dataset,
    flatten_record,
)
from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.eval.dataset_cards import (
    dataset_card,
)
from chemworld.eval.verify import verify_records


def write_path(path, actions, **kwargs):
    env = gym.make("ChemWorld", seed=0, **kwargs)
    try:
        env.reset(seed=0)
        task = {**env.unwrapped.task_info(), **env.unwrapped.evaluator_provenance()}
        with TrajectoryLogger(path) as logger:
            for step, action in enumerate(actions, 1):
                obs, reward, terminated, truncated, info = env.step(action)
                logger.log(
                    task_info=task,
                    step=step,
                    action=action,
                    observation=obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata={},
                )
        records = load_jsonl(path)
        assert verify_records(records, tolerance=0.0).verified
        return records, info
    finally:
        env.close()


def outcome_block(root):
    trajectories = root / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=False)
    charge = [
        {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
        {"operation": "add_reagent", "amount_mol": 0.012},
    ]
    negative_actions = [
        *charge,
        {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": 0},
        {
            "operation": "heat",
            "target_temperature_K": 380.0,
            "duration_s": 1800.0,
            "stirring_speed_rpm": 600.0,
        },
        {"operation": "quench"},
        {"operation": "cool_crystallize", "target_temperature_K": 320.0, "duration_s": 3600.0},
        {"operation": "measure", "instrument": "particle_size"},
        {"operation": "filter_crystals"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]
    negative, _ = write_path(
        trajectories / "negative.jsonl",
        negative_actions,
        task_id="reaction-to-crystallization",
        full_process_contract_id="phase-resolved-process-v5",
        episode_mode_override="single_experiment",
        budget_override=40,
    )
    assert negative[-1]["environment_outcome"]["observation"]["crystal_size"] is None
    assert negative[-1]["raw_signal"]["sample_outcome"]["status"] == "negative_result"
    open_records, _ = write_path(trajectories / "open.jsonl", charge, task_id="reaction-to-assay")
    truncated, _ = write_path(
        trajectories / "truncated.jsonl", charge, task_id="reaction-to-assay", budget_override=2
    )
    assert truncated[-1]["truncated"]
    card = CampaignResourceCard(
        card_id="outcome-demo",
        operation_attempt_limit=20,
        vessel_start_limit=3,
        final_assay_limit=3,
        nonfinal_instrument_use_limit=4,
        stock_limits={"solvent_L": 0.1, "reagent_mol": 0.1},
    )
    campaign, info = write_path(
        trajectories / "campaign.jsonl",
        [*charge, {"operation": "discard_batch", "reason": "stop this batch"}, charge[0]],
        task_id="reaction-to-assay",
        episode_mode_override="campaign",
        budget_override=20,
        campaign_resource_card=card,
    )
    assert [r["experiment_index"] for r in campaign] == [0, 0, 0, 1]
    terminal = campaign[2]["environment_outcome"]["last_terminal_summary"]
    assert terminal["outcome"] == "discarded" and not terminal["final_assay"]
    assert terminal["batch_id"] == "batch-0001"
    assert info["campaign_resources"]["state"]["stocks_used"]["solvent_L"] == pytest.approx(0.05)
    assert info["campaign_resources"]["state"]["vessel_starts"] == 2
    export = export_dataset(root, output=root / "export.jsonl", format="jsonl")
    exported = load_jsonl(root / "export.jsonl")
    assert export.record_count == 18 == len(exported)
    assert sum(r["transaction_status"] != "committed" for r in exported) == 1
    assert all(
        r["evaluation_outcome"]["leaderboard_score"] is None
        for r in [*open_records, *truncated, *campaign]
    )
    flat = flatten_record(negative[-1])
    assert json.loads(flat["environment_outcome"])["observation"]["crystal_size"] is None
    assert json.loads(flat["observed_mask"])["crystal_size"] is False
    assert json.loads(flat["sample_outcome"])["status"] == "negative_result"
    assert flat["measurement_cost"] > 0.0 and flat["sample_consumed"] > 0.0
    assert flatten_record(open_records[-1])["leaderboard_score"] is None
    card_payload = dataset_card(root)
    counts = card_payload["outcome_counts"]
    assert counts["operation_count"] == 18
    assert counts["experiment_count"] == 5
    assert counts["final_assay_experiment_count"] == counts["negative_final_assay_count"] == 1
    assert counts["discarded_experiment_count"] == counts["truncated_open_experiment_count"] == 1
    assert counts["open_experiment_count"] == 2
    assert counts["committed_action_count"] == 17 and counts["rejected_action_count"] == 1
    assert counts["unclassified_action_count"] == 0
    assert card_payload["replay_verification"]["group_count"] == 4
    assert card_payload["replay_verification"]["verified"]
    (root / "dataset-card.json").write_text(json.dumps(card_payload, indent=2), encoding="utf-8")
    return counts


def test_negative_discarded_truncated_and_unfinished_experiments_remain_in_export(tmp_path):
    outcome_block(tmp_path)
