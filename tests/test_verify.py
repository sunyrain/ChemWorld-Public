from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemworld.agents.base import BaseAgent
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import make_agent, run_agent
from chemworld.eval.verify import verify_records


class _CompositionCloseoutAgent(BaseAgent):
    def act(self, history):  # type: ignore[no-untyped-def]
        actions = (
            {"operation": "add_solvent", "solvent": 0, "volume_L": 0.025},
            {"operation": "add_reagent", "amount_mol": 0.01},
            {"operation": "measure", "instrument": "ph_meter"},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        )
        return actions[len(history)]


def test_verify_records_accepts_valid_trajectory(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("random"),
        world_split="public-dev",
        budget=3,
        objective="balanced",
        seed=12,
        output_path=path,
    )
    result = verify_records(load_jsonl(path))
    assert result.verified
    assert result.checked_steps == 3


@pytest.mark.parametrize("mutation", ["reorder", "status", "duplicate"])
def test_check_order_is_irrelevant_but_check_content_is_verified(tmp_path, mutation):
    path = tmp_path / "check-order.jsonl"
    run_agent(
        env_id="ChemWorld", agent=make_agent("random"), world_split="public-dev",
        budget=3, objective="balanced", seed=12, output_path=path,
    )
    records = load_jsonl(path)
    checks = records[0]["constitution_checks"]
    checks.reverse()
    if mutation == "status":
        checks[0]["passed"] = not checks[0]["passed"]
    elif mutation == "duplicate":
        checks.append(dict(checks[0]))
    result = verify_records(records)
    assert result.verified == (mutation == "reorder"), result.mismatches
    if mutation != "reorder":
        assert any(m["field"].startswith("constitution_checks.") for m in result.mismatches)


def test_verify_records_restores_composition_campaign_resource_card(tmp_path) -> None:
    composition_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "world-authoring"
        / "composed-equilibrium-characterization-v0.1.json"
    )
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    card = CampaignResourceCard(
        card_id="verify-composition-closeout",
        operation_attempt_limit=5,
        vessel_start_limit=1,
        final_assay_limit=1,
        nonfinal_instrument_use_limit=1,
        stock_limits={"reagent_mol": 0.01, "solvent_L": 0.025},
        per_instrument_limits={"ph_meter": 1},
    )
    path = tmp_path / "composition-closeout.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=_CompositionCloseoutAgent(),
        world_split="public-test",
        budget=5,
        objective="balanced",
        seed=0,
        composition=composition,
        output_path=path,
        campaign_resource_card=card,
        budget_override=5,
        episode_mode_override="campaign",
    )
    records = load_jsonl(path)

    assert records[-1]["terminated"] is True
    result = verify_records(records, tolerance=0.0)
    assert result.verified, result.mismatches


def test_verify_records_rejects_tampered_trajectory(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("random"),
        world_split="public-dev",
        budget=3,
        objective="balanced",
        seed=13,
        output_path=path,
    )
    records = load_jsonl(path)
    records[0]["evaluation_outcome"]["online_transition_reward"] = 999.0
    result = verify_records(records)
    assert not result.verified
    assert result.mismatches


def test_verify_records_rejects_tampered_observation(tmp_path) -> None:
    path = tmp_path / "observation.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    records[-1]["environment_outcome"]["observation"]["score"] = 0.0

    result = verify_records(records)

    assert not result.verified
    assert any(mismatch["field"] == "observation.score" for mismatch in result.mismatches)


def test_verify_records_rejects_mechanism_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "mechanism_hash.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    records[0]["mechanism_hash"] = "tampered-mechanism-hash"

    result = verify_records(records)

    assert not result.verified
    assert any(
        mismatch["step"] == 0 and mismatch["field"] == "mechanism_hash"
        for mismatch in result.mismatches
    )


def test_intervention_trajectory_requires_exact_replay_context(tmp_path) -> None:
    path = tmp_path / "mechanism_intervention.jsonl"
    intervention = {
        "kind": "mechanism_family",
        "mode": "topology_family",
        "severity": 0.8,
    }
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("random"),
        world_split="public-dev",
        budget=3,
        objective="balanced",
        seed=17,
        task_id="flow-reaction-optimization",
        output_path=path,
        budget_override=3,
        world_interventions=[intervention],
    )
    records = load_jsonl(path)

    assert records[0]["mechanism_family_intervention_hash"]
    assert "world_interventions" not in records[0]
    missing_context = verify_records(records)
    exact_context = verify_records(records, world_interventions=[intervention])
    wrong_context = verify_records(
        records,
        world_interventions=[{**intervention, "severity": 0.7}],
    )

    assert not missing_context.verified
    assert any(item["field"] == "world_interventions" for item in missing_context.mismatches)
    assert exact_context.verified, exact_context.mismatches
    assert not wrong_context.verified
    assert any(
        item["field"] in {"mechanism_hash", "mechanism_family_intervention_hash"}
        for item in wrong_context.mismatches
    )


def test_verify_records_rejects_contract_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "contract_hash.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    records[0]["scoring_contract_hash"] = "tampered-scoring-contract-hash"
    records[0]["observation_contract_hash"] = "tampered-observation-contract-hash"

    result = verify_records(records)

    assert not result.verified
    mismatch_fields = {mismatch["field"] for mismatch in result.mismatches}
    assert "scoring_contract_hash" in mismatch_fields
    assert "observation_contract_hash" in mismatch_fields


def test_verify_records_rejects_mid_trajectory_contract_hash_tampering(tmp_path) -> None:
    path = tmp_path / "mid_contract_hash.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    assert len(records) > 2
    records[1]["scoring_contract_hash"] = "tampered-mid-scoring-contract-hash"
    records[1]["runtime_profile_hash"] = "tampered-mid-runtime-profile-hash"

    result = verify_records(records)

    assert not result.verified
    mismatch_fields = {mismatch["field"] for mismatch in result.mismatches}
    assert "scoring_contract_hash" in mismatch_fields
    assert "runtime_profile_hash" in mismatch_fields


def test_verify_records_rejects_task_profile_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "task_profile_hash.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    records[0]["task_contract_hash"] = "tampered-task-contract-hash"
    records[0]["runtime_profile_hash"] = "tampered-runtime-profile-hash"

    result = verify_records(records)

    assert not result.verified
    mismatch_fields = {mismatch["field"] for mismatch in result.mismatches}
    assert "task_contract_hash" in mismatch_fields
    assert "runtime_profile_hash" in mismatch_fields


def test_verify_records_rejects_runtime_v2_transaction_metadata_tampering(
    tmp_path,
) -> None:
    path = tmp_path / "transaction_metadata.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    records[0]["kernel_id"] = "chemworld.operation.fake"
    records[0]["transaction_status"] = "rolled_back"
    records[0]["affected_ledgers"] = ["process", "fake_ledger"]
    records[0]["world_events"][0]["event_type"] = "fake_event"

    result = verify_records(records)

    assert not result.verified
    mismatch_fields = {mismatch["field"] for mismatch in result.mismatches}
    assert "kernel_id" in mismatch_fields
    assert "transaction_status" in mismatch_fields
    assert "affected_ledgers.length" in mismatch_fields
    assert "world_events.0.event_type" in mismatch_fields


def test_verify_records_rejects_runtime_v2_state_patch_tampering(tmp_path) -> None:
    path = tmp_path / "state_patch.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    records[0]["state_patches_summary"][0]["summary"]["delta_cost"] += 0.25

    result = verify_records(records)

    assert not result.verified
    assert any(
        mismatch["field"] == "state_patches_summary.0.summary.delta_cost"
        for mismatch in result.mismatches
    )


def test_verify_records_rejects_invalid_schema() -> None:
    with pytest.raises(ValueError, match="Trajectory is empty"):
        verify_records([])


def test_verify_records_uses_task_budget_for_early_termination(tmp_path) -> None:
    path = tmp_path / "reaction_to_assay.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=path,
    )
    records = load_jsonl(path)
    assert len(records) < records[0]["budget"]
    assert records[-1]["terminated"]
    assert not records[-1]["truncated"]

    result = verify_records(records)
    assert result.verified
    assert result.mismatches == []
