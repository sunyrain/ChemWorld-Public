from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.agent_interface import action_schema
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.eval.verify import verify_records
from chemworld.foundation import equipment_settings
from chemworld.foundation.samples import SampleLedger, StoredSample


def cycle_actions(*, dilute=False):
    heat = lambda target, duration: {  # noqa: E731
        "operation": "heat",
        "target_temperature_K": target,
        "duration_s": duration,
        "stirring_speed_rpm": 600,
    }
    cool = {
        "operation": "cool_crystallize",
        "target_temperature_K": 278.15 if dilute else 260.0,
        "duration_s": 1800 if dilute else 7200,
    }
    resuspend = {
        "operation": "resuspend_crystals",
        "volume_L": 0.015 if dilute else 0.001,
        "solvent": 0,
    }
    return [
        resuspend,
        {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
        {"operation": "add_reagent", "amount_mol": 0.012},
        {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": 0},
        heat(350, 900),
        {"operation": "quench"},
        {"operation": "seed_crystals", "seed_mass_g": 0.003 if dilute else 0.03},
        cool,
        *[{"operation": "filter_crystals"}, resuspend, heat(380, 900), cool] * 2,
        {"operation": "filter_crystals"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def total_inventory(state):
    amounts = state.species_amounts.copy()
    for species, amount in state.samples.total_amounts_mol().items():
        amounts[species] = amounts.get(species, 0.0) + amount
    return amounts


def run_cycles(path, seed):
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=seed,
        full_process_contract_id="phase-resolved-process-v5",
        episode_mode_override="single_experiment",
        budget_override=60,
    )
    try:
        env.reset(seed=seed)
        task = {**env.unwrapped.task_info(), **env.unwrapped.evaluator_provenance()}
        rows = []
        seed_reference = None
        with TrajectoryLogger(path) as logger:
            for step, action in enumerate(cycle_actions(), 1):
                before = env.unwrapped._state
                obs, reward, terminated, truncated, info = env.step(action)
                after = env.unwrapped._state
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
                assert info["transaction_status"] == (
                    "rolled_back" if step == 1 else "committed"
                ), (step, action, info.get("preconditions"), info.get("rollback_reason"))
                if step == 1:
                    assert total_inventory(before) == total_inventory(after)
                    assert before.samples == after.samples
                if (
                    step >= 6
                    and action["operation"] != "seed_crystals"
                    and action.get("instrument") != "final_assay"
                ):
                    assert total_inventory(after) == pytest.approx(
                        total_inventory(before), abs=1e-10
                    )
                if action["operation"] == "resuspend_crystals" and step > 1:
                    old_volume = before.volume_L + sum(
                        s.volume_L for s in before.samples.samples.values()
                    )
                    new_volume = after.volume_L + sum(
                        s.volume_L for s in after.samples.samples.values()
                    )
                    assert new_volume == pytest.approx(old_volume + action["volume_L"], abs=1e-12)
                    retained = before.phases.phases["solid"].species_amounts_mol.copy()
                    for species, amount in before.phases.phases[
                        "cake_liquor"
                    ].species_amounts_mol.items():
                        retained[species] = retained.get(species, 0.0) + amount
                    assert after.species_amounts == pytest.approx(retained, abs=1e-12)
                    assert after.ledger.cost > before.ledger.cost
                if step > 8 and action["operation"] == "heat":
                    assert after.metadata["last_energy_transition"]["phase_change_heat_J"] > 0
                    assert after.samples == before.samples
                if action["operation"] == "filter_crystals":
                    assert sum(p.volume_L for p in after.phases.phases.values()) == pytest.approx(
                        after.volume_L
                    )
                if step >= 7:
                    settings = equipment_settings(after.equipment, "crystallizer")
                    seed_total = sum(s.seed_target_mol for s in after.samples.samples.values())
                    seed_total += settings.get("seed_target_mol", 0.0)
                    seed_total += settings.get("dissolved_seed_target_mol", 0.0)
                    if seed_reference is None:
                        seed_reference = seed_total
                    if action.get("instrument") != "final_assay":
                        assert seed_total == pytest.approx(seed_reference, abs=1e-12)
                rows.append(
                    {
                        "step": step,
                        "status": info["transaction_status"],
                        "stored_samples": len(after.samples.samples),
                    }
                )
        assert len(env.unwrapped._state.samples.samples) == 2
        records = load_jsonl(path)
        result = verify_records(records, tolerance=0.0)
        assert result.verified, result.to_dict()
        assert records[-1]["instrument"] == "final_assay"
        assert terminated and not truncated
        return rows
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_recrystallizations_preserve_retained_samples_and_replay(tmp_path, seed):
    run_cycles(tmp_path / "trajectory.jsonl", seed)


def test_sample_ledger_rejects_duplicate_identity_and_copies_input():
    amounts = {"P": 0.01}
    sample = StoredSample("s1", "reactor", "transfer", 0, 0.02, amounts, 298.15, 0, True)
    ledger = SampleLedger().append(sample)
    amounts["P"] = 100
    assert ledger.total_amounts_mol() == {"P": 0.01}
    with pytest.raises(ValueError, match="already exists"):
        ledger.append(sample)


def test_invalid_resuspension_preserves_physical_state(tmp_path):
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        full_process_contract_id="phase-resolved-process-v5",
        budget_override=60,
        episode_mode_override="single_experiment",
    )
    try:
        env.reset(seed=0)
        for action in cycle_actions()[1:9]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        state = env.unwrapped._state
        assert equipment_settings(state.equipment, "crystal_filter")["crystals_filtered"]
        schema = action_schema(env, "resuspend_crystals")
        assert schema["operation"] == "resuspend_crystals"
        for action in [
            {"operation": "resuspend_crystals", "volume_L": 0.2, "solvent": 0},
            {"operation": "resuspend_crystals", "volume_L": float("nan"), "solvent": 0},
            {"operation": "resuspend_crystals", "volume_L": 0.015, "solvent": 99},
        ]:
            assert env.step(action)[-1]["transaction_status"] != "committed"
            assert env.unwrapped._state.phases == state.phases
            assert env.unwrapped._state.samples == state.samples
        assert env.step({"operation": "terminate"})[-1]["transaction_status"] == "committed"
        assert env.step(cycle_actions()[0])[-1]["transaction_status"] != "committed"
    finally:
        env.close()


def test_replay_detects_changed_sample_inventory(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    run_cycles(path, 0)
    records = deepcopy(load_jsonl(path))
    records[-1]["environment_outcome"]["sample_inventory"][0]["volume_L"] += 0.001
    result = verify_records(records)
    assert not result.verified
    assert any("sample_inventory" in item["field"] for item in result.mismatches)


def test_dilute_recrystallization_remains_a_closable_negative_result():
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        full_process_contract_id="phase-resolved-process-v5",
        budget_override=60,
        episode_mode_override="single_experiment",
    )
    try:
        env.reset(seed=0)
        for action in cycle_actions(dilute=True)[1:12]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        assert env.step({"operation": "filter_crystals"})[-1]["transaction_status"] == "rolled_back"
        assert env.step({"operation": "terminate"})[-1]["transaction_status"] == "committed"
        info = env.step({"operation": "measure", "instrument": "final_assay"})[-1]
        assert info["transaction_status"] == "committed"
        assert info["raw_signal"]["sample_outcome"]["status"] == "negative_result"
        assert len(info["sample_inventory"]) == 1
    finally:
        env.close()


def test_campaign_stock_preflight_blocks_unfunded_resuspension():
    card = CampaignResourceCard(
        card_id="resuspension-stock",
        operation_attempt_limit=30,
        vessel_start_limit=2,
        final_assay_limit=2,
        nonfinal_instrument_use_limit=4,
        stock_limits={"solvent_L": 0.025, "reagent_mol": 0.1, "catalyst_mol": 0.1, "seed_g": 0.1},
    )
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        full_process_contract_id="phase-resolved-process-v5",
        budget_override=60,
        campaign_resource_card=card,
    )
    try:
        env.reset(seed=0)
        for action in cycle_actions()[1:9]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        before = env.unwrapped._state
        info = env.step(cycle_actions()[0])[-1]
        assert info["transaction_status"] != "committed"
        assert info["constraint_flags"]["campaign_resource_rejected"]
        assert env.unwrapped._state.samples == before.samples
        assert env.unwrapped._state.phases == before.phases
    finally:
        env.close()


def test_campaign_new_batch_retains_isolated_samples():
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        full_process_contract_id="phase-resolved-process-v5",
        budget_override=60,
        episode_mode_override="campaign",
    )
    try:
        env.reset(seed=0)
        for action in cycle_actions()[1:]:
            info = env.step(action)[-1]
            assert info["transaction_status"] == "committed"
        assert info["next_experiment_ready"]
        stored = env.unwrapped._state.samples
        assert len(stored.samples) == 2
        info = env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 0})[-1]
        assert info["experiment_index"] == 1
        assert env.unwrapped._state.samples == stored
        assert len(info["sample_inventory"]) == 2
    finally:
        env.close()
