from __future__ import annotations

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.foundation import instrument_equipment_id, upsert_equipment_record
from chemworld.runtime import (
    make_chemworld_constitution,
)
from chemworld.world.state_factory import initial_chemworld_state


def _run_recipe(target_temperature_K: float, duration_s: float) -> tuple[float, dict[str, float]]:
    env = gym.make("ChemWorld", budget=8, seed=4, debug_truth=True)
    env.reset(seed=4)
    actions = [
        {"operation": "add_solvent", "volume_L": 0.03, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.01},
        {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": 1},
        {
            "operation": "heat",
            "target_temperature_K": target_temperature_K,
            "duration_s": duration_s,
            "stirring_speed_rpm": 700.0,
        },
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]
    reward = 0.0
    observation: dict[str, float] = {}
    for action in actions:
        obs, reward, _, _, info = env.step(action)
        observation = {key: float(value[0]) for key, value in obs.items()}
        assert not info["constraint_flags"]["precondition_failed"]
    env.close()
    return reward, observation


def test_registers_and_event_sequence_measures() -> None:
    reward, observation = _run_recipe(385.0, 1200.0)
    assert reward > 0.0
    assert observation["conversion"] > 0.0
    assert observation["cost"] > 0.0


def test_temperature_and_time_have_qualitative_effects() -> None:
    _, cold_short = _run_recipe(330.0, 300.0)
    _, warm = _run_recipe(390.0, 1800.0)
    _, hot_long = _run_recipe(445.0, 5000.0)

    assert warm["conversion"] >= cold_short["conversion"]
    assert hot_long["degradation_warning"] >= warm["degradation_warning"]
    assert hot_long["safety_risk"] >= warm["safety_risk"]


def test_missing_operation_is_retained_as_invalid_transaction() -> None:
    env = gym.make("ChemWorld", budget=2, seed=7)
    env.reset(seed=7)
    try:
        _, reward, terminated, truncated, info = env.step({"temperature": 120.0})
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info["transaction_status"] == "validation_failed"
        assert info["operation_type"] == "invalid"
        assert info["preconditions"]["action_schema_valid"] is False
        assert info["cost_components"]["precondition_failure"] > 0.0
    finally:
        env.close()


def test_failed_final_assay_does_not_leak_observation() -> None:
    env = gym.make("ChemWorld", budget=2, seed=7)
    try:
        obs, _ = env.reset(seed=7)
        del obs
        validation = env.unwrapped.operation_validator.validate(
            {"operation": "measure", "instrument": "final_assay"},
            env.unwrapped._state,
        )
        assert not validation.is_valid
        assert validation.dispatchable_to_runtime
        observation, reward, terminated, truncated, info = env.step(
            {"operation": "measure", "instrument": "final_assay"}
        )

        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert info["instrument"] == "final_assay"
        assert info["instrument_source"] is None
        assert info["observed_keys"] == []
        assert info["leaderboard_score"] is None
        assert info["measurement_cost"] == 0.0
        assert info["sample_consumed"] == 0.0
        assert info["transaction_status"] == "rolled_back"
        assert info["rollback_reason"] == "precondition_failed"
        assert info["world_events"][0]["event_type"] == "operation_rejected"
        assert info["world_events"][-1]["event_type"] == "transaction_rollback"
        assert info["state_patches_summary"][-1]["patch_type"] == "rollback_penalty"
        assert info["constraint_flags"]["precondition_failed"]
        assert "measure_final_requires_terminated" in info["error_message"]
        assert all(not observed for observed in info["observed_mask"].values())
        assert all(value[0] != value[0] for value in observation.values())
    finally:
        env.close()


def test_successful_final_assay_terminates_episode() -> None:
    env = gym.make("ChemWorld", budget=8, seed=4)
    try:
        env.reset(seed=4)
        actions = [
            {"operation": "add_solvent", "volume_L": 0.03, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.01},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1200.0,
                "stirring_speed_rpm": 700.0,
            },
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
        for action in actions[:-1]:
            env.step(action)
        _, _, terminated, truncated, info = env.step(actions[-1])

        assert terminated
        assert not truncated
        assert info["leaderboard_score"] is not None
        with pytest.raises(RuntimeError):
            env.step({"operation": "measure", "instrument": "final_assay"})
    finally:
        env.close()


def test_constitution_rejects_repeated_final_assay() -> None:
    constitution = make_chemworld_constitution()
    state = initial_chemworld_state().replace(
        volume_L=0.02,
        terminated=True,
        equipment=upsert_equipment_record(
            initial_chemworld_state().equipment,
            equipment_id=instrument_equipment_id("final_assay"),
            equipment_type="instrument",
            attached_vessel_id="batch_reactor",
            status="completed",
            settings={"instrument_id": "final_assay", "last_time_s": 0.0},
        ),
    )

    preconditions = constitution.check_preconditions(
        "measure",
        state,
        {"instrument": "final_assay"},
    )

    assert not preconditions["measure_final_not_repeated"]
