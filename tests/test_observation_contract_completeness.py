from __future__ import annotations

import json

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.envs.spaces import OBSERVATION_KEYS
from chemworld.eval.metrics import evaluate_records
from chemworld.eval.runner import make_agent, run_agent
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.operations import EQUILIBRIUM_OBSERVATION_KEYS


def test_legacy_instrument_observables_keep_the_fixed_gym_projection() -> None:
    instrument_observables = {
        key
        for spec in DomainModuleRegistry.default().instrument_specs().values()
        if spec.native_observer is None
        for key in spec.contract.observable_keys
    }

    assert instrument_observables <= set(OBSERVATION_KEYS)


def test_equilibrium_metrics_flow_through_gym_rl_and_trajectory(tmp_path) -> None:
    trajectory = tmp_path / "equilibrium.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("codex_subagent_replay"),
        world_split="public-test",
        budget=24,
        objective="balanced",
        seed=0,
        task_id="equilibrium-characterization",
        output_path=trajectory,
    )
    records = load_jsonl(trajectory)
    measurement_records = [
        record
        for record in records
        if set(record["observed_keys"]).intersection(EQUILIBRIUM_OBSERVATION_KEYS)
    ]

    assert measurement_records
    assert any(
        trajectory_value(record, 'observation')["equilibrium_confidence"] is not None
        for record in measurement_records
    )
    latest_agent_view = trajectory_value(measurement_records[-1], 'agent_view')
    rl_keys = latest_agent_view["rl"]["keys"]
    assert "pH_normalized" in rl_keys
    assert "equilibrium_confidence" in rl_keys
    result = evaluate_records(records)
    assert result.mean_equilibrium_confidence > 0.0
    assert result.mean_pH_normalized > 0.0

    serialized = json.dumps(measurement_records[-1])
    assert "pH_normalized" in serialized


def test_equilibrium_observation_space_contains_all_public_equilibrium_keys() -> None:
    env = gym.make("ChemWorld", task_id="equilibrium-characterization", seed=0)
    try:
        observation, _ = env.reset(seed=0)
        assert set(EQUILIBRIUM_OBSERVATION_KEYS) <= set(observation)
        assert set(EQUILIBRIUM_OBSERVATION_KEYS) <= set(env.observation_space.spaces)
    finally:
        env.close()


def test_consecutive_measurements_replace_the_typed_observation_cache() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=7)
    try:
        env.reset(seed=7)
        env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 1})
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        first, _, _, _, _ = env.step({"operation": "measure", "instrument": "hplc"})
        second, _, _, _, _ = env.step({"operation": "measure", "instrument": "hplc"})

        first_yield = float(first["yield"][0])
        second_yield = float(second["yield"][0])
        cached_yield = env.unwrapped._state.process.last_observation["yield"]
        assert first_yield != second_yield
        assert cached_yield == pytest.approx(second_yield)
        assert env.unwrapped._state.process.last_observed_mask["yield"] is True
    finally:
        env.close()
