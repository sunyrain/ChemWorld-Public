from __future__ import annotations

import gymnasium as gym
import numpy as np

import chemworld  # noqa: F401


def test_env_registers_and_steps() -> None:
    env = gym.make("ChemWorld", world_split="public-dev", budget=3, seed=7)
    obs, info = env.reset(seed=7)
    assert "score" in obs
    assert info["env_id"] == "ChemWorld"

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert 0.0 <= reward <= 1.0
    assert not terminated
    assert not truncated
    assert "constraint_flags" in info
    assert "observed_mask" in info
    if info["observed_mask"].get("yield"):
        assert 0.0 <= float(obs["yield"][0]) <= 1.0
    else:
        assert np.isnan(float(obs["yield"][0]))
    env.close()


def test_budget_truncates_episode() -> None:
    env = gym.make("ChemWorld", budget=1, seed=1)
    env.reset(seed=1)
    _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert truncated
    env.close()

