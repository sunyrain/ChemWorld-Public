"""Gym adapter over the single native experimental session."""

from typing import Any

import gymnasium as gym
import numpy as np

from chemworld.envs.spaces import (
    OBSERVATION_KEYS,
    make_action_space,
    make_observation_space,
    to_observation,
)
from chemworld.runtime.engine import ChemWorldRuntime

__all__ = ["OBSERVATION_KEYS", "ChemWorldEnv"]


class ChemWorldEnv(ChemWorldRuntime, gym.Env[dict[str, np.ndarray], dict[str, Any]]):
    """Native execution with Gym formatting and reward/ending tuple adaptation.

    Inheritance preserves established evaluator and wrapper state access;
    there is no second session, state copy, or execution loop behind this view.
    """

    def __init__(self, **kwargs: Any) -> None:
        ChemWorldRuntime.__init__(self, **kwargs)
        self.action_space = make_action_space(modules=self.domain_modules)
        self.observation_space = make_observation_space()

    @property
    def _last_observation(self) -> dict[str, np.ndarray]:
        return to_observation(self.observe().values)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        reading, info = ChemWorldRuntime.reset(self, seed=seed, options=options)
        return to_observation(reading.values), info

    def step(self, action):
        result = self.execute(action)
        return (
            to_observation(result.observation.values),
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )
