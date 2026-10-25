"""Uniform random search baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.recipe_sequence import RecipeSequenceMixin
from chemworld.agents.task_recipes import sample_task_recipe


class RandomAgent(RecipeSequenceMixin, BaseAgent):
    """Uniform random search over complete task-valid experiment recipes."""

    name = "random"

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self.rng = np.random.default_rng(seed)

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending
        return self._start_recipe(sample_task_recipe(self.task_info, self.rng))


class RandomRecipeAgent(RandomAgent):
    """Backward-compatible alias with an explicit recipe-search name."""

    name = "random_recipe"
