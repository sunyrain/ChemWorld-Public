"""Latin hypercube search baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.recipe_sequence import RecipeSequenceMixin
from chemworld.agents.task_recipes import (
    task_recipe_dimension,
    task_recipe_event_count,
    task_recipe_from_unit_vector,
)


class LatinHypercubeAgent(RecipeSequenceMixin, BaseAgent):
    name = "latin_hypercube"

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self.rng = np.random.default_rng(seed)
        event_count = task_recipe_event_count(task_info)
        budget = max(1, int(task_info["budget"]) // event_count)
        dims = task_recipe_dimension(task_info)
        design = np.zeros((budget, dims), dtype=float)
        for dim in range(dims):
            bins = (np.arange(budget, dtype=float) + self.rng.random(budget)) / budget
            self.rng.shuffle(bins)
            design[:, dim] = bins
        self._actions = [task_recipe_from_unit_vector(task_info, row) for row in design]

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending

        index = min(len(self._recipe_history), len(self._actions) - 1)
        return self._start_recipe(dict(self._actions[index]))
