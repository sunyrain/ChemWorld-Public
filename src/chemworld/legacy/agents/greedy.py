"""Greedy local perturbation baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.interaction import InteractionCapabilities
from chemworld.agents.recipe_sequence import RecipeSequenceMixin
from chemworld.agents.task_recipes import (
    TASK_RECIPE_SPACE_VERSION,
    sample_task_recipe,
    task_recipe_categorical_coordinates,
    task_recipe_from_unit_vector,
    task_recipe_to_vector,
)


class GreedyLocalAgent(RecipeSequenceMixin, BaseAgent):
    name = "greedy_local"
    exploration_probability = 0.20

    def __init__(self, warmup: int = 5, perturbation_scale: float = 0.18) -> None:
        self.warmup = warmup
        self.perturbation_scale = perturbation_scale

    def interaction_capabilities(self) -> InteractionCapabilities:
        capabilities = super().interaction_capabilities()
        return InteractionCapabilities(
            decision_scope=capabilities.decision_scope,
            consumes_intermediate_observations=capabilities.consumes_intermediate_observations,
            consumes_spectra=capabilities.consumes_spectra,
            adapts_within_experiment=capabilities.adapts_within_experiment,
            adapts_across_experiments=True,
            emits_structured_decision_audit=capabilities.emits_structured_decision_audit,
        )

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self.rng = np.random.default_rng(seed)

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending

        recipe_history = self._recipe_history
        if len(recipe_history) < self.warmup:
            return self._start_recipe(sample_task_recipe(self.task_info, self.rng))

        best = max(recipe_history, key=lambda item: item.reward)
        vector = task_recipe_to_vector(best.action)
        categorical = dict(task_recipe_categorical_coordinates(self.task_info))
        continuous_indices = [
            index for index in range(vector.size) if index not in categorical
        ]
        candidate = np.asarray(vector, dtype=float).copy()
        if continuous_indices:
            candidate[continuous_indices] = np.clip(
                candidate[continuous_indices]
                + self.rng.normal(
                    0.0,
                    self.perturbation_scale,
                    size=len(continuous_indices),
                ),
                0.0,
                1.0,
            )
        if self.rng.random() < self.exploration_probability:
            coordinate = int(self.rng.integers(0, candidate.size))
            if coordinate in categorical:
                category_count = categorical[coordinate]
                current = min(int(float(vector[coordinate]) * category_count), category_count - 1)
                alternatives = [item for item in range(category_count) if item != current]
                selected = int(self.rng.choice(alternatives))
                candidate[coordinate] = (selected + 0.5) / category_count
            else:
                candidate[coordinate] = float(self.rng.random())
        recipe = task_recipe_from_unit_vector(self.task_info, candidate)
        return self._start_recipe(recipe)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "search_policy": "task_recipe_local_perturbation",
                "search_space_version": TASK_RECIPE_SPACE_VERSION,
                "recipe_encoding": "continuous_plus_nominal_material_mutation",
                "material_distance_assumption": "none",
                "warmup": self.warmup,
                "perturbation_scale": self.perturbation_scale,
                "exploration_probability": self.exploration_probability,
            }
        )
        return manifest
