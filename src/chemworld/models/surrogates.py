"""Learnable local world-model interfaces.

These interfaces describe an agent-side belief or surrogate, not the hidden
physical world itself. Keeping them outside `chemworld.foundation` preserves the
boundary between world law and learned local world models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from chemworld.data.records import trajectory_value


@dataclass(frozen=True)
class BeliefState:
    predictions: dict[str, float] = field(default_factory=dict)
    uncertainties: dict[str, float] = field(default_factory=dict)
    observations: int = 0
    best_observed_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> BeliefState:
        """Build a compact belief summary from public trajectory records."""

        scores = [
            float(trajectory_value(record, 'leaderboard_score'))
            for record in records
            if trajectory_value(record, 'leaderboard_score', None) is not None
        ]
        estimates: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        for record in records:
            for key, value in record.get("processed_estimate", {}).items():
                if value is not None:
                    estimates[key] = float(value)
            for key, value in record.get("uncertainty", {}).items():
                if value is not None:
                    uncertainty[key] = float(value)
        return cls(
            predictions=estimates,
            uncertainties=uncertainty,
            observations=len(records),
            best_observed_score=max(scores) if scores else None,
            metadata={
                "source": "trajectory_records",
                "measured_steps": sum(bool(record.get("observed_keys")) for record in records),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions,
            "uncertainties": self.uncertainties,
            "observations": self.observations,
            "best_observed_score": self.best_observed_score,
            "metadata": self.metadata,
        }


class SurrogateModel(Protocol):
    def fit(self, trajectory: list[dict[str, Any]]) -> None:
        """Fit the local world model from trajectory records."""

    def predict(self, action_or_recipe: dict[str, Any]) -> dict[str, float]:
        """Predict observation or score for a candidate experiment."""

    def uncertainty(self, action_or_recipe: dict[str, Any]) -> dict[str, float]:
        """Return uncertainty estimates for a candidate experiment."""

    def recommend(
        self,
        history: list[dict[str, Any]],
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """Recommend the next experiment under constraints."""
