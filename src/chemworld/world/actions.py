"""Action catalog and terminal recipe-vector helpers for the shared ChemWorld law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chemworld.foundation.solvents import SOLVENT_IDS

CATALYSTS = ("cat_a", "cat_b", "cat_c", "cat_d")
SOLVENTS = SOLVENT_IDS
ELECTROLYTE_PROFILES = (
    "low_support_acetate",
    "high_support_acetate",
    "acidic_high_transport",
    "precipitation_prone",
)


@dataclass(frozen=True)
class ContinuousBound:
    low: float
    high: float
    unit: str


ACTION_BOUNDS: dict[str, ContinuousBound] = {
    "temperature": ContinuousBound(40.0, 160.0, "degC"),
    "time": ContinuousBound(0.25, 4.0, "h"),
    "initial_concentration": ContinuousBound(0.10, 2.00, "mol/L"),
    "stirring_speed": ContinuousBound(100.0, 1200.0, "rpm"),
}

ACTION_KEYS = (*ACTION_BOUNDS.keys(), "catalyst", "solvent")


def _scalar(value: Any) -> float:
    try:
        values = np.asarray(value).reshape(-1)
        if values.size != 1:
            raise ValueError("action coordinate must be scalar")
        scalar = float(values[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("action coordinate must be scalar and numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError("action coordinates must be finite")
    return scalar


def _categorical_index(value: Any, *, cardinality: int, label: str, clip: bool) -> int:
    coordinate = _scalar(value)
    if not coordinate.is_integer():
        raise ValueError(f"{label} must be an integer categorical index")
    index = int(coordinate)
    if clip:
        return int(np.clip(index, 0, cardinality - 1))
    if not 0 <= index < cardinality:
        raise ValueError(f"{label} index {index} is outside [0, {cardinality - 1}]")
    return index


def canonicalize_action(action: dict[str, Any], *, clip: bool = True) -> dict[str, float | int]:
    """Return a normalized terminal-recipe action dictionary."""

    missing = [key for key in ACTION_KEYS if key not in action]
    if missing:
        raise ValueError(f"Action is missing required keys: {missing}")

    normalized: dict[str, float | int] = {}
    for key, bound in ACTION_BOUNDS.items():
        value = _scalar(action[key])
        if clip:
            value = float(np.clip(value, bound.low, bound.high))
        elif not bound.low <= value <= bound.high:
            raise ValueError(f"{key}={value} is outside [{bound.low}, {bound.high}]")
        normalized[key] = value

    catalyst = _categorical_index(
        action["catalyst"],
        cardinality=len(CATALYSTS),
        label="catalyst",
        clip=clip,
    )
    solvent = _categorical_index(
        action["solvent"],
        cardinality=len(SOLVENTS),
        label="solvent",
        clip=clip,
    )

    normalized["catalyst"] = catalyst
    normalized["solvent"] = solvent
    return normalized


def sample_random_action(rng: np.random.Generator) -> dict[str, float | int]:
    """Sample one uniformly random terminal-recipe action."""

    action: dict[str, float | int] = {
        key: float(rng.uniform(bound.low, bound.high)) for key, bound in ACTION_BOUNDS.items()
    }
    action["catalyst"] = int(rng.integers(0, len(CATALYSTS)))
    action["solvent"] = int(rng.integers(0, len(SOLVENTS)))
    return action


def action_to_vector(action: dict[str, Any]) -> np.ndarray:
    """Map a mixed terminal-recipe action to normalized numeric features."""

    normalized = canonicalize_action(action)
    features: list[float] = []
    for key, bound in ACTION_BOUNDS.items():
        value = float(normalized[key])
        features.append((value - bound.low) / (bound.high - bound.low))

    catalyst = int(normalized["catalyst"])
    solvent = int(normalized["solvent"])
    features.extend(1.0 if catalyst == index else 0.0 for index in range(len(CATALYSTS)))
    features.extend(1.0 if solvent == index else 0.0 for index in range(len(SOLVENTS)))
    return np.asarray(features, dtype=float)


def vector_to_action(vector: np.ndarray) -> dict[str, float | int]:
    """Map a normalized continuous vector to a valid terminal-recipe action."""

    coordinates = np.asarray(vector, dtype=float).reshape(-1)
    if coordinates.size < 6:
        raise ValueError("Expected at least 6 coordinates")
    if not np.all(np.isfinite(coordinates[:6])):
        raise ValueError("Action vector coordinates must be finite")

    action: dict[str, float | int] = {}
    for index, (key, bound) in enumerate(ACTION_BOUNDS.items()):
        coordinate = float(np.clip(coordinates[index], 0.0, 1.0))
        action[key] = bound.low + coordinate * (bound.high - bound.low)
    action["catalyst"] = int(
        np.clip(
            round(float(coordinates[4]) * (len(CATALYSTS) - 1)),
            0,
            len(CATALYSTS) - 1,
        )
    )
    action["solvent"] = int(
        np.clip(
            round(float(coordinates[5]) * (len(SOLVENTS) - 1)),
            0,
            len(SOLVENTS) - 1,
        )
    )
    return action


__all__ = [
    "ACTION_BOUNDS",
    "ACTION_KEYS",
    "CATALYSTS",
    "SOLVENTS",
    "ContinuousBound",
    "action_to_vector",
    "canonicalize_action",
    "sample_random_action",
    "vector_to_action",
]
