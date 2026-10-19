"""Gymnasium spaces and observation array codec for ChemWorldEnv."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from chemworld.foundation.samples import CONTAINER_IDS
from chemworld.world.actions import CATALYSTS, ELECTROLYTE_PROFILES, SOLVENTS
from chemworld.world.control_contract import CONTROL_CHOICES, CONTROL_NUMERIC
from chemworld.world.operations import (
    INSTRUMENTS,
    OPERATION_TYPES,
    PUBLIC_OBSERVATION_KEYS,
    operation_contracts,
)
from chemworld.world.spectral_contract import SPECTRAL_BOUNDS

OBSERVATION_KEYS = PUBLIC_OBSERVATION_KEYS


class NullableScalarBox(spaces.Box):
    """A scalar Box that treats NaN as a valid missing observation."""

    def contains(self, x: object) -> bool:
        try:
            array = np.asarray(x, dtype=self.dtype)
        except (TypeError, ValueError):
            return False
        if array.shape != self.shape:
            return False
        # NaN is the sole missing-value sentinel in the public observation
        # contract.  Treating +/-Inf as another kind of missing value hides
        # numerical failures from Gym/RL consumers.
        if np.any(np.isinf(array)):
            return False
        finite = np.isfinite(array)
        if not np.any(finite):
            return bool(np.all(np.isnan(array)))
        return bool(
            np.all(array[finite] >= self.low[finite]) and np.all(array[finite] <= self.high[finite])
        )


class ConditionalActionDict(spaces.Dict):
    """A Gym-compatible carrier with operation-conditional semantic samples.

    Gym's ordinary ``Dict.sample`` emits every coordinate.  ChemWorld actions
    are tagged unions, so only the selected operation's declared fields are a
    valid event.  The complete ``spaces`` mapping remains available to vector
    adapters while ``sample`` and ``contains`` respect the tagged-union
    contract.
    """

    def sample(self, mask: Mapping[str, Any] | None = None) -> dict[str, Any]:
        carrier = super().sample(mask=dict(mask) if mask is not None else None)
        operation_index = int(carrier["operation"])
        operation = OPERATION_TYPES[operation_index]
        required = operation_contracts()[operation].required_fields
        return {"operation": operation_index, **{key: carrier[key] for key in required}}

    def contains(self, x: object) -> bool:
        if not isinstance(x, Mapping) or "operation" not in x:
            return False
        operation_value = x["operation"]
        if isinstance(operation_value, (bool, np.bool_)):
            return False
        try:
            operation_index = int(operation_value)
        except (TypeError, ValueError, OverflowError):
            return False
        if not 0 <= operation_index < len(OPERATION_TYPES):
            return False
        required = operation_contracts()[OPERATION_TYPES[operation_index]].required_fields
        if set(x) != {"operation", *required}:
            return False
        if not self.spaces["operation"].contains(operation_value):
            return False
        return all(self.spaces[key].contains(x[key]) for key in required)


def make_action_space() -> spaces.Dict:
    """Build the public ChemWorld action space."""

    return ConditionalActionDict(
        {
            **{
                key: spaces.Box(bounds[0], bounds[1], shape=(1,), dtype=np.float32)
                for key, bounds in SPECTRAL_BOUNDS.items()
            },
            "operation": spaces.Discrete(len(OPERATION_TYPES)),
            "container": spaces.Discrete(len(CONTAINER_IDS)),
            "source_container": spaces.Discrete(len(CONTAINER_IDS)),
            "destination_container": spaces.Discrete(len(CONTAINER_IDS)),
            "capacity_L": spaces.Box(0.0001, 0.1, shape=(1,), dtype=np.float32),
            "mixing": spaces.Discrete(2),
            "component": spaces.Discrete(16),
            **{key: spaces.Discrete(len(choices)) for key, choices in CONTROL_CHOICES.items()},
            **{
                key: spaces.Box(low, high, shape=(1,), dtype=np.float32)
                for key, (low, high, _unit, _default) in CONTROL_NUMERIC.items()
            },
            "vessel": spaces.Discrete(16),
            "connection": spaces.Discrete(64),
            "amount_mol": spaces.Box(0.0, 0.040, shape=(1,), dtype=np.float32),
            "volume_L": spaces.Box(0.0, 0.080, shape=(1,), dtype=np.float32),
            "catalyst_amount_mol": spaces.Box(0.0, 0.005, shape=(1,), dtype=np.float32),
            "target_temperature_K": spaces.Box(250.0, 520.0, shape=(1,), dtype=np.float32),
            "duration_s": spaces.Box(0.0, 14_400.0, shape=(1,), dtype=np.float32),
            "stirring_speed_rpm": spaces.Box(100.0, 1200.0, shape=(1,), dtype=np.float32),
            "sample_volume_L": spaces.Box(0.0, 0.002, shape=(1,), dtype=np.float32),
            "instrument": spaces.Discrete(len(INSTRUMENTS)),
            "catalyst": spaces.Discrete(len(CATALYSTS)),
            "solvent": spaces.Discrete(len(SOLVENTS)),
            "phase": spaces.Discrete(3),
            "target_phase": spaces.Discrete(3),
            "extractant": spaces.Discrete(len(SOLVENTS)),
            "wash_volume_L": spaces.Box(0.0, 0.040, shape=(1,), dtype=np.float32),
            "transfer_fraction": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "seed_mass_g": spaces.Box(0.0, 0.050, shape=(1,), dtype=np.float32),
            "reflux_ratio": spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32),
            "flow_rate_mL_min": spaces.Box(0.0, 20.0, shape=(1,), dtype=np.float32),
            "residence_time_s": spaces.Box(1.0, 7200.0, shape=(1,), dtype=np.float32),
            "potential_V": spaces.Box(-3.0, 3.0, shape=(1,), dtype=np.float32),
            "current_mA": spaces.Box(0.0, 500.0, shape=(1,), dtype=np.float32),
            "electrolyte_profile": spaces.Discrete(len(ELECTROLYTE_PROFILES)),
        }
    )


def make_observation_space() -> spaces.Dict:
    """Build the public ChemWorld observation space."""

    return spaces.Dict(
        {
            key: NullableScalarBox(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            for key in OBSERVATION_KEYS
        }
    )


def value_or_default(values: dict[str, float | None], key: str, default: float = 0.0) -> float:
    value = values.get(key)
    return default if value is None else float(value)


def to_observation(values: dict[str, float | None]) -> dict[str, np.ndarray]:
    def scalar_value(key: str) -> float:
        value = values.get(key)
        if value is None:
            return np.nan
        if isinstance(value, bool):
            raise ValueError(f"observation value {key!r} must be numeric, not boolean")
        resolved = float(value)
        if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError(f"observation value {key!r} must be finite and lie in [0, 1]")
        return resolved

    return {
        key: np.array(
            [scalar_value(key)],
            dtype=np.float32,
        )
        for key in OBSERVATION_KEYS
    }


def empty_observation() -> dict[str, np.ndarray]:
    return {
        key: np.array(
            [0.0 if key in {"cost", "safety_risk", "score"} else np.nan],
            dtype=np.float32,
        )
        for key in OBSERVATION_KEYS
    }


__all__ = [
    "OBSERVATION_KEYS",
    "NullableScalarBox",
    "empty_observation",
    "make_action_space",
    "make_observation_space",
    "to_observation",
    "value_or_default",
]
