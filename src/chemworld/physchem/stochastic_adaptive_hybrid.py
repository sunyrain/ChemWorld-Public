"""Hysteresis classification on the caller's sole, unrounded mol inventory.

This module currently defines the adaptive model and its pure rate/classification
view. It does not yet advance a native world or own any material, time or RNG.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from math import isfinite, prod, ulp
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from chemworld.foundation.discrete_inventory import (
    AVOGADRO,
    CONTINUOUS_ADAPTIVE_ALGORITHM,
    MAX_COUNT,
)
from chemworld.physchem.reaction_rate_laws import mass_action_rate
from chemworld.physchem.stochastic_elementary import ElementaryChannel, compile_elementary_channels
from chemworld.physchem.stoichiometric_rates import stoichiometric_amount_rates

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from chemworld.physchem.reaction_network import ReactionNetworkSpec

ADAPTIVE_HYBRID_SCHEMA = "native-adaptive-hybrid-v1"
ADAPTIVE_HYBRID_ALGORITHM = CONTINUOUS_ADAPTIVE_ALGORITHM
MOLECULAR_QUANTUM = 1 / AVOGADRO


def _real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(name + " must be a finite real number")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + " must be " + ("positive" if positive else "nonnegative"))
    return float(value)


@dataclass(frozen=True)
class AdaptiveHybridSpec:
    volume_L: float
    temperature_K: float = 298.15
    lower_limit: float = 800.0
    upper_limit: float = 1000.0
    step_s: float = 0.001
    partition_steps: int = 1
    method: str = "LSODA"
    rtol: float = 1e-8
    atol_particles: float = 1e-10
    max_steps_per_advance: int = 100000
    max_events_per_advance: int = 100000
    max_evaluations_per_advance: int = 1000000

    def __post_init__(self) -> None:
        for name in ("volume_L", "temperature_K", "step_s", "rtol", "atol_particles"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        for name in ("lower_limit", "upper_limit"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if self.lower_limit > self.upper_limit or self.upper_limit > MAX_COUNT:
            raise ValueError(
                "adaptive thresholds require 0 <= lower <= upper <= resolved count limit"
            )
        for name in (
            "partition_steps",
            "max_steps_per_advance",
            "max_events_per_advance",
            "max_evaluations_per_advance",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.method not in {"RK4", "LSODA"}:
            raise ValueError("adaptive hybrid method must be RK4 or LSODA")
        if not isfinite(self.partition_interval_s) or not isfinite(AVOGADRO * self.volume_L):
            raise ValueError("adaptive geometry or partition interval overflows")

    @property
    def partition_interval_s(self) -> float:
        return self.step_s * self.partition_steps

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ADAPTIVE_HYBRID_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdaptiveHybridSpec:
        if not isinstance(payload, Mapping):
            raise ValueError("adaptive hybrid declaration must be an object")
        values = dict(payload)
        if values.pop("schema", None) != ADAPTIVE_HYBRID_SCHEMA:
            raise ValueError("unsupported adaptive hybrid physical schema")
        if set(values) != {f.name for f in fields(cls)}:
            raise ValueError("adaptive hybrid declaration requires its exact fields")
        return cls(**values)


@dataclass(frozen=True)
class AdaptivePartition:
    low_species: tuple[str, ...]
    stochastic_reactions: tuple[str, ...]
    deterministic_reactions: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveHybridModel:
    network: ReactionNetworkSpec
    spec: AdaptiveHybridSpec
    channels: tuple[ElementaryChannel, ...] = field(init=False)
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AdaptiveHybridSpec):
            raise ValueError("adaptive model requires its explicit physical declaration")
        channels = compile_elementary_channels(self.network)
        if self.network.stochastic is not None:
            raise ValueError("adaptive hybrid cannot compete with an integer SSA declaration")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(
            self,
            "signature",
            sha256(
                json.dumps(
                    [self.network.to_dict(), self.spec.to_dict()],
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )

    def populations(self, amounts_mol: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(amounts_mol, Mapping) or set(amounts_mol) != set(
            self.network.species_ids
        ):
            raise ValueError("adaptive mol inventory must cover exactly the original species")
        values = {}
        for key in self.network.species_ids:
            amount = _real(amounts_mol[key], "adaptive amount")
            value = amount / MOLECULAR_QUANTUM
            if not isfinite(value) or value > MAX_COUNT or 8 * AVOGADRO * ulp(amount) > 1e-4:
                raise ValueError("adaptive molecular quantum is not numerically resolved")
            # Recover an exact integer preimage only if it maps to the identical
            # original mol bit pattern. No material is rounded or replaced.
            integer = round(value)
            values[key] = float(integer) if integer * MOLECULAR_QUANTUM == amount else value
        return values

    def partition(
        self, amounts_mol: Mapping[str, float], *, previous_low: Iterable[str] | None = None
    ) -> AdaptivePartition:
        values = self.populations(amounts_mol)
        if previous_low is None:
            middle = (self.spec.lower_limit + self.spec.upper_limit) / 2
            low = {key for key, value in values.items() if value < middle}
        else:
            if isinstance(previous_low, str):
                raise ValueError("adaptive low species must be an identity sequence")
            previous_low = tuple(previous_low)
            if (
                len(previous_low) != len(set(previous_low))
                or not set(previous_low) <= values.keys()
            ):
                raise ValueError("adaptive low species must be unique original identities")
            low = set(previous_low)
            for key, value in values.items():
                if value < self.spec.lower_limit:
                    low.add(key)
                elif value > self.spec.upper_limit:
                    low.discard(key)
        jumps = tuple(c.reaction_id for c in self.channels if c.participants & low)
        deterministic = tuple(c.reaction_id for c in self.channels if c.reaction_id not in jumps)
        return AdaptivePartition(
            tuple(k for k in self.network.species_ids if k in low), jumps, deterministic
        )

    def _vector(self, populations: ArrayLike, *, signed: bool = False) -> dict[str, float]:
        values = np.asarray(populations)
        if values.dtype.kind not in "iuf" or values.shape != (len(self.network.species_ids),):
            raise ValueError("adaptive stage populations must match the original species")
        if not np.isfinite(values).all() or (not signed and np.any(values < 0)):
            raise ValueError("adaptive stage populations must be finite and physically nonnegative")
        return dict(zip(self.network.species_ids, map(float, values), strict=True))

    def deterministic_drift(
        self, populations: ArrayLike, stochastic_reactions: Iterable[str]
    ) -> NDArray[np.float64]:
        values = self._vector(populations, signed=True)
        jumps = set(stochastic_reactions)
        if not jumps <= set(self.network.reaction_ids):
            raise ValueError("adaptive partition has an unknown reaction")
        concentrations = {k: n / (AVOGADRO * self.spec.volume_L) for k, n in values.items()}
        rates = {
            c.reaction_id: 0.0
            if c.reaction_id in jumps
            else mass_action_rate(
                k=c.k,
                concentrations_mol_L=concentrations,
                reactants=dict(c.orders),
                allow_signed_trial=True,
            )
            for c in self.channels
        }
        drift = stoichiometric_amount_rates(
            self.network.species_ids,
            ((c.reaction_id, dict(c.changes)) for c in self.channels),
            rates,
            rate_scale=AVOGADRO * self.spec.volume_L,
        )
        result = np.array([drift[k] for k in self.network.species_ids])
        if not np.isfinite(result).all():
            raise ArithmeticError("adaptive deterministic rate is nonfinite")
        return result

    def propensities(
        self, populations: ArrayLike, stochastic_reactions: Iterable[str]
    ) -> tuple[float, ...]:
        values = self._vector(populations)
        jumps = set(stochastic_reactions)
        if not jumps <= set(self.network.reaction_ids):
            raise ValueError("adaptive partition has an unknown reaction")
        result = []
        for channel in self.channels:
            rate = 0.0
            if channel.reaction_id in jumps and all(
                values[k] >= -n for k, n in channel.changes if n < 0
            ):
                factors = [values[k] - j for k, order in channel.orders for j in range(order)]
                if all(factor >= 0 for factor in factors):
                    order = sum(order for _, order in channel.orders)
                    rate = (
                        channel.k * (AVOGADRO * self.spec.volume_L) ** (1 - order) * prod(factors)
                    )
            if not isfinite(rate) or rate < 0:
                raise ArithmeticError("adaptive propensity is not finite and nonnegative")
            result.append(rate)
        return tuple(result)
