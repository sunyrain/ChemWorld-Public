"""Pure frozen-propensity adaptive hybrid candidate on the original mol inventory.

All mode checks use a declared absolute time grid. Poisson clocks are paused
while a channel is deterministic; no material is rounded when its label changes.
The caller remains responsible for the original transaction and shared clock.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, log1p
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, cast

import numpy as np

from chemworld.physchem.runge_kutta4 import rk4_step
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.physchem.stochastic_adaptive_hybrid import (
    ADAPTIVE_HYBRID_ALGORITHM,
    MOLECULAR_QUANTUM,
    AdaptiveHybridModel,
    AdaptiveHybridSpec,
    _real,
)
from chemworld.physchem.stochastic_kinetics import EventBudgetError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from chemworld.physchem.reaction_network import ReactionNetworkSpec


class AdaptivePending(TypedDict):
    schema: str
    channel_ids: list[str]
    left_time_s: float
    left_populations: list[float]
    propensities: list[float]
    times_s: list[float | None]
    dormant_hazards: list[float | None]


class _AdaptiveCounters(TypedDict):
    events_committed: int
    segments_committed: int
    partition_checks: int
    draws_committed: int


class AdaptiveContinuation(_AdaptiveCounters):
    algorithm: str
    model_signature: str
    inventory_signature: str
    time_s: float
    grid_index: int
    low_species: list[str]
    rng_state: Mapping[str, Any]
    pending: AdaptivePending | None
    firings: dict[str, int]


class AdaptivePartitionChange(TypedDict):
    time_s: float
    before: tuple[str, ...]
    after: tuple[str, ...]


_PENDING_SCHEMA = "adaptive-frozen-reaction-clocks-v1"
_ClockCounterKey: TypeAlias = Literal[
    "events_committed", "segments_committed", "partition_checks", "draws_committed", "grid_index"
]
_COUNTERS: tuple[
    Literal["events_committed", "segments_committed", "partition_checks", "draws_committed"], ...
] = ("events_committed", "segments_committed", "partition_checks", "draws_committed")


def _signature(model: AdaptiveHybridModel, amounts: Mapping[str, float], time: float) -> str:
    return sha256(
        json.dumps([model.signature, amounts, time], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def initial_adaptive_continuation(
    network: ReactionNetworkSpec, spec: AdaptiveHybridSpec,
    amounts_mol: Mapping[str, float], *, seed: int,
) -> AdaptiveContinuation:
    model = AdaptiveHybridModel(network, spec)
    model.populations(amounts_mol)
    if type(seed) is not int or seed < 0:
        raise ValueError("adaptive process seed must be an explicit nonnegative integer")
    return {
        "algorithm": ADAPTIVE_HYBRID_ALGORITHM,
        "model_signature": model.signature,
        "inventory_signature": _signature(model, amounts_mol, 0.0),
        "time_s": 0.0,
        "grid_index": 0,
        "low_species": list(model.partition(amounts_mol).low_species),
        "rng_state": deepcopy(np.random.PCG64(seed).state),
        "pending": None,
        "firings": dict.fromkeys(network.reaction_ids, 0),
        **cast(_AdaptiveCounters, dict.fromkeys(_COUNTERS, 0)),
    }


class _Work:
    def __init__(self, model: AdaptiveHybridModel) -> None:
        self.model = model
        self.values = {
            "drift_evaluations": 0,
            "propensity_evaluations": 0,
            "segments": 0,
            "events": 0,
            "partition_checks": 0,
            "random_draws": 0,
        }

    def rates(self, populations: ArrayLike, jumps: Iterable[str]) -> tuple[float, ...]:
        self.charge("propensity_evaluations", len(self.model.channels))
        return self.model.propensities(populations, jumps)

    def drift(self, populations: ArrayLike, jumps: Iterable[str]) -> NDArray[np.float64]:
        self.charge("drift_evaluations", 1)
        return self.model.deterministic_drift(populations, jumps)

    def charge(self, field: str, count: int) -> None:
        self.values[field] += count
        total = self.values["drift_evaluations"] + self.values["propensity_evaluations"]
        if total > self.model.spec.max_evaluations_per_advance:
            raise EventBudgetError("adaptive rate evaluation budget exhausted")


def _jumps(model: AdaptiveHybridModel, low: Iterable[str]) -> tuple[str, ...]:
    return tuple(c.reaction_id for c in model.channels if c.participants & set(low))


def _continuation(
    model: AdaptiveHybridModel, amounts: Mapping[str, float],
    continuation: Mapping[str, object], work: _Work, *, verify_rates: bool = True,
) -> tuple[AdaptiveContinuation, np.random.Generator]:
    expected = {
        "algorithm",
        "model_signature",
        "inventory_signature",
        "time_s",
        "grid_index",
        "low_species",
        "rng_state",
        "pending",
        "firings",
        *_COUNTERS,
    }
    state = cast(AdaptiveContinuation, deepcopy(dict(continuation)))
    if set(state) != expected or state["algorithm"] != ADAPTIVE_HYBRID_ALGORITHM:
        raise ValueError("unsupported adaptive hybrid continuation")
    time = _real(state["time_s"], "adaptive continuation time")
    if (
        state["model_signature"] != model.signature
        or state["inventory_signature"] != _signature(model, amounts, time)
        or any(type(state[k]) is not int or state[k] < 0
               for k in cast(tuple[_ClockCounterKey, ...], (*_COUNTERS, "grid_index")))
    ):
        raise ValueError("adaptive continuation differs from the original model/inventory")
    start, end = (
        state["grid_index"] * model.spec.step_s,
        (state["grid_index"] + 1) * model.spec.step_s,
    )
    if not isfinite(end) or not start <= time < end:
        raise ValueError("adaptive clock is outside its canonical numerical interval")
    low = state["low_species"]
    if (
        not isinstance(low, list)
        or len(low) != len(set(low))
        or low != [key for key in model.network.species_ids if key in low]
    ):
        raise ValueError("adaptive species labels must use the original canonical identities")
    firings = state["firings"]
    if (
        not isinstance(firings, dict)
        or set(firings) != set(model.network.reaction_ids)
        or any(type(n) is not int or n < 0 for n in firings.values())
        or sum(firings.values()) != state["events_committed"]
    ):
        raise ValueError("adaptive firing history is invalid")
    rng = np.random.Generator(np.random.PCG64(0))
    rng.bit_generator.state = state["rng_state"]
    pending = state["pending"]
    if pending is None:
        if time != 0 or any(state[k] for k in _COUNTERS) or state["grid_index"] != 0:
            raise ValueError("adaptive pending clocks cannot be erased after advancement")
        if low != list(model.partition(amounts).low_species):
            raise ValueError("fresh adaptive labels disagree with the declared initial midpoint")
        return state, rng
    keys = {
        "schema",
        "channel_ids",
        "left_time_s",
        "left_populations",
        "propensities",
        "times_s",
        "dormant_hazards",
    }
    if (
        not isinstance(pending, dict)
        or set(pending) != keys
        or pending["schema"] != _PENDING_SCHEMA
        or pending["channel_ids"] != list(model.network.reaction_ids)
        or not start <= _real(pending["left_time_s"], "adaptive slab start") <= time
    ):
        raise ValueError("invalid adaptive pending numerical slab")
    model._vector(pending["left_populations"])
    for name in ("propensities", "times_s", "dormant_hazards"):
        if not isinstance(pending[name], list) or len(pending[name]) != len(model.channels):
            raise ValueError("adaptive pending clocks have invalid dimensions")
    rates = tuple(_real(value, "adaptive saved propensity") for value in pending["propensities"])
    if verify_rates and rates != work.rates(pending["left_populations"], _jumps(model, low)):
        raise ValueError("adaptive frozen propensities differ from the saved left state")
    for rate, when, hazard in zip(
        rates, pending["times_s"], pending["dormant_hazards"], strict=True
    ):
        if rate > 0:
            if (
                when is None
                or isinstance(when, bool)
                or not isfinite(when)
                or when <= time
                or hazard is not None
            ):
                raise ValueError("invalid active adaptive reaction clock")
        elif when is not None or (hazard is not None and _real(hazard, "paused hazard") <= 0):
            raise ValueError("invalid paused adaptive reaction clock")
    return state, rng


def validate_adaptive_continuation(
    network: ReactionNetworkSpec, spec: AdaptiveHybridSpec,
    amounts_mol: Mapping[str, float], continuation: Mapping[str, object],
    *, numerical: bool = True,
) -> None:
    model = AdaptiveHybridModel(network, spec)
    model.populations(amounts_mol)
    _continuation(model, amounts_mol, continuation, _Work(model), verify_rates=numerical)


def _pending(
    model: AdaptiveHybridModel, values: Sequence[float] | NDArray[np.float64],
    time: float, low: Iterable[str], old: AdaptivePending | None,
    rng: np.random.Generator, work: _Work, fired: int | None = None,
) -> AdaptivePending:
    rates = work.rates(values, _jumps(model, low))
    times: list[float | None] = []
    dormant: list[float | None] = []
    for index, rate in enumerate(rates):
        hazard = None
        same_time = None
        if old is not None and index != fired:
            previous, when = old["propensities"][index], old["times_s"][index]
            if previous > 0:
                hazard = previous * (cast(float, when) - time)
                if hazard <= 0 or not isfinite(hazard):
                    raise ArithmeticError("adaptive residual hazard cannot be represented")
                if rate == previous:
                    same_time = when
            else:
                hazard = old["dormant_hazards"][index]
        if rate > 0:
            if hazard is None:
                hazard = -log1p(-float(rng.random()))
                work.values["random_draws"] += 1
                if not isfinite(hazard) or hazard <= 0:
                    raise ArithmeticError("adaptive Poisson target cannot be represented")
            when = same_time if same_time is not None else time + hazard / rate
            if not isfinite(when) or when <= time:
                raise ArithmeticError("adaptive reaction time cannot advance its clock")
            times.append(when)
            dormant.append(None)
        else:
            times.append(None)
            dormant.append(hazard)
    return {
        "schema": _PENDING_SCHEMA,
        "channel_ids": list(model.network.reaction_ids),
        "left_time_s": time,
        "left_populations": list(map(float, values)),
        "propensities": list(rates),
        "times_s": times,
        "dormant_hazards": dormant,
    }


@dataclass(frozen=True)
class AdaptiveHybridAdvance:
    amounts_mol: dict[str, float]
    continuation: AdaptiveContinuation
    actual_duration_s: float
    events: tuple[tuple[float, str], ...]
    work: dict[str, int]
    partition_changes: tuple[AdaptivePartitionChange, ...]


@dataclass(frozen=True)
class AdaptiveHybridRebind:
    continuation: AdaptiveContinuation
    work: dict[str, int]
    partition_change: AdaptivePartitionChange | None


def rebind_adaptive_continuation(
    network: ReactionNetworkSpec, spec: AdaptiveHybridSpec,
    old_amounts_mol: Mapping[str, float], new_amounts_mol: Mapping[str, float],
    continuation: Mapping[str, object], *, force_refresh: bool = False,
) -> AdaptiveHybridRebind:
    """Rebind after the caller's actual material transaction; never move its stock.

    The old clock's unspent hazard survives both rate changes and mode changes.
    Previously unused channels draw from the same persisted process stream only
    when they become active. Material receipts remain the caller's responsibility.
    """
    model = AdaptiveHybridModel(network, spec)
    model.populations(old_amounts_mol)
    new = model.populations(new_amounts_mol)
    work = _Work(model)
    state, rng = _continuation(model, old_amounts_mol, continuation, work)
    if type(force_refresh) is not bool:
        raise ValueError("adaptive material refresh policy must be explicitly Boolean")
    if old_amounts_mol == new_amounts_mol and not force_refresh:
        return AdaptiveHybridRebind(state, dict(work.values), None)
    before = tuple(state["low_species"])
    after = model.partition(new_amounts_mol, previous_low=before).low_species
    state["low_species"] = list(after)
    state["partition_checks"] += 1
    work.values["partition_checks"] += 1
    values = [new[k] for k in network.species_ids]
    state["pending"] = _pending(model, values, state["time_s"], after, state["pending"], rng, work)
    state["rng_state"] = deepcopy(rng.bit_generator.state)
    state["draws_committed"] += work.values["random_draws"]
    state["inventory_signature"] = _signature(model, new_amounts_mol, state["time_s"])
    change: AdaptivePartitionChange | None = (
        {"time_s": state["time_s"], "before": before, "after": after} if before != after else None
    )
    return AdaptiveHybridRebind(state, dict(work.values), change)


def advance_adaptive_hybrid(
    network: ReactionNetworkSpec, spec: AdaptiveHybridSpec,
    amounts_mol: Mapping[str, float], continuation: Mapping[str, object], duration_s: float,
) -> AdaptiveHybridAdvance:
    model = AdaptiveHybridModel(network, spec)
    populations = model.populations(amounts_mol)
    values = np.array([populations[k] for k in network.species_ids])
    initial = values.copy()
    duration = _real(duration_s, "adaptive requested duration")
    work = _Work(model)
    state, rng = _continuation(model, amounts_mol, continuation, work)
    time = state["time_s"]
    target = time + duration
    if not isfinite(target) or (duration > 0 and target <= time):
        raise ValueError("adaptive requested endpoint is not representable")
    events: list[tuple[float, str]] = []
    transitions: list[AdaptivePartitionChange] = []

    def amounts() -> dict[str, float]:
        return {
            key: amounts_mol[key]
            if values[i] == initial[i]
            else float(values[i] * MOLECULAR_QUANTUM)
            for i, key in enumerate(network.species_ids)
        }

    if duration and state["pending"] is None:
        state["pending"] = _pending(model, values, time, state["low_species"], None, rng, work)
    while time < target:
        pending = cast(AdaptivePending, state["pending"])
        grid_end = (state["grid_index"] + 1) * spec.step_s
        when, index = min(
            ((t, i) for i, t in enumerate(pending["times_s"]) if t is not None),
            default=(float("inf"), None),
        )
        reached = min(target, grid_end, when)
        if reached <= time:
            raise ArithmeticError("adaptive numerical interval cannot advance its clock")
        if work.values["segments"] >= spec.max_steps_per_advance:
            raise EventBudgetError("adaptive numerical segment budget exhausted")
        jumps = _jumps(model, state["low_species"])
        if len(jumps) < len(model.channels):

            def rhs(
                _at: float, y: NDArray[np.float64], _jumps: tuple[str, ...] = jumps,
            ) -> NDArray[np.float64]:
                return work.drift(y, _jumps)

            if spec.method == "RK4":
                candidate = np.array(rk4_step(rhs, time, values, reached - time).values)
            else:
                report = solve_ode(
                    rhs,
                    values,
                    time_span_s=(time, reached),
                    policy=ODESolverPolicy(
                        "native_adaptive_hybrid_lsoda_v1",
                        method="LSODA",
                        rtol=spec.rtol,
                        atol=spec.atol_particles,
                        max_step_s=spec.step_s,
                    ),
                    evaluation_times_s=[reached],
                )
                report.raise_for_failure("adaptive deterministic interval")
                candidate = report.y[:, -1]
            if not np.isfinite(candidate).all() or np.any(candidate < 0):
                raise ArithmeticError(
                    "adaptive deterministic candidate has negative/nonfinite stock"
                )
            values = candidate
        time = reached
        work.values["segments"] += 1
        state["segments_committed"] += 1
        fired = None
        if when == time:
            if work.values["events"] >= spec.max_events_per_advance:
                raise EventBudgetError("adaptive stochastic event budget exhausted")
            channel = model.channels[cast(int, index)]
            candidate = values.copy()
            for key, change in channel.changes:
                position = network.species_ids.index(key)
                candidate[position] += change
                if candidate[position] == values[position]:
                    raise ArithmeticError("adaptive molecular jump is not numerically resolved")
            if np.any(candidate < 0) or not np.isfinite(candidate).all():
                raise ArithmeticError(
                    "adaptive frozen-hazard jump lacks its actual material quantum"
                )
            values, fired = candidate, index
            state["firings"][channel.reaction_id] += 1
            state["events_committed"] += 1
            work.values["events"] += 1
            events.append((time, channel.reaction_id))
        if time == grid_end:
            state["grid_index"] += 1
            if state["grid_index"] % spec.partition_steps == 0:
                before = tuple(state["low_species"])
                after = model.partition(amounts(), previous_low=before).low_species
                state["low_species"] = list(after)
                state["partition_checks"] += 1
                work.values["partition_checks"] += 1
                if before != after:
                    transitions.append({"time_s": time, "before": before, "after": after})
        if fired is not None or time == grid_end:
            state["pending"] = _pending(
                model, values, time, state["low_species"], pending, rng, work, fired
            )
    final = amounts()
    model.populations(final)
    state["time_s"] = target
    state["rng_state"] = deepcopy(rng.bit_generator.state)
    state["draws_committed"] += work.values["random_draws"]
    state["inventory_signature"] = _signature(model, final, target)
    return AdaptiveHybridAdvance(
        final, state, duration, tuple(events), dict(work.values), tuple(transitions)
    )
