"""Exact dependency-updated reaction clocks inside the original SSA candidate.

Absolute active clock times make observation interval splits inert. A channel
whose propensity becomes zero retains its remaining internal Poisson hazard;
reactivation resumes that hazard. RNG, inventory and time remain owned by the
caller's original process ledger, never by this module or its dependency cache.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from heapq import heapify, heappop, heappush
from math import isfinite, log1p
from typing import Any

import numpy as np

from chemworld.physchem.stochastic_kinetics import (
    EventBudgetError,
    EventChannel,
    EventPolicy,
    StochasticAdvance,
    _signature,
    validate_counts,
)

PENDING_SCHEMA = "native-next-reaction-clocks-v1"


@lru_cache(maxsize=128)
def channel_dependencies(channels: tuple[EventChannel, ...]) -> tuple[tuple[int, ...], ...]:
    """Static reactant dependencies; the firing channel always renews its clock."""
    consumers: dict[str, set[int]] = {}
    for index, channel in enumerate(channels):
        for species, _ in channel.reactants:
            consumers.setdefault(species, set()).add(index)
    result = []
    for index, channel in enumerate(channels):
        affected = {index}
        for species, change in channel.changes:
            if change:
                affected.update(consumers.get(species, ()))
        result.append(tuple(sorted(affected)))
    return tuple(result)


def _positive(value: Any) -> bool:
    return type(value) in {int, float} and isfinite(value) and value > 0


def _validate_pending(
    pending: dict[str, Any], channels: tuple[EventChannel, ...],
    propensities: list[float], start: float,
) -> None:
    if not isinstance(pending, dict) or pending.get("schema") != PENDING_SCHEMA:
        raise ValueError("invalid next-reaction pending schema")
    if pending.get("channel_ids") != [c.reaction_id for c in channels]:
        raise ValueError("next-reaction pending channel identities disagree")
    for name in ("propensities", "times_s", "dormant_hazards"):
        if not isinstance(pending.get(name), list) or len(pending[name]) != len(channels):
            raise ValueError("invalid next-reaction pending dimensions")
    for old, expected, when, hazard in zip(
        pending["propensities"], propensities, pending["times_s"], pending["dormant_hazards"],
        strict=True,
    ):
        if type(old) not in {int, float} or not isfinite(old) or old != expected:
            raise ValueError("next-reaction cached propensity disagrees with inventory")
        if old > 0:
            if not _positive(when) or when <= start or hazard is not None:
                raise ValueError("invalid active next-reaction clock")
        elif when is not None or (hazard is not None and not _positive(hazard)):
            raise ValueError("invalid dormant next-reaction clock")


def advance_next_reaction(
    channels: tuple[EventChannel, ...], counts: dict[str, int], state: dict[str, Any],
    spec: EventPolicy, rng: np.random.Generator, duration_s: float,
    *, stop_after_boundary: bool = False,
) -> StochasticAdvance:
    """Consume the validated, copied original SSA candidate; no second runtime."""
    start = float(state["time_s"])
    end = start + duration_s
    evaluations = draws = 0
    pending = state["pending"]

    def draw_hazard() -> float:
        nonlocal draws
        draws += 1
        hazard = -log1p(-float(rng.random()))
        if hazard <= 0 or not isfinite(hazard):
            raise ArithmeticError("next-reaction internal waiting hazard cannot be represented")
        return hazard

    def scheduled(time: float, hazard: float, propensity: float) -> float:
        when = time + hazard / propensity
        if not isfinite(when) or when <= time:
            raise ArithmeticError("next-reaction event time cannot be represented")
        return when

    if pending is not None or duration_s > 0:
        propensities = [float(channel.propensity(counts)) for channel in channels]
        evaluations += len(channels)
        if pending is not None:
            _validate_pending(pending, channels, propensities, start)
        else:
            pending = {
                "schema": PENDING_SCHEMA,
                "channel_ids": [c.reaction_id for c in channels],
                "propensities": propensities,
                "times_s": [
                    scheduled(start, draw_hazard(), value) if value > 0 else None
                    for value in propensities
                ],
                "dormant_hazards": [None] * len(channels),
            }
    events: list[tuple[float, str]] = []
    if pending is not None and duration_s > 0:
        propensities = pending["propensities"]
        times = pending["times_s"]
        hazards = pending["dormant_hazards"]
        dependencies = channel_dependencies(channels)
        queue = [(when, i) for i, when in enumerate(times) if when is not None]
        heapify(queue)
        while queue:
            time, fired = heappop(queue)
            if times[fired] != time:
                continue
            if time > end:
                break
            if len(events) >= spec.max_events_per_advance:
                raise EventBudgetError(
                    "next-reaction event budget exceeded; candidate not committed"
                )
            channel = channels[fired]
            for species, change in channel.changes:
                counts[species] += change
            validate_counts(counts)
            events.append((time, channel.reaction_id))
            for index in dependencies[fired]:
                old = propensities[index]
                value = float(channels[index].propensity(counts))
                evaluations += 1
                if index != fired and value == old:
                    continue
                old_time = times[index]
                if index == fired:
                    hazard = draw_hazard() if value > 0 else None
                elif old > 0:
                    hazard = old * (old_time - time)
                    if not _positive(hazard):
                        raise ArithmeticError("next-reaction residual hazard cannot be represented")
                else:
                    hazard = hazards[index]
                    if value > 0 and hazard is None:
                        hazard = draw_hazard()
                when = None
                if value > 0:
                    if not (hazard is not None):
                        raise ValueError("positive propensity requires a sampled hazard")
                    when = scheduled(time, hazard, value)
                propensities[index] = value
                times[index] = when
                hazards[index] = None if value > 0 else hazard
                if when is not None:
                    heappush(queue, (when, index))
            if len(queue) > 2 * len(channels) + 16:
                queue = [(when, i) for i, when in enumerate(times) if when is not None]
                heapify(queue)
            if stop_after_boundary:
                end = time
                break
    state.update(
        time_s=end, pending=pending, rng_state=deepcopy(rng.bit_generator.state),
        signature=_signature(counts, channels, spec),
        events_committed=state["events_committed"] + len(events),
    )
    return StochasticAdvance(
        counts, state, tuple(events), end-start if stop_after_boundary else duration_s,
        {"propensity_evaluations": evaluations, "random_draws": draws},
    )
