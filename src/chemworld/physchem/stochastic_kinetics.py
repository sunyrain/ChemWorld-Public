"""Finite, isothermal Gillespie direct strategy; no session or inventory owner.

For a deterministic extent rate k prod(c_i**r_i), events/s are
k (N_A V)**(1-sum(r_i)) prod((n_i)_{r_i}). Equivalently the stochastic
constant multiplying prod(comb(n_i,r_i)) includes prod(r_i!). No net-rate
subtraction, fractional molecules, automatic ODE switch or hidden upstream.
"""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import comb, factorial, fsum, isfinite, log

import numpy as np

from chemworld.foundation.discrete_inventory import (
    AVOGADRO,
    MAX_COUNT,
    counts_to_mol,
    mol_to_counts,
    validate_counts,
)

__all__ = [
    "AVOGADRO",
    "MAX_COUNT",
    "EventBudgetError",
    "EventChannel",
    "StochasticAdvance",
    "StochasticSpec",
    "advance_stochastic",
    "compile_channels",
    "counts_to_mol",
    "initial_continuation",
    "mol_to_counts",
    "validate_counts",
]


@dataclass(frozen=True)
class StochasticSpec:
    volume_L: float
    temperature_K: float = 298.15
    max_events_per_advance: int = 1_000_000
    algorithm: str = "gillespie-direct-v1"

    def __post_init__(self):
        if any(not isfinite(x) or x <= 0 for x in (self.volume_L, self.temperature_K)):
            raise ValueError("SSA requires finite positive constant volume and temperature")
        if type(self.max_events_per_advance) is not int or self.max_events_per_advance <= 0:
            raise ValueError("SSA event budget must be a positive integer")
        if self.algorithm != "gillespie-direct-v1":
            raise ValueError("unsupported stochastic algorithm")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EventChannel:
    reaction_id: str
    changes: tuple
    reactants: tuple
    constant: float

    def propensity(self, counts):
        value = self.constant
        for key, order in self.reactants:
            n = counts.get(key, 0)
            if n < order:
                return 0.0
            value *= comb(n, order)
        if not isfinite(value) or value < 0:
            raise ValueError("invalid SSA propensity")
        return value


def compile_channels(network, spec):
    if any(
        not float(value).is_integer() for s in network.species for value in s.composition.values()
    ):
        raise ValueError("discrete chemistry requires integer atomic formulas")
    channels = []
    for reaction in network.reactions:
        if (
            reaction.reversible
            or reaction.rate_law.equation_id != "mass_action"
            or reaction.rate_law.uses_activities
            or reaction.kinetic_forward_orders != reaction.reactants
            or set(reaction.rate_law.parameters) != {"k"}
        ):
            raise ValueError(
                "SSA supports elementary irreversible mass_action channels; "
                "declare reverse chemistry as a separate channel"
            )
        if any(not float(v).is_integer() for v in reaction.stoichiometry.values()):
            raise ValueError("SSA stoichiometry must be integer")
        k = reaction.rate_law.parameters["k"]
        if isinstance(k, bool) or not isfinite(k) or k < 0:
            raise ValueError("SSA rate constant must be finite and nonnegative")
        reactants = tuple(sorted((key, int(n)) for key, n in reaction.reactants.items()))
        order = sum(n for _, n in reactants)
        if order > 3:
            raise ValueError("SSA supports elementary molecularity at most three")
        constant = float(k) * (AVOGADRO * spec.volume_L) ** (1 - order)
        for _, n in reactants:
            constant *= factorial(n)
        if not isfinite(constant):
            raise ValueError("SSA stochastic constant overflow")
        channels.append(
            EventChannel(
                reaction.reaction_id,
                tuple(sorted((key, int(n)) for key, n in reaction.stoichiometry.items())),
                reactants,
                constant,
            )
        )
    return tuple(channels)


def initial_continuation(seed, time_s=0.0):
    if type(seed) is not int or seed < 0 or not isfinite(time_s) or time_s < 0:
        raise ValueError("SSA seed/time must be explicit nonnegative values")
    return {
        "algorithm": "gillespie-direct-v1",
        "time_s": float(time_s),
        "rng_state": deepcopy(np.random.PCG64(seed).state),
        "pending": None,
        "signature": None,
        "events_committed": 0,
    }


def _signature(counts, channels, spec):
    return sha256(
        json.dumps(
            [counts, [asdict(c) for c in channels], spec.volume_L, spec.temperature_K],
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StochasticAdvance:
    counts: dict
    continuation: dict
    events: tuple
    actual_duration_s: float


class EventBudgetError(RuntimeError):
    """Numerical work limit, not a physical event; candidate must roll back."""


def advance_stochastic(network, counts, continuation, duration_s):
    """Pure candidate computation. Inputs and RNG are never mutated on failure."""
    if network.stochastic is None:
        raise ValueError("network has not declared a stochastic strategy")
    spec = StochasticSpec(**network.stochastic)
    channels = network.stochastic_channels
    counts = validate_counts(counts)
    if set(counts) - set(network.species_ids):
        raise ValueError("unknown SSA species")
    counts = {key: counts.get(key, 0) for key in network.species_ids}
    state = deepcopy(continuation)
    if not isfinite(duration_s) or duration_s < 0:
        raise ValueError("SSA duration must be finite and nonnegative")
    start = float(state["time_s"])
    if (
        state["algorithm"] != spec.algorithm
        or not isfinite(start)
        or start < 0
        or type(state["events_committed"]) is not int
        or state["events_committed"] < 0
    ):
        raise ValueError("invalid SSA continuation")
    end = start + duration_s
    if not isfinite(end) or (duration_s > 0 and end <= start):
        raise ValueError("SSA endpoint cannot be represented")
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state["rng_state"]
    signature = _signature(counts, channels, spec)
    if state["signature"] != signature:
        # A genuine instantaneous inventory/rate intervention invalidates the
        # old exponential event, but preserves the independent process stream.
        state["pending"] = None
    pending = state["pending"]
    if pending is not None and (
        not isfinite(pending["time_s"])
        or pending["time_s"] < start
        or type(pending["channel"]) is not int
        or not 0 <= pending["channel"] < len(channels)
    ):
        raise ValueError("invalid pending SSA event")
    time = start
    events = []
    while duration_s > 0:
        if pending is None:
            propensities = [c.propensity(counts) for c in channels]
            total = fsum(propensities)
            if not isfinite(total):
                raise ValueError("nonfinite total SSA propensity")
            if total == 0:
                break
            event_time = time - log(1.0 - rng.random()) / total
            if not isfinite(event_time) or event_time <= time:
                raise ArithmeticError("SSA event time cannot be represented")
            target = rng.random() * total
            cumulative = 0.0
            index = len(channels) - 1
            for i, propensity in enumerate(propensities):
                cumulative += propensity
                if target < cumulative:
                    index = i
                    break
            pending = {"time_s": event_time, "channel": index}
        if pending["time_s"] > end:
            break
        if len(events) >= spec.max_events_per_advance:
            raise EventBudgetError("SSA event budget exceeded; candidate not committed")
        time = pending["time_s"]
        channel = channels[pending["channel"]]
        if channel.propensity(counts) <= 0:
            raise ValueError("pending event is incompatible with current inventory")
        for key, change in channel.changes:
            counts[key] += change
        validate_counts(counts)
        events.append((time, channel.reaction_id))
        pending = None
    state.update(
        time_s=end,
        rng_state=deepcopy(rng.bit_generator.state),
        pending=pending,
        signature=_signature(counts, channels, spec),
        events_committed=state["events_committed"] + len(events),
    )
    return StochasticAdvance(counts, state, tuple(events), duration_s)
