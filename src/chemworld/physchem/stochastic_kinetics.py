"""Finite, isothermal Gillespie direct strategy; no session or inventory owner.

For a deterministic extent rate k prod(c_i**r_i), events/s are
k (N_A V)**(1-sum(r_i)) prod((n_i)_{r_i}). Equivalently the stochastic
constant multiplying prod(comb(n_i,r_i)) includes prod(r_i!). No net-rate
subtraction, fractional molecules, automatic ODE switch or hidden upstream.
"""

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
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
    "StochasticLeap",
    "StochasticSpec",
    "advance_stochastic",
    "compile_channels",
    "counts_to_mol",
    "initial_continuation",
    "mol_to_counts",
    "validate_counts",
]

TAU_ALGORITHM = "adaptive-explicit-tau-leap-v1"


@dataclass(frozen=True)
class StochasticSpec:
    volume_L: float
    temperature_K: float = 298.15
    max_events_per_advance: int = 1_000_000
    algorithm: str = "gillespie-direct-v1"
    tau_policy: object = None

    def __post_init__(self) -> None:
        if any(not isfinite(x) or x <= 0 for x in (self.volume_L, self.temperature_K)):
            raise ValueError("SSA requires finite positive constant volume and temperature")
        if type(self.max_events_per_advance) is not int or self.max_events_per_advance <= 0:
            raise ValueError("SSA event budget must be a positive integer")
        if self.algorithm not in {"gillespie-direct-v1", "gibson-bruck-next-reaction-v1",
                                  TAU_ALGORITHM}:
            raise ValueError("unsupported stochastic algorithm")
        if self.algorithm == TAU_ALGORITHM:
            from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy

            policy = self.tau_policy
            if not isinstance(policy, TauLeapPolicy):
                policy = TauLeapPolicy.from_dict(policy)
            object.__setattr__(self, "tau_policy", policy)
        elif self.tau_policy is not None:
            raise ValueError("tau_policy requires the explicit tau-leap algorithm")

    def to_dict(self):
        # Keep the historical exact-method codec and field order unchanged.
        result = {"volume_L": self.volume_L, "temperature_K": self.temperature_K,
                  "max_events_per_advance": self.max_events_per_advance,
                  "algorithm": self.algorithm}
        if self.tau_policy is not None:
            result["tau_policy"] = self.tau_policy.to_dict()
        return result


@dataclass(frozen=True)
class EventProcessSpec:
    """Numerical event policy for already compiled channels, without fake T/V.

    The caller binds its own immutable source and fixed inputs in context. This
    does not declare chemical composition, thermodynamics or a live inventory.
    """

    algorithm: str
    max_events_per_advance: int
    context: str
    tau_policy: object = None

    def __post_init__(self) -> None:
        if self.algorithm not in {"gillespie-direct-v1", "gibson-bruck-next-reaction-v1",
                                  TAU_ALGORITHM}:
            raise ValueError("unsupported compiled event algorithm")
        if type(self.max_events_per_advance) is not int or self.max_events_per_advance < 1:
            raise ValueError("compiled event budget must be a positive integer")
        if not isinstance(self.context, str) or not self.context:
            raise ValueError("compiled channels require an explicit source context")
        if self.algorithm == TAU_ALGORITHM:
            from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy

            if not isinstance(self.tau_policy, TauLeapPolicy):
                object.__setattr__(self, "tau_policy", TauLeapPolicy.from_dict(self.tau_policy))
        elif self.tau_policy is not None:
            raise ValueError("tau_policy requires the explicit tau-leap algorithm")


@dataclass(frozen=True)
class EventChannel:
    reaction_id: str
    changes: tuple
    reactants: tuple
    constant: float

    @property
    def has_regenerated_reactants(self) -> bool:
        """Full reactants include material retained in or produced by this channel."""
        return dict(self.reactants) != {key: -v for key, v in self.changes if v < 0}

    def firing_limit(self, counts: Mapping[str, int]) -> int:
        """Isolated-channel availability, without borrowing other leap products."""
        orders = dict(self.reactants)
        if any(counts.get(key, 0) < n for key, n in self.reactants):
            return 0
        return min(
            (1 + (counts.get(key, 0) - orders[key]) // -change
             for key, change in self.changes if change < 0),
            default=MAX_COUNT,
        )

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
        orders = reaction.kinetic_forward_orders
        if (any(not float(n).is_integer() or n <= 0 for n in orders.values())
                or any(orders.get(key, 0) < n for key, n in reaction.reactants.items())):
            raise ValueError("SSA requires full integer elementary reactants")
        reactants = tuple(sorted((key, int(n)) for key, n in orders.items()))
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


def initial_continuation(seed, time_s=0.0, *, algorithm="gillespie-direct-v1"):
    if type(seed) is not int or seed < 0 or not isfinite(time_s) or time_s < 0:
        raise ValueError("SSA seed/time must be explicit nonnegative values")
    if algorithm not in {"gillespie-direct-v1", "gibson-bruck-next-reaction-v1", TAU_ALGORITHM}:
        raise ValueError("unsupported stochastic algorithm")
    return {
        "algorithm": algorithm,
        "time_s": float(time_s),
        "rng_state": deepcopy(np.random.PCG64(seed).state),
        "pending": None,
        "signature": None,
        "events_committed": 0,
    }


def _signature(counts, channels, spec):
    if isinstance(spec, EventProcessSpec):
        values = ["compiled-event-channels-v1", counts, [asdict(c) for c in channels],
                  spec.context, spec.algorithm]
    else:
        # Historical physical-network signatures and draw order stay exact.
        values = [counts, [asdict(c) for c in channels], spec.volume_L, spec.temperature_K]
    if spec.algorithm == TAU_ALGORITHM:
        values.extend((spec.algorithm, spec.tau_policy.to_dict()))
    return sha256(
        json.dumps(
            values,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StochasticLeap:
    """Aggregate integer firings at a numerical endpoint, not SSA event times."""

    start_time_s: float
    end_time_s: float
    reaction_counts: tuple


@dataclass(frozen=True)
class StochasticAdvance:
    counts: dict
    continuation: dict
    events: tuple
    actual_duration_s: float
    work: dict = field(default_factory=dict)
    batches: tuple = ()


class EventBudgetError(RuntimeError):
    """Numerical work limit, not a physical event; candidate must roll back."""


def _direct_pending(propensities, total, time, rng):
    """Original direct draw order, shared with the declared tau SSA fallback."""
    event_time = time - log(1.0 - rng.random()) / total
    if not isfinite(event_time) or event_time <= time:
        raise ArithmeticError("SSA event time cannot be represented")
    target = rng.random() * total
    cumulative = 0.0
    index = len(propensities) - 1
    for i, propensity in enumerate(propensities):
        cumulative += propensity
        if target < cumulative:
            index = i
            break
    return {"time_s": event_time, "channel": index}


def advance_stochastic(network, counts, continuation, duration_s):
    """Pure candidate computation. Inputs and RNG are never mutated on failure."""
    if network.stochastic is None:
        raise ValueError("network has not declared a stochastic strategy")
    spec = StochasticSpec(**network.stochastic)
    channels = network.stochastic_channels
    return advance_event_channels(channels, network.species_ids, spec,
                                  counts, continuation, duration_s)


def advance_event_channels(channels, species_ids, spec, counts, continuation, duration_s, *,
                           stop_after_boundary=False, critical_reaction_ids=frozenset(),
                           calendar_deadline_s=None, fluctuation_scaled_tau=False):
    """Shared pure direct/next/tau kernel; callers retain state and source units."""
    if type(stop_after_boundary) is not bool:
        raise ValueError("event boundary selection must be an explicit Boolean")
    if type(fluctuation_scaled_tau) is not bool:
        raise ValueError("tau fluctuation selection must be an explicit Boolean")
    if not isinstance(spec, (StochasticSpec, EventProcessSpec)):
        raise ValueError("compiled channels require an explicit event policy")
    counts = validate_counts(counts)
    if set(counts) - set(species_ids):
        raise ValueError("unknown SSA species")
    counts = {key: counts.get(key, 0) for key in species_ids}
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
    if (not isinstance(critical_reaction_ids, frozenset)
            or any(not isinstance(identity, str) for identity in critical_reaction_ids)
            or (critical_reaction_ids and critical_reaction_ids -
                {channel.reaction_id for channel in channels})):
        raise ValueError("critical source reactions must be a frozen set of known channel IDs")
    if calendar_deadline_s is not None:
        from math import ulp

        if (type(calendar_deadline_s) not in {int, float} or not isfinite(calendar_deadline_s)
                or calendar_deadline_s < start):
            raise ValueError("source calendar deadline must be finite and not precede the clock")
        tolerance = 8*max(ulp(end), ulp(calendar_deadline_s))
        if end > calendar_deadline_s+tolerance:
            raise ValueError("requested advance must not cross the source calendar deadline")
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state["rng_state"]
    signature = _signature(counts, channels, spec)
    if spec.algorithm == TAU_ALGORITHM:
        from chemworld.physchem.stochastic_tau_leap import advance_tau_leap

        return advance_tau_leap(channels, counts, state, spec, rng, duration_s, signature,
                                stop_after_boundary=stop_after_boundary,
                                critical_reaction_ids=critical_reaction_ids,
                                calendar_deadline_s=calendar_deadline_s,
                                fluctuation_scaled_tau=fluctuation_scaled_tau)
    if state["signature"] != signature:
        # A genuine instantaneous inventory/rate intervention invalidates the
        # old exponential event, but preserves the independent process stream.
        state["pending"] = None
    if spec.algorithm == "gibson-bruck-next-reaction-v1":
        from chemworld.physchem.stochastic_next_reaction import advance_next_reaction

        return advance_next_reaction(channels, counts, state, spec, rng, duration_s,
                                     stop_after_boundary=stop_after_boundary)
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
    evaluations = draws = 0
    while duration_s > 0:
        if pending is None:
            propensities = [c.propensity(counts) for c in channels]
            evaluations += len(channels)
            total = fsum(propensities)
            if not isfinite(total):
                raise ValueError("nonfinite total SSA propensity")
            if total == 0:
                break
            pending = _direct_pending(propensities, total, time, rng)
            draws += 2
        if pending["time_s"] > end:
            break
        if len(events) >= spec.max_events_per_advance:
            raise EventBudgetError("SSA event budget exceeded; candidate not committed")
        time = pending["time_s"]
        channel = channels[pending["channel"]]
        evaluations += 1
        if channel.propensity(counts) <= 0:
            raise ValueError("pending event is incompatible with current inventory")
        for key, change in channel.changes:
            counts[key] += change
        validate_counts(counts)
        events.append((time, channel.reaction_id))
        pending = None
        if stop_after_boundary:
            end = time
            break
    state.update(
        time_s=end,
        rng_state=deepcopy(rng.bit_generator.state),
        pending=pending,
        signature=_signature(counts, channels, spec),
        events_committed=state["events_committed"] + len(events),
    )
    return StochasticAdvance(
        counts, state, tuple(events), end-start if stop_after_boundary else duration_s,
        {"propensity_evaluations": evaluations, "random_draws": draws},
    )
