"""Nonnegative explicit Poisson leaps on the original stochastic candidate.

CGP species-based selection and critical-channel protection, with declared direct
fallback. Rejected negative proposals are redrawn at half tau; this does not
claim rejection-unbiased paths. Saved endpoint batches give step-held output and
exact numerical-path continuation under arbitrary observation splits.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import fsum, isfinite, log, sqrt
from typing import Any

import numpy as np

from chemworld.physchem.stochastic_kinetics import (
    MAX_COUNT,
    TAU_ALGORITHM,
    EventBudgetError,
    EventChannel,
    StochasticAdvance,
    StochasticLeap,
    StochasticSpec,
    _direct_pending,
    _signature,
    validate_counts,
)

PENDING_SCHEMA = "native-tau-leap-packet-v1"


@dataclass(frozen=True)
class TauLeapPolicy:
    epsilon: float = .03
    max_tau_s: float = .1
    critical_threshold: int = 10
    ssa_threshold: float = 10.
    max_steps_per_advance: int = 100000
    max_rejections_per_leap: int = 32

    def __post_init__(self) -> None:
        for name in ("epsilon", "max_tau_s", "ssa_threshold"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not isfinite(value) or value <= 0:
                raise ValueError(f"tau {name} must be finite and positive")
        if self.epsilon > 1:
            raise ValueError("tau epsilon must be at most one")
        for name in ("critical_threshold", "max_steps_per_advance", "max_rejections_per_leap"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"tau {name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> TauLeapPolicy:
        if not isinstance(value, Mapping):
            raise ValueError("tau-leap requires an explicit tau_policy mapping")
        if set(value) - cls.__dataclass_fields__.keys():
            raise ValueError("unknown tau policy field")
        return cls(**dict(value))


def _critical(channels: tuple[EventChannel, ...], counts: dict[str, int],
              propensities: list[float], policy: TauLeapPolicy,
              critical_reaction_ids: frozenset[str] = frozenset()) -> list[bool]:
    result = []
    for channel, a in zip(channels, propensities, strict=True):
        if channel.reaction_id in critical_reaction_ids:
            result.append(a > 0)
        elif channel.has_regenerated_reactants:
            limit = channel.firing_limit(counts)
            # Protect the last feasible firing even at critical_threshold=1.
            result.append(a > 0 and (limit <= 1 or limit < policy.critical_threshold))
        else:
            # Preserve every original channel's classification and draw order.
            result.append(a > 0 and min(
                (counts[k] // -v for k, v in channel.changes if v < 0),
                default=MAX_COUNT) < policy.critical_threshold)
    return result


def _full_reactant_batch_available(channels: tuple[EventChannel, ...],
                                   counts: dict[str, int], firings: list[int]) -> bool:
    """Conservative extra guard only for newly admitted full-reactant channels.

    Same-leap production is not credited toward these per-channel limits.
    This is still the declared rejection-biased Poisson approximation.
    """
    return all(not n or not channel.has_regenerated_reactants
               or n <= channel.firing_limit(counts)
               for channel, n in zip(channels, firings, strict=True))


def _species_factors(channels: tuple[EventChannel, ...], counts: dict[str, int]
                     ) -> dict[str, float]:
    result: dict[str, float] = {}
    for channel in channels:
        order = sum(r for _, r in channel.reactants)
        for key, multiplicity in channel.reactants:
            n = counts[key]
            # Below a repeated-reactant threshold the finite one-molecule floor
            # controls selection; no division by n-1/n-2 or invented propensity.
            if n < multiplicity:
                factor = float("inf")
            else:
                factor = order / multiplicity * sum(n / (n - j)
                                                    for j in range(multiplicity))
            result[key] = max(result.get(key, 1.), factor)
    return result


def _select_tau(channels: tuple[EventChannel, ...], counts: dict[str, int],
                propensities: list[float], critical: list[bool], policy: TauLeapPolicy,
                fluctuation_scaled_tau: bool = False) -> float:
    factors = _species_factors(channels, counts)
    drift = dict.fromkeys(factors, 0.)
    variance = dict.fromkeys(factors, 0.)
    for channel, a, is_critical in zip(channels, propensities, critical, strict=True):
        if not is_critical:
            for key, change in channel.changes:
                if key in factors:
                    drift[key] += change * a
                    variance[key] += change * change * a
    tau = policy.max_tau_s
    for key, factor in factors.items():
        # Source-boundary profiles also limit changes on the particle noise
        # scale. This is a stricter upper bound, not a global error guarantee.
        population = sqrt(counts[key]) if fluctuation_scaled_tau else counts[key]
        bound = max(policy.epsilon * population / factor, 1.)
        if not isfinite(drift[key]) or not isfinite(variance[key]):
            raise ArithmeticError("tau selection overflow")
        if drift[key]:
            tau = min(tau, bound / abs(drift[key]))
        if variance[key]:
            tau = min(tau, bound * bound / variance[key])
    if not isfinite(tau) or tau <= 0:
        raise ArithmeticError("tau step cannot be represented")
    return tau


def _updated(counts: dict[str, int], channels: tuple[EventChannel, ...],
             firings: list[int]) -> dict[str, int]:
    candidate = dict(counts)
    for channel, number in zip(channels, firings, strict=True):
        for key, change in channel.changes:
            candidate[key] += change * number
    return candidate


def _validate_packet(packet: Any, channels: tuple[EventChannel, ...],
                     counts: dict[str, int], state: dict[str, Any], spec: StochasticSpec,
                     signature: str, critical_reaction_ids: frozenset[str] = frozenset(),
                     fluctuation_scaled_tau: bool = False) -> int:
    policy = spec.tau_policy
    if not isinstance(policy, TauLeapPolicy):
        raise ValueError("tau policy was not compiled")
    fields = {"schema", "algorithm", "policy", "mode", "channel_ids", "signature",
              "start_time_s", "end_time_s", "firings"}
    if not isinstance(packet, dict) or set(packet) != fields:
        raise ValueError("invalid tau pending packet fields")
    # Dict equality alone would accept 10.0 for an integer field (or True for
    # one). Apply the same strict codec used by the authored declaration.
    packet_policy = TauLeapPolicy.from_dict(packet["policy"])
    if (packet["schema"] != PENDING_SCHEMA or packet["algorithm"] != TAU_ALGORITHM
            or packet["policy"] != policy.to_dict()
            or packet_policy != policy
            or packet["mode"] not in {"leap", "ssa"}
            or packet["channel_ids"] != [c.reaction_id for c in channels]
            or not isinstance(packet["signature"], str)
            or packet["signature"] != state["signature"]):
        raise ValueError("tau pending declaration or signature disagrees")
    origin, end = packet["start_time_s"], packet["end_time_s"]
    if (any(type(t) not in {int, float} or not isfinite(t) for t in (origin, end))
            or not 0 <= origin <= state["time_s"] < end
            or (packet["mode"] == "leap" and end > origin + policy.max_tau_s)):
        raise ValueError("invalid tau pending clock interval")
    numbers = packet["firings"]
    if (not isinstance(numbers, list) or len(numbers) != len(channels)
            or any(type(n) is not int or n < 0 or n > MAX_COUNT for n in numbers)
            or (packet["mode"] == "ssa" and sum(numbers) != 1)):
        raise ValueError("invalid tau pending firing counts")
    if signature != state["signature"]:
        return 0  # Genuine inventory/rate intervention discards this valid packet.
    propensities = [c.propensity(counts) for c in channels]
    if any(n and a <= 0 for n, a in zip(numbers, propensities, strict=True)):
        raise ValueError("tau pending firing has zero initial propensity")
    if not _full_reactant_batch_available(channels, counts, numbers):
        raise ValueError("tau pending exceeds full-reactant availability")
    critical = _critical(channels, counts, propensities, policy, critical_reaction_ids)
    if packet["mode"] == "leap":
        if critical_reaction_ids and all(a == 0 or flag for a, flag in
                                         zip(propensities, critical, strict=True)):
            raise ValueError("tau pending leap violates the source critical-channel policy")
        if sum(n for n, flag in zip(numbers, critical, strict=True) if flag) > 1:
            raise ValueError("tau pending has multiple critical firings")
        tau = _select_tau(channels, counts, propensities, critical, policy, fluctuation_scaled_tau)
        if end > origin + tau:
            raise ValueError("tau pending exceeds selected step")
    validate_counts(_updated(counts, channels, numbers))
    return len(channels)


def advance_tau_leap(channels: tuple[EventChannel, ...], counts: dict[str, int],
                     state: dict[str, Any], spec: StochasticSpec, rng: np.random.Generator,
                     duration_s: float, signature: str, *,
                     stop_after_boundary: bool = False,
                     critical_reaction_ids: frozenset[str] = frozenset(),
                     calendar_deadline_s: float | None = None,
                     fluctuation_scaled_tau: bool = False) -> StochasticAdvance:
    """Work only on the caller's copied candidate and its original process RNG."""
    policy = spec.tau_policy
    if not isinstance(policy, TauLeapPolicy):
        raise ValueError("tau policy was not compiled")
    start = float(state["time_s"])
    end = start + duration_s
    if calendar_deadline_s is not None:
        end = min(end, calendar_deadline_s)
    packet = state["pending"]
    validation_evaluations = 0
    if packet is not None:
        validation_evaluations = _validate_packet(
            packet, channels, counts, state, spec, signature, critical_reaction_ids,
            fluctuation_scaled_tau)
    if state["signature"] != signature:
        packet = None
    calendar_replanned = bool(packet is not None and packet["mode"] == "leap"
        and calendar_deadline_s is not None and packet["end_time_s"] > calendar_deadline_s)
    if calendar_replanned:
        # A real new calendar intervention can shorten a previously unknown
        # future interval. Validate first, then discard its uncommitted draws.
        packet = None
    events: list[tuple[float, str]] = []
    batches: list[StochasticLeap] = []
    work = dict.fromkeys(("propensity_evaluations", "random_draws", "poisson_draws",
                          "uniform_draws", "steps_attempted", "rejected_leaps", "accepted_leaps",
                          "ssa_steps", "firings_planned", "firings_committed", "leap_firings"), 0)
    work["propensity_evaluations"] = validation_evaluations
    if calendar_deadline_s is not None:
        work["calendar_replanned_packets"] = int(calendar_replanned)
    time = start

    def attempted() -> None:
        if work["steps_attempted"] >= policy.max_steps_per_advance:
            raise EventBudgetError("tau internal-step budget exceeded; candidate not committed")
        work["steps_attempted"] += 1

    def plan() -> dict[str, Any] | None:
        propensities = [float(c.propensity(counts)) for c in channels]
        work["propensity_evaluations"] += len(channels)
        total = fsum(propensities)
        if not isfinite(total):
            raise ValueError("nonfinite total tau propensity")
        if total == 0:
            return None
        critical = _critical(channels, counts, propensities, policy, critical_reaction_ids)
        tau = _select_tau(channels, counts, propensities, critical, policy, fluctuation_scaled_tau)
        if calendar_deadline_s is not None:
            remaining = calendar_deadline_s-time
            if remaining <= 0:
                raise ValueError("source calendar boundary must settle before another tau proposal")
            tau = min(tau, remaining)
        critical_total = fsum(a for a, flag in zip(propensities, critical, strict=True) if flag)
        rejections = 0
        while True:
            attempted()
            numbers = [0] * len(channels)
            if tau * total < policy.ssa_threshold or critical_total == total:
                event = _direct_pending(propensities, total, time, rng)
                work["uniform_draws"] += 2
                mode, when = "ssa", event["time_s"]
                numbers[event["channel"]] = 1
            else:
                mode = "leap"
                critical_wait = float("inf")
                if critical_total:
                    critical_wait = -log(1. - float(rng.random())) / critical_total
                    work["uniform_draws"] += 1
                    if not isfinite(critical_wait) or critical_wait <= 0:
                        raise ArithmeticError("tau critical waiting time cannot be represented")
                step = min(tau, critical_wait)
                when = time + step
                if calendar_deadline_s is not None:
                    when = min(when, calendar_deadline_s)
                if not isfinite(when) or when <= time:
                    raise ArithmeticError("tau endpoint cannot be represented")
                if critical_wait <= tau:
                    target = float(rng.random()) * critical_total
                    work["uniform_draws"] += 1
                    cumulative = 0.
                    for index, (a, flag) in enumerate(zip(propensities, critical, strict=True)):
                        if flag:
                            cumulative += a
                            if target < cumulative:
                                numbers[index] = 1
                                break
                for index, (a, flag) in enumerate(zip(propensities, critical, strict=True)):
                    if not flag and a > 0:
                        mean = a * step
                        if not isfinite(mean) or mean > MAX_COUNT:
                            raise ArithmeticError("tau Poisson mean exceeds inventory bound")
                        numbers[index] = int(rng.poisson(mean))
                        work["poisson_draws"] += 1
            candidate = _updated(counts, channels, numbers)
            if (any(n < 0 for n in candidate.values())
                    or not _full_reactant_batch_available(channels, counts, numbers)):
                work["rejected_leaps"] += 1
                rejections += 1
                if rejections >= policy.max_rejections_per_leap:
                    raise EventBudgetError("tau rejection budget exceeded; candidate not committed")
                tau /= 2
                continue
            validate_counts(candidate)
            if any(n > MAX_COUNT for n in numbers):
                raise ArithmeticError("tau firing count exceeds representability bound")
            work["firings_planned"] += sum(numbers)
            return {"schema": PENDING_SCHEMA, "algorithm": TAU_ALGORITHM,
                    "policy": policy.to_dict(), "mode": mode,
                    "channel_ids": [c.reaction_id for c in channels],
                    "signature": _signature(counts, channels, spec),
                    "start_time_s": time, "end_time_s": when, "firings": numbers}

    while duration_s > 0:
        if packet is None:
            packet = plan()
        if packet is None or packet["end_time_s"] > end:
            break
        firing_count = sum(packet["firings"])
        if work["firings_committed"] + firing_count > spec.max_events_per_advance:
            raise EventBudgetError("tau firing budget exceeded; candidate not committed")
        counts = _updated(counts, channels, packet["firings"])
        time = packet["end_time_s"]
        work["firings_committed"] += firing_count
        if packet["mode"] == "ssa":
            index = packet["firings"].index(1)
            events.append((time, channels[index].reaction_id))
            work["ssa_steps"] += 1
        else:
            batches.append(StochasticLeap(packet["start_time_s"], time, tuple(
                (c.reaction_id, n) for c, n in zip(channels, packet["firings"], strict=True))))
            work["accepted_leaps"] += 1
            work["leap_firings"] += firing_count
        packet = None
        if stop_after_boundary:
            end = time
            break
    state.update(time_s=end, pending=packet, rng_state=deepcopy(rng.bit_generator.state),
                 signature=_signature(counts, channels, spec),
                 events_committed=state["events_committed"] + work["firings_committed"])
    work["random_draws"] = work["uniform_draws"] + work["poisson_draws"]
    diagnostics: dict[str, Any] = {**work,
        "pending_firings": sum(packet["firings"]) if packet else 0,
        "output_hold_s": (end - packet["start_time_s"]
                          if packet and packet["mode"] == "leap" else 0.),
        "approximate": True, "output_interpolation": "step-held-at-leap-endpoints",
        "rejection_unbiased": False,
        "min_tau_s": min((b.end_time_s - b.start_time_s for b in batches), default=None),
        "max_tau_s": max((b.end_time_s - b.start_time_s for b in batches), default=None)}
    return StochasticAdvance(counts, state, tuple(events),
        end-start if stop_after_boundary else duration_s, diagnostics, tuple(batches))
