"""Stoichiometric PDMP jumps across an explicit discrete/continuous partition.

Continuous solver coordinates are molecule equivalents of the one mol ledger.
Transfer receipts count reactions, never material. Availability roots split the
original ODE solve without consuming or resetting any Poisson threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from math import isfinite, log1p, prod, ulp
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from chemworld.foundation.discrete_inventory import (
    AVOGADRO,
    CROSS_PARTITION_ALGORITHM,
    CROSS_PARTITION_TRANSFER_SCHEMA,
    counts_to_mol,
    validate_cross_partition_transition,
)
from chemworld.physchem.solver_backend import ODEEvent, ODESolverPolicy, solve_ode
from chemworld.physchem.stochastic_hybrid import (
    HAZARD_SCHEMA,
    HybridAdvance,
    HybridChannel,
    HybridEventBudgetError,
    StochasticHybridSpec,
    _amounts,
    _HazardRoot,
    _signature,
    compile_hybrid,
)

if TYPE_CHECKING:
    from chemworld.physchem.reaction_network import ReactionNetworkSpec

CROSS_PARTITION_SCHEMA = "cross-partition-stochastic-hybrid-v1"


@dataclass(frozen=True)
class CrossPartitionHybridSpec(StochasticHybridSpec):
    algorithm: ClassVar[str] = CROSS_PARTITION_ALGORITHM

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "schema": CROSS_PARTITION_SCHEMA}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CrossPartitionHybridSpec:
        values = dict(payload)
        if values.pop("schema", None) != CROSS_PARTITION_SCHEMA:
            raise ValueError("unsupported cross-partition hybrid physical model schema")
        values.setdefault("ode_method", "DOP853")
        if set(values) != {f.name for f in fields(cls)}:
            raise ValueError("cross-partition model requires its exact declared fields")
        return cls(**values)


def transfer_receipt(network: ReactionNetworkSpec, spec: StochasticHybridSpec) -> dict[str, Any]:
    if not isinstance(spec, CrossPartitionHybridSpec):
        raise ValueError("cross-partition execution requires its explicit model schema")
    _, jumps = compile_hybrid(network, spec, allow_cross_partition=True)
    discrete = set(spec.discrete_species)
    channels = {c.reaction_id: dict(c.changes) for c in jumps
                if set(dict(c.changes)) & discrete and set(dict(c.changes)) - discrete}
    if not channels:
        raise ValueError("cross-partition hybrid requires a real cross-partition jump")
    return {"schema": CROSS_PARTITION_TRANSFER_SCHEMA, "channel_stoichiometry": channels,
            "firings": dict.fromkeys(channels, 0)}


def validate_transfer(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                      before: Mapping[str, float], after: Mapping[str, float],
                      before_receipt: dict[str, Any], after_receipt: dict[str, Any]
                      ) -> dict[str, float]:
    declaration = transfer_receipt(network, spec)
    for receipt in (before_receipt, after_receipt):
        if (not isinstance(receipt, dict)
                or receipt.get("channel_stoichiometry") != declaration["channel_stoichiometry"]):
            raise ValueError("cross-partition receipt differs from original reaction network")
    return validate_cross_partition_transition(before, after, spec.discrete_species,
        {s.species_id: s.composition for s in network.species}, before_receipt, after_receipt)


def initial_cross_continuation(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                               amounts_mol: Mapping[str, float], *, seed: int,
                               time_s: float = 0.) -> dict[str, Any]:
    amounts, _ = _amounts(network, spec, amounts_mol)
    receipt = transfer_receipt(network, spec)
    validate_transfer(network, spec, amounts, amounts, receipt, receipt)
    if type(seed) is not int or seed < 0 or time_s != 0:
        raise ValueError("cross-partition initialization requires a fresh seed and zero clock")
    return {"algorithm": CROSS_PARTITION_ALGORITHM,
            "discrete_species": list(spec.discrete_species), "time_s": float(time_s),
            "rng_state": deepcopy(np.random.PCG64(seed).state), "pending": None,
            "signature": _signature(network, spec, amounts), "events_committed": 0,
            "partition_transfer": receipt}


def _continuation(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                  amounts: Mapping[str, float], continuation: Mapping[str, Any]
                  ) -> tuple[dict[str, Any], np.random.Generator]:
    state = deepcopy(dict(continuation))
    if (set(state) != {"algorithm", "discrete_species", "time_s", "rng_state", "pending",
                       "signature", "events_committed", "partition_transfer"}
            or state["algorithm"] != CROSS_PARTITION_ALGORITHM
            or state["discrete_species"] != list(spec.discrete_species)
            or isinstance(state["time_s"], bool) or not isfinite(state["time_s"])
            or state["time_s"] < 0 or type(state["events_committed"]) is not int
            or state["events_committed"] < 0
            or state["signature"] != _signature(network, spec, amounts)):
        raise ValueError("cross-partition continuation does not match original inventory")
    receipt = state["partition_transfer"]
    validate_transfer(network, spec, amounts, amounts, receipt, receipt)
    if sum(receipt["firings"].values()) > state["events_committed"]:
        raise ValueError("cross-partition receipt exceeds committed events")
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state["rng_state"]
    pending = state["pending"]
    if pending is None:
        if state["events_committed"] != 0 or state["time_s"] != 0:
            raise ValueError("cross-partition pending hazards cannot be erased")
    elif (not isinstance(pending, dict)
          or set(pending) != {"schema", "channel_ids", "hazards", "targets"}
          or pending["schema"] != HAZARD_SCHEMA
          or pending["channel_ids"] != list(spec.stochastic_reactions)
          or len(pending["hazards"]) != len(spec.stochastic_reactions)
          or len(pending["targets"]) != len(spec.stochastic_reactions)
          or any(isinstance(h, bool) or isinstance(e, bool) or not isfinite(h)
                 or not isfinite(e) or h < 0 or e <= h
                 for h, e in zip(pending["hazards"], pending["targets"], strict=True))):
        raise ValueError("invalid cross-partition pending integrated hazards")
    return state, rng


@dataclass(frozen=True)
class _AvailabilityRoot:
    index: int
    required: int
    eligible: bool
    terminal: bool = True

    @property
    def direction(self) -> int:
        return -1 if self.eligible else 1

    def __call__(self, _time: float, vector: np.ndarray) -> float:
        value = float(vector[self.index]) - self.required
        # Root-only ULP convention prevents a constant threshold plateau or a
        # just-consumed root from looping at t=0. No material is moved or rounded;
        # actual propensities and jump acceptance still use strict availability.
        epsilon = 8 * ulp(float(self.required))
        return (epsilon if self.eligible else -epsilon) if abs(value) <= epsilon else value


def advance_cross_partition(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                             amounts_mol: Mapping[str, float], continuation: Mapping[str, Any],
                             duration_s: float) -> HybridAdvance:
    if not isinstance(spec, CrossPartitionHybridSpec):
        raise ValueError("cross-partition execution requires its explicit model schema")
    continuous_channels, jumps = compile_hybrid(network, spec, allow_cross_partition=True)
    amounts, counts = _amounts(network, spec, amounts_mol)
    memory, rng = _continuation(network, spec, amounts, continuation)
    start = time = memory["time_s"]
    if isinstance(duration_s, bool) or not isfinite(duration_s) or duration_s < 0:
        raise ValueError("cross-partition duration must be finite and nonnegative")
    end = start + duration_s
    if not isfinite(end) or (duration_s > 0 and end <= start):
        raise ValueError("cross-partition endpoint cannot be represented")
    work = {"random_draws": 0, "rhs_evaluations": 0, "propensity_evaluations": 0,
            "segments": 0, "availability_crossings": 0, "events_committed": 0}
    if duration_s == 0:
        return HybridAdvance(amounts, memory, (), 0., work)

    def draw() -> float:
        value = -log1p(-float(rng.random()))
        work["random_draws"] += 1
        if value <= 0 or not isfinite(value):
            raise ArithmeticError("cross-partition threshold cannot be represented")
        return value

    if memory["pending"] is None:
        memory["pending"] = {"schema": HAZARD_SCHEMA,
            "channel_ids": list(spec.stochastic_reactions),
            "hazards": [0.] * len(jumps), "targets": [draw() for _ in jumps]}
    pending = memory["pending"]
    hazards, targets = np.asarray(pending["hazards"]), np.asarray(pending["targets"])
    keys = tuple(k for k in network.species_ids if k not in spec.discrete_species)
    positions = {key: i for i, key in enumerate(keys)}
    discrete = set(spec.discrete_species)
    quantum_mol = 1 / AVOGADRO
    scale = spec.volume_L / quantum_mol
    boundaries = sorted({(positions[key], -n) for c in jumps for key, n in c.changes
                         if key not in discrete and n < 0})
    # Divide by the representable mol quantum rather than multiplying by N_A:
    # (1/N_A)*N_A can be below one and falsely deactivate a whole final quantum.
    # This is a change of units, never integer rounding of continuous material.
    x = np.asarray([amounts[k] / quantum_mol for k in keys])
    events: list[dict[str, Any]] = []

    def rate(channel: HybridChannel, values: Mapping[str, float]) -> float:
        if channel.stochastic and any(values[k] < -n for k, n in channel.changes
                                      if k not in discrete and n < 0):
            return 0.
        value = channel.k * scale
        for key, order in channel.orders:
            if channel.stochastic and key in discrete:
                n = counts[key]
                if n < order:
                    return 0.
                value *= prod(range(n - order + 1, n + 1)) / scale**order
            else:
                # Only internal signed solver trials use this rate extension.
                # Every accepted ODE node and every final mol stock is checked.
                value *= (max(0., values[key]) / scale)**order
        if not isfinite(value) or value < 0:
            raise ArithmeticError("cross-partition mass-action rate is nonfinite")
        return value

    def rhs(_time: float, vector: np.ndarray) -> np.ndarray:
        work["rhs_evaluations"] += 1
        values = {**counts, **dict(zip(keys, vector[:len(keys)], strict=True))}
        derivatives = dict.fromkeys(keys, 0.)
        for channel in continuous_channels:
            extent = rate(channel, values)
            for key, n in channel.changes:
                derivatives[key] += n * extent
        risks = [rate(c, values) for c in jumps]
        work["propensity_evaluations"] += len(jumps)
        return np.asarray([*(derivatives[k] for k in keys), *risks])

    def availability_sides() -> list[bool]:
        derivative = rhs(time, np.concatenate((x, hazards)))
        return [bool(derivative[i] >= 0) if abs(float(x[i]) - n) <= 8 * ulp(float(n))
                else bool(x[i] >= n) for i, n in boundaries]

    side = availability_sides()
    while time < end:
        roots: tuple[ODEEvent, ...] = (
                 *(_HazardRoot(i, len(keys), targets) for i in range(len(jumps))),
                 *(_AvailabilityRoot(i, n, side[j]) for j, (i, n) in enumerate(boundaries)))
        policy = ODESolverPolicy(CROSS_PARTITION_ALGORITHM, method=spec.ode_method, rtol=spec.rtol,
            atol=spec.atol, max_step_s=spec.max_step_s,
            event_names=(*spec.stochastic_reactions,
                         *(f"available:{keys[i]}:{n}" for i, n in boundaries)))
        report = solve_ode(rhs, np.concatenate((x, hazards)), time_span_s=(time, end),
                           policy=policy, events=roots)
        work["segments"] += 1
        report.raise_for_failure("cross-partition continuous segment")
        if np.any(report.y[:len(keys)] < 0):
            raise ArithmeticError("cross-partition accepted segment has negative inventory")
        terminal = report.terminal_point
        new_time, vector = ((float(report.t[-1]), report.y[:, -1]) if terminal is None
                            else terminal[:2])
        if new_time <= time:
            raise ArithmeticError("cross-partition root cannot advance the clock")
        time = new_time
        x, hazards = vector[:len(keys)].copy(), vector[len(keys):].copy()
        if terminal is None:
            break
        index = terminal[2]
        if index >= len(jumps):
            side[index - len(jumps)] = not side[index - len(jumps)]
            work["availability_crossings"] += 1
            continue
        if len(events) >= spec.max_events_per_advance:
            raise HybridEventBudgetError(events, work)
        channel = jumps[index]
        values = {**counts, **dict(zip(keys, x, strict=True))}
        if rate(channel, values) <= 0:
            raise ArithmeticError("cross-partition jump has no available stoichiometric quantum")
        for key, n in channel.changes:
            if key in discrete:
                counts[key] += n
            else:
                previous = float(x[positions[key]])
                candidate = previous + n
                if candidate < 0 or candidate == previous or not isfinite(candidate):
                    raise ArithmeticError("cross-partition jump cannot represent its quantum")
                x[positions[key]] = candidate
        counts_to_mol(counts)
        side = availability_sides()
        firings = memory["partition_transfer"]["firings"]
        if channel.reaction_id in firings:
            firings[channel.reaction_id] += 1
        hazards[index] = targets[index]
        new_target = float(targets[index]) + draw()
        if new_target <= targets[index]:
            raise ArithmeticError("cross-partition cumulative threshold cannot be represented")
        targets[index] = new_target
        events.append({"time_s": time, "reaction_id": channel.reaction_id})
    amounts.update(counts_to_mol(counts))
    # An unchanged solver coordinate preserves the original mol bit pattern;
    # a no-op must not accumulate conversion round-off over repeated waits.
    amounts.update({k: (amounts[k] if float(v) == amounts[k] / quantum_mol
                        else float(v) * quantum_mol)
                    for k, v in zip(keys, x, strict=True)})
    validate_transfer(network, spec, amounts_mol, amounts, continuation["partition_transfer"],
                      memory["partition_transfer"])
    memory.update(time_s=end, rng_state=deepcopy(rng.bit_generator.state),
                  signature=_signature(network, spec, amounts),
                  events_committed=memory["events_committed"] + len(events))
    pending.update(hazards=hazards.tolist(), targets=targets.tolist())
    _continuation(network, spec, amounts, memory)
    work["events_committed"] = len(events)
    return HybridAdvance(amounts, memory, tuple(events), duration_s, work)
