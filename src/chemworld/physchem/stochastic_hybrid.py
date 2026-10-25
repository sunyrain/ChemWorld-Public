"""Fixed-partition chemical PDMP on a view of the caller's sole mol inventory.

Continuous chemistry and molecular jumps share one reaction network. Integrated
Poisson hazards, not frozen-rate waiting times, locate jumps. No thermal model,
automatic repartition, inventory clipping, or hidden material reservoir is used.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from math import fsum, isfinite, log1p, prod
from typing import TYPE_CHECKING, Any

import numpy as np

from chemworld.foundation.discrete_inventory import AVOGADRO, counts_to_mol, mol_to_counts
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.physchem.stochastic_kinetics import EventBudgetError

if TYPE_CHECKING:
    from chemworld.physchem.reaction_network import ReactionNetworkSpec

HYBRID_ALGORITHM = "integrated-hazard-pdmp-v1"
HYBRID_SCHEMA = "fixed-partition-stochastic-hybrid-v1"
HAZARD_SCHEMA = "integrated-poisson-hazards-v1"


@dataclass(frozen=True)
class StochasticHybridSpec:
    discrete_species: tuple[str, ...]
    stochastic_reactions: tuple[str, ...]
    volume_L: float
    temperature_K: float = 298.15
    max_events_per_advance: int = 10000
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step_s: float | None = None
    ode_method: str = "DOP853"

    def __post_init__(self) -> None:
        if not isinstance(self.ode_method, str) or self.ode_method not in {"DOP853", "RK45"}:
            raise ValueError("unknown hybrid deterministic integration method")
        for name in ("discrete_species", "stochastic_reactions"):
            values = getattr(self, name)
            if isinstance(values, str):
                raise ValueError("hybrid partition requires a sequence of identities")
            values = tuple(values)
            if (not values or len(values) != len(set(values))
                    or any(not isinstance(v, str) or not v for v in values)):
                raise ValueError("hybrid partition requires unique nonempty identities")
            object.__setattr__(self, name, values)
        for value in (self.volume_L, self.temperature_K, self.rtol, self.atol):
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError("hybrid geometry, temperature and tolerances must be positive")
        if (type(self.max_events_per_advance) is not int or self.max_events_per_advance < 1
                or (self.max_step_s is not None and (
                    isinstance(self.max_step_s, bool) or not isfinite(self.max_step_s)
                    or self.max_step_s <= 0))):
            raise ValueError("hybrid work budget and maximum step must be positive")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.ode_method == "DOP853":
            values.pop("ode_method")
        return {"schema": HYBRID_SCHEMA, **values,
                "discrete_species": list(self.discrete_species),
                "stochastic_reactions": list(self.stochastic_reactions)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StochasticHybridSpec:
        values = dict(payload)
        if values.pop("schema", None) != HYBRID_SCHEMA:
            raise ValueError("unsupported hybrid physical model schema")
        values.setdefault("ode_method", "DOP853")
        if set(values) != {f.name for f in fields(cls)}:
            raise ValueError("hybrid physical model requires its exact declared fields")
        return cls(**values)


@dataclass(frozen=True)
class HybridChannel:
    reaction_id: str
    changes: tuple[tuple[str, int], ...]
    orders: tuple[tuple[str, int], ...]
    k: float
    stochastic: bool


def compile_hybrid(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                   *, allow_cross_partition: bool = False
                   ) -> tuple[tuple[HybridChannel, ...], tuple[HybridChannel, ...]]:
    """Validate separate net stocks and explicit zero-net catalytic orders."""
    discrete, jumps = set(spec.discrete_species), set(spec.stochastic_reactions)
    if (network.stochastic is not None or not discrete < set(network.species_ids)
            or not jumps < set(network.reaction_ids)):
        raise ValueError("hybrid requires explicit proper partitions and no competing SSA model")
    if any(not float(n).is_integer() for s in network.species for n in s.composition.values()):
        raise ValueError("hybrid chemistry requires integer atomic formulas")
    compiled = []
    for reaction in network.reactions:
        orders = reaction.kinetic_forward_orders
        if (reaction.reversible or reaction.rate_law.equation_id != "mass_action"
                or reaction.rate_law.uses_activities
                or set(reaction.rate_law.parameters) != {"k"}
                or any(not float(n).is_integer() for n in reaction.stoichiometry.values())
                or any(not float(n).is_integer() or n <= 0 for n in orders.values())
                or sum(orders.values()) > 3
                or any(orders.get(key) != n for key, n in reaction.reactants.items())
                or any(key in reaction.stoichiometry
                       for key in orders.keys() - reaction.reactants.keys())):
            raise ValueError("hybrid requires elementary mass action with zero-net catalysts")
        stochastic = reaction.reaction_id in jumps
        changed = set(reaction.stoichiometry)
        if (stochastic and not allow_cross_partition and not changed <= discrete) or (
                not stochastic and changed & discrete):
            raise ValueError("hybrid channel changes inventory outside its fixed partition")
        k = reaction.rate_law.parameters["k"]
        if isinstance(k, bool) or not isfinite(k) or k < 0:
            raise ValueError("hybrid mass-action constant must be finite and nonnegative")
        compiled.append(HybridChannel(reaction.reaction_id,
            tuple(sorted((key, int(n)) for key, n in reaction.stoichiometry.items())),
            tuple(sorted((key, int(n)) for key, n in orders.items())), float(k), stochastic))
    # Declaration order, rather than incidental network order, owns Poisson streams.
    return (tuple(c for c in compiled if not c.stochastic),
            tuple(next(c for c in compiled if c.reaction_id == key)
                  for key in spec.stochastic_reactions))


def _amounts(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
             values: Mapping[str, float]) -> tuple[dict[str, float], dict[str, int]]:
    if set(values) != set(network.species_ids):
        raise ValueError("hybrid inventory must cover exactly the declared species")
    if any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("hybrid inventory must be finite and nonnegative")
    amounts = {key: float(values[key]) for key in network.species_ids}
    counts = mol_to_counts({key: amounts[key] for key in spec.discrete_species})
    return amounts, counts


def _signature(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
               amounts: Mapping[str, float]) -> str:
    # Execution-local continuation binding, never a repository/release hash.
    return sha256(json.dumps([network.to_dict(), spec.to_dict(), amounts],
                            sort_keys=True, allow_nan=False).encode()).hexdigest()


def initial_hybrid_continuation(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                                amounts_mol: Mapping[str, float], *, seed: int,
                                time_s: float = 0.) -> dict[str, Any]:
    if getattr(spec, "algorithm", HYBRID_ALGORITHM) != HYBRID_ALGORITHM:
        from chemworld.physchem.stochastic_cross_partition import initial_cross_continuation

        return initial_cross_continuation(network, spec, amounts_mol, seed=seed, time_s=time_s)
    compile_hybrid(network, spec)
    amounts, _ = _amounts(network, spec, amounts_mol)
    if type(seed) is not int or seed < 0 or time_s != 0:
        raise ValueError("hybrid initialization requires a nonnegative seed and a fresh zero clock")
    return {"algorithm": HYBRID_ALGORITHM, "discrete_species": list(spec.discrete_species),
            "time_s": float(time_s), "rng_state": deepcopy(np.random.PCG64(seed).state),
            "pending": None, "signature": _signature(network, spec, amounts),
            "events_committed": 0}


def _continuation(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                  amounts: Mapping[str, float], continuation: Mapping[str, Any]
                  ) -> tuple[dict[str, Any], np.random.Generator]:
    state = deepcopy(dict(continuation))
    if (set(state) != {"algorithm", "discrete_species", "time_s", "rng_state", "pending",
                       "signature", "events_committed"}
            or state["algorithm"] != HYBRID_ALGORITHM
            or state["discrete_species"] != list(spec.discrete_species)
            or isinstance(state["time_s"], bool) or not isfinite(state["time_s"])
            or state["time_s"] < 0 or type(state["events_committed"]) is not int
            or state["events_committed"] < 0
            or state["signature"] != _signature(network, spec, amounts)):
        raise ValueError("hybrid continuation does not match its declared inventory and partition")
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state["rng_state"]
    pending = state["pending"]
    if pending is None:
        if state["events_committed"] != 0 or state["time_s"] != 0:
            raise ValueError("hybrid pending hazards cannot be erased during continuation")
    elif (not isinstance(pending, dict)
          or set(pending) != {"schema", "channel_ids", "hazards", "targets"}
          or pending["schema"] != HAZARD_SCHEMA
          or pending["channel_ids"] != list(spec.stochastic_reactions)
          or len(pending["hazards"]) != len(spec.stochastic_reactions)
          or len(pending["targets"]) != len(spec.stochastic_reactions)
          or any(isinstance(h, bool) or isinstance(e, bool) or not isfinite(h)
                 or not isfinite(e) or h < 0 or e <= h
                 for h, e in zip(pending["hazards"], pending["targets"], strict=True))):
        raise ValueError("invalid hybrid pending integrated hazards")
    return state, rng


def hybrid_element_totals(network: ReactionNetworkSpec, amounts: Mapping[str, float],
                          species: Sequence[str] | set[str]) -> dict[str, float]:
    keys = set(species)
    atoms = sorted({a for s in network.species if s.species_id in keys for a in s.composition})
    return {a: fsum(amounts[s.species_id] * s.composition.get(a, 0)
                    for s in network.species if s.species_id in keys) for a in atoms}


def validate_hybrid_conservation(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                                 before: Mapping[str, float], after: Mapping[str, float]) -> None:
    """Both partitions close separately; a large continuous stock cannot hide one atom."""
    _, old_counts = _amounts(network, spec, before)
    _, new_counts = _amounts(network, spec, after)
    atoms = {a for s in network.species if s.species_id in spec.discrete_species
             for a in s.composition}
    if any(sum((new_counts[s.species_id] - old_counts[s.species_id])
               * int(s.composition.get(a, 0)) for s in network.species
               if s.species_id in spec.discrete_species) != 0 for a in atoms):
        raise ValueError("hybrid integer atom inventory is not conserved")
    continuous = set(network.species_ids) - set(spec.discrete_species)
    old = hybrid_element_totals(network, before, continuous)
    new = hybrid_element_totals(network, after, continuous)
    if any(abs(old[a] - new[a]) > max(1e-30, 1e-9 * max(abs(old[a]), abs(new[a]))) for a in old):
        raise ValueError("hybrid continuous atom inventory is not conserved")


@dataclass(frozen=True)
class HybridAdvance:
    amounts_mol: dict[str, float]
    continuation: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    actual_duration_s: float
    work: dict[str, int]


@dataclass(frozen=True)
class _HazardRoot:
    index: int
    offset: int
    targets: np.ndarray
    terminal: bool = True
    direction: int = -1

    def __call__(self, _time: float, vector: np.ndarray) -> float:
        return float(self.targets[self.index] - vector[self.offset + self.index])


class HybridEventBudgetError(EventBudgetError):
    def __init__(self, events: list[dict[str, Any]], work: dict[str, int]) -> None:
        super().__init__("hybrid event budget exhausted; entire candidate rolls back")
        self.events, self.work = tuple(events), dict(work)


def advance_stochastic_hybrid(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                              amounts_mol: Mapping[str, float], continuation: Mapping[str, Any],
                              duration_s: float) -> HybridAdvance:
    """Pure candidate: all inventory, RNG and hazard mutations are local until accepted."""
    if getattr(spec, "algorithm", HYBRID_ALGORITHM) != HYBRID_ALGORITHM:
        from chemworld.physchem.stochastic_cross_partition import advance_cross_partition

        return advance_cross_partition(network, spec, amounts_mol, continuation, duration_s)
    continuous_channels, jump_channels = compile_hybrid(network, spec)
    amounts, counts = _amounts(network, spec, amounts_mol)
    memory, rng = _continuation(network, spec, amounts, continuation)
    start = time = memory["time_s"]
    if isinstance(duration_s, bool) or not isfinite(duration_s) or duration_s < 0:
        raise ValueError("hybrid duration must be finite and nonnegative")
    end = start + duration_s
    if not isfinite(end) or (duration_s > 0 and end <= start):
        raise ValueError("hybrid endpoint cannot be represented")
    work = {"random_draws": 0, "rhs_evaluations": 0, "propensity_evaluations": 0,
            "segments": 0, "events_committed": 0}
    if duration_s == 0:
        return HybridAdvance(amounts, memory, (), 0., work)

    def draw() -> float:
        threshold = -log1p(-float(rng.random()))
        work["random_draws"] += 1
        if threshold <= 0 or not isfinite(threshold):
            raise ArithmeticError("hybrid exponential threshold cannot be represented")
        return threshold

    if memory["pending"] is None:
        # One independent Poisson stream per channel, including currently dormant
        # channels: continuous production can activate one inside an ODE segment.
        memory["pending"] = {"schema": HAZARD_SCHEMA,
            "channel_ids": list(spec.stochastic_reactions),
            "hazards": [0.] * len(jump_channels), "targets": [draw() for _ in jump_channels]}
    pending = memory["pending"]
    hazards, targets = np.asarray(pending["hazards"]), np.asarray(pending["targets"])
    keys = tuple(k for k in network.species_ids if k not in spec.discrete_species)
    scale = max((amounts[k] for k in keys), default=0.) or 1 / AVOGADRO
    discrete = set(spec.discrete_species)
    concentration_scale = AVOGADRO * spec.volume_L
    events: list[dict[str, Any]] = []

    def rate(channel: HybridChannel, values: Mapping[str, float]) -> float:
        value = channel.k * (concentration_scale if channel.stochastic else spec.volume_L)
        for key, order in channel.orders:
            if channel.stochastic and key in discrete:
                n = counts[key]
                if n < order:
                    return 0.
                # Falling factorial, not a macroscopic N**order approximation.
                factor = prod(range(n - order + 1, n + 1))
                value *= factor / concentration_scale**order
            else:
                # Rate-domain extension for signed internal ODE trials only.
                # Every accepted solver node and final inventory is checked below;
                # negative accepted material is rejected, never clipped.
                value *= (max(0., values[key]) / spec.volume_L)**order
        if not isfinite(value) or value < 0:
            raise ArithmeticError("hybrid mass-action rate is nonfinite")
        return value

    def rhs(_time: float, vector: np.ndarray) -> np.ndarray:
        work["rhs_evaluations"] += 1
        values = {**counts_to_mol(counts), **dict(zip(keys, vector[:len(keys)] * scale,
                                                   strict=True))}
        derivatives = dict.fromkeys(keys, 0.)
        for channel in continuous_channels:
            extent = rate(channel, values)
            for key, coefficient in channel.changes:
                derivatives[key] += coefficient * extent
        propensities = [rate(c, values) for c in jump_channels]
        work["propensity_evaluations"] += len(jump_channels)
        return np.asarray([*(derivatives[k] / scale for k in keys), *propensities])

    while time < end:
        roots = tuple(_HazardRoot(index, len(keys), targets)
                      for index in range(len(jump_channels)))
        policy = ODESolverPolicy(HYBRID_ALGORITHM, method=spec.ode_method, rtol=spec.rtol,
            atol=spec.atol, max_step_s=spec.max_step_s,
            event_names=tuple(c.reaction_id for c in jump_channels))
        vector = np.asarray([*(amounts[k] / scale for k in keys), *hazards])
        report = solve_ode(rhs, vector, time_span_s=(time, end), policy=policy, events=roots)
        work["segments"] += 1
        report.raise_for_failure("hybrid continuous segment")
        if np.any(report.y[:len(keys)] < 0):
            raise ArithmeticError("hybrid accepted segment has negative continuous inventory")
        terminal = report.terminal_point
        new_time, vector = ((float(report.t[-1]), report.y[:, -1]) if terminal is None
                            else terminal[:2])
        if new_time <= time:
            raise ArithmeticError("hybrid jump time cannot advance the clock")
        time = new_time
        amounts.update({k: float(v) * scale for k, v in zip(keys, vector[:len(keys)], strict=True)})
        hazards = vector[len(keys):].copy()
        if terminal is None:
            break
        if len(events) >= spec.max_events_per_advance:
            raise HybridEventBudgetError(events, work)
        index = terminal[2]
        channel = jump_channels[index]
        for key, coefficient in channel.changes:
            counts[key] += coefficient
        amounts.update(counts_to_mol(counts))  # Validates strict integer availability and bound.
        hazards[index] = targets[index]
        new_target = float(targets[index]) + draw()
        if new_target <= targets[index]:
            raise ArithmeticError("hybrid cumulative threshold cannot be represented")
        targets[index] = new_target
        events.append({"time_s": time, "reaction_id": channel.reaction_id})
    validate_hybrid_conservation(network, spec, amounts_mol, amounts)
    memory.update(time_s=end, rng_state=deepcopy(rng.bit_generator.state),
                  signature=_signature(network, spec, amounts),
                  events_committed=memory["events_committed"] + len(events))
    pending.update(hazards=hazards.tolist(), targets=targets.tolist())
    # Recheck stored roots strictly so corrupt/overshot risk cannot enter a snapshot.
    _continuation(network, spec, amounts, memory)
    work["events_committed"] = len(events)
    return HybridAdvance(amounts, memory, tuple(events), duration_s, work)
