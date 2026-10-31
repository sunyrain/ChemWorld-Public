"""Transactional RI5 advancement over a caller-owned mol inventory.

Fixed mesh endpoints use RI5 with linear or fractional weak output. An explicit
truncated policy ends numerical steps at the requested time and checks that
endpoint physically. None is a strong Brownian bridge. Continuation holds solver
memory only. No stock, global RNG or independently advancing clock lives here.
"""

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import cast

import numpy as np

from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.stochastic_kinetics import EventBudgetError
from chemworld.physchem.stochastic_ri5 import RI5Draw, _finite_array, draw_ri5, ri5_step
from chemworld.physchem.stochastic_sde_model import ReactionSDE
from chemworld.physchem.stochastic_sde_specs import SDE_ALGORITHM, finite_real

SDE_PACKET_SCHEMA = "ri5-linear-dense-packet-v1"
SDE_FRACTIONAL_PACKET_SCHEMA = "ri5-fractional-dense-packet-v1"
SDE_TRUNCATED_PACKET_SCHEMA = "ri5-truncated-endpoint-packet-v1"
_COUNTERS = ("completed_steps", "proposals", "rejections", "backtracks", "invalidations")
_FIELDS = {
    "algorithm",
    "model_signature",
    "inventory_signature",
    "time_s",
    "mesh_time_s",
    "control_values",
    "volume_L",
    "rng_state",
    "pending",
    *_COUNTERS,
}


def _amounts(model, values):
    if not isinstance(values, Mapping) or set(values) != set(model.network.species_ids):
        raise ValueError("SDE inventory must cover exactly the declared species")
    result = {
        key: finite_real(values[key], "SDE mol inventory") for key in model.network.species_ids
    }
    if any(v < 0 or not isfinite(v * AVOGADRO) for v in result.values()):
        raise ValueError("SDE inventory must be nonnegative and representable in particle units")
    return result


def _signature(model, amounts, time_s):
    return sha256(
        json.dumps([model.signature, amounts, time_s], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _rng(value):
    if (
        not isinstance(value, Mapping)
        or set(value) != {"bit_generator", "state", "has_uint32", "uinteger"}
        or value["bit_generator"] != "PCG64"
        or not isinstance(value["state"], Mapping)
        or set(value["state"]) != {"state", "inc"}
        or any(
            type(value["state"][k]) is not int or not 0 <= value["state"][k] < 2**128
            for k in ("state", "inc")
        )
        or value["state"]["inc"] % 2 != 1
        or type(value["has_uint32"]) is not int
        or value["has_uint32"] not in {0, 1}
        or type(value["uinteger"]) is not int
        or not 0 <= value["uinteger"] < 2**32
    ):
        raise ValueError("SDE requires the exact portable PCG64 process random state")
    result = np.random.Generator(np.random.PCG64(0))
    result.bit_generator.state = deepcopy(dict(value))
    return result


def sde_model(network, spec, continuation):
    """Rebuild callbacks from the actual declaration and its sole live inputs."""
    if not isinstance(continuation, Mapping):
        raise ValueError("SDE requires declared continuation data")
    return ReactionSDE(
        network,
        spec,
        control_values=continuation.get("control_values"),
        volume_L=continuation.get("volume_L"),
    )


def _validate_current_callbacks(model, amounts, time_s):
    """A bounded current-domain check for initialization/material/control changes."""
    values = np.array([amounts[key] * AVOGADRO for key in model.network.species_ids])
    try:
        _finite_array(model.drift(time_s, values), (len(values),), "current drift")
        if model.noise_dimension:
            _finite_array(
                model.diffusion(time_s, values),
                (len(values), model.noise_dimension),
                "current diffusion",
            )
    except (ValueError, ArithmeticError) as exc:
        raise ValueError(
            "SDE current drift or diffusion is undefined for the declared state"
        ) from exc


def initial_sde_continuation(network, spec, amounts_mol, *, seed, time_s=0.0):
    model = ReactionSDE(network, spec)
    amounts = _amounts(model, amounts_mol)
    time_s = finite_real(time_s, "SDE initial clock")
    if type(seed) is not int or seed < 0 or time_s != 0:
        raise ValueError("SDE initialization requires a nonnegative seed and a fresh zero clock")
    _validate_current_callbacks(model, amounts, time_s)
    return {
        "algorithm": SDE_ALGORITHM,
        "model_signature": model.signature,
        "inventory_signature": _signature(model, amounts, time_s),
        "time_s": time_s,
        "mesh_time_s": time_s,
        "control_values": dict(cast(Mapping[str, float], model.control_values)),
        "volume_L": model.volume_L,
        "rng_state": deepcopy(np.random.PCG64(seed).state),
        "pending": None,
        **dict.fromkeys(_COUNTERS, 0),
    }


class _Work:
    def __init__(self, model):
        self.model = model
        self.evaluations = 0
        self.packets_created = 0

    def step(self, start, values, end, draw):
        m = self.model.noise_dimension
        required = 3 + (1 + 4 * m if m else 0)
        if self.evaluations + required > self.model.spec.max_evaluations_per_advance:
            raise EventBudgetError("SDE callback evaluation budget exhausted")
        self.evaluations += required
        result = ri5_step(self.model.drift, self.model.diffusion, start, values, end - start, draw)
        return list(result.values)


def _populations(value, dimension):
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError("SDE numerical packet requires a declared population vector")
    result = [finite_real(v, "SDE packet population") for v in value]
    if any(v < 0 for v in result):
        raise ValueError("SDE numerical packet endpoints must be nonnegative")
    return result


def _packet_schema(model):
    return {
        "linear": SDE_PACKET_SCHEMA,
        "ri5_fractional": SDE_FRACTIONAL_PACKET_SCHEMA,
        "ri5_truncated": SDE_TRUNCATED_PACKET_SCHEMA,
    }[model.spec.output_policy]


def _dense_amounts(model, packet, time_s, work):
    if time_s == packet["end_time_s"]:
        values = packet["right_populations"]
    elif time_s == packet["left_time_s"]:
        values = packet["left_populations"]
    elif model.spec.output_policy == "ri5_fractional":
        values = work.step(
            packet["left_time_s"],
            packet["left_populations"],
            time_s,
            RI5Draw.from_dict(packet["draw"]),
        )
        if any(not isfinite(v) or v < 0 for v in values):
            raise ValueError("SDE fractional RI5 output is outside its nonnegative physical domain")
    elif model.spec.output_policy == "ri5_truncated":
        raise ValueError("SDE truncated output must be an accepted numerical endpoint")
    else:
        fraction = (time_s - packet["left_time_s"]) / (packet["end_time_s"] - packet["left_time_s"])
        # Convex, with no subtraction of nearly equal large endpoint populations.
        values = [
            (1 - fraction) * left + fraction * right
            for left, right in zip(
                packet["left_populations"], packet["right_populations"], strict=True
            )
        ]
    return {
        key: float(value) / AVOGADRO
        for key, value in zip(model.network.species_ids, values, strict=True)
    }


def _continuation(model, amounts, continuation, work, *, verify_endpoint=True):
    if not isinstance(continuation, Mapping) or set(continuation) != _FIELDS:
        raise ValueError("unknown SDE continuation fields")
    state = deepcopy(dict(continuation))
    time_s = finite_real(state["time_s"], "SDE continuation clock")
    mesh_time_s = finite_real(state["mesh_time_s"], "SDE continuation mesh anchor")
    if (
        state["algorithm"] != SDE_ALGORITHM
        or state["control_values"] != dict(cast(Mapping[str, float], model.control_values))
        or finite_real(state["volume_L"], "SDE continuation volume", positive=True)
        != model.volume_L
        or state["model_signature"] != model.signature
        or time_s < 0
        or not 0 <= mesh_time_s <= time_s
        or state["inventory_signature"] != _signature(model, amounts, time_s)
        or any(type(state[k]) is not int or state[k] < 0 for k in _COUNTERS)
        or state["completed_steps"] + state["rejections"] > state["proposals"]
        or state["backtracks"] > state["proposals"] * model.spec.max_root_iterations
    ):
        raise ValueError("SDE continuation does not match its model, inventory or numerical work")
    rng = _rng(state["rng_state"])
    packet = state["pending"]
    if packet is None and mesh_time_s != time_s:
        raise ValueError("SDE pending packet cannot be erased inside a numerical step")
    if packet is not None:
        if model.spec.output_policy == "ri5_truncated":
            raise ValueError("SDE truncated endpoint policy cannot carry pending dense output")
        if (
            not isinstance(packet, dict)
            or set(packet)
            != {
                "schema",
                "left_time_s",
                "end_time_s",
                "left_populations",
                "right_populations",
                "draw",
                "proposal_index",
            }
            or packet["schema"] != _packet_schema(model)
            or type(packet["proposal_index"]) is not int
            or packet["proposal_index"] != state["proposals"]
            or state["completed_steps"] + state["rejections"] >= state["proposals"]
        ):
            raise ValueError("unknown or inconsistent SDE pending numerical packet")
        left = finite_real(packet["left_time_s"], "SDE packet left clock")
        end = finite_real(packet["end_time_s"], "SDE packet right clock")
        if (
            not 0 <= left == mesh_time_s < time_s < end
            or end - left > model.spec.step_s
            or end - left < model.spec.min_step_s
        ):
            raise ValueError("SDE pending packet does not bracket the current clock")
        packet["left_populations"] = _populations(packet["left_populations"], len(amounts))
        packet["right_populations"] = _populations(packet["right_populations"], len(amounts))
        draw = RI5Draw.from_dict(packet["draw"])
        if draw.dimension != model.noise_dimension:
            raise ValueError("SDE pending random variables do not match the model channels")
        if (verify_endpoint or model.spec.output_policy == "linear") and (
            _dense_amounts(model, packet, time_s, work) != amounts
        ):
            raise ValueError("SDE current inventory differs from its pending dense output")
        if (
            verify_endpoint
            and work.step(left, packet["left_populations"], end, draw)
            != packet["right_populations"]
        ):
            raise ValueError("SDE pending endpoint does not match its saved RI5 random variables")
    return state, rng


def validate_sde_continuation(network, spec, amounts_mol, continuation, *, verify_endpoint=True):
    """Validate solver data; a trusted advance performs its own budgeted endpoint check.

    The original runtime uses metadata-only checks around that one advance.
    Snapshot restoration keeps complete numerical verification.
    """
    model = sde_model(network, spec, continuation)
    amounts = _amounts(model, amounts_mol)
    _continuation(model, amounts, continuation, _Work(model), verify_endpoint=verify_endpoint)


def _new_packet(model, amounts, state, rng, work, target):
    if work.packets_created >= model.spec.max_steps_per_advance:
        raise EventBudgetError("SDE numerical step budget exhausted")
    start = state["time_s"]
    end = start + model.spec.step_s
    truncated = model.spec.output_policy == "ri5_truncated"
    if truncated:
        end = min(end, target)
    if not isfinite(end) or end <= start:
        raise ValueError("SDE numerical step cannot advance its finite clock")
    # The representable interval is bounded by the declared maximum.
    if end - start > model.spec.step_s:
        end = float(np.nextafter(end, start))
    # A requested final interval may be shorter than the physical-search floor.
    # Its own length then becomes the floor: accept it as a whole or redraw,
    # without unbounded sub-minimum backtracking or clipping material.
    minimum = min(model.spec.min_step_s, end - start) if truncated else model.spec.min_step_s
    if end <= start or end - start < minimum:
        raise ValueError("SDE numerical mesh cannot represent its declared minimum step")
    values = [amounts[key] * AVOGADRO for key in model.network.species_ids]
    for _ in range(model.spec.max_rejections_per_step):
        draw = draw_ri5(rng, model.noise_dimension)
        state["proposals"] += 1
        right = work.step(start, values, end, draw)
        accepted_end = end
        if any(v < 0 for v in right):
            if model.spec.boundary_policy == "reject":
                raise ValueError("SDE negative candidate rejected without inventory clipping")
            lower, upper, accepted = start, end, None
            # A bounded feasible-side bracket, not a claim of a universal first root.
            for _root in range(model.spec.max_root_iterations):
                midpoint = lower + (upper - lower) / 2
                if midpoint <= lower or midpoint >= upper:
                    break
                if midpoint - start < minimum:
                    midpoint = start + minimum
                    if not lower < midpoint < upper:
                        break
                candidate = work.step(start, values, midpoint, draw)
                state["backtracks"] += 1
                if all(v >= 0 for v in candidate):
                    lower, accepted = midpoint, candidate
                else:
                    upper = midpoint
            if accepted is None:
                state["rejections"] += 1
                continue
            accepted_end, right = lower, accepted
        work.packets_created += 1
        return {
            "schema": _packet_schema(model),
            "left_time_s": start,
            "end_time_s": accepted_end,
            "left_populations": values,
            "right_populations": right,
            "draw": draw.to_dict(),
            "proposal_index": state["proposals"],
        }
    raise EventBudgetError("SDE physical-boundary redraw budget exhausted")


@dataclass(frozen=True)
class SDEAdvance:
    amounts_mol: dict
    continuation: dict
    actual_duration_s: float
    diagnostics: dict


def advance_sde(network, spec, amounts_mol, continuation, duration_s):
    """Build a complete candidate; caller commits through its original transaction."""
    model = sde_model(network, spec, continuation)
    amounts = _amounts(model, amounts_mol)
    duration = finite_real(duration_s, "SDE requested duration")
    if duration < 0:
        raise ValueError("SDE requested duration cannot be negative")
    work = _Work(model)
    state, rng = _continuation(model, amounts, continuation, work)
    target = state["time_s"] + duration
    if not isfinite(target) or (duration > 0 and target <= state["time_s"]):
        raise ValueError("SDE requested duration cannot advance its finite clock")
    old_counters = {k: state[k] for k in _COUNTERS}
    while state["time_s"] < target:
        packet = state["pending"]
        if packet is None:
            packet = _new_packet(model, amounts, state, rng, work, target)
        reached = min(target, packet["end_time_s"])
        amounts = _dense_amounts(model, packet, reached, work)
        state["time_s"] = reached
        if reached == packet["end_time_s"]:
            state["completed_steps"] += 1
            state["mesh_time_s"] = reached
            state["pending"] = None
        else:
            state["pending"] = packet
    state["rng_state"] = deepcopy(rng.bit_generator.state)
    state["inventory_signature"] = _signature(model, amounts, state["time_s"])
    return SDEAdvance(
        amounts,
        state,
        duration,
        {
            "packets_created": work.packets_created,
            "evaluations": work.evaluations,
            **{k: state[k] - old_counters[k] for k in _COUNTERS},
        },
    )


def rebind_sde_continuation(
    network,
    spec,
    old_amounts_mol,
    new_amounts_mol,
    continuation,
    *,
    control_values=None,
    volume_L=None,
    force_invalidate=False,
):
    """Invalidate solver lookahead after an actual caller-owned material change.

    The consumed process stream is retained. The material transaction, not this
    helper, owns the new stock and validates its external source/sink receipts.
    """
    model = sde_model(network, spec, continuation)
    old = _amounts(model, old_amounts_mol)
    new = _amounts(model, new_amounts_mol)
    state, _ = _continuation(model, old, continuation, _Work(model))
    if type(force_invalidate) is not bool:
        raise ValueError("SDE material invalidation policy must be explicitly Boolean")
    if control_values is not None and not isinstance(control_values, Mapping):
        raise ValueError("SDE control change must be a declared mapping")
    current = {**cast(Mapping[str, float], model.control_values), **(control_values or {})}
    changed = ReactionSDE(
        network,
        spec,
        control_values=current,
        volume_L=model.volume_L if volume_L is None else volume_L,
    )
    if force_invalidate or old != new or model.signature != changed.signature:
        _validate_current_callbacks(changed, new, state["time_s"])
        state["pending"] = None
        state["mesh_time_s"] = state["time_s"]
        state["invalidations"] += 1
        state["control_values"] = dict(cast(Mapping[str, float], changed.control_values))
        state["volume_L"] = changed.volume_L
        state["model_signature"] = changed.signature
        state["inventory_signature"] = _signature(changed, new, state["time_s"])
    return state
