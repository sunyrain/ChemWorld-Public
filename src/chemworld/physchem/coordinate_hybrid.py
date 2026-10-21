"""Restart common coupled segments at coordinate roots and queue deadlines.

This is a pure candidate operation. The existing runtime owns the state,
single clock, resource settlement, transaction, and continuation persistence.
No event is located by advancing an auxiliary simulator or nudging time.
"""

from dataclasses import dataclass, field, replace
from math import copysign, isfinite, nextafter
from types import MappingProxyType

from chemworld.physchem.coordinate_events import (
    CoordinateEventError,
    CoordinateEventMemory,
    _finite_real,
    _integer,
    settle_coordinate_events,
)
from chemworld.physchem.coupled_system import CoupledEvent


@dataclass(frozen=True)
class CoordinateRoot:
    root_id: str
    value: object
    derivative: object

    def __post_init__(self):
        if (not isinstance(self.root_id, str) or not self.root_id
                or not callable(self.value) or not callable(self.derivative)):
            raise ValueError("coordinate roots need identity and pure value/derivative callbacks")


@dataclass(frozen=True)
class CoordinateHybridMemory:
    events: CoordinateEventMemory
    root_values: object = field(default_factory=dict)

    def __post_init__(self):
        from collections.abc import Mapping

        if (not isinstance(self.events, CoordinateEventMemory)
                or not isinstance(self.root_values, Mapping)
                or any(not isinstance(k, str) or not k or not _finite_real(v)
                       for k, v in self.root_values.items())):
            raise ValueError("invalid coordinate root continuation")
        object.__setattr__(self, "root_values", MappingProxyType(dict(self.root_values)))

    @classmethod
    def initialize(cls, events):
        return cls(CoordinateEventMemory.initialize(events))

    def to_dict(self):
        return {"format": "chemworld_coordinate_hybrid_v1", "events": self.events.to_dict(),
                "root_values": dict(self.root_values)}

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {"format", "events", "root_values"}
                or payload["format"] != "chemworld_coordinate_hybrid_v1"):
            raise ValueError("unknown coordinate hybrid continuation fields/format")
        return cls(CoordinateEventMemory.from_dict(payload["events"]), payload["root_values"])


@dataclass(frozen=True)
class CoordinateHybridResult:
    initial_state: object
    final_state: object
    memory: CoordinateHybridMemory
    trajectory: tuple
    events: tuple
    segment_diagnostics: tuple
    stopping_event: str | None = None

    @property
    def actual_duration_s(self):
        return self.final_state.time_s-self.initial_state.time_s


def coordinate_root_sides(state, roots, root_values):
    """Right-side signs only at an exact or recorded numerical root candidate.

    A nonzero derivative is required there; tangencies and static equalities
    are outside this transverse-root profile rather than assigned a fake sign.
    """
    signs = {}
    for root in roots:
        value = root.value(state)
        if not _finite_real(value):
            raise ValueError("coordinate root must be finite")
        if value == 0 or (root.root_id in root_values and value == root_values[root.root_id]):
            slope = root.derivative(state)
            if not _finite_real(slope) or slope == 0:
                raise ValueError("coordinate guard is not a supported transverse root")
            signs[root.root_id] = 1 if slope > 0 else -1
        else:
            signs[root.root_id] = 1 if value > 0 else -1
    return signs


def advance_coordinate_events(system, state, duration_s, memory, events, *, roots=(),
                              trigger_values=None, max_segments=10000, max_events=1000,
                              max_pending=1024):
    """Propose shared continuous segments and atomic same-time coordinate jumps.

    ``trigger_values(state, signs)`` evaluates guards using supplied outgoing
    root signs. Root residuals are numerical continuation, never a second live
    amount or clock. Unresolvable zero-time restarts fail with their prefix.
    """
    if not _finite_real(duration_s) or duration_s < 0:
        raise ValueError("hybrid duration must be finite nonnegative seconds")
    if any(not _integer(v, 1) for v in (max_segments, max_events, max_pending)):
        raise ValueError("hybrid segment/event/queue budgets must be positive integers")
    if not isinstance(memory, CoordinateHybridMemory):
        raise ValueError("hybrid advance requires its known continuation")
    events, roots = tuple(events), tuple(roots)
    if any(not isinstance(r, CoordinateRoot) for r in roots):
        raise ValueError("hybrid roots require known declarations")
    identities = {r.root_id for r in roots}
    if (len(identities) != len(roots)
            or identities & {e.event_id for e in system.all_events}
            or set(memory.root_values)-identities):
        raise ValueError("hybrid roots and continuation must match unique declared identities")
    if roots and not callable(trigger_values):
        raise ValueError("hybrid root guards require their installed comparison evaluator")
    end_time = state.time_s+float(duration_s)
    if not isfinite(end_time) or (duration_s > 0 and end_time <= state.time_s):
        raise ValueError("hybrid duration exceeds shared clock range/resolution")
    candidate, continuation = state, memory
    trajectory, executed, diagnostics = [state], [], []
    stopping_event = None
    try:
        while True:
            boundary_values = dict(continuation.root_values)

            def observe(current, boundaries=boundary_values):
                return trigger_values(current, coordinate_root_sides(
                    current, roots, boundaries))

            settled = settle_coordinate_events(system, candidate, continuation.events, events,
                max_events=max_events-len(executed),
                trigger_values=observe if trigger_values is not None else None,
                max_pending=max_pending)
            executed.extend(settled.events)
            if settled.state != candidate:
                trajectory.append(settled.state)
            candidate = settled.state
            active = {r.root_id: r.value(candidate) for r in roots
                      if r.root_id in boundary_values
                      and r.value(candidate) == boundary_values[r.root_id]}
            continuation = CoordinateHybridMemory(settled.memory, active)
            if candidate.time_s >= end_time or stopping_event is not None:
                break
            if len(diagnostics) >= max_segments:
                raise ValueError("coordinate hybrid segment budget exceeded")
            next_time = min([end_time, *(p.due_time_s for p in continuation.events.pending)])
            if next_time <= candidate.time_s:
                raise ValueError("coordinate hybrid queue made no clock progress")
            signs = coordinate_root_sides(candidate, roots, active)
            start_time = candidate.time_s
            start_values = {r.root_id: r.value(candidate) for r in roots}
            root_events = []
            for root in roots:
                sign = signs[root.root_id]

                def value(current, spec=root, side=sign, start=start_time, values=start_values):
                    raw = spec.value(current)
                    # Root initialization only: avoid immediately locating the
                    # root just consumed. No physical time/state is perturbed.
                    if (current.time_s == start and raw == values[spec.root_id]):
                        return copysign(nextafter(0., 1.), side)
                    return raw

                root_events.append(CoupledEvent(root.root_id, value, direction=-sign))
            segment = replace(system, events=(*system.events, *root_events)).advance(
                candidate, next_time-candidate.time_s)
            diagnostics.append(dict(segment.solver_diagnostic))
            if segment.actual_duration_s <= 0:
                raise ValueError("coordinate root restart made no resolvable clock progress")
            candidate = segment.final_state
            trajectory.extend(segment.trajectory[1:])
            active = {}
            if segment.stopping_event in identities:
                active[segment.stopping_event] = next(
                    r.value(candidate) for r in roots if r.root_id == segment.stopping_event)
            else:
                stopping_event = segment.stopping_event
            continuation = CoordinateHybridMemory(continuation.events, active)
        return CoordinateHybridResult(state, candidate, continuation, tuple(trajectory),
            tuple(executed), tuple(diagnostics), stopping_event)
    except CoordinateEventError as exc:
        raise CoordinateEventError(str(exc), (*executed, *exc.executed_prefix)) from exc
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        raise CoordinateEventError(str(exc), executed) from exc
