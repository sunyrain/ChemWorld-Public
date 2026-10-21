"""Atomic formal-coordinate events on the existing shared coupled candidate.

This module advances no clock and owns no simulator state. The caller supplies
the current coupled candidate and persists the returned continuation with its
existing entity owner. Material/energy jumps require their physical contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from math import isfinite, ulp
from numbers import Real
from types import MappingProxyType

from chemworld.physchem.coupled_system import CoupledState, KineticOnlyThermodynamics

MEMORY_FORMAT = "chemworld_coordinate_event_memory_v1"


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite_real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _payload(payload, targets=None):
    if (not isinstance(payload, Mapping)
            or any(not isinstance(k, str) or not k or not _finite_real(v)
                   for k, v in payload.items())
            or (targets is not None and set(payload) != set(targets))):
        raise ValueError("coordinate event payload must cover finite declared targets exactly")
    return MappingProxyType({key: float(value) for key, value in payload.items()})


def coordinate_jump(state, payload):
    """Simultaneously replace supplied coordinates, preserving every other field."""
    if set(payload)-state.coordinates.keys():
        raise ValueError("coordinate event cannot add undeclared coordinates")
    return replace(state, coordinates={**state.coordinates, **payload})


@dataclass(frozen=True)
class CoordinateEventSpec:
    event_id: str
    targets: tuple[str, ...]
    trigger: object
    capture: object
    apply: object = coordinate_jump
    delay_s: object = lambda _state: 0.
    priority: object = None
    initial_value: bool = True
    persistent: bool = True
    use_trigger_values: bool = True

    def __post_init__(self):
        object.__setattr__(self, "targets", tuple(self.targets))
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("coordinate event requires a stable identity")
        if (len(self.targets) != len(set(self.targets))
                or any(not isinstance(t, str) or not t for t in self.targets)):
            raise ValueError("coordinate event requires unique target identities")
        if (any(not callable(f) for f in (self.trigger, self.capture, self.apply, self.delay_s))
                or (self.priority is not None and not callable(self.priority))):
            raise ValueError("coordinate event requires installed pure callbacks")
        if any(type(v) is not bool for v in (self.initial_value, self.persistent,
                                            self.use_trigger_values)):
            raise ValueError("coordinate event flags must be explicit Booleans")


@dataclass(frozen=True)
class PendingCoordinateEvent:
    event_id: str
    generation: int
    sequence: int
    due_time_s: float
    payload: object = None

    def __post_init__(self):
        if (not isinstance(self.event_id, str) or not self.event_id
                or not _integer(self.generation, 1) or not _integer(self.sequence)
                or not _finite_real(self.due_time_s) or self.due_time_s < 0):
            raise ValueError("invalid pending coordinate event identity or due time")
        if self.payload is not None:
            object.__setattr__(self, "payload", _payload(self.payload))

    def to_dict(self):
        return {"event_id": self.event_id, "generation": self.generation,
                "sequence": self.sequence, "due_time_s": self.due_time_s,
                "payload": None if self.payload is None else dict(self.payload)}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {
                "event_id", "generation", "sequence", "due_time_s", "payload"}:
            raise ValueError("unknown pending coordinate event fields")
        return cls(**payload)


@dataclass(frozen=True)
class CoordinateEventMemory:
    previous: object
    generations: object
    pending: tuple = ()
    next_sequence: int = 0

    def __post_init__(self):
        if (not isinstance(self.previous, Mapping) or not isinstance(self.generations, Mapping)
                or set(self.previous) != set(self.generations)
                or any(not isinstance(k, str) or not k or type(v) is not bool
                       for k, v in self.previous.items())
                or any(not _integer(v) for v in self.generations.values())
                or not _integer(self.next_sequence)
                or self.next_sequence != sum(self.generations.values())):
            raise ValueError("invalid coordinate event trigger/generation continuation")
        pending = tuple(self.pending)
        if any(not isinstance(p, PendingCoordinateEvent) for p in pending):
            raise ValueError("pending queue requires known coordinate event records")
        if (len({p.sequence for p in pending}) != len(pending)
                or len({(p.event_id, p.generation) for p in pending}) != len(pending)
                or any(p.event_id not in self.generations
                    or p.generation > self.generations[p.event_id]
                    or p.sequence >= self.next_sequence
                    for p in pending)):
            raise ValueError("pending queue contradicts event generation/sequence counters")
        object.__setattr__(self, "previous", MappingProxyType(dict(self.previous)))
        object.__setattr__(self, "generations", MappingProxyType(dict(self.generations)))
        object.__setattr__(self, "pending", pending)

    @classmethod
    def initialize(cls, events):
        events = _events(events)
        return cls({key: event.initial_value for key, event in events.items()},
                   dict.fromkeys(events, 0))

    def to_dict(self):
        return {"format": MEMORY_FORMAT, "previous": dict(self.previous),
                "generations": dict(self.generations), "next_sequence": self.next_sequence,
                "pending": [p.to_dict() for p in self.pending]}

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {
                "format", "previous", "generations", "next_sequence", "pending"}
                or payload["format"] != MEMORY_FORMAT or not isinstance(payload["pending"], list)):
            raise ValueError("unknown coordinate event continuation format/fields")
        return cls(payload["previous"], payload["generations"],
                   tuple(PendingCoordinateEvent.from_dict(p) for p in payload["pending"]),
                   payload["next_sequence"])


def _events(events):
    events = tuple(events)
    if any(not isinstance(event, CoordinateEventSpec) for event in events):
        raise ValueError("coordinate events require their registered declaration type")
    result = {event.event_id: event for event in events}
    if len(result) != len(events):
        raise ValueError("duplicate coordinate event identity")
    return result


def _check_memory(memory, events):
    if not isinstance(memory, CoordinateEventMemory) or set(memory.previous) != set(events):
        raise ValueError("coordinate event continuation does not match its declarations")
    for pending in memory.pending:
        event = events[pending.event_id]
        if event.use_trigger_values != (pending.payload is not None):
            raise ValueError("pending event capture convention differs from its declaration")
        if pending.payload is not None:
            _payload(pending.payload, event.targets)


def refresh_coordinate_triggers(state, memory, events, *, trigger_values=None, max_pending=1024):
    """Observe one shared candidate; schedule false-to-true edges and cancel lost guards.

    A root coordinator may supply explicit one-sided trigger values at a root;
    this function does not infer a second time or move the supplied candidate.
    """
    events = _events(events)
    _check_memory(memory, events)
    if not _integer(max_pending, 1) or len(memory.pending) > max_pending:
        raise ValueError("coordinate event pending queue exceeds its positive declared budget")
    overrides = {} if trigger_values is None else dict(trigger_values)
    if set(overrides)-events.keys() or any(type(v) is not bool for v in overrides.values()):
        raise ValueError("invalid one-sided coordinate trigger values")
    previous, generations = dict(memory.previous), dict(memory.generations)
    truths = {}
    for key, event in events.items():
        truth = overrides[key] if key in overrides else event.trigger(state)
        if type(truth) is not bool:
            raise ValueError("coordinate event trigger must return a Boolean")
        truths[key] = truth
    # All cancellations concern the same candidate and precede capacity checks.
    # Queue capacity must not depend on the declarations' XML order.
    pending = [p for p in memory.pending if events[p.event_id].persistent or truths[p.event_id]]
    sequence = memory.next_sequence
    for key, event in events.items():
        truth = truths[key]
        if truth and not previous[key]:
            if len(pending) >= max_pending:
                raise ValueError("coordinate event pending queue budget exceeded")
            delay = event.delay_s(state)
            if not _finite_real(delay) or delay < 0:
                raise ValueError("coordinate event delay must be finite nonnegative seconds")
            due = state.time_s+float(delay)
            if not isfinite(due) or (delay > 0 and due <= state.time_s):
                raise ValueError("coordinate event delay exceeds clock range/resolution")
            captured = (_payload(event.capture(state), event.targets)
                        if event.use_trigger_values else None)
            generations[key] += 1
            pending.append(PendingCoordinateEvent(key, generations[key], sequence, due, captured))
            sequence += 1
        previous[key] = truth
    return CoordinateEventMemory(previous, generations, tuple(pending), sequence)


class CoordinateEventError(RuntimeError):
    def __init__(self, message, executed_prefix=()):
        super().__init__(message)
        self.executed_prefix = tuple(executed_prefix)


@dataclass(frozen=True)
class CoordinateEventSettlement:
    state: CoupledState
    memory: CoordinateEventMemory
    events: tuple


def settle_coordinate_events(system, state, memory, events, *, max_events=1000,
                             trigger_values=None, max_pending=1024):
    """Return an atomic proposal at the same physical time; retain errors' prefixes.

    Priorities are reevaluated after each execution. Simultaneous unspecified
    or tied priorities are refused instead of treating XML order as a standard.
    """
    if not isinstance(system.thermodynamics, KineticOnlyThermodynamics):
        raise ValueError("coordinate events require explicit formal-kinetic energy scope")
    if not _integer(max_events):
        raise ValueError("coordinate event execution budget must be a nonnegative integer")
    events = _events(events)
    system.validate_state(state)
    executed = []
    candidate, continuation = state, memory
    try:
        def observe(current):
            return trigger_values(current) if callable(trigger_values) else trigger_values

        continuation = refresh_coordinate_triggers(candidate, continuation, events.values(),
            trigger_values=observe(candidate), max_pending=max_pending)
        while True:
            eligible = []
            for pending in continuation.pending:
                tolerance = 8*max(ulp(candidate.time_s), ulp(pending.due_time_s))
                if pending.due_time_s < candidate.time_s-tolerance:
                    raise ValueError("coordinate event deadline was passed without execution")
                if abs(pending.due_time_s-candidate.time_s) <= tolerance:
                    event = events[pending.event_id]
                    priority = None if event.priority is None else event.priority(candidate)
                    if event.priority is not None and not _finite_real(priority):
                        raise ValueError("coordinate event priority must be finite")
                    eligible.append((priority, pending))
            if not eligible:
                break
            if len(executed) >= max_events:
                raise ValueError("coordinate event cascade exceeded its execution budget")
            if len(eligible) > 1 and any(priority is None for priority, _ in eligible):
                raise ValueError("simultaneous coordinate events have unspecified priorities")
            if len(eligible) == 1:
                priority, pending = eligible[0]
            else:
                priority = max(p for p, _ in eligible)
                winners = [item for p, item in eligible if p == priority]
                if len(winners) != 1:
                    raise ValueError("simultaneous coordinate events have tied priorities")
                pending = winners[0]
            event = events[pending.event_id]
            payload = (pending.payload if pending.payload is not None else
                       _payload(event.capture(candidate), event.targets))
            proposed = event.apply(candidate, payload)
            if not isinstance(proposed, CoupledState) or any(
                    getattr(proposed, item.name) != getattr(candidate, item.name)
                    for item in fields(CoupledState) if item.name != "coordinates"):
                raise ValueError("coordinate event cannot change clock or physical material/energy")
            system.validate_state(proposed)
            continuation = replace(continuation, pending=tuple(
                p for p in continuation.pending if p.sequence != pending.sequence))
            executed.append({"event_id": pending.event_id, "generation": pending.generation,
                "sequence": pending.sequence, "time_s": candidate.time_s,
                "due_time_s": pending.due_time_s, "priority": priority, "payload": dict(payload)})
            candidate = proposed
            continuation = refresh_coordinate_triggers(candidate, continuation, events.values(),
                trigger_values=observe(candidate) if callable(trigger_values) else None,
                max_pending=max_pending)
        return CoordinateEventSettlement(candidate, continuation, tuple(executed))
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        raise CoordinateEventError(str(exc), executed) from exc
