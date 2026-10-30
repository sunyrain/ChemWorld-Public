"""Pure source-event orchestration on the original integer candidate and queue."""

from dataclasses import dataclass, field, replace
from math import isfinite

from chemworld.physchem.coordinate_events import (
    CoordinateEventError,
    CoordinateEventMemory,
    _check_memory,
    settle_coordinate_events,
)
from chemworld.physchem.coupled_system import KineticOnlyThermodynamics
from chemworld.physchem.sbml_stochastic import MEMORY_FORMAT, PROCESS_FIELDS, initial_memory

BOUNDARY_MEMORY_FORMAT = "chemworld_sbml_integer_events_v1"


@dataclass(frozen=True)
class IntegerEventValidator:
    """Queue validation only; no integrator, inventory, clock or evolving state."""

    integer: object
    thermodynamics: KineticOnlyThermodynamics = field(default_factory=KineticOnlyThermodynamics)

    def validate_state(self, state):
        self.integer.compile(state.coordinates, state.time_s)


@dataclass(frozen=True)
class IntegerEventAdvance:
    coordinates: dict
    memory: dict
    actual_duration_s: float
    events: tuple
    batches: tuple
    source_events: tuple
    work: dict


def initial_boundary_memory(model, seed):
    return {
        "format": BOUNDARY_MEMORY_FORMAT,
        "process": initial_memory(seed, model.solver_method)["process"],
        "event_queue": CoordinateEventMemory.initialize(model.events).to_dict(),
    }


def advance_integer_events(model, state, memory, duration_s):
    """Settle original queue boundaries between actual kernel reaction/batch steps.

    Numerical reaction/batch endpoints at a source deadline precede source-event
    settlement at that same time. This is the declared right-continuous numerical
    path for tau, not an assertion of exact within-leap source event locations.
    """
    if not model.integer_kinetics.requires_boundaries:
        raise ValueError("source boundary coordination requires an explicit source profile")
    if type(duration_s) not in {int, float} or not isfinite(duration_s) or duration_s < 0:
        raise ValueError("integer boundary duration must be finite nonnegative seconds")
    if (
        not isinstance(memory, dict)
        or set(memory) != {"format", "process", "event_queue"}
        or memory["format"] != BOUNDARY_MEMORY_FORMAT
    ):
        raise ValueError("unknown integer source-event continuation fields or format")
    queue = CoordinateEventMemory.from_dict(memory["event_queue"])
    _check_memory(queue, {event.event_id: event for event in model.events})
    if len(queue.pending) > model.maximum_pending_events:
        raise ValueError("integer source-event queue exceeds declared pending budget")
    start, target = state.time_s, state.time_s + duration_s
    if not isfinite(target) or (duration_s > 0 and target <= start):
        raise ValueError("integer source duration exceeds native clock resolution")
    integer, candidate = model.integer_kinetics, state
    validator = IntegerEventValidator(integer)
    process = memory["process"]
    if (
        not isinstance(process, dict)
        or set(process) != PROCESS_FIELDS
        or type(process["events_committed"]) is not int
        or process["events_committed"] < 0
    ):
        raise ValueError("integer source process fields and firing counter must be valid")
    events, batches, source_events, work = [], [], [], {}
    segments = 0
    initial_firings = process["events_committed"]

    def step(duration, *, boundary, calendar_deadline_s=None):
        nonlocal candidate, process
        values, updated, result = integer.advance(
            candidate.coordinates,
            {"format": MEMORY_FORMAT, "process": process},
            candidate.time_s,
            duration,
            model.solver_method,
            model.maximum_events,
            model.tau_policy,
            stop_after_boundary=boundary,
            critical_reaction_ids=model.guard_critical_reactions,
            calendar_deadline_s=calendar_deadline_s,
        )
        process = updated["process"]
        candidate = replace(candidate, coordinates=values, time_s=process["time_s"])
        for key, value in result.work.items():
            if type(value) is int and key != "pending_firings":
                work[key] = work.get(key, 0) + value
        return result

    try:
        while True:
            settled = settle_coordinate_events(
                validator,
                candidate,
                queue,
                model.events,
                max_events=model.maximum_events - len(source_events),
                trigger_values=model.roots.trigger_values,
                max_pending=model.maximum_pending_events,
            )
            source_events.extend(settled.events)
            candidate, queue = settled.state, settled.memory
            guard = model.roots.next_boundary(candidate)
            calendar = [pending.due_time_s for pending in queue.pending]
            if guard is not None:
                calendar.append(guard)
            deadline = min(calendar) if calendar else None
            # Real source assignments/controls invalidate incompatible plans via
            # the existing signature; a zero-duration rebind never draws anew.
            step(0.0, boundary=False, calendar_deadline_s=deadline)
            if candidate.time_s >= target:
                break
            if segments >= model.maximum_segments:
                raise ValueError("integer source-event segment budget exceeded")
            endpoint = min(target, target if deadline is None else deadline)
            if endpoint <= candidate.time_s:
                raise ValueError("integer source boundary has no resolvable clock progress")
            result = step(endpoint - candidate.time_s, boundary=True, calendar_deadline_s=deadline)
            segments += 1
            if result.actual_duration_s <= 0:
                raise ValueError("integer kernel returned no positive source-boundary progress")
            events.extend(result.events)
            batches.extend(result.batches)
            if process["events_committed"] - initial_firings > model.maximum_events:
                raise ValueError("integer source reaction-firing budget exceeded")
            if (
                model.tau_policy is not None
                and work.get("steps_attempted", 0) > model.tau_policy.max_steps_per_advance
            ):
                raise ValueError("integer source tau internal-step budget exceeded")
        work.update(boundary_segments=segments, source_events_executed=len(source_events))
        return IntegerEventAdvance(
            dict(candidate.coordinates),
            {"format": BOUNDARY_MEMORY_FORMAT, "process": process, "event_queue": queue.to_dict()},
            candidate.time_s - start,
            tuple(events),
            tuple(batches),
            tuple(source_events),
            work,
        )
    except CoordinateEventError as exc:
        raise CoordinateEventError(str(exc), (*source_events, *exc.executed_prefix)) from exc
    except (ArithmeticError, KeyError, ValueError, RuntimeError) as exc:
        raise CoordinateEventError(str(exc), source_events) from exc
