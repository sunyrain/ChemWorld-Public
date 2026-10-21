"""Same-clock atomic queue proposals, replay, priority and failure semantics."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.coordinate_events import (
    CoordinateEventError,
    CoordinateEventMemory,
    CoordinateEventSpec,
    PendingCoordinateEvent,
    refresh_coordinate_triggers,
    settle_coordinate_events,
)
from chemworld.physchem.coupled_system import (
    CoupledCoordinate,
    CoupledState,
    CoupledSystem,
    KineticOnlyThermodynamics,
)


def setup(values=None):
    values = {"x": 1., "y": 2., "z": 0.} if values is None else values
    state = CoupledState({}, 298.15, 1., coordinates=values)
    system = CoupledSystem(KineticOnlyThermodynamics(), isothermal=True,
                           coordinates=tuple(CoupledCoordinate(k, "1", 1., 0.) for k in values))
    return system, state


def event(name="e", **kwargs):
    defaults = {"targets": ("x",), "trigger": lambda _: True,
                "capture": lambda _: {"x": 4.}, "initial_value": False}
    return CoordinateEventSpec(name, **{**defaults, **kwargs})


def settle(system, state, events, memory=None, **kwargs):
    memory = CoordinateEventMemory.initialize(events) if memory is None else memory
    return settle_coordinate_events(system, state, memory, events, **kwargs)


def test_initial_flag_and_atomic_swap_preserve_shared_clock_and_original_candidate():
    system, state = setup()
    swap = event(targets=("x", "y"), capture=lambda s: {
        "x": s.coordinates["y"], "y": s.coordinates["x"]})
    result = settle(system, state, (swap,))
    assert dict(result.state.coordinates) == {"x": 2., "y": 1., "z": 0.}
    assert state.coordinates["x"] == 1 and result.state.time_s == state.time_s == 0
    assert len(result.events) == 1 and result.memory.pending == ()
    assert settle(system, state, (replace(swap, initial_value=True),)).events == ()


@pytest.mark.parametrize("capture_early,expected", [(True, 1.), (False, 7.)])
def test_delayed_capture_convention_and_json_queue_replay(capture_early, expected):
    system, state = setup()
    delayed = event(delay_s=lambda _: 2., use_trigger_values=capture_early,
                    capture=lambda s: {"x": s.coordinates["y"]}, targets=("x",))
    state = replace(state, coordinates={**state.coordinates, "y": 1.})
    first = settle(system, state, (delayed,))
    assert first.events == () and first.memory.pending[0].due_time_s == 2
    restored = CoordinateEventMemory.from_dict(json.loads(json.dumps(first.memory.to_dict())))
    due = replace(state, time_s=2., coordinates={**state.coordinates, "y": 7.})
    result = settle(system, due, (delayed,), restored)
    assert result.state.coordinates["x"] == expected
    assert result.memory.next_sequence == 1 and result.memory.generations["e"] == 1


@pytest.mark.parametrize("persistent,remaining", [(False, 0), (True, 1)])
def test_nonpersistent_delay_is_cancelled_when_trigger_becomes_false(persistent, remaining):
    system, state = setup()
    delayed = event(delay_s=lambda _: 2., persistent=persistent,
                    trigger=lambda s: s.coordinates["y"] > 0)
    first = settle(system, state, (delayed,))
    off = replace(state, time_s=1., coordinates={**state.coordinates, "y": 0.})
    second = settle(system, off, (delayed,), first.memory)
    assert len(second.memory.pending) == remaining
    again = refresh_coordinate_triggers(state, second.memory, (delayed,))
    assert again.generations["e"] == 2 and len(again.pending) == remaining+1


def test_priorities_are_reevaluated_after_each_execution():
    system, state = setup()
    first = event("first", priority=lambda _: 10., capture=lambda _: {"x": 4.})
    second = event("second", targets=("y",), priority=lambda s: s.coordinates["x"],
                   capture=lambda _: {"y": 5.})
    third = event("third", targets=("z",), priority=lambda _: 3., capture=lambda _: {"z": 6.})
    result = settle(system, state, (third, second, first))
    assert [e["event_id"] for e in result.events] == ["first", "second", "third"]
    assert result.events[1]["priority"] == 4.


@pytest.mark.parametrize("priority,match", [(None, "unspecified"), (lambda _: 1., "tied")])
def test_ambiguous_simultaneous_order_is_refused_without_mutating_owner(priority, match):
    system, state = setup()
    events = (event("a", priority=priority), event("b", priority=priority))
    memory = CoordinateEventMemory.initialize(events)
    with pytest.raises(CoordinateEventError, match=match) as exc:
        settle(system, state, events, memory)
    assert exc.value.executed_prefix == () and memory.next_sequence == 0
    assert state.coordinates["x"] == 1.


@pytest.mark.parametrize("priority", [lambda _: None, lambda _: float("nan"), lambda _: True])
def test_declared_priority_requires_a_finite_real(priority):
    system, state = setup()
    with pytest.raises(CoordinateEventError, match="priority must be finite"):
        settle(system, state, (event(priority=priority),))


def test_missed_deadline_and_unrepresentable_delay_are_failures():
    system, state = setup()
    delayed = event(delay_s=lambda _: 1.)
    first = settle(system, state, (delayed,))
    with pytest.raises(CoordinateEventError, match="deadline"):
        settle(system, replace(state, time_s=2.), (delayed,), first.memory)
    with pytest.raises(CoordinateEventError, match="resolution"):
        settle(system, replace(state, time_s=1e20), (delayed,))


def test_cascade_budget_retains_executed_prefix_but_no_partial_commit():
    system, state = setup()
    up = event("up", trigger=lambda s: s.coordinates["x"] == 1,
               capture=lambda _: {"x": 2.})
    down = event("down", trigger=lambda s: s.coordinates["x"] == 2,
                 capture=lambda _: {"x": 1.})
    memory = CoordinateEventMemory.initialize((up, down))
    with pytest.raises(CoordinateEventError, match="budget") as exc:
        settle(system, state, (up, down), memory, max_events=3)
    assert [e["event_id"] for e in exc.value.executed_prefix] == ["up", "down", "up"]
    assert memory.next_sequence == 0 and state.coordinates["x"] == 1


@pytest.mark.parametrize("apply", [lambda s, _: replace(s, time_s=1.),
    lambda s, _: replace(s, boundary_heat_J=1.),
    lambda s, _: replace(s, amounts_mol={"fake": 1.})])
def test_formal_coordinate_jump_cannot_write_physical_inventory_energy_or_clock(apply):
    system, state = setup()
    with pytest.raises(CoordinateEventError, match="cannot change clock"):
        settle(system, state, (event(apply=apply),))


@pytest.mark.parametrize("payload", [{"x": -1.}, {"x": float("nan")}, {"bad": 1.}])
def test_complete_candidate_bounds_and_payload_validation_are_atomic(payload):
    system, state = setup()
    with pytest.raises(CoordinateEventError):
        settle(system, state, (event(capture=lambda _: payload),))
    assert state.coordinates["x"] == 1


def test_continuation_rejects_unknown_fields_and_inconsistent_generations():
    events = (event(),)
    payload = CoordinateEventMemory.initialize(events).to_dict()
    with pytest.raises(ValueError, match="format/fields"):
        CoordinateEventMemory.from_dict({**payload, "time_s": 0.})
    with pytest.raises(ValueError, match="counters"):
        CoordinateEventMemory({"e": True}, {"e": 1},
                              (PendingCoordinateEvent("e", 2, 0, 1.),), 1)


def test_pending_budget_rejects_overlapping_generations_without_mutating_continuation():
    system, state = setup()
    delayed = event(delay_s=lambda _: 10., trigger=lambda s: s.coordinates["y"] > 0)
    first = settle(system, state, (delayed,), max_pending=1)
    off = replace(state, coordinates={**state.coordinates, "y": 0.})
    second = settle(system, off, (delayed,), first.memory, max_pending=1)
    with pytest.raises(CoordinateEventError, match="queue budget"):
        settle(system, state, (delayed,), second.memory, max_pending=1)
    assert second.memory.generations["e"] == 1 and len(second.memory.pending) == 1


def test_all_same_candidate_cancellations_free_capacity_before_new_scheduling():
    system, state = setup()
    old = event("old", delay_s=lambda _: 10., persistent=False,
                trigger=lambda s: s.coordinates["y"] > 0)
    new = event("new", delay_s=lambda _: 10., trigger=lambda s: s.coordinates["y"] == 0)
    events = (new, old)  # new is observed before the cancelled declaration
    first = settle(system, state, events, max_pending=1)
    off = replace(state, coordinates={**state.coordinates, "y": 0.})
    second = settle(system, off, events, first.memory, max_pending=1)
    assert [p.event_id for p in second.memory.pending] == ["new"]
    assert second.memory.next_sequence == 2
