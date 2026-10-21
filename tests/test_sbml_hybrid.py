"""Shared segment restarts, source triggers, queue deadlines and exact continuation."""

import json
import math

import pytest

from chemworld.physchem.coordinate_events import CoordinateEventError
from chemworld.physchem.coordinate_hybrid import (
    CoordinateHybridMemory,
    advance_coordinate_events,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.sbml_events import SBMLTriggerRoots, sbml_coordinate_events
from chemworld.physchem.sbml_kinetics import SBMLKinetics, sbml_continuous_system
from test_sbml import MODEL, model


def advance(frame, state, duration, memory=None, form="ode", **kwargs):
    events = sbml_coordinate_events(frame)
    program = SBMLTriggerRoots(frame)
    return advance_coordinate_events(sbml_continuous_system(frame, numerical_form=form),
        state, duration, CoordinateHybridMemory.initialize(events) if memory is None else memory,
        events, roots=program.roots, trigger_values=program.trigger_values, **kwargs)


def initial(frame):
    return CoupledState({}, 298.15, .03, coordinates=frame.initial_values)


@pytest.mark.parametrize("form", ["ode", "residual"])
def test_time_root_delay_and_assignment_restart_common_continuous_equations(form):
    frame = SBMLKinetics(model())
    result = advance(frame, initial(frame), 3., form=form)
    assert result.actual_duration_s == 3.
    assert len(result.events) == 1
    assert result.events[0]["time_s"] == pytest.approx(2.5, abs=1e-9)
    assert result.final_state.coordinates["N:A"] == pytest.approx(.02*math.exp(-.3*.5),
                                                                rel=3e-6)
    assert result.final_state.coordinates["P:p"] == pytest.approx(5.05, abs=1e-9)
    assert len(result.segment_diagnostics) == 3
    assert result.memory.events.pending == ()


def test_split_at_strict_root_restores_pending_capture_without_duplicate_trigger():
    frame = SBMLKinetics(model())
    first = advance(frame, initial(frame), 2.)
    assert first.events == () and len(first.memory.events.pending) == 1
    restored = CoordinateHybridMemory.from_dict(json.loads(json.dumps(first.memory.to_dict())))
    second = advance(frame, first.final_state, 1., restored)
    whole = advance(frame, initial(frame), 3.)
    assert len(second.events) == 1 and second.memory.events.next_sequence == 1
    assert second.final_state.coordinates == pytest.approx(whole.final_state.coordinates,
                                                          abs=2e-9)


@pytest.mark.parametrize("persistent,count", [(False, 0), (True, 1)])
def test_window_trigger_cancels_only_nonpersistent_delayed_instances(persistent, count):
    trigger = '''<apply><and/><apply><gt/><csymbol
        definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol><cn>2</cn>
        </apply><apply><lt/><csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">
        time</csymbol><cn>2.25</cn></apply></apply>'''
    start = MODEL.index('<math', MODEL.index('<trigger'))
    end = MODEL.index('</math>', start)+len('</math>')
    text = MODEL[:start]+'<math xmlns="http://www.w3.org/1998/Math/MathML">'+trigger+'</math>'+MODEL[end:]
    text = text.replace('persistent="false"', f'persistent="{str(persistent).lower()}"')
    frame = SBMLKinetics(model(text))
    result = advance(frame, initial(frame), 3.)
    assert len(result.events) == count
    assert result.memory.events.generations["switch"] == 1
    assert result.memory.events.pending == ()


def test_state_guard_root_restarts_on_the_post_assignment_side():
    start = MODEL.index('<math', MODEL.index('<trigger'))
    end = MODEL.index('</math>', start)+len('</math>')
    text = MODEL[:start]+'''<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/>
        <ci>p</ci><cn>3.2</cn></apply></math>'''+MODEL[end:]
    frame = SBMLKinetics(model(text))
    result = advance(frame, initial(frame), 3.)
    assert len(result.events) == 1
    assert result.events[0]["time_s"] == pytest.approx(2.5, abs=1e-8)
    assert result.final_state.coordinates["P:p"] == pytest.approx(5.05, abs=1e-9)


def test_event_execution_budget_is_global_across_continuous_segments():
    frame = SBMLKinetics(model())
    result = advance(frame, initial(frame), 3., max_events=1)
    assert len(result.events) == 1


def test_chained_inequality_with_static_equal_pair_has_no_fictitious_root():
    text = MODEL.replace('<cn>2</cn>\n</apply></math></trigger>',
                         '<cn>2</cn><cn>2</cn></apply></math></trigger>')
    frame = SBMLKinetics(model(text))
    result = advance(frame, initial(frame), 3.)
    assert result.events == ()


def test_math_root_comparison_views_do_not_change_raw_assignment_expressions():
    frame = SBMLKinetics(model())
    expression = frame.model.declaration.items("listOfEvents")[0].of("trigger")[0].mathematics
    node = expression._root
    scope = frame.context(frame.initial_values, 0.)
    assert expression.evaluate(scope) is False
    assert expression.evaluate(scope, comparison_values={node: True}) is True
    assert expression.evaluate(scope) is False
    with pytest.raises(ValueError, match="Boolean comparisons"):
        expression.evaluate(scope, comparison_values={node: 1})


@pytest.mark.parametrize("op", ["eq", "neq"])
def test_point_equality_guards_are_explicitly_outside_transverse_profile(op):
    frame = SBMLKinetics(model(MODEL.replace('<gt/>', f'<{op}/>')))
    with pytest.raises(ValueError, match="unsupported SBML event guard"):
        SBMLTriggerRoots(frame)


def test_zero_time_segment_budget_failure_keeps_original_state_and_prefix():
    frame = SBMLKinetics(model())
    state = initial(frame)
    with pytest.raises(CoordinateEventError, match="segment budget") as exc:
        advance(frame, state, 3., max_segments=2)
    assert len(exc.value.executed_prefix) == 1
    assert state.time_s == 0. and state.coordinates == frame.initial_values
