"""Source event callbacks and common pre-execution compartment semantics."""

from dataclasses import replace

import pytest

from chemworld.physchem.coordinate_events import CoordinateEventMemory, settle_coordinate_events
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.sbml_events import sbml_coordinate_events
from chemworld.physchem.sbml_kinetics import SBMLKinetics, sbml_continuous_system
from reference.run_sbml_input_block import cases
from test_sbml import MODEL, model


def test_source_delay_capture_priority_and_units_use_shared_seconds():
    frame = SBMLKinetics(model(list(cases())[5][3]))  # mmol / mL / minute
    events = sbml_coordinate_events(frame)
    system = sbml_continuous_system(frame)
    state = CoupledState({}, 298.15, .03, time_s=180., coordinates=frame.initial_values)
    first = settle_coordinate_events(system, state,
                                     CoordinateEventMemory.initialize(events), events)
    assert first.events == () and first.memory.pending[0].due_time_s == 210.
    state = replace(state, time_s=210., coordinates={**state.coordinates, "P:p": 9.})
    result = settle_coordinate_events(system, state, first.memory, events)
    assert result.events[0]["priority"] == 9.  # evaluated at execution, not capture
    assert result.state.coordinates["N:A"] == pytest.approx(2e-5)
    assert result.state.coordinates["P:p"] == 5.
    assert result.state.time_s == 210.


def test_same_event_concentration_and_volume_assignments_use_pre_event_volume():
    frame = SBMLKinetics(model())
    values = frame.assigned_values(frame.initial_values, 0., {"A": 2., "c1": .04})
    assert values["N:A"] == pytest.approx(.02)
    assert values["V:c1"] == .04
    assert frame.context(values, 0.)["A"] == .5
    reversed_order = frame.assigned_values(frame.initial_values, 0., {"c1": .04, "A": 2.})
    assert reversed_order == values
    unchanged_amount = frame.assigned_values(frame.initial_values, 0., {"c1": .04})
    assert unchanged_amount["N:A"] == .01
    assert frame.context(unchanged_amount, 0.)["A"] == .25


def test_derived_volume_recomputes_after_amount_targets_use_the_pre_event_view():
    text = MODEL.replace('</listOfRules>', '''<assignmentRule variable="c1"><math
        xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><cn>.01</cn><ci>p</ci>
        </apply></math></assignmentRule></listOfRules>''')
    frame = SBMLKinetics(model(text))
    assert "V:c1" not in frame.initial_values
    values = frame.assigned_values(frame.initial_values, 0., {"A": 2., "p": 5.})
    assert values["N:A"] == .06
    assert frame.quantities(values, 0.)[1]["c1"] == .05
    assert frame.context(values, 0.)["A"] == 1.2


@pytest.mark.parametrize("assignment", [{"c2": .1}, {"v": 5.}, {"missing": 1.},
                                         {"A": -2.}, {"c1": 0.}])
def test_invalid_or_derived_targets_and_invalid_post_jump_state_are_refused(assignment):
    frame = SBMLKinetics(model())
    with pytest.raises(ValueError):
        frame.assigned_values(frame.initial_values, 0., assignment)
    assert frame.initial_values["N:A"] == .01
