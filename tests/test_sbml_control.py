"""Native external model controls, atomic event reobservation and continuation."""

import json
from xml.etree import ElementTree as ET

import pytest

from chemworld.physchem.sbml_system import SBMLObservable, parameter_observable
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import model_record
from chemworld.world.sbml_world import sbml_world
from reference.run_sbml_trajectory_block import CORE, MATH, cases
from test_sbml import model


def runtime(text=None, **kwargs):
    parsed = model() if text is None else model(text)
    r = ChemWorldRuntime(composition=sbml_world(parsed, maximum_step_s=.02,
        controlled_parameters=("k", "p"), **kwargs))
    r.reset(seed=1)
    return r


def control(r, values):
    return r.execute({"operation": "set_model_parameters", "values": values})


@pytest.mark.parametrize("method", ["BDF", "LSODA"])
def test_declared_inputs_use_one_native_owner_and_solve_after_control(method):
    r = runtime(list(cases())[6][3], solver_method=method)
    assert r.execute({"operation": "wait", "duration_s": .5}).info[
        "operation_outcome"] == "success"
    before = model_record(r._state).settings["coordinates"]
    result = control(r, {"k": 0., "p": 4.})
    assert result.info["operation_outcome"] == "success"
    assert r._state.ledger.time_s == .5
    after = model_record(r._state).settings["coordinates"]
    assert after["N:A"] == before["N:A"] and after["N:B"] == before["N:B"]
    assert after["P:k"] == 0. and after["P:p"] == 4.
    assert r.execute({"operation": "wait", "duration_s": .5}).info[
        "operation_outcome"] == "success"
    after = model_record(r._state).settings["coordinates"]
    assert after["N:A"] == before["N:A"] and after["N:B"] == before["N:B"]
    assert after["P:p"] == pytest.approx(4.05)
    assert r._state.ledger.time_s == 1.
    services = r.runtime.domain_services
    assert services.formal_models is services.reaction_thermal.formal_models


def test_batch_control_does_not_trigger_on_a_partial_parameter_assignment():
    tree = ET.fromstring(model().to_xml())
    trigger = next(tree.iter("{"+CORE+"}trigger"))
    trigger.remove(trigger[0])
    trigger.append(ET.fromstring('<math xmlns="'+MATH+'"><apply><and/>'
        '<apply><gt/><ci>p</ci><cn>4</cn></apply>'
        '<apply><gt/><ci>k</ci><cn>1</cn></apply></apply></math>'))
    r = runtime(ET.tostring(tree, encoding="unicode"))
    assert control(r, {"k": 2.}).info["operation_outcome"] == "success"
    before = model_record(r._state).settings["coordinates"]
    assert control(r, {"p": 5., "k": 0.}).info["operation_outcome"] == "success"
    owned = model_record(r._state).settings
    assert owned["coordinates"]["N:A"] == before["N:A"]
    assert owned["continuation"]["events"]["pending"] == []
    assert owned["continuation"]["events"]["next_sequence"] == 0
    assert control(r, {"k": 2.}).info["operation_outcome"] == "success"
    assert len(model_record(r._state).settings["continuation"]["events"]["pending"]) == 1


@pytest.mark.parametrize("values", [{}, {"unknown": 1.}, {"k": True}, {"k": float("nan")},
    {"k": {"nested": 1}}, {"k": "0.2"}, {"k": 0., "unknown": 2.}])
def test_bad_or_undeclared_batch_cannot_partially_change_the_owned_model(values):
    r = runtime()
    before = model_record(r._state).to_dict()
    result = control(r, values)
    assert result.info["operation_outcome"] != "success"
    assert model_record(r._state).to_dict() == before
    assert r._state.ledger.time_s == 0.


def test_source_constant_event_assignments_remain_forbidden_for_an_external_control():
    r = runtime()
    frame = r.compiled_composition.spec.physics.models_of("sbml")["formal_workspace"].frame
    with pytest.raises(ValueError):
        frame.assigned_values(frame.initial_values, 0., {"k": 2.})


def test_controlled_snapshot_restores_live_inputs_and_captures_without_reinitializing():
    r = runtime()
    assert r.execute({"operation": "wait", "duration_s": 2.25}).info[
        "operation_outcome"] == "success"
    pending = model_record(r._state).settings["continuation"]
    assert control(r, {"k": 1., "p": 9.}).info["operation_outcome"] == "success"
    assert model_record(r._state).settings["continuation"] == pending
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(r.snapshot().to_dict())))
    action = {"operation": "wait", "duration_s": .75}
    assert r.execute(action).info["operation_outcome"] == "success"
    assert restored.execute(action).info["operation_outcome"] == "success"
    assert restored._state.to_dict() == r._state.to_dict()


def test_zero_time_event_budget_failure_rolls_back_all_control_inputs_and_event_effects():
    tree = ET.fromstring(list(cases())[15][3])
    events = next(tree.iter("{"+CORE+"}listOfEvents"))
    events.append(ET.fromstring('<event xmlns="'+CORE+'" id="second_cascade" '
        'useValuesFromTriggerTime="true"><trigger initialValue="false" persistent="true">'
        '<math xmlns="'+MATH+'"><apply><gt/><ci>A</ci><cn>2</cn></apply></math></trigger>'
        '<listOfEventAssignments><eventAssignment variable="B"><math xmlns="'+MATH+'">'
        '<cn>0.04</cn></math></eventAssignment></listOfEventAssignments></event>'))
    payload = sbml_world(model(ET.tostring(tree, encoding="unicode")), maximum_step_s=.02,
        controlled_parameters=("p", "k")).to_dict()
    payload["physics"]["models"]["sbml"]["bindings"]["formal_workspace"]["maximum_events"] = 1
    r = ChemWorldRuntime(composition=payload)
    r.reset(seed=5)
    before = model_record(r._state).to_dict()
    result = control(r, {"p": 5., "k": 8.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert model_record(r._state).to_dict() == before
    assert r._state.ledger.time_s == 0.


@pytest.mark.parametrize("index,scale", [(0, 1.), (1, 60.)])
def test_declared_parameter_and_reaction_readouts_use_coherent_native_units(index, scale):
    parsed = model(list(cases())[index][3])
    r = runtime(list(cases())[index][3], readout_relative_sd=0., observables=(
        parameter_observable(parsed, "k"), SBMLObservable("reaction_rate", "J")))
    result = r.execute({"operation": "measure", "instrument": "formal_quantity_meter"})
    assert result.info["operation_outcome"] == "success"
    channels = {c.channel_id: c for c in r.observe().channels}
    assert channels["parameter_k"].unit == "Hz"
    assert channels["parameter_k"].value == pytest.approx(.2/scale)
    assert channels["reaction_rate_J"].unit == "mol/s"
    assert channels["reaction_rate_J"].value == pytest.approx(
        .003*(.001 if index else 1.)/scale)
    assert all(c.uncertainty == 0. for c in channels.values())
    assert r._state.ledger.time_s == 0.


def test_a_discretely_controlled_parameter_can_hold_exactly_at_a_guard_threshold():
    tree = ET.fromstring(model().to_xml())
    trigger = next(tree.iter("{"+CORE+"}trigger"))
    trigger.remove(trigger[0])
    trigger.append(ET.fromstring('<math xmlns="'+MATH+'"><apply><gt/><ci>k</ci><cn>1</cn>'
                                '</apply></math>'))
    r = runtime(ET.tostring(tree, encoding="unicode"))
    assert control(r, {"k": 1.}).info["operation_outcome"] == "success"
    assert r.execute({"operation": "wait", "duration_s": .5}).info[
        "operation_outcome"] == "success"
    assert model_record(r._state).settings["continuation"]["events"]["pending"] == []
    assert control(r, {"k": 2.}).info["operation_outcome"] == "success"
    assert r.execute({"operation": "wait", "duration_s": .5}).info[
        "operation_outcome"] == "success"
    assert model_record(r._state).settings["coordinates"]["N:A"] == pytest.approx(.02)


def test_mixed_chained_guard_reobserves_held_inputs_instead_of_caching_their_initial_truth():
    tree = ET.fromstring(model().to_xml())
    trigger = next(tree.iter("{"+CORE+"}trigger"))
    trigger.remove(trigger[0])
    trigger.append(ET.fromstring('<math xmlns="'+MATH+'"><apply><lt/><ci>k</ci><cn>1</cn>'
        '<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
        '</apply></math>'))
    r = runtime(ET.tostring(tree, encoding="unicode"))
    assert control(r, {"k": 2.}).info["operation_outcome"] == "success"
    assert r.execute({"operation": "wait", "duration_s": 2.}).info[
        "operation_outcome"] == "success"
    memory = model_record(r._state).settings["continuation"]["events"]
    assert memory["pending"] == [] and memory["next_sequence"] == 0
    assert control(r, {"k": .5}).info["operation_outcome"] == "success"
    assert len(model_record(r._state).settings["continuation"]["events"]["pending"]) == 1
