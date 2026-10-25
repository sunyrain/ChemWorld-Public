"""Actual source events, symbolic rules, atomic failures and persistent native queues."""

import json
import subprocess
import sys
from xml.etree import ElementTree as ET

import pytest

from chemworld.physchem.sbml_stochastic import ALGORITHMS
from chemworld.physchem.sbml_system import SBMLObservable, SBMLSystemSpec
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import model_record
from chemworld.world.sbml_world import sbml_world
from test_sbml import model
from test_sbml_stochastic51 import CORE, MATH, source
from test_sedml_stochastic51j import plain, protocol

METHODS = sorted(ALGORITHMS)
TAU = "adaptive-explicit-tau-leap-v1"
TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)


def math(body):
    return '<math xmlns="' + MATH + '">' + body + "</math>"


def comparison(operation, left, right):
    return "<apply><" + operation + "/>" + left + right + "</apply>"


def clock(value, operation="geq"):
    return comparison(operation, TIME, "<cn>" + str(value) + "</cn>")


def event(
    identity="change",
    guard=None,
    values=(("C", "<cn>5</cn>"),),
    *,
    delay=None,
    priority=None,
    persistent=True,
    captured=True,
    initial=False,
):
    guard = clock(0.25) if guard is None else guard
    xml = '<event id="' + identity + '" useValuesFromTriggerTime="' + str(captured).lower() + '">'
    xml += (
        '<trigger initialValue="'
        + str(initial).lower()
        + '" persistent="'
        + str(persistent).lower()
    )
    xml += '">' + math(guard) + "</trigger>"
    if priority is not None:
        xml += "<priority>" + math("<cn>" + str(priority) + "</cn>") + "</priority>"
    if delay is not None:
        xml += "<delay>" + math("<cn>" + str(delay) + "</cn>") + "</delay>"
    return (
        xml
        + "<listOfEventAssignments>"
        + "".join(
            '<eventAssignment variable="' + name + '">' + math(value) + "</eventAssignment>"
            for name, value in values
        )
        + "</listOfEventAssignments></event>"
    )


def parsed(*, events="", rules="", constraints="", original=None):
    original = source() if original is None else original
    tree = ET.fromstring(original.to_xml())
    declaration = tree.find("{" + CORE + "}model")
    for parameter in tree.iter("{" + CORE + "}parameter"):
        if parameter.get("id") == "k":
            parameter.set("constant", "false")
    for label, value in (
        ("listOfRules", rules),
        ("listOfConstraints", constraints),
        ("listOfEvents", events),
    ):
        if value:
            node = ET.fromstring(
                "<" + label + ' xmlns="' + CORE + '">' + value + "</" + label + ">"
            )
            declaration.append(node)
    return model(ET.tostring(tree, encoding="unicode"), items_per_mole=original.items_per_mole)


def world(method, original, **kwargs):
    return sbml_world(
        original,
        numerical_form="stochastic",
        solver_method=method,
        seed=17,
        tau_policy=TauLeapPolicy(max_tau_s=0.05) if method == TAU else None,
        maximum_events=100000,
        budget=100,
        readout_relative_sd=0,
        observables=(
            SBMLObservable("amount", "A"),
            SBMLObservable("amount", "B"),
            SBMLObservable("amount", "C"),
            SBMLObservable("volume", "c"),
        ),
        **kwargs,
    )


def runtime(method, original, **kwargs):
    result = ChemWorldRuntime(composition=world(method, original, **kwargs))
    result.reset(seed=37)
    return result


def owned(session):
    return plain(model_record(session._state).settings)


def counts(session):
    spec = session.compiled_composition.spec.physics.models_of("sbml")["formal_workspace"]
    return spec.integer_kinetics.counts(owned(session)["coordinates"], session._state.ledger.time_s)


def wait(session, duration):
    result = session.execute({"operation": "wait", "duration_s": duration})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def diagnostics(result):
    values = result.info["outcome_diagnostics"]
    return values.get("local_outcomes", {}).get("formal_workspace", values)


@pytest.mark.parametrize("method", METHODS)
def test_initial_event_settles_same_unique_coordinates_without_process_draw(method):
    session = runtime(method, parsed(events=event(guard="<true/>")))
    state = owned(session)
    assert counts(session)["C"] == 5 and state["continuation"]["process"]["time_s"] == 0
    queue = state["continuation"]["event_queue"]
    assert queue["generations"] == {"change": 1} and queue["pending"] == []
    assert state["continuation"]["process"]["events_committed"] == 0
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_time_event_stops_real_reactions_at_the_source_boundary(method):
    original = parsed(events=event(guard=clock(0.5), values=(("k", "<cn>0</cn>"),)))
    session = runtime(method, original)
    first = wait(session, 0.5)
    state = owned(session)
    assert 0 < counts(session)["B"] < 80 and state["coordinates"]["P:k"] == 0
    assert diagnostics(first)["source_events_executed"] == 1
    wait(session, 1.0)
    assert owned(session)["coordinates"] == state["coordinates"]
    assert (
        owned(session)["continuation"]["process"]["rng_state"]
        == state["continuation"]["process"]["rng_state"]
    )
    assert session._state.ledger.time_s == 1.5
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_inventory_guard_runs_after_first_actual_reaction_not_after_long_wait(method):
    original = parsed(
        events=event(
            guard=comparison("lt", "<ci>A</ci>", "<cn>80</cn>"), values=(("k", "<cn>0</cn>"),)
        )
    )
    session = runtime(method, original)
    wait(session, 2.0)
    assert counts(session) == {"A": 79, "B": 1, "C": 3}
    assert owned(session)["continuation"]["event_queue"]["generations"]["change"] == 1
    session.close()


def test_tau_guard_observes_whole_original_leap_batches_with_declared_step_held_semantics():
    original = parsed(
        original=source(counts={"A": 10000, "B": 0, "C": 3}),
        events=event(
            guard=comparison("lt", "<ci>A</ci>", "<cn>10000</cn>"), values=(("k", "<cn>0</cn>"),)
        ),
    )
    declaration = sbml_world(
        original,
        numerical_form="stochastic",
        solver_method=TAU,
        seed=17,
        tau_policy=TauLeapPolicy(epsilon=0.1, max_tau_s=0.1, ssa_threshold=0.01),
        maximum_events=100000,
    )
    session = ChemWorldRuntime(composition=declaration)
    session.reset(seed=37)
    result = wait(session, 1.0)
    assert 0 < counts(session)["A"] < 9999 and counts(session)["A"] + counts(session)["B"] == 10000
    assert diagnostics(result)["stochastic_work"]["leap_firings"] > 1
    assert diagnostics(result)["source_events_executed"] == 1
    session.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("captured", (True, False))
def test_delayed_payload_and_queue_survive_full_json_and_keep_trigger_or_execution_values(
    method, captured
):
    original = parsed(
        events=event(guard=clock(0.2), values=(("C", "<ci>A</ci>"),), delay=0.5, captured=captured)
    )
    session = runtime(method, original)
    wait(session, 0.3)
    checkpoint = plain(session.snapshot().to_dict())
    queue = owned(session)["continuation"]["event_queue"]
    assert len(queue["pending"]) == 1
    if captured:
        expected = queue["pending"][0]["payload"]["C"]
    else:
        pending = ChemWorldRuntime.from_snapshot(checkpoint)
        wait(pending, 0.4)
        expected = counts(pending)["A"]
        pending.close()
    restored = ChemWorldRuntime.from_snapshot(checkpoint)
    first, second = wait(session, 0.7), wait(restored, 0.7)
    assert plain(session._state.to_dict()) == plain(restored._state.to_dict())
    assert plain(first.info) == plain(second.info)
    assert counts(session)["C"] == expected
    session.close()
    restored.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("persistent", (True, False))
def test_lost_time_guard_cancels_only_nonpersistent_original_queue_entries(method, persistent):
    guard = "<apply><and/>" + clock(0.2) + clock(0.4, "lt") + "</apply>"
    session = runtime(method, parsed(events=event(guard=guard, delay=0.5, persistent=persistent)))
    wait(session, 1.0)
    assert counts(session)["C"] == (5 if persistent else 3)
    assert owned(session)["continuation"]["event_queue"]["pending"] == []
    session.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("reverse", (True, False))
def test_priorities_use_actual_post_execution_values_without_xml_order(method, reverse):
    high = event("high", priority=2)
    low = event(
        "low",
        priority=1,
        values=(("C", "<apply><plus/><ci>C</ci><cn>1</cn></apply>"),),
        captured=False,
    )
    session = runtime(method, parsed(events=low + high if reverse else high + low))
    result = wait(session, 1.0)
    assert counts(session)["C"] == 6 and diagnostics(result)["source_events_executed"] == 2
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_source_cascade_and_state_equality_run_at_the_original_jump_boundary(method):
    first = event("first", guard=comparison("eq", "<ci>B</ci>", "<cn>1</cn>"))
    second = event(
        "second", guard=comparison("geq", "<ci>C</ci>", "<cn>5</cn>"), values=(("C", "<cn>6</cn>"),)
    )
    session = runtime(method, parsed(events=first + second))
    wait(session, 1.0)
    assert counts(session)["C"] == 6
    assert owned(session)["continuation"]["event_queue"]["generations"] == {"first": 1, "second": 1}
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_recurrent_state_edges_keep_real_queue_generations_and_positive_clock_steps(method):
    original = parsed(
        events=event(
            guard=comparison("lt", "<ci>A</ci>", "<cn>80</cn>"), values=(("A", "<cn>80</cn>"),)
        )
    )
    session = runtime(method, original)
    wait(session, 1.0)
    state = owned(session)
    assert counts(session)["A"] == 80 and counts(session)["B"] > 1
    assert state["continuation"]["event_queue"]["generations"]["change"] == counts(session)["B"]
    assert session._state.ledger.time_s == state["continuation"]["process"]["time_s"] == 1
    session.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("failure", ("fractional", "tie", "queue", "steps"))
def test_invalid_source_assignments_or_budgets_roll_back_all_process_and_queue_state(
    method, failure
):
    value = "<cn>0.5</cn>" if failure == "fractional" else "<cn>5</cn>"
    events = event(values=(("C", value),), delay=0.5 if failure == "queue" else None)
    if failure in {"tie", "queue"}:
        events += event("second", delay=0.5 if failure == "queue" else None)
    declaration = world(method, parsed(events=events)).to_dict()
    spec = declaration["physics"]["models"]["sbml"]["bindings"]["formal_workspace"]
    if failure == "queue":
        spec["maximum_pending_events"] = 1
    if failure == "steps":
        spec["maximum_segments"] = 1
    from chemworld.world.composition import WorldCompositionSpec

    session = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(declaration))
    session.reset(seed=37)
    before = owned(session)
    result = session.execute({"operation": "wait", "duration_s": 1.0})
    assert result.info["transaction_status"] != "committed"
    assert owned(session) == before and session._state.ledger.time_s == 0
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_source_minutes_origin_and_delay_convert_once_to_shared_seconds(method):
    original = parsed(original=source(time="minute"), events=event(guard=clock(5.5), delay=0.25))
    session = runtime(method, original, initial_model_time=5.0)
    wait(session, 40.0)
    assert counts(session)["C"] == 3
    queue = owned(session)["continuation"]["event_queue"]
    assert queue["pending"][0]["due_time_s"] == 45
    wait(session, 20.0)
    assert counts(session)["C"] == 5 and session._state.ledger.time_s == 60
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_source_volume_and_concentration_assignments_use_original_pre_jump_units(method):
    original = source(concentration=True, coefficient=0.0)
    tree = ET.fromstring(original.to_xml())
    next(n for n in tree.iter("{" + CORE + "}compartment") if n.get("id") == "c").set(
        "constant", "false"
    )
    original = model(ET.tostring(tree, encoding="unicode"), items_per_mole=original.items_per_mole)
    original = parsed(
        original=original, events=event(values=(("c", "<cn>2e-15</cn>"), ("A", "<cn>1e16</cn>")))
    )
    session = runtime(method, original)
    wait(session, 1.0)
    assert counts(session)["A"] == 10 and owned(session)["coordinates"]["V:c"] == 2e-15
    session.close()


def ruled(order=1, *, expression=None, local=False):
    original = source({"A": order}, {"B": order})
    tree = ET.fromstring(original.to_xml())
    group = next(tree.iter("{" + CORE + "}listOfParameters"))
    ET.SubElement(group, "{" + CORE + "}parameter", {"id": "q", "constant": "false", "value": "0"})
    law = next(tree.iter("{" + CORE + "}kineticLaw"))
    law.remove(next(law.iter("{" + MATH + "}math")))
    law.append(ET.fromstring(math("<ci>q</ci>")))
    if local:
        group = ET.SubElement(law, "{" + CORE + "}listOfLocalParameters")
        ET.SubElement(group, "{" + CORE + "}localParameter", {"id": "q", "value": "0.5"})
        law.remove(next(law.iter("{" + MATH + "}math")))
        law.insert(0, ET.fromstring(math("<apply><times/><ci>q</ci><ci>A</ci></apply>")))
    expression = (
        "<apply><times/><ci>k</ci>" + ("<ci>A</ci>" * order) + "</apply>"
        if expression is None
        else expression
    )
    rules = '<assignmentRule variable="q">' + math(expression) + "</assignmentRule>"
    return parsed(
        original=model(
            ET.tostring(tree, encoding="unicode"), items_per_mole=original.items_per_mole
        ),
        rules=rules,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("order", (1, 2, 3))
def test_assignment_rule_is_symbolically_expanded_to_full_falling_factorial_orders(method, order):
    spec = SBMLSystemSpec(
        ruled(order),
        numerical_form="stochastic",
        solver_method=method,
        tau_policy=TauLeapPolicy() if method == TAU else None,
    )
    (channel,) = spec.integer_kinetics.compile(spec.frame.initial_values, 0.0)
    expected = 0.2
    for j in range(order):
        expected *= 80 - j
    assert dict(channel.reactants) == {"A": order}
    assert channel.propensity({"A": 80, "B": 0, "C": 3}) == pytest.approx(expected, rel=1e-12)
    session = runtime(method, spec.model)
    wait(session, 0.1)
    assert counts(session)["A"] + counts(session)["B"] == 80
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_reaction_local_parameter_shadows_global_assignment_without_capture(method):
    spec = SBMLSystemSpec(
        ruled(local=True),
        numerical_form="stochastic",
        solver_method=method,
        tau_policy=TauLeapPolicy() if method == TAU else None,
    )
    (channel,) = spec.integer_kinetics.compile(spec.frame.initial_values, 0.0)
    assert channel.propensity({"A": 80, "B": 0, "C": 3}) == 40


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "invalid",
    ("rate", "time_rule", "nonlinear_guard", "time_equality", "time_constraint", "non_elementary"),
)
def test_continuously_varying_or_unproved_source_profiles_reject_before_runtime(method, invalid):
    if invalid == "rate":
        original = parsed(rules='<rateRule variable="k">' + math("<cn>1</cn>") + "</rateRule>")
    elif invalid == "time_rule":
        original = ruled(expression=TIME)
    elif invalid == "nonlinear_guard":
        original = parsed(
            events=event(
                guard=comparison("geq", "<apply><times/>" + TIME + TIME + "</apply>", "<cn>1</cn>")
            )
        )
    elif invalid == "time_equality":
        original = parsed(events=event(guard=clock(1, "eq")))
    elif invalid == "time_constraint":
        original = parsed(constraints="<constraint>" + math(clock(0)) + "</constraint>")
    else:
        original = ruled(expression="<apply><divide/><ci>k</ci><ci>A</ci></apply>")
    with pytest.raises(ValueError):
        world(method, original)


@pytest.mark.parametrize("method", METHODS)
def test_real_pending_source_queue_new_process_suffix_is_full_state_info_exact(method, tmp_path):
    session = runtime(method, parsed(events=event(guard=clock(0.2), delay=0.5)))
    wait(session, 0.3)
    prefix, destination = tmp_path / "prefix.json", tmp_path / "actual.json"
    prefix.write_text(json.dumps(session.snapshot().to_dict()), encoding="utf-8")
    code = """import json,sys
from chemworld.runtime import ChemWorldRuntime
s=ChemWorldRuntime.from_snapshot(json.load(open(sys.argv[1],encoding='utf-8')))
r=s.execute({'operation':'wait','duration_s':.7})
with open(sys.argv[2],'w',encoding='utf-8') as f:
 json.dump({'state':s._state.to_dict(),'info':r.info},f)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(prefix), str(destination)], check=True, timeout=60
    )
    result = wait(session, 0.7)
    assert json.loads(destination.read_text(encoding="utf-8")) == plain(
        {"state": session._state.to_dict(), "info": result.info}
    )
    session.close()


@pytest.mark.parametrize("method", ("KISAO:0000029", "KISAO:0000027", "KISAO:0000084"))
def test_original_standard_experiment_cursor_consumes_source_events_at_all_reset_boundaries(method):
    from chemworld.eval.sedml_continuation import SEDMLExperimentSession

    plan = protocol(method, parsed=parsed(events=event()))
    session = SEDMLExperimentSession(plan, seed=37)
    session.advance(max_courses=1)
    restored = SEDMLExperimentSession.from_dict(plain(session.to_dict()))
    session.advance()
    restored.advance()
    assert plain(session.to_dict()) == plain(restored.to_dict())
    runtime = ChemWorldRuntime.from_snapshot(session.result().snapshots["scan"]["m1"])
    assert counts(runtime)["C"] == 5
    runtime.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("valid", (True, False))
def test_state_constraints_are_checked_at_each_actual_reaction_and_failure_is_atomic(method, valid):
    condition = (
        comparison("eq", "<apply><plus/><ci>A</ci><ci>B</ci></apply>", "<cn>80</cn>")
        if valid
        else comparison("leq", "<ci>B</ci>", "<cn>0</cn>")
    )
    original = parsed(constraints="<constraint>" + math(condition) + "</constraint>")
    session = runtime(method, original)
    before = owned(session)
    result = session.execute({"operation": "wait", "duration_s": 1.0})
    if valid:
        assert result.info["transaction_status"] == "committed"
        assert counts(session)["A"] + counts(session)["B"] == 80
    else:
        assert result.info["transaction_status"] != "committed"
        assert owned(session) == before and session._state.ledger.time_s == 0
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_real_parameter_control_settles_source_event_at_zero_time_without_process_draw(method):
    original = parsed(events=event(guard=comparison("gt", "<ci>k</ci>", "<cn>0.5</cn>")))
    session = runtime(method, original, controlled_parameters=("k",))
    before = owned(session)["continuation"]["process"]["rng_state"]
    result = session.execute({"operation": "set_model_parameters", "values": {"k": 1.0}})
    assert result.info["transaction_status"] == "committed"
    assert diagnostics(result)["source_events_executed"] == 1 and counts(session)["C"] == 5
    assert owned(session)["continuation"]["process"]["rng_state"] == before
    assert session._state.ledger.time_s == 0
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_snapshot_cannot_forge_an_unsettled_source_edge(method):
    session = runtime(method, parsed(events=event(guard=clock(0.2), delay=0.5)))
    wait(session, 0.3)
    snapshot = plain(session.snapshot().to_dict())
    payload = owned(session)

    # Replace only the persisted queue in the original EquipmentRecord payload.
    def corrupt(node):
        if isinstance(node, dict):
            if node.get("format") == payload["continuation"]["format"]:
                node["event_queue"]["previous"]["change"] = False
                return True
            return any(corrupt(value) for value in node.values())
        if isinstance(node, list):
            return any(corrupt(value) for value in node)
        return False

    assert corrupt(snapshot["world_state"])
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(snapshot)
    session.close()


@pytest.mark.parametrize("method", METHODS)
def test_low_level_default_cannot_silently_skip_source_boundary_coordination(method):
    from chemworld.physchem.sbml_stochastic import initial_memory

    spec = SBMLSystemSpec(
        parsed(events=event()),
        numerical_form="stochastic",
        solver_method=method,
        tau_policy=TauLeapPolicy() if method == TAU else None,
    )
    with pytest.raises(ValueError, match="boundary coordination"):
        spec.integer_kinetics.advance(
            spec.frame.initial_values,
            initial_memory(17, method),
            0.0,
            1.0,
            method,
            100000,
            spec.tau_policy,
        )


@pytest.mark.parametrize("method", METHODS)
def test_reported_kernel_work_includes_all_real_zero_time_rebind_calls(method, monkeypatch):
    from chemworld.physchem.sbml_integer_events import advance_integer_events
    from chemworld.runtime.sbml_models import _view

    session = runtime(method, parsed(events=event()))
    spec = session.compiled_composition.spec.physics.models_of("sbml")["formal_workspace"]
    state = owned(session)
    original = type(spec.integer_kinetics).advance
    calls = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args[4], result[2].work))
        return result

    monkeypatch.setattr(type(spec.integer_kinetics), "advance", observed)
    result = advance_integer_events(
        spec, _view(session._state, spec, state["coordinates"]), state["continuation"], 1.0
    )
    assert any(duration == 0 for duration, _ in calls)
    assert any(duration > 0 for duration, _ in calls)
    for counter in ("propensity_evaluations", "random_draws"):
        assert result.work[counter] == sum(row.get(counter, 0) for _, row in calls)
    session.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("corruption", ("missing_counter", "fractional_counter", "extra_field"))
def test_source_process_schema_rejects_corruption_before_reading_its_counter(method, corruption):
    from chemworld.physchem.sbml_integer_events import BOUNDARY_MEMORY_FORMAT

    session = runtime(method, parsed(events=event()))
    snapshot = plain(session.snapshot().to_dict())

    def corrupt(node):
        if isinstance(node, dict):
            if node.get("format") == BOUNDARY_MEMORY_FORMAT:
                process = node["process"]
                if corruption == "missing_counter":
                    process.pop("events_committed")
                elif corruption == "fractional_counter":
                    process["events_committed"] = 0.5
                else:
                    process["extra"] = 0
                return True
            return any(corrupt(value) for value in node.values())
        if isinstance(node, list):
            return any(corrupt(value) for value in node)
        return False

    assert corrupt(snapshot["world_state"])
    with pytest.raises(ValueError, match="process fields"):
        ChemWorldRuntime.from_snapshot(snapshot)
    session.close()
