"""Declared KiSAO policies consume the original integer world and experiment cursor."""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from chemworld.data.combine_archive import COMBINE_URI, ArchiveEntry, CombineArchive
from chemworld.data.sedml_simulations import resolve_simulations
from chemworld.data.sedml_stochastic import INTEGER_METHODS, SEDMLStochasticAlgorithm
from chemworld.eval.sedml_continuation import SEDMLExperimentSession
from chemworld.eval.sedml_experiment import SEDMLExecutionError, run_uniform_course
from chemworld.physchem.sbml_stochastic import initial_memory
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import model_record
from chemworld.world.sedml_protocol import SEDMLProtocol
from test_sbml_stochastic51 import source
from test_sedml_declarations import SEDML, document

METHODS = tuple(INTEGER_METHODS)
TAU = "KISAO:0000084"


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def options(values):
    return (
        "<listOfAlgorithmParameters>"
        + "".join(
            '<algorithmParameter kisaoID="' + key + '" value="' + str(value) + '"/>'
            for key, value in values
        )
        + "</listOfAlgorithmParameters>"
    )


def text(method=METHODS[0], parameters=(("KISAO:0000488", 17),), *, reset=True, root="scan"):
    start = SEDML.index("<algorithm kisaoID=")
    end = SEDML.index("</algorithm>", start) + len("</algorithm>")
    algorithm = '<algorithm kisaoID="' + method + '">' + options(parameters) + "</algorithm>"
    result = SEDML[:start] + algorithm + SEDML[end:]
    result = result.replace('resetModel="true"', 'resetModel="' + str(reset).lower() + '"')
    return result.replace(
        'taskReference="scan" symbol="urn:sedml:symbol:time"',
        'taskReference="' + root + '" target="/s:sbml/s:model/s:listOfSpecies/'
        's:species[@id=\'A\']" symbol="KISAO:0000836"',
    )


def protocol(method=METHODS[0], *, xml=None, parsed=None, noise=0, **kwargs):
    return SEDMLProtocol(
        document(text(method, **kwargs) if xml is None else xml),
        {"model.xml": source() if parsed is None else parsed},
        readout_relative_sd=noise,
    )


def native_state(runtime):
    return plain(model_record(runtime._state).settings)


@pytest.mark.parametrize("method", METHODS)
def test_requested_algorithm_and_seed_bind_original_formal_process(method):
    plan = protocol(method)
    policy = plan.tasks.simulations["s1"].algorithm
    assert isinstance(policy, SEDMLStochasticAlgorithm) and policy.seed == 17
    world = plan.worlds["m1"]
    spec = world.physics.models_of("sbml")["formal_workspace"]
    assert spec.numerical_form == "stochastic" and spec.solver_method == INTEGER_METHODS[method]
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=999)
    process = native_state(runtime)["continuation"]["process"]
    assert (
        process["rng_state"] == initial_memory(17, INTEGER_METHODS[method])["process"]["rng_state"]
    )
    assert process["time_s"] == 0 and process["events_committed"] == 0
    assert process["signature"] and process["pending"] is None
    assert runtime._state.species_amounts == {} and runtime._state.volume_L == 0
    restored = SEDMLProtocol.from_dict(plain(plan.to_dict()))
    assert restored.worlds["m1"].to_dict() == world.to_dict()
    runtime.close()


def test_tau_options_are_consumed_once_in_source_time_and_keep_separate_work_budgets():
    values = (
        ("KISAO:0000488", 37),
        ("KISAO:0000415", 123),
        ("KISAO:0000467", 0.02),
        ("KISAO:0000228", 0.04),
        ("KISAO:0000230", 0.5),
        ("KISAO:0000249", 7),
    )
    plan = protocol(TAU, parameters=values, parsed=source(time="minute"))
    spec = plan.worlds["m1"].physics.models_of("sbml")["formal_workspace"]
    assert spec.tau_policy.max_tau_s == 1.2
    assert spec.tau_policy.epsilon == 0.04 and spec.tau_policy.ssa_threshold == 0.5
    assert spec.tau_policy.critical_threshold == 7
    assert spec.tau_policy.max_steps_per_advance == 123 and spec.maximum_events == 1000000
    defaults = SEDMLStochasticAlgorithm(TAU).native_policy(seconds_per_source_time=60)
    assert defaults.tau_policy.max_tau_s == 0.1 and defaults.seed == 0


@pytest.mark.parametrize("method", METHODS[:2])
def test_exact_algorithms_consume_internal_event_budget(method):
    plan = protocol(method, parameters=(("KISAO:0000415", 23),))
    spec = plan.worlds["m1"].physics.models_of("sbml")["formal_workspace"]
    assert spec.maximum_events == 23 and spec.tau_policy is None


@pytest.mark.parametrize("method", ("KISAO:0000040", "KISAO:0000039", "KISAO:0000030"))
def test_unimplemented_variants_have_no_algorithm_substitution(method):
    with pytest.raises(ValueError, match="no KiSAO substitution"):
        protocol(method)


@pytest.mark.parametrize(
    "key,value",
    (
        ("KISAO:0000488", -1),
        ("KISAO:0000488", 2**32),
        ("KISAO:0000488", "1.0"),
        ("KISAO:0000415", 0),
        ("KISAO:0000415", 1000001),
        ("KISAO:0000415", "2.5"),
        ("KISAO:0000467", 0),
        ("KISAO:0000467", "NaN"),
        ("KISAO:0000228", 1.01),
        ("KISAO:0000228", 0),
        ("KISAO:0000230", 0),
        ("KISAO:0000249", 0),
        ("KISAO:0000249", "2.5"),
        ("KISAO:0000209", 0.01),
        ("KISAO:9999999", 1),
    ),
)
def test_invalid_or_inapplicable_options_reject_complete_protocol_before_execution(key, value):
    with pytest.raises(ValueError):
        protocol(TAU, parameters=((key, value),))


@pytest.mark.parametrize("method", METHODS[:2])
@pytest.mark.parametrize(
    "key", ("KISAO:0000467", "KISAO:0000228", "KISAO:0000230", "KISAO:0000249")
)
def test_exact_methods_do_not_silently_ignore_tau_options(method, key):
    with pytest.raises(ValueError):
        protocol(method, parameters=((key, 1),))


def test_duplicate_options_reject_and_global_seed_is_validated_before_override():
    with pytest.raises(ValueError, match="duplicate"):
        protocol(parameters=(("KISAO:0000488", 17), ("KISAO:0000488", 37)))
    global_values = options((("KISAO:0000488", 37), ("KISAO:0000415", 100)))
    plan = protocol(xml=text().replace("</sedML>", global_values + "</sedML>"))
    algorithm = plan.tasks.simulations["s1"].algorithm
    assert algorithm.seed == 17 and algorithm.maximum_steps == 100
    with pytest.raises(ValueError):
        protocol(xml=text().replace("</sedML>", options((("KISAO:0000488", -1),)) + "</sedML>"))
    with pytest.raises(ValueError, match="no supported applicable"):
        protocol(xml=text().replace("</sedML>", options((("KISAO:0000209", 0.01),)) + "</sedML>"))


def test_mixed_simulations_apply_each_global_option_only_to_its_supported_algorithm():
    ode = '<uniformTimeCourse id="ode" initialTime="0" outputStartTime="0" '
    ode += 'outputEndTime="4" numberOfSteps="10"><algorithm kisaoID="KISAO:0000288"/>'
    ode += "</uniformTimeCourse>"
    xml = text().replace("</listOfSimulations>", ode + "</listOfSimulations>")
    xml = xml.replace(
        "</sedML>", options((("KISAO:0000209", 1e-7), ("KISAO:0000488", 37))) + "</sedML>"
    )
    simulations = resolve_simulations(document(xml))
    assert simulations["ode"].algorithm.rtol == 1e-7
    assert simulations["s1"].algorithm.seed == 17


@pytest.mark.parametrize("method", METHODS)
def test_course_matches_native_operations_without_process_draws_from_measurement(method):
    plan = protocol(method, root="t1")
    course = run_uniform_course(plan, "t1", seed=37)
    manual = ChemWorldRuntime(composition=plan.worlds["m1"])
    manual.reset(seed=37)
    for record in course.history:
        before = native_state(manual)
        clock = manual._state.ledger.time_s
        result = manual.execute(record.action)
        assert result.info["transaction_status"] == "committed"
        if record.action["operation"] == "measure":
            assert native_state(manual) == before and manual._state.ledger.time_s == clock
    assert native_state(manual) == native_state(course.runtime)
    assert manual._state.ledger.time_s == 4
    quantity = plan.reports.bindings["dg", "time"][2]
    array = course.source_array(quantity)
    assert array.unit == "item" and array.shape == (11,)
    assert array.values[0] == pytest.approx(80)
    assert all(v >= 0 and v == pytest.approx(round(v)) for v in array.values)
    assert all("rng_state" not in json.dumps(r.public_view) for r in course.history)
    course.runtime.close()
    manual.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("reset", (True, False))
def test_whole_experiment_restores_full_course_cursor_and_reset_controls(method, reset):
    plan = protocol(method, reset=reset, noise=0.01)
    session = SEDMLExperimentSession(plan, seed=37, observation_noise_mode="keyed")
    session.advance(max_courses=1)
    checkpoint = plain(session.to_dict())
    restored = SEDMLExperimentSession.from_dict(checkpoint)
    assert restored.to_dict() == checkpoint
    session.advance()
    while not restored.complete:
        restored.advance(max_courses=1)
        restored = SEDMLExperimentSession.from_dict(plain(restored.to_dict()))
    assert plain(restored.to_dict()) == plain(session.to_dict())
    assert plain(asdict(restored.result())) == plain(asdict(session.result()))
    assert restored.result().reports["report"][0]["data"]["shape"] == [3, 11]
    controls = [
        r for r in restored.result().history if r.action["operation"] == "set_model_parameters"
    ]
    assert len(controls) == (0 if reset else 3)


@pytest.mark.parametrize("method", METHODS)
def test_xml_reinitialization_preserves_explicit_seed_and_process_policy(method):
    plan = protocol(method)
    world = plan.world_with_parameters("m1", {"k": 0.3})
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=123)
    state = native_state(runtime)
    process = state["continuation"]["process"]
    assert (
        process["rng_state"] == initial_memory(17, INTEGER_METHODS[method])["process"]["rng_state"]
    )
    assert process["time_s"] == 0 and process["events_committed"] == 0
    assert process["signature"] and process["pending"] is None
    assert state["coordinates"]["P:k"] == 0.3
    assert (
        world.physics.models_of("sbml")["formal_workspace"].solver_method == INTEGER_METHODS[method]
    )
    runtime.close()


@pytest.mark.parametrize("method", METHODS)
def test_same_explicit_process_seed_is_reused_by_reset_true_not_derived_from_detector_seed(method):
    xml = text(method).replace(
        "<value>0.1</value><value>0.2</value><value>0.4</value>",
        "<value>0.2</value><value>0.2</value><value>0.2</value>",
    )
    first = SEDMLExperimentSession(protocol(xml=xml), seed=7)
    second = SEDMLExperimentSession(protocol(xml=xml), seed=999)
    first.advance()
    second.advance()
    a = first.result().reports["report"][0]["data"]["values"]
    b = second.result().reports["report"][0]["data"]["values"]
    assert a[:11] == a[11:22] == a[22:] and a == b


@pytest.mark.parametrize("method", METHODS)
def test_combine_archive_entry_runs_same_source_policy(method):
    parsed = source()
    archive = CombineArchive(
        (
            ArchiveEntry("model.xml", COMBINE_URI + "sbml", parsed.to_xml().encode()),
            ArchiveEntry("experiment.sedml", COMBINE_URI + "sed-ml", text(method).encode(), True),
        ),
        parsed.source,
    )
    restored = CombineArchive.from_bytes(archive.to_bytes(), source=parsed.source)
    plan = SEDMLProtocol.from_archive(restored, items_per_mole=parsed.items_per_mole)
    session = SEDMLExperimentSession(plan)
    session.advance()
    assert session.complete and session.result().reports["report"][0]["data"]["shape"] == [3, 11]


@pytest.mark.parametrize("method", METHODS)
def test_new_process_resumes_exact_complete_experiment_checkpoint(method, tmp_path):
    session = SEDMLExperimentSession(protocol(method, reset=False), seed=37)
    session.advance(max_courses=1)
    checkpoint = tmp_path / "prefix.json"
    checkpoint.write_text(json.dumps(session.to_dict()), encoding="utf-8")
    destination = tmp_path / "suffix.json"
    code = """import json,sys
from dataclasses import asdict
from chemworld.eval.sedml_continuation import SEDMLExperimentSession
s=SEDMLExperimentSession.from_dict(json.load(open(sys.argv[1],encoding='utf-8')))
s.advance()
with open(sys.argv[2],'w',encoding='utf-8') as f:
 json.dump({'checkpoint':s.to_dict(),'result':asdict(s.result())},f,allow_nan=False)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(checkpoint), str(destination)], check=True, timeout=60
    )
    session.advance()
    assert json.loads(destination.read_text(encoding="utf-8")) == plain(
        {"checkpoint": session.to_dict(), "result": asdict(session.result())}
    )


@pytest.mark.parametrize("method", METHODS)
def test_exhausted_algorithm_budget_retains_failed_candidate_without_advancing_clock(method):
    plan = protocol(method, root="t1", parameters=(("KISAO:0000415", 1),))
    runtime = ChemWorldRuntime(composition=plan.worlds["m1"])
    runtime.reset(seed=37)
    before = native_state(runtime)
    with pytest.raises(SEDMLExecutionError) as caught:
        run_uniform_course(plan, "t1", runtime=runtime)
    assert caught.value.snapshot is not None
    assert native_state(runtime) == before and runtime._state.ledger.time_s == 0
    runtime.close()


@pytest.mark.parametrize("method", METHODS)
def test_nested_repeated_task_keeps_original_branch_axes_and_full_cursor(method):
    outer = """<repeatedTask id="outer" range="outer_grid" resetModel="true"
concatenate="false"><listOfRanges><vectorRange id="outer_grid"><value>0</value>
<value>1</value></vectorRange></listOfRanges><listOfSubTasks><subTask task="scan"/>
</listOfSubTasks></repeatedTask>"""
    xml = text(method, root="outer").replace("</listOfTasks>", outer + "</listOfTasks>")
    session = SEDMLExperimentSession(protocol(xml=xml), seed=37)
    session.advance(max_courses=4)
    restored = SEDMLExperimentSession.from_dict(plain(session.to_dict()))
    session.advance()
    restored.advance()
    assert plain(restored.to_dict()) == plain(session.to_dict())
    assert session.progress()["completed_courses"] == 6
    assert session.result().reports["report"][0]["data"]["shape"] == [2, 3, 11]


@pytest.mark.parametrize("method", METHODS)
def test_process_rng_and_coordinates_are_independent_of_detector_stream(method):
    plan = protocol(method, noise=0.1, root="t1")
    first = run_uniform_course(plan, "t1", seed=7)
    second = run_uniform_course(plan, "t1", seed=999, observation_noise_mode="keyed")
    assert native_state(first.runtime) == native_state(second.runtime)
    first.runtime.close()
    second.runtime.close()


@pytest.mark.parametrize("method", METHODS)
def test_nonzero_source_origin_and_source_minutes_consume_original_shared_clock(method):
    xml = text(method, root="t1").replace('initialTime="0"', 'initialTime="5"')
    xml = xml.replace('outputStartTime="0"', 'outputStartTime="6"')
    xml = xml.replace('outputEndTime="4"', 'outputEndTime="9"')
    plan = protocol(xml=xml, parsed=source(time="minute"))
    course = run_uniform_course(plan, "t1")
    state = native_state(course.runtime)
    assert course.runtime._state.ledger.time_s == 240
    assert state["continuation"]["process"]["time_s"] == 240
    assert course.native_channels["$time"].values[0] == 60
    assert course.native_channels["$time"].sample_keys[0] == (360.0,)
    course.runtime.close()
