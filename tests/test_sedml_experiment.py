"""Actual primitive operations, public measurements and shared-clock continuation."""

import json
import math

import pytest

from chemworld.eval.sedml_experiment import (
    SEDMLExecutionError,
    run_sedml_experiment,
    run_uniform_course,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sedml_protocol import SEDMLProtocol
from reference.run_sedml_initialization_diagnostic import model_xml
from test_sbml import model
from test_sedml_declarations import SEDML, document


def protocol():
    text = SEDML.replace('taskReference="scan" symbol="urn:sedml:symbol:time"',
        '''taskReference="t1" target="/s:sbml/s:model/s:listOfSpecies/s:species[@id='A']"
symbol="KISAO:0000836"''')
    return SEDMLProtocol(document(text), {"model.xml": model()})


def test_simple_course_uses_actual_wait_and_gauge_records_and_evaluates_its_report():
    plan = protocol()
    result = run_uniform_course(plan, "t1", seed=7)
    assert result.runtime.lab_status()["time_s"] == 4.
    assert len(result.history) == 21
    assert [record.action["operation"] for record in result.history].count("measure") == 11
    assert all(record.info["operation_outcome"] == "success" for record in result.history)
    quantity = plan.reports.bindings["dg", "time"][2]
    array = result.source_array(quantity)
    assert array.unit == "mole" and array.shape == (11,)
    assert array.sample_keys == tuple((i*.4,) for i in range(11))
    assert array.values[0] == pytest.approx(.01)
    report = plan.reports.evaluate({("dg", "time"): array})
    assert report["report"][0]["data"] == array.to_dict()
    assert "coordinates" not in json.dumps([record.public_view for record in result.history])


def test_second_invocation_preserves_native_time_state_and_pending_event_queue_after_json_restore():
    plan = protocol()
    first = run_uniform_course(plan, "t1", seed=8)
    snapshot = json.loads(json.dumps(first.runtime.snapshot().to_dict(), allow_nan=False))
    restored = ChemWorldRuntime.from_snapshot(snapshot)
    continued = run_uniform_course(plan, "t1", runtime=first.runtime)
    replayed = run_uniform_course(plan, "t1", runtime=restored)
    assert continued.runtime.lab_status()["time_s"] == 8.
    assert continued.native_channels == replayed.native_channels
    assert continued.runtime.snapshot().to_dict() == replayed.runtime.snapshot().to_dict()
    assert continued.native_channels["$time"].values[0] == 4.
    assert continued.native_channels["$time"].values[-1] == 8.


def test_a_report_with_only_time_uses_the_public_clock_without_a_fake_material_measurement():
    plan = SEDMLProtocol(document(), {"model.xml": model()})
    result = run_uniform_course(plan, "t1", seed=1)
    assert set(result.native_channels) == {"$time"}
    assert len(result.history) == 10
    assert all(record.action["operation"] == "wait" for record in result.history)


def test_unknown_or_repeated_task_is_rejected_before_a_native_session_is_created():
    plan = protocol()
    for identity in ("missing", "scan"):
        with pytest.raises(ValueError):
            run_uniform_course(plan, identity)


def scan_protocol(*, reset, concatenate=False):
    text = SEDML.replace('resetModel="true"', 'resetModel="'+str(reset).lower()+'"'
        ).replace('concatenate="false"', 'concatenate="'+str(concatenate).lower()+'"'
        ).replace('symbol="urn:sedml:symbol:time"',
            '''target="/s:sbml/s:model/s:listOfSpecies/s:species[@id='A']"''')
    source = model_xml("").replace('id="k" value="2" units="molar"',
        'id="k" value="0.2" units="per_second"').replace('<ci>rho</ci>', '<ci>k</ci>')
    return SEDMLProtocol(document(text), {"model.xml": model(source)})


@pytest.mark.parametrize("reset,concatenate", [(True, False), (True, True), (False, False),
                                              (False, True)])
def test_full_scan_runs_native_control_or_fresh_input_resets_and_preserves_report_axes(
        reset, concatenate):
    plan = scan_protocol(reset=reset, concatenate=concatenate)
    result = run_sedml_experiment(plan, seed=12)
    data = result.reports["report"][0]["data"]
    assert data["shape"] == ([33] if concatenate else [3, 11])
    assert len(data["sample_keys"]) == len(data["values"]) == 33
    assert len(result.iterations) == 3
    assert all(record.info["operation_outcome"] == "success" for record in result.history)
    values = data["values"]
    for index, k in enumerate((.1, .2, .4)):
        prior = 0. if reset else 4*sum((.1, .2, .4)[:index])
        assert values[index*11] == pytest.approx(math.exp(-prior), rel=2e-7)
        assert values[index*11+10] == pytest.approx(math.exp(-prior-4*k), rel=2e-7)
    controls = [record for record in result.history
                if record.action["operation"] == "set_model_parameters"]
    assert len(controls) == (0 if reset else 3)


@pytest.mark.parametrize("noise_mode", ["sequential", "keyed"])
def test_scan_numeric_reports_repeat_for_the_same_seed_and_native_snapshots_restore(noise_mode):
    original = scan_protocol(reset=True).to_dict()
    original["readout_relative_sd"] = .02
    plan = SEDMLProtocol.from_dict(original)
    first = run_sedml_experiment(plan, seed=4, observation_noise_mode=noise_mode)
    second = run_sedml_experiment(plan, seed=4, observation_noise_mode=noise_mode)
    assert first.reports == second.reports
    assert [(r.action, r.observation) for r in first.history] == [
        (r.action, r.observation) for r in second.history]
    # Fresh native campaigns have independent UUIDs. Full continuation preserves
    # the actual saved campaign identity rather than manufacturing a new one.
    saved = json.loads(json.dumps(first.snapshots["scan"]["m1"], allow_nan=False))
    left, right = (ChemWorldRuntime.from_snapshot(saved) for _ in range(2))
    for session in (left, right):
        assert session.execute({"operation": "wait", "duration_s": .5}).info[
            "operation_outcome"] == "success"
        assert session.execute({"operation": "measure", "instrument": "formal_quantity_meter"}
            ).info["operation_outcome"] == "success"
    assert left.snapshot().to_dict() == right.snapshot().to_dict()


def test_nested_resets_preserve_both_repeat_axes_and_reset_the_continued_inner_clock():
    original = scan_protocol(reset=False).to_dict()
    outer = '''<repeatedTask id="outer" range="outer_grid" resetModel="true" concatenate="false">
<listOfRanges><vectorRange id="outer_grid"><value>1</value><value>2</value></vectorRange>
</listOfRanges><listOfSubTasks><subTask task="scan"/></listOfSubTasks></repeatedTask>'''
    text = original["document"]["xml"].replace('</listOfTasks>', outer+'</listOfTasks>'
        ).replace('taskReference="scan"', 'taskReference="outer"')
    original["document"]["xml"] = text
    result = run_sedml_experiment(SEDMLProtocol.from_dict(original))
    data = result.reports["report"][0]["data"]
    assert data["shape"] == [2, 3, 11]
    assert data["axes"] == ["repeat:outer", "repeat:scan", "time"]
    assert data["values"][:33] == pytest.approx(data["values"][33:], rel=1e-12)
    restored = ChemWorldRuntime.from_snapshot(result.snapshots["outer"]["m1"])
    assert restored.lab_status()["time_s"] == 12.


def test_ordered_subtasks_continue_the_same_model_and_get_a_separate_output_axis():
    original = scan_protocol(reset=False).to_dict()
    original["document"]["xml"] = original["document"]["xml"].replace('<subTask task="t1"/>',
        '<subTask task="t1" order="2"/><subTask task="t1" order="1"/>')
    result = run_sedml_experiment(SEDMLProtocol.from_dict(original))
    data = result.reports["report"][0]["data"]
    assert data["shape"] == [3, 2, 11]
    assert data["values"][21] == pytest.approx(math.exp(-.8), rel=2e-7)
    assert data["values"][-1] == pytest.approx(math.exp(-5.6), rel=2e-7)


def test_model_dependent_functional_range_reads_the_current_native_parameter_each_iteration():
    original = scan_protocol(reset=False).to_dict()
    extra = '''<functionalRange id="dependent" range="grid"><listOfVariables>
<variable id="current_k" modelReference="m1"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']"/></listOfVariables>
<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>grid</ci>
<apply><times/><cn>0.1</cn><ci>current_k</ci></apply></apply></math></functionalRange>'''
    text = original["document"]["xml"].replace('range="grid"', 'range="dependent"'
        ).replace('<ci>grid</ci>', '<ci>dependent</ci>'
        ).replace('</listOfRanges>', extra+'</listOfRanges>')
    original["document"]["xml"] = text
    result = run_sedml_experiment(SEDMLProtocol.from_dict(original))
    assert [i["range_values"]["dependent"] for i in result.iterations] == pytest.approx(
        [.14, .214, .4214])
    assert result.reports["report"][0]["data"]["values"][-1] == pytest.approx(
        math.exp(-4*(.14+.214+.4214)), rel=2e-7)


def test_native_rejection_keeps_failed_action_history_and_the_actual_unchanged_snapshot():
    original = scan_protocol(reset=False).to_dict()
    original["document"]["xml"] = original["document"]["xml"].replace(
        '<value>0.1</value>', '<value>-0.1</value>')
    constraint = '''<listOfConstraints><constraint><math xmlns="http://www.w3.org/1998/Math/MathML">
<apply><geq/><ci>k</ci><cn>0</cn></apply></math></constraint></listOfConstraints>'''
    original["resources"]["model.xml"]["raw_text"] = original["resources"]["model.xml"][
        "raw_text"].replace('</model>', constraint+'</model>')
    with pytest.raises(SEDMLExecutionError) as failure:
        run_sedml_experiment(SEDMLProtocol.from_dict(original))
    error = failure.value
    assert len(error.history) == 1
    assert error.history[0].info["operation_outcome"] == "rejected"
    restored = ChemWorldRuntime.from_snapshot(error.snapshots["scan"]["m1"])
    assert restored.lab_status()["time_s"] == 0.


def test_sequential_direct_parameter_reads_see_the_batch_shadow_and_commit_only_once():
    original = scan_protocol(reset=False).to_dict()
    second = '''<setValue modelReference="m1" range="grid"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']/@value"><listOfVariables>
<variable id="read_k" modelReference="m1"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']"/></listOfVariables>
<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>read_k</ci>
<ci>grid</ci></apply></math></setValue>'''
    # Insert at the repeated-task boundary, after its first parameter change.
    text = original["document"]["xml"].replace('</setValue>\n</listOfChanges>',
        '</setValue>'+second+'\n</listOfChanges>')
    assert second in text
    original["document"]["xml"] = text
    result = run_sedml_experiment(SEDMLProtocol.from_dict(original))
    controls = [record.action for record in result.history
                if record.action["operation"] == "set_model_parameters"]
    assert [action["values"]["k"] for action in controls] == pytest.approx([.2, .4, .8])
    assert result.reports["report"][0]["data"]["values"][-1] == pytest.approx(
        math.exp(-5.6), rel=2e-7)


def test_data_generator_domain_failure_preserves_all_native_history_and_original_arrays():
    original = scan_protocol(reset=False).to_dict()
    original["document"]["xml"] = original["document"]["xml"].replace('<ci>time</ci>',
        '<apply><ln/><apply><minus/><ci>time</ci></apply></apply>')
    with pytest.raises(SEDMLExecutionError) as failure:
        run_sedml_experiment(SEDMLProtocol.from_dict(original))
    error = failure.value
    assert all(record.info["operation_outcome"] == "success" for record in error.history)
    assert len(error.variable_arrays["('dg', 'time')"]["values"]) == 33
    restored = ChemWorldRuntime.from_snapshot(error.snapshots["scan"]["m1"])
    assert restored.lab_status()["time_s"] == 12.
