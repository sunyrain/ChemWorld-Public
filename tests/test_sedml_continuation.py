"""Whole-task checkpointing preserves native state, resources, noise and graph position."""

import json
import math
import subprocess
import sys
from dataclasses import asdict

import pytest

from chemworld.data.sbml import SBMLModel
from chemworld.data.sedml_declarations import SEDMLDocument
from chemworld.eval.sedml_continuation import SEDMLExperimentSession
from chemworld.eval.sedml_experiment import SEDMLExecutionError, run_sedml_experiment
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sedml_protocol import SEDMLProtocol
from reference.run_sedml_initialization_diagnostic import experiment_xml, model_xml
from reference.sedml_experiment_cases import cases
from test_sbml import model


def protocol(identity, noise=0.01):
    case = next(case for case in cases() if case["id"] == identity)
    source = model().source
    plan = SEDMLProtocol(
        SEDMLDocument(case["sedml_xml"], source),
        {"models/model.xml": SBMLModel(case["model_xml"], source)},
        document_location="experiments/experiment.sedml",
        readout_relative_sd=noise,
    )
    return plan, case


@pytest.mark.parametrize("identity", [case["id"] for case in cases()])
@pytest.mark.parametrize("noise_mode", ["sequential", "keyed"])
def test_checkpoint_at_each_course_resumes_the_same_native_branch(identity, noise_mode):
    plan, case = protocol(identity)
    session = SEDMLExperimentSession(plan, seed=37, observation_noise_mode=noise_mode)
    session.advance(max_courses=1)
    checkpoint = json.loads(json.dumps(session.to_dict(), allow_nan=False))
    resumed = SEDMLExperimentSession.from_dict(checkpoint)
    assert resumed.to_dict() == checkpoint
    assert resumed.progress()["completed_courses"] == 1
    session.advance()
    while not resumed.complete:
        resumed.advance(max_courses=1)
        resumed = SEDMLExperimentSession.from_dict(resumed.to_dict())
    assert resumed.to_dict() == session.to_dict()
    assert asdict(resumed.result()) == asdict(session.result())
    assert len(resumed.result().history) == case["expected_operations"]
    assert [record.step for record in resumed.result().history] == list(
        range(1, case["expected_operations"] + 1)
    )


def test_unstarted_and_complete_continuations_do_not_repeat_physical_work():
    plan, case = protocol("C06", noise=0)
    session = SEDMLExperimentSession.from_dict(SEDMLExperimentSession(plan).to_dict())
    with pytest.raises(ValueError, match="final"):
        session.result()
    session.advance()
    restored = SEDMLExperimentSession.from_dict(session.to_dict())
    before = restored.to_dict()
    restored.advance()
    assert restored.to_dict() == before
    assert len(restored.result().history) == case["expected_operations"]
    original = run_sedml_experiment(plan)
    assert original.reports == restored.result().reports
    assert [record.action for record in original.history] == [
        record.action for record in restored.result().history
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "future"),
        ("root_index", 999),
        ("courses_completed", True),
        ("courses_completed", 999),
        ("noise_mode", "different"),
    ],
)
def test_inconsistent_or_unknown_continuation_policies_are_rejected(field, value):
    plan, _ = protocol("C08")
    session = SEDMLExperimentSession(plan)
    session.advance(max_courses=1)
    checkpoint = session.to_dict()
    checkpoint[field] = value
    with pytest.raises(ValueError):
        SEDMLExperimentSession.from_dict(checkpoint)


def test_snapshot_world_recipe_must_match_the_protocol_even_if_both_parse():
    plan, _ = protocol("C07")
    session = SEDMLExperimentSession(plan)
    session.advance(max_courses=1)
    checkpoint = session.to_dict()
    checkpoint["protocol"]["readout_relative_sd"] = 0.2
    with pytest.raises(ValueError, match="model/seed/noise"):
        SEDMLExperimentSession.from_dict(checkpoint)


def test_failure_before_first_session_keeps_the_original_error_and_failed_cursor(monkeypatch):
    plan, _ = protocol("C01")
    session = SEDMLExperimentSession(plan)

    def fail(*args, **kwargs):
        raise ValueError("declared initialization failed")

    monkeypatch.setattr(ChemWorldRuntime, "reset", fail)
    with pytest.raises(SEDMLExecutionError, match="declared initialization failed") as caught:
        session.advance()
    assert caught.value.history == ()
    assert caught.value.snapshot is None
    assert caught.value.snapshots == {"course": {}}
    checkpoint = caught.value.experiment_checkpoint
    restored = SEDMLExperimentSession.from_dict(checkpoint)
    with pytest.raises(RuntimeError, match="terminal"):
        restored.advance()


def test_fresh_protocols_have_independent_opaque_identities_despite_equal_seeds():
    plan, _ = protocol("C01", noise=0)
    first, second = SEDMLExperimentSession(plan), SEDMLExperimentSession(plan)
    assert first.identity_namespace != second.identity_namespace
    first.advance()
    second.advance()
    assert first.result().reports == second.result().reports
    assert (
        first.result().history[0].info["campaign_id"]
        != second.result().history[0].info["campaign_id"]
    )


@pytest.mark.parametrize(
    "identity", [False, "world-1", "episode-" + "G" * 32, "episode-" + "0" * 31]
)
def test_native_replay_identity_validation_precedes_reset_mutations(identity):
    plan, _ = protocol("C01", noise=0)
    runtime = ChemWorldRuntime(composition=plan.worlds["m"])
    runtime.reset(seed=7)
    before = runtime.snapshot().to_dict()
    with pytest.raises(ValueError, match="opaque"):
        runtime.reset(seed=99, options={"replay_campaign_id": identity})
    assert runtime.snapshot().to_dict() == before


@pytest.mark.parametrize("change", ["missing_child", "range", "unit", "frame_field"])
def test_invalid_stack_cannot_skip_outputs_or_stall_its_restored_runner(change):
    plan, _ = protocol("C08")
    session = SEDMLExperimentSession(plan)
    session.advance(max_courses=1)
    checkpoint = session.to_dict()
    frame = checkpoint["stack"][0]
    if change == "missing_child":
        frame["phase"] = "collect"
    elif change == "range":
        frame["ranges"] = {}
    elif change == "unit":
        frame["parts"][0]["m"]["$time"]["unit"] = "minute"
    else:
        frame["unknown"] = 1
    with pytest.raises(ValueError):
        SEDMLExperimentSession.from_dict(checkpoint)


def test_native_constraint_failure_retains_spent_operations_and_is_terminal_after_restore():
    plan, _ = protocol("C01", noise=0)
    constraint = (
        '<listOfConstraints><constraint><math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<apply><gt/><ci>A</ci><cn>0.8</cn></apply></math></constraint></listOfConstraints>"
    )
    original = plan.resources["models/model.xml"]
    constrained = SBMLModel(
        original.to_xml().replace("</model>", constraint + "</model>"), original.source
    )
    plan = SEDMLProtocol(
        plan.document, {"models/model.xml": constrained}, document_location=plan.document_location
    )
    session = SEDMLExperimentSession(plan)
    with pytest.raises(SEDMLExecutionError) as caught:
        session.advance()
    assert len(caught.value.history) > 1
    assert caught.value.history[-1].info["operation_outcome"] != "success"
    checkpoint = caught.value.experiment_checkpoint
    restored = SEDMLExperimentSession.from_dict(checkpoint)
    assert restored.to_dict() == checkpoint
    with pytest.raises(RuntimeError, match="terminal"):
        restored.advance()


def test_completed_root_outputs_and_a_new_root_continue_without_replaying_the_first():
    plan, _ = protocol("C07")
    generator = (
        '<dataGenerator id="dg_single"><listOfVariables><variable id="single_time" '
        'taskReference="course" symbol="urn:sedml:symbol:time"/></listOfVariables>'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>single_time</ci></math></dataGenerator>'
    )
    text = plan.document.to_xml().replace(
        "</listOfDataGenerators>", generator + "</listOfDataGenerators>"
    )
    text = text.replace(
        "</listOfDataSets>",
        '<dataSet id="single_column" label="single" dataReference="dg_single"/></listOfDataSets>',
    )
    plan = SEDMLProtocol(
        SEDMLDocument(text, plan.document.source),
        plan.resources,
        document_location=plan.document_location,
        readout_relative_sd=0.01,
    )
    session = SEDMLExperimentSession(plan)
    session.advance(max_courses=1)
    assert session.progress()["completed_roots"] == 1
    resumed = SEDMLExperimentSession.from_dict(session.to_dict())
    session.advance()
    resumed.advance()
    assert resumed.to_dict() == session.to_dict()
    assert len(resumed.result().history) == 52
    assert set(resumed.result().snapshots) == {"course", "scan"}


def test_new_process_resumes_the_whole_scan_with_a_preexisting_delayed_event(tmp_path):
    provenance = model().source
    document = SEDMLDocument(
        experiment_xml("E", False).replace('outputEndTime="1"', 'outputEndTime="0.1"'), provenance
    )
    plan = SEDMLProtocol(document, {"E.xml": SBMLModel(model_xml("E"), provenance)})
    session = SEDMLExperimentSession(plan, seed=37)
    session.advance(max_courses=1)
    checkpoint = tmp_path / "experiment.json"
    checkpoint.write_text(json.dumps(session.to_dict(), allow_nan=False), encoding="utf-8")
    result_file = tmp_path / "continued.json"
    code = """import json,sys
from pathlib import Path
from chemworld.eval.sedml_continuation import SEDMLExperimentSession
session=SEDMLExperimentSession.from_dict(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
while not session.complete:
    session.advance(max_courses=1)
Path(sys.argv[2]).write_text(json.dumps(session.to_dict(),allow_nan=False),encoding="utf-8")
"""
    child = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", code, str(checkpoint), str(result_file)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    session.advance()
    assert json.loads(result_file.read_text(encoding="utf-8")) == session.to_dict()
    columns = {column["label"]: column["data"] for column in session.result().reports["report"]}
    assert columns["a"]["values"][-1] == pytest.approx(2 * math.exp(-0.005), rel=2e-7)
