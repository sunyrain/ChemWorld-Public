"""Assessment fixtures are synthetic data, never independent scientific evidence."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import run_battery_parameters45 as runner
from battery_parameters45_cases import CASE_IDS, RESPONSE_ORDER, case_data, routes, stage_times

from chemworld.physchem.battery_models import F
from chemworld.physchem.battery_parameter_sets import battery_parameter_set


@pytest.fixture(scope="module")
def fixture_histories():
    """Artificial consistent bookkeeping and cutoffs test the assessment machinery."""
    parameters = battery_parameter_set()
    manifest = {key: {"kind": "number", "value": value.value}
                for key, value in parameters.scalar_inputs.items()}
    manifest.update({law.source_field: {"kind": "function", "function_id": law.function_id}
                     for law in parameters.functions})
    manifest.update({key: {"kind": "metadata", "value": value}
                     for key, value in parameters.metadata.items()})
    specs = {n: parameters.build_spm(maximum_current_A=10, radial_cells=n) for n in (32, 64)}
    result = {}
    for route in routes():
        data, spec = case_data(route["unit"]), specs[route["radial_cells"]]
        native = route["implementation"] == "native"
        count = route["radial_cells"]
        mesh = {}
        for electrode in ("negative", "positive"):
            particle = getattr(spec, electrode)
            edges = (np.asarray(particle.mesh.faces_m) if native
                     else np.linspace(0, particle.radius_m, count + 1))
            nodes = (np.asarray(particle.mesh.centers_m) if native
                     else (edges[1:] + edges[:-1]) / 2)
            mesh[electrode] = {"edges_m": edges.tolist(), "nodes_m": nodes.tolist(),
                               "radius_m": particle.radius_m, "cells": count,
                               "class": "DiffusionMesh" if native else "Uniform1DSubMesh"}
        events = [{"name": name, "type": "TERMINATION"}
                  for name in ("Minimum voltage [V]", "Maximum voltage [V]")]
        contract = {"mesh": mesh, "F_C_mol": F, "spec": spec.to_dict(),
                    "pybamm_version": "25.4.2", "casadi_version": "3.6.7",
                    "parameter_updates": {}, "source_parameter_manifest": manifest,
                    "original_events": events, "all_events": [*events, *[
                        {"name": str(i), "type": "TERMINATION"} for i in range(4)]],
                    "rhs_size": 2 * count + 4, "algebraic_size": 1}
        initial = spec.initial_state()
        original = {e: sum(initial.amounts_mol[k] for k in getattr(spec, e).keys(e))
                    for e in ("negative", "positive")}
        charge, time_s, segments = 0., 0., []
        for index, stage in enumerate(data["stages"]):
            duration = stage["duration_s"] if stage["expected"] == "complete" else 1000.
            times = sorted({0., duration, *(t for t in stage_times(stage) if t <= duration)})
            current = stage["target"] if stage["unit"] == "A" else 1.
            records = []
            for relative in times:
                q = charge + current * relative
                if stage["unit"] == "V":
                    voltage = stage["target"]
                elif stage["expected"] == "minimum-voltage":
                    voltage = 4 - 1.5 * relative / duration
                elif stage["expected"] == "maximum-voltage":
                    voltage = 4 + .2 * relative / duration
                else:
                    voltage = 4.
                row = [voltage, current, q, 4 * q,
                       original["negative"] - q / F, original["positive"] + q / F]
                fractions = {e: row[4 + i] / getattr(spec, e).capacity_mol
                             for i, e in enumerate(("negative", "positive"))}
                row.extend((fractions["negative"], fractions["positive"],
                            fractions["negative"], fractions["positive"], .1, 4.1, .01, -.01))
                record = {"time_s": time_s + relative, "row": row,
                          "profiles": {e: {"fraction": [fractions[e]] * count}
                                       for e in ("negative", "positive")},
                          "native_state": {"synthetic_fixture": True},
                          "source_y": [0.] * (2 * count + 5),
                          "source_rhs": [0.] * (2 * count + 4),
                          "source_algebraic": [0.],
                          "source_particle_Li_mol": row[4] + row[5],
                          "source_total_Li_including_electrolyte_mol": row[4] + row[5] + .01,
                          "source_events": {event["name"]: [1.]
                                            for event in contract["all_events"]}}
                records.append(record)
            boundary = deepcopy(records[0])
            boundary["differential_state_unchanged_by_boundary_root"] = True
            boundary["boundary_operation"] = "source algebraic root only; no integration"
            segment = {"index": index, "request": deepcopy(stage),
                       "boundary_t0": boundary, "records": records,
                       "actual_start_time_s": time_s, "actual_end_time_s": time_s + duration,
                       "original_reported_times_s": [r["time_s"] for r in records],
                       "termination": "final time" if stage["expected"] == "complete"
                       else "event: " + stage["expected"],
                       "original_residual_reports": [{"variables": [{"name": "fixture"}],
                                                      "points": [{"y": [0.], "ydot": [0.]}]}],
                       "original_all_ts": [[r["time_s"] for r in records]],
                       "original_all_ys": [np.zeros((2 * count + 5, len(records))).tolist()]}
            segments.append(segment)
            time_s += duration
            charge += current * duration
        result[runner.route_id(route)] = {"contract": deepcopy(contract), "segments": segments,
                                          "synthetic_fixture_not_scientific_evidence": True}
    return result


def test_complete_plan_and_absence_have_the_same_fixed_check_denominator():
    plan = runner.plan()
    assert (plan["histories"], plan["authored_stages"], plan["authored_observation_slots"]) \
        == (24, 48, 2048)
    assert (plan["comparisons"], plan["comparison_stages"], plan["check_denominator"]) \
        == (24, 48, 2544)
    assessment = runner.assess_histories({})
    assert assessment["evaluated_checks"] == assessment["failed_checks"] == 2544
    assert len({c["name"] for c in assessment["checks"]}) == 2544
    assert assessment["all_passed"] is False
    assert all(unit["passed"] is False for unit in assessment["units"])


def test_cutoff_fixture_passes_without_requiring_all_post_cutoff_request_slots(fixture_histories):
    assessment = runner.assess_histories(fixture_histories)
    assert assessment["all_passed"], [c for c in assessment["checks"] if not c["passed"]][:8]
    assert assessment["evaluated_checks"] == 2544
    assert sum(len(s["records"]) for h in fixture_histories.values() for s in h["segments"]) < 2048
    assert len(assessment["comparisons"]) == 24
    assert sum(len(c["stages"]) for c in assessment["comparisons"]) == 48


def test_all_fourteen_scalar_channels_include_ocp_and_kinetic_overpotential(fixture_histories):
    histories = deepcopy(fixture_histories)
    column = RESPONSE_ORDER.index("positive_overpotential_V")
    histories["P02/source/fine"]["segments"][0]["records"][1]["row"][column] += .02
    assessment = runner.assess_histories(histories)
    failures = {c["name"]: c for c in assessment["checks"] if not c["passed"]}
    name = "P02/cross-fine/stage-0/common:positive_overpotential_V"
    assert failures[name]["limit"] == .005
    assert failures["P02/source-refinement/stage-0/common:positive_overpotential_V"]["limit"] \
        == .0025


def test_unpaired_unfavorable_tail_endpoint_and_event_identity_cannot_disappear(fixture_histories):
    histories = deepcopy(fixture_histories)
    stage = histories["P03/source/fine"]["segments"][0]
    stage["records"][-1]["time_s"] += 1
    stage["records"][-1]["row"][3] += 100
    stage["original_reported_times_s"][-1] += 1
    stage["actual_end_time_s"] += 1
    assessment = runner.assess_histories(histories)
    paired = next(c for c in assessment["comparisons"]
                  if c["unit"] == "P03" and c["kind"] == "cross-fine")["stages"][0]
    endpoint = next(c for c in paired["checks"] if c["name"].endswith("endpoint:work_J"))
    assert endpoint["max_abs"] == 100 and endpoint["passed"] is False
    assert paired["tail_audit"]["right"]["unpaired_duration_s"] == 1
    histories["P03/source/fine"]["segments"][0]["termination"] = "event: Maximum voltage [V]"
    assessment = runner.assess_histories(histories)
    assert any(c["name"] == "P03/cross-fine/stage-0/termination-kind" and not c["passed"]
               for c in assessment["checks"])
    assert assessment["evaluated_checks"] == 2544


def test_missing_and_malformed_histories_keep_the_complete_denominator(fixture_histories):
    histories = deepcopy(fixture_histories)
    del histories["P04/source/fine"]
    histories["P05/native/fine"]["segments"][0]["records"][0]["row"] = [float("nan")]
    histories["P01/source/coarse"] = ["malformed non-dictionary history"]
    assessment = runner.assess_histories(histories)
    assert not assessment["all_passed"]
    assert assessment["evaluated_checks"] == assessment["check_denominator"] == 2544
    assert assessment["available_histories"] == 22


def test_archives_are_verified_exclusive_and_retain_nonfinite_failure_data(tmp_path):
    path = tmp_path / "trace.json.gz"
    value = {"partial_history": {"y": np.array([1., np.nan, np.inf])}}
    archive = runner.archive_new(path, value)
    saved = path.read_bytes()
    assert archive["archive_sha256"] and archive["payload_sha256"]
    restored = runner.read_archive(path)
    assert restored["partial_history"]["y"] == [1., {"__nonfinite_float__": "nan"},
                                                 {"__nonfinite_float__": "inf"}]
    with pytest.raises(FileExistsError):
        runner.archive_new(path, {"replacement": True})
    assert path.read_bytes() == saved


@pytest.mark.skipif(os.name != "posix", reason="route process-group budget is POSIX-only")
def test_supervisor_budget_preserves_latest_checkpoint_and_never_reuses_route_directory(
        tmp_path, monkeypatch):
    directory = tmp_path / "route"
    calls = []

    class Process:
        returncode = None
        pid = 987654321

        def __init__(self, *args, **kwargs):
            calls.append(self)
            assert kwargs["start_new_session"] is True
            runner.archive_new(directory / "progress-0000.json.gz",
                               {"segments": [], "boundary_t0": {"source_y": [1, 2]},
                                "phase": "before-source-integration"})

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def kill_group(pid, sig):
        assert pid == Process.pid
        assert sig in (signal.SIGTERM, signal.SIGKILL)
        calls[-1].returncode = -int(sig)

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner.os, "killpg", kill_group)
    events = []
    trace, artifact = runner.execute_route(routes()[0], directory, 1e-9, progress=events.append)
    assert artifact["timed_out"] and trace["exception"]["type"] == "RouteTimeout"
    assert trace["partial_history"]["boundary_t0"]["source_y"] == [1, 2]
    assert (directory / "supervisor-failure.json.gz").exists()
    with pytest.raises(FileExistsError):
        runner.execute_route(routes()[0], directory, 1e-9, progress=events.append)
    assert len(calls) == 1


@pytest.mark.skipif(os.name != "posix",
                    reason="Runner explicitly supports POSIX process groups only")
def test_real_timeout_terminates_grandchild_that_ignores_term(tmp_path, monkeypatch):
    directory = tmp_path / "route"
    identities, ready = tmp_path / "pids.json", tmp_path / "grandchild-ready"
    child_code = (
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(60)"
    )
    parent_code = (
        "import os,subprocess,sys,time,json;from pathlib import Path;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"Path({str(identities)!r}).write_text(json.dumps({{'leader':os.getpid(),"
        "'grandchild':p.pid,'pgid':os.getpgrp()}));time.sleep(60)"
    )
    original_popen = subprocess.Popen
    processes = []

    def stub_worker_command(*args, **kwargs):
        process = original_popen([sys.executable, "-c", parent_code], **kwargs)
        processes.append(process)
        return process

    def running(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                                text=True, check=False).stdout.strip()
        return bool(status) and not status.startswith("Z")

    try:
        with monkeypatch.context() as context:
            context.setattr(runner.subprocess, "Popen", stub_worker_command)
            trace, artifact = runner.execute_route(routes()[0], directory, 1.5,
                                                    progress=lambda event: None)
        pids = json.loads(identities.read_text())
        assert ready.exists(), "Grandchild must have installed its SIGTERM-ignore handler"
        assert pids["pgid"] == pids["leader"] == processes[0].pid
        assert trace["exception"]["type"] == "RouteTimeout" and artifact["timed_out"]
        deadline = time.monotonic() + 3
        while running(pids["grandchild"]) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not running(pids["leader"])
        assert not running(pids["grandchild"])
        assert processes[0].returncode == -signal.SIGTERM
    finally:
        if identities.exists():
            for pid in json.loads(identities.read_text()).values():
                if pid == os.getpid():
                    continue
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


def test_non_posix_budget_is_rejected_before_creating_a_route_directory(tmp_path, monkeypatch):
    directory = tmp_path / "unsupported-route"
    with monkeypatch.context() as context:
        context.setattr(runner.os, "name", "nt")
        with pytest.raises(RuntimeError, match=r"POSIX.*unsupported"):
            runner.execute_route(routes()[0], directory, 1, progress=lambda event: None)
    assert not directory.exists()


def test_worker_preserves_exception_and_partial_prefix(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    route = routes()[0]
    runner.write_new(request, {"route": route, "case": case_data(route["unit"])})

    def failure(*args, **kwargs):
        exc = RuntimeError("fixture scientific call failure")
        exc.partial_history = {"segments": [{"raw_y": [1, 2]}]}
        raise exc

    monkeypatch.setattr(runner, "native_time", failure)
    assert runner._worker(request, tmp_path) == 1
    saved = runner.read_archive(tmp_path / "final.json.gz")
    assert saved["exception"]["message"] == "fixture scientific call failure"
    assert saved["partial_history"]["segments"][0]["raw_y"] == [1, 2]


@pytest.fixture
def initial_event_trace(fixture_histories):
    route = next(r for r in routes() if runner.route_id(r) == "P04/source/coarse")
    original = fixture_histories[runner.route_id(route)]
    boundary = deepcopy(original["segments"][0]["records"][-1])
    boundary["row"][:2] = [4.3, -1.]
    boundary.update(differential_state_unchanged_by_boundary_root=True,
                    boundary_operation="source algebraic root only; no integration")
    boundary["source_events"]["Maximum voltage [V]"] = [-.1]
    trace = {
        "exception": {"type": "SolverError", "message":
                      "Events ['Maximum voltage [V]'] are non-positive at initial conditions"},
        "partial_history": {"contract": deepcopy(original["contract"]),
                            "segments": [deepcopy(original["segments"][0])],
                            "failed_stage": 1, "failed_request": case_data("P04")["stages"][1],
                            "failed_boundary_t0": boundary, "last_completed_time_s": 1800.}}
    return route, trace


def test_complete_original_t0_proof_classifies_without_counting_a_raw_integration(
        initial_event_trace):
    route, trace = initial_event_trace
    report, views = runner._route_assessment(route, trace)
    assert report["passed"] and report["physical_initial_source_stop"]
    assert report["raw_completed_stages"] == 1
    assert report["available_physical_stages"] == len(views) == 2
    counts = runner.raw_saved_counts([trace])
    assert counts["saved_raw_stages"] == 1
    assert counts["saved_raw_observations"] == 31


@pytest.mark.parametrize("mutation", (
    lambda t: t.__setitem__("exception", "malformed exception"),
    lambda t: t["exception"].__setitem__("type", "RuntimeError"),
    lambda t: t["exception"].__setitem__("message", "initial conditions algebraic root failure"),
    lambda t: t["partial_history"].pop("failed_stage"),
    lambda t: t["partial_history"].__setitem__("failed_stage", True),
    lambda t: t["partial_history"].pop("failed_request"),
    lambda t: t["partial_history"].__setitem__("failed_request", "malformed request"),
    lambda t: t["partial_history"].__setitem__("failed_boundary_t0", None),
    lambda t: t["partial_history"]["failed_boundary_t0"].pop("source_y"),
    lambda t: t["partial_history"]["failed_boundary_t0"].__setitem__("source_rhs", [np.nan]),
    lambda t: t["partial_history"]["failed_boundary_t0"].pop("source_algebraic"),
    lambda t: t["partial_history"]["failed_boundary_t0"].__setitem__("source_algebraic", [np.inf]),
    lambda t: t["partial_history"]["failed_boundary_t0"].__setitem__(
        "differential_state_unchanged_by_boundary_root", False),
    lambda t: t["partial_history"]["failed_boundary_t0"].__setitem__("row", [0.]),
    lambda t: t["partial_history"]["failed_boundary_t0"]["profiles"].pop("positive"),
    lambda t: t["partial_history"]["failed_boundary_t0"]["source_events"].pop(
        "Maximum voltage [V]"),
    lambda t: t["partial_history"]["failed_boundary_t0"]["source_events"].__setitem__(
        "Maximum voltage [V]", [1.]),
    lambda t: t["partial_history"]["failed_boundary_t0"]["source_events"].__setitem__(
        "Maximum voltage [V]", [np.nan]),
    lambda t: t["partial_history"]["segments"][-1].__setitem__("records", []),
))
def test_missing_or_numeric_t0_proof_fails_closed_with_full_denominator(
        initial_event_trace, mutation):
    route, trace = initial_event_trace
    mutation(trace)
    assert runner._initial_source_stop(trace, trace["partial_history"]) is None
    report, _ = runner._route_assessment(route, trace)
    assert not report["passed"] and not report["physical_initial_source_stop"]
    assert report["raw_completed_stages"] == 1
    assessment = runner.assess_histories({runner.route_id(route): trace})
    assert assessment["evaluated_checks"] == assessment["check_denominator"] == 2544
    assert not assessment["all_passed"]


def test_ordinary_source_retention_also_requires_algebraic_and_unchanged_boundary(
        fixture_histories):
    route = next(r for r in routes() if runner.route_id(r) == "P02/source/coarse")
    for boundary, field, value in ((False, "source_algebraic", []),
                                    (True, "differential_state_unchanged_by_boundary_root", False)):
        trace = deepcopy(fixture_histories[runner.route_id(route)])
        record = trace["segments"][0]["boundary_t0"] if boundary \
            else trace["segments"][0]["records"][0]
        record[field] = value
        report, _ = runner._route_assessment(route, trace)
        assert not report["passed"]
        assert any(c["name"].endswith("raw-state-retention") and not c["passed"]
                   for c in report["checks"])


@pytest.mark.skipif(os.name != "posix", reason="route execution is POSIX-only")
def test_actual_source_exception_prefix_is_counted_in_route_and_final_manifest(
        tmp_path, monkeypatch):
    import battery_parameters45_pybamm as source

    data = case_data("P02")
    data["purpose"] = "actual 0.01s exception-prefix tool fixture, not qualification"
    data["stages"] = tuple(dict(data["stages"][0], duration_s=.01, samples=2) for _ in range(2))
    original, calls = source.initial_state_record, []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected failure after one actual source stage")
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "initial_state_record", fail_second)
    with pytest.raises(RuntimeError, match="one actual source stage") as caught:
        source.reference_time(data, 4)
    trace = {"exception": {"type": type(caught.value).__name__, "message": str(caught.value)},
             "partial_history": caught.value.partial_history}
    assert runner.raw_saved_counts([trace]) == {"saved_raw_stages": 1, "saved_raw_observations": 2}
    route = next(r for r in routes() if runner.route_id(r) == "P02/source/coarse")
    assert runner._route_assessment(route, trace)[0]["raw_completed_stages"] == 1
    note, output = tmp_path / "fixture-note.txt", tmp_path / "fixture-output"
    note.write_text("Artificial manifest contract only; no fixed-route execution.\n")
    monkeypatch.setattr(runner, "_provenance", lambda protocol: {"fixture": True})

    def fake_route(*args, **kwargs):
        return trace, {"route": route, "exception": trace["exception"],
                       "timed_out": False, "elapsed_s": 0.}

    monkeypatch.setattr(runner, "execute_route", fake_route)
    assert runner.main(["--execute", "--route", runner.route_id(route), "--protocol", str(note),
                        "--output-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["saved_raw_stages"] == 1 and manifest["saved_raw_observations"] == 2
    assert manifest["returned_histories"] == 0 and manifest["exceptions"] == 1
    assert manifest["check_denominator"] == manifest["evaluated_checks"] == 2544


def test_plan_cli_does_not_create_outputs_or_call_any_science(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("no prospective route may execute")

    monkeypatch.setattr(runner, "execute_route", forbidden)
    output = tmp_path / "must-not-exist"
    assert runner.main(["--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["check_denominator"] == 2544
    assert not output.exists()


def test_provenance_reads_the_actual_pinned_endpoint_control_and_energy_sources(tmp_path):
    note = tmp_path / "fixture-note.txt"
    note.write_text("Tool-contract note only, never an experiment authorization.\n")
    provenance = runner._provenance(note)
    names = {Path(name).name for name in provenance["files_sha256"]}
    assert {"lithium_ion_parameters.py", "function_control_external_circuit.py",
            "discharge_throughput.py", "Chen2020.py"} <= names
    assert provenance["versions"]["pybamm"] == "25.4.2"
    assert provenance["versions"]["casadi"] == "3.6.7"


def test_real_source_worker_cli_archives_short_original_states_and_checkpoints(tmp_path):
    route = next(r for r in routes() if r["unit"] == "P02" and r["implementation"] == "source")
    route = dict(route, radial_cells=4)
    data = case_data("P02")
    data["purpose"] = "0.01 s source worker tool contract, not qualification"
    data["stages"] = (dict(data["stages"][0], duration_s=.01, samples=2),)
    request = tmp_path / "request.json"
    runner.write_new(request, {"route": route, "case": data})
    result = subprocess.run([sys.executable, "-u", runner.__file__, "--worker-request",
                             str(request),
                             "--worker-directory", str(tmp_path)], capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    trace = runner.read_archive(tmp_path / "final.json.gz")
    assert trace["segments"][0]["actual_end_time_s"] == .01
    assert len(trace["segments"][0]["records"]) == 2
    assert trace["segments"][0]["records"][0]["row"][2:4] == [0, 0]
    assert len(trace["segments"][0]["original_all_ys"][0]) == 13
    checkpoints = [runner.read_archive(p) for p in sorted(tmp_path.glob("progress-*.json.gz"))]
    assert {p["phase"] for p in checkpoints} == {
        "before-source-integration", "source-solution-returned", "source-stage-captured"}


def test_short_native_observer_keeps_original_full_residual_y_and_receipts():
    data = case_data("P06")
    data["purpose"] = "three 0.01 s native tool contracts, not fixed qualification units"
    data["stages"] = tuple(dict(stage, duration_s=.01, samples=2)
                           for stage in data["stages"])
    checkpoints = []
    trace = runner.native_time(data, 4, progress_callback=lambda p: checkpoints.append(deepcopy(p)))
    assert len(trace["segments"]) == 3
    assert trace["contract"]["binding_validated_against_actual_spec"]
    for stage in trace["segments"]:
        reports = stage["original_residual_reports"]
        assert len(reports) == 1
        report = reports[0]
        assert len(report["variables"]) == 4 * 2 + 5 + 2 + 2
        assert {report["variables"][i]["name"] for i in report["algebraic_indices"]} \
            == {"battery:current_A", "battery:voltage_V"}
        assert all(len(point["y"]) == len(point["ydot"]) == len(report["variables"])
                   for point in report["points"])
        assert stage["records"][-1]["native_state"]["coordinates"]["discharged_charge_C"] \
            == stage["records"][-1]["row"][2]
    assert trace["segments"][-1]["records"][-1]["row"][0] == pytest.approx(4.05, abs=1e-7)
    assert any(p["phase"] == "native-residual-report-returned" for p in checkpoints)
    assert tuple(f"P{i:02d}" for i in range(1, 7)) == CASE_IDS
