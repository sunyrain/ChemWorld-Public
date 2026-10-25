"""Installed-only whole experiment, actual-operation replay and fresh-process suffix."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from itertools import product
from pathlib import Path


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--integer-events", action="store_true")
    parser.add_argument("--resume-native", type=Path)
    args = parser.parse_args()
    installed = args.installed.resolve()
    sys.path.insert(0, str(installed))
    import chemworld
    from chemworld.data.combine_archive import CombineArchive
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.sedml_continuation import SEDMLExperimentSession
    from chemworld.eval.verify import verify_records
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.sbml_models import model_record
    from chemworld.world.sedml_protocol import SEDMLProtocol

    assert Path(chemworld.__file__).resolve().is_relative_to(installed)
    assert importlib.util.find_spec("COPASI") is None

    def origins():
        data = {
            name: str(Path(module.__file__).resolve())
            for name, module in sys.modules.items()
            if name.startswith("chemworld") and getattr(module, "__file__", None)
        }
        assert data and all(Path(path).is_relative_to(installed) for path in data.values())
        return data

    def completion(session):
        session.advance()
        assert session.complete
        return plain({"checkpoint": session.to_dict(), "result": asdict(session.result())})

    def native_suffix(snapshot, logger=None, start_step=0, live_runtime=None):
        runtime = live_runtime or ChemWorldRuntime.from_snapshot(snapshot)
        try:
            assert plain(runtime.snapshot().to_dict()) == snapshot
            rows = []
            for index, action in enumerate((
                {"operation": "wait", "duration_s": 2.0},
                {"operation": "measure", "instrument": "formal_quantity_meter"},
            ), start_step + 1):
                before = plain(model_record(runtime._state).settings)
                clock = runtime._state.ledger.time_s
                result = original_execute(runtime, action)
                assert result.info["transaction_status"] == "committed", result.info
                after = plain(model_record(runtime._state).settings)
                if action["operation"] == "measure":
                    assert before == after and clock == runtime._state.ledger.time_s
                assert after["continuation"]["format"] == "chemworld_sbml_integer_events_v1"
                assert after["continuation"]["event_queue"]["pending"] == []
                assert after["coordinates"]["P:memory"] > 0
                rows.append(plain({"state": runtime._state.to_dict(), "info": result.info,
                                   "observation": result.observation.to_dict(),
                                   "reward": result.reward, "terminated": result.terminated,
                                   "truncated": result.truncated}))
                if logger is not None:
                    logger.log(
                        task_info={**runtime.task_info(), **runtime.evaluator_provenance()},
                        step=index, action=action,
                        observation=observation_to_json(to_observation(result.observation.values)),
                        reward=result.reward, terminated=result.terminated,
                        truncated=result.truncated, info=result.info,
                        agent_metadata={"name": "installed-sbml51k-native-suffix"},
                    )
            return {"rows": rows}
        finally:
            runtime.close()

    original_execute = ChemWorldRuntime.execute
    if args.resume_native:
        data = native_suffix(json.loads(args.resume_native.read_text(encoding="utf-8")))
        data["module_origins"] = origins()
        args.output.write_text(json.dumps(data), encoding="utf-8")
        return
    if args.resume:
        session = SEDMLExperimentSession.from_dict(
            json.loads(args.resume.read_text(encoding="utf-8"))
        )
        data = completion(session)
        data["module_origins"] = origins()
        args.output.write_text(json.dumps(data), encoding="utf-8")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    assert args.fixtures is not None
    rows, started = [], time.monotonic()
    for method, reset, noise in product(
        ("direct", "next", "coarse"), (True, False), ("sequential", "keyed")
    ):
        identity = method + "-" + str(reset).lower() + "-" + noise
        directory = args.output / identity
        directory.mkdir()
        row, captured = {"id": identity, "passed": False}, {}
        try:
            fixture = args.fixtures / (method + "-" + str(reset).lower())
            if reset:
                provenance = InputSource(
                    DataSourceProvenance(
                        "sedml51j-installed",
                        0,
                        "Actual installed S02 standard archive",
                        "1",
                        "CC0-1.0",
                    ),
                    "synthetic",
                )
                archive = CombineArchive.from_bytes(
                    fixture.with_suffix(".omex").read_bytes(), source=provenance
                )
                plan = SEDMLProtocol.from_archive(
                    archive, items_per_mole=6.02214076e23, readout_relative_sd=0.02
                )
            else:
                plan = SEDMLProtocol.from_dict(
                    json.loads(fixture.with_suffix(".json").read_text(encoding="utf-8"))
                )

            def capture(runtime, action, *, captured=captured, directory=directory):
                before = plain(model_record(runtime._state).settings)
                clock = runtime._state.ledger.time_s
                result = original_execute(runtime, action)
                assert result.info["transaction_status"] == "committed"
                after = plain(model_record(runtime._state).settings)
                if action["operation"] == "measure":
                    assert before == after and clock == runtime._state.ledger.time_s
                if action["operation"] == "set_model_parameters":
                    assert clock == runtime._state.ledger.time_s
                    assert (
                        before["continuation"]["process"]["rng_state"]
                        == after["continuation"]["process"]["rng_state"]
                    )
                campaign = result.info["campaign_id"]
                if campaign not in captured:
                    path = directory / (campaign + ".jsonl")
                    captured[campaign] = {
                        "logger": TrajectoryLogger(path),
                        "path": path,
                        "steps": 0,
                        "runtime": runtime,
                    }
                entry = captured[campaign]
                entry["steps"] += 1
                entry["logger"].log(
                    task_info={**runtime.task_info(), **runtime.evaluator_provenance()},
                    step=entry["steps"],
                    action=action,
                    observation=observation_to_json(to_observation(result.observation.values)),
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                    agent_metadata={"name": "installed-sedml51j"},
                )
                return result

            ChemWorldRuntime.execute = capture
            session = SEDMLExperimentSession(plan, seed=37, observation_noise_mode=noise)
            session.advance(max_courses=1)
            checkpoint = plain(session.to_dict())
            if args.integer_events:
                runtime = ChemWorldRuntime.from_snapshot(checkpoint["runner"]["sessions"]["m"])
                try:
                    settings = model_record(runtime._state).settings
                    assert settings["continuation"]["format"] == "chemworld_sbml_integer_events_v1"
                    pending = settings["continuation"]["event_queue"]["pending"]
                    assert len(pending) == 1 and pending[0]["event_id"] == "long_capture"
                    assert pending[0]["due_time_s"] == 4.4
                    spec = runtime.compiled_composition.spec.physics.models_of("sbml")[
                        "formal_workspace"]
                    assert spec.integer_kinetics.requires_boundaries
                    assert spec.frame.integer_quantities
                finally:
                    runtime.close()
            prefix = directory / "checkpoint.json"
            prefix.write_text(json.dumps(checkpoint), encoding="utf-8")
            expected = completion(session)
            ChemWorldRuntime.execute = original_execute
            for entry in captured.values():
                entry["logger"].close()
            (directory / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
            assert session.progress()["completed_courses"] == 3
            assert all(
                column["data"]["shape"] == [3, 25] for column in session.result().reports["report"]
            )
            assert completion(SEDMLExperimentSession.from_dict(checkpoint)) == expected
            fresh = directory / "fresh.json"
            child = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-X",
                    "utf8",
                    str(Path(__file__).resolve()),
                    "--installed",
                    str(installed),
                    "--resume",
                    str(prefix),
                    "--output",
                    str(fresh),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=120,
            )
            for label, output in (("stdout", child.stdout), ("stderr", child.stderr)):
                (directory / ("fresh." + label)).write_text(output, encoding="utf-8")
            assert child.returncode == 0, child.stdout + child.stderr
            actual = json.loads(fresh.read_text(encoding="utf-8"))
            assert {k: actual[k] for k in ("checkpoint", "result")} == expected
            replayed, recorded = 0, len(session.result().history)
            for entry in captured.values():
                replay = verify_records(load_jsonl(entry["path"]), tolerance=0)
                assert replay.verified, replay.mismatches
                assert replay.checked_steps == entry["steps"]
                replayed += replay.checked_steps
            assert replayed == recorded
            if args.integer_events:
                snapshot = session.result().snapshots["scan"]["m"]
                final = ChemWorldRuntime.from_snapshot(snapshot)
                try:
                    settings = model_record(final._state).settings
                    pending = settings["continuation"]["event_queue"]["pending"]
                    assert final._state.ledger.time_s == (3 if reset else 9)
                    if reset:
                        assert len(pending) == 1 and pending[0]["event_id"] == "long_capture"
                        assert settings["coordinates"]["P:memory"] == 0
                    else:
                        assert pending == [] and settings["coordinates"]["P:memory"] > 0
                    campaign = final.snapshot().to_dict()["continuation"]["_campaign_id"]
                finally:
                    final.close()
                snapshot_path = directory / "native-checkpoint.json"
                snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
                entry = captured[campaign]
                suffix_path = directory / "native-suffix.jsonl"
                with TrajectoryLogger(suffix_path) as logger:
                    native_expected = native_suffix(
                        snapshot, logger, entry["steps"], live_runtime=entry["runtime"])
                (directory / "native-expected.json").write_text(
                    json.dumps(native_expected), encoding="utf-8")
                assert native_suffix(snapshot) == native_expected
                native_fresh = directory / "native-fresh.json"
                native_child = subprocess.run(
                    [sys.executable, "-I", "-X", "utf8", str(Path(__file__).resolve()),
                     "--installed", str(installed), "--resume-native", str(snapshot_path),
                     "--output", str(native_fresh)],
                    cwd=directory, capture_output=True, text=True, timeout=120,
                )
                for label, output in (("stdout", native_child.stdout),
                                      ("stderr", native_child.stderr)):
                    (directory / ("native-fresh." + label)).write_text(output, encoding="utf-8")
                assert native_child.returncode == 0, native_child.stdout + native_child.stderr
                native_actual = json.loads(native_fresh.read_text(encoding="utf-8"))
                assert {"rows": native_actual["rows"]} == native_expected
                combined = directory / "native-combined.jsonl"
                combined.write_bytes(entry["path"].read_bytes() + suffix_path.read_bytes())
                replay = verify_records(load_jsonl(combined), tolerance=0)
                assert replay.verified, replay.mismatches
                assert replay.checked_steps == entry["steps"] + 2
                row.update(native_suffix_steps=2,
                           native_combined_replayed_steps=replay.checked_steps,
                           native_new_process_exact=True,
                           native_child_loaded_modules=len(native_actual["module_origins"]))
            row.update(
                passed=True,
                courses=3,
                recorded_steps=recorded,
                replayed_steps=replayed,
                native_sessions=len(captured),
                new_process_exact=True,
                child_loaded_modules=len(actual["module_origins"]),
            )
        except Exception as exc:
            row.update(error=repr(exc), traceback=traceback.format_exc())
        finally:
            ChemWorldRuntime.execute = original_execute
            for entry in captured.values():
                entry["logger"].close()
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": len(rows),
                    "total": 12,
                    "elapsed_s": time.monotonic() - started,
                    "rate_per_s": len(rows) / (time.monotonic() - started),
                    "eta_s": (12-len(rows)) * (time.monotonic()-started) / len(rows),
                    **row,
                }
            ),
            flush=True,
        )
    result = {
        "schema": ("chemworld-sbml51k-installed-v1" if args.integer_events
                   else "chemworld-sedml51j-installed-v1"),
        "protocols": rows,
        "passed": len(rows) == 12 and all(r["passed"] for r in rows),
        "module_origins": origins(),
        "elapsed_s": time.monotonic() - started,
        "sdk_importable": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
