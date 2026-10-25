"""Installed-only original SBML world, common verifier and new-process suffixes."""

import argparse
import importlib.util
import json
import subprocess
import sys
import traceback
from itertools import product
from pathlib import Path


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    target = args.installed.resolve()
    sys.path.insert(0, str(target))
    import chemworld
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.verify import verify_records
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.sbml_models import model_record
    from chemworld.world.composition import WorldCompositionSpec

    assert Path(chemworld.__file__).resolve().is_relative_to(target)
    assert importlib.util.find_spec("COPASI") is None

    def origins():
        result = {
            name: str(Path(module.__file__).resolve())
            for name, module in sys.modules.items()
            if name.startswith("chemworld") and getattr(module, "__file__", None)
        }
        assert result and all(Path(p).is_relative_to(target) for p in result.values())
        return result

    suffix = {"operation": "wait", "duration_s": 1.0}
    if args.resume:
        r = ChemWorldRuntime.from_snapshot(json.loads(args.resume.read_text(encoding="utf-8")))
        try:
            result = r.execute(suffix)
            assert result.info["transaction_status"] == "committed", result.info
            args.output.write_text(
                json.dumps(
                    {"state": r._state.to_dict(), "info": result.info, "module_origins": origins()}
                ),
                encoding="utf-8",
            )
        finally:
            r.close()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    assert args.fixtures is not None
    actions = (
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "measure", "instrument": "formal_quantity_meter"},
        {"operation": "set_model_parameters", "values": {"k0": 3e-18}},
        {"operation": "wait", "duration_s": 0.875},
        {"operation": "measure", "instrument": "formal_quantity_meter"},
        suffix,
        {"operation": "measure", "instrument": "formal_quantity_meter"},
        {"operation": "terminate"},
    )
    units = []
    for name, (entry, cls), noise in product(
        ("direct", "next", "tau"),
        (("native", ChemWorldRuntime), ("gym", ChemWorldEnv)),
        ("sequential", "keyed"),
    ):
        identity = name + "-" + entry + "-" + noise
        row = {"id": identity, "recorded_steps": 0, "replayed_steps": 0, "passed": False}
        session = None
        try:
            world = WorldCompositionSpec.from_dict(
                json.loads((args.fixtures / (name + ".json")).read_text(encoding="utf-8"))
            )
            spec = world.physics.models_of("sbml")["formal_workspace"]
            assert spec.numerical_form == "stochastic"
            (channel,) = spec.integer_kinetics.compile(spec.frame.initial_values, 0)
            assert dict(channel.reactants) == {"A": 1, "B": 1}
            assert abs(channel.propensity({"A": 20, "B": 180}) - 0.002 * 20 * 180) < 1e-12
            session = cls(composition=world, observation_noise_mode=noise)
            session.reset(seed=7)
            trajectory = args.output / (identity + ".jsonl")
            with TrajectoryLogger(trajectory) as logger:
                for index, action in enumerate(actions, 1):
                    old = normalized(model_record(session._state).settings)
                    old_time = session._state.ledger.time_s
                    if entry == "native":
                        result = session.execute(action)
                        observation = to_observation(result.observation.values)
                        reward, terminated, truncated, info = (
                            0.0,
                            result.terminated,
                            result.truncated,
                            result.info,
                        )
                    else:
                        observation, reward, terminated, truncated, info = session.step(action)
                    assert info["transaction_status"] == "committed", info
                    assert "rng_state" not in json.dumps(info)
                    current = normalized(model_record(session._state).settings)
                    counts = spec.integer_kinetics.counts(
                        current["coordinates"], session._state.ledger.time_s
                    )
                    assert sum(counts.values()) == 200
                    assert session._state.species_amounts == {} and session._state.volume_L == 0
                    assert (
                        current["continuation"]["process"]["time_s"] == session._state.ledger.time_s
                    )
                    if action["operation"] == "measure":
                        assert current == old and session._state.ledger.time_s == old_time
                    if index == 3:
                        assert session._state.ledger.time_s == old_time
                        assert (
                            current["continuation"]["process"]["rng_state"]
                            == old["continuation"]["process"]["rng_state"]
                        )
                        assert current["continuation"]["process"]["pending"] is None
                    logger.log(
                        task_info={**session.task_info(), **session.evaluator_provenance()},
                        step=index,
                        action=action,
                        observation=observation_to_json(observation),
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        agent_metadata={"name": "installed-sbml51i"},
                    )
                    row["recorded_steps"] = index
                    if index == 5:
                        checkpoint = args.output / (identity + "-checkpoint.json")
                        payload = normalized(session.snapshot().to_dict())
                        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
                        restored = ChemWorldRuntime.from_snapshot(payload)
                        try:
                            result = restored.execute(suffix)
                            assert result.info["transaction_status"] == "committed", result.info
                            expected = normalized(
                                {"state": restored._state.to_dict(), "info": result.info}
                            )
                        finally:
                            restored.close()
                        fresh = args.output / (identity + "-fresh.json")
                        child = subprocess.run(
                            [
                                sys.executable,
                                "-I",
                                "-X",
                                "utf8",
                                str(Path(__file__).resolve()),
                                "--installed",
                                str(target),
                                "--resume",
                                str(checkpoint),
                                "--output",
                                str(fresh),
                            ],
                            cwd=args.output,
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        for label, text in (("stdout", child.stdout), ("stderr", child.stderr)):
                            (args.output / (identity + "-fresh." + label)).write_text(
                                text, encoding="utf-8"
                            )
                        assert child.returncode == 0, child.stdout + child.stderr
                        actual = json.loads(fresh.read_text(encoding="utf-8"))
                        assert {k: actual[k] for k in ("state", "info")} == expected
                        row.update(
                            json_restore=True,
                            new_process_exact=True,
                            child_loaded_modules=len(actual["module_origins"]),
                        )
                    if index == 6:
                        assert (
                            normalized({"state": session._state.to_dict(), "info": info})
                            == expected
                        )
                        assert current["continuation"]["process"]["events_committed"] > 0
            replay = verify_records(load_jsonl(trajectory), tolerance=0)
            assert replay.verified, replay.mismatches
            assert replay.checked_steps == 8
            row.update(passed=True, replayed_steps=replay.checked_steps)
        except Exception as exc:
            row.update(error=repr(exc), traceback=traceback.format_exc())
        finally:
            if session is not None:
                session.close()
        units.append(row)
        print(json.dumps({"completed": len(units), "total": 12, **row}), flush=True)
    result = {
        "passed": len(units) == 12 and all(r["passed"] for r in units),
        "protocols": units,
        "module_origins": origins(),
        "sdk_importable": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
