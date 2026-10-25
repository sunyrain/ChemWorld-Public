"""Eight fixed controlled RK45 protocols against the actual installed distribution."""

import argparse
import importlib.util
import json
import subprocess
import sys
import traceback
from itertools import product
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--integer", action="store_true")
    args = parser.parse_args()
    target = args.installed.resolve()
    sys.path.insert(0, str(target))
    import chemworld
    from chemworld import physchem
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.verify import verify_records
    from chemworld.physchem import stochastic_kinetics
    from chemworld.physchem.stochastic_hybrid import HybridAdvance
    from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.composition import WorldCompositionSpec

    assert Path(chemworld.__file__).resolve().is_relative_to(target)
    assert importlib.util.find_spec("COPASI") is None
    public = (
        "StochasticHybridSpec",
        "CrossPartitionHybridSpec",
        "initial_hybrid_continuation",
        "advance_stochastic_hybrid",
        "initial_cross_continuation",
        "advance_cross_partition",
    )
    for name in public:
        assert getattr(physchem, name) is not None
    assert HybridAdvance.__module__ == "chemworld.physchem.stochastic_hybrid"
    public = (
        *("chemworld.physchem." + name for name in public),
        "chemworld.physchem.stochastic_hybrid.HybridAdvance",
    )
    if args.integer:
        public = (
            *(
                "chemworld.physchem.stochastic_kinetics." + name
                for name in (
                    "EventChannel",
                    "StochasticSpec",
                    "advance_stochastic",
                    "initial_continuation",
                )
            ),
            "chemworld.physchem.stochastic_tau_leap.TauLeapPolicy",
        )
        assert all(
            getattr(stochastic_kinetics, name) is not None
            for name in (
                "EventChannel",
                "StochasticSpec",
                "advance_stochastic",
                "initial_continuation",
            )
        )
        assert TauLeapPolicy.__module__ == "chemworld.physchem.stochastic_tau_leap"
    suffix = {"operation": "wait", "duration_s": 1.0 if args.integer else 2.0}
    if args.resume:
        session = ChemWorldRuntime.from_snapshot(
            json.loads(args.resume.read_text(encoding="utf-8"))
        )
        try:
            result = session.execute(suffix)
            assert result.info["transaction_status"] == "committed", result.info
            args.output.write_text(
                json.dumps({"state": session._state.to_dict(), "info": result.info}),
                encoding="utf-8",
            )
        finally:
            session.close()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    assert args.fixtures is not None
    kinds = ("direct", "next", "tau") if args.integer else ("feedback", "cross")
    fixtures = {
        kind: json.loads((args.fixtures / (kind + ".json")).read_text(encoding="utf-8"))
        for kind in kinds
    }
    actions = [
        {"operation": "wait", "duration_s": 1.0},
        {"operation": "wait", "duration_s": 2.0},
        {"operation": "wait", "duration_s": 1.0},
        {"operation": "terminate"},
    ]
    if args.integer:
        actions = [
            {"operation": "wait", "duration_s": 1.0},
            {"operation": "measure", "instrument": "molecule_counter"},
            {
                "operation": "add_component",
                "component": 1,
                "amount_mol": 10 / stochastic_kinetics.AVOGADRO,
            },
            {"operation": "wait", "duration_s": 2.0},
            {"operation": "measure", "instrument": "molecule_capture"},
            suffix,
            {"operation": "measure", "instrument": "molecule_counter"},
            {"operation": "terminate"},
        ]
    units = []
    for kind, (entry, cls), noise in product(
        kinds,
        (("native", ChemWorldRuntime), ("gym", ChemWorldEnv)),
        ("sequential", "keyed"),
    ):
        name = kind + "-" + entry + "-" + noise
        row = {"name": name, "recorded_steps": 0, "replayed_steps": 0, "passed": False}
        session = None
        try:
            world = WorldCompositionSpec.from_dict(fixtures[kind])
            if args.integer:
                algorithms = {
                    "direct": "gillespie-direct-v1",
                    "next": "gibson-bruck-next-reaction-v1",
                    "tau": "adaptive-explicit-tau-leap-v1",
                }
                assert world.physics.network.stochastic["algorithm"] == algorithms[kind]
                channel = world.physics.network.stochastic_channels[0]
                assert dict(channel.reactants) == {"A": 1, "B": 1}
                assert (
                    abs(channel.propensity({"A": 39, "B": 261, "C": 0}) - 0.002 * 39 * 261) < 1e-10
                )
            else:
                assert all(
                    spec.ode_method == "RK45"
                    for spec in world.physics.models_of("stochastic_hybrid").values()
                )
            session = cls(composition=world, observation_noise_mode=noise)
            session.reset(seed=7)
            path = args.output / (name + ".jsonl")
            with TrajectoryLogger(path) as logger:
                for index, action in enumerate(actions, 1):
                    if args.integer and index in (2, 7):
                        old_rng = json.loads(
                            json.dumps(session._state.process.stochastic_continuation["rng_state"])
                        )
                    if entry == "native":
                        result = session.execute(action)
                        observation = to_observation(result.observation.values)
                        reward, terminated, truncated, info = (
                            result.reward,
                            result.terminated,
                            result.truncated,
                            result.info,
                        )
                    else:
                        observation, reward, terminated, truncated, info = session.step(action)
                    assert info["transaction_status"] == "committed", info
                    if args.integer:
                        assert "rng_state" not in json.dumps(info)
                        if index in (2, 7):
                            assert (
                                session._state.process.stochastic_continuation["rng_state"]
                                == old_rng
                            )
                        stochastic_kinetics.mol_to_counts(session._state.species_amounts)
                    logger.log(
                        task_info={**session.task_info(), **session.evaluator_provenance()},
                        step=index,
                        action=action,
                        observation=observation_to_json(observation),
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        agent_metadata={"name": "installed-hybrid51g"},
                    )
                    row["recorded_steps"] = index
                    if index == (5 if args.integer else 1):
                        payload = json.loads(json.dumps(session.snapshot().to_dict()))
                        checkpoint = args.output / (name + "-checkpoint.json")
                        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
                        restored = ChemWorldRuntime.from_snapshot(payload)
                        try:
                            result = restored.execute(suffix)
                            assert result.info["transaction_status"] == "committed", result.info
                            expected = json.loads(
                                json.dumps(
                                    {"state": restored._state.to_dict(), "info": result.info}
                                )
                            )
                        finally:
                            restored.close()
                        fresh = args.output / (name + "-fresh-state.json")
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
                                *(["--integer"] if args.integer else []),
                            ],
                            cwd=args.output,
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        for label, value in (("stdout", child.stdout), ("stderr", child.stderr)):
                            (args.output / (name + "-fresh." + label)).write_text(
                                value, encoding="utf-8"
                            )
                        assert child.returncode == 0, child.stdout + child.stderr
                        assert json.loads(fresh.read_text(encoding="utf-8")) == expected
                        (args.output / (name + "-expected-state.json")).write_text(
                            json.dumps(expected), encoding="utf-8"
                        )
                        row.update(json_restore=True, new_process_exact=True)
                    if args.integer and index == 6:
                        assert (
                            json.loads(
                                json.dumps({"state": session._state.to_dict(), "info": info})
                            )
                            == expected
                        )
                        row["events_committed"] = session._state.process.stochastic_continuation[
                            "events_committed"
                        ]
                        assert row["events_committed"] > 0
            records = load_jsonl(path)
            replay = verify_records(records, tolerance=0)
            assert replay.verified, replay.mismatches
            assert replay.checked_steps == len(actions)
            row.update(replayed_steps=len(actions), passed=True)
        except Exception as exc:
            row.update(error=repr(exc), traceback=traceback.format_exc())
        finally:
            if session is not None:
                session.close()
        units.append(row)
        print(json.dumps({"completed": len(units), "total": len(kinds) * 4, **row}), flush=True)
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name.startswith("chemworld") and getattr(module, "__file__", None)
    }
    assert all(Path(origin).is_relative_to(target) for origin in origins.values())
    result = {
        "passed": len(units) == len(kinds) * 4 and all(row["passed"] for row in units),
        "protocols": units,
        "public_exports": public,
        "module_origins": origins,
        "sdk_importable": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
