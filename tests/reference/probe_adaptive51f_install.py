"""Eight fixed protocols against an actual installed adaptive world distribution."""

import argparse
import importlib.util
import json
import subprocess
import sys
import traceback
from dataclasses import replace
from itertools import product
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--autocatalytic", action="store_true")
    args = parser.parse_args()
    target = args.installed.resolve()
    sys.path.insert(0, str(target))
    import chemworld
    from chemworld import physchem
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.verify import verify_records
    from chemworld.physchem import AdaptiveHybridSpec
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.biochemical_worlds import biochemical_network
    from chemworld.world.stochastic_adaptive_world import stochastic_adaptive_world

    assert Path(chemworld.__file__).resolve().is_relative_to(target)
    assert importlib.util.find_spec("COPASI") is None
    public = (
        "AdaptiveHybridSpec",
        "AdaptiveHybridModel",
        "AdaptivePartition",
        "AdaptiveHybridAdvance",
        "AdaptiveHybridRebind",
        "initial_adaptive_continuation",
        "advance_adaptive_hybrid",
        "rebind_adaptive_continuation",
        "RK4Step",
        "rk4_step",
    )
    for name in public:
        assert getattr(physchem, name) is not None
    suffix = {"operation": "wait", "duration_s": 0.625}
    if args.resume:
        session = ChemWorldRuntime.from_snapshot(
            json.loads(args.resume.read_text(encoding="utf-8"))
        )
        try:
            result = session.execute(suffix)
            assert result.info["transaction_status"] == "committed", result.info
            args.output.write_text(json.dumps(session._state.to_dict()), encoding="utf-8")
        finally:
            session.close()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    quantum = 1 / 6.02214076e23
    actions = [
        {"operation": "wait", "duration_s": 0.375},
        {"operation": "add_component", "component": 0, "amount_mol": 0.5 * quantum},
        {"operation": "measure", "instrument": "sde_amount_meter"},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "wait", "duration_s": 0.25},
        {"operation": "measure", "instrument": "sde_amount_meter"},
        {"operation": "add_component", "component": 0, "amount_mol": 0.5 * quantum},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "measure", "instrument": "sde_amount_meter"},
        {"operation": "wait", "duration_s": 0.125},
    ]
    original = biochemical_network("reversible")
    network = replace(
        original,
        stochastic=None,
        reactions=tuple(
            replace(reaction, rate_law=replace(reaction.rate_law, parameters={"k": k}))
            for reaction, k in zip(original.reactions, (1.0, 0.0), strict=True)
        ),
    )
    if args.autocatalytic:
        from chemworld.physchem.reaction_network import (
            RateLawSpec,
            ReactionNetworkSpec,
            ReactionSpec,
            SpeciesSpec,
        )

        network = ReactionNetworkSpec(
            "installed-autocatalytic",
            (SpeciesSpec("A", "CH2O"), SpeciesSpec("B", "CH2O")),
            (
                ReactionSpec.from_equation(
                    reaction_id="autocatalytic",
                    equation="A + B => 2 A",
                    forward_orders={"A": 1, "B": 1},
                    rate_law=RateLawSpec(
                        "autocatalytic", "mass_action", {"k": 0.002 * (6.02214076e23 * 1e-15)}
                    ),
                ),
            ),
        )
    units = []
    for method, (entry, cls), noise in product(
        ("RK4", "LSODA"),
        (("native", ChemWorldRuntime), ("gym", ChemWorldEnv)),
        ("sequential", "keyed"),
    ):
        name = method + "-" + entry + "-" + noise
        row = {"name": name, "recorded_steps": 0, "replayed_steps": 0, "passed": False}
        session = None
        try:
            spec = AdaptiveHybridSpec(
                volume_L=1e-15,
                method=method,
                step_s=0.125 if args.autocatalytic else 0.25,
                lower_limit=39 if args.autocatalytic else 45,
                upper_limit=40 if args.autocatalytic else 45,
            )
            world = stochastic_adaptive_world(
                network,
                spec,
                {"A": 39.25 * quantum, "B": 260.75 * quantum}
                if args.autocatalytic
                else {"A": 50.25 * quantum, "B": 50.75 * quantum},
                seed=17,
            )
            session = cls(composition=world, observation_noise_mode=noise)
            session.reset(seed=7)
            path = args.output / (name + ".jsonl")
            rhs_evaluations = 0
            with TrajectoryLogger(path) as logger:
                for index, action in enumerate(actions, 1):
                    if index == 6:
                        session.reset(options={"scope": "material"})
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
                    rhs_evaluations += (
                        info.get("outcome_diagnostics", {})
                        .get("stochastic_work", {})
                        .get("drift_evaluations", 0)
                    )
                    logger.log(
                        task_info={**session.task_info(), **session.evaluator_provenance()},
                        step=index,
                        action=action,
                        observation=observation_to_json(observation),
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        agent_metadata={"name": "installed-adaptive51f"},
                    )
                    row["recorded_steps"] = index
                    if index == 3:
                        payload = json.loads(json.dumps(session.snapshot().to_dict()))
                        checkpoint = args.output / (name + "-checkpoint.json")
                        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
                        restored = ChemWorldRuntime.from_snapshot(payload)
                        try:
                            result = restored.execute(suffix)
                            assert result.info["transaction_status"] == "committed", result.info
                            expected = json.loads(json.dumps(restored._state.to_dict()))
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
            records = load_jsonl(path)
            replay = verify_records(records, tolerance=0)
            assert replay.verified, replay.mismatches
            assert replay.checked_steps == 10
            assert len(records[5]["initialization_replay"]) == 1
            if args.autocatalytic:
                events = session._state.process.stochastic_continuation["events_committed"]
                assert events > 0 and rhs_evaluations > 0
                row.update(events_committed=events, rhs_evaluations=rhs_evaluations)
            row.update(replayed_steps=10, passed=True)
        except Exception as exc:
            row.update(error=repr(exc), traceback=traceback.format_exc())
        finally:
            if session is not None:
                session.close()
        units.append(row)
        print(json.dumps({"completed": len(units), "total": 8, **row}), flush=True)
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name.startswith("chemworld") and getattr(module, "__file__", None)
    }
    assert all(Path(origin).is_relative_to(target) for origin in origins.values())
    result = {
        "passed": len(units) == 8 and all(row["passed"] for row in units),
        "protocols": units,
        "public_exports": public,
        "module_origins": origins,
        "sdk_importable": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
