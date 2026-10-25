"""Run only against a genuinely installed distribution, outside the checkout."""

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-policy", choices=("linear", "ri5_fractional", "ri5_truncated"), default="linear"
    )
    args = parser.parse_args()
    target = args.installed.resolve()
    sys.path.insert(0, str(target))
    import chemworld
    from chemworld import physchem
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.verify import verify_records
    from chemworld.foundation.discrete_inventory import AVOGADRO
    from chemworld.physchem import SDEControl, StochasticSDESpec
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.biochemical_worlds import biochemical_network
    from chemworld.world.stochastic_sde_world import stochastic_sde_world

    assert Path(chemworld.__file__).resolve().is_relative_to(target)
    assert importlib.util.find_spec("COPASI") is None
    public_names = (
        "RI5Draw",
        "RI5Step",
        "draw_ri5",
        "ri5_step",
        "SDEParameter",
        "ReactionFlux",
        "ReactionNoise",
        "SDEControl",
        "StochasticSDESpec",
        "ReactionSDE",
        "initial_sde_continuation",
        "advance_sde",
        "rebind_sde_continuation",
    )
    for name in public_names:
        assert getattr(physchem, name) is not None
    suffix = {"operation": "wait", "duration_s": 0.375}
    if args.resume:
        session = ChemWorldRuntime.from_snapshot(
            json.loads(args.resume.read_text(encoding="utf-8"))
        )
        try:
            assert (
                session.compiled_composition.spec.physics.models_of("stochastic_sde")[
                    session._state.vessel_id
                ].output_policy
                == args.output_policy
            )
            result = session.execute(suffix)
            assert result.info["transaction_status"] == "committed"
            args.output.write_text(json.dumps(session._state.to_dict()), encoding="utf-8")
        finally:
            session.close()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    actions = [
        {"operation": "wait", "duration_s": 0.0625},
        {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}},
        {"operation": "measure", "instrument": "sde_aliquot_meter"},
        {"operation": "wait", "duration_s": 0.03125},
        {"operation": "measure", "instrument": "sde_amount_meter"},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "set_sde_parameters", "values": {"forward_scale": 0}},
        {"operation": "measure", "instrument": "sde_aliquot_meter"},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "terminate"},
    ]
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = StochasticSDESpec(
        volume_L=1e-15,
        step_s=0.125,
        minimum_volume_L=0.75e-15,
        output_policy=args.output_policy,
        controls=(
            SDEControl("forward_scale", "forward", "rate_multiplier", 0, 4),
            SDEControl("reverse_scale", "reverse", "rate_multiplier", 0, 4),
        ),
    )
    world = stochastic_sde_world(
        network,
        spec,
        {"A": 100.5 / AVOGADRO, "B": 20.25 / AVOGADRO},
        seed=123,
    )
    units = []
    for entry, cls in (("native", ChemWorldRuntime), ("gym", ChemWorldEnv)):
        for noise in ("sequential", "keyed"):
            name = entry + "-" + noise
            session = cls(composition=world, observation_noise_mode=noise)
            path = args.output / (name + ".jsonl")
            session.reset(seed=7)
            try:
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
                        logger.log(
                            task_info={**session.task_info(), **session.evaluator_provenance()},
                            step=index,
                            action=action,
                            observation=observation_to_json(observation),
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            info=info,
                            agent_metadata={"name": "installed-sde51e"},
                        )
                        if index == 4:
                            payload = json.loads(json.dumps(session.snapshot().to_dict()))
                            checkpoint = args.output / (name + "-checkpoint.json")
                            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
                            restored = ChemWorldRuntime.from_snapshot(payload)
                            try:
                                result = restored.execute(suffix)
                                assert result.info["transaction_status"] == "committed"
                                expected = json.loads(json.dumps(restored._state.to_dict()))
                            finally:
                                restored.close()
                            fresh_path = args.output / (name + "-fresh-state.json")
                            process = subprocess.run(
                                [
                                    sys.executable,
                                    "-I",
                                    "-X",
                                    "utf8",
                                    str(Path(__file__).resolve()),
                                    "--installed",
                                    str(target),
                                    "--output-policy",
                                    args.output_policy,
                                    "--resume",
                                    str(checkpoint),
                                    "--output",
                                    str(fresh_path),
                                ],
                                cwd=args.output,
                                capture_output=True,
                                text=True,
                                timeout=120,
                            )
                            (args.output / (name + "-fresh.stdout")).write_text(
                                process.stdout, encoding="utf-8"
                            )
                            (args.output / (name + "-fresh.stderr")).write_text(
                                process.stderr, encoding="utf-8"
                            )
                            assert process.returncode == 0, process.stdout + process.stderr
                            assert json.loads(fresh_path.read_text(encoding="utf-8")) == expected
                            # This host-side fork does not advance the source session.
                records = load_jsonl(path)
                verified = verify_records(records, tolerance=0)
                assert verified.verified, verified.mismatches
                assert verified.checked_steps == 10
                assert len(records[5]["initialization_replay"]) == 1
                assert "sample_amount_A" in path.read_text(encoding="utf-8")
                units.append(
                    {
                        "name": name,
                        "recorded_steps": 10,
                        "replayed_steps": 10,
                        "json_restore": True,
                        "new_process_exact": True,
                    }
                )
                print(json.dumps(units[-1]), flush=True)
            finally:
                session.close()
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name.startswith("chemworld") and getattr(module, "__file__", None)
    }
    assert all(Path(origin).is_relative_to(target) for origin in origins.values())
    result = {
        "passed": len(units) == 4,
        "output_policy": args.output_policy,
        "protocols": units,
        "public_exports": public_names,
        "module_origins": origins,
        "sdk_importable": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
