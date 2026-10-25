"""Actual native/Gym/CLI/Agent calls; recording wrappers never replace physical methods."""

import argparse
import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def trace(session, info):
    from chemworld.data.logging import to_builtin

    return to_builtin(
        {
            "state": session._state.to_dict(),
            "reading": info["native_observation"],
            "queue": info["lab_clock"],
            "outcome": info["operation_outcome"],
            "transaction_status": info["transaction_status"],
            "duration_s": info["actual_duration_s"],
            "balance": info["material_balance"],
            "initializations": session._lifecycle_history,
            "rng_state": session._rng.bit_generator.state,
        }
    )


def run_entry(kind, recipe, noise, output):
    import pytest

    import chemworld.eval.runner as runner_module
    import chemworld.runtime as runtime_module
    from chemworld.agents.base import BaseAgent
    from chemworld.cli import main as cli_main
    from chemworld.data.logging import TrajectoryLogger
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.envs.spaces import to_observation
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.composition import WorldCompositionSpec

    class FixedSequence(BaseAgent):
        name = "fixed-sbml-entry-no-provider"

        def act(self, history):
            return deepcopy(recipe["actions"][len(history)])

    output.mkdir()
    traces = []
    path = output / "trajectory.jsonl"
    original_runtime = ChemWorldRuntime
    with (output / "trace.jsonl").open("x", encoding="utf-8") as stream:

        def record_step(session, info):
            record = trace(session, info)
            traces.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if kind == "native" and len(traces) == recipe["checkpoint"]:
                write(output / "checkpoint.json", session.snapshot().to_dict())
            if len(traces) == len(recipe["actions"]):
                write(output / "final.json", session.snapshot().to_dict())

        def factory(*args, **kwargs):
            session = original_runtime(*args, **kwargs)
            execute = session.execute

            def recorded(action):
                result = execute(action)
                record_step(session, result.info)
                return result

            session.execute = recorded
            return session

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runtime_module, "ChemWorldRuntime", factory)
            patch.setattr(runner_module, "ChemWorldRuntime", factory)
            if kind in ("native", "gym"):
                world = WorldCompositionSpec.from_dict(recipe["composition"])
                session = (
                    factory(
                        composition=world,
                        observation_noise_mode=noise,
                        observation_seed_override=12,
                    )
                    if kind == "native"
                    else ChemWorldEnv(
                        composition=world,
                        observation_noise_mode=noise,
                        observation_seed_override=12,
                    )
                )
                session.reset(seed=7)
                try:
                    with TrajectoryLogger(path) as logger:
                        for index, action in enumerate(recipe["actions"], 1):
                            if kind == "native":
                                result = session.execute(action)
                                observation = to_observation(result.observation.values)
                                reward, ended, truncated, info = (
                                    result.reward,
                                    result.terminated,
                                    result.truncated,
                                    result.info,
                                )
                            else:
                                observation, reward, ended, truncated, info = session.step(action)
                                record_step(session, info)
                            logger.log(
                                task_info={**session.task_info(), **session.evaluator_provenance()},
                                step=index,
                                action=action,
                                observation=observation,
                                reward=reward,
                                terminated=ended,
                                truncated=truncated,
                                info=info,
                                agent_metadata={"agent_id": "fixed-sbml-entry"},
                            )
                finally:
                    session.close()
            elif kind == "cli":
                composition = output / "composition.json"
                write(composition, recipe["composition"])
                files = []
                for index, action in enumerate(recipe["actions"]):
                    action_file = output / f"action-{index}.json"
                    write(action_file, action)
                    files.append(str(action_file))
                with (
                    (output / "stdout.txt").open("x", encoding="utf-8") as stdout,
                    redirect_stdout(stdout),
                ):
                    cli_main(
                        [
                            "render",
                            "--composition",
                            str(composition),
                            "--seed",
                            "7",
                            "--noise-mode",
                            noise,
                            "--observation-seed",
                            "12",
                            "--actions",
                            *files,
                            "--format",
                            "json",
                            "--trajectory",
                            str(path),
                        ]
                    )
            elif kind == "agent":
                with redirect_stdout(io.StringIO()):
                    runner_module.run_agent(
                        env_id="ChemWorld",
                        agent=FixedSequence(),
                        composition=recipe["composition"],
                        world_split="public-dev",
                        budget=80,
                        objective="balanced",
                        seed=7,
                        observation_noise_mode=noise,
                        observation_seed=12,
                        output_path=path,
                    )
            else:
                raise ValueError("unknown actual entry adapter")
    return traces


def physical_contract(recipe, traces):
    expected = ["success"] * len(recipe["actions"])
    if recipe["failure"]:
        expected[1] = "failure"
    if len(traces) != len(expected):
        return False
    for index, record in enumerate(traces):
        if (record["outcome"] == "success") != (expected[index] == "success"):
            return False
        if expected[index] == "success" and record["transaction_status"] != "committed":
            return False
    if recipe["failure"]:
        before, after = traces[:2]
        fields = ("equipment", "lab_clock", "samples", "state_slots")
        if (
            any(before["state"].get(key) != after["state"].get(key) for key in fields)
            or before["state"]["ledger"]["time_s"] != after["state"]["ledger"]["time_s"]
            or before["rng_state"] != after["rng_state"]
        ):
            return False
    return traces[-1]["state"]["terminated"]


def pending_count(snapshot):
    equipment = snapshot["world_state"]["equipment"]
    records = [
        record
        for record in equipment.values()
        if record["settings"].get("format") == "chemworld_formal_model_state_v1"
    ]
    if len(records) != 1:
        raise ValueError("checkpoint must contain the one formal model owner")
    return len(records[0]["settings"]["continuation"]["events"]["pending"])


def resume(snapshot_file, actions_file, output):
    from chemworld.runtime import ChemWorldRuntime

    runtime = ChemWorldRuntime.from_snapshot(json.loads(snapshot_file.read_text(encoding="utf-8")))
    actions = json.loads(actions_file.read_text(encoding="utf-8"))
    traces = []
    with (output / "trace.jsonl").open("x", encoding="utf-8") as stream:
        for action in actions:
            result = runtime.execute(action)
            record = trace(runtime, result.info)
            traces.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    write(output / "final.json", runtime.snapshot().to_dict())
    print(f"native resumed suffix {len(traces)}/{len(actions)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resume(args.snapshot, args.actions, args.output)
