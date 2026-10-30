"""Command line tools for the programmable ChemWorld environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld import ENV_ID
from chemworld.data.datasets import (
    export_dataset,
)
from chemworld.data.logging import load_jsonl
from chemworld.eval.dataset_cards import (
    dataset_card,
)
from chemworld.eval.verify import verify_records
from chemworld.tasks import get_task, list_tasks
from chemworld.world.recipes import compile_recipe, validate_recipe
from chemworld.world.scenario import get_scenario_card, list_scenarios
from chemworld.wrappers import validate_event_action


def _verify(args: argparse.Namespace) -> None:
    records = load_jsonl(args.submission)
    result = verify_records(records, tolerance=args.tolerance).to_dict()
    if args.constitution:
        failures: list[dict[str, object]] = []
        for record in records:
            checks = record.get("constitution_checks", [])
            if not checks:
                failures.append(
                    {"step": record.get("step"), "message": "missing constitution checks"}
                )
                continue
            for name, passed in record.get("preconditions", {}).items():
                if not passed:
                    failures.append(
                        {
                            "step": record.get("step"),
                            "precondition": name,
                            "message": "failed action precondition",
                        }
                    )
            for check in checks:
                if not check.get("passed", False):
                    failures.append({"step": record.get("step"), "check": check})
        result["constitution_checked"] = True
        result["constitution_passed"] = not failures
        result["constitution_failures"] = failures
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["verified"] or not result.get("constitution_passed", True):
        raise SystemExit(1)


def _tasks_list(args: argparse.Namespace) -> None:
    tasks = list_tasks()
    if args.format == "summary":
        payload = [
            {
                "task_id": task.task_id,
                "scenario_id": task.scenario_id,
                "difficulty": task.difficulty,
                "budget": task.budget,
                "episode_mode": task.episode_mode,
                "objective": task.objective,
                "success_metrics": list(task.success_metrics),
                "description": task.description,
            }
            for task in tasks
        ]
    else:
        payload = [task.to_dict() for task in tasks]
    print(json.dumps(payload, indent=2, sort_keys=True))


def _tasks_show(args: argparse.Namespace) -> None:
    print(json.dumps(get_task(args.task_id).to_dict(), indent=2, sort_keys=True))


def _scenarios_list(args: argparse.Namespace) -> None:
    scenarios = list_scenarios()
    if args.format == "summary":
        payload = [
            {
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "split": scenario.split,
                "difficulty": scenario.difficulty,
                "parameter_profile": scenario.parameter_profile,
                "initial_state_id": scenario.initial_state_id,
                "expected_qualitative_behavior": list(scenario.expected_qualitative_behavior),
            }
            for scenario in scenarios
        ]
    else:
        payload = [scenario.to_dict() for scenario in scenarios]
    print(json.dumps(payload, indent=2, sort_keys=True))


def _scenarios_show(args: argparse.Namespace) -> None:
    try:
        card = get_scenario_card(args.scenario_id, split=args.split)
    except KeyError as exc:
        raise ValueError(
            f"Unknown scenario {args.scenario_id!r}; use 'chemworld scenarios list'."
        ) from exc
    print(
        json.dumps(
            card,
            indent=2,
            sort_keys=True,
        )
    )


def _load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from a file path or an inline JSON argument.

    CLI examples are often copied into a shell one action at a time.  Accepting
    an inline object keeps that workflow equivalent to the file based form,
    while retaining file paths for larger compositions, recipes and actions.
    """

    raw = str(path)
    try:
        candidate = Path(raw)
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{path} must be a JSON object file path or an inline JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_action(args: argparse.Namespace) -> None:
    from chemworld.runtime import ChemWorldRuntime

    task = get_task(args.task)
    action = _load_json_file(args.action)
    env = ChemWorldRuntime(task_id=task.task_id, seed=args.seed)
    try:
        env.reset(seed=args.seed)
        result = validate_event_action(action, env)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["valid"]:
            raise SystemExit(1)
    finally:
        env.close()


def _validate_recipe(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    recipe = _load_json_file(args.recipe)
    task_info = task.to_dict()
    result = validate_recipe(recipe, task_info=task_info).to_dict()
    result["task_id"] = task.task_id
    if result["valid"]:
        result["compiled_steps"] = compile_recipe(recipe, task_info=task_info)
        result["compiled_step_count"] = len(result["compiled_steps"])
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


def _datasets_export(args: argparse.Namespace) -> None:
    output = args.output
    if output is None:
        suffix = "jsonl" if args.format == "jsonl" else "parquet"
        output = str(Path("datasets") / f"chemworld_dataset.{suffix}")
    result = export_dataset(args.submission, output=output, format=args.format).to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))


def _datasets_card(args: argparse.Namespace) -> None:
    print(json.dumps(dataset_card(args.dataset), indent=2, sort_keys=True))


def _render(args: argparse.Namespace) -> None:
    from chemworld.data.logging import TrajectoryLogger, to_builtin
    from chemworld.envs.spaces import to_observation
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.lifecycle import initialization_boundaries

    boundaries = initialization_boundaries(
        _load_json_file(args.initializations) if args.initializations else None
    )
    if any(index >= len(args.actions) for index in boundaries):
        raise ValueError("initialization boundary must precede a recorded action")

    outputs = [Path(output).resolve() for output in (args.trajectory, args.snapshot) if output]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Trajectory and snapshot outputs must be different files")
    for output in outputs:
        if output is not None and Path(output).exists():
            raise ValueError(f"Output already exists: {output}")
    kwargs = (
        {"task_id": args.task}
        if args.composition is None
        else {
            "composition": _load_json_file(args.composition),
        }
    )
    env = ChemWorldRuntime(
        **kwargs,
        seed=args.seed,
        render_mode="ansi",
        observation_noise_mode=args.noise_mode,
        observation_seed_override=args.observation_seed,
    )
    logger = None
    history = []
    try:
        env.reset(seed=args.seed)
        task_info = {**env.task_info(), **env.evaluator_provenance()}
        logger = TrajectoryLogger(args.trajectory) if args.trajectory else None
        for step, action_path in enumerate(args.actions, start=1):
            if step - 1 in boundaries:
                env.reset(options=boundaries[step - 1])
            action = _load_json_file(action_path)
            result = env.execute(action)
            if logger is not None:
                logger.log(
                    task_info={**task_info, **env.evaluator_provenance()},
                    step=step,
                    action=action,
                    observation=to_observation(result.observation.values),
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                    agent_metadata={"agent_id": "cli-event-sequence"},
                )
            history.append(
                {
                    "action": action,
                    "outcome": result.info["operation_outcome"],
                    "transaction": result.info["transaction_status"],
                    "actual_duration_s": result.info["actual_duration_s"],
                    "measurement_status": result.info["measurement_status"],
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                }
            )
        if args.snapshot is not None:
            env.snapshot().save(args.snapshot)
        if args.format == "json":
            print(
                json.dumps(
                    to_builtin({"steps": history, "observation": env.observe().to_dict()}), indent=2
                )
            )
        else:
            print(env.render())
    finally:
        if logger is not None:
            logger.close()
        env.close()


def _inspect_constitution(args: argparse.Namespace) -> None:
    from chemworld.runtime import ChemWorldRuntime

    kwargs: dict[str, Any] = {
        "world_split": args.world_split,
        "budget": args.budget,
        "objective": args.objective,
        "seed": args.seed,
    }
    env = ChemWorldRuntime(**kwargs) if args.env == ENV_ID else gym.make(args.env, **kwargs)
    try:
        env.reset(seed=args.seed)
        base = getattr(env, "unwrapped", env)
        if not hasattr(base, "constitution_summary"):
            raise SystemExit(f"{args.env} does not expose a foundation constitution")
        print(json.dumps(base.constitution_summary(), indent=2, sort_keys=True))
    finally:
        env.close()


def _capabilities(args):
    from chemworld.examples.native_capabilities import capability_rows, markdown, summary_rows

    if args.format == "markdown":
        print(markdown())
    elif args.format == "summary":
        print(json.dumps(summary_rows(), indent=2, sort_keys=True))
    else:
        print(json.dumps(capability_rows(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chemworld",
        description=(
            "Programmable physical-chemistry environments. Agents and rewards are user-owned."
        ),
        epilog="Historical benchmark commands: python -m chemworld.legacy.cli --help",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "capabilities", help="Inspect installed domain, operation and instrument declarations."
    )
    p.add_argument("--format", choices=("json", "markdown", "summary"), default="json")
    p.set_defaults(func=_capabilities)
    p = commands.add_parser("verify", help="Replay recorded operations; does not score an agent.")
    p.add_argument("--submission", required=True)
    p.add_argument("--tolerance", type=float, default=1e-5)
    p.add_argument("--constitution", action="store_true")
    p.set_defaults(func=_verify)
    p = commands.add_parser("tasks", help="Inspect existing environment presets.")
    sub = p.add_subparsers(dest="tasks_command", required=True)
    q = sub.add_parser("list")
    q.add_argument("--format", choices=("json", "summary"), default="json")
    q.set_defaults(func=_tasks_list)
    q = sub.add_parser("show")
    q.add_argument("task_id")
    q.set_defaults(func=_tasks_show)
    p = commands.add_parser("scenarios", help="Inspect world scenarios.")
    sub = p.add_subparsers(dest="scenarios_command", required=True)
    q = sub.add_parser("list")
    q.add_argument("--format", choices=("json", "summary"), default="json")
    q.set_defaults(func=_scenarios_list)
    q = sub.add_parser("show")
    q.add_argument("scenario_id")
    q.add_argument("--split", default="public-dev")
    q.set_defaults(func=_scenarios_show)
    p = commands.add_parser(
        "validate-action", help="Validate a physical action without executing it."
    )
    p.add_argument("--task", required=True)
    p.add_argument("--action", required=True, help="JSON object file or inline JSON object")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_validate_action)
    p = commands.add_parser(
        "validate-recipe", help="Validate a recipe against an environment preset."
    )
    p.add_argument("--task", required=True)
    p.add_argument("--recipe", required=True, help="JSON object file or inline JSON object")
    p.set_defaults(func=_validate_recipe)
    p = commands.add_parser("datasets", help="Export and inspect recorded interaction data.")
    sub = p.add_subparsers(dest="datasets_command", required=True)
    q = sub.add_parser("export")
    q.add_argument("--submission", required=True)
    q.add_argument("--format", choices=("jsonl", "parquet"), required=True)
    q.add_argument("--output")
    q.set_defaults(func=_datasets_export)
    q = sub.add_parser("card")
    q.add_argument("--dataset", required=True)
    q.set_defaults(func=_datasets_card)
    p = commands.add_parser("render", help="Execute supplied actions in a native physical session.")
    world = p.add_mutually_exclusive_group(required=True)
    world.add_argument("--task")
    world.add_argument("--composition", help="Explicit world composition JSON.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--observation-seed", type=int)
    p.add_argument(
        "--actions",
        nargs="*",
        default=[],
        metavar="ACTION_JSON",
        help="JSON object files or inline JSON objects, in execution order",
    )
    p.add_argument("--noise-mode", choices=("sequential", "keyed"), default="sequential")
    p.add_argument("--format", choices=("ansi", "json"), default="ansi")
    p.add_argument("--trajectory")
    p.add_argument("--snapshot")
    p.add_argument("--initializations")
    p.set_defaults(func=_render)
    p = commands.add_parser("inspect-constitution", help="Inspect physical constraints.")
    p.add_argument("--env", default=ENV_ID)
    p.add_argument("--world-split", default="public-dev")
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--objective", default="balanced")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_inspect_constitution)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.command_line = ["chemworld", *(argv if argv is not None else sys.argv[1:])]
    try:
        args.func(args)
    except (OSError, ValueError, gym.error.UnregisteredEnv) as exc:
        parser.exit(2, f"chemworld: error: {exc}\n")


if __name__ == "__main__":
    main()
