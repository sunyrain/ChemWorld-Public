"""Command line interface for the ChemWorld environment and benchmark."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld import ENV_ID
from chemworld.data.datasets import dataset_card, export_dataset
from chemworld.data.logging import load_jsonl
from chemworld.data.submission import (
    build_example_submission_bundle,
    init_submission_bundle,
    summarize_submission_bundle,
    validate_submission_bundle,
    write_submission_manifest,
)
from chemworld.eval.baseline_report import (
    CORE_BASELINE_AGENTS,
    SERIOUS_BASELINE_AGENTS,
    generate_baseline_report,
)
from chemworld.eval.leaderboard import aggregate_leaderboard, load_results
from chemworld.eval.private_artifact import (
    sign_private_eval_results,
    verify_private_eval_artifact,
)
from chemworld.eval.result_artifacts import build_verified_evaluation_result
from chemworld.eval.runner import make_agent, run_agent
from chemworld.eval.seed_suite import official_seed_suite, official_seeds_for_task
from chemworld.eval.suite import run_suite
from chemworld.eval.verify import verify_records
from chemworld.task_design import serious_task_readiness_manifest
from chemworld.tasks import CORE_TASK_IDS, SERIOUS_TASK_IDS, get_task, get_task_card, list_tasks
from chemworld.world.recipes import compile_recipe, validate_recipe
from chemworld.world.scenario import get_scenario_card, list_scenarios
from chemworld.wrappers import validate_event_action


def _resolve_run_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.task is None:
        return {
            "env": args.env,
            "world_split": args.world_split,
            "budget": args.budget,
            "objective": args.objective,
            "seed": 42 if args.seed is None else args.seed,
            "threshold": args.threshold,
            "task_id": None,
        }
    task = get_task(args.task)
    return {
        "env": task.env_id,
        "world_split": task.world_split,
        "budget": task.budget,
        "objective": task.objective,
        "seed": task.seeds[0] if args.seed is None else args.seed,
        "threshold": task.threshold,
        "task_id": task.task_id,
    }


def _resolve_suite_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.task is None:
        return {
            "env": args.env,
            "world_splits": args.world_splits,
            "seeds": args.seeds,
            "budget": args.budget,
            "objective": args.objective,
            "threshold": args.threshold,
            "task_id": None,
        }
    task = get_task(args.task)
    return {
        "env": task.env_id,
        "world_splits": [task.world_split],
        "seeds": official_seeds_for_task(task.task_id),
        "budget": task.budget,
        "objective": task.objective,
        "threshold": task.threshold,
        "task_id": task.task_id,
    }


def _run(args: argparse.Namespace) -> None:
    config = _resolve_run_args(args)
    output = args.output
    if output is None:
        output = (
            f"runs/{args.agent}_{config['env']}_{config['world_split']}_"
            f"{config['objective']}_seed{config['seed']}.jsonl"
        )
    agent = make_agent(args.agent)
    command = args.command_line
    history = run_agent(
        env_id=config["env"],
        agent=agent,
        world_split=config["world_split"],
        budget=config["budget"],
        objective=config["objective"],
        seed=config["seed"],
        task_id=config["task_id"],
        output_path=output,
    )
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = str(Path(output).with_suffix(".manifest.json"))
    write_submission_manifest(
        path=manifest_path,
        trajectory_path=output,
        agent_manifest=agent.manifest(),
        command=command,
        notes={
            "task_id": config["task_id"],
            "env_id": config["env"],
            "world_split": config["world_split"],
            "objective": config["objective"],
            "budget": config["budget"],
            "seed": config["seed"],
        },
    )
    print(
        json.dumps(
            {"trajectory": output, "manifest": manifest_path, "steps": len(history)},
            indent=2,
            sort_keys=True,
        )
    )


def _evaluate(args: argparse.Namespace) -> None:
    records = load_jsonl(args.submission)
    result = build_verified_evaluation_result(
        records,
        trajectory_path=args.submission,
        threshold=args.threshold,
    )
    if args.output is None:
        output = Path("results") / (Path(args.submission).stem + ".json")
    else:
        output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"result": str(output), **result}, indent=2, sort_keys=True))


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


def _expand_patterns(patterns: list[str]) -> list[str]:
    expanded: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        expanded.extend(matches if matches else [pattern])
    return expanded


def _leaderboard(args: argparse.Namespace) -> None:
    paths = _expand_patterns(args.results)
    rows = aggregate_leaderboard(load_results(paths), replay_verify=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


def _suite(args: argparse.Namespace) -> None:
    config = _resolve_suite_args(args)
    results = run_suite(
        agent_name=args.agent,
        env_id=config["env"],
        world_splits=config["world_splits"],
        seeds=config["seeds"],
        budget=config["budget"],
        objective=config["objective"],
        output_dir=args.output_dir,
        threshold=config["threshold"],
        task_id=config["task_id"],
    )
    print(
        json.dumps(
            {
                "runs": len(results),
                "output_dir": args.output_dir,
                "task_id": config["task_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _baselines_report(args: argparse.Namespace) -> None:
    task_ids = _resolve_task_preset(args.preset, args.tasks)
    agents = _resolve_agent_preset(args.preset, args.agents)
    report = generate_baseline_report(
        task_ids=task_ids,
        agents=agents,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def _private_eval_sign(args: argparse.Namespace) -> None:
    run_log = {
        "command": args.command_line,
        "notes": args.notes,
    }
    artifact = sign_private_eval_results(
        result_paths=args.results,
        output_path=args.output,
        salt=args.salt,
        run_log=run_log,
    )
    print(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))


def _private_eval_verify(args: argparse.Namespace) -> None:
    valid = verify_private_eval_artifact(args.artifact, salt=args.salt)
    print(json.dumps({"artifact": args.artifact, "signature_valid": valid}, indent=2))
    if not valid:
        raise SystemExit(1)


def _resolve_task_preset(preset: str | None, tasks: list[str] | None) -> list[str]:
    if tasks:
        return tasks
    if preset == "serious":
        return list(SERIOUS_TASK_IDS)
    return list(CORE_TASK_IDS)


def _resolve_agent_preset(preset: str | None, agents: list[str] | None) -> list[str]:
    if agents:
        return agents
    if preset == "serious":
        return list(SERIOUS_BASELINE_AGENTS)
    return list(CORE_BASELINE_AGENTS)


def _tasks_list(args: argparse.Namespace) -> None:
    del args
    print(json.dumps([task.to_dict() for task in list_tasks()], indent=2, sort_keys=True))


def _tasks_show(args: argparse.Namespace) -> None:
    print(json.dumps(get_task(args.task_id).to_dict(), indent=2, sort_keys=True))


def _tasks_card(args: argparse.Namespace) -> None:
    print(json.dumps(get_task_card(args.task_id), indent=2, sort_keys=True))


def _tasks_readiness(args: argparse.Namespace) -> None:
    del args
    print(json.dumps(serious_task_readiness_manifest(), indent=2, sort_keys=True))


def _scenarios_list(args: argparse.Namespace) -> None:
    del args
    payload = [scenario.to_dict() for scenario in list_scenarios()]
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
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_action(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    action = _load_json_file(args.action)
    env = gym.make(task.env_id, task_id=task.task_id, seed=args.seed)
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
    env = gym.make("ChemWorld", task_id=args.task, seed=args.seed, render_mode="ansi")
    try:
        env.reset(seed=args.seed)
        for action_path in args.actions:
            action = _load_json_file(action_path)
            env.step(action)
        rendered: Any = env.render()
        print(rendered)
    finally:
        env.close()


def _submission_init(args: argparse.Namespace) -> None:
    manifest = init_submission_bundle(
        args.path,
        agent_name=args.agent_name,
        agent_family=args.agent_family,
        task_id=args.task_id,
        seeds=args.seeds,
        command=args.command_text,
        dependency_file=args.dependency_file,
    )
    print(json.dumps({"path": args.path, "manifest": manifest}, indent=2, sort_keys=True))


def _submission_validate(args: argparse.Namespace) -> None:
    result = validate_submission_bundle(args.path).to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


def _submission_summarize(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_submission_bundle(args.path), indent=2, sort_keys=True))


def _submission_example(args: argparse.Namespace) -> None:
    result = build_example_submission_bundle(
        args.path,
        task_id=args.task_id,
        agent_name=args.agent,
        seeds=args.seeds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["validation"]["valid"]:
        raise SystemExit(1)
    if not all(item["verified"] for item in result["verification"]):
        raise SystemExit(1)


def _seeds_show(args: argparse.Namespace) -> None:
    task_ids = (
        [task.task_id for task in list_tasks()]
        if args.all_tasks
        else args.tasks
    )
    print(json.dumps(official_seed_suite(task_ids), indent=2, sort_keys=True))


def _inspect_constitution(args: argparse.Namespace) -> None:
    kwargs: dict[str, Any] = {
        "world_split": args.world_split,
        "budget": args.budget,
        "objective": args.objective,
        "seed": args.seed,
    }
    env = gym.make(args.env, **kwargs)
    try:
        env.reset(seed=args.seed)
        if not hasattr(env.unwrapped, "constitution_summary"):
            raise SystemExit(f"{args.env} does not expose a foundation constitution")
        print(json.dumps(env.unwrapped.constitution_summary(), indent=2, sort_keys=True))
    finally:
        env.close()


def _lab(args: argparse.Namespace) -> None:
    from chemworld.lab.server import serve

    serve(args.host, args.port, open_browser=not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chemworld")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an agent and write a trajectory JSONL.")
    run_parser.add_argument("--env", default=ENV_ID)
    run_parser.add_argument("--task")
    run_parser.add_argument("--agent", default="random")
    run_parser.add_argument("--world-split", default="public-dev")
    run_parser.add_argument("--budget", type=int, default=30)
    run_parser.add_argument("--objective", default="balanced")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--threshold", type=float, default=0.75)
    run_parser.add_argument("--output")
    run_parser.add_argument("--manifest")
    run_parser.set_defaults(func=_run)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate one trajectory JSONL.")
    eval_parser.add_argument("--submission", required=True)
    eval_parser.add_argument("--threshold", type=float, default=0.75)
    eval_parser.add_argument("--output")
    eval_parser.set_defaults(func=_evaluate)

    verify_parser = subparsers.add_parser("verify", help="Replay and verify one trajectory JSONL.")
    verify_parser.add_argument("--submission", required=True)
    verify_parser.add_argument("--tolerance", type=float, default=1.0e-5)
    verify_parser.add_argument("--constitution", action="store_true")
    verify_parser.set_defaults(func=_verify)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Aggregate result JSON files.")
    leaderboard_parser.add_argument("--results", nargs="+", required=True)
    leaderboard_parser.add_argument("--output")
    leaderboard_parser.set_defaults(func=_leaderboard)

    suite_parser = subparsers.add_parser("suite", help="Run an agent across splits and seeds.")
    suite_parser.add_argument("--env", default=ENV_ID)
    suite_parser.add_argument("--task")
    suite_parser.add_argument("--agent", default="random")
    suite_parser.add_argument(
        "--world-splits",
        nargs="+",
        default=["public-test", "private-eval"],
    )
    suite_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    suite_parser.add_argument("--budget", type=int, default=30)
    suite_parser.add_argument("--objective", default="balanced")
    suite_parser.add_argument("--threshold", type=float, default=0.75)
    suite_parser.add_argument("--output-dir", default="runs/suite")
    suite_parser.set_defaults(func=_suite)

    baselines_parser = subparsers.add_parser("baselines", help="Generate baseline reports.")
    baselines_subparsers = baselines_parser.add_subparsers(
        dest="baselines_command",
        required=True,
    )
    baselines_report_parser = baselines_subparsers.add_parser(
        "report",
        help="Run official baselines and write a task-based report.",
    )
    baselines_report_parser.add_argument(
        "--preset",
        choices=["core", "serious"],
        default="core",
        help="Use a frozen task/agent preset unless --tasks or --agents override it.",
    )
    baselines_report_parser.add_argument(
        "--tasks",
        nargs="+",
    )
    baselines_report_parser.add_argument(
        "--agents",
        nargs="+",
    )
    baselines_report_parser.add_argument("--seeds", nargs="+", type=int)
    baselines_report_parser.add_argument("--output-dir", default="runs/baseline_report")
    baselines_report_parser.set_defaults(func=_baselines_report)

    private_eval_parser = subparsers.add_parser(
        "private-eval",
        help="Create and verify signed private-eval artifacts.",
    )
    private_eval_subparsers = private_eval_parser.add_subparsers(
        dest="private_eval_command",
        required=True,
    )
    private_sign_parser = private_eval_subparsers.add_parser(
        "sign",
        help="Sign private-eval result JSON files with CHEMWORLD_PRIVATE_EVAL_SALT.",
    )
    private_sign_parser.add_argument("--results", nargs="+", required=True)
    private_sign_parser.add_argument("--output", required=True)
    private_sign_parser.add_argument("--salt")
    private_sign_parser.add_argument("--notes", default="")
    private_sign_parser.set_defaults(func=_private_eval_sign)
    private_verify_parser = private_eval_subparsers.add_parser(
        "verify",
        help="Verify a signed private-eval artifact with a maintainer salt.",
    )
    private_verify_parser.add_argument("--artifact", required=True)
    private_verify_parser.add_argument("--salt", required=True)
    private_verify_parser.set_defaults(func=_private_eval_verify)

    tasks_parser = subparsers.add_parser("tasks", help="Inspect benchmark task specs.")
    tasks_subparsers = tasks_parser.add_subparsers(dest="tasks_command", required=True)
    tasks_list_parser = tasks_subparsers.add_parser("list", help="List registered tasks.")
    tasks_list_parser.set_defaults(func=_tasks_list)
    tasks_show_parser = tasks_subparsers.add_parser("show", help="Show one registered task.")
    tasks_show_parser.add_argument("task_id")
    tasks_show_parser.set_defaults(func=_tasks_show)
    tasks_card_parser = tasks_subparsers.add_parser("card", help="Show one task card.")
    tasks_card_parser.add_argument("task_id")
    tasks_card_parser.set_defaults(func=_tasks_card)
    tasks_readiness_parser = tasks_subparsers.add_parser(
        "readiness",
        help="Review serious benchmark task contracts and evidence.",
    )
    tasks_readiness_parser.set_defaults(func=_tasks_readiness)

    seeds_parser = subparsers.add_parser("seeds", help="Inspect official seed suites.")
    seeds_subparsers = seeds_parser.add_subparsers(dest="seeds_command", required=True)
    seeds_show_parser = seeds_subparsers.add_parser(
        "show",
        help="Show the official core seed suite.",
    )
    seeds_show_parser.add_argument("--tasks", nargs="+")
    seeds_show_parser.add_argument("--all-tasks", action="store_true")
    seeds_show_parser.set_defaults(func=_seeds_show)

    scenarios_parser = subparsers.add_parser("scenarios", help="Inspect scenario specs.")
    scenarios_subparsers = scenarios_parser.add_subparsers(
        dest="scenarios_command",
        required=True,
    )
    scenarios_list_parser = scenarios_subparsers.add_parser("list", help="List scenarios.")
    scenarios_list_parser.set_defaults(func=_scenarios_list)
    scenarios_show_parser = scenarios_subparsers.add_parser("show", help="Show one scenario.")
    scenarios_show_parser.add_argument("scenario_id")
    scenarios_show_parser.add_argument("--split")
    scenarios_show_parser.set_defaults(func=_scenarios_show)

    validate_action_parser = subparsers.add_parser(
        "validate-action",
        help="Validate one event action against a task's initial state.",
    )
    validate_action_parser.add_argument("--task", required=True)
    validate_action_parser.add_argument("--action", required=True)
    validate_action_parser.add_argument("--seed", type=int, default=0)
    validate_action_parser.set_defaults(func=_validate_action)

    validate_recipe_parser = subparsers.add_parser(
        "validate-recipe",
        help="Validate and compile a recipe JSON file.",
    )
    validate_recipe_parser.add_argument("--task", required=True)
    validate_recipe_parser.add_argument("--recipe", required=True)
    validate_recipe_parser.set_defaults(func=_validate_recipe)

    datasets_parser = subparsers.add_parser("datasets", help="Export trajectory datasets.")
    datasets_subparsers = datasets_parser.add_subparsers(
        dest="datasets_command",
        required=True,
    )
    datasets_export_parser = datasets_subparsers.add_parser(
        "export",
        help="Export a submission or trajectory to dataset format.",
    )
    datasets_export_parser.add_argument("--submission", required=True)
    datasets_export_parser.add_argument("--format", choices=["jsonl", "parquet"], required=True)
    datasets_export_parser.add_argument("--output")
    datasets_export_parser.set_defaults(func=_datasets_export)
    datasets_card_parser = datasets_subparsers.add_parser("card", help="Summarize a dataset.")
    datasets_card_parser.add_argument("--dataset", required=True)
    datasets_card_parser.set_defaults(func=_datasets_card)

    render_parser = subparsers.add_parser("render", help="Render an ANSI environment summary.")
    render_parser.add_argument("--task", required=True)
    render_parser.add_argument("--seed", type=int, default=0)
    render_parser.add_argument("--actions", nargs="*", default=[])
    render_parser.set_defaults(func=_render)

    submission_parser = subparsers.add_parser("submission", help="Manage submission bundles.")
    submission_subparsers = submission_parser.add_subparsers(
        dest="submission_command",
        required=True,
    )
    submission_init_parser = submission_subparsers.add_parser(
        "init",
        help="Create a local submission bundle skeleton.",
    )
    submission_init_parser.add_argument("path")
    submission_init_parser.add_argument("--agent-name", default="unknown")
    submission_init_parser.add_argument("--agent-family", default="unknown")
    submission_init_parser.add_argument("--task-id", default="reaction-optimization-standard")
    submission_init_parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    submission_init_parser.add_argument("--command-text", nargs="*", default=[])
    submission_init_parser.add_argument("--dependency-file", default="dependency_notes.md")
    submission_init_parser.set_defaults(func=_submission_init)
    submission_validate_parser = submission_subparsers.add_parser(
        "validate",
        help="Validate a submission bundle.",
    )
    submission_validate_parser.add_argument("path")
    submission_validate_parser.set_defaults(func=_submission_validate)
    submission_summarize_parser = submission_subparsers.add_parser(
        "summarize",
        help="Summarize a submission bundle.",
    )
    submission_summarize_parser.add_argument("path")
    submission_summarize_parser.set_defaults(func=_submission_summarize)
    submission_example_parser = submission_subparsers.add_parser(
        "example",
        help="Build a complete valid submission bundle example.",
    )
    submission_example_parser.add_argument("path")
    submission_example_parser.add_argument("--task-id", default="reaction-to-purification")
    submission_example_parser.add_argument("--agent", default="tool_using_llm_stub")
    submission_example_parser.add_argument("--seeds", nargs="+", type=int)
    submission_example_parser.set_defaults(func=_submission_example)

    constitution_parser = subparsers.add_parser(
        "inspect-constitution",
        help="Inspect an environment's physical constitution checklist.",
    )
    constitution_parser.add_argument("--env", default=ENV_ID)
    constitution_parser.add_argument("--world-split", default="public-dev")
    constitution_parser.add_argument("--budget", type=int, default=30)
    constitution_parser.add_argument("--objective", default="balanced")
    constitution_parser.add_argument("--seed", type=int, default=42)
    constitution_parser.set_defaults(func=_inspect_constitution)

    lab_parser = subparsers.add_parser(
        "lab",
        help="Open the provider-free Student Lab in a local browser.",
    )
    lab_parser.add_argument("--host", default="127.0.0.1")
    lab_parser.add_argument("--port", type=int, default=8876)
    lab_parser.add_argument("--no-browser", action="store_true")
    lab_parser.set_defaults(func=_lab)
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
