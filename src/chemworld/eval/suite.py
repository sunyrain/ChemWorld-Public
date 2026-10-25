"""Batch evaluation suites across splits and seeds."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from chemworld.data.logging import load_jsonl
from chemworld.eval.result_artifacts import build_verified_evaluation_result
from chemworld.eval.runner import EvaluationPolicy, run_agent
from chemworld.legacy.agent_registry import make_agent


def run_suite(
    *,
    agent_name: str,
    env_id: str,
    world_splits: list[str],
    seeds: list[int],
    budget: int,
    objective: str,
    output_dir: str | Path,
    threshold: float = 0.75,
    task_id: str | None = None,
    budget_override: int | None = None,
    evaluation_policy: EvaluationPolicy = "task_contract",
) -> list[dict[str, Any]]:
    """Run an agent across a matrix of splits and seeds, then evaluate runs."""

    root = Path(output_dir)
    trajectory_dir = root / "trajectories"
    result_dir = root / "results"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for split in world_splits:
        for seed in seeds:
            trajectory_path = trajectory_dir / (
                f"{agent_name}_{env_id}_{split}_{objective}_seed{seed}.jsonl"
            )
            agent = make_agent(agent_name)
            run_wall_start = time.perf_counter()
            process_start = time.process_time()
            run_agent(
                env_id=env_id,
                agent=agent,
                world_split=split,
                budget=budget,
                objective=objective,
                seed=seed,
                task_id=task_id,
                output_path=trajectory_path,
                budget_override=budget_override,
                evaluation_policy=evaluation_policy,
            )
            run_wall_time = time.perf_counter() - run_wall_start
            evaluation_wall_start = time.perf_counter()
            records = load_jsonl(trajectory_path)
            result = build_verified_evaluation_result(
                records,
                trajectory_path=trajectory_path,
                threshold=threshold,
            )
            evaluation_wall_time = time.perf_counter() - evaluation_wall_start
            process_cpu_time = time.process_time() - process_start
            result.update(_maturity_payload(records[0]))
            result["resource_usage"] = _resource_usage_payload(
                records=records,
                result=result,
                orchestration_wall_time_s=run_wall_time,
                evaluation_wall_time_s=evaluation_wall_time,
                process_cpu_time_s=process_cpu_time,
            )
            result_path = result_dir / (trajectory_path.stem + ".json")
            result["result_path"] = str(result_path.resolve())
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
            results.append(result)

    manifest_path = root / "suite_results.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    return results


def _maturity_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "kernel_maturity": record.get("kernel_maturity", {}),
        "physics_maturity": record.get("physics_maturity"),
        "proxy_allowed": bool(record.get("proxy_allowed", False)),
    }


def _resource_usage_payload(
    *,
    records: list[dict[str, Any]],
    result: dict[str, Any],
    orchestration_wall_time_s: float,
    evaluation_wall_time_s: float,
    process_cpu_time_s: float,
) -> dict[str, Any]:
    """Promote the final cumulative runner ledger into the suite result."""

    ledger = records[-1].get("method_resources", {})
    if ledger.get("schema_version") != "chemworld-method-resource-ledger-0.1":
        raise ValueError("trajectory is missing the vNext method resource ledger")
    if ledger.get("accounting_complete") is not True:
        raise ValueError("trajectory method resource accounting is incomplete")
    operation_count = int(ledger.get("operation_count", -1))
    experiment_count = int(ledger.get("complete_experiment_count", -1))
    if operation_count != int(result["steps"]):
        raise ValueError("method operation ledger does not match verified trajectory steps")
    if experiment_count != int(result["final_assay_count"]):
        raise ValueError("method experiment ledger does not match verified final assays")
    agent_usage = ledger.get("agent_usage", {})
    return {
        "schema_version": "chemworld-resource-usage-0.2",
        "run_wall_time_s": float(ledger["run_wall_time_s"]),
        "orchestration_wall_time_s": orchestration_wall_time_s,
        "evaluation_wall_time_s": evaluation_wall_time_s,
        "total_wall_time_s": orchestration_wall_time_s + evaluation_wall_time_s,
        "process_cpu_time_s": process_cpu_time_s,
        "step_count": operation_count,
        "complete_experiment_count": experiment_count,
        "model_call_count": int(agent_usage.get("model_call_count", 0)),
        "input_token_count": int(agent_usage.get("input_token_count", 0)),
        "output_token_count": int(agent_usage.get("output_token_count", 0)),
        "monetary_cost_usd": float(agent_usage.get("monetary_cost_usd", 0.0)),
        "training_environment_step_count": int(
            agent_usage.get("training_environment_step_count", 0)
        ),
        "cpu_time_s": float(agent_usage.get("cpu_time_s", 0.0)),
        "gpu_time_s": float(agent_usage.get("gpu_time_s", 0.0)),
        "method_ledger": ledger,
    }
