"""Submission manifest helpers for reproducible benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.data.submission import (
    validate_submission_bundle,
)


def summarize_submission_bundle(path: str | Path) -> dict[str, Any]:
    """Summarize score, risk, and seed coverage for a local submission bundle."""

    from chemworld.eval.metrics import evaluate_records

    root = Path(path)
    validation = validate_submission_bundle(root)
    if not validation.valid:
        return {"valid": False, "validation": validation.to_dict()}

    trajectory_paths = sorted((root / "trajectories").glob("*.jsonl"))
    evaluations = []
    seeds: set[int] = set()
    task_maturity: dict[str, dict[str, Any]] = {}
    for trajectory_path in trajectory_paths:
        records = load_jsonl(trajectory_path)
        evaluations.append(evaluate_records(records).to_dict())
        seeds.update(int(record["seed"]) for record in records)
        first = records[0]
        task_id = str(trajectory_value(first, 'benchmark_task_id', None) or first["task_id"])
        maturity_payload = {
            "kernel_maturity": first["kernel_maturity"],
            "physics_maturity": first["physics_maturity"],
            "proxy_allowed": bool(first["proxy_allowed"]),
        }
        existing = task_maturity.get(task_id)
        if existing is not None and existing != maturity_payload:
            return {
                "valid": False,
                "validation": validation.to_dict(),
                "errors": [f"mixed maturity metadata for benchmark task {task_id!r}"],
            }
        task_maturity[task_id] = maturity_payload

    total_scores = [float(item["total_score"]) for item in evaluations]
    risks = [float(item["mean_safety_risk"]) for item in evaluations]
    best_scores = [float(item["final_best_score"]) for item in evaluations]
    return {
        "valid": True,
        "path": str(root),
        "trajectory_count": len(trajectory_paths),
        "result_count": validation.result_count,
        "seeds": sorted(seeds),
        "mean_total_score": sum(total_scores) / len(total_scores),
        "mean_safety_risk": sum(risks) / len(risks),
        "max_final_best_score": max(best_scores),
        "task_maturity": task_maturity,
        "physics_maturity_levels": sorted(
            {item["physics_maturity"] for item in task_maturity.values()}
        ),
    }
