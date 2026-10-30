"""Submission manifest helpers for reproducible benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chemworld.data.logging import load_jsonl
from chemworld.data.submission import (
    init_submission_bundle,
    validate_submission_bundle,
)
from chemworld.data.validation import validate_records
from chemworld.eval.submissions import summarize_submission_bundle
from chemworld.tasks import get_task


def build_example_submission_bundle(
    path: str | Path,
    *,
    task_id: str = "reaction-to-purification",
    agent_name: str = "tool_using_llm_stub",
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Build a complete, valid local submission bundle example.

    The bundle is intentionally generated through the same runner, evaluator,
    verifier, and validator used by normal submissions.
    """

    from chemworld.eval.metrics import evaluate_records
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records
    from chemworld.legacy.agent_registry import make_agent

    task = get_task(task_id)
    resolved_seeds = list(task.seeds[:1] if seeds is None else seeds)
    root = Path(path)
    command = [
        "python",
        "-m",
        "chemworld.legacy.cli",
        "submission",
        "example",
        str(root),
        "--task-id",
        task_id,
        "--agent",
        agent_name,
        "--seeds",
        *(str(seed) for seed in resolved_seeds),
    ]
    agent_family = make_agent(agent_name).__class__.__name__
    init_submission_bundle(
        root,
        agent_name=agent_name,
        agent_family=agent_family,
        task_id=task_id,
        seeds=resolved_seeds,
        command=command,
        dependency_file="dependency_notes.md",
        llm_metadata={
            "online_model": None,
            "mode": "offline deterministic example",
            "requires_network": False,
        },
    )

    verification_results: list[dict[str, Any]] = []
    for seed in resolved_seeds:
        trajectory_path = (
            root / "trajectories" / f"{agent_name}_{task_id}_seed{seed}.jsonl"
        )
        result_path = root / "results" / f"{agent_name}_{task_id}_seed{seed}.json"
        explanation_path = (
            root / "explanations" / f"{agent_name}_{task_id}_seed{seed}.json"
        )
        run_agent(
            env_id=task.env_id,
            agent=make_agent(agent_name),
            world_split=task.world_split,
            budget=task.budget,
            objective=task.objective,
            seed=seed,
            task_id=task.task_id,
            output_path=trajectory_path,
        )
        records = load_jsonl(trajectory_path)
        validate_records(records)
        verification = verify_records(records).to_dict()
        verification_results.append(
            {
                "seed": seed,
                "trajectory_path": str(trajectory_path),
                "verified": verification["verified"],
                "checked_steps": verification["checked_steps"],
                "max_abs_error": verification["max_abs_error"],
            }
        )
        evaluation = evaluate_records(records, threshold=task.threshold).to_dict()
        evaluation.update(
            {
                "task_id": task.task_id,
                "task_contract_hash": records[0].get("task_contract_hash"),
                "runtime_profile_hash": records[0].get("runtime_profile_hash"),
                "mechanism_hash": records[0].get("mechanism_hash"),
                "scoring_contract_hash": records[0].get("scoring_contract_hash"),
                "observation_contract_hash": records[0].get("observation_contract_hash"),
                "trajectory_path": str(trajectory_path),
                "verification": verification,
            }
        )
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2, sort_keys=True)
        explanation = {
            "schema_version": "chemworld-explanation-0.1",
            "task_id": task.task_id,
            "agent_name": agent_name,
            "seed": seed,
            "hypothesis": (
                "Moderate reaction conditions followed by phase handling should "
                "increase final assay score while keeping risk controlled."
            ),
            "learned_mechanism": (
                "The public observations are consistent with a target reaction "
                "competing against byproduct/degradation and downstream recovery losses."
            ),
            "failure_analysis": (
                "This example is a reproducibility artifact, not an optimized submission; "
                "it uses a deterministic tool-agent plan and does not tune per seed."
            ),
            "next_experiment_rationale": (
                "Compare a lower-temperature reaction with a longer extraction stage and "
                "use HPLC before final assay to update the local belief state."
            ),
            "public_evidence": {
                "steps": len(records),
                "final_best_score": evaluation["final_best_score"],
                "total_score": evaluation["total_score"],
                "invalid_action_count": evaluation["invalid_action_count"],
                "final_assay_count": evaluation["final_assay_count"],
            },
        }
        with explanation_path.open("w", encoding="utf-8") as handle:
            json.dump(explanation, handle, indent=2, sort_keys=True)

    validation = validate_submission_bundle(root)
    summary = summarize_submission_bundle(root)
    return {
        "path": str(root),
        "task_id": task_id,
        "agent_name": agent_name,
        "seeds": resolved_seeds,
        "validation": validation.to_dict(),
        "summary": summary,
        "verification": verification_results,
        "reproducible_command": command,
    }
