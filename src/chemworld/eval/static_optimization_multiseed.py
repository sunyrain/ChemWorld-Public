"""Aggregate audited S0 static optimization runs across world seeds."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.static_campaign_statistics import score_summary

STATIC_OPTIMIZATION_MULTISEED_VERSION = (
    "chemworld-static-optimization-multiseed-0.1-s0-dev"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return dict(score_summary(values))


def _best_so_far(values: Sequence[float]) -> list[float]:
    best: list[float] = []
    current = float("-inf")
    for value in values:
        current = max(current, float(value))
        best.append(current)
    return best


def _curve_summary(curves: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    if not curves:
        return []
    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        raise ValueError("S0 score curves must have a common horizon")
    horizon = lengths.pop()
    return [
        {
            "experiment_index": index,
            **_summary([float(curve[index]) for curve in curves]),
        }
        for index in range(horizon)
    ]


def aggregate_static_optimization_runs(
    run_roots: Sequence[str | Path],
) -> dict[str, Any]:
    """Build one audited development summary from seed-level S0 run roots."""

    roots = [Path(root) for root in run_roots]
    if not roots:
        raise ValueError("at least one S0 run root is required")

    run_rows: list[dict[str, Any]] = []
    task_rows: dict[str, list[dict[str, Any]]] = {}
    method_hashes: set[str] = set()
    method_ids: set[str] = set()
    provider_modes: set[str] = set()
    protocol_ids: set[str] = set()
    formal_flags: set[bool] = set()
    benchmark_claim_flags: set[bool] = set()
    seen_seeds: set[int] = set()

    for root in roots:
        report_path = root / "report.json"
        audit_path = root / "postrun_audit.json"
        if not report_path.exists() or not audit_path.exists():
            raise ValueError(f"S0 run is missing report or audit: {root}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        seed_values = {int(cell["cell"]["world_seed"]) for cell in report["cells"]}
        if len(seed_values) != 1:
            raise ValueError(f"S0 run mixes world seeds: {root}")
        seed = seed_values.pop()
        if seed in seen_seeds:
            raise ValueError(f"duplicate S0 world seed: {seed}")
        seen_seeds.add(seed)

        method_hashes.add(str(report["method_config_sha256"]))
        method_ids.update(str(item) for item in report["method_ids"])
        provider_modes.add(str(report["provider_mode"]))
        protocol_ids.add(str(report["protocol_id"]))
        formal_flags.add(bool(report.get("formal_result", False)))
        benchmark_claim_flags.add(bool(report.get("benchmark_claim_allowed", False)))
        audit_passed = all(
            (
                audit.get("static_world_verified") is True,
                audit.get("no_mechanism_fields_in_plans") is True,
                audit.get("report_receipt_hashes_match") is True,
                audit.get("replay", {}).get("all_verified") is True,
                (
                    audit.get("known_horizon_visible") is True
                    and audit.get("final_synthesis_present") is True
                    and audit.get("final_recommendation_validation_matches") is True
                    and audit.get("atomic_executor_verified") is True
                    if report.get("recommendation_stage_present") is True
                    else True
                ),
            )
        )

        declared_by_cell_id = {
            str(item["cell_id"]): item.get("score")
            for item in audit.get("world_understanding", {}).get("cells", [])
            if item.get("status") == "scored"
        }
        cells = []
        prompt_estimates: list[int] = []
        for cell in report["cells"]:
            scores = [float(value) for value in cell["scores"]]
            if not scores:
                raise ValueError(f"S0 cell has no scores: {root}")
            cell_id = cell["cell"].get("cell_id")
            row: dict[str, Any] = {
                "seed": seed,
                "task_id": str(cell["cell"]["task_id"]),
                "cell_status": str(cell["cell_status"]),
                "completed_experiment_count": int(cell["completed_experiment_count"]),
                "scores": scores,
                "best_so_far_scores": _best_so_far(scores),
                "first_score": scores[0],
                "last_score": scores[-1],
                "best_score": max(scores),
                "best_experiment_index": scores.index(max(scores)),
                "last_minus_first_score": scores[-1] - scores[0],
                "best_minus_first_score": max(scores) - scores[0],
                "recommendation_type": (
                    cell["final_synthesis"]["recommendation"][
                        "recommendation_type"
                    ]
                    if cell.get("final_synthesis") is not None
                    else None
                ),
                "validated_recommendation_score": (
                    float(
                        cell["validation"][
                            "primary_validated_recommendation_score_mean"
                        ]
                    )
                    if cell.get("validation") is not None
                    else None
                ),
                "validated_incumbent_score": (
                    float(cell["validation"]["validated_incumbent_score_mean"])
                    if cell.get("validation") is not None
                    else None
                ),
                "recommendation_gain_over_incumbent": (
                    float(
                        cell["validation"][
                            "recommendation_gain_over_incumbent_mean"
                        ]
                    )
                    if cell.get("validation") is not None
                    else None
                ),
                "predictive_directional_accuracy": (
                    float(cell["predictive_validation"]["score"]["directional_accuracy"])
                    if cell.get("predictive_validation") is not None
                    else None
                ),
                "predictive_confidence_brier_score": (
                    float(
                        cell["predictive_validation"]["score"][
                            "confidence_brier_score"
                        ]
                    )
                    if cell.get("predictive_validation") is not None
                    else None
                ),
                "predictive_nontrivial_actual_effect_rate": (
                    float(
                        cell["predictive_validation"]["score"][
                            "nontrivial_actual_effect_rate"
                        ]
                    )
                    if cell.get("predictive_validation") is not None
                    else None
                ),
                "declared_world_understanding": declared_by_cell_id.get(
                    str(cell_id)
                ) if cell_id is not None else None,
            }
            prompt_estimates.extend(
                int(experiment["decision_audit"]["prompt_estimated_tokens"])
                for experiment in cell["experiments"]
                if experiment.get("decision_audit") is not None
            )
            cells.append(row)
            task_rows.setdefault(row["task_id"], []).append(row)

        run_rows.append(
            {
                "seed": seed,
                "run_root": str(root),
                "report_sha256": _sha256(report_path),
                "audit_sha256": _sha256(audit_path),
                "protocol_sha256": str(report["protocol_sha256"]),
                "protocol_id": str(report["protocol_id"]),
                "completed_cell_count": int(report["completed_cell_count"]),
                "cell_count": int(report["cell_count"]),
                "completed_experiment_count": int(report["completed_experiment_count"]),
                "planned_experiment_count": int(report["planned_experiment_count"]),
                "provider_call_count": int(report["provider_call_count"]),
                "provider_attempt_count": int(report["provider_attempt_count"]),
                "provider_reported_total_tokens": int(
                    report["provider_reported_total_tokens"]
                ),
                "recommendation_stage_present": bool(
                    report.get("recommendation_stage_present", False)
                ),
                "completed_synthesis_call_count": int(
                    report.get("completed_synthesis_call_count", 0)
                ),
                "completed_validation_experiment_count": int(
                    report.get("completed_validation_experiment_count", 0)
                ),
                "total_physical_experiment_count": int(
                    report.get(
                        "total_physical_experiment_count",
                        report["completed_experiment_count"],
                    )
                ),
                "accounting_complete": bool(report["accounting_complete"]),
                "audit_passed": audit_passed,
                "max_prompt_estimated_tokens": max(prompt_estimates)
                if prompt_estimates
                else None,
                "cells": cells,
            }
        )

    if (
        len(method_hashes) != 1
        or len(method_ids) != 1
        or len(provider_modes) != 1
        or len(protocol_ids) != 1
        or len(formal_flags) != 1
        or len(benchmark_claim_flags) != 1
    ):
        raise ValueError("S0 runs do not share one frozen protocol, method, and provider")

    run_rows.sort(key=lambda item: item["seed"])
    aggregated_tasks = []
    for task_id, rows in sorted(task_rows.items()):
        rows.sort(key=lambda item: item["seed"])
        score_curves = [row["scores"] for row in rows]
        best_curves = [row["best_so_far_scores"] for row in rows]
        aggregated_tasks.append(
            {
                "task_id": task_id,
                "seed_count": len(rows),
                "seeds": [row["seed"] for row in rows],
                "first_score": _summary([row["first_score"] for row in rows]),
                "last_score": _summary([row["last_score"] for row in rows]),
                "best_score": _summary([row["best_score"] for row in rows]),
                "last_minus_first_score": _summary(
                    [row["last_minus_first_score"] for row in rows]
                ),
                "best_minus_first_score": _summary(
                    [row["best_minus_first_score"] for row in rows]
                ),
                "best_experiment_index": _summary(
                    [row["best_experiment_index"] for row in rows]
                ),
                "validated_recommendation_score": _summary(
                    [
                        row["validated_recommendation_score"]
                        for row in rows
                        if row["validated_recommendation_score"] is not None
                    ]
                ),
                "validated_incumbent_score": _summary(
                    [
                        row["validated_incumbent_score"]
                        for row in rows
                        if row["validated_incumbent_score"] is not None
                    ]
                ),
                "recommendation_gain_over_incumbent": _summary(
                    [
                        row["recommendation_gain_over_incumbent"]
                        for row in rows
                        if row["recommendation_gain_over_incumbent"] is not None
                    ]
                ),
                "recommendation_type_counts": {
                    recommendation_type: sum(
                        row["recommendation_type"] == recommendation_type
                        for row in rows
                    )
                    for recommendation_type in (
                        "tested",
                        "interpolated",
                        "extrapolated",
                    )
                },
                "predictive_directional_accuracy": _summary(
                    [
                        row["predictive_directional_accuracy"]
                        for row in rows
                        if row["predictive_directional_accuracy"] is not None
                    ]
                ),
                "predictive_confidence_brier_score": _summary(
                    [
                        row["predictive_confidence_brier_score"]
                        for row in rows
                        if row["predictive_confidence_brier_score"] is not None
                    ]
                ),
                "predictive_nontrivial_actual_effect_rate": _summary(
                    [
                        row["predictive_nontrivial_actual_effect_rate"]
                        for row in rows
                        if row["predictive_nontrivial_actual_effect_rate"] is not None
                    ]
                ),
                "declared_world_understanding": {
                    metric: _summary(
                        [
                            float(row["declared_world_understanding"][metric])
                            for row in rows
                            if row["declared_world_understanding"] is not None
                        ]
                    )
                    for metric in (
                        "structural_edge_f1",
                        "directional_accuracy",
                        "mechanism_tag_f1",
                        "unsupported_claim_rate",
                        "confidence_brier_score",
                    )
                },
                "score_curve": _curve_summary(score_curves),
                "best_so_far_curve": _curve_summary(best_curves),
                "seed_rows": rows,
            }
        )

    all_audits_passed = all(row["audit_passed"] for row in run_rows)
    all_runs_completed = all(
        row["completed_cell_count"] == row["cell_count"]
        and row["completed_experiment_count"] == row["planned_experiment_count"]
        for row in run_rows
    )
    formal_result = next(iter(formal_flags))
    benchmark_claim_allowed = next(iter(benchmark_claim_flags))
    return {
        "schema_version": STATIC_OPTIMIZATION_MULTISEED_VERSION,
        "formal_result": formal_result,
        "benchmark_claim_allowed": benchmark_claim_allowed,
        "formal_optimization_estimand": formal_result,
        "development_only": not formal_result,
        "last_score_is_final_recommendation": False,
        "last_score_semantics": "last_executed_trial_only",
        "method_id": next(iter(method_ids)),
        "method_config_sha256": next(iter(method_hashes)),
        "protocol_id": next(iter(protocol_ids)),
        "provider_mode": next(iter(provider_modes)),
        "recommendation_stage_present": all(
            row["recommendation_stage_present"] for row in run_rows
        ),
        "primary_metric": (
            "validated_final_recommendation_score_mean"
            if all(row["recommendation_stage_present"] for row in run_rows)
            else "exploration_descriptives"
        ),
        "seed_count": len(run_rows),
        "seeds": [row["seed"] for row in run_rows],
        "run_count": len(run_rows),
        "all_runs_completed": all_runs_completed,
        "all_audits_passed": all_audits_passed,
        "completed_cell_count": sum(row["completed_cell_count"] for row in run_rows),
        "planned_cell_count": sum(row["cell_count"] for row in run_rows),
        "completed_experiment_count": sum(
            row["completed_experiment_count"] for row in run_rows
        ),
        "planned_experiment_count": sum(
            row["planned_experiment_count"] for row in run_rows
        ),
        "provider_call_count": sum(row["provider_call_count"] for row in run_rows),
        "provider_attempt_count": sum(
            row["provider_attempt_count"] for row in run_rows
        ),
        "provider_reported_total_tokens": sum(
            row["provider_reported_total_tokens"] for row in run_rows
        ),
        "completed_synthesis_call_count": sum(
            row["completed_synthesis_call_count"] for row in run_rows
        ),
        "completed_validation_experiment_count": sum(
            row["completed_validation_experiment_count"] for row in run_rows
        ),
        "total_physical_experiment_count": sum(
            row["total_physical_experiment_count"] for row in run_rows
        ),
        "accounting_complete": all(row["accounting_complete"] for row in run_rows),
        "known_billed_cost_usd": None,
        "tasks": aggregated_tasks,
        "runs": run_rows,
        "interpretation": (
            "Multi-seed S0 fixed-world summary for one frozen method and declared protocol. "
            "These statistics characterize the sampled world seeds; they do not estimate a "
            "provider causal effect or broad generalization. The last score is the final "
            "executed exploration trial, not the submitted final recommendation."
        ),
    }


__all__ = [
    "STATIC_OPTIMIZATION_MULTISEED_VERSION",
    "aggregate_static_optimization_runs",
]
