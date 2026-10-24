"""Paired-seed validity and prospective-power diagnostics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from statistics import NormalDist, fmean
from typing import Any

import numpy as np

from chemworld.agents.task_recipes import (
    task_recipe_dimension,
    task_recipe_event_count,
)
from chemworld.data.records import trajectory_value
from chemworld.tasks import SERIOUS_TASK_IDS, get_task

VALIDITY_POWER_SCHEMA_VERSION = "chemworld-validity-power-audit-0.1"
DEFAULT_METHOD_PAIRS = (
    ("gp_bo", "random"),
    ("safe_gp_bo", "random"),
    ("lhs", "random"),
    ("scripted_chemistry", "random"),
    ("tool_using_llm_stub", "random"),
)
ADAPTIVE_METHOD_PAIRS = (
    ("gp_bo", "random"),
    ("safe_gp_bo", "random"),
)


def campaign_record_prefix(
    records: Sequence[dict[str, Any]],
    complete_experiments: int,
) -> list[dict[str, Any]]:
    """Return records through the requested successful final assay."""

    if complete_experiments < 1:
        raise ValueError("complete_experiments must be positive")
    terminal_indices = [
        index
        for index, record in enumerate(records)
        if trajectory_value(record, 'leaderboard_score', None) is not None
    ]
    if len(terminal_indices) < complete_experiments:
        raise ValueError(
            f"Campaign has {len(terminal_indices)} complete experiments; "
            f"requested {complete_experiments}"
        )
    return [dict(record) for record in records[: terminal_indices[complete_experiments - 1] + 1]]


def minimum_learning_capacity(task_info: dict[str, Any]) -> int:
    """Return the minimum complete experiments for a learning diagnostic."""

    return max(8, task_recipe_dimension(task_info) + 2)


def calibrated_validity_budget(task_info: dict[str, Any]) -> int:
    """Return a step budget that admits the minimum complete experiments."""

    return task_recipe_event_count(task_info) * minimum_learning_capacity(task_info)


def audit_validity_power(
    results: Sequence[dict[str, Any]],
    *,
    task_ids: Sequence[str] = SERIOUS_TASK_IDS,
    method_pairs: Sequence[tuple[str, str]] = DEFAULT_METHOD_PAIRS,
    adaptive_method_pairs: Sequence[tuple[str, str]] = ADAPTIVE_METHOD_PAIRS,
    metric: str = "total_score",
    practical_effect: float = 0.05,
    alpha: float = 0.05,
    target_power: float = 0.80,
    planned_seed_count: int = 20,
    bootstrap_samples: int = 5_000,
) -> dict[str, Any]:
    """Audit task discrimination using paired task-seed observations.

    The best-known reference is deliberately descriptive: it is the strongest
    observed method on each paired seed. It is not a mathematical oracle and
    must not be used as an immutable upper bound.
    """

    if not 0.0 < practical_effect <= 1.0:
        raise ValueError("practical_effect must be in (0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if planned_seed_count < 2:
        raise ValueError("planned_seed_count must be at least two")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")

    index = _index_results(results, metric=metric)
    reports: dict[str, dict[str, Any]] = {}
    compact_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        if task_id not in index:
            raise ValueError(f"No results for task {task_id!r}")
        task = get_task(task_id)
        methods = index[task_id]
        common_seeds = sorted(set.intersection(*(set(rows) for rows in methods.values())))
        if len(common_seeds) < 2:
            raise ValueError(f"Task {task_id!r} has fewer than two paired seeds")

        method_reports: dict[str, dict[str, Any]] = {}
        for method, by_seed in sorted(methods.items()):
            paired_rows = [by_seed[seed] for seed in common_seeds]
            values = np.asarray([float(row[metric]) for row in paired_rows], dtype=float)
            method_reports[method] = {
                "seed_count": len(common_seeds),
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "success_rate": float(
                    np.mean(
                        [row.get("sample_efficiency_step") is not None for row in paired_rows]
                    )
                ),
                "mean_bo_acquisition_recipe_count": float(
                    np.mean(
                        [
                            float(row.get("bo_acquisition_recipe_count", 0.0))
                            for row in paired_rows
                        ]
                    )
                ),
            }
            compact_rows.extend(
                _compact_seed_row(task_id, method, seed, by_seed[seed])
                for seed in common_seeds
            )

        comparisons: dict[str, dict[str, Any]] = {}
        for method_a, method_b in method_pairs:
            if method_a not in methods or method_b not in methods:
                continue
            differences = np.asarray(
                [
                    float(methods[method_a][seed][metric])
                    - float(methods[method_b][seed][metric])
                    for seed in common_seeds
                ],
                dtype=float,
            )
            comparisons[f"{method_a}__minus__{method_b}"] = _paired_effect(
                differences,
                identity=f"{task_id}:{method_a}:{method_b}:{metric}",
                practical_effect=practical_effect,
                alpha=alpha,
                target_power=target_power,
                bootstrap_samples=bootstrap_samples,
            )

        best_known = {
            seed: max(float(by_seed[seed][metric]) for by_seed in methods.values())
            for seed in common_seeds
        }
        best_known_gaps = {
            method: float(
                np.mean(
                    [best_known[seed] - float(by_seed[seed][metric]) for seed in common_seeds]
                )
            )
            for method, by_seed in sorted(methods.items())
        }
        method_means = [float(item["mean"]) for item in method_reports.values()]
        all_values = np.asarray(
            [float(by_seed[seed][metric]) for by_seed in methods.values() for seed in common_seeds],
            dtype=float,
        )
        dimension = task_recipe_dimension(task.to_dict())
        event_count = task_recipe_event_count(task.to_dict())
        evaluated_budgets = {
            int(
                row.get(
                    "evaluation_budget_steps",
                    row.get("diagnostic_budget_steps", task.budget),
                )
            )
            for by_seed in methods.values()
            for seed, row in by_seed.items()
            if seed in common_seeds
        }
        if len(evaluated_budgets) != 1:
            raise ValueError(f"Task {task_id!r} mixes evaluation budgets")
        evaluated_budget = next(iter(evaluated_budgets))
        capacity = evaluated_budget // event_count
        required_capacity = minimum_learning_capacity(task.to_dict())
        adaptive_effects = [
            float(comparisons[f"{a}__minus__{b}"]["mean_paired_effect"])
            for a, b in adaptive_method_pairs
            if f"{a}__minus__{b}" in comparisons
        ]
        adaptive_value_detected = any(effect >= practical_effect for effect in adaptive_effects)
        required_seed_counts = [
            int(item["required_seed_count_for_practical_effect"])
            for key, item in comparisons.items()
            if key in {f"{a}__minus__{b}" for a, b in adaptive_method_pairs}
        ]
        all_method_seeds = set.union(*(set(rows) for rows in methods.values()))
        checks = {
            "paired_seed_coverage": len(common_seeds) == len(all_method_seeds),
            "no_global_floor_or_ceiling_saturation": bool(
                np.mean(all_values <= 0.05) < 0.90 and np.mean(all_values >= 0.95) < 0.90
            ),
            "method_discrimination_at_sesoi": max(method_means) - min(method_means)
            >= practical_effect,
            "adaptive_value_at_sesoi": adaptive_value_detected,
            "learning_opportunity_adequate": capacity >= required_capacity,
            "planned_seed_depth_powered": bool(required_seed_counts)
            and max(required_seed_counts) <= planned_seed_count,
        }
        reports[task_id] = {
            "task_contract_hash": task.contract_hash,
            "paired_seeds": common_seeds,
            "method_count": len(methods),
            "experiment_opportunity": {
                "official_task_budget_steps": task.budget,
                "evaluated_budget_steps": evaluated_budget,
                "recipe_dimension": dimension,
                "events_per_complete_experiment": event_count,
                "complete_experiment_capacity": capacity,
                "minimum_learning_capacity": required_capacity,
            },
            "score_distribution": {
                "minimum": float(np.min(all_values)),
                "maximum": float(np.max(all_values)),
                "method_mean_spread": max(method_means) - min(method_means),
            },
            "methods": method_reports,
            "comparisons": comparisons,
            "best_known_reference": {
                "kind": "paired_seed_hindsight_best_observed",
                "is_oracle": False,
                "scores_by_seed": {str(seed): best_known[seed] for seed in common_seeds},
                "mean_gap_by_method": best_known_gaps,
            },
            "checks": checks,
            "validity_ready": all(checks.values()),
        }

    adaptive_ready_count = sum(
        bool(report["checks"]["adaptive_value_at_sesoi"]) for report in reports.values()
    )
    capacity_ready_count = sum(
        bool(report["checks"]["learning_opportunity_adequate"])
        for report in reports.values()
    )
    _apply_holm_correction(reports, alpha=alpha)
    return {
        "schema_version": VALIDITY_POWER_SCHEMA_VERSION,
        "status": (
            "validity_ready"
            if all(report["validity_ready"] for report in reports.values())
            else "blocked"
        ),
        "metric": metric,
        "practical_effect_threshold": practical_effect,
        "alpha": alpha,
        "target_power": target_power,
        "planned_seed_count": planned_seed_count,
        "adaptive_method_pairs": [list(pair) for pair in adaptive_method_pairs],
        "paired_design_required": True,
        "multiple_comparison_policy": "holm_within_comparison_family",
        "oracle_policy": {
            "random_sample_maximum_is_oracle": False,
            "reported_reference": "paired_seed_hindsight_best_observed",
            "future_method_may_exceed_reference": True,
        },
        "task_count": len(reports),
        "adaptive_value_task_count": adaptive_ready_count,
        "learning_capacity_task_count": capacity_ready_count,
        "tasks": reports,
        "compact_seed_metrics": compact_rows,
    }


def _apply_holm_correction(
    reports: dict[str, dict[str, Any]],
    *,
    alpha: float,
) -> None:
    comparison_keys = sorted(
        {
            key
            for report in reports.values()
            for key in report["comparisons"]
        }
    )
    for comparison_key in comparison_keys:
        family = [
            (task_id, float(report["comparisons"][comparison_key]["sign_flip_p_value"]))
            for task_id, report in reports.items()
            if comparison_key in report["comparisons"]
        ]
        adjusted = _holm_adjusted_p_values(family)
        for task_id, adjusted_p_value in adjusted.items():
            comparison = reports[task_id]["comparisons"][comparison_key]
            comparison["holm_adjusted_p_value"] = adjusted_p_value
            comparison["significant_after_holm"] = adjusted_p_value <= alpha
            comparison["multiplicity_family_size"] = len(family)


def _holm_adjusted_p_values(family: list[tuple[str, float]]) -> dict[str, float]:
    if not family:
        return {}
    ordered = sorted(family, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running_max = 0.0
    adjusted: dict[str, float] = {}
    for rank, (identity, p_value) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * p_value)
        adjusted[identity] = min(1.0, running_max)
    return adjusted


def _index_results(
    results: Sequence[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    index: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for result in results:
        task_id = str(result.get("task_id") or result.get("benchmark_task_id") or "")
        method = str(result.get("baseline_agent") or result.get("agent_name") or "")
        if not task_id or not method or "seed" not in result or metric not in result:
            raise ValueError("Each result needs task, method, seed, and requested metric")
        seed = int(result["seed"])
        if seed in index[task_id][method]:
            raise ValueError(f"Duplicate task-method-seed result: {task_id}/{method}/{seed}")
        value = float(result[metric])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite metric for {task_id}/{method}/{seed}")
        index[task_id][method][seed] = result
    return {
        task: {method: dict(rows) for method, rows in methods.items()}
        for task, methods in index.items()
    }


def _paired_effect(
    differences: np.ndarray,
    *,
    identity: str,
    practical_effect: float,
    alpha: float,
    target_power: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    count = int(differences.size)
    if count < 2:
        raise ValueError("Paired effect requires at least two differences")
    mean_effect = float(np.mean(differences))
    sample_std = float(np.std(differences, ddof=1))
    encoded = json.dumps(
        {"identity": identity, "differences": differences.tolist()},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big"))
    samples = rng.choice(differences, size=(bootstrap_samples, count), replace=True)
    means = np.mean(samples, axis=1)
    lower, upper = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0))
    p_value = _sign_flip_p_value(differences, rng=rng)
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(target_power)
    required = max(
        2,
        math.ceil(((z_alpha + z_power) * sample_std / practical_effect) ** 2),
    )
    tolerance = 1.0e-12
    return {
        "seed_count": count,
        "mean_paired_effect": mean_effect,
        "sample_std_paired_effect": sample_std,
        "paired_bootstrap_ci": [float(lower), float(upper)],
        "sign_flip_p_value": p_value,
        "wins": int(np.sum(differences > tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "losses": int(np.sum(differences < -tolerance)),
        "cohen_dz": None if sample_std <= tolerance else mean_effect / sample_std,
        "required_seed_count_for_practical_effect": required,
        "power_approximation": "two-sided paired normal approximation",
    }


def _sign_flip_p_value(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(np.mean(differences)))
    count = int(differences.size)
    if count <= 16:
        statistics = (
            abs(fmean(sign * value for sign, value in zip(signs, differences, strict=True)))
            for signs in itertools.product((-1.0, 1.0), repeat=count)
        )
        total = 2**count
        extreme = sum(value >= observed - 1.0e-15 for value in statistics)
        return extreme / total
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(20_000, count), replace=True)
    statistics = np.abs(np.mean(signs * differences, axis=1))
    return float((np.sum(statistics >= observed - 1.0e-15) + 1) / (statistics.size + 1))


def _compact_seed_row(
    task_id: str,
    method: str,
    seed: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    keys: Iterable[str] = (
        "total_score",
        "final_best_score",
        "area_under_best_score",
        "safety_aware_score",
        "cost_aware_score",
        "invalid_action_rate",
        "bo_initial_recipe_count",
        "bo_acquisition_recipe_count",
    )
    return {
        "task_id": task_id,
        "method": method,
        "seed": seed,
        **{key: result.get(key) for key in keys},
        "threshold_reached": result.get("sample_efficiency_step") is not None,
    }


__all__ = [
    "ADAPTIVE_METHOD_PAIRS",
    "DEFAULT_METHOD_PAIRS",
    "VALIDITY_POWER_SCHEMA_VERSION",
    "audit_validity_power",
    "calibrated_validity_budget",
    "campaign_record_prefix",
    "minimum_learning_capacity",
]
