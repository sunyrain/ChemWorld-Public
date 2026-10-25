"""Pure design and gates for Work II mechanism-oracle qualification."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import qmc

MECHANISM_ORACLE_VERSION = "chemworld-work-ii-mechanism-oracle-qualification-0.2"
INITIAL_POPULATION_SIZE = 128
OPTIMIZER_GENERATIONS = 20
EXPECTED_OPTIMIZER_REQUESTS = INITIAL_POPULATION_SIZE * (OPTIMIZER_GENERATIONS + 1)
TARGET_GRID_SIDE = 9
FULL_PERTURBATION_COUNT = 64
VALIDATION_CANDIDATE_COUNT = 8
VALIDATION_REPLICATES = 3
VALIDATION_EXECUTION_COUNT = VALIDATION_CANDIDATE_COUNT * VALIDATION_REPLICATES


def deterministic_oracle_seed(task_id: str, world_seed: int) -> int:
    digest = hashlib.sha256(f"work-ii-mechanism-oracle:{task_id}:{world_seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def balanced_initial_population(
    task_id: str,
    world_seed: int,
    schema: Sequence[Mapping[str, Any]],
) -> list[list[float]]:
    dimension = len(schema)
    if dimension < 2:
        raise ValueError("mechanism-oracle design requires at least two coordinates")
    population = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=deterministic_oracle_seed(task_id, world_seed),
    ).random_base2(m=7)
    categorical = [
        (int(item["coordinate"]), int(item["category_count"]))
        for item in schema
        if item.get("kind") == "categorical"
    ]
    combination_count = math.prod(count for _, count in categorical) or 1
    for row_index, vector in enumerate(population):
        combination = row_index % combination_count
        for coordinate, count in categorical:
            category = combination % count
            combination //= count
            vector[coordinate] = (category + 0.5) / count
    return population.astype(float).tolist()


def local_target_grid(
    optimum: Sequence[float],
    target_indices: tuple[int, int],
) -> list[dict[str, Any]]:
    offsets = np.linspace(-0.16, 0.16, TARGET_GRID_SIDE)
    rows: list[dict[str, Any]] = []
    for i_value, x_offset in enumerate(offsets):
        for j_value, y_offset in enumerate(offsets):
            vector = [float(value) for value in optimum]
            vector[target_indices[0]] = float(
                np.clip(vector[target_indices[0]] + x_offset, 0.02, 0.98)
            )
            vector[target_indices[1]] = float(
                np.clip(vector[target_indices[1]] + y_offset, 0.02, 0.98)
            )
            rows.append(
                {
                    "vector": vector,
                    "grid_i": i_value,
                    "grid_j": j_value,
                    "offset": [float(x_offset), float(y_offset)],
                }
            )
    return rows


def full_dimensional_perturbations(
    task_id: str,
    world_seed: int,
    optimum: Sequence[float],
    schema: Sequence[Mapping[str, Any]],
) -> list[list[float]]:
    dimension = len(schema)
    design = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=deterministic_oracle_seed(task_id + ":local", world_seed),
    ).random_base2(m=6)
    categorical = {int(item["coordinate"]) for item in schema if item.get("kind") == "categorical"}
    output: list[list[float]] = []
    for raw in design:
        vector = [float(value) for value in optimum]
        for coordinate in range(dimension):
            if coordinate in categorical:
                continue
            vector[coordinate] = float(
                np.clip(vector[coordinate] + 0.16 * (raw[coordinate] - 0.5), 0.02, 0.98)
            )
        output.append(vector)
    return output


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def select_validation_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int = VALIDATION_CANDIDATE_COUNT,
    minimum_distance: float = 0.08,
) -> list[dict[str, Any]]:
    eligible = sorted(
        (row for row in rows if row.get("status") == "completed" and bool(row.get("safe"))),
        key=lambda row: (-float(row["score"]), str(row.get("evaluation_id", ""))),
    )
    selected: list[dict[str, Any]] = []
    for row in eligible:
        vector = [float(value) for value in row["vector"]]
        if (
            selected
            and min(_distance(vector, item["vector"]) for item in selected) < minimum_distance
        ):
            continue
        selected.append(
            {
                "candidate_rank": len(selected) + 1,
                "source_evaluation_id": str(row.get("evaluation_id", "")),
                "vector": vector,
                "oracle_score": float(row["score"]),
                "oracle_risk": float(row["safety_risk"]),
            }
        )
        if len(selected) == count:
            break
    return selected


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return dict.fromkeys(
            ("count", "minimum", "p10", "median", "p90", "maximum", "mean", "sample_sd"),
            None,
        ) | {"count": 0}
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)) if array.size > 1 else None,
    }


def _pooled_validation_sigma(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[float]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        groups.setdefault(int(row["candidate_rank"]), []).append(float(row["score"]))
    variances = [
        float(np.var(values, ddof=1))
        for values in groups.values()
        if len(values) == VALIDATION_REPLICATES
    ]
    return {
        "complete_group_count": len(variances),
        "planned_group_count": VALIDATION_CANDIDATE_COUNT,
        "sigma": math.sqrt(sum(variances) / len(variances)) if variances else None,
    }


def _grid_structure(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_indices: tuple[int, int],
    optimum_score: float,
    tolerance: float,
    safety_limit: float,
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_index = {(int(row["grid_i"]), int(row["grid_j"])): row for row in completed}
    relative = [
        row
        for row in completed
        if bool(row.get("safe")) and float(row["score"]) >= optimum_score - tolerance
    ]
    x_values = [float(row["vector"][target_indices[0]]) for row in relative]
    y_values = [float(row["vector"][target_indices[1]]) for row in relative]
    x_span = max(x_values) - min(x_values) if x_values else 0.0
    y_span = max(y_values) - min(y_values) if y_values else 0.0
    interior = any(
        0.10 <= float(row["vector"][target_indices[0]]) <= 0.90
        and 0.10 <= float(row["vector"][target_indices[1]]) <= 0.90
        for row in relative
    )

    slope = 0.0
    curvature = 0.0
    for fixed in range(TARGET_GRID_SIDE):
        x_line = [by_index.get((index, fixed)) for index in range(TARGET_GRID_SIDE)]
        y_line = [by_index.get((fixed, index)) for index in range(TARGET_GRID_SIDE)]
        for line in (x_line, y_line):
            if all(item is not None for item in line):
                scores = [float(item["score"]) for item in line if item is not None]
                slope = max(slope, max(scores) - min(scores))
                for index in range(1, len(scores) - 1):
                    curvature = max(
                        curvature,
                        abs((scores[index - 1] + scores[index + 1]) / 2.0 - scores[index]),
                    )
    corners = [
        by_index.get((0, 0)),
        by_index.get((0, TARGET_GRID_SIDE - 1)),
        by_index.get((TARGET_GRID_SIDE - 1, 0)),
        by_index.get((TARGET_GRID_SIDE - 1, TARGET_GRID_SIDE - 1)),
    ]
    interaction = 0.0
    if all(item is not None for item in corners):
        ll, lh, hl, hh = (float(item["score"]) for item in corners if item is not None)
        interaction = abs(((hh + ll) - (hl + lh)) / 2.0)
    risks = [float(row["safety_risk"]) for row in completed]
    safety_crossing = bool(risks) and min(risks) < safety_limit <= max(risks)
    return {
        "completed_count": len(completed),
        "relative_high_quality_count": len(relative),
        "target_x_span": x_span,
        "target_y_span": y_span,
        "relative_high_quality_has_interior": interior,
        "directional_slope_effect": slope,
        "curvature_effect": curvature,
        "interaction_effect": interaction,
        "safety_frontier_crossing": safety_crossing,
    }


def analyze_mechanism_oracle_world(
    mechanism_rows: Sequence[Mapping[str, Any]],
    target_grid_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    optimizer_request_count: int,
    optimizer_generation_count: int,
    task_threshold: float,
    safety_limit: float,
    primary_metric: str,
    target_indices: tuple[int, int],
    require_safety_frontier: bool,
) -> dict[str, Any]:
    completed = [row for row in mechanism_rows if row.get("status") == "completed"]
    physical_failures = [
        row for row in mechanism_rows if row.get("status") == "physical_failure"
    ]
    classified = [
        row
        for row in mechanism_rows
        if row.get("status") in {"completed", "physical_failure"}
    ]
    unclassified = [
        row
        for row in mechanism_rows
        if row.get("status") not in {"completed", "physical_failure"}
    ]
    safe = [row for row in completed if bool(row.get("safe"))]
    optimum = max(safe, key=lambda row: float(row["score"])) if safe else None
    optimum_score = float(optimum["score"]) if optimum is not None else None
    score_distribution = _distribution([float(row["score"]) for row in safe])
    primary_distribution = _distribution([float(row["metrics"][primary_metric]) for row in safe])
    validation_noise = _pooled_validation_sigma(validation_rows)
    sigma = float(validation_noise["sigma"] or 0.0)
    tolerance = max(0.05, 6.0 * sigma)
    dynamic_range = (
        float(score_distribution["p90"]) - float(score_distribution["p10"])
        if score_distribution["p90"] is not None and score_distribution["p10"] is not None
        else None
    )
    primary_range = (
        float(primary_distribution["maximum"]) - float(primary_distribution["minimum"])
        if primary_distribution["maximum"] is not None
        and primary_distribution["minimum"] is not None
        else None
    )
    grid = _grid_structure(
        target_grid_rows,
        target_indices=target_indices,
        optimum_score=float(optimum_score or 0.0),
        tolerance=tolerance,
        safety_limit=safety_limit,
    )
    structure_effect_gate = max(0.03, 6.0 * sigma)
    local_structure = max(
        float(grid["directional_slope_effect"]),
        float(grid["curvature_effect"]),
        float(grid["interaction_effect"]),
    ) >= structure_effect_gate or bool(grid["safety_frontier_crossing"])
    validation_completed = [row for row in validation_rows if row.get("status") == "completed"]
    validation_exact = [
        row
        for row in validation_completed
        if isinstance(row.get("exact_replay"), Mapping)
        and row["exact_replay"].get("verified") is True
    ]
    optimum_validation = [
        float(row["score"]) for row in validation_completed if int(row["candidate_rank"]) == 1
    ]
    optimum_observed_median = (
        float(np.median(optimum_validation))
        if len(optimum_validation) == VALIDATION_REPLICATES
        else None
    )
    frontier = [row for row in completed if 0.30 <= float(row["safety_risk"]) <= 0.40]
    threshold_reaching = [row for row in safe if float(row["score"]) >= task_threshold]
    completion_fraction = len(completed) / len(mechanism_rows) if mechanism_rows else 0.0
    classification_fraction = (
        len(classified) / len(mechanism_rows) if mechanism_rows else 0.0
    )
    checks = {
        "optimizer_requests_complete": optimizer_request_count == EXPECTED_OPTIMIZER_REQUESTS,
        "optimizer_generations_complete": optimizer_generation_count == OPTIMIZER_GENERATIONS,
        "all_mechanism_outcomes_classified": len(classified) == len(mechanism_rows),
        "all_observed_validation_completed": len(validation_completed)
        == VALIDATION_EXECUTION_COUNT,
        "all_observed_validation_exact_replay": len(validation_exact) == VALIDATION_EXECUTION_COUNT,
        "validation_candidate_groups": validation_noise["complete_group_count"] >= 7,
        "safe_oracle_optimum_exists": optimum is not None,
        "observed_optimum_agreement": optimum_score is not None
        and optimum_observed_median is not None
        and abs(optimum_observed_median - optimum_score) <= tolerance,
        "safe_dynamic_range": dynamic_range is not None and dynamic_range >= max(0.10, 6.0 * sigma),
        "primary_metric_range": primary_range is not None and primary_range >= 0.10,
        "relative_high_quality_basin": grid["relative_high_quality_count"] >= 5
        and grid["target_x_span"] >= 0.05
        and grid["target_y_span"] >= 0.05
        and grid["relative_high_quality_has_interior"],
        "local_law_identifiable": local_structure,
        "safety_frontier_coverage": (
            len(frontier) >= math.ceil(0.05 * len(completed))
            and optimum is not None
            and float(optimum["safety_risk"]) < safety_limit
            if require_safety_frontier
            else True
        ),
    }
    return {
        "mechanism_evaluation_count": len(mechanism_rows),
        "mechanism_completed_count": len(completed),
        "mechanism_physical_failure_count": len(physical_failures),
        "mechanism_unclassified_count": len(unclassified),
        "mechanism_failed_count": len(unclassified),
        "mechanism_completion_fraction": completion_fraction,
        "mechanism_classified_count": len(classified),
        "mechanism_classification_fraction": classification_fraction,
        "optimizer_request_count": optimizer_request_count,
        "optimizer_generation_count": optimizer_generation_count,
        "safe_mechanism_count": len(safe),
        "historical_threshold_reaching_count": len(threshold_reaching),
        "historical_task_threshold": task_threshold,
        "safe_score_distribution": score_distribution,
        "primary_metric": primary_metric,
        "primary_metric_distribution": primary_distribution,
        "safe_dynamic_range_p90_p10": dynamic_range,
        "primary_metric_range": primary_range,
        "validation_noise": validation_noise,
        "relative_high_quality_tolerance": tolerance,
        "structure_effect_gate": structure_effect_gate,
        "oracle_optimum": (
            None
            if optimum is None
            else {
                "evaluation_id": str(optimum.get("evaluation_id", "")),
                "vector": [float(value) for value in optimum["vector"]],
                "score": float(optimum["score"]),
                "safety_risk": float(optimum["safety_risk"]),
                "metrics": dict(optimum["metrics"]),
            }
        ),
        "optimum_observed_median_score": optimum_observed_median,
        "target_grid": grid,
        "local_law_identifiable": local_structure,
        "safety_frontier_count": len(frontier),
        "observed_validation_completed_count": len(validation_completed),
        "observed_validation_exact_replay_count": len(validation_exact),
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "EXPECTED_OPTIMIZER_REQUESTS",
    "FULL_PERTURBATION_COUNT",
    "INITIAL_POPULATION_SIZE",
    "MECHANISM_ORACLE_VERSION",
    "OPTIMIZER_GENERATIONS",
    "TARGET_GRID_SIDE",
    "VALIDATION_CANDIDATE_COUNT",
    "VALIDATION_EXECUTION_COUNT",
    "VALIDATION_REPLICATES",
    "analyze_mechanism_oracle_world",
    "balanced_initial_population",
    "deterministic_oracle_seed",
    "full_dimensional_perturbations",
    "local_target_grid",
    "select_validation_candidates",
]
