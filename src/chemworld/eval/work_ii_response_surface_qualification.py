"""Deterministic design and gates for Work II Q1 response-surface qualification."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

WORK_II_Q1_RESPONSE_SURFACE_VERSION = "chemworld-work-ii-q1-response-surface-0.3"
BROAD_RECIPE_COUNT = 384
LOCAL_ANCHOR_COUNT = 28
LOCAL_RECIPES_PER_ANCHOR = 4
REPEAT_ANCHOR_COUNT = 8
REPEATS_PER_ANCHOR = 2
ADAPTIVE_RECIPE_COUNT = (
    LOCAL_ANCHOR_COUNT * LOCAL_RECIPES_PER_ANCHOR
    + REPEAT_ANCHOR_COUNT * REPEATS_PER_ANCHOR
)
TOTAL_RECIPE_COUNT = BROAD_RECIPE_COUNT + ADAPTIVE_RECIPE_COUNT
LOCAL_OFFSET = 0.04


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def deterministic_design_seed(task_id: str, world_seed: int) -> int:
    digest = hashlib.sha256(
        f"work-ii-q1-response-surface:{task_id}:{world_seed}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def broad_sobol_design(task_id: str, world_seed: int, dimension: int) -> list[list[float]]:
    if dimension < 2:
        raise ValueError("response-surface dimension must be at least two")
    design = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=deterministic_design_seed(task_id, world_seed),
    ).random_base2(m=9)
    return design[:BROAD_RECIPE_COUNT].astype(float).tolist()


def _categorical_signature(
    vector: Sequence[float], schema: Sequence[Mapping[str, Any]]
) -> tuple[int, ...]:
    signature: list[int] = []
    for coordinate in schema:
        if coordinate.get("kind") != "categorical":
            continue
        index = int(coordinate["coordinate"])
        count = int(coordinate["category_count"])
        signature.append(min(int(float(vector[index]) * count), count - 1))
    return tuple(signature)


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("recipe_id", ""))


def _score(row: Mapping[str, Any]) -> float:
    return _finite(row.get("score"), "row.score")


def _safety(row: Mapping[str, Any]) -> float:
    return _finite(row.get("safety_risk"), "row.safety_risk")


def _vector(row: Mapping[str, Any]) -> list[float]:
    value = row.get("vector")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("row.vector must be a list")
    return [_finite(item, "row.vector") for item in value]


def _interior(row: Mapping[str, Any], target_indices: tuple[int, int]) -> bool:
    vector = _vector(row)
    return all(0.10 <= vector[index] <= 0.90 for index in target_indices)


def _local_variation(
    row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: Sequence[Mapping[str, Any]],
    target_indices: tuple[int, int],
) -> float:
    vector = _vector(row)
    signature = _categorical_signature(vector, schema)
    distances: list[tuple[float, float, str]] = []
    for other in rows:
        if _row_id(other) == _row_id(row):
            continue
        other_vector = _vector(other)
        if _categorical_signature(other_vector, schema) != signature:
            continue
        distance = math.dist(
            [vector[index] for index in target_indices],
            [other_vector[index] for index in target_indices],
        )
        if distance > 0.0:
            distances.append((distance, abs(_score(row) - _score(other)), _row_id(other)))
    nearest = sorted(distances)[:8]
    if not nearest:
        return 0.0
    return max(delta / distance for distance, delta, _ in nearest)


def select_adaptive_anchors(
    broad_rows: Sequence[Mapping[str, Any]],
    *,
    schema: Sequence[Mapping[str, Any]],
    target_indices: tuple[int, int],
    task_threshold: float,
    safety_limit: float,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in broad_rows
        if row.get("status") == "completed"
        and _safety(row) <= safety_limit
        and _interior(row, target_indices)
    ]
    if len(candidates) < LOCAL_ANCHOR_COUNT:
        raise ValueError("broad design lacks 28 feasible interior adaptive anchors")
    variation = {
        _row_id(row): _local_variation(
            row,
            candidates,
            schema=schema,
            target_indices=target_indices,
        )
        for row in candidates
    }
    rankings = {
        "high_quality": sorted(candidates, key=lambda row: (-_score(row), _row_id(row))),
        "threshold_near": sorted(
            candidates,
            key=lambda row: (abs(_score(row) - task_threshold), _row_id(row)),
        ),
        "safety_frontier": sorted(
            candidates,
            key=lambda row: (abs(_safety(row) - safety_limit), _row_id(row)),
        ),
        "local_variation": sorted(
            candidates,
            key=lambda row: (-variation[_row_id(row)], _row_id(row)),
        ),
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for stratum, ranking in rankings.items():
        added = 0
        for row in ranking:
            recipe_id = _row_id(row)
            if recipe_id in used:
                continue
            selected.append(
                {
                    "anchor_id": f"anchor-{len(selected) + 1:02d}",
                    "source_recipe_id": recipe_id,
                    "stratum": stratum,
                    "vector": _vector(row),
                    "score": _score(row),
                    "safety_risk": _safety(row),
                    "local_variation": variation[recipe_id],
                }
            )
            used.add(recipe_id)
            added += 1
            if added == 7:
                break
        if added != 7:
            raise ValueError(f"adaptive anchor stratum is incomplete: {stratum}")
    return selected


def _offset_coordinate(value: float, sign: int) -> float:
    shifted = value + sign * LOCAL_OFFSET
    if shifted < 0.02:
        shifted = 0.02 + (0.02 - shifted)
    if shifted > 0.98:
        shifted = 0.98 - (shifted - 0.98)
    return min(max(shifted, 0.02), 0.98)


def build_adaptive_design(
    anchors: Sequence[Mapping[str, Any]],
    *,
    target_indices: tuple[int, int],
) -> list[dict[str, Any]]:
    if len(anchors) != LOCAL_ANCHOR_COUNT:
        raise ValueError("adaptive design requires exactly 28 anchors")
    designs: list[dict[str, Any]] = []
    signs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    for anchor in anchors:
        base = [float(item) for item in anchor["vector"]]
        for x_sign, y_sign in signs:
            vector = list(base)
            vector[target_indices[0]] = _offset_coordinate(
                vector[target_indices[0]], x_sign
            )
            vector[target_indices[1]] = _offset_coordinate(
                vector[target_indices[1]], y_sign
            )
            designs.append(
                {
                    "phase": "local_refinement",
                    "anchor_id": anchor["anchor_id"],
                    "source_recipe_id": anchor["source_recipe_id"],
                    "offset": [x_sign, y_sign],
                    "vector": vector,
                }
            )
    repeat_pool = [
        anchor
        for stratum in ("high_quality", "safety_frontier")
        for anchor in anchors
        if anchor["stratum"] == stratum
    ][:REPEAT_ANCHOR_COUNT]
    if len(repeat_pool) != REPEAT_ANCHOR_COUNT:
        raise ValueError("adaptive design lacks eight repeat anchors")
    for anchor in repeat_pool:
        for repeat_index in range(1, REPEATS_PER_ANCHOR + 1):
            designs.append(
                {
                    "phase": "noise_repeat",
                    "anchor_id": anchor["anchor_id"],
                    "source_recipe_id": anchor["source_recipe_id"],
                    "repeat_index": repeat_index,
                    "vector": [float(item) for item in anchor["vector"]],
                }
            )
    if len(designs) != ADAPTIVE_RECIPE_COUNT:
        raise RuntimeError("adaptive response-surface denominator drifted")
    return designs


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
            "mean": None,
            "sample_sd": None,
        }
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


def _pooled_noise_sigma(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    adaptive_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in adaptive_rows:
        if row.get("phase") != "noise_repeat" or row.get("status") != "completed":
            continue
        source = str(row["source_recipe_id"])
        groups.setdefault(source, [])
        value = _mapping_metric(row, metric)
        groups[source].append(value)
    variances: list[float] = []
    complete_groups = 0
    for source, repeated in sorted(groups.items()):
        original = rows_by_id.get(source)
        if original is None or original.get("status") != "completed" or len(repeated) != 2:
            continue
        values = [_mapping_metric(original, metric), *repeated]
        variances.append(float(np.var(np.asarray(values), ddof=1)))
        complete_groups += 1
    sigma = math.sqrt(sum(variances) / len(variances)) if variances else None
    return {
        "metric": metric,
        "planned_group_count": REPEAT_ANCHOR_COUNT,
        "complete_group_count": complete_groups,
        "sigma": sigma,
    }


def _mapping_metric(row: Mapping[str, Any], metric: str) -> float:
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    return _finite(metrics.get(metric), f"row.metrics.{metric}")


def _largest_component(vectors: Sequence[Sequence[float]], radius: float = 0.15) -> int:
    if not vectors:
        return 0
    array = np.asarray(vectors, dtype=float)
    tree = cKDTree(array)
    neighbors = tree.query_ball_tree(tree, r=radius)
    visited: set[int] = set()
    largest = 0
    for start in range(len(vectors)):
        if start in visited:
            continue
        stack = [start]
        size = 0
        visited.add(start)
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest


def _local_effects(
    broad_rows: Sequence[Mapping[str, Any]],
    adaptive_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    broad_by_id = {_row_id(row): row for row in broad_rows}
    grouped: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = {}
    anchor_source: dict[str, str] = {}
    for row in adaptive_rows:
        if row.get("phase") != "local_refinement" or row.get("status") != "completed":
            continue
        anchor_id = str(row["anchor_id"])
        offset = row.get("offset")
        if not isinstance(offset, Sequence) or len(offset) != 2:
            continue
        grouped.setdefault(anchor_id, {})[(int(offset[0]), int(offset[1]))] = row
        anchor_source[anchor_id] = str(row["source_recipe_id"])
    effects: list[dict[str, Any]] = []
    for anchor_id, corners in sorted(grouped.items()):
        required = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        source = broad_by_id.get(anchor_source[anchor_id])
        if set(corners) != required or source is None:
            continue
        mm = _score(corners[(-1, -1)])
        mp = _score(corners[(-1, 1)])
        pm = _score(corners[(1, -1)])
        pp = _score(corners[(1, 1)])
        gradient_x = ((pm + pp) - (mm + mp)) / 2.0
        gradient_y = ((mp + pp) - (mm + pm)) / 2.0
        interaction = ((pp + mm) - (pm + mp)) / 2.0
        effects.append(
            {
                "anchor_id": anchor_id,
                "source_recipe_id": anchor_source[anchor_id],
                "center_score": _score(source),
                "gradient_x_contrast": gradient_x,
                "gradient_y_contrast": gradient_y,
                "interaction_contrast": interaction,
                "maximum_effect": max(
                    abs(gradient_x), abs(gradient_y), abs(interaction)
                ),
            }
        )
    return effects


def analyze_q1_world(
    broad_rows: Sequence[Mapping[str, Any]],
    adaptive_rows: Sequence[Mapping[str, Any]],
    *,
    target_indices: tuple[int, int],
    task_threshold: float,
    safety_limit: float,
    primary_metric: str,
    require_safety_frontier: bool,
) -> dict[str, Any]:
    rows = [*broad_rows, *adaptive_rows]
    completed = [row for row in rows if row.get("status") == "completed"]
    exact = [
        row
        for row in completed
        if isinstance(row.get("exact_replay"), Mapping)
        and row["exact_replay"].get("verified") is True
    ]
    feasible = [row for row in completed if _safety(row) <= safety_limit]
    successful = [row for row in feasible if _score(row) >= task_threshold]
    scores = [_score(row) for row in completed]
    feasible_scores = [_score(row) for row in feasible]
    primary_values = [_mapping_metric(row, primary_metric) for row in feasible]
    score_distribution = _distribution(scores)
    feasible_distribution = _distribution(feasible_scores)
    primary_distribution = _distribution(primary_values)
    broad_by_id = {_row_id(row): row for row in broad_rows}
    score_noise = _pooled_noise_sigma(
        broad_by_id, adaptive_rows, metric="score"
    )
    primary_noise = _pooled_noise_sigma(
        broad_by_id, adaptive_rows, metric=primary_metric
    )
    score_sigma = score_noise["sigma"]
    primary_sigma = primary_noise["sigma"]
    score_effect_gate = max(0.10, 3.0 * float(score_sigma or 0.0))
    primary_effect_gate = max(0.10, 3.0 * float(primary_sigma or 0.0))
    sorted_by_score = sorted(completed, key=lambda row: (-_score(row), _row_id(row)))
    top_count = max(1, math.ceil(0.05 * len(sorted_by_score)))
    top_rows = sorted_by_score[:top_count]
    top_interior = any(_interior(row, target_indices) for row in top_rows)
    basin_size = _largest_component([_vector(row) for row in successful])
    effects = _local_effects(broad_rows, adaptive_rows)
    qualifying_effects = [
        row for row in effects if float(row["maximum_effect"]) >= score_effect_gate
    ]
    gradient_sign_reversal = any(
        any(float(row[field]) >= score_effect_gate for row in effects)
        and any(float(row[field]) <= -score_effect_gate for row in effects)
        for field in ("gradient_x_contrast", "gradient_y_contrast")
    )
    interaction_count = sum(
        abs(float(row["interaction_contrast"])) >= score_effect_gate
        for row in effects
    )
    frontier = [row for row in completed if 0.30 <= _safety(row) <= 0.40]
    dynamic_range = (
        float(feasible_distribution["p90"]) - float(feasible_distribution["p10"])
        if feasible_distribution["p90"] is not None
        and feasible_distribution["p10"] is not None
        else None
    )
    primary_range = (
        float(primary_distribution["maximum"])
        - float(primary_distribution["minimum"])
        if primary_distribution["maximum"] is not None
        and primary_distribution["minimum"] is not None
        else None
    )
    floor_fraction = (
        sum(score <= 1.0e-9 for score in scores) / len(scores) if scores else None
    )
    ceiling_fraction = (
        sum(score >= 1.0 - 1.0e-9 for score in scores) / len(scores)
        if scores
        else None
    )
    checks = {
        "all_512_completed": len(completed) == TOTAL_RECIPE_COUNT,
        "all_512_exact_replay": len(exact) == TOTAL_RECIPE_COUNT,
        "absolute_reachability": len(successful) >= max(5, math.ceil(0.01 * len(completed))),
        "dynamic_range": dynamic_range is not None and dynamic_range >= 0.15,
        "primary_metric_effect": primary_range is not None
        and primary_range >= primary_effect_gate,
        "floor_not_saturated": floor_fraction is not None and floor_fraction <= 0.20,
        "ceiling_not_saturated": ceiling_fraction is not None
        and ceiling_fraction <= 0.20,
        "top_region_has_target_interior": top_interior,
        "nonisolated_success_basin": basin_size >= 5,
        "local_parametric_structure": len(qualifying_effects) >= 2
        and (gradient_sign_reversal or interaction_count >= 2),
        "safety_frontier_coverage": (
            len(frontier) >= math.ceil(0.05 * len(completed))
            if require_safety_frontier
            else True
        ),
    }
    return {
        "planned_recipe_count": TOTAL_RECIPE_COUNT,
        "completed_recipe_count": len(completed),
        "exact_replay_count": len(exact),
        "failed_recipe_count": TOTAL_RECIPE_COUNT - len(completed),
        "feasible_recipe_count": len(feasible),
        "successful_recipe_count": len(successful),
        "safety_frontier_recipe_count": len(frontier),
        "score_distribution": score_distribution,
        "feasible_score_distribution": feasible_distribution,
        "primary_metric": primary_metric,
        "primary_metric_distribution": primary_distribution,
        "dynamic_range_p90_p10": dynamic_range,
        "primary_metric_range": primary_range,
        "floor_fraction": floor_fraction,
        "ceiling_fraction": ceiling_fraction,
        "largest_success_component": basin_size,
        "top_five_percent_count": top_count,
        "top_region_has_target_interior": top_interior,
        "score_noise": score_noise,
        "primary_metric_noise": primary_noise,
        "score_effect_gate": score_effect_gate,
        "primary_metric_effect_gate": primary_effect_gate,
        "local_family_count": len(effects),
        "qualifying_local_family_count": len(qualifying_effects),
        "interaction_family_count": interaction_count,
        "gradient_sign_reversal": gradient_sign_reversal,
        "local_effects": effects,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "ADAPTIVE_RECIPE_COUNT",
    "BROAD_RECIPE_COUNT",
    "TOTAL_RECIPE_COUNT",
    "WORK_II_Q1_RESPONSE_SURFACE_VERSION",
    "analyze_q1_world",
    "broad_sobol_design",
    "build_adaptive_design",
    "deterministic_design_seed",
    "select_adaptive_anchors",
]
