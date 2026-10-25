"""Public-history-only candidate portfolios for static optimization agents."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from chemworld.agents._probability import normal_cdf as _normal_cdf


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)


def _expected_improvement(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    *,
    best: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma = np.maximum(np.asarray(standard_deviation, dtype=float), 1.0e-9)
    improvement = np.asarray(mean, dtype=float) - float(best) - float(xi)
    z = improvement / sigma
    return improvement * _normal_cdf(z) + sigma * _normal_pdf(z)


def _canonicalize_nominal(
    matrix: np.ndarray,
    categorical: tuple[tuple[int, int], ...],
) -> np.ndarray:
    canonical = np.asarray(np.clip(matrix, 0.0, 1.0), dtype=float).copy()
    for coordinate, category_count in categorical:
        categories = np.minimum(
            np.floor(canonical[:, coordinate] * category_count).astype(int),
            category_count - 1,
        )
        canonical[:, coordinate] = (categories + 0.5) / category_count
    return canonical


def _encode(
    matrix: np.ndarray,
    categorical: tuple[tuple[int, int], ...],
) -> np.ndarray:
    values = np.asarray(np.clip(matrix, 0.0, 1.0), dtype=float)
    categorical_map = dict(categorical)
    continuous = [
        coordinate
        for coordinate in range(values.shape[1])
        if coordinate not in categorical_map
    ]
    blocks = [values[:, continuous]]
    for coordinate, category_count in categorical:
        categories = np.minimum(
            np.floor(values[:, coordinate] * category_count).astype(int),
            category_count - 1,
        )
        one_hot = np.zeros((len(values), category_count), dtype=float)
        one_hot[np.arange(len(values)), categories] = 1.0
        blocks.append(one_hot)
    return np.column_stack(blocks)


def _minimum_distance(
    candidates: np.ndarray,
    history: np.ndarray,
    categorical: tuple[tuple[int, int], ...],
) -> np.ndarray:
    encoded_candidates = _encode(candidates, categorical)
    encoded_history = _encode(history, categorical)
    return np.sqrt(
        np.min(
            np.sum(
                np.square(
                    encoded_candidates[:, np.newaxis, :]
                    - encoded_history[np.newaxis, :, :]
                ),
                axis=2,
            ),
            axis=1,
        )
    )


def _candidate_matrix(
    history: np.ndarray,
    scores: np.ndarray,
    *,
    categorical: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = history.shape[1]
    categorical_map = dict(categorical)
    continuous = [
        coordinate for coordinate in range(dimension) if coordinate not in categorical_map
    ]
    global_candidates = rng.random((1536, dimension))

    leader_count = min(3, len(history))
    leaders = history[np.argsort(scores)[-leader_count:]]
    local_rows = []
    rows_per_leader = max(1, 768 // leader_count)
    for leader in leaders:
        local = np.tile(leader, (rows_per_leader, 1))
        if continuous:
            local[:, continuous] = np.clip(
                local[:, continuous]
                + rng.normal(0.0, 0.18, size=(rows_per_leader, len(continuous))),
                0.0,
                1.0,
            )
        for row in local:
            for coordinate, category_count in categorical:
                if rng.random() < 0.30:
                    row[coordinate] = (
                        int(rng.integers(0, category_count)) + 0.5
                    ) / category_count
        local_rows.append(local)

    boundary_rows: list[np.ndarray] = []
    for leader in leaders:
        for coordinate in continuous:
            for boundary in (0.0, 1.0):
                candidate = np.array(leader, copy=True)
                candidate[coordinate] = boundary
                boundary_rows.append(candidate)
        for coordinate, category_count in categorical:
            for category in range(category_count):
                candidate = np.array(leader, copy=True)
                candidate[coordinate] = (category + 0.5) / category_count
                boundary_rows.append(candidate)
    for _ in range(256):
        candidate = rng.random(dimension)
        if continuous:
            candidate[continuous] = rng.integers(0, 2, size=len(continuous))
        boundary_rows.append(candidate)

    matrix = np.vstack(
        [
            global_candidates,
            *local_rows,
            np.asarray(boundary_rows, dtype=float),
        ]
    )
    matrix = _canonicalize_nominal(matrix, categorical)
    matrix = np.unique(np.round(matrix, decimals=12), axis=0)
    boundary_mask = np.zeros(len(matrix), dtype=bool)
    # Deduplication changes row positions, so identify boundary candidates by
    # canonical membership rather than relying on the pre-dedup row offset.
    canonical_boundaries = {
        tuple(row)
        for row in np.round(
            _canonicalize_nominal(
                np.asarray(boundary_rows, dtype=float),
                categorical,
            ),
            decimals=12,
        )
    }
    boundary_mask[:] = [
        tuple(row) in canonical_boundaries for row in np.round(matrix, decimals=12)
    ]
    return matrix, boundary_mask


def public_surrogate_candidate_portfolio(
    history_vectors: Sequence[Sequence[float]],
    scores: Sequence[float],
    *,
    categorical: tuple[tuple[int, int], ...],
    seed: int,
) -> list[dict[str, Any]]:
    """Build a deterministic mixed-surrogate portfolio from public outcomes only."""

    history = np.asarray(history_vectors, dtype=float)
    score_values = np.asarray(scores, dtype=float)
    if history.ndim != 2 or len(history) < 8:
        raise ValueError("candidate portfolio requires at least eight history vectors")
    if score_values.shape != (len(history),):
        raise ValueError("candidate portfolio scores do not match history")
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(score_values)):
        raise ValueError("candidate portfolio inputs must be finite")

    rng = np.random.default_rng(int(seed))
    candidates, boundary_mask = _candidate_matrix(
        history,
        score_values,
        categorical=categorical,
        rng=rng,
    )
    x_train = _encode(history, categorical)
    x_candidates = _encode(candidates, categorical)

    gp = GaussianProcessRegressor(
        kernel=Matern(length_scale=np.ones(x_train.shape[1]), nu=2.5)
        + WhiteKernel(noise_level=1.0e-4),
        normalize_y=True,
        alpha=1.0e-8,
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x_train, score_values)
    gp_mean, gp_std = gp.predict(x_candidates, return_std=True)
    gp_ei = _expected_improvement(
        gp_mean,
        gp_std,
        best=float(np.max(score_values)),
    )

    rf = RandomForestRegressor(
        n_estimators=192,
        min_samples_leaf=2,
        random_state=int(seed) + 1,
        n_jobs=1,
    )
    rf.fit(x_train, score_values)
    tree_predictions = np.vstack(
        [tree.predict(x_candidates) for tree in rf.estimators_]
    )
    rf_mean = tree_predictions.mean(axis=0)
    rf_std = tree_predictions.std(axis=0)
    rf_ei = _expected_improvement(
        rf_mean,
        rf_std,
        best=float(np.max(score_values)),
    )

    minimum_distance = _minimum_distance(candidates, history, categorical)
    novel = minimum_distance >= 0.02
    consensus_mean = 0.5 * (gp_mean + rf_mean)
    boundary_priority = np.where(boundary_mask, minimum_distance, -np.inf)
    acquisition_rankings = (
        ("gp_ei", gp_ei, "gaussian_process_expected_improvement"),
        ("rf_ei", rf_ei, "random_forest_expected_improvement"),
        ("surrogate_consensus", consensus_mean, "cross_surrogate_predicted_score"),
        ("maximin_global", minimum_distance, "maximum_distance_from_public_history"),
        ("boundary_challenge", boundary_priority, "explicit_boundary_or_nominal_challenge"),
        ("gp_uncertainty", gp_std, "gaussian_process_uncertainty"),
    )

    selected_indices: set[int] = set()
    portfolio: list[dict[str, Any]] = []
    for candidate_id, acquisition, source in acquisition_rankings:
        ranking_values = np.where(novel, acquisition, -np.inf)
        ranking = np.argsort(ranking_values)[::-1]
        selected = next(
            (
                int(index)
                for index in ranking
                if np.isfinite(ranking_values[index])
                and int(index) not in selected_indices
            ),
            None,
        )
        if selected is None:
            continue
        selected_indices.add(selected)
        portfolio.append(
            {
                "candidate_id": candidate_id,
                "candidate_generation_policy": source,
                "search_vector": [
                    float(value) for value in candidates[selected]
                ],
                "public_surrogate_summary": {
                    "gp_predicted_score_mean": float(gp_mean[selected]),
                    "gp_predicted_score_std": float(gp_std[selected]),
                    "gp_expected_improvement": float(gp_ei[selected]),
                    "rf_predicted_score_mean": float(rf_mean[selected]),
                    "rf_predicted_score_std": float(rf_std[selected]),
                    "rf_expected_improvement": float(rf_ei[selected]),
                    "minimum_encoded_distance_to_history": float(
                        minimum_distance[selected]
                    ),
                    "explicit_boundary_candidate": bool(boundary_mask[selected]),
                },
            }
        )
    if len(portfolio) != len(acquisition_rankings):
        raise RuntimeError("candidate portfolio could not produce six distinct candidates")
    return portfolio


__all__ = ["public_surrogate_candidate_portfolio"]
