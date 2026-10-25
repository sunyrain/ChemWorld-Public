"""Provider-free reviewer controls for Work II law compression and action transfer."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy import linalg, stats

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_prior_discovery import parse_work_ii_law_summary

CONTROL_SCHEMA_VERSION = "chemworld-work-ii-reviewer-controls-0.1"
LAW_THRESHOLDS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _correlation(x: Sequence[float], y: Sequence[float], kind: str) -> dict[str, Any]:
    if len(x) != len(y) or len(x) < 3:
        return {"n": len(x), "coefficient": None, "p_value": None}
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "coefficient": None, "p_value": None}
    result = stats.spearmanr(x, y) if kind == "spearman" else stats.pearsonr(x, y)
    return {
        "n": len(x),
        "coefficient": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _bootstrap_correlation(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    kind: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)
    cluster_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        draw = [row for cluster_id in sampled for row in grouped[str(cluster_id)]]
        x = [float(row["law_normalized_mae"]) for row in draw]
        y = [float(row[outcome]) for row in draw]
        result = _correlation(x, y, kind)
        if result["coefficient"] is not None:
            estimates.append(float(result["coefficient"]))
    if not estimates:
        return {
            "cluster_count": len(cluster_ids),
            "replicate_count": replicates,
            "valid_replicate_count": 0,
            "percentile_95_interval": None,
        }
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return {
        "cluster_count": len(cluster_ids),
        "replicate_count": replicates,
        "valid_replicate_count": len(estimates),
        "percentile_95_interval": [float(low), float(high)],
    }


def _task_id(row: Mapping[str, Any]) -> str:
    parts = str(row["cluster_id"]).split("--")
    if len(parts) < 3:
        raise ValueError(f"cannot recover task from cluster ID: {row['cluster_id']}")
    return parts[1]


def analyze_w2_50(
    summary: Mapping[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260827,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    raw_rows = summary.get("cell_rows")
    if not isinstance(raw_rows, list):
        raise ValueError("W2-50 summary lacks cell rows")
    eligible: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("W2-50 cell row is malformed")
        if raw.get("status") != "completed_uncontaminated":
            failures.append(
                {
                    "cell_id": raw.get("cell_id"),
                    "cluster_id": raw.get("cluster_id"),
                    "status": raw.get("status"),
                }
            )
            continue
        row = {
            "cell_id": str(raw["cell_id"]),
            "cluster_id": str(raw["cluster_id"]),
            "task_id": _task_id(raw),
            "world_seed": int(raw["world_seed"]),
            "arm": str(raw["arm"]),
            "law_normalized_mae": _finite(
                raw.get("law_normalized_mae"), "law_normalized_mae"
            ),
            "normalized_regret": _finite(raw.get("normalized_regret"), "normalized_regret"),
            "selected_rank": int(raw["selected_rank"]),
            "top1_selected": bool(raw["top1_selected"]),
        }
        eligible.append(row)
    if len(raw_rows) != 45 or len(eligible) != 42 or len(failures) != 3:
        raise ValueError("W2-50 frozen 45/42/3 denominator drifted")

    def relationship(rows: Sequence[Mapping[str, Any]], *, offset: int) -> dict[str, Any]:
        laws = [float(row["law_normalized_mae"]) for row in rows]
        result: dict[str, Any] = {"cell_count": len(rows)}
        for outcome_index, outcome in enumerate(("normalized_regret", "selected_rank")):
            values = [float(row[outcome]) for row in rows]
            result[outcome] = {}
            for kind_index, kind in enumerate(("spearman", "pearson")):
                entry = _correlation(laws, values, kind)
                entry["cluster_bootstrap"] = _bootstrap_correlation(
                    rows,
                    outcome=outcome,
                    kind=kind,
                    replicates=bootstrap_replicates,
                    seed=seed + offset + outcome_index * 10 + kind_index,
                )
                result[outcome][kind] = entry
        return result

    by_task: dict[str, Any] = {}
    for task_index, task_id in enumerate(sorted({str(row["task_id"]) for row in eligible})):
        rows = [row for row in eligible if row["task_id"] == task_id]
        by_task[task_id] = relationship(rows, offset=100 * (task_index + 1))
        if progress is not None:
            progress(
                {
                    "stage": "w2_50_task_complete",
                    "completed_tasks": task_index + 1,
                    "total_tasks": 3,
                    "task_id": task_id,
                }
            )

    threshold_rows: list[dict[str, Any]] = []
    for threshold in LAW_THRESHOLDS:
        counts = {
            "adequate_law_correct_action": 0,
            "adequate_law_wrong_action": 0,
            "inadequate_law_correct_action": 0,
            "inadequate_law_wrong_action": 0,
        }
        for row in eligible:
            adequate = float(row["law_normalized_mae"]) <= threshold
            correct = bool(row["top1_selected"])
            key = (
                ("adequate" if adequate else "inadequate")
                + "_law_"
                + ("correct" if correct else "wrong")
                + "_action"
            )
            counts[key] += 1
        threshold_rows.append({"law_mae_threshold": threshold, **counts})
    return {
        "scheduled_cell_count": 45,
        "eligible_cell_count": len(eligible),
        "retained_failure_count": len(failures),
        "world_cluster_count": len({row["cluster_id"] for row in eligible}),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "overall": relationship(eligible, offset=0),
        "by_task": by_task,
        "threshold_sensitivity": threshold_rows,
        "cell_rows": eligible,
        "retained_failures": failures,
    }


def _basis_value(spec: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    basis = str(spec["basis"])
    inputs = list(map(str, spec["input_ids"]))
    if basis == "categorical_level":
        return float(features[inputs[0]] == spec["category_value"])
    if basis.startswith("conditional_"):
        power = {"conditional_linear": 1, "conditional_quadratic": 2,
                 "conditional_cubic": 3}[basis]
        return float(features[inputs[0]] == spec["category_value"]) * float(
            features[inputs[1]]
        ) ** power
    values = [float(features[item]) for item in inputs]
    if basis == "linear":
        return values[0]
    if basis == "quadratic":
        return values[0] ** 2
    if basis == "cubic":
        return values[0] ** 3
    if basis == "interaction":
        return values[0] * values[1]
    raise ValueError(f"unknown basis {basis}")


def _candidate_bases(
    queries: Sequence[Mapping[str, Any]], allowed_feature_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    features = [query["feature_values"] for query in queries]
    specs: list[dict[str, Any]] = []
    numeric: list[str] = []
    for feature_id in allowed_feature_ids:
        values = [row[feature_id] for row in features]
        if all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
            numeric.append(feature_id)
            for basis in ("linear", "quadratic", "cubic"):
                specs.append({"basis": basis, "input_ids": [feature_id]})
        for category in sorted(set(values), key=lambda item: (str(type(item)), repr(item))):
            specs.append(
                {
                    "basis": "categorical_level",
                    "input_ids": [feature_id],
                    "category_value": category,
                }
            )
    for left, right in combinations(numeric, 2):
        specs.append({"basis": "interaction", "input_ids": [left, right]})
    for condition_id in allowed_feature_ids:
        categories = sorted(
            {row[condition_id] for row in features},
            key=lambda item: (str(type(item)), repr(item)),
        )
        for category in categories:
            for numeric_id in numeric:
                if numeric_id == condition_id:
                    continue
                for basis in (
                    "conditional_linear",
                    "conditional_quadratic",
                    "conditional_cubic",
                ):
                    specs.append(
                        {
                            "basis": basis,
                            "input_ids": [condition_id, numeric_id],
                            "category_value": category,
                        }
                    )
    unique_specs: list[dict[str, Any]] = []
    columns: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for spec in specs:
        column = np.asarray([_basis_value(spec, row) for row in features], dtype=float)
        if not np.all(np.isfinite(column)):
            continue
        scale = max(float(np.max(np.abs(column))), 1.0)
        signature = tuple(np.round(column / scale, 12))
        if signature in seen or np.allclose(column, column[0], atol=1e-12, rtol=0.0):
            continue
        seen.add(signature)
        unique_specs.append(spec)
        columns.append(column)
    matrix = np.column_stack(columns) if columns else np.empty((len(queries), 0))
    return unique_specs, matrix


def _solve_subset(
    matrix: np.ndarray, target: np.ndarray, train_indices: Sequence[int], selected: Sequence[int]
) -> tuple[float, np.ndarray, np.ndarray]:
    train = np.asarray(train_indices, dtype=int)
    design = np.column_stack([np.ones(len(matrix)), matrix[:, list(selected)]])
    coefficients = np.linalg.lstsq(design[train], target[train], rcond=None)[0]
    predictions = np.clip(design @ coefficients, 0.0, 1.0)
    error = float(np.mean(np.abs(predictions[train] - target[train])))
    return error, coefficients, predictions


def _fit_metric(
    matrix: np.ndarray,
    target: Sequence[float],
    *,
    term_budget: int,
    train_indices: Sequence[int],
) -> dict[str, Any]:
    y = np.asarray(target, dtype=float)
    train = np.asarray(train_indices, dtype=int)
    budget = max(0, min(int(term_budget), 64, len(train) - 1, matrix.shape[1]))
    best: tuple[float, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    if budget <= 2:
        candidates = [()] if budget == 0 else []
        if budget >= 1:
            candidates.extend((index,) for index in range(matrix.shape[1]))
        if budget >= 2:
            candidates.extend(combinations(range(matrix.shape[1]), 2))
        for selected in candidates:
            error, coefficients, predictions = _solve_subset(matrix, y, train, selected)
            candidate = (error, tuple(selected), coefficients, predictions)
            if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
                best = candidate
    else:
        centered = matrix[train] - np.mean(matrix[train], axis=0, keepdims=True)
        _, diagonal, pivots = linalg.qr(centered, mode="economic", pivoting=True)
        diagonal_values = np.abs(np.diag(diagonal))
        tolerance = max(centered.shape) * np.finfo(float).eps * (
            float(diagonal_values[0]) if len(diagonal_values) else 0.0
        )
        rank = int(np.sum(diagonal_values > tolerance))
        selected = tuple(sorted(map(int, pivots[: min(budget, rank)])))
        error, coefficients, predictions = _solve_subset(matrix, y, train, selected)
        best = (error, selected, coefficients, predictions)
    assert best is not None
    error, selected, coefficients, predictions = best
    return {
        "train_mae": error,
        "intercept": float(coefficients[0]),
        "selected_indices": list(selected),
        "coefficients": [float(value) for value in coefficients[1:]],
        "predictions": [float(value) for value in predictions],
    }


def _law_payload(
    *,
    summary_id: str,
    allowed_feature_ids: Sequence[str],
    metric_fits: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metric_laws = []
    for metric_id, fit in metric_fits.items():
        terms = []
        for term_index, (spec_index, coefficient) in enumerate(
            zip(fit["selected_indices"], fit["coefficients"], strict=True), start=1
        ):
            spec = deepcopy(dict(specs[int(spec_index)]))
            spec.update(
                {
                    "term_id": f"{metric_id}-oracle-{term_index:02d}",
                    "coefficient": float(coefficient),
                }
            )
            terms.append(spec)
        metric_laws.append(
            {
                "metric_id": metric_id,
                "intercept": float(fit["intercept"]),
                "link": "identity",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "terms": terms,
            }
        )
    return {
        "schema_version": "chemworld-work-ii-law-summary-0.1",
        "summary_id": summary_id,
        "feature_ids": list(allowed_feature_ids),
        "metric_laws": metric_laws,
        "evidence_ids": [],
        "applicability": "In-domain typed-law schema capacity control on registered queries.",
        "limitations": ["not a global mechanistic-recovery claim"],
        "confidence": 1.0,
    }


def _validated_predictions(
    payload: Mapping[str, Any],
    *,
    queries: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    law = parse_work_ii_law_summary(
        payload,
        allowed_feature_ids=contract["allowed_feature_ids"],
        allowed_metric_ids=contract["allowed_metric_ids"],
        required_metric_ids=contract["required_metric_ids"],
        evidence_catalog=contract["evidence_catalog"],
    )
    result: dict[str, dict[str, float]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        result[query_id] = law.predict(query["feature_values"])
        for metric_id, value in result[query_id].items():
            if not math.isclose(
                value, expected[query_id][metric_id], rel_tol=0.0, abs_tol=1.0e-10
            ):
                raise ValueError("typed-law execution differs from fitted design-matrix prediction")
    return result


def _mean_error(
    predictions: Mapping[str, Mapping[str, float]],
    targets: Mapping[str, Mapping[str, float]],
) -> float:
    errors = [
        abs(float(predictions[query_id][metric_id]) - float(value))
        for query_id, metrics in targets.items()
        for metric_id, value in metrics.items()
    ]
    return mean(errors)


def _cell_capacity(
    row: Mapping[str, Any],
    binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_summary: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = row["checkpoint_error"]
    stage = str(checkpoint["effective_final_stage"])
    score = checkpoint["checkpoint_scores"][stage]
    targets: dict[str, dict[str, float]] = defaultdict(dict)
    truths: dict[str, dict[str, float]] = defaultdict(dict)
    for term in score["terms"]:
        query_id = str(term["query_id"])
        metric_id = str(term["metric_id"])
        targets[query_id][metric_id] = float(term["predicted_mean"])
        truths[query_id][metric_id] = float(term["evaluator_truth"])
    queries = [
        query for query in plan["queries"] if str(query["query_id"]) in set(targets)
    ]
    if len(queries) != len(targets):
        raise ValueError(f"{row['cell_id']} query denominator drifted")
    contract = plan["law_summary_contract"]
    metric_ids = list(contract["required_metric_ids"])
    snapshots = source_summary["analysis"]["belief_snapshots"]
    snapshot = next((item for item in snapshots if item.get("stage") == stage), snapshots[-1])
    submitted = snapshot["law_summary"]
    budgets = {
        str(metric["metric_id"]): len(metric["terms"]) for metric in submitted["metric_laws"]
    }
    specs, matrix = _candidate_bases(queries, contract["allowed_feature_ids"])
    query_ids = [str(query["query_id"]) for query in queries]

    def fit_all(budget_by_metric: Mapping[str, int], label: str) -> tuple[dict[str, Any], dict]:
        fits = {}
        expected: dict[str, dict[str, float]] = {query_id: {} for query_id in query_ids}
        for metric_id in metric_ids:
            target = [targets[query_id][metric_id] for query_id in query_ids]
            fit = _fit_metric(
                matrix,
                target,
                term_budget=budget_by_metric[metric_id],
                train_indices=range(len(queries)),
            )
            fits[metric_id] = fit
            for query_index, query_id in enumerate(query_ids):
                expected[query_id][metric_id] = fit["predictions"][query_index]
        payload = _law_payload(
            summary_id=f"{label}-{canonical_json_sha256({'cell': row['cell_id']})[:12]}",
            allowed_feature_ids=contract["allowed_feature_ids"],
            metric_fits=fits,
            specs=specs,
        )
        predictions = _validated_predictions(
            payload, queries=queries, contract=contract, expected=expected
        )
        return payload, predictions

    full_payload, full_predictions = fit_all(dict.fromkeys(metric_ids, 64), "full")
    matched_payload, matched_predictions = fit_all(budgets, "matched")

    loo_predictions: dict[str, dict[str, float]] = {query_id: {} for query_id in query_ids}
    for held_out in range(len(queries)):
        train = [index for index in range(len(queries)) if index != held_out]
        fits = {}
        expected: dict[str, dict[str, float]] = {query_id: {} for query_id in query_ids}
        for metric_id in metric_ids:
            fit = _fit_metric(
                matrix,
                [targets[query_id][metric_id] for query_id in query_ids],
                term_budget=64,
                train_indices=train,
            )
            fits[metric_id] = fit
            for query_index, query_id in enumerate(query_ids):
                expected[query_id][metric_id] = fit["predictions"][query_index]
        payload = _law_payload(
            summary_id=f"loo-{held_out}-{canonical_json_sha256({'cell': row['cell_id']})[:12]}",
            allowed_feature_ids=contract["allowed_feature_ids"],
            metric_fits=fits,
            specs=specs,
        )
        predictions = _validated_predictions(
            payload, queries=queries, contract=contract, expected=expected
        )
        query_id = query_ids[held_out]
        loo_predictions[query_id] = predictions[query_id]

    agent_target_error = float(row["law_summary"]["prediction_consistency_normalized_mae"])
    agent_truth_error = float(row["law_summary"]["normalized_mae"])
    full_target_error = _mean_error(full_predictions, targets)
    full_truth_error = _mean_error(full_predictions, truths)
    matched_target_error = _mean_error(matched_predictions, targets)
    matched_truth_error = _mean_error(matched_predictions, truths)
    loo_target_error = _mean_error(loo_predictions, targets)
    return {
        "cell_id": row["cell_id"],
        "block": row["block"],
        "locus_id": row["locus_id"],
        "task_id": row["task_id"],
        "world_cluster_id": row["world_cluster_id"],
        "world_seed": row["world_seed"],
        "prior_arm": row["prior_arm"],
        "effective_final_stage": stage,
        "query_count": len(query_ids),
        "query_metric_term_count": sum(len(metrics) for metrics in targets.values()),
        "candidate_basis_count": len(specs),
        "participant_term_budget_by_metric": budgets,
        "participant_law_to_final_prediction_mae": agent_target_error,
        "participant_law_to_truth_mae": agent_truth_error,
        "full_schema_to_final_prediction_mae": full_target_error,
        "full_schema_to_truth_mae": full_truth_error,
        "term_matched_to_final_prediction_mae": matched_target_error,
        "term_matched_to_truth_mae": matched_truth_error,
        "leave_one_query_out_to_final_prediction_mae": loo_target_error,
        "participant_distillation_gap": agent_target_error - full_target_error,
        "same_schema_residual_gap": full_target_error,
        "participant_excess_over_term_matched": agent_target_error - matched_target_error,
        "full_schema_law": full_payload,
        "term_matched_law": matched_payload,
        "source_summary_path": binding["summary"]["path"],
    }


def analyze_schema_capacity(
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evaluator_root: Path,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    rows = dataset.get("cell_rows")
    bindings = manifest.get("cell_bindings")
    if not isinstance(rows, list) or len(rows) != 135:
        raise ValueError("current-composite analysis dataset must contain 135 cells")
    if not isinstance(bindings, list) or len(bindings) != 135:
        raise ValueError("current-composite input manifest must bind 135 cells")
    binding_by_cell = {str(item["cell_id"]): item for item in bindings}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        cell_id = str(row["cell_id"])
        try:
            binding = binding_by_cell[cell_id]
            plan_path = evaluator_root / "truth" / str(row["world_cluster_id"]) / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            source_summary = json.loads(
                Path(binding["summary"]["path"]).read_text(encoding="utf-8")
            )
            results.append(_cell_capacity(row, binding, plan, source_summary))
        except Exception as error:
            failures.append(
                {"cell_id": cell_id, "type": type(error).__name__, "message": str(error)[:1000]}
            )
        if progress is not None and (row_index % 5 == 0 or row_index == len(rows)):
            progress(
                {
                    "stage": "schema_capacity_progress",
                    "completed_cells": row_index,
                    "total_cells": len(rows),
                    "successful_cells": len(results),
                    "failed_cells": len(failures),
                }
            )
    fields = (
        "participant_law_to_final_prediction_mae",
        "participant_law_to_truth_mae",
        "full_schema_to_final_prediction_mae",
        "full_schema_to_truth_mae",
        "term_matched_to_final_prediction_mae",
        "term_matched_to_truth_mae",
        "leave_one_query_out_to_final_prediction_mae",
        "participant_distillation_gap",
        "same_schema_residual_gap",
        "participant_excess_over_term_matched",
    )
    aggregate = {
        "cell_weighted_mean": {
            field: mean(float(row[field]) for row in results) if results else None
            for field in fields
        },
        "full_schema_near_exact_cell_count": sum(
            float(row["full_schema_to_final_prediction_mae"]) <= 1.0e-10 for row in results
        ),
        "term_matched_near_exact_cell_count": sum(
            float(row["term_matched_to_final_prediction_mae"]) <= 1.0e-10 for row in results
        ),
        "participant_outperformed_full_schema_cell_count": sum(
            float(row["participant_law_to_final_prediction_mae"])
            < float(row["full_schema_to_final_prediction_mae"]) - 1.0e-10
            for row in results
        ),
    }
    by_task: dict[str, Any] = {}
    for task_id in sorted({str(row["task_id"]) for row in results}):
        subset = [row for row in results if row["task_id"] == task_id]
        by_task[task_id] = {
            "cell_count": len(subset),
            "mean": {field: mean(float(row[field]) for row in subset) for field in fields},
        }
    return {
        "scheduled_cell_count": 135,
        "completed_cell_count": len(results),
        "failed_cell_count": len(failures),
        "provider_call_count": 0,
        "participant_physical_experiment_count": 0,
        "aggregate": aggregate,
        "by_task": by_task,
        "cell_rows": results,
        "failures": failures,
    }


__all__ = [
    "CONTROL_SCHEMA_VERSION",
    "LAW_THRESHOLDS",
    "analyze_schema_capacity",
    "analyze_w2_50",
]
