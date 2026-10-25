"""Frozen Work II prediction-error and cluster-contrast calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

WORK_II_ANALYSIS_SNAPSHOT_STAGES = (
    "pre_evidence",
    "after_experiment_1",
    "after_experiment_2",
    "final",
)
WORK_II_ANALYSIS_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")


class WorkIIAnalysisError(ValueError):
    """Raised when a formal Work II analysis record violates the frozen contract."""


@dataclass(frozen=True)
class PredictionErrorSummary:
    error: float
    term_count: int
    terms: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "term_count": self.term_count,
            "terms": [dict(item) for item in self.terms],
        }


@dataclass(frozen=True)
class ClusterCorrectionContrast:
    misindexed_improvement: float
    aligned_improvement: float
    primary_contrast: float

    def to_dict(self) -> dict[str, float]:
        return {
            "misindexed_improvement": self.misindexed_improvement,
            "aligned_improvement": self.aligned_improvement,
            "primary_contrast": self.primary_contrast,
        }


FAILURE_AWARE_IMPROVEMENT_BOUNDS = (-1.0, 1.0)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkIIAnalysisError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise WorkIIAnalysisError(f"{field} must be finite")
    return number


def score_prediction_error(
    predictions: Sequence[Mapping[str, Any]],
    evaluator_truth: Mapping[str, Mapping[str, Any]],
    *,
    metric_scales: Mapping[str, float] | None = None,
    default_metric_scale: float = 1.0,
) -> PredictionErrorSummary:
    """Return bounded normalized MAE over exact registered query/metric pairs."""

    if not math.isfinite(default_metric_scale) or default_metric_scale <= 0.0:
        raise WorkIIAnalysisError("default_metric_scale must be finite and positive")
    prediction_by_query: dict[str, Mapping[str, Any]] = {}
    for item in predictions:
        query_id = str(item.get("query_id", ""))
        if not query_id or query_id in prediction_by_query:
            raise WorkIIAnalysisError("prediction query IDs must be non-empty and unique")
        prediction_by_query[query_id] = item
    if set(prediction_by_query) != set(evaluator_truth):
        raise WorkIIAnalysisError("prediction queries must exactly match evaluator truth queries")

    scales = dict(metric_scales or {})
    terms: list[dict[str, Any]] = []
    for query_id in evaluator_truth:
        truth_metrics = evaluator_truth[query_id]
        metrics = prediction_by_query[query_id].get("metrics")
        if not isinstance(metrics, Sequence) or isinstance(metrics, str | bytes):
            raise WorkIIAnalysisError(f"predictions[{query_id}].metrics must be a list")
        prediction_metrics: dict[str, Mapping[str, Any]] = {}
        for metric in metrics:
            if not isinstance(metric, Mapping):
                raise WorkIIAnalysisError("prediction metric rows must be objects")
            metric_id = str(metric.get("metric_id", ""))
            if not metric_id or metric_id in prediction_metrics:
                raise WorkIIAnalysisError("prediction metric IDs must be non-empty and unique")
            prediction_metrics[metric_id] = metric
        if set(prediction_metrics) != set(truth_metrics):
            raise WorkIIAnalysisError(
                f"prediction metrics for {query_id} must exactly match evaluator truth"
            )
        for metric_id in truth_metrics:
            predicted = _finite_number(
                prediction_metrics[metric_id].get("mean"),
                field=f"{query_id}.{metric_id}.mean",
            )
            truth = _finite_number(
                truth_metrics[metric_id],
                field=f"{query_id}.{metric_id}.truth",
            )
            scale = _finite_number(
                scales.get(metric_id, default_metric_scale),
                field=f"metric_scales.{metric_id}",
            )
            if scale <= 0.0:
                raise WorkIIAnalysisError(f"metric scale for {metric_id} must be positive")
            terms.append(
                {
                    "query_id": query_id,
                    "metric_id": metric_id,
                    "predicted_mean": predicted,
                    "evaluator_truth": truth,
                    "metric_scale": scale,
                    "normalized_absolute_error": min(abs(predicted - truth) / scale, 1.0),
                }
            )
    if not terms:
        raise WorkIIAnalysisError("prediction error requires at least one query/metric term")
    return PredictionErrorSummary(
        error=sum(item["normalized_absolute_error"] for item in terms) / len(terms),
        term_count=len(terms),
        terms=tuple(terms),
    )


def correction_contrast(
    *,
    misindexed_pre: float,
    misindexed_final: float | None,
    aligned_pre: float,
    aligned_final: float | None,
) -> ClusterCorrectionContrast:
    """Apply frozen zero-improvement missing-final handling and calculate H3."""

    mis_pre = _finite_number(misindexed_pre, field="misindexed_pre")
    aligned_pre_value = _finite_number(aligned_pre, field="aligned_pre")
    mis_final = (
        mis_pre
        if misindexed_final is None
        else _finite_number(misindexed_final, field="misindexed_final")
    )
    aligned_final_value = (
        aligned_pre_value
        if aligned_final is None
        else _finite_number(aligned_final, field="aligned_final")
    )
    mis_improvement = mis_pre - mis_final
    aligned_improvement = aligned_pre_value - aligned_final_value
    return ClusterCorrectionContrast(
        misindexed_improvement=mis_improvement,
        aligned_improvement=aligned_improvement,
        primary_contrast=mis_improvement - aligned_improvement,
    )


def score_cell_checkpoint_errors(
    analysis: Mapping[str, Any],
    evaluator_truth: Mapping[str, Mapping[str, Any]],
    *,
    terminal_state: str,
    snapshot_stages: Sequence[str] | None = None,
    metric_scales: Mapping[str, float] | None = None,
    default_metric_scale: float = 1.0,
) -> dict[str, Any]:
    """Score one retained cell and apply the frozen missing-final rules."""

    if terminal_state not in {"completed", "right_censored", "failed"}:
        raise WorkIIAnalysisError("terminal_state is outside the formal contract")
    stages = tuple(
        str(stage)
        for stage in (
            WORK_II_ANALYSIS_SNAPSHOT_STAGES
            if snapshot_stages is None
            else snapshot_stages
        )
    )
    if (
        len(stages) < 2
        or stages[0] != "pre_evidence"
        or stages[-1] != "final"
        or len(set(stages)) != len(stages)
    ):
        raise WorkIIAnalysisError("snapshot stages must be unique from pre_evidence to final")
    raw_snapshots = analysis.get("belief_snapshots", [])
    if not isinstance(raw_snapshots, Sequence) or isinstance(
        raw_snapshots, str | bytes
    ):
        raw_snapshots = []
    scored: dict[str, dict[str, Any]] = {}
    unscorable: list[dict[str, str]] = []
    seen_stages: set[str] = set()
    for snapshot_index, snapshot in enumerate(raw_snapshots):
        if not isinstance(snapshot, Mapping):
            unscorable.append(
                {
                    "snapshot_index": str(snapshot_index),
                    "stage": "unknown",
                    "reason": "snapshot_not_an_object",
                }
            )
            continue
        stage = str(snapshot.get("stage", ""))
        if stage not in stages or stage in seen_stages:
            unscorable.append(
                {
                    "snapshot_index": str(snapshot_index),
                    "stage": stage or "unknown",
                    "reason": "unknown_or_duplicate_stage",
                }
            )
            continue
        seen_stages.add(stage)
        predictions = snapshot.get("predictions")
        if not isinstance(predictions, Sequence) or isinstance(
            predictions, str | bytes
        ):
            unscorable.append(
                {
                    "snapshot_index": str(snapshot_index),
                    "stage": stage,
                    "reason": "predictions_not_a_list",
                }
            )
            continue
        try:
            score = score_prediction_error(
                predictions,
                evaluator_truth,
                metric_scales=metric_scales,
                default_metric_scale=default_metric_scale,
            )
        except WorkIIAnalysisError as error:
            unscorable.append(
                {
                    "snapshot_index": str(snapshot_index),
                    "stage": stage,
                    "reason": str(error),
                }
            )
            continue
        scored[stage] = score.to_dict()

    pre = scored.get("pre_evidence")
    final = scored.get("final")
    effective_final_stage: str | None = None
    missing_rule: str
    primary_improvement: float
    if pre is None:
        effective_pre_error = None
        effective_final_error = None
        primary_improvement = 0.0
        missing_rule = "missing_or_unscorable_pre_sets_primary_improvement_to_zero"
    else:
        effective_pre_error = float(pre["error"])
        if final is not None:
            effective_final_stage = "final"
            effective_final_error = float(final["error"])
            missing_rule = "observed_final"
        elif terminal_state == "right_censored":
            available = [
                stage
                for stage in stages[1:-1]
                if stage in scored
            ]
            if available:
                effective_final_stage = available[-1]
                effective_final_error = float(scored[effective_final_stage]["error"])
                missing_rule = "right_censored_carries_last_valid_checkpoint_to_final"
            else:
                effective_final_stage = "pre_evidence"
                effective_final_error = effective_pre_error
                missing_rule = "right_censored_without_later_checkpoint_sets_zero_improvement"
        else:
            effective_final_stage = "pre_evidence"
            effective_final_error = effective_pre_error
            missing_rule = "missing_final_with_valid_pre_sets_zero_improvement"
        primary_improvement = effective_pre_error - effective_final_error

    return {
        "terminal_state": terminal_state,
        "scheduled_snapshot_count": len(stages),
        "observed_snapshot_count": len(raw_snapshots),
        "scored_snapshot_count": len(scored),
        "checkpoint_scores": scored,
        "unscorable_snapshots": unscorable,
        "effective_pre_error": effective_pre_error,
        "effective_final_error": effective_final_error,
        "effective_final_stage": effective_final_stage,
        "primary_improvement": primary_improvement,
        "confirmatory_improvement_bounds": (
            [primary_improvement, primary_improvement]
            if pre is not None and final is not None and terminal_state == "completed"
            else list(FAILURE_AWARE_IMPROVEMENT_BOUNDS)
        ),
        "missing_failure_rule": missing_rule,
    }


def build_cluster_correction_record(
    arm_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build H1/H2 and the H3 cluster contrast from one retained arm triplet."""

    if set(arm_records) != set(WORK_II_ANALYSIS_ARMS):
        raise WorkIIAnalysisError("cluster analysis requires the exact three-arm triplet")
    improvements: dict[str, float] = {}
    improvement_bounds: dict[str, tuple[float, float]] = {}
    pre_errors: dict[str, float | None] = {}
    for arm in WORK_II_ANALYSIS_ARMS:
        record = arm_records[arm]
        improvements[arm] = _finite_number(
            record.get("primary_improvement"),
            field=f"{arm}.primary_improvement",
        )
        raw_bounds = record.get("confirmatory_improvement_bounds")
        if (
            not isinstance(raw_bounds, Sequence)
            or isinstance(raw_bounds, str | bytes)
            or len(raw_bounds) != 2
        ):
            raw_bounds = [improvements[arm], improvements[arm]]
        lower = _finite_number(raw_bounds[0], field=f"{arm}.improvement_lower")
        upper = _finite_number(raw_bounds[1], field=f"{arm}.improvement_upper")
        if not -1.0 <= lower <= upper <= 1.0:
            raise WorkIIAnalysisError(f"{arm}.confirmatory_improvement_bounds are invalid")
        improvement_bounds[arm] = (lower, upper)
        pre = record.get("effective_pre_error")
        pre_errors[arm] = (
            None
            if pre is None
            else _finite_number(pre, field=f"{arm}.effective_pre_error")
        )
    misindexed_improvement = improvements["misindexed_nominal"]
    aligned_improvement = improvements["aligned_nominal"]
    misindexed_lower, _ = improvement_bounds["misindexed_nominal"]
    aligned_lower, aligned_upper = improvement_bounds["aligned_nominal"]
    h1 = (
        None
        if pre_errors["opaque"] is None or pre_errors["aligned_nominal"] is None
        else pre_errors["opaque"] - pre_errors["aligned_nominal"]
    )
    h2 = (
        None
        if pre_errors["misindexed_nominal"] is None or pre_errors["opaque"] is None
        else pre_errors["misindexed_nominal"] - pre_errors["opaque"]
    )
    return {
        "arm_primary_improvements": improvements,
        "arm_pre_errors": pre_errors,
        "H1_prior_utility": h1,
        "H2_prior_vulnerability": h2,
        "H3_misindexed_improvement": misindexed_improvement,
        "H3_aligned_improvement": aligned_improvement,
        "H3_primary_contrast": misindexed_improvement - aligned_improvement,
        "arm_confirmatory_improvement_bounds": {
            arm: list(improvement_bounds[arm]) for arm in WORK_II_ANALYSIS_ARMS
        },
        "H3_primary_contrast_lower_bound": misindexed_lower - aligned_upper,
        "H3_misindexed_improvement_lower_bound": misindexed_lower,
        "H3_aligned_improvement_lower_bound": aligned_lower,
        "H3_primary_contrast_formula": (
            "(E_misindexed_pre-E_misindexed_final)-"
            "(E_aligned_pre-E_aligned_final)"
        ),
    }


__all__ = [
    "FAILURE_AWARE_IMPROVEMENT_BOUNDS",
    "WORK_II_ANALYSIS_ARMS",
    "WORK_II_ANALYSIS_SNAPSHOT_STAGES",
    "ClusterCorrectionContrast",
    "PredictionErrorSummary",
    "WorkIIAnalysisError",
    "build_cluster_correction_record",
    "correction_contrast",
    "score_cell_checkpoint_errors",
    "score_prediction_error",
]
