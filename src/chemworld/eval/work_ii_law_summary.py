"""Evaluator-owned execution of final Work II law summaries.

The participant submits a typed law summary inside its persistent session.  This
module executes that summary on the registered held-out query coordinates and
compares its predictions with evaluator truth and the final explicit checkpoint
predictions.  It reports descriptive public evidence only; private transfer is
still required for a reusable-law claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_analysis import WorkIIAnalysisError, score_prediction_error
from chemworld.eval.work_ii_prior_discovery import parse_work_ii_law_summary

WORK_II_LAW_SUMMARY_EVALUATION_VERSION = (
    "chemworld-work-ii-law-summary-evaluation-0.1"
)


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _prediction_truth(
    predictions: object,
) -> dict[str, dict[str, float]] | None:
    rows = _sequence(predictions)
    if not rows:
        return None
    truth: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in truth:
            return None
        metrics: dict[str, float] = {}
        for metric in _sequence(row.get("metrics")):
            if not isinstance(metric, Mapping):
                return None
            metric_id = str(metric.get("metric_id", ""))
            mean = _numeric(metric.get("mean"))
            if not metric_id or metric_id in metrics or mean is None:
                return None
            metrics[metric_id] = mean
        if not metrics:
            return None
        truth[query_id] = metrics
    return truth


def _metadata(raw_summary: object) -> dict[str, Any]:
    if not isinstance(raw_summary, Mapping):
        return {
            "present": False,
            "schema_version": None,
            "schema_version_matches": False,
            "summary_id": None,
            "feature_count": 0,
            "metric_law_count": 0,
            "term_count": 0,
            "evidence_reference_count": 0,
            "confidence": None,
        }
    features = _sequence(raw_summary.get("feature_ids"))
    metric_laws = _sequence(raw_summary.get("metric_laws"))
    evidence = _sequence(raw_summary.get("evidence_ids"))
    term_count = sum(
        len(_sequence(row.get("terms")))
        for row in metric_laws
        if isinstance(row, Mapping)
    )
    return {
        "present": True,
        "schema_version": raw_summary.get("schema_version"),
        "schema_version_matches": (
            raw_summary.get("schema_version") == "chemworld-work-ii-law-summary-0.1"
        ),
        "summary_id": raw_summary.get("summary_id"),
        "feature_count": len(features),
        "metric_law_count": len(metric_laws),
        "term_count": term_count,
        "evidence_reference_count": len(evidence),
        "confidence": _numeric(raw_summary.get("confidence")),
    }


def evaluate_final_law_summary(
    raw_summary: object,
    *,
    truth_plan: Mapping[str, Any],
    evaluator_truth: Mapping[str, Mapping[str, Any]],
    final_checkpoint_predictions: object,
    effective_pre_error: object,
    effective_final_error: object,
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one final law summary on all registered evaluator queries."""

    record: dict[str, Any] = {
        "schema_version": WORK_II_LAW_SUMMARY_EVALUATION_VERSION,
        **_metadata(raw_summary),
        "evaluation_contract_sha256": canonical_json_sha256(evaluation_contract),
        "evaluator_executability_status": "not_evaluated_missing_final_law_summary",
        "continuous_prediction_validity_status": "not_evaluated",
        "registered_query_count": 0,
        "registered_query_metric_count": 0,
        "normalized_mae": None,
        "pre_to_law_summary_improvement": None,
        "summary_minus_effective_final_error": None,
        "prediction_consistency_normalized_mae": None,
        "query_predictions": [],
        "normalized_error_terms": [],
        "evaluation_error": None,
    }
    if not isinstance(raw_summary, Mapping):
        record["status"] = "missing_final_law_summary"
        return record

    contract = truth_plan.get("law_summary_contract")
    if not isinstance(contract, Mapping):
        record.update(
            {
                "status": "law_summary_evaluation_failed",
                "evaluator_executability_status": "failed_missing_truth_plan_contract",
                "evaluation_error": "truth plan lacks its law-summary contract",
            }
        )
        return record
    queries = _sequence(truth_plan.get("queries"))
    try:
        summary = parse_work_ii_law_summary(
            raw_summary,
            allowed_feature_ids=list(contract.get("allowed_feature_ids", [])),
            allowed_metric_ids=list(contract.get("allowed_metric_ids", [])),
            evidence_catalog=list(contract.get("evidence_catalog", [])),
            required_metric_ids=list(contract.get("required_metric_ids", [])),
        )
        prediction_rows: list[dict[str, Any]] = []
        for query in queries:
            if not isinstance(query, Mapping):
                raise ValueError("truth plan contains a malformed query")
            feature_values = query.get("feature_values")
            if not isinstance(feature_values, Mapping):
                raise ValueError("truth-plan query lacks feature values")
            predicted = summary.predict(feature_values)
            metric_ids = [str(item) for item in _sequence(query.get("metric_ids"))]
            prediction_rows.append(
                {
                    "query_id": str(query.get("query_id", "")),
                    "metrics": [
                        {"metric_id": metric_id, "mean": predicted[metric_id]}
                        for metric_id in metric_ids
                    ],
                }
            )
        score = score_prediction_error(prediction_rows, evaluator_truth)
        consistency_error: float | None = None
        final_truth = _prediction_truth(final_checkpoint_predictions)
        if final_truth is not None:
            try:
                consistency_error = score_prediction_error(
                    prediction_rows,
                    final_truth,
                ).error
            except WorkIIAnalysisError:
                consistency_error = None
    except (KeyError, ValueError, WorkIIAnalysisError) as error:
        record.update(
            {
                "status": "law_summary_evaluation_failed",
                "evaluator_executability_status": "failed_registered_query_execution",
                "continuous_prediction_validity_status": "not_evaluated_execution_failed",
                "evaluation_error": str(error),
            }
        )
        return record

    pre_error = _numeric(effective_pre_error)
    final_error = _numeric(effective_final_error)
    record.update(
        {
            "status": "evaluated",
            "evaluator_executability_status": "passed_registered_query_execution",
            "continuous_prediction_validity_status": (
                "evaluated_descriptive_no_public_binary_threshold"
            ),
            "registered_query_count": len(prediction_rows),
            "registered_query_metric_count": score.term_count,
            "normalized_mae": score.error,
            "pre_to_law_summary_improvement": (
                None if pre_error is None else pre_error - score.error
            ),
            "summary_minus_effective_final_error": (
                None if final_error is None else score.error - final_error
            ),
            "prediction_consistency_normalized_mae": consistency_error,
            "query_predictions": prediction_rows,
            "normalized_error_terms": [dict(item) for item in score.terms],
        }
    )
    return record


__all__ = [
    "WORK_II_LAW_SUMMARY_EVALUATION_VERSION",
    "evaluate_final_law_summary",
]
