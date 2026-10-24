"""Executable evaluation endpoints for every registered task success metric.

Task cards describe *what* should be measured.  This module binds those names to
the concrete evaluation layer that produces the value.  Keeping this separate
from the online reward prevents trajectory-, artifact-, and campaign-level
metrics from being mistaken for final-assay observation fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import numpy as np

from chemworld.data.records import trajectory_value
from chemworld.eval.explanations import score_mechanism_explanation
from chemworld.world.operations import PUBLIC_OBSERVATION_KEYS

TASK_METRIC_ENDPOINT_VERSION = "chemworld-task-metric-endpoints-1.0"

_MINIMIZE_METRICS = frozenset(
    {
        "constraint_violations",
        "cost",
        "crystal_fines_fraction",
        "equilibrium_residual",
        "process_mass_balance_error",
        "public_private_gap",
        "safety_risk",
        "solvent_loss",
    }
)
_TRAJECTORY_METRICS = {
    "constraint_violations": (
        "trajectory_constraint_event_count",
        ("trajectory_records",),
        "nonnegative_integer",
    ),
    "final_assay_score": (
        "last_committed_final_assay_score",
        ("trajectory_records",),
        "unit_interval",
    ),
    "sample_efficiency": (
        "threshold_hit_by_completed_experiment",
        ("trajectory_records", "task_threshold"),
        "unit_interval",
    ),
    "trajectory_validity": (
        "committed_valid_operation_fraction_with_final_assay_gate",
        ("trajectory_records",),
        "unit_interval",
    ),
    "validator_use": (
        "validator_evidence_fraction",
        ("trajectory_records",),
        "unit_interval",
    ),
}
_ARTIFACT_METRICS = {
    "explanation": "structured_explanation_rubric",
    "failure_analysis": "structured_failure_analysis_rubric",
    "mechanism_explanation": "transparent_mechanism_rubric",
}
_PREDICTIVE_METRICS = {
    "local_model_quality": "frozen_holdout_one_minus_rmse",
    "uncertainty": "frozen_holdout_normal_interval_score",
}
_PAIRED_SPLIT_METRICS = {
    "public_private_gap": "absolute_public_private_mean_shift",
    "rank_confidence": "world_bootstrap_pairwise_rank_agreement",
}


@dataclass(frozen=True)
class MetricEndpoint:
    """A serializable binding from a declared metric to its evaluator."""

    metric_id: str
    source_layer: str
    evaluator_id: str
    direction: str
    required_inputs: tuple[str, ...]
    output_contract: str
    implementation_status: str = "executable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "source_layer": self.source_layer,
            "evaluator_id": self.evaluator_id,
            "direction": self.direction,
            "required_inputs": list(self.required_inputs),
            "output_contract": self.output_contract,
            "implementation_status": self.implementation_status,
        }


def metric_endpoint(metric_id: str) -> MetricEndpoint:
    """Return the executable endpoint contract for one success metric."""

    direction = "minimize" if metric_id in _MINIMIZE_METRICS else "maximize"
    if metric_id in PUBLIC_OBSERVATION_KEYS:
        return MetricEndpoint(
            metric_id=metric_id,
            source_layer="terminal_observation",
            evaluator_id="last_committed_final_assay_observation",
            direction=direction,
            required_inputs=("trajectory_records",),
            output_contract="finite_scalar",
        )
    if metric_id in _TRAJECTORY_METRICS:
        evaluator_id, required_inputs, output_contract = _TRAJECTORY_METRICS[metric_id]
        return MetricEndpoint(
            metric_id=metric_id,
            source_layer="trajectory_aggregate",
            evaluator_id=evaluator_id,
            direction=direction,
            required_inputs=required_inputs,
            output_contract=output_contract,
        )
    if metric_id in _ARTIFACT_METRICS:
        return MetricEndpoint(
            metric_id=metric_id,
            source_layer="structured_artifact",
            evaluator_id=_ARTIFACT_METRICS[metric_id],
            direction=direction,
            required_inputs=("structured_explanation_artifact", "trajectory_records"),
            output_contract="unit_interval",
        )
    if metric_id in _PREDICTIVE_METRICS:
        return MetricEndpoint(
            metric_id=metric_id,
            source_layer="predictive_holdout",
            evaluator_id=_PREDICTIVE_METRICS[metric_id],
            direction=direction,
            required_inputs=(
                "frozen_holdout_targets",
                "predictive_means",
                "predictive_standard_deviations",
            ),
            output_contract="unit_interval",
        )
    if metric_id in _PAIRED_SPLIT_METRICS:
        return MetricEndpoint(
            metric_id=metric_id,
            source_layer="paired_split_campaign",
            evaluator_id=_PAIRED_SPLIT_METRICS[metric_id],
            direction=direction,
            required_inputs=(
                "aligned_public_world_scores_by_method",
                "aligned_private_world_scores_by_method",
            ),
            output_contract="unit_interval",
        )
    return MetricEndpoint(
        metric_id=metric_id,
        source_layer="unbound",
        evaluator_id="none",
        direction=direction,
        required_inputs=(),
        output_contract="undefined",
        implementation_status="missing",
    )


def build_task_metric_contract(success_metrics: tuple[str, ...]) -> dict[str, Any]:
    """Build and hash the ordered metric endpoint contract for a task."""

    endpoints = [metric_endpoint(metric).to_dict() for metric in success_metrics]
    payload = {
        "schema_version": TASK_METRIC_ENDPOINT_VERSION,
        "success_metrics": list(success_metrics),
        "endpoints": endpoints,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **payload,
        "all_metrics_bound": all(
            endpoint["implementation_status"] == "executable" for endpoint in endpoints
        ),
        "contract_hash": hashlib.sha256(canonical).hexdigest(),
    }


def evaluate_task_metrics(
    *,
    success_metrics: tuple[str, ...],
    records: list[dict[str, Any]],
    threshold: float,
    structured_artifact: dict[str, Any] | None = None,
    predictive_holdout: dict[str, Any] | None = None,
    paired_split: dict[str, Any] | None = None,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate all declared metrics without silently fabricating missing inputs."""

    if not records:
        raise ValueError("trajectory_records cannot be empty")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    predictive_values = (
        _predictive_holdout_values(predictive_holdout)
        if predictive_holdout is not None
        else None
    )
    paired_values = (
        _paired_split_values(
            paired_split,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if paired_split is not None
        else None
    )
    results: dict[str, dict[str, Any]] = {}
    for metric_id in success_metrics:
        endpoint = metric_endpoint(metric_id)
        if endpoint.implementation_status != "executable":
            results[metric_id] = _missing_result(endpoint, "metric endpoint is not implemented")
            continue
        if endpoint.source_layer == "terminal_observation":
            value = _terminal_observation_value(records, metric_id)
            results[metric_id] = _value_result(endpoint, value)
        elif endpoint.source_layer == "trajectory_aggregate":
            value, details = _trajectory_value(records, metric_id, float(threshold))
            results[metric_id] = _value_result(endpoint, value, details=details)
        elif endpoint.source_layer == "structured_artifact":
            if structured_artifact is None:
                results[metric_id] = _missing_result(
                    endpoint, "structured_explanation_artifact was not supplied"
                )
            else:
                value, details = _artifact_value(
                    records,
                    metric_id,
                    structured_artifact,
                )
                results[metric_id] = _value_result(endpoint, value, details=details)
        elif endpoint.source_layer == "predictive_holdout":
            if predictive_values is None:
                results[metric_id] = _missing_result(
                    endpoint, "frozen predictive holdout was not supplied"
                )
            else:
                results[metric_id] = _value_result(
                    endpoint,
                    predictive_values[metric_id],
                    details=predictive_values["details"],
                )
        elif endpoint.source_layer == "paired_split_campaign":
            if paired_values is None:
                results[metric_id] = _missing_result(
                    endpoint, "aligned public/private campaign was not supplied"
                )
            else:
                results[metric_id] = _value_result(
                    endpoint,
                    paired_values[metric_id],
                    details=paired_values["details"],
                )
    return {
        "schema_version": TASK_METRIC_ENDPOINT_VERSION,
        "metrics": results,
        "all_declared_metrics_implemented": all(
            result["implementation_status"] == "executable"
            for result in results.values()
        ),
        "all_required_inputs_present": all(
            result["evaluation_status"] == "evaluated" for result in results.values()
        ),
    }


def _missing_result(endpoint: MetricEndpoint, reason: str) -> dict[str, Any]:
    return {
        **endpoint.to_dict(),
        "evaluation_status": "not_evaluated_missing_input",
        "value": None,
        "details": {"reason": reason},
    }


def _value_result(
    endpoint: MetricEndpoint,
    value: float,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = _finite(value, endpoint.metric_id)
    return {
        **endpoint.to_dict(),
        "evaluation_status": "evaluated",
        "value": value,
        "details": details or {},
    }


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _final_assay_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if trajectory_value(record, 'leaderboard_score', None) is not None
        and record.get("transaction_status", record.get("info", {}).get("transaction_status"))
        in {None, "committed"}
    ]


def _terminal_observation_value(records: list[dict[str, Any]], metric_id: str) -> float:
    assays = _final_assay_records(records)
    if not assays:
        raise ValueError("trajectory has no committed final assay")
    final = assays[-1]
    if metric_id == "score":
        return _finite(trajectory_value(final, 'leaderboard_score'), "leaderboard_score")
    observation = trajectory_value(final, 'observation', {})
    if not isinstance(observation, dict) or observation.get(metric_id) is None:
        raise ValueError(f"final assay does not expose declared metric: {metric_id}")
    return _finite(observation[metric_id], f"observation.{metric_id}")


def _constraint_flags(record: dict[str, Any]) -> dict[str, Any]:
    flags = record.get("constraint_flags")
    if not isinstance(flags, dict):
        flags = record.get("info", {}).get("constraint_flags", {})
    return flags if isinstance(flags, dict) else {}


def _transaction_status(record: dict[str, Any]) -> str | None:
    status = record.get("transaction_status")
    if status is None:
        status = record.get("info", {}).get("transaction_status")
    return None if status is None else str(status)


def _trajectory_value(
    records: list[dict[str, Any]],
    metric_id: str,
    threshold: float,
) -> tuple[float, dict[str, Any]]:
    assays = _final_assay_records(records)
    if metric_id == "final_assay_score":
        if not assays:
            raise ValueError("trajectory has no committed final assay")
        value = _finite(trajectory_value(assays[-1], 'leaderboard_score'), "leaderboard_score")
        return value, {"committed_final_assay_count": len(assays)}
    if metric_id == "sample_efficiency":
        scores = [
            _finite(trajectory_value(record, "leaderboard_score"), "leaderboard_score")
            for record in assays
        ]
        hit = next((index + 1 for index, score in enumerate(scores) if score >= threshold), None)
        value = 0.0 if hit is None else 1.0 - (hit - 1) / max(len(scores), 1)
        return value, {
            "completed_experiment_count": len(scores),
            "threshold": threshold,
            "threshold_hit_complete_experiment": hit,
        }
    invalid_count = 0
    for record in records:
        flags = _constraint_flags(record)
        invalid_count += int(
            _transaction_status(record) not in {None, "committed"}
            or any(
                bool(flags.get(key, False))
                for key in (
                    "unsafe",
                    "precondition_failed",
                    "constitution_failed",
                    "phase_mass_balance_failed",
                )
            )
        )
    if metric_id == "constraint_violations":
        return float(invalid_count), {"trajectory_record_count": len(records)}
    if metric_id == "trajectory_validity":
        valid_fraction = 1.0 - invalid_count / len(records)
        value = valid_fraction if assays else 0.0
        return float(np.clip(value, 0.0, 1.0)), {
            "invalid_or_constraint_event_count": invalid_count,
            "committed_final_assay_count": len(assays),
            "final_assay_gate_passed": bool(assays),
        }
    if metric_id == "validator_use":
        trace = _latest_agent_trace(records)
        if not trace:
            return 0.0, {"trace_entry_count": 0, "validator_evidence_count": 0}
        validator_count = sum(_has_validator_evidence(entry) for entry in trace)
        return validator_count / len(trace), {
            "trace_entry_count": len(trace),
            "validator_evidence_count": validator_count,
        }
    raise ValueError(f"unsupported trajectory metric: {metric_id}")


def _latest_agent_trace(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in reversed(records):
        trace = record.get("agent_trace")
        if isinstance(trace, list):
            return [entry for entry in trace if isinstance(entry, dict)]
    return []


def _has_validator_evidence(entry: dict[str, Any]) -> bool:
    return bool(
        entry.get("validator_result")
        or entry.get("uses_validator_feedback")
        or entry.get("adaptation_source") == "validator"
    )


def _artifact_value(
    records: list[dict[str, Any]],
    metric_id: str,
    artifact: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    known_evidence_ids = {
        str(record.get("run_id") or record.get("step"))
        for record in records
        if record.get("run_id") is not None or record.get("step") is not None
    }
    cited = artifact.get("evidence_ids", [])
    if not isinstance(cited, list):
        cited = []
    cited_ids = {str(item) for item in cited}
    evidence_precision = (
        len(cited_ids & known_evidence_ids) / len(cited_ids) if cited_ids else 0.0
    )
    mechanism_payload = {
        "hypothesis": artifact.get("hypothesis", ""),
        "learned_mechanism": artifact.get("learned_mechanism", ""),
        "limitations": artifact.get("limitations", ""),
        "next_experiment": artifact.get("next_experiment", ""),
    }
    mechanism = score_mechanism_explanation(mechanism_payload).normalized
    failure_text = str(artifact.get("failure_analysis", "")).casefold()
    failure_items = {
        "failure_named": bool(failure_text.strip()),
        "cause": any(term in failure_text for term in ("cause", "because", "原因", "由于")),
        "limitation": any(
            term in failure_text for term in ("limit", "uncertain", "局限", "不确定")
        ),
        "corrective_action": any(
            term in failure_text
            for term in ("next", "correct", "mitigate", "下一", "改进", "缓解")
        ),
    }
    failure_score = fmean(float(value) for value in failure_items.values())
    if metric_id == "mechanism_explanation":
        value = 0.8 * mechanism + 0.2 * evidence_precision
    elif metric_id == "failure_analysis":
        value = 0.8 * failure_score + 0.2 * evidence_precision
    elif metric_id == "explanation":
        value = 0.4 * mechanism + 0.4 * failure_score + 0.2 * evidence_precision
    else:
        raise ValueError(f"unsupported artifact metric: {metric_id}")
    return float(np.clip(value, 0.0, 1.0)), {
        "mechanism_rubric_score": mechanism,
        "failure_rubric_score": failure_score,
        "evidence_reference_precision": evidence_precision,
        "known_evidence_id_count": len(known_evidence_ids),
        "cited_evidence_id_count": len(cited_ids),
    }


def _predictive_holdout_values(payload: dict[str, Any]) -> dict[str, Any]:
    targets = _unit_interval_array(payload.get("targets"), "targets")
    means = _unit_interval_array(payload.get("predictive_means"), "predictive_means")
    deviations = np.asarray(payload.get("predictive_standard_deviations"), dtype=float)
    if targets.size == 0 or targets.shape != means.shape or targets.shape != deviations.shape:
        raise ValueError("predictive holdout arrays must be non-empty and aligned")
    if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0.0):
        raise ValueError("predictive_standard_deviations must be finite and positive")
    rmse = float(np.sqrt(np.mean(np.square(means - targets))))
    alpha = 0.10
    z = 1.6448536269514722
    lower = means - z * deviations
    upper = means + z * deviations
    interval_scores = (upper - lower) + (2.0 / alpha) * (
        np.maximum(lower - targets, 0.0) + np.maximum(targets - upper, 0.0)
    )
    mean_interval_score = float(np.mean(interval_scores))
    return {
        "local_model_quality": float(np.clip(1.0 - rmse, 0.0, 1.0)),
        "uncertainty": 1.0 / (1.0 + mean_interval_score),
        "details": {
            "holdout_count": int(targets.size),
            "rmse": rmse,
            "normal_90_interval_mean_score": mean_interval_score,
        },
    }


def _unit_interval_array(value: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite one-dimensional array")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{field} must be in [0, 1]")
    return array


def _paired_split_values(
    payload: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    public = _score_matrix(payload.get("public_scores_by_method"), "public")
    private = _score_matrix(payload.get("private_scores_by_method"), "private")
    if set(public) != set(private) or len(public) < 2:
        raise ValueError("public/private campaigns require the same two or more methods")
    methods = sorted(public)
    world_count = len(public[methods[0]])
    if any(
        len(public[method]) != world_count or len(private[method]) != world_count
        for method in methods
    ):
        raise ValueError("all public/private method rows must align by world")
    public_means = np.asarray([np.mean(public[method]) for method in methods])
    private_means = np.asarray([np.mean(private[method]) for method in methods])
    absolute_gap = float(np.mean(np.abs(private_means - public_means)))
    rng = np.random.default_rng(bootstrap_seed)
    agreements = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sampled = rng.integers(0, world_count, size=world_count)
        sampled_public = [float(np.mean(public[method][sampled])) for method in methods]
        sampled_private = [float(np.mean(private[method][sampled])) for method in methods]
        agreements[index] = _pairwise_agreement(sampled_public, sampled_private)
    rank_confidence = float(np.mean(agreements))
    return {
        "public_private_gap": absolute_gap,
        "rank_confidence": rank_confidence,
        "details": {
            "method_count": len(methods),
            "world_count": world_count,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "rank_agreement_bootstrap_95_interval": [
                float(np.quantile(agreements, 0.025)),
                float(np.quantile(agreements, 0.975)),
            ],
        },
    }


def _score_matrix(value: Any, field: str) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        raise ValueError(f"{field}_scores_by_method must be an object")
    result = {
        str(method): _unit_interval_array(scores, f"{field}.{method}")
        for method, scores in value.items()
    }
    if not result or any(scores.size == 0 for scores in result.values()):
        raise ValueError(f"{field}_scores_by_method cannot be empty")
    return result


def _pairwise_agreement(left: list[float], right: list[float]) -> float:
    agreements: list[float] = []
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if left_delta == 0.0 and right_delta == 0.0:
                agreements.append(1.0)
            elif left_delta == 0.0 or right_delta == 0.0:
                agreements.append(0.5)
            else:
                agreements.append(float((left_delta > 0.0) == (right_delta > 0.0)))
    return float(fmean(agreements)) if agreements else 0.0


__all__ = [
    "TASK_METRIC_ENDPOINT_VERSION",
    "MetricEndpoint",
    "build_task_metric_contract",
    "evaluate_task_metrics",
    "metric_endpoint",
]
