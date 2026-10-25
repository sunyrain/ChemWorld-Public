"""Frozen held-out predictive checks for electrochemical S0 world understanding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Protocol

import numpy as np

from chemworld._serialization import canonical_json_sha256 as _canonical_sha256
from chemworld.agents.electrochemical_single_stage import (
    ELECTROCHEMICAL_SINGLE_STAGE_MEASUREMENT_SLOTS,
    electrochemical_single_stage_parameters_from_unit_vector,
    electrochemical_single_stage_unit_vector_from_parameters,
)
from chemworld.agents.task_recipes import (
    electrochemical_recipe_parameters_from_unit_vector,
    electrochemical_recipe_unit_vector_from_parameters,
)
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_ADAPTIVE_TWO_STAGE,
    ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE,
    normalize_electrochemical_workflow_mode,
)

ELECTROCHEMICAL_PREDICTIVE_VERSION = "chemworld-electrochemical-predictive-0.1-s0-dev"
ELECTROCHEMICAL_SINGLE_STAGE_PREDICTIVE_VERSION = (
    "chemworld-electrochemical-predictive-0.4-material-single-stage-s0-dev"
)
ELECTROCHEMICAL_STANDARDIZED_PREDICTIVE_VERSION = (
    "chemworld-electrochemical-predictive-0.3-standardized-single-stage-s0-dev"
)
STANDARDIZED_PREDICTIVE_ANCHOR_ID = "balanced-standardized-anchor-v0.1"
PREDICTION_DIRECTIONS = ("increase", "decrease", "no_material_change")
PREDICTIVE_QUERY_COUNT = 3
PREDICTIVE_PAIRED_REPLICATE_COUNT = 2
PREDICTIVE_PHYSICAL_EXPERIMENT_COUNT = (
    PREDICTIVE_QUERY_COUNT * PREDICTIVE_PAIRED_REPLICATE_COUNT * 2
)
PREDICTIVE_DIRECTION_THRESHOLD = 0.01
PREDICTIVE_MEASUREMENT_SLOTS = (
    "diagnostic-01-ph_meter",
    "diagnostic-02-uvvis",
    "diagnostic-03-uvvis",
)
PREDICTIVE_QUERY_METRICS: dict[str, tuple[str, ...]] = {
    "controlled_potential_V": (
        "selective_product_yield",
        "energy_efficiency",
        "leaderboard_score",
    ),
    "controlled_current_mA": (
        "electrochemical_conversion",
        "ohmic_efficiency",
        "leaderboard_score",
    ),
    "electrolyte_profile": ("transport_efficiency", "pH_normalized", "leaderboard_score"),
}
SINGLE_STAGE_PREDICTIVE_MEASUREMENT_SLOTS = tuple(
    str(item["slot_id"]) for item in ELECTROCHEMICAL_SINGLE_STAGE_MEASUREMENT_SLOTS
)
SINGLE_STAGE_PREDICTIVE_QUERY_METRICS: dict[str, tuple[str, ...]] = {
    "potential_V": (
        "selective_product_yield",
        "energy_efficiency",
        "leaderboard_score",
    ),
    "current_mA": (
        "electrochemical_conversion",
        "ohmic_efficiency",
        "leaderboard_score",
    ),
    "electrolyte_profile": (
        "transport_efficiency",
        "pH_normalized",
        "leaderboard_score",
    ),
}
PREDICTIVE_METRIC_SOURCES = {
    metric_id: (
        "terminal_summary.leaderboard_score"
        if metric_id == "leaderboard_score"
        else "measurement_evidence.closeout-final-assay.processed_estimate"
    )
    for metrics in PREDICTIVE_QUERY_METRICS.values()
    for metric_id in metrics
}
_DIRECTION_THRESHOLDS = {
    "selective_product_yield": PREDICTIVE_DIRECTION_THRESHOLD,
    "electrochemical_conversion": PREDICTIVE_DIRECTION_THRESHOLD,
    "energy_efficiency": PREDICTIVE_DIRECTION_THRESHOLD,
    "ohmic_efficiency": PREDICTIVE_DIRECTION_THRESHOLD,
    "transport_efficiency": PREDICTIVE_DIRECTION_THRESHOLD,
    "pH_normalized": PREDICTIVE_DIRECTION_THRESHOLD,
    "leaderboard_score": PREDICTIVE_DIRECTION_THRESHOLD,
}
STANDARDIZED_PREDICTIVE_REFERENCE_PARAMETERS: dict[str, int | float] = {
    "electrolyte_profile": 1,
    "solvent": 0,
    "reagent_amount_mol": 0.015,
    "potential_V": 0.8,
    "current_mA": 180.0,
    "duration_s": 2100.0,
}
STANDARDIZED_PREDICTIVE_INTERVENTIONS: dict[str, dict[str, int | float]] = {
    "potential_V": {
        **STANDARDIZED_PREDICTIVE_REFERENCE_PARAMETERS,
        "potential_V": 1.0,
    },
    "current_mA": {
        **STANDARDIZED_PREDICTIVE_REFERENCE_PARAMETERS,
        "current_mA": 220.0,
    },
    "electrolyte_profile": {
        **STANDARDIZED_PREDICTIVE_REFERENCE_PARAMETERS,
        "electrolyte_profile": 2,
    },
}


def predictive_schema_version(electrochemical_workflow_mode: str) -> str:
    mode = normalize_electrochemical_workflow_mode(electrochemical_workflow_mode)
    if mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE:
        return ELECTROCHEMICAL_SINGLE_STAGE_PREDICTIVE_VERSION
    return ELECTROCHEMICAL_PREDICTIVE_VERSION


def predictive_measurement_slots(
    electrochemical_workflow_mode: str,
) -> tuple[str, ...]:
    mode = normalize_electrochemical_workflow_mode(electrochemical_workflow_mode)
    if mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE:
        return SINGLE_STAGE_PREDICTIVE_MEASUREMENT_SLOTS
    return PREDICTIVE_MEASUREMENT_SLOTS


def predictive_query_metrics(
    electrochemical_workflow_mode: str,
) -> dict[str, tuple[str, ...]]:
    mode = normalize_electrochemical_workflow_mode(electrochemical_workflow_mode)
    if mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE:
        return SINGLE_STAGE_PREDICTIVE_QUERY_METRICS
    return PREDICTIVE_QUERY_METRICS


@dataclass(frozen=True)
class ElectrochemicalPredictionQuery:
    schema_version: str
    standardized_measurement_slots: tuple[str, ...]
    query_id: str
    reference_experiment_index: int
    intervention_variable: str
    reference_recipe_parameters: dict[str, int | float]
    intervention_recipe_parameters: dict[str, int | float]
    metric_ids: tuple[str, ...]
    metric_sources: dict[str, str]
    direction_thresholds: dict[str, float]
    query_sha256: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "reference_experiment_index": self.reference_experiment_index,
            "intervention_variable": self.intervention_variable,
            "reference_recipe_parameters": dict(self.reference_recipe_parameters),
            "intervention_recipe_parameters": dict(self.intervention_recipe_parameters),
            "metric_ids": list(self.metric_ids),
            "metric_sources": dict(self.metric_sources),
            "direction_thresholds": dict(self.direction_thresholds),
            "direction_labels": list(PREDICTION_DIRECTIONS),
            "standardized_measurement_slots": list(
                self.standardized_measurement_slots
            ),
            "query_sha256": self.query_sha256,
        }


@dataclass(frozen=True)
class CounterfactualMetricPrediction:
    metric_id: str
    direction: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "direction": self.direction,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CounterfactualQueryPrediction:
    query_id: str
    metric_predictions: tuple[CounterfactualMetricPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "metric_predictions": [item.to_dict() for item in self.metric_predictions],
        }


class _CounterfactualPredictionQuery(Protocol):
    @property
    def query_id(self) -> str: ...

    @property
    def metric_ids(self) -> tuple[str, ...]: ...


def build_electrochemical_prediction_queries(
    history: Sequence[Mapping[str, Any]],
    *,
    electrochemical_workflow_mode: str = (
        ELECTROCHEMICAL_WORKFLOW_ADAPTIVE_TWO_STAGE
    ),
) -> tuple[ElectrochemicalPredictionQuery, ...]:
    """Build three deterministic unseen one-factor interventions from the incumbent."""

    if not history:
        raise ValueError("predictive queries require completed exploration history")
    workflow_mode = normalize_electrochemical_workflow_mode(
        electrochemical_workflow_mode
    )
    query_metrics = predictive_query_metrics(workflow_mode)
    measurement_slots = predictive_measurement_slots(workflow_mode)
    schema_version = predictive_schema_version(workflow_mode)
    history_hashes = {
        _canonical_sha256(_recipe_parameters(item["plan"], workflow_mode))
        for item in history
    }
    reference, prepared_queries = _select_reference_and_interventions(
        history,
        electrochemical_workflow_mode=workflow_mode,
        history_hashes=history_hashes,
    )
    reference_index = int(reference["experiment_index"])
    base = _recipe_parameters(reference["plan"], workflow_mode)
    queries: list[ElectrochemicalPredictionQuery] = []
    for query_id, variable, intervention in prepared_queries:
        metrics = query_metrics[variable]
        thresholds = {
            metric_id: _DIRECTION_THRESHOLDS[metric_id] for metric_id in metrics
        }
        query_core = _query_core(
            query_id=query_id,
            reference_experiment_index=reference_index,
            intervention_variable=variable,
            reference_recipe_parameters=base,
            intervention_recipe_parameters=intervention,
            metric_ids=metrics,
            metric_sources={
                metric_id: PREDICTIVE_METRIC_SOURCES[metric_id]
                for metric_id in metrics
            },
            direction_thresholds=thresholds,
            schema_version=schema_version,
            standardized_measurement_slots=measurement_slots,
        )
        queries.append(
            ElectrochemicalPredictionQuery(
                schema_version=schema_version,
                standardized_measurement_slots=measurement_slots,
                query_id=query_id,
                reference_experiment_index=reference_index,
                intervention_variable=variable,
                reference_recipe_parameters=dict(base),
                intervention_recipe_parameters=intervention,
                metric_ids=metrics,
                metric_sources={
                    metric_id: PREDICTIVE_METRIC_SOURCES[metric_id]
                    for metric_id in metrics
                },
                direction_thresholds=thresholds,
                query_sha256=_canonical_sha256(query_core),
            )
        )
    validate_prediction_queries(
        queries,
        history=history,
        electrochemical_workflow_mode=workflow_mode,
    )
    return tuple(queries)


def build_standardized_electrochemical_prediction_queries(
) -> tuple[ElectrochemicalPredictionQuery, ...]:
    """Build the history-independent anchor qualified across balanced S0 worlds."""

    reference = dict(STANDARDIZED_PREDICTIVE_REFERENCE_PARAMETERS)
    electrochemical_single_stage_unit_vector_from_parameters(reference)
    query_ids = {
        "potential_V": "standardized-potential",
        "current_mA": "standardized-current",
        "electrolyte_profile": "standardized-electrolyte-profile",
    }
    queries: list[ElectrochemicalPredictionQuery] = []
    for variable in ("potential_V", "current_mA", "electrolyte_profile"):
        intervention = dict(STANDARDIZED_PREDICTIVE_INTERVENTIONS[variable])
        electrochemical_single_stage_unit_vector_from_parameters(intervention)
        metrics = SINGLE_STAGE_PREDICTIVE_QUERY_METRICS[variable]
        thresholds = {
            metric_id: _DIRECTION_THRESHOLDS[metric_id] for metric_id in metrics
        }
        sources = {
            metric_id: PREDICTIVE_METRIC_SOURCES[metric_id]
            for metric_id in metrics
        }
        query_id = query_ids[variable]
        core = _query_core(
            query_id=query_id,
            reference_experiment_index=-1,
            intervention_variable=variable,
            reference_recipe_parameters=reference,
            intervention_recipe_parameters=intervention,
            metric_ids=metrics,
            metric_sources=sources,
            direction_thresholds=thresholds,
            schema_version=ELECTROCHEMICAL_STANDARDIZED_PREDICTIVE_VERSION,
            standardized_measurement_slots=SINGLE_STAGE_PREDICTIVE_MEASUREMENT_SLOTS,
        )
        queries.append(
            ElectrochemicalPredictionQuery(
                schema_version=ELECTROCHEMICAL_STANDARDIZED_PREDICTIVE_VERSION,
                standardized_measurement_slots=(
                    SINGLE_STAGE_PREDICTIVE_MEASUREMENT_SLOTS
                ),
                query_id=query_id,
                reference_experiment_index=-1,
                intervention_variable=variable,
                reference_recipe_parameters=dict(reference),
                intervention_recipe_parameters=intervention,
                metric_ids=metrics,
                metric_sources=sources,
                direction_thresholds=thresholds,
                query_sha256=_canonical_sha256(core),
            )
        )
    validate_standardized_prediction_queries(queries)
    return tuple(queries)


def validate_standardized_prediction_queries(
    queries: Sequence[ElectrochemicalPredictionQuery],
) -> None:
    expected_ids = (
        "standardized-potential",
        "standardized-current",
        "standardized-electrolyte-profile",
    )
    if tuple(query.query_id for query in queries) != expected_ids:
        raise ValueError("standardized predictive query IDs or order changed")
    reference = dict(STANDARDIZED_PREDICTIVE_REFERENCE_PARAMETERS)
    for query in queries:
        if query.schema_version != ELECTROCHEMICAL_STANDARDIZED_PREDICTIVE_VERSION:
            raise ValueError("standardized predictive schema version changed")
        if query.reference_experiment_index != -1:
            raise ValueError("standardized predictive reference must be history-independent")
        if query.reference_recipe_parameters != reference:
            raise ValueError("standardized predictive reference recipe changed")
        expected_intervention = STANDARDIZED_PREDICTIVE_INTERVENTIONS[
            query.intervention_variable
        ]
        if query.intervention_recipe_parameters != expected_intervention:
            raise ValueError("standardized predictive intervention recipe changed")
        changed = {
            key
            for key in reference
            if reference[key] != query.intervention_recipe_parameters[key]
        }
        if changed != {query.intervention_variable}:
            raise ValueError("standardized predictive query is not one-factor")
        expected_metrics = SINGLE_STAGE_PREDICTIVE_QUERY_METRICS[
            query.intervention_variable
        ]
        if query.metric_ids != expected_metrics:
            raise ValueError("standardized predictive metrics changed")
        expected_hash = _canonical_sha256(
            _query_core(
                query_id=query.query_id,
                reference_experiment_index=-1,
                intervention_variable=query.intervention_variable,
                reference_recipe_parameters=reference,
                intervention_recipe_parameters=expected_intervention,
                metric_ids=query.metric_ids,
                metric_sources=query.metric_sources,
                direction_thresholds=query.direction_thresholds,
                schema_version=query.schema_version,
                standardized_measurement_slots=query.standardized_measurement_slots,
            )
        )
        if query.query_sha256 != expected_hash:
            raise ValueError("standardized predictive query hash mismatch")


def validate_prediction_queries(
    queries: Sequence[ElectrochemicalPredictionQuery],
    *,
    history: Sequence[Mapping[str, Any]],
    electrochemical_workflow_mode: str = (
        ELECTROCHEMICAL_WORKFLOW_ADAPTIVE_TWO_STAGE
    ),
) -> None:
    workflow_mode = normalize_electrochemical_workflow_mode(
        electrochemical_workflow_mode
    )
    expected_query_metrics = predictive_query_metrics(workflow_mode)
    expected_slots = predictive_measurement_slots(workflow_mode)
    expected_schema_version = predictive_schema_version(workflow_mode)
    expected_query_ids = (
        (
            "predictive-potential",
            "predictive-current",
            "predictive-electrolyte-profile",
        )
        if workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
        else (
            "predictive-controlled-potential",
            "predictive-controlled-current",
            "predictive-electrolyte-profile",
        )
    )
    if len(queries) != PREDICTIVE_QUERY_COUNT or tuple(
        query.query_id for query in queries
    ) != expected_query_ids:
        raise ValueError("predictive query IDs or order do not match the frozen contract")
    if not history:
        raise ValueError("predictive queries require completed exploration history")
    expected_reference, _ = _select_reference_and_interventions(
        history,
        electrochemical_workflow_mode=workflow_mode,
        history_hashes={
            _canonical_sha256(_recipe_parameters(item["plan"], workflow_mode))
            for item in history
        },
    )
    expected_reference_index = int(expected_reference["experiment_index"])
    expected_reference_recipe = _recipe_parameters(
        expected_reference["plan"], workflow_mode
    )
    history_hashes = {
        _canonical_sha256(_recipe_parameters(item["plan"], workflow_mode))
        for item in history
    }
    for query in queries:
        if query.schema_version != expected_schema_version:
            raise ValueError("predictive query schema version mismatch")
        if query.standardized_measurement_slots != expected_slots:
            raise ValueError("predictive query measurement slots mismatch")
        if query.reference_experiment_index != expected_reference_index:
            raise ValueError("predictive query reference does not match the frozen tie rule")
        if query.reference_recipe_parameters != expected_reference_recipe:
            raise ValueError("predictive query reference recipe does not match exploration history")
        _unit_vector_from_parameters(
            dict(query.reference_recipe_parameters), workflow_mode
        )
        changed = {
            key
            for key in query.reference_recipe_parameters
            if query.reference_recipe_parameters[key]
            != query.intervention_recipe_parameters[key]
        }
        if changed != {query.intervention_variable}:
            raise ValueError(
                f"predictive query {query.query_id!r} is not a one-factor intervention"
            )
        _unit_vector_from_parameters(
            dict(query.intervention_recipe_parameters), workflow_mode
        )
        if _canonical_sha256(query.intervention_recipe_parameters) in history_hashes:
            raise ValueError(f"predictive query {query.query_id!r} repeats an explored recipe")
        if query.metric_ids != expected_query_metrics[query.intervention_variable]:
            raise ValueError(f"predictive query {query.query_id!r} has the wrong metric set")
        expected_thresholds = {
            metric_id: _DIRECTION_THRESHOLDS[metric_id]
            for metric_id in query.metric_ids
        }
        if query.direction_thresholds != expected_thresholds:
            raise ValueError(
                f"predictive query {query.query_id!r} has the wrong direction thresholds"
            )
        expected_sources = {
            metric_id: PREDICTIVE_METRIC_SOURCES[metric_id]
            for metric_id in query.metric_ids
        }
        if query.metric_sources != expected_sources:
            raise ValueError(
                f"predictive query {query.query_id!r} has the wrong metric sources"
            )
        expected_hash = _canonical_sha256(
            _query_core(
                query_id=query.query_id,
                reference_experiment_index=query.reference_experiment_index,
                intervention_variable=query.intervention_variable,
                reference_recipe_parameters=query.reference_recipe_parameters,
                intervention_recipe_parameters=query.intervention_recipe_parameters,
                metric_ids=query.metric_ids,
                metric_sources=query.metric_sources,
                direction_thresholds=query.direction_thresholds,
                schema_version=query.schema_version,
                standardized_measurement_slots=(
                    query.standardized_measurement_slots
                ),
            )
        )
        if query.query_sha256 != expected_hash:
            raise ValueError(f"predictive query {query.query_id!r} hash mismatch")


def parse_counterfactual_predictions(
    payload: object,
    *,
    queries: Sequence[_CounterfactualPredictionQuery],
) -> tuple[CounterfactualQueryPrediction, ...]:
    if not isinstance(payload, list):
        raise ValueError("counterfactual_predictions must be a list")
    if len(payload) != len(queries):
        raise ValueError("counterfactual_predictions must answer every frozen query exactly once")
    payload_by_id: dict[str, Mapping[str, Any]] = {}
    for item in payload:
        if not isinstance(item, Mapping) or set(item) != {"query_id", "metric_predictions"}:
            raise ValueError("counterfactual prediction fields do not match the contract")
        query_id = str(item["query_id"])
        if query_id in payload_by_id:
            raise ValueError("counterfactual prediction query IDs must be unique")
        payload_by_id[query_id] = item
    predictions: list[CounterfactualQueryPrediction] = []
    for query in queries:
        item = payload_by_id.get(query.query_id)
        if item is None:
            raise ValueError(f"missing counterfactual prediction for {query.query_id!r}")
        metrics_payload = item["metric_predictions"]
        if not isinstance(metrics_payload, list):
            raise ValueError("metric_predictions must be a list")
        metrics_by_id: dict[str, CounterfactualMetricPrediction] = {}
        for metric_payload in metrics_payload:
            if not isinstance(metric_payload, Mapping) or set(metric_payload) != {
                "metric_id",
                "direction",
                "confidence",
            }:
                raise ValueError("metric prediction fields do not match the contract")
            metric_id = str(metric_payload["metric_id"])
            if metric_id in metrics_by_id:
                raise ValueError("metric prediction IDs must be unique within a query")
            direction = str(metric_payload["direction"])
            if direction not in PREDICTION_DIRECTIONS:
                raise ValueError("metric prediction direction is unsupported")
            confidence = metric_payload["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, int | float):
                raise ValueError("metric prediction confidence must be numeric")
            confidence_float = float(confidence)
            if not isfinite(confidence_float) or not 0.0 <= confidence_float <= 1.0:
                raise ValueError("metric prediction confidence must be finite and in [0,1]")
            metrics_by_id[metric_id] = CounterfactualMetricPrediction(
                metric_id=metric_id,
                direction=direction,
                confidence=confidence_float,
            )
        if set(metrics_by_id) != set(query.metric_ids):
            raise ValueError(f"counterfactual prediction for {query.query_id!r} has wrong metrics")
        predictions.append(
            CounterfactualQueryPrediction(
                query_id=query.query_id,
                metric_predictions=tuple(
                    metrics_by_id[metric_id] for metric_id in query.metric_ids
                ),
            )
        )
    if set(payload_by_id) != {query.query_id for query in queries}:
        raise ValueError("counterfactual_predictions contains an unknown query ID")
    return tuple(predictions)


def classify_metric_direction(delta: float, threshold: float) -> str:
    if not isfinite(delta) or threshold <= 0.0 or not isfinite(threshold):
        raise ValueError("predictive direction requires finite delta and positive threshold")
    if delta >= threshold:
        return "increase"
    if delta <= -threshold:
        return "decrease"
    return "no_material_change"


def score_predictive_validation(
    predictions: Sequence[CounterfactualQueryPrediction],
    query_results: Sequence[Mapping[str, Any]],
    *,
    queries: Sequence[ElectrochemicalPredictionQuery],
) -> dict[str, Any]:
    query_map = {query.query_id: query for query in queries}
    if len(query_map) != len(queries):
        raise ValueError("predictive queries must have unique query IDs")
    prediction_map = {
        (query.query_id, metric.metric_id): metric
        for query in predictions
        for metric in query.metric_predictions
    }
    expected_prediction_keys = {
        (query.query_id, metric_id)
        for query in queries
        for metric_id in query.metric_ids
    }
    if set(prediction_map) != expected_prediction_keys:
        raise ValueError("predictive predictions do not exactly cover the frozen query set")
    result_map: dict[str, Mapping[str, Any]] = {}
    for result in query_results:
        query_id = str(result["query_id"])
        if query_id in result_map:
            raise ValueError("predictive validation query results must be unique")
        result_map[query_id] = result
    if set(result_map) != set(query_map):
        raise ValueError("predictive validation results do not match the frozen query set")
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = query.query_id
        result = result_map[query_id]
        if str(result.get("query_sha256")) != query.query_sha256:
            raise ValueError("predictive validation query hash mismatch")
        metrics_payload = result.get("metric_results")
        if not isinstance(metrics_payload, list):
            raise ValueError("predictive metric_results must be a list")
        metric_map: dict[str, Mapping[str, Any]] = {}
        for metric in metrics_payload:
            if not isinstance(metric, Mapping):
                raise ValueError("predictive metric result must be an object")
            metric_id = str(metric["metric_id"])
            if metric_id in metric_map:
                raise ValueError("predictive metric results must be unique within a query")
            metric_map[metric_id] = metric
        if set(metric_map) != set(query.metric_ids):
            raise ValueError("predictive metric results do not match the frozen metric set")
        for metric_id in query.metric_ids:
            metric = metric_map[metric_id]
            prediction = prediction_map.get((query_id, metric_id))
            if prediction is None:
                raise ValueError("predictive validation result has no matching prediction")
            actual_direction = str(metric["actual_direction"])
            if actual_direction not in PREDICTION_DIRECTIONS:
                raise ValueError("predictive validation actual direction is unsupported")
            reference_mean = float(metric["reference_mean"])
            intervention_mean = float(metric["intervention_mean"])
            delta = float(metric["delta"])
            threshold = float(metric["direction_threshold"])
            metric_source = str(metric["metric_source"])
            if not all(
                isfinite(value)
                for value in (reference_mean, intervention_mean, delta, threshold)
            ):
                raise ValueError("predictive validation metric values must be finite")
            if not isclose(
                delta,
                intervention_mean - reference_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("predictive validation delta does not match paired means")
            if threshold != query.direction_thresholds[metric_id]:
                raise ValueError("predictive validation direction threshold mismatch")
            if metric_source != query.metric_sources[metric_id]:
                raise ValueError("predictive validation metric source mismatch")
            if actual_direction != classify_metric_direction(delta, threshold):
                raise ValueError("predictive validation actual direction was misclassified")
            correct = prediction.direction == actual_direction
            rows.append(
                {
                    "query_id": query_id,
                    "metric_id": metric_id,
                    "predicted_direction": prediction.direction,
                    "actual_direction": actual_direction,
                    "confidence": prediction.confidence,
                    "correct": correct,
                    "brier_term": (prediction.confidence - float(correct)) ** 2,
                    "reference_mean": reference_mean,
                    "intervention_mean": intervention_mean,
                    "delta": delta,
                    "direction_threshold": threshold,
                    "metric_source": metric_source,
                }
            )
    if len(rows) != len(prediction_map):
        raise ValueError("predictive validation did not score every prediction exactly once")
    schema_versions = {query.schema_version for query in queries}
    if len(schema_versions) != 1:
        raise ValueError("predictive queries mix schema versions")
    return {
        "schema_version": next(iter(schema_versions)),
        "query_count": len(queries),
        "prediction_count": len(rows),
        "directional_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "confidence_brier_score": sum(row["brier_term"] for row in rows) / len(rows),
        "nontrivial_actual_effect_rate": (
            sum(row["actual_direction"] != "no_material_change" for row in rows) / len(rows)
        ),
        "rows": rows,
    }


def metric_value_from_result(result: Mapping[str, Any], metric_id: str) -> float:
    if metric_id == "leaderboard_score":
        return float(result["terminal_summary"]["leaderboard_score"])
    final_assays = [
        evidence
        for evidence in result["measurement_evidence"]
        if evidence.get("measurement_slot_id") == "closeout-final-assay"
    ]
    if len(final_assays) != 1:
        raise ValueError("predictive result must contain exactly one closeout final assay")
    processed = final_assays[0].get("processed_estimate")
    if not isinstance(processed, Mapping) or metric_id not in processed:
        raise ValueError(f"predictive metric {metric_id!r} is missing from final assay")
    value = float(processed[metric_id])
    if not isfinite(value):
        raise ValueError(f"predictive metric {metric_id!r} must be finite")
    return value


def _recipe_parameters(
    plan: Mapping[str, Any], electrochemical_workflow_mode: str
) -> dict[str, int | float]:
    parameters = plan.get("recipe_parameters")
    if isinstance(parameters, Mapping):
        vector = _unit_vector_from_parameters(
            dict(parameters), electrochemical_workflow_mode
        )
    else:
        vector = np.asarray(plan["search_vector"], dtype=float)
    return _parameters_from_unit_vector(vector, electrochemical_workflow_mode)


def _unit_vector_from_parameters(
    parameters: Mapping[str, int | float], electrochemical_workflow_mode: str
) -> np.ndarray:
    if (
        normalize_electrochemical_workflow_mode(electrochemical_workflow_mode)
        == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
    ):
        return electrochemical_single_stage_unit_vector_from_parameters(parameters)
    return electrochemical_recipe_unit_vector_from_parameters(dict(parameters))


def _parameters_from_unit_vector(
    vector: np.ndarray, electrochemical_workflow_mode: str
) -> dict[str, int | float]:
    if (
        normalize_electrochemical_workflow_mode(electrochemical_workflow_mode)
        == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
    ):
        return electrochemical_single_stage_parameters_from_unit_vector(vector)
    return electrochemical_recipe_parameters_from_unit_vector(vector)


def _select_reference(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for item in history:
        score = float(item["terminal_summary"]["leaderboard_score"])
        experiment_index = int(item["experiment_index"])
        if not isfinite(score) or experiment_index < 0:
            raise ValueError("predictive reference selection requires finite scored history")
        scored.append((score, experiment_index, item))
    return max(scored, key=lambda item: (item[0], -item[1]))[2]


def _select_reference_and_interventions(
    history: Sequence[Mapping[str, Any]],
    *,
    electrochemical_workflow_mode: str,
    history_hashes: set[str],
) -> tuple[
    Mapping[str, Any],
    tuple[tuple[str, str, dict[str, int | float]], ...],
]:
    workflow_mode = normalize_electrochemical_workflow_mode(
        electrochemical_workflow_mode
    )
    candidates: tuple[Mapping[str, Any], ...]
    if workflow_mode != ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE:
        candidates = (_select_reference(history),)
    else:
        candidates = tuple(
            item
            for _score, _negative_index, item in sorted(
                (
                    (
                        float(item["terminal_summary"]["leaderboard_score"]),
                        -int(item["experiment_index"]),
                        item,
                    )
                    for item in history
                ),
                reverse=True,
            )
        )
    for reference in candidates:
        base = _recipe_parameters(reference["plan"], workflow_mode)
        prepared: list[tuple[str, str, dict[str, int | float]]] = []
        try:
            for query_id, variable, values in _query_specs(base, workflow_mode):
                intervention = _first_valid_unseen_intervention(
                    base,
                    variable=variable,
                    candidates=values,
                    history_hashes=history_hashes,
                    electrochemical_workflow_mode=workflow_mode,
                )
                prepared.append((query_id, variable, intervention))
        except ValueError:
            continue
        return reference, tuple(prepared)
    raise ValueError(
        "cannot construct an unseen valid intervention for a complete predictive query set"
    )


def _query_specs(
    base: Mapping[str, int | float], workflow_mode: str
) -> tuple[tuple[str, str, tuple[int | float, ...]], ...]:
    potential_variable = (
        "potential_V"
        if workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
        else "controlled_potential_V"
    )
    current_variable = (
        "current_mA"
        if workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
        else "controlled_current_mA"
    )
    return (
        (
            (
                "predictive-potential"
                if workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
                else "predictive-controlled-potential"
            ),
            potential_variable,
            _potential_candidates(base, variable=potential_variable),
        ),
        (
            (
                "predictive-current"
                if workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
                else "predictive-controlled-current"
            ),
            current_variable,
            _current_candidates(base, variable=current_variable),
        ),
        (
            "predictive-electrolyte-profile",
            "electrolyte_profile",
            tuple(
                (int(base["electrolyte_profile"]) + offset) % 4
                for offset in (1, 2, 3)
            ),
        ),
    )


def _query_core(
    *,
    query_id: str,
    reference_experiment_index: int,
    intervention_variable: str,
    reference_recipe_parameters: Mapping[str, int | float],
    intervention_recipe_parameters: Mapping[str, int | float],
    metric_ids: Sequence[str],
    metric_sources: Mapping[str, str],
    direction_thresholds: Mapping[str, float],
    schema_version: str,
    standardized_measurement_slots: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "query_id": query_id,
        "reference_experiment_index": reference_experiment_index,
        "intervention_variable": intervention_variable,
        "reference_recipe_parameters": dict(reference_recipe_parameters),
        "intervention_recipe_parameters": dict(intervention_recipe_parameters),
        "metric_ids": list(metric_ids),
        "metric_sources": dict(metric_sources),
        "direction_thresholds": dict(direction_thresholds),
        "standardized_measurement_slots": list(standardized_measurement_slots),
    }


def _potential_candidates(
    base: Mapping[str, int | float], *, variable: str
) -> tuple[float, ...]:
    current = float(base[variable])
    return (
        *(current + delta for delta in (0.20, -0.20, 0.35, -0.35)),
        0.65,
        0.85,
        1.05,
        1.25,
        1.45,
        1.65,
    )


def _current_candidates(
    base: Mapping[str, int | float], *, variable: str
) -> tuple[float, ...]:
    current = float(base[variable])
    return (
        *(current + delta for delta in (40.0, -40.0, 70.0, -70.0)),
        20.0,
        50.0,
        80.0,
        120.0,
        170.0,
        210.0,
    )


def _first_valid_unseen_intervention(
    base: Mapping[str, int | float],
    *,
    variable: str,
    candidates: Sequence[int | float],
    history_hashes: set[str],
    electrochemical_workflow_mode: str,
) -> dict[str, int | float]:
    for value in candidates:
        intervention = dict(base)
        intervention[variable] = int(value) if variable == "electrolyte_profile" else float(value)
        if intervention[variable] == base[variable]:
            continue
        try:
            vector = _unit_vector_from_parameters(
                intervention, electrochemical_workflow_mode
            )
        except ValueError:
            continue
        normalized = _parameters_from_unit_vector(
            vector, electrochemical_workflow_mode
        )
        if _canonical_sha256(normalized) not in history_hashes:
            return normalized
    raise ValueError(f"cannot construct an unseen valid intervention for {variable!r}")


__all__ = [
    "ELECTROCHEMICAL_PREDICTIVE_VERSION",
    "ELECTROCHEMICAL_SINGLE_STAGE_PREDICTIVE_VERSION",
    "ELECTROCHEMICAL_STANDARDIZED_PREDICTIVE_VERSION",
    "PREDICTION_DIRECTIONS",
    "PREDICTIVE_DIRECTION_THRESHOLD",
    "PREDICTIVE_MEASUREMENT_SLOTS",
    "PREDICTIVE_METRIC_SOURCES",
    "PREDICTIVE_PAIRED_REPLICATE_COUNT",
    "PREDICTIVE_PHYSICAL_EXPERIMENT_COUNT",
    "PREDICTIVE_QUERY_COUNT",
    "PREDICTIVE_QUERY_METRICS",
    "SINGLE_STAGE_PREDICTIVE_MEASUREMENT_SLOTS",
    "SINGLE_STAGE_PREDICTIVE_QUERY_METRICS",
    "STANDARDIZED_PREDICTIVE_ANCHOR_ID",
    "STANDARDIZED_PREDICTIVE_INTERVENTIONS",
    "STANDARDIZED_PREDICTIVE_REFERENCE_PARAMETERS",
    "CounterfactualMetricPrediction",
    "CounterfactualQueryPrediction",
    "ElectrochemicalPredictionQuery",
    "build_electrochemical_prediction_queries",
    "build_standardized_electrochemical_prediction_queries",
    "classify_metric_direction",
    "metric_value_from_result",
    "parse_counterfactual_predictions",
    "predictive_measurement_slots",
    "predictive_query_metrics",
    "predictive_schema_version",
    "score_predictive_validation",
    "validate_prediction_queries",
    "validate_standardized_prediction_queries",
]
