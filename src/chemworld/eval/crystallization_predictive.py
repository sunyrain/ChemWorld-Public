"""Frozen held-out predictive checks for reaction-to-crystallization S0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from chemworld.agents.crystallization_single_stage import (
    CRYSTALLIZATION_SINGLE_STAGE_MEASUREMENT_SLOTS,
    crystallization_single_stage_parameters_from_unit_vector,
    crystallization_single_stage_unit_vector_from_parameters,
)

CRYSTALLIZATION_PREDICTIVE_VERSION = (
    "chemworld-reaction-crystallization-predictive-0.1-s0-dev"
)
PREDICTIVE_QUERY_COUNT = 3
PREDICTIVE_PAIRED_REPLICATE_COUNT = 2
PREDICTIVE_PHYSICAL_EXPERIMENT_COUNT = (
    PREDICTIVE_QUERY_COUNT * PREDICTIVE_PAIRED_REPLICATE_COUNT * 2
)
PREDICTIVE_DIRECTION_THRESHOLD = 0.01
PREDICTIVE_MEASUREMENT_SLOTS = tuple(
    str(item["slot_id"]) for item in CRYSTALLIZATION_SINGLE_STAGE_MEASUREMENT_SLOTS
)
PREDICTIVE_QUERY_METRICS = {
    "reaction_temperature_K": ("yield", "selectivity", "leaderboard_score"),
    "seed_mass_g": ("crystal_purity", "crystal_size", "leaderboard_score"),
    "crystallization_temperature_K": (
        "crystal_yield",
        "crystal_fines_fraction",
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


@dataclass(frozen=True)
class CrystallizationPredictionQuery:
    schema_version: str
    query_id: str
    reference_experiment_index: int
    intervention_variable: str
    reference_recipe_parameters: dict[str, int | float]
    intervention_recipe_parameters: dict[str, int | float]
    metric_ids: tuple[str, ...]
    metric_sources: dict[str, str]
    direction_thresholds: dict[str, float]
    standardized_measurement_slots: tuple[str, ...]
    query_sha256: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CRYSTALLIZATION_PREDICTIVE_VERSION,
            "query_id": self.query_id,
            "reference_experiment_index": self.reference_experiment_index,
            "intervention_variable": self.intervention_variable,
            "reference_recipe_parameters": dict(self.reference_recipe_parameters),
            "intervention_recipe_parameters": dict(self.intervention_recipe_parameters),
            "metric_ids": list(self.metric_ids),
            "metric_sources": dict(self.metric_sources),
            "direction_thresholds": dict(self.direction_thresholds),
            "standardized_measurement_slots": list(self.standardized_measurement_slots),
            "query_sha256": self.query_sha256,
        }


def build_crystallization_prediction_queries(
    history: Sequence[Mapping[str, Any]],
) -> tuple[CrystallizationPredictionQuery, ...]:
    if not history:
        raise ValueError("crystallization predictive validation requires experiment history")
    history_vectors = {_vector_hash(_parameters_from_history(item)) for item in history}
    reference, prepared = _select_reference_and_interventions(history, history_vectors)
    reference_parameters = _parameters_from_history(reference)
    reference_index = int(reference["experiment_index"])
    queries: list[CrystallizationPredictionQuery] = []
    for query_id, variable, intervention in prepared:
        metric_ids = tuple(PREDICTIVE_QUERY_METRICS[variable])
        metric_sources = {
            metric_id: PREDICTIVE_METRIC_SOURCES[metric_id]
            for metric_id in metric_ids
        }
        direction_thresholds = dict.fromkeys(
            metric_ids, PREDICTIVE_DIRECTION_THRESHOLD
        )
        core = {
            "schema_version": CRYSTALLIZATION_PREDICTIVE_VERSION,
            "query_id": query_id,
            "reference_experiment_index": reference_index,
            "intervention_variable": variable,
            "reference_recipe_parameters": dict(reference_parameters),
            "intervention_recipe_parameters": dict(intervention),
            "metric_ids": list(metric_ids),
            "metric_sources": metric_sources,
            "direction_thresholds": direction_thresholds,
            "standardized_measurement_slots": list(PREDICTIVE_MEASUREMENT_SLOTS),
        }
        queries.append(
            CrystallizationPredictionQuery(
                schema_version=CRYSTALLIZATION_PREDICTIVE_VERSION,
                query_id=query_id,
                reference_experiment_index=reference_index,
                intervention_variable=variable,
                reference_recipe_parameters=dict(reference_parameters),
                intervention_recipe_parameters=dict(intervention),
                metric_ids=metric_ids,
                metric_sources=dict(metric_sources),
                direction_thresholds=dict(direction_thresholds),
                standardized_measurement_slots=PREDICTIVE_MEASUREMENT_SLOTS,
                query_sha256=_canonical_sha256(core),
            )
        )
    return tuple(queries)


def _select_reference_and_interventions(
    history: Sequence[Mapping[str, Any]], history_vectors: set[str]
) -> tuple[
    Mapping[str, Any], tuple[tuple[str, str, dict[str, int | float]], ...]
]:
    candidates = sorted(
        history,
        key=lambda item: (
            float(item["terminal_summary"]["leaderboard_score"]),
            -int(item["experiment_index"]),
        ),
        reverse=True,
    )
    for reference in candidates:
        base = _parameters_from_history(reference)
        prepared: list[tuple[str, str, dict[str, int | float]]] = []
        try:
            for query_id, variable, values in _query_specs(base):
                intervention = _first_valid_unseen_intervention(
                    base,
                    variable=variable,
                    candidates=values,
                    history_vectors=history_vectors,
                )
                prepared.append((query_id, variable, intervention))
        except ValueError:
            continue
        return reference, tuple(prepared)
    raise ValueError(
        "cannot construct an unseen valid intervention for a complete crystallization query set"
    )


def _query_specs(
    base: Mapping[str, int | float],
) -> tuple[tuple[str, str, tuple[float, ...]], ...]:
    reaction_temperature = float(base["reaction_temperature_K"])
    seed_mass = float(base["seed_mass_g"])
    crystallization_temperature = float(base["crystallization_temperature_K"])
    coupled_maximum = min(315.0, reaction_temperature - 55.0)
    return (
        (
            "predictive-reaction-temperature",
            "reaction_temperature_K",
            (
                reaction_temperature + 20.0,
                reaction_temperature - 20.0,
                reaction_temperature + 35.0,
                reaction_temperature - 35.0,
                333.15,
                378.15,
                423.15,
            ),
        ),
        (
            "predictive-seed-mass",
            "seed_mass_g",
            (
                seed_mass + 0.004,
                seed_mass - 0.004,
                seed_mass + 0.008,
                seed_mass - 0.008,
                0.001,
                0.006,
                0.010,
                0.015,
            ),
        ),
        (
            "predictive-crystallization-temperature",
            "crystallization_temperature_K",
            (
                crystallization_temperature + 10.0,
                crystallization_temperature - 10.0,
                crystallization_temperature + 20.0,
                crystallization_temperature - 20.0,
                270.0,
                278.15,
                288.15,
                coupled_maximum,
            ),
        ),
    )


def _first_valid_unseen_intervention(
    base: Mapping[str, int | float],
    *,
    variable: str,
    candidates: Sequence[float],
    history_vectors: set[str],
) -> dict[str, int | float]:
    for value in candidates:
        intervention = dict(base)
        intervention[variable] = float(value)
        try:
            vector = crystallization_single_stage_unit_vector_from_parameters(intervention)
        except ValueError:
            continue
        normalized = crystallization_single_stage_parameters_from_unit_vector(vector)
        if normalized == dict(base) or _vector_hash(normalized) in history_vectors:
            continue
        return normalized
    raise ValueError(f"no valid unseen crystallization intervention for {variable}")


def _parameters_from_history(item: Mapping[str, Any]) -> dict[str, int | float]:
    plan = item.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("predictive history is missing a plan")
    parameters = plan.get("recipe_parameters")
    if isinstance(parameters, Mapping):
        vector = crystallization_single_stage_unit_vector_from_parameters(parameters)
        return crystallization_single_stage_parameters_from_unit_vector(vector)
    return crystallization_single_stage_parameters_from_unit_vector(
        np.asarray(plan.get("search_vector"), dtype=float)
    )


def _vector_hash(parameters: Mapping[str, int | float]) -> str:
    vector = crystallization_single_stage_unit_vector_from_parameters(parameters)
    return _canonical_sha256([float(value) for value in vector])


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CRYSTALLIZATION_PREDICTIVE_VERSION",
    "PREDICTIVE_DIRECTION_THRESHOLD",
    "PREDICTIVE_MEASUREMENT_SLOTS",
    "PREDICTIVE_METRIC_SOURCES",
    "PREDICTIVE_PAIRED_REPLICATE_COUNT",
    "PREDICTIVE_PHYSICAL_EXPERIMENT_COUNT",
    "PREDICTIVE_QUERY_COUNT",
    "PREDICTIVE_QUERY_METRICS",
    "CrystallizationPredictionQuery",
    "build_crystallization_prediction_queries",
]
