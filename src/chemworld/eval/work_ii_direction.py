"""Direction diagnostics shared by Work II readiness and post-run evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _prediction_rows_by_query(predictions: object) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in _sequence(predictions):
        if not isinstance(row, Mapping):
            continue
        query_id = str(row.get("query_id", ""))
        metrics = {
            str(metric.get("metric_id")): float(metric["mean"])
            for metric in _sequence(row.get("metrics"))
            if isinstance(metric, Mapping)
            and isinstance(metric.get("mean"), int | float)
            and not isinstance(metric.get("mean"), bool)
        }
        if query_id and metrics:
            result[query_id] = metrics
    return result


def temperature_direction_diagnostic(
    predictions: object,
    *,
    truth_plan: Mapping[str, Any],
    reference_temperature_K: float,
    temperature_tolerance_K: float,
) -> dict[str, Any]:
    by_query = _prediction_rows_by_query(predictions)
    grouped: dict[float, dict[str, list[float]]] = {}
    for query in _sequence(truth_plan.get("queries")):
        if not isinstance(query, Mapping):
            continue
        features = _mapping(query.get("feature_values"))
        query_id = str(query.get("query_id", ""))
        metrics = by_query.get(query_id)
        temperature = features.get("reaction_temperature_K")
        duration = features.get("reaction_duration_s")
        if (
            metrics is None
            or "score" not in metrics
            or isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or isinstance(duration, bool)
            or not isinstance(duration, int | float)
        ):
            continue
        if float(temperature) <= reference_temperature_K - temperature_tolerance_K:
            side = "lower_temperature"
        elif float(temperature) >= reference_temperature_K + temperature_tolerance_K:
            side = "higher_temperature"
        else:
            continue
        grouped.setdefault(float(duration), {}).setdefault(side, []).append(metrics["score"])
    contrasts: list[float] = []
    for sides in grouped.values():
        lower = sides.get("lower_temperature", [])
        higher = sides.get("higher_temperature", [])
        if lower and higher:
            contrasts.append(sum(lower) / len(lower) - sum(higher) / len(higher))
    mean_contrast = sum(contrasts) / len(contrasts) if contrasts else None
    preferred_side = (
        None
        if mean_contrast is None or mean_contrast == 0.0
        else "lower_temperature"
        if mean_contrast > 0.0
        else "higher_temperature"
    )
    return {
        "paired_duration_count": len(contrasts),
        "lower_minus_higher_mean_score_contrast": mean_contrast,
        "preferred_side": preferred_side,
    }


def controlled_potential_direction_diagnostic(
    predictions: object,
    *,
    truth_plan: Mapping[str, Any],
    reference_context: Mapping[str, Any],
    tolerance_V: float = 0.02,
) -> dict[str, Any]:
    """Compare lower/higher controlled-potential sides at matched current values."""

    by_query = _prediction_rows_by_query(predictions)
    probe = reference_context.get("probe_potential_V")
    if isinstance(probe, bool) or not isinstance(probe, int | float):
        return {
            "paired_current_count": 0,
            "lower_minus_higher_mean_score_contrast": None,
            "preferred_side": None,
        }
    grouped: dict[float, dict[str, list[float]]] = {}
    for query in _sequence(truth_plan.get("queries")):
        if not isinstance(query, Mapping):
            continue
        features = _mapping(query.get("feature_values"))
        metrics = by_query.get(str(query.get("query_id", "")))
        potential = features.get("controlled_potential_V")
        current = features.get("controlled_current_mA")
        if (
            metrics is None
            or "score" not in metrics
            or isinstance(potential, bool)
            or not isinstance(potential, int | float)
            or isinstance(current, bool)
            or not isinstance(current, int | float)
        ):
            continue
        if float(potential) <= float(probe) - float(tolerance_V):
            side = "lower_controlled_potential"
        elif float(potential) >= float(probe) + float(tolerance_V):
            side = "higher_controlled_potential"
        else:
            continue
        grouped.setdefault(round(float(current), 8), {}).setdefault(side, []).append(
            metrics["score"]
        )
    contrasts: list[float] = []
    for sides in grouped.values():
        lower = sides.get("lower_controlled_potential", [])
        higher = sides.get("higher_controlled_potential", [])
        if lower and higher:
            contrasts.append(sum(lower) / len(lower) - sum(higher) / len(higher))
    mean_contrast = sum(contrasts) / len(contrasts) if contrasts else None
    preferred_side = (
        None
        if mean_contrast is None or mean_contrast == 0.0
        else "lower_controlled_potential"
        if mean_contrast > 0.0
        else "higher_controlled_potential"
    )
    return {
        "paired_current_count": len(contrasts),
        "lower_minus_higher_mean_score_contrast": mean_contrast,
        "preferred_side": preferred_side,
    }


def registered_temperature_direction(config: Mapping[str, Any]) -> dict[str, Any]:
    aligned = _mapping(_mapping(config.get("prior_arms")).get("aligned_nominal"))
    initial_model = _mapping(aligned.get("initial_world_model"))
    claim = _mapping(_mapping(initial_model.get("model")).get("claim"))
    relation = claim.get("expected_relation")
    if not isinstance(relation, str):
        return {"preferred_side": None, "source": None, "claim": None}
    higher = "higher-temperature side" in relation
    lower = "lower-temperature side" in relation
    source = "aligned_nominal.initial_world_model.model.claim.expected_relation"
    if not higher or not lower:
        return {"preferred_side": None, "source": source, "claim": relation}
    preferred_side = (
        "higher_temperature"
        if relation.index("higher-temperature side") < relation.index("lower-temperature side")
        else "lower_temperature"
    )
    return {"preferred_side": preferred_side, "source": source, "claim": relation}


def registered_control_direction(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read a controlled-potential/current directional claim from an initial model."""

    aligned = _mapping(_mapping(config.get("prior_arms")).get("aligned_nominal"))
    initial_model = _mapping(aligned.get("initial_world_model"))
    claim = _mapping(_mapping(initial_model.get("model")).get("claim"))
    relation = claim.get("expected_relation")
    if not isinstance(relation, str):
        return {"preferred_side": None, "source": None, "claim": None, "axis": None}
    for axis in ("potential", "current"):
        higher_label = f"higher-controlled-{axis} side"
        lower_label = f"lower-controlled-{axis} side"
        if higher_label in relation and lower_label in relation:
            preferred_side = (
                f"higher_controlled_{axis}"
                if relation.index(higher_label) < relation.index(lower_label)
                else f"lower_controlled_{axis}"
            )
            return {
                "preferred_side": preferred_side,
                "source": ("aligned_nominal.initial_world_model.model.claim.expected_relation"),
                "claim": relation,
                "axis": axis,
            }
    return {
        "preferred_side": None,
        "source": "aligned_nominal.initial_world_model.model.claim.expected_relation",
        "claim": relation,
        "axis": None,
    }


def temperature_direction_contract(
    registered: Mapping[str, Any],
    held_out_truth: Mapping[str, Any],
) -> dict[str, Any]:
    registered_side = registered.get("preferred_side")
    held_out_side = held_out_truth.get("preferred_side")
    if registered_side is None or held_out_side is None:
        status = "undefined"
    elif registered_side == held_out_side:
        status = "stable"
    else:
        status = "query_subset_conflict"
    return {
        "status": status,
        "registered_side": registered_side,
        "held_out_truth_side": held_out_side,
        "recovery_scoring_authorized": status == "stable",
        "interpretation": (
            "Binary direction recovery is scored only when the frozen aligned-prior direction and "
            "the evaluator-held query direction agree. Prediction and executable-law errors remain "
            "scored directly against evaluator truth regardless of this diagnostic."
        ),
    }


def truth_prediction_rows(evaluator_truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": str(query_id),
            "metrics": [
                {"metric_id": str(metric_id), "mean": float(value)}
                for metric_id, value in _mapping(metrics).items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            ],
        }
        for query_id, metrics in evaluator_truth.items()
    ]


__all__ = [
    "controlled_potential_direction_diagnostic",
    "registered_control_direction",
    "registered_temperature_direction",
    "temperature_direction_contract",
    "temperature_direction_diagnostic",
    "truth_prediction_rows",
]
