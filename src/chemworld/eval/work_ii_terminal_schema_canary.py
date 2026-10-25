"""Development-only terminal prediction-schema canary evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from statistics import mean
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_b4_decision import (
    B4_ARMS,
    evaluate_b4_decision,
)
from chemworld.eval.work_ii_reviewer_followup import B3_FAMILIES, B3_METRIC_IDS
from chemworld.eval.work_ii_study_b import score_prediction_payload

CANARY_VERSION = "chemworld-work-ii-terminal-schema-canary-cell-0.1"
SUMMARY_VERSION = "chemworld-work-ii-terminal-schema-canary-summary-0.1"
FIXED_CONTEXT_SUMMARY_VERSION = (
    "chemworld-work-ii-terminal-schema-fixed-context-summary-0.1"
)
CONDITIONS = ("full_32", "lean_ranking")


def law_output_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "status",
            "mechanism_family",
            "estimated_reference_exponent",
            "confidence",
            "typed_law",
            "law_summary",
        ],
        "properties": {
            "status": {"type": "string", "const": "final_law_committed"},
            "mechanism_family": {"type": "string", "enum": list(B3_FAMILIES)},
            "estimated_reference_exponent": {
                "type": "number",
                "minimum": 0.25,
                "maximum": 3.0,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "typed_law": {
                "type": "object",
                "additionalProperties": False,
                "required": ["law_type", "mechanism_family", "reference_exponent"],
                "properties": {
                    "law_type": {
                        "type": "string",
                        "const": "reference_coefficient_power",
                    },
                    "mechanism_family": {
                        "type": "string",
                        "enum": list(B3_FAMILIES),
                    },
                    "reference_exponent": {
                        "type": "number",
                        "minimum": 0.25,
                        "maximum": 3.0,
                    },
                },
            },
            "law_summary": {"type": "string", "minLength": 1, "maxLength": 1200},
        },
    }


def terminal_output_schema(
    action_queries: Sequence[Mapping[str, Any]], *, condition: str
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown terminal schema condition: {condition}")
    query_ids = [str(item["query_id"]) for item in action_queries]
    properties: dict[str, Any] = {
        "status": {"type": "string", "const": "terminal_ranking_complete"},
        "ranking": {
            "type": "array",
            "minItems": len(query_ids),
            "maxItems": len(query_ids),
            "uniqueItems": True,
            "items": {"type": "string", "enum": query_ids},
        },
        "selected_action_query_id": {"type": "string", "enum": query_ids},
        "selection_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "mechanism_application": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1200,
        },
    }
    required = [
        "status",
        "ranking",
        "selected_action_query_id",
        "selection_confidence",
        "mechanism_application",
    ]
    if condition == "full_32":
        properties["predictions"] = {
            "type": "array",
            "minItems": len(query_ids),
            "maxItems": len(query_ids),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query_id", "metrics"],
                "properties": {
                    "query_id": {"type": "string", "enum": query_ids},
                    "metrics": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(B3_METRIC_IDS),
                        "properties": {
                            metric_id: {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            }
                            for metric_id in B3_METRIC_IDS
                        },
                    },
                },
            },
        }
        required.append("predictions")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def validate_law_payload(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "final_law_committed":
        errors.append("law status is invalid")
    family = payload.get("mechanism_family")
    exponent = payload.get("estimated_reference_exponent")
    confidence = payload.get("confidence")
    typed = payload.get("typed_law")
    if family not in B3_FAMILIES:
        errors.append("law mechanism family is invalid")
    if (
        isinstance(exponent, bool)
        or not isinstance(exponent, int | float)
        or not 0.25 <= float(exponent) <= 3.0
    ):
        errors.append("law exponent is invalid")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("law confidence is invalid")
    if not isinstance(typed, Mapping):
        errors.append("typed law is unavailable")
    else:
        if typed.get("law_type") != "reference_coefficient_power":
            errors.append("typed law type is invalid")
        if typed.get("mechanism_family") != family:
            errors.append("typed law family differs from the committed family")
        typed_exponent = typed.get("reference_exponent")
        if (
            isinstance(typed_exponent, bool)
            or not isinstance(typed_exponent, int | float)
            or not isinstance(exponent, int | float)
            or isinstance(exponent, bool)
            or not math.isclose(float(typed_exponent), float(exponent), abs_tol=1.0e-12)
        ):
            errors.append("typed law exponent differs from the committed exponent")
    if not isinstance(payload.get("law_summary"), str) or not payload["law_summary"].strip():
        errors.append("law summary is unavailable")
    return errors


def validate_terminal_payload(
    payload: Mapping[str, Any],
    action_queries: Sequence[Mapping[str, Any]],
    *,
    condition: str,
) -> list[str]:
    errors: list[str] = []
    query_ids = {str(item["query_id"]) for item in action_queries}
    if payload.get("status") != "terminal_ranking_complete":
        errors.append("terminal status is invalid")
    ranking = payload.get("ranking")
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(query_ids)
        or set(map(str, ranking)) != query_ids
    ):
        errors.append("terminal ranking denominator is invalid")
    selected = payload.get("selected_action_query_id")
    if not isinstance(selected, str) or selected not in query_ids:
        errors.append("terminal selected action is invalid")
    elif not isinstance(ranking, list) or not ranking or selected != ranking[0]:
        errors.append("terminal selected action is not ranking[0]")
    confidence = payload.get("selection_confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("terminal selection confidence is invalid")
    if (
        not isinstance(payload.get("mechanism_application"), str)
        or not payload["mechanism_application"].strip()
    ):
        errors.append("terminal mechanism application is unavailable")
    predictions = payload.get("predictions")
    if condition == "lean_ranking":
        if predictions is not None:
            errors.append("lean terminal payload unexpectedly contains predictions")
        return errors
    if condition != "full_32":
        errors.append("terminal condition is invalid")
        return errors
    if not isinstance(predictions, list) or len(predictions) != len(query_ids):
        errors.append("full terminal prediction denominator is invalid")
        return errors
    observed: dict[str, set[str]] = {}
    for item in predictions:
        if not isinstance(item, Mapping) or not isinstance(item.get("metrics"), Mapping):
            errors.append("full terminal prediction row is malformed")
            continue
        query_id = str(item.get("query_id"))
        if query_id in observed:
            errors.append("full terminal prediction query IDs are not unique")
        observed[query_id] = {str(metric) for metric in item["metrics"]}
        for value in item["metrics"].values():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append("full terminal prediction value is invalid")
    if set(observed) != query_ids:
        errors.append("full terminal prediction query coverage is invalid")
    if any(metrics != set(B3_METRIC_IDS) for metrics in observed.values()):
        errors.append("full terminal prediction metric coverage is invalid")
    return errors


def _ranking_tau(ranking: Sequence[str], truth_order: Sequence[str]) -> float:
    predicted_position = {query_id: index for index, query_id in enumerate(ranking)}
    truth_position = {query_id: index for index, query_id in enumerate(truth_order)}
    concordant = 0
    discordant = 0
    for left, right in combinations(truth_order, 2):
        predicted_delta = predicted_position[left] - predicted_position[right]
        truth_delta = truth_position[left] - truth_position[right]
        if predicted_delta * truth_delta > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = concordant + discordant
    return 0.0 if denominator == 0 else (concordant - discordant) / denominator


def evaluate_terminal_payload(
    cell: Mapping[str, Any], payload: Mapping[str, Any], *, condition: str
) -> dict[str, Any]:
    ranking = [str(item) for item in payload["ranking"]]
    truth = cell["scoring_truth"]
    truth_order = sorted(
        map(str, truth),
        key=lambda query_id: (-float(truth[query_id]["score"]), query_id),
    )
    decision = evaluate_b4_decision(
        cell,
        {
            "decision_type": "execute_candidate",
            "selected_action_query_id": str(payload["selected_action_query_id"]),
        },
    )
    result: dict[str, Any] = {
        **decision,
        "participant_ranking": ranking,
        "hidden_truth_ranking": truth_order,
        "ranking_kendall_tau": _ranking_tau(ranking, truth_order),
        "selection_confidence": float(payload["selection_confidence"]),
    }
    if condition == "full_32":
        result["prediction_evaluation"] = score_prediction_payload(payload, truth)
    return result


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _terminal_usage_delta(receipts: Sequence[Mapping[str, Any]], field: str) -> float:
    if len(receipts) < 2:
        return 0.0
    law_usage = receipts[0].get("usage")
    terminal_usage = receipts[1].get("usage")
    law_usage = law_usage if isinstance(law_usage, Mapping) else {}
    terminal_usage = terminal_usage if isinstance(terminal_usage, Mapping) else {}
    law_value = law_usage.get(field)
    terminal_value = terminal_usage.get(field)
    if not isinstance(terminal_value, int | float) or isinstance(terminal_value, bool):
        return 0.0
    if not isinstance(law_value, int | float) or isinstance(law_value, bool):
        return float(terminal_value)
    return max(0.0, float(terminal_value) - float(law_value))


def summarize_canary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        members = [item for item in results if item.get("condition") == condition]
        completed = [item for item in members if item.get("status") == "completed"]
        evaluations = [
            item["terminal_evaluation"]
            for item in completed
            if isinstance(item.get("terminal_evaluation"), Mapping)
        ]
        receipt_pairs = [
            item["provider_receipts"]
            for item in completed
            if isinstance(item.get("provider_receipts"), list)
            and len(item["provider_receipts"]) >= 2
            and all(isinstance(receipt, Mapping) for receipt in item["provider_receipts"][:2])
        ]
        by_condition[condition] = {
            "scheduled_session_count": len(members),
            "completed_session_count": len(completed),
            "failed_session_count": len(members) - len(completed),
            "participant_schema_failure_count": sum(
                item.get("failure", {}).get("classification") == "participant_schema"
                for item in members
                if isinstance(item.get("failure"), Mapping)
            ),
            "mean_selected_rank": _mean_or_none(
                [float(item["selected_rank"]) for item in evaluations]
            ),
            "mean_normalized_regret": _mean_or_none(
                [float(item["normalized_policy_regret"]) for item in evaluations]
            ),
            "mean_ranking_kendall_tau": _mean_or_none(
                [float(item["ranking_kendall_tau"]) for item in evaluations]
            ),
            "mean_selected_minus_random": _mean_or_none(
                [float(item["selected_minus_random_candidate_mean"]) for item in evaluations]
            ),
            "mean_terminal_elapsed_s": _mean_or_none(
                [float(receipts[1].get("elapsed_s", 0.0)) for receipts in receipt_pairs]
            ),
            "mean_terminal_output_tokens": _mean_or_none(
                [
                    _terminal_usage_delta(receipts, "output_tokens")
                    for receipts in receipt_pairs
                ]
            ),
            "mean_terminal_reasoning_output_tokens": _mean_or_none(
                [
                    _terminal_usage_delta(receipts, "reasoning_output_tokens")
                    for receipts in receipt_pairs
                ]
            ),
            "mean_prediction_mae": _mean_or_none(
                [
                    float(item["prediction_evaluation"]["mean_normalized_absolute_error"])
                    for item in evaluations
                    if isinstance(item.get("prediction_evaluation"), Mapping)
                ]
            ),
        }
    matched_rows: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in results:
        if item.get("status") == "completed" and item.get("arm") in B4_ARMS:
            by_arm[str(item["arm"])][str(item["condition"])] = item
    for arm in B4_ARMS:
        pair = by_arm.get(arm, {})
        if set(pair) != set(CONDITIONS):
            continue
        full = pair["full_32"]["terminal_evaluation"]
        lean = pair["lean_ranking"]["terminal_evaluation"]
        matched_rows.append(
            {
                "arm": arm,
                "lean_minus_full_selected_rank": (
                    float(lean["selected_rank"]) - float(full["selected_rank"])
                ),
                "lean_minus_full_normalized_regret": (
                    float(lean["normalized_policy_regret"])
                    - float(full["normalized_policy_regret"])
                ),
                "lean_minus_full_ranking_kendall_tau": (
                    float(lean["ranking_kendall_tau"])
                    - float(full["ranking_kendall_tau"])
                ),
            }
        )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "status": (
            "completed"
            if len(results) == 6 and all(item.get("status") == "completed" for item in results)
            else "incomplete"
        ),
        "scheduled_session_count": 6,
        "observed_session_count": len(results),
        "completed_session_count": sum(item.get("status") == "completed" for item in results),
        "failed_session_count": sum(item.get("status") != "completed" for item in results),
        "participant_physical_experiment_count": 0,
        "by_condition": by_condition,
        "matched_arm_rows": matched_rows,
        "cell_results": [deepcopy(dict(item)) for item in results],
        "interpretation_status": "development_descriptive_only",
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    return summary


def summarize_fixed_context_replay(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        members = [item for item in results if item.get("condition") == condition]
        completed = [item for item in members if item.get("status") == "completed"]
        evaluations = [
            item["terminal_evaluation"]
            for item in completed
            if isinstance(item.get("terminal_evaluation"), Mapping)
        ]
        receipts = [
            item["provider_receipts"][0]
            for item in completed
            if isinstance(item.get("provider_receipts"), list)
            and item["provider_receipts"]
            and isinstance(item["provider_receipts"][0], Mapping)
        ]

        def provider_error_associated(item: Mapping[str, Any]) -> bool:
            item_receipts = item.get("provider_receipts")
            return bool(
                isinstance(item_receipts, list)
                and any(
                    isinstance(receipt, Mapping) and bool(receipt.get("provider_errors"))
                    for receipt in item_receipts
                )
            )

        by_condition[condition] = {
            "scheduled_session_count": len(members),
            "completed_session_count": len(completed),
            "failed_session_count": len(members) - len(completed),
            "participant_schema_failure_count": sum(
                item.get("failure", {}).get("classification") == "participant_schema"
                for item in members
                if isinstance(item.get("failure"), Mapping)
            ),
            "provider_error_associated_failure_count": sum(
                item.get("status") != "completed" and provider_error_associated(item)
                for item in members
            ),
            "payload_failure_without_provider_error_count": sum(
                item.get("status") != "completed" and not provider_error_associated(item)
                for item in members
            ),
            "mean_elapsed_s": _mean_or_none(
                [float(item.get("elapsed_s", 0.0)) for item in receipts]
            ),
            "mean_output_tokens": _mean_or_none(
                [float(item.get("usage", {}).get("output_tokens", 0)) for item in receipts]
            ),
            "mean_reasoning_output_tokens": _mean_or_none(
                [
                    float(item.get("usage", {}).get("reasoning_output_tokens", 0))
                    for item in receipts
                ]
            ),
            "mean_selected_rank": _mean_or_none(
                [float(item["selected_rank"]) for item in evaluations]
            ),
            "mean_normalized_regret": _mean_or_none(
                [float(item["normalized_policy_regret"]) for item in evaluations]
            ),
            "mean_ranking_kendall_tau": _mean_or_none(
                [float(item["ranking_kendall_tau"]) for item in evaluations]
            ),
            "mean_selected_minus_random": _mean_or_none(
                [float(item["selected_minus_random_candidate_mean"]) for item in evaluations]
            ),
            "mean_prediction_mae": _mean_or_none(
                [
                    float(item["prediction_evaluation"]["mean_normalized_absolute_error"])
                    for item in evaluations
                    if isinstance(item.get("prediction_evaluation"), Mapping)
                ]
            ),
        }
    by_replicate: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in results:
        replicate = item.get("replicate")
        if (
            item.get("status") == "completed"
            and isinstance(replicate, int)
            and not isinstance(replicate, bool)
        ):
            by_replicate[replicate][str(item["condition"])] = item
    paired_rows: list[dict[str, Any]] = []
    for replicate in sorted(by_replicate):
        pair = by_replicate[replicate]
        if set(pair) != set(CONDITIONS):
            continue
        full = pair["full_32"]["terminal_evaluation"]
        lean = pair["lean_ranking"]["terminal_evaluation"]
        paired_rows.append(
            {
                "replicate": replicate,
                "lean_minus_full_selected_rank": (
                    float(lean["selected_rank"]) - float(full["selected_rank"])
                ),
                "lean_minus_full_normalized_regret": (
                    float(lean["normalized_policy_regret"])
                    - float(full["normalized_policy_regret"])
                ),
                "lean_minus_full_ranking_kendall_tau": (
                    float(lean["ranking_kendall_tau"])
                    - float(full["ranking_kendall_tau"])
                ),
            }
        )
    summary: dict[str, Any] = {
        "schema_version": FIXED_CONTEXT_SUMMARY_VERSION,
        "status": (
            "completed"
            if len(results) == 6 and all(item.get("status") == "completed" for item in results)
            else "incomplete"
        ),
        "scheduled_session_count": 6,
        "observed_session_count": len(results),
        "completed_session_count": sum(item.get("status") == "completed" for item in results),
        "failed_session_count": sum(item.get("status") != "completed" for item in results),
        "participant_physical_experiment_count": 0,
        "fixed_mechanism_family": "FAMILY_B_POWER",
        "fixed_reference_exponent": 1.75,
        "by_condition": by_condition,
        "paired_replicate_rows": paired_rows,
        "cell_results": [deepcopy(dict(item)) for item in results],
        "interpretation_status": "development_fixed_context_descriptive_only",
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    return summary


__all__ = [
    "CANARY_VERSION",
    "CONDITIONS",
    "FIXED_CONTEXT_SUMMARY_VERSION",
    "SUMMARY_VERSION",
    "evaluate_terminal_payload",
    "law_output_schema",
    "summarize_canary",
    "summarize_fixed_context_replay",
    "terminal_output_schema",
    "validate_law_payload",
    "validate_terminal_payload",
]
