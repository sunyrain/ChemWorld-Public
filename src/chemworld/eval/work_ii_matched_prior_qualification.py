"""Pure construction and gates for Work II matched local priors."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

MATCHED_PRIOR_VERSION = "chemworld-work-ii-matched-prior-qualification-0.3"
TEMPERATURES_K = tuple(float(value) for value in range(370, 471, 10))
DURATIONS_S = tuple(float(value) for value in range(300, 6301, 600))
BASELINE_TEMPERATURE_K = 420.0
BASELINE_DURATION_S = 3300.0
BASELINE_TEMPERATURE_TOLERANCE_K = 10.0
BASELINE_DURATION_TOLERANCE_S = 600.0
MINIMUM_SAFE_FIT_COUNT = 24
MINIMUM_SAFE_HELD_OUT_COUNT = 40
HELD_OUT_QUERY_COUNT = 16
MAXIMUM_ALIGNED_NORMALIZED_MAE = 0.20
MAXIMUM_BASELINE_UTILITY_GAP = 0.05
MINIMUM_DISAGREEMENT_FRACTION = 0.25
MINIMUM_BLIND_ERROR_MARGIN = 0.05
MINIMUM_REGION_SUPPORT = 3
MINIMUM_REPRESENTATIVE_GRID_DISTANCE = 4


def select_reference_candidate(
    source_report: Mapping[str, Any],
    *,
    maximum_score_gap: float = 0.05,
    minimum_non_target_distance: float = 0.05,
) -> dict[str, Any]:
    analysis = _mapping(source_report.get("analysis"), "analysis")
    optimum = _mapping(analysis.get("oracle_optimum"), "analysis.oracle_optimum")
    optimum_vector = _float_vector(optimum.get("vector"), "oracle optimum vector", length=8)
    optimum_score = float(optimum["score"])
    candidates = source_report.get("validation_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("source report validation_candidates must be a sequence")
    for raw in candidates:
        candidate = _mapping(raw, "validation candidate")
        rank = int(candidate["candidate_rank"])
        if rank <= 1:
            continue
        vector = _float_vector(candidate.get("vector"), "candidate vector", length=8)
        score_gap = optimum_score - float(candidate["oracle_score"])
        non_target_distance = float(
            np.linalg.norm(np.asarray(vector[2:], dtype=float) - optimum_vector[2:])
        )
        if score_gap <= maximum_score_gap and non_target_distance >= minimum_non_target_distance:
            return {
                "candidate_rank": rank,
                "vector": vector,
                "score_gap": score_gap,
                "non_target_distance": non_target_distance,
            }
    raise ValueError("no validation candidate satisfies the frozen reference-context gates")


def rounded_reference_context(vector: Sequence[float]) -> dict[str, Any]:
    values = _float_vector(vector, "reference vector", length=8)
    context = {
        "reagent_amount_mol": _round_increment(_scale(values[2], 0.003, 0.030), 0.001),
        "stirring_speed_rpm": _round_increment(_scale(values[3], 100.0, 1200.0), 50.0),
        "catalyst": min(int(values[4] * 4), 3),
        "catalyst_amount_mol": _round_increment(
            _scale(values[5], 0.00008, 0.00055), 0.000025
        ),
        "solvent": min(int(values[6] * 4), 3),
        "solvent_volume_L": _round_increment(_scale(values[7], 0.005, 0.050), 0.0025),
    }
    context["reagent_amount_mol"] = _clip(context["reagent_amount_mol"], 0.003, 0.030)
    context["stirring_speed_rpm"] = _clip(
        context["stirring_speed_rpm"], 100.0, 1200.0
    )
    context["catalyst_amount_mol"] = _clip(
        context["catalyst_amount_mol"], 0.00008, 0.00055
    )
    context["solvent_volume_L"] = _clip(context["solvent_volume_L"], 0.005, 0.050)
    return context


def surface_design(reference_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = dict(reference_context)
    normalized_context = (
        _unscale(float(context["reagent_amount_mol"]), 0.003, 0.030),
        _unscale(float(context["stirring_speed_rpm"]), 100.0, 1200.0),
        (int(context["catalyst"]) + 0.5) / 4.0,
        _unscale(float(context["catalyst_amount_mol"]), 0.00008, 0.00055),
        (int(context["solvent"]) + 0.5) / 4.0,
        _unscale(float(context["solvent_volume_L"]), 0.005, 0.050),
    )
    rows: list[dict[str, Any]] = []
    for grid_i, temperature in enumerate(TEMPERATURES_K):
        for grid_j, duration in enumerate(DURATIONS_S):
            rows.append(
                {
                    "query_id": f"t{temperature:.0f}-d{duration:.0f}",
                    "grid_i": grid_i,
                    "grid_j": grid_j,
                    "split": "fit" if grid_i % 2 == 0 and grid_j % 2 == 0 else "held_out",
                    "temperature_K": temperature,
                    "duration_s": duration,
                    "vector": [
                        _unscale(temperature, 250.0, 470.0),
                        _unscale(duration, 1.0, 14_400.0),
                        *normalized_context,
                    ],
                }
            )
    return rows


def analyze_matched_prior_world(
    surface_rows: Sequence[Mapping[str, Any]],
    *,
    validation_sigma: float,
    reference_context: Mapping[str, Any],
    world_token: str,
) -> dict[str, Any]:
    if len(surface_rows) != len(TEMPERATURES_K) * len(DURATIONS_S):
        raise ValueError("matched-prior surface must contain exactly 121 rows")
    platform_failures = [row for row in surface_rows if row.get("status") == "failed"]
    classified = [
        row for row in surface_rows if row.get("status") in {"completed", "physical_failure"}
    ]
    safe_fit = [row for row in surface_rows if _safe_completed(row) and row["split"] == "fit"]
    safe_held_out = [
        row for row in surface_rows if _safe_completed(row) and row["split"] == "held_out"
    ]
    common_checks = {
        "all_surface_queries_classified": len(classified) == len(surface_rows),
        "zero_platform_failures": not platform_failures,
        "safe_fit_count": len(safe_fit) >= MINIMUM_SAFE_FIT_COUNT,
        "safe_held_out_count": len(safe_held_out) >= MINIMUM_SAFE_HELD_OUT_COUNT,
    }
    failures: list[str] = []
    if not all(common_checks.values()):
        failures.extend(key for key, passed in common_checks.items() if not passed)
        return {
            "passed": False,
            "checks": common_checks,
            "failures": failures,
            "surface_count": len(surface_rows),
            "classified_count": len(classified),
            "platform_failure_count": len(platform_failures),
            "physical_failure_count": sum(
                row.get("status") == "physical_failure" for row in surface_rows
            ),
            "safe_fit_count": len(safe_fit),
            "safe_held_out_count": len(safe_held_out),
        }

    score_model = fit_quadratic_model(safe_fit, "score")
    risk_model = fit_quadratic_model(safe_fit, "safety_risk")
    aligned_rows = _prediction_rows(safe_held_out, score_model, risk_model)
    score_range = _nonzero_range(float(row["score"]) for row in safe_held_out)
    risk_range = _nonzero_range(float(row["safety_risk"]) for row in safe_held_out)
    aligned_score_mae = _mean(
        abs(float(row["aligned_score"]) - float(row["score"])) / score_range
        for row in aligned_rows
    )
    aligned_risk_mae = _mean(
        abs(float(row["aligned_risk"]) - float(row["safety_risk"])) / risk_range
        for row in aligned_rows
    )
    common_checks.update(
        {
            "aligned_score_normalized_mae":
                aligned_score_mae <= MAXIMUM_ALIGNED_NORMALIZED_MAE,
            "aligned_risk_normalized_mae":
                aligned_risk_mae <= MAXIMUM_ALIGNED_NORMALIZED_MAE,
        }
    )

    disagreement_gate = max(0.03, 6.0 * float(validation_sigma))
    candidates = []
    for axis in ("temperature", "duration"):
        candidate = _evaluate_reflection_candidate(
            aligned_rows,
            score_model=score_model,
            risk_model=risk_model,
            axis=axis,
            score_range=score_range,
            risk_range=risk_range,
            disagreement_gate=disagreement_gate,
        )
        candidates.append(candidate)
    selected = next((candidate for candidate in candidates if candidate["passed"]), None)
    common_checks["qualified_reflection_exists"] = selected is not None
    if selected is None:
        failures.extend(key for key, passed in common_checks.items() if not passed)
        failures.append("no_frozen_reflection_candidate_passed")
        return {
            "passed": False,
            "checks": common_checks,
            "failures": sorted(set(failures)),
            "surface_count": len(surface_rows),
            "classified_count": len(classified),
            "platform_failure_count": len(platform_failures),
            "physical_failure_count": sum(
                row.get("status") == "physical_failure" for row in surface_rows
            ),
            "safe_fit_count": len(safe_fit),
            "safe_held_out_count": len(safe_held_out),
            "aligned_score_normalized_mae": aligned_score_mae,
            "aligned_risk_normalized_mae": aligned_risk_mae,
            "disagreement_gate": disagreement_gate,
            "reflection_candidates": candidates,
        }

    query_rows = select_balanced_held_out_queries(aligned_rows, count=HELD_OUT_QUERY_COUNT)
    public_priors = build_public_priors(
        selected,
        reference_context=reference_context,
    )
    leakage = audit_public_priors(public_priors)
    matched = audit_supplied_prior_matching(
        public_priors["supplied_a"], public_priors["supplied_b"]
    )
    common_checks.update(
        {
            "held_out_query_count": len(query_rows) == HELD_OUT_QUERY_COUNT,
            "public_prior_leakage_free": leakage["passed"],
            "supplied_prior_matching": matched["passed"],
        }
    )
    blind = _blind_assignment(selected, world_token=world_token)
    passed = all(common_checks.values()) and bool(selected["passed"])
    failures.extend(key for key, value in common_checks.items() if not value)
    failures.extend(leakage["failures"])
    failures.extend(matched["failures"])
    return {
        "passed": passed,
        "checks": common_checks,
        "failures": sorted(set(failures)),
        "surface_count": len(surface_rows),
        "classified_count": len(classified),
        "platform_failure_count": len(platform_failures),
        "physical_failure_count": sum(
            row.get("status") == "physical_failure" for row in surface_rows
        ),
        "safe_fit_count": len(safe_fit),
        "safe_held_out_count": len(safe_held_out),
        "aligned_score_normalized_mae": aligned_score_mae,
        "aligned_risk_normalized_mae": aligned_risk_mae,
        "disagreement_gate": disagreement_gate,
        "selected_reflection": selected,
        "reflection_candidates": candidates,
        "blind_identification": blind,
        "public_priors": public_priors,
        "prior_matching": matched,
        "leakage_audit": leakage,
        "held_out_queries": [
            {
                "query_id": str(row["query_id"]),
                "grid_i": int(row["grid_i"]),
                "grid_j": int(row["grid_j"]),
                "temperature_K": float(row["temperature_K"]),
                "duration_s": float(row["duration_s"]),
            }
            for row in query_rows
        ],
        "quadratic_models": {
            "feature_order": [
                "intercept",
                "temperature",
                "duration",
                "temperature2",
                "interaction",
                "duration2",
            ],
            "temperature_center_K": BASELINE_TEMPERATURE_K,
            "temperature_scale_K": 50.0,
            "duration_center_s": BASELINE_DURATION_S,
            "duration_scale_s": 3000.0,
            "score": score_model,
            "safety_risk": risk_model,
        },
    }


def fit_quadratic_model(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    if len(rows) < 6:
        raise ValueError("quadratic fit requires at least six rows")
    design = np.asarray(
        [
            _quadratic_features(
                float(row["temperature_K"]), float(row["duration_s"])
            )
            for row in rows
        ],
        dtype=float,
    )
    target = np.asarray([float(row[metric]) for row in rows], dtype=float)
    penalty = np.eye(design.shape[1], dtype=float) * 1.0e-6
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {
        "metric": metric,
        "coefficients": coefficients.astype(float).tolist(),
        "training_minimum": float(np.min(target)),
        "training_maximum": float(np.max(target)),
    }


def predict_quadratic(model: Mapping[str, Any], temperature_K: float, duration_s: float) -> float:
    coefficients = np.asarray(model["coefficients"], dtype=float)
    value = float(coefficients @ _quadratic_features(temperature_K, duration_s))
    return _clip(value, float(model["training_minimum"]), float(model["training_maximum"]))


def select_balanced_held_out_queries(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    candidates = [dict(row) for row in rows]
    if len(candidates) < count:
        raise ValueError("not enough held-out queries for balanced selection")
    selected: list[dict[str, Any]] = []
    center = (5.0, 5.0)
    first = min(
        candidates,
        key=lambda row: (
            _grid_distance((float(row["grid_i"]), float(row["grid_j"])), center),
            int(row["grid_i"]),
            int(row["grid_j"]),
        ),
    )
    selected.append(first)
    remaining = [row for row in candidates if row["query_id"] != first["query_id"]]
    while len(selected) < count:
        next_row = max(
            remaining,
            key=lambda row: (
                min(
                    _grid_distance(
                        (float(row["grid_i"]), float(row["grid_j"])),
                        (float(item["grid_i"]), float(item["grid_j"])),
                    )
                    for item in selected
                ),
                -int(row["grid_i"]),
                -int(row["grid_j"]),
            ),
        )
        selected.append(next_row)
        remaining = [row for row in remaining if row["query_id"] != next_row["query_id"]]
    return selected


def build_public_priors(
    selected_reflection: Mapping[str, Any],
    *,
    reference_context: Mapping[str, Any],
) -> dict[str, Any]:
    axis = str(selected_reflection["axis"])
    aligned_side = str(selected_reflection["aligned_preferred_side"])
    opposite = {
        "higher_temperature": "lower_temperature",
        "lower_temperature": "higher_temperature",
        "longer_duration": "shorter_duration",
        "shorter_duration": "longer_duration",
    }[aligned_side]
    common = {
        "schema_version": "chemworld-work-ii-initial-world-model-0.2",
        "locus": "parametric",
        "context_contract": {
            "reference_context": dict(reference_context),
            "target_controls": ["reaction_temperature_K", "reaction_duration_s"],
            "approximate_reference_region": {
                "reaction_temperature_K": BASELINE_TEMPERATURE_K,
                "reaction_duration_s": BASELINE_DURATION_S,
                "temperature_tolerance_K": BASELINE_TEMPERATURE_TOLERANCE_K,
                "duration_tolerance_s": BASELINE_DURATION_TOLERANCE_S,
            },
        },
    }

    def supplied(side: str) -> dict[str, Any]:
        return {
            **common,
            "availability": "supplied_incomplete_model",
            "model": {
                "claim": {
                    "directional_axis": (
                        "reaction_temperature_K" if axis == "temperature" else "reaction_duration_s"
                    ),
                    "expected_relation": _directional_claim(side),
                    "safety_relation": (
                        "Treat an observed temperature-bound rollback as evidence against the "
                        "attempted setting, not as missing platform output."
                    ),
                },
                "confidence": 0.70,
                "scope_limit": (
                    "This is an incomplete local process model. Experimental evidence is "
                    "authoritative outside the stated context."
                ),
            },
            "interpretation": (
                "The supplied model may be reliable or shifted. Experimental evidence is "
                "authoritative."
            ),
        }

    return {
        "opaque": {
            **common,
            "availability": "opaque_for_target_locus",
            "model": None,
            "interpretation": (
                "No task-specific temperature-duration law is supplied. Experimental evidence "
                "is authoritative."
            ),
        },
        "supplied_a": supplied(aligned_side),
        "supplied_b": supplied(opposite),
    }


def audit_supplied_prior_matching(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    left_shape = _json_shape(left)
    right_shape = _json_shape(right)
    left_words = _word_count(left)
    right_words = _word_count(right)
    checks = {
        "schema_shape_equal": left_shape == right_shape,
        "word_count_equal": left_words == right_words,
        "confidence_equal": _nested(left, "model", "confidence")
        == _nested(right, "model", "confidence"),
        "context_equal": _nested(left, "context_contract") == _nested(right, "context_contract"),
        "only_directional_claim_differs": _difference_paths(left, right)
        == ["model.claim.expected_relation"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "left_word_count": left_words,
        "right_word_count": right_words,
        "difference_paths": _difference_paths(left, right),
        "failures": [key for key, value in checks.items() if not value],
    }


def audit_public_priors(priors: Mapping[str, Any]) -> dict[str, Any]:
    rendered = json.dumps(priors, ensure_ascii=False, sort_keys=True).lower()
    forbidden_patterns = (
        r"\baligned\b",
        r"\bmisspecified\b",
        r"\bmisindexed\b",
        r"\bprior[_ -]?arm\b",
        r"\boracle\b",
        r"\bworld[_ -]?seed\b",
        r"\bscreening[_ -]?seed\b",
        r"\bhidden mechanism\b",
        r"\bruns[/\\]",
    )
    hits = [pattern for pattern in forbidden_patterns if re.search(pattern, rendered)]
    return {
        "passed": not hits,
        "forbidden_pattern_hits": hits,
        "failures": [f"participant_visible_leak:{pattern}" for pattern in hits],
    }


def held_out_query_contract(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    reference_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": str(row["query_id"]),
            "feature_values": {
                **dict(reference_context),
                "reaction_temperature_K": float(row["temperature_K"]),
                "reaction_duration_s": float(row["duration_s"]),
            },
            "metric_ids": ["yield", "selectivity", "safety_risk", "score"],
        }
        for row in selected_rows
    ]


def _evaluate_reflection_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_model: Mapping[str, Any],
    risk_model: Mapping[str, Any],
    axis: str,
    score_range: float,
    risk_range: float,
    disagreement_gate: float,
) -> dict[str, Any]:
    output = []
    for row in rows:
        temperature = float(row["temperature_K"])
        duration = float(row["duration_s"])
        reflected_temperature = (
            2.0 * BASELINE_TEMPERATURE_K - temperature if axis == "temperature" else temperature
        )
        reflected_duration = (
            2.0 * BASELINE_DURATION_S - duration if axis == "duration" else duration
        )
        misspecified_score = predict_quadratic(
            score_model, reflected_temperature, reflected_duration
        )
        misspecified_risk = predict_quadratic(risk_model, reflected_temperature, reflected_duration)
        aligned_joint = 0.5 * (
            abs(float(row["aligned_score"]) - float(row["score"])) / score_range
            + abs(float(row["aligned_risk"]) - float(row["safety_risk"])) / risk_range
        )
        misspecified_joint = 0.5 * (
            abs(misspecified_score - float(row["score"])) / score_range
            + abs(misspecified_risk - float(row["safety_risk"])) / risk_range
        )
        output.append(
            {
                **dict(row),
                "misspecified_score": misspecified_score,
                "misspecified_risk": misspecified_risk,
                "aligned_joint_error": aligned_joint,
                "misspecified_joint_error": misspecified_joint,
                "evidence_margin": misspecified_joint - aligned_joint,
            }
        )
    baseline = [
        row
        for row in output
        if abs(float(row["temperature_K"]) - BASELINE_TEMPERATURE_K)
        <= BASELINE_TEMPERATURE_TOLERANCE_K
        and abs(float(row["duration_s"]) - BASELINE_DURATION_S)
        <= BASELINE_DURATION_TOLERANCE_S
    ]
    baseline_gap = max(
        (abs(float(row["aligned_score"]) - float(row["misspecified_score"])) for row in baseline),
        default=math.inf,
    )
    disagreements = [
        row
        for row in output
        if abs(float(row["aligned_score"]) - float(row["misspecified_score"]))
        >= disagreement_gate
        or (
            float(row["aligned_risk"]) < 0.35
            <= float(row["misspecified_risk"])
        )
        or (
            float(row["misspecified_risk"]) < 0.35
            <= float(row["aligned_risk"])
        )
    ]
    aligned_joint_error = _mean(float(row["aligned_joint_error"]) for row in output)
    misspecified_joint_error = _mean(
        float(row["misspecified_joint_error"]) for row in output
    )
    center_index = 5
    low_side = [
        row
        for row in output
        if int(row["grid_i" if axis == "temperature" else "grid_j"]) < center_index
        and float(row["evidence_margin"]) >= MINIMUM_BLIND_ERROR_MARGIN
    ]
    high_side = [
        row
        for row in output
        if int(row["grid_i" if axis == "temperature" else "grid_j"]) > center_index
        and float(row["evidence_margin"]) >= MINIMUM_BLIND_ERROR_MARGIN
    ]
    low_representative = max(low_side, key=lambda row: float(row["evidence_margin"]), default=None)
    high_representative = max(
        high_side, key=lambda row: float(row["evidence_margin"]), default=None
    )
    representative_distance = (
        abs(int(low_representative["grid_i"]) - int(high_representative["grid_i"]))
        + abs(int(low_representative["grid_j"]) - int(high_representative["grid_j"]))
        if low_representative is not None and high_representative is not None
        else 0
    )
    low_mean = _mean(
        float(row["aligned_score"])
        for row in output
        if int(row["grid_i" if axis == "temperature" else "grid_j"]) < center_index
    )
    high_mean = _mean(
        float(row["aligned_score"])
        for row in output
        if int(row["grid_i" if axis == "temperature" else "grid_j"]) > center_index
    )
    preferred_side = (
        "higher_temperature"
        if axis == "temperature" and high_mean >= low_mean
        else "lower_temperature"
        if axis == "temperature"
        else "longer_duration"
        if high_mean >= low_mean
        else "shorter_duration"
    )
    checks = {
        "baseline_utility_matched": baseline_gap <= MAXIMUM_BASELINE_UTILITY_GAP,
        "held_out_disagreement": len(disagreements) / len(output)
        >= MINIMUM_DISAGREEMENT_FRACTION,
        "blind_identification_margin": misspecified_joint_error - aligned_joint_error
        >= MINIMUM_BLIND_ERROR_MARGIN,
        "low_side_falsification_region": len(low_side) >= MINIMUM_REGION_SUPPORT,
        "high_side_falsification_region": len(high_side) >= MINIMUM_REGION_SUPPORT,
        "representatives_separated": representative_distance
        >= MINIMUM_REPRESENTATIVE_GRID_DISTANCE,
    }
    return {
        "axis": axis,
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_maximum_score_gap": baseline_gap,
        "disagreement_count": len(disagreements),
        "held_out_count": len(output),
        "disagreement_fraction": len(disagreements) / len(output),
        "aligned_joint_normalized_error": aligned_joint_error,
        "misspecified_joint_normalized_error": misspecified_joint_error,
        "blind_error_margin": misspecified_joint_error - aligned_joint_error,
        "low_side_support": len(low_side),
        "high_side_support": len(high_side),
        "representative_grid_distance": representative_distance,
        "falsification_representatives": [
            _compact_representative(low_representative),
            _compact_representative(high_representative),
        ],
        "aligned_preferred_side": preferred_side,
    }


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    score_model: Mapping[str, Any],
    risk_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "aligned_score": predict_quadratic(
                score_model, float(row["temperature_K"]), float(row["duration_s"])
            ),
            "aligned_risk": predict_quadratic(
                risk_model, float(row["temperature_K"]), float(row["duration_s"])
            ),
        }
        for row in rows
    ]


def _blind_assignment(selected: Mapping[str, Any], *, world_token: str) -> dict[str, Any]:
    parity = int(hashlib.sha256(world_token.encode()).hexdigest()[:2], 16) % 2
    aligned_label, alternative_label = (
        ("law-X", "law-Y") if parity == 0 else ("law-Y", "law-X")
    )
    lower_error_label = (
        aligned_label
        if float(selected["aligned_joint_normalized_error"])
        < float(selected["misspecified_joint_normalized_error"])
        else alternative_label
    )
    return {
        "labels": sorted((aligned_label, alternative_label)),
        "lower_error_label": lower_error_label,
        "identified_aligned_law": lower_error_label == aligned_label,
        "error_margin": float(selected["blind_error_margin"]),
    }


def _directional_claim(side: str) -> str:
    phrases = {
        "higher_temperature": (
            "Relative to the stated reference region, the higher-temperature side should retain "
            "safe balanced performance more reliably than the lower-temperature side."
        ),
        "lower_temperature": (
            "Relative to the stated reference region, the lower-temperature side should retain "
            "safe balanced performance more reliably than the higher-temperature side."
        ),
        "longer_duration": (
            "Relative to the stated reference region, the longer-duration side should retain safe "
            "balanced performance more reliably than the shorter-duration side."
        ),
        "shorter_duration": (
            "Relative to the stated reference region, the shorter-duration side should retain safe "
            "balanced performance more reliably than the longer-duration side."
        ),
    }
    return phrases[side]


def _quadratic_features(temperature_K: float, duration_s: float) -> np.ndarray:
    x_value = (float(temperature_K) - BASELINE_TEMPERATURE_K) / 50.0
    y_value = (float(duration_s) - BASELINE_DURATION_S) / 3000.0
    return np.asarray(
        [1.0, x_value, y_value, x_value * x_value, x_value * y_value, y_value * y_value],
        dtype=float,
    )


def _compact_representative(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "query_id": str(row["query_id"]),
        "grid_i": int(row["grid_i"]),
        "grid_j": int(row["grid_j"]),
        "temperature_K": float(row["temperature_K"]),
        "duration_s": float(row["duration_s"]),
        "evidence_margin": float(row["evidence_margin"]),
    }


def _safe_completed(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "completed" and bool(row.get("safe"))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _float_vector(value: Any, name: str, *, length: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    output = [float(item) for item in value]
    if not all(math.isfinite(item) for item in output):
        raise ValueError(f"{name} must be finite")
    return output


def _scale(value: float, low: float, high: float) -> float:
    return low + float(value) * (high - low)


def _unscale(value: float, low: float, high: float) -> float:
    return _clip((float(value) - low) / (high - low), 0.0, 1.0)


def _round_increment(value: float, increment: float) -> float:
    return float(round(float(value) / increment) * increment)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), low), high))


def _mean(values: Any) -> float:
    output = [float(value) for value in values]
    return float(sum(output) / len(output)) if output else math.nan


def _nonzero_range(values: Any) -> float:
    output = [float(value) for value in values]
    return max(max(output) - min(output), 1.0e-9)


def _grid_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _word_count(value: Any) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", json.dumps(value, sort_keys=True)))


def _json_shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_shape(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_json_shape(child) for child in value]
    return type(value).__name__


def _difference_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_difference_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            paths.extend(_difference_paths(left_item, right_item, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


__all__ = [
    "BASELINE_DURATION_S",
    "BASELINE_TEMPERATURE_K",
    "DURATIONS_S",
    "MATCHED_PRIOR_VERSION",
    "TEMPERATURES_K",
    "analyze_matched_prior_world",
    "audit_public_priors",
    "audit_supplied_prior_matching",
    "build_public_priors",
    "fit_quadratic_model",
    "held_out_query_contract",
    "predict_quadratic",
    "rounded_reference_context",
    "select_balanced_held_out_queries",
    "select_reference_candidate",
    "surface_design",
]
