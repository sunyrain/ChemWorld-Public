"""Construction and gates for the electrochemical Work II matched-prior block."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chemworld.agents.task_recipes import (
    electrochemical_recipe_parameters_from_unit_vector,
    electrochemical_recipe_unit_vector_from_parameters,
)
from chemworld.eval.work_ii_matched_prior_qualification import (
    audit_public_priors,
    audit_supplied_prior_matching,
    select_balanced_held_out_queries,
)

MATCHED_PRIOR_VERSION = "chemworld-work-ii-electrochemical-matched-prior-qualification-0.3"
GRID_COORDINATES = tuple(round(0.25 + 0.05 * index, 12) for index in range(11))
GRID_SIZE = len(GRID_COORDINATES)
HELD_OUT_QUERY_COUNT = 16
MINIMUM_SAFE_FIT_COUNT = 24
MINIMUM_SAFE_HELD_OUT_COUNT = 40
MAXIMUM_ALIGNED_NORMALIZED_MAE = 0.20
MAXIMUM_BASELINE_UTILITY_GAP = 0.05
MINIMUM_DISAGREEMENT_FRACTION = 0.25
MINIMUM_BLIND_ERROR_MARGIN = 0.05
MINIMUM_REGION_SUPPORT = 3
MINIMUM_REPRESENTATIVE_GRID_DISTANCE = 4
REFLECTION_STRENGTHS = (1.0,)


def select_reference_candidate(
    source_report: Mapping[str, Any],
    *,
    maximum_score_gap: float = 0.05,
    minimum_non_target_distance: float = 0.05,
) -> dict[str, Any]:
    analysis = _mapping(source_report.get("analysis"), "analysis")
    optimum = _mapping(analysis.get("oracle_optimum"), "analysis.oracle_optimum")
    optimum_vector = _float_vector(optimum.get("vector"), "oracle optimum vector", length=9)
    optimum_score = float(optimum["score"])
    candidates = source_report.get("validation_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("source report validation_candidates must be a sequence")
    for raw in candidates:
        candidate = _mapping(raw, "validation candidate")
        rank = int(candidate["candidate_rank"])
        if rank <= 1:
            continue
        vector = _float_vector(candidate.get("vector"), "candidate vector", length=9)
        score_gap = optimum_score - float(candidate["oracle_score"])
        non_target = [0, 1, 2, 3, 4, 5, 8]
        distance = float(
            np.linalg.norm(np.asarray(vector)[non_target] - np.asarray(optimum_vector)[non_target])
        )
        if score_gap <= maximum_score_gap and distance >= minimum_non_target_distance:
            return {
                "candidate_rank": rank,
                "vector": vector,
                "score_gap": score_gap,
                "non_target_distance": distance,
            }
    raise ValueError("no electrochemical reference context satisfies the frozen gates")


def rounded_reference_context(vector: Sequence[float]) -> dict[str, Any]:
    values = _float_vector(vector, "reference vector", length=9)
    params = electrochemical_recipe_parameters_from_unit_vector(np.asarray(values, dtype=float))
    context: dict[str, Any] = {
        "electrolyte_profile": int(params["electrolyte_profile"]),
        "solvent": int(params["solvent"]),
        "reagent_amount_mol": _clip(
            _round_increment(float(params["reagent_amount_mol"]), 0.001), 0.003, 0.030
        ),
        "probe_potential_V": _clip(
            _round_increment(float(params["probe_potential_V"]), 0.01), 0.65, 1.25
        ),
        "probe_current_mA": _clip(
            _round_increment(float(params["probe_current_mA"]), 1.0), 15.0, 90.0
        ),
        "probe_duration_s": _clip(
            _round_increment(float(params["probe_duration_s"]), 30.0), 180.0, 900.0
        ),
        "controlled_duration_s": _clip(
            _round_increment(float(params["controlled_duration_s"]), 60.0), 300.0, 3600.0
        ),
    }
    return context


def _context_vector(context: Mapping[str, Any]) -> np.ndarray:
    payload = {
        **dict(context),
        "controlled_potential_V": float(context["probe_potential_V"]) + 0.02,
        "controlled_current_mA": float(context["probe_current_mA"]) + 1.0,
    }
    return electrochemical_recipe_unit_vector_from_parameters(payload)


def surface_design(reference_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = _context_vector(reference_context)
    rows: list[dict[str, Any]] = []
    for grid_i, potential_coordinate in enumerate(GRID_COORDINATES):
        for grid_j, current_coordinate in enumerate(GRID_COORDINATES):
            vector = np.array(base, copy=True)
            vector[6] = potential_coordinate
            vector[7] = current_coordinate
            params = electrochemical_recipe_parameters_from_unit_vector(vector)
            rows.append(
                {
                    "query_id": f"p{grid_i:02d}-i{grid_j:02d}",
                    "grid_i": grid_i,
                    "grid_j": grid_j,
                    "split": "fit" if grid_i % 2 == 0 and grid_j % 2 == 0 else "held_out",
                    "potential_coordinate": potential_coordinate,
                    "current_coordinate": current_coordinate,
                    "controlled_potential_V": float(params["controlled_potential_V"]),
                    "controlled_current_mA": float(params["controlled_current_mA"]),
                    "vector": vector.tolist(),
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
    expected_count = GRID_SIZE * GRID_SIZE
    if len(surface_rows) != expected_count:
        raise ValueError(
            f"electrochemical matched-prior surface must contain exactly {expected_count} rows"
        )
    platform_failures = [row for row in surface_rows if row.get("status") == "failed"]
    classified = [
        row for row in surface_rows if row.get("status") in {"completed", "physical_failure"}
    ]
    safe_fit = [row for row in surface_rows if _safe_completed(row) and row["split"] == "fit"]
    safe_held_out = [
        row for row in surface_rows if _safe_completed(row) and row["split"] == "held_out"
    ]
    checks: dict[str, bool] = {
        "all_surface_queries_classified": len(classified) == expected_count,
        "zero_platform_failures": not platform_failures,
        "safe_fit_count": len(safe_fit) >= MINIMUM_SAFE_FIT_COUNT,
        "safe_held_out_count": len(safe_held_out) >= MINIMUM_SAFE_HELD_OUT_COUNT,
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        return _early_result(surface_rows, checks, failures, safe_fit, safe_held_out)

    score_model = fit_quadratic_model(safe_fit, "score")
    score_range = _nonzero_range(float(row["score"]) for row in safe_held_out)
    aligned_rows = _prediction_rows(safe_held_out, score_model)
    aligned_mae = _mean(
        abs(float(row["aligned_score"]) - float(row["score"])) / score_range for row in aligned_rows
    )
    checks["aligned_score_normalized_mae"] = aligned_mae <= MAXIMUM_ALIGNED_NORMALIZED_MAE
    disagreement_gate = max(0.03, 6.0 * float(validation_sigma))
    candidates = [
        _evaluate_reflection_candidate(
            aligned_rows,
            score_model=score_model,
            axis=axis,
            reflection_strength=strength,
            score_range=score_range,
            disagreement_gate=disagreement_gate,
        )
        for axis in ("potential", "current")
        for strength in REFLECTION_STRENGTHS
    ]
    selected = next((candidate for candidate in candidates if candidate["passed"]), None)
    checks["qualified_reflection_exists"] = selected is not None
    if selected is None:
        failures.extend(key for key, passed in checks.items() if not passed)
        return _early_result(
            surface_rows,
            checks,
            [*sorted(set(failures)), "no_frozen_reflection_candidate_passed"],
            safe_fit,
            safe_held_out,
            extra={
                "aligned_score_normalized_mae": aligned_mae,
                "disagreement_gate": disagreement_gate,
                "reflection_candidates": candidates,
            },
        )

    query_rows = select_balanced_held_out_queries(aligned_rows, count=HELD_OUT_QUERY_COUNT)
    public_priors = build_public_priors(selected, reference_context=reference_context)
    leakage = audit_public_priors(public_priors)
    matching = audit_supplied_prior_matching(
        public_priors["supplied_a"], public_priors["supplied_b"]
    )
    checks.update(
        {
            "held_out_query_count": len(query_rows) == HELD_OUT_QUERY_COUNT,
            "public_prior_leakage_free": leakage["passed"],
            "supplied_prior_matching": matching["passed"],
        }
    )
    blind = _blind_assignment(selected, world_token=world_token)
    passed = all(checks.values()) and bool(selected["passed"])
    failures.extend(key for key, value in checks.items() if not value)
    failures.extend(leakage["failures"])
    failures.extend(matching["failures"])
    return {
        "passed": passed,
        "checks": checks,
        "failures": sorted(set(failures)),
        "surface_count": len(surface_rows),
        "classified_count": len(classified),
        "platform_failure_count": len(platform_failures),
        "physical_failure_count": sum(
            row.get("status") == "physical_failure" for row in surface_rows
        ),
        "safe_fit_count": len(safe_fit),
        "safe_held_out_count": len(safe_held_out),
        "aligned_score_normalized_mae": aligned_mae,
        "disagreement_gate": disagreement_gate,
        "selected_reflection": selected,
        "reflection_candidates": candidates,
        "blind_identification": blind,
        "public_priors": public_priors,
        "prior_matching": matching,
        "leakage_audit": leakage,
        "held_out_queries": [
            {
                "query_id": str(row["query_id"]),
                "grid_i": int(row["grid_i"]),
                "grid_j": int(row["grid_j"]),
                "potential_coordinate": float(row["potential_coordinate"]),
                "current_coordinate": float(row["current_coordinate"]),
                "controlled_potential_V": float(row["controlled_potential_V"]),
                "controlled_current_mA": float(row["controlled_current_mA"]),
            }
            for row in query_rows
        ],
        "quadratic_model": {
            "feature_order": [
                "intercept",
                "potential",
                "current",
                "potential2",
                "interaction",
                "current2",
            ],
            "score": score_model,
        },
    }


def build_public_priors(
    selected_reflection: Mapping[str, Any], *, reference_context: Mapping[str, Any]
) -> dict[str, Any]:
    axis = str(selected_reflection["axis"])
    preferred = str(selected_reflection["aligned_preferred_side"])
    opposite = {
        "higher_controlled_potential": "lower_controlled_potential",
        "lower_controlled_potential": "higher_controlled_potential",
        "higher_controlled_current": "lower_controlled_current",
        "lower_controlled_current": "higher_controlled_current",
    }[preferred]
    common = {
        "schema_version": "chemworld-work-ii-initial-world-model-0.2",
        "locus": "parametric",
        "context_contract": {
            "reference_context": dict(reference_context),
            "target_controls": ["controlled_potential_V", "controlled_current_mA"],
            "relative_to": ["probe_potential_V", "probe_current_mA"],
            "coordinate_center": 0.5,
        },
    }

    def supplied(side: str) -> dict[str, Any]:
        return {
            **common,
            "availability": "supplied_incomplete_model",
            "model": {
                "claim": {
                    "directional_axis": "controlled_potential_V"
                    if axis == "potential"
                    else "controlled_current_mA",
                    "expected_relation": _directional_claim(side),
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
                "No task-specific potential/current law is supplied. Experimental evidence "
                "is authoritative."
            ),
        },
        "supplied_a": supplied(preferred),
        "supplied_b": supplied(opposite),
    }


def held_out_query_contract(
    selected_rows: Sequence[Mapping[str, Any]], *, reference_context: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": str(row["query_id"]),
            "feature_values": {
                **dict(reference_context),
                "controlled_potential_V": float(row["controlled_potential_V"]),
                "controlled_current_mA": float(row["controlled_current_mA"]),
            },
            "metric_ids": [
                "selective_product_yield",
                "electrochemical_selectivity",
                "faradaic_efficiency",
                "energy_efficiency",
                "safety_risk",
                "score",
            ],
        }
        for row in selected_rows
    ]


def fit_quadratic_model(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    design = np.asarray(
        [
            _quadratic_features(
                float(row["potential_coordinate"]), float(row["current_coordinate"])
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


def predict_quadratic(
    model: Mapping[str, Any], potential_coordinate: float, current_coordinate: float
) -> float:
    value = float(
        np.asarray(model["coefficients"], dtype=float)
        @ _quadratic_features(potential_coordinate, current_coordinate)
    )
    return _clip(value, float(model["training_minimum"]), float(model["training_maximum"]))


def _evaluate_reflection_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_model: Mapping[str, Any],
    axis: str,
    reflection_strength: float,
    score_range: float,
    disagreement_gate: float,
) -> dict[str, Any]:
    output = []
    for row in rows:
        p = float(row["potential_coordinate"])
        c = float(row["current_coordinate"])
        reflected_p = 1.0 - p if axis == "potential" else p
        reflected_c = 1.0 - c if axis == "current" else c
        reflected = predict_quadratic(score_model, reflected_p, reflected_c)
        in_reference_neighborhood = abs(p - 0.5) <= 0.05 and abs(c - 0.5) <= 0.05
        misspecified = (
            float(row["aligned_score"])
            if in_reference_neighborhood
            else float(row["aligned_score"])
            + reflection_strength * (reflected - float(row["aligned_score"]))
        )
        aligned_error = abs(float(row["aligned_score"]) - float(row["score"])) / score_range
        misspecified_error = abs(misspecified - float(row["score"])) / score_range
        output.append(
            {
                **dict(row),
                "misspecified_score": misspecified,
                "aligned_joint_error": aligned_error,
                "misspecified_joint_error": misspecified_error,
                "evidence_margin": misspecified_error - aligned_error,
            }
        )
    baseline = [
        row
        for row in output
        if abs(float(row["potential_coordinate"]) - 0.5) <= 0.05
        and abs(float(row["current_coordinate"]) - 0.5) <= 0.05
    ]
    baseline_gap = max(
        (abs(float(row["aligned_score"]) - float(row["misspecified_score"])) for row in baseline),
        default=math.inf,
    )
    disagreements = [
        row
        for row in output
        if abs(float(row["aligned_score"]) - float(row["misspecified_score"])) >= disagreement_gate
    ]
    aligned_error = _mean(float(row["aligned_joint_error"]) for row in output)
    misspecified_error = _mean(float(row["misspecified_joint_error"]) for row in output)
    grid_axis = "grid_i" if axis == "potential" else "grid_j"
    low = [
        row
        for row in output
        if int(row[grid_axis]) < 5 and float(row["evidence_margin"]) >= MINIMUM_BLIND_ERROR_MARGIN
    ]
    high = [
        row
        for row in output
        if int(row[grid_axis]) > 5 and float(row["evidence_margin"]) >= MINIMUM_BLIND_ERROR_MARGIN
    ]
    low_rep = max(low, key=lambda row: float(row["evidence_margin"]), default=None)
    high_rep = max(high, key=lambda row: float(row["evidence_margin"]), default=None)
    distance = (
        (
            abs(int(low_rep["grid_i"]) - int(high_rep["grid_i"]))
            + abs(int(low_rep["grid_j"]) - int(high_rep["grid_j"]))
        )
        if low_rep and high_rep
        else 0
    )
    low_mean = _mean(float(row["aligned_score"]) for row in output if int(row[grid_axis]) < 5)
    high_mean = _mean(float(row["aligned_score"]) for row in output if int(row[grid_axis]) > 5)
    stem = "controlled_potential" if axis == "potential" else "controlled_current"
    preferred = f"higher_{stem}" if high_mean >= low_mean else f"lower_{stem}"
    checks = {
        "baseline_utility_matched": baseline_gap <= MAXIMUM_BASELINE_UTILITY_GAP,
        "held_out_disagreement": len(disagreements) / len(output) >= MINIMUM_DISAGREEMENT_FRACTION,
        "blind_identification_margin": misspecified_error - aligned_error
        >= MINIMUM_BLIND_ERROR_MARGIN,
        "low_side_falsification_region": len(low) >= MINIMUM_REGION_SUPPORT,
        "high_side_falsification_region": len(high) >= MINIMUM_REGION_SUPPORT,
        "representatives_separated": distance >= MINIMUM_REPRESENTATIVE_GRID_DISTANCE,
    }
    return {
        "axis": axis,
        "reflection_strength": reflection_strength,
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_maximum_score_gap": baseline_gap,
        "disagreement_count": len(disagreements),
        "held_out_count": len(output),
        "disagreement_fraction": len(disagreements) / len(output),
        "aligned_joint_normalized_error": aligned_error,
        "misspecified_joint_normalized_error": misspecified_error,
        "blind_error_margin": misspecified_error - aligned_error,
        "low_side_support": len(low),
        "high_side_support": len(high),
        "representative_grid_distance": distance,
        "aligned_preferred_side": preferred,
    }


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "aligned_score": predict_quadratic(
                model, float(row["potential_coordinate"]), float(row["current_coordinate"])
            ),
        }
        for row in rows
    ]


def _blind_assignment(selected: Mapping[str, Any], *, world_token: str) -> dict[str, Any]:
    parity = int(hashlib.sha256(world_token.encode()).hexdigest()[:2], 16) % 2
    aligned, alternative = ("law-X", "law-Y") if parity == 0 else ("law-Y", "law-X")
    lower_error = (
        aligned
        if float(selected["aligned_joint_normalized_error"])
        < float(selected["misspecified_joint_normalized_error"])
        else alternative
    )
    return {
        "labels": sorted((aligned, alternative)),
        "lower_error_label": lower_error,
        "identified_aligned_law": lower_error == aligned,
        "error_margin": float(selected["blind_error_margin"]),
    }


def _directional_claim(side: str) -> str:
    phrases = {
        "higher_controlled_potential": (
            "Relative to the probe and stated reference context, the "
            "higher-controlled-potential side should retain balanced performance more "
            "reliably than the lower-controlled-potential side."
        ),
        "lower_controlled_potential": (
            "Relative to the probe and stated reference context, the "
            "lower-controlled-potential side should retain balanced performance more "
            "reliably than the higher-controlled-potential side."
        ),
        "higher_controlled_current": (
            "Relative to the probe and stated reference context, the "
            "higher-controlled-current side should retain balanced performance more reliably "
            "than the lower-controlled-current side."
        ),
        "lower_controlled_current": (
            "Relative to the probe and stated reference context, the "
            "lower-controlled-current side should retain balanced performance more reliably "
            "than the higher-controlled-current side."
        ),
    }
    return phrases[side]


def _quadratic_features(potential: float, current: float) -> np.ndarray:
    x = (float(potential) - 0.5) / 0.25
    y = (float(current) - 0.5) / 0.25
    return np.asarray([1.0, x, y, x * x, x * y, y * y], dtype=float)


def _early_result(
    surface_rows: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
    failures: Sequence[str],
    safe_fit: Sequence[Mapping[str, Any]],
    safe_held_out: Sequence[Mapping[str, Any]],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "passed": False,
        "checks": dict(checks),
        "failures": sorted(set(failures)),
        "surface_count": len(surface_rows),
        "classified_count": sum(
            row.get("status") in {"completed", "physical_failure"} for row in surface_rows
        ),
        "platform_failure_count": sum(row.get("status") == "failed" for row in surface_rows),
        "physical_failure_count": sum(
            row.get("status") == "physical_failure" for row in surface_rows
        ),
        "safe_fit_count": len(safe_fit),
        "safe_held_out_count": len(safe_held_out),
    }
    if extra:
        result.update(dict(extra))
    return result


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


def _round_increment(value: float, increment: float) -> float:
    return float(round(float(value) / increment) * increment)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), low), high))


def _mean(values: Sequence[float] | Any) -> float:
    output = [float(value) for value in values]
    return float(sum(output) / len(output)) if output else math.nan


def _nonzero_range(values: Sequence[float] | Any) -> float:
    output = [float(value) for value in values]
    return max(max(output) - min(output), 1.0e-9)


__all__ = [
    "GRID_COORDINATES",
    "HELD_OUT_QUERY_COUNT",
    "MATCHED_PRIOR_VERSION",
    "REFLECTION_STRENGTHS",
    "analyze_matched_prior_world",
    "build_public_priors",
    "held_out_query_contract",
    "rounded_reference_context",
    "select_reference_candidate",
    "surface_design",
]
