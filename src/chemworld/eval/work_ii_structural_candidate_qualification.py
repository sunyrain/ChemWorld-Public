"""Frozen Work II structural-candidate designs and qualification analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

STRUCTURAL_QUALIFICATION_VERSION = "chemworld-work-ii-structural-candidate-qualification-0.2"
WORLD_SEEDS = (0, 1, 2, 3, 4)
GRID_LEVELS = (0, 1, 2)
VALIDATION_REPLICATES = 3
EFFECT_FLOOR = 0.03
NOISE_MULTIPLIER = 6.0
MINIMUM_DISAGREEMENT_FRACTION = 0.40

ELECTROCHEMICAL_METRICS = (
    "selective_product_yield",
    "faradaic_efficiency",
    "transport_efficiency",
    "ohmic_efficiency",
    "energy_efficiency",
    "safety_risk",
    "score",
)
CRYSTALLIZATION_METRICS = (
    "crystal_yield",
    "crystal_size",
    "crystal_csd_quality",
    "crystal_fines_fraction",
    "score",
)


def candidate_specs() -> dict[str, dict[str, Any]]:
    return {
        "electrochemical_transport": {
            "task_id": "electrochemical-conversion",
            "config": (
                "configs/benchmark/"
                "work_ii_electrochemical_parametric_initial_model_pilot.json"
            ),
            "axis_names": ("controlled_potential_V", "controlled_current_mA"),
            "axis_levels": ((0.75, 1.05, 1.35), (15.0, 91.0, 190.0)),
            "fixed_context": {
                "electrolyte_profile": 0,
                "solvent": 0,
                "reagent_amount_mol": 0.012,
                "probe_potential_V": 0.80,
                "probe_current_mA": 90.0,
                "probe_duration_s": 300.0,
                "controlled_duration_s": 1800.0,
            },
            "metrics": ELECTROCHEMICAL_METRICS,
            "model_metrics": (
                "selective_product_yield",
                "faradaic_efficiency",
                "transport_efficiency",
            ),
        },
        "crystallization_nucleation_growth": {
            "task_id": "reaction-to-crystallization",
            "config": "configs/benchmark/work_ii_crystallization_campaign.json",
            "axis_names": ("seed_mass_g", "crystallization_temperature_K"),
            "axis_levels": ((0.001, 0.008, 0.015), (310.0, 290.0, 270.0)),
            "fixed_context": {
                "catalyst": 0,
                "solvent": 0,
                "reagent_amount_mol": 0.015,
                "reaction_temperature_K": 398.15,
                "reaction_duration_s": 7200.0,
                "stirring_speed_rpm": 675.0,
                "catalyst_amount_mol": 0.000315,
                "crystallization_duration_s": 7200.0,
            },
            "metrics": CRYSTALLIZATION_METRICS,
            "model_metrics": (
                "crystal_yield",
                "crystal_csd_quality",
                "crystal_fines_fraction",
            ),
        },
    }


def registered_queries(candidate_id: str) -> list[dict[str, Any]]:
    spec = candidate_specs()[candidate_id]
    axis_a, axis_b = spec["axis_names"]
    levels_a, levels_b = spec["axis_levels"]
    rows: list[dict[str, Any]] = []
    for axis_a_index, axis_a_value in enumerate(levels_a):
        for axis_b_index, axis_b_value in enumerate(levels_b):
            feature_values = {
                **spec["fixed_context"],
                axis_a: axis_a_value,
                axis_b: axis_b_value,
            }
            rows.append(
                {
                    "query_id": f"a{axis_a_index}-b{axis_b_index}",
                    "phase": "main_grid",
                    "axis_a_index": axis_a_index,
                    "axis_b_index": axis_b_index,
                    "feature_values": feature_values,
                    "metric_ids": list(spec["metrics"]),
                }
            )
    for group_index, (axis_a_index, axis_b_index) in enumerate(
        validation_groups(candidate_id)
    ):
        for replicate in range(1, VALIDATION_REPLICATES + 1):
            feature_values = {
                **spec["fixed_context"],
                axis_a: levels_a[axis_a_index],
                axis_b: levels_b[axis_b_index],
            }
            rows.append(
                {
                    "query_id": f"validation-g{group_index}-r{replicate}",
                    "phase": "noisy_validation",
                    "validation_group": group_index,
                    "replicate": replicate,
                    "axis_a_index": axis_a_index,
                    "axis_b_index": axis_b_index,
                    "feature_values": feature_values,
                    "metric_ids": list(spec["metrics"]),
                }
            )
    return rows


def validation_groups(candidate_id: str) -> tuple[tuple[int, int], ...]:
    if candidate_id == "electrochemical_transport":
        return ((1, 0), (1, 1), (1, 2))
    if candidate_id == "crystallization_nucleation_growth":
        return ((0, 1), (1, 1), (2, 1))
    raise ValueError(f"unknown structural candidate: {candidate_id}")


def analyze_candidate_world(
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = candidate_specs()[candidate_id]
    main = [row for row in rows if row.get("phase") == "main_grid"]
    validation = [row for row in rows if row.get("phase") == "noisy_validation"]
    groups = validation_groups(candidate_id)
    expected = 9 + len(groups) * VALIDATION_REPLICATES
    checks: dict[str, bool] = {
        "fixed_query_count": len(rows) == expected,
        "main_grid_count": len(main) == 9,
        "validation_count": len(validation) == 9,
        "all_outcomes_classified": all(
            row.get("status") in {"completed", "physical_failure", "platform_failure"}
            for row in rows
        ),
        "zero_platform_failures": not any(
            row.get("status") == "platform_failure" for row in rows
        ),
        "all_exact_replay": all(row.get("exact_replay") is True for row in rows),
    }
    completed_main = [row for row in main if row.get("status") == "completed"]
    completed_validation = [
        row for row in validation if row.get("status") == "completed"
    ]
    checks["complete_main_surface"] = len(completed_main) == 9
    checks["complete_validation_surface"] = len(completed_validation) == 9
    if not all(checks.values()):
        return _early_result(candidate_id, rows, checks)

    sigma = _validation_sigma(completed_validation, spec["metrics"], groups=groups)
    if candidate_id == "electrochemical_transport":
        effects = _electrochemical_effects(completed_main, sigma)
        model = _model_qualification(
            completed_main,
            completed_validation,
            sigma=sigma,
            metrics=spec["model_metrics"],
            aligned_features=_electrochemical_aligned_features,
            misspecified_features=_electrochemical_misspecified_features,
            validation_groups=groups,
            target_axis="b",
            candidate_id=candidate_id,
        )
    elif candidate_id == "crystallization_nucleation_growth":
        effects = _crystallization_effects(completed_main, sigma)
        model = _model_qualification(
            completed_main,
            completed_validation,
            sigma=sigma,
            metrics=spec["model_metrics"],
            aligned_features=_crystallization_aligned_features,
            misspecified_features=_crystallization_misspecified_features,
            validation_groups=groups,
            target_axis="a",
            candidate_id=candidate_id,
        )
    else:
        raise ValueError(f"unknown structural candidate: {candidate_id}")
    checks.update(
        {
            "axis_a_effect": bool(effects["axis_a"]["passed"]),
            "axis_b_effect": bool(effects["axis_b"]["passed"]),
            "topology_signature": bool(effects["topology_signature"]["passed"]),
            "baseline_error_matched": bool(model["checks"]["baseline_error_matched"]),
            "held_out_disagreement": bool(model["checks"]["held_out_disagreement"]),
            "low_counterexample_region": bool(model["checks"]["low_counterexample_region"]),
            "high_counterexample_region": bool(model["checks"]["high_counterexample_region"]),
            "blind_identification": bool(model["checks"]["blind_identification"]),
            "prior_schema_matched": bool(model["checks"]["prior_schema_matched"]),
            "prior_word_count_matched": bool(model["checks"]["prior_word_count_matched"]),
        }
    )
    return {
        "candidate_id": candidate_id,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, passed in checks.items() if not passed),
        "denominators": _denominators(rows),
        "validation_sigma": sigma,
        "effects": effects,
        "model_qualification": model,
        "prior_arms": build_prior_arms(candidate_id),
    }


def build_prior_arms(candidate_id: str) -> dict[str, Any]:
    if candidate_id == "electrochemical_transport":
        aligned_claim = (
            "At fixed potential, increasing current beyond the middle regime gives diminishing "
            "product benefit while transport and Faradaic efficiencies decline measurably."
        )
        misspecified_claim = (
            "At fixed potential, increasing current beyond the middle regime gives continuing "
            "product benefit while transport and Faradaic efficiencies remain approximately stable."
        )
        target = "current_transport_limitation"
        intervention = ["controlled_potential_V", "controlled_current_mA"]
    elif candidate_id == "crystallization_nucleation_growth":
        aligned_claim = (
            "At fixed cooling, increasing seed mass changes crystal-size quality and fines, with "
            "the seed effect depending measurably on cooling severity."
        )
        misspecified_claim = (
            "At fixed cooling, increasing seed mass leaves crystal-size quality and fines stable, "
            "with the outcome governed primarily by cooling severity."
        )
        target = "seed_mediated_nucleation_growth"
        intervention = ["seed_mass_g", "crystallization_temperature_K"]
    else:
        raise ValueError(f"unknown structural candidate: {candidate_id}")
    common = {
        "schema_version": "chemworld-work-ii-initial-world-model-0.3",
        "locus": "structural_mechanistic",
        "target": target,
        "intervention_controls": intervention,
        "confidence": 0.70,
        "scope_limit": (
            "This is an incomplete local causal model. Public experimental evidence is "
            "authoritative outside the registered context."
        ),
    }
    return {
        "opaque": {
            **common,
            "availability": "opaque_for_target_locus",
            "claim": None,
        },
        "aligned_nominal": {
            **common,
            "availability": "supplied_incomplete_model",
            "claim": aligned_claim,
        },
        "misindexed_nominal": {
            **common,
            "availability": "supplied_incomplete_model",
            "claim": misspecified_claim,
        },
    }


def _validation_sigma(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    groups: Sequence[tuple[int, int]],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in metrics:
        group_sigmas = []
        for group_index in range(len(groups)):
            values = [
                float(row["metrics"][metric])
                for row in rows
                if int(row["validation_group"]) == group_index
            ]
            if len(values) != VALIDATION_REPLICATES:
                raise ValueError("validation group does not contain three completed replicates")
            group_sigmas.append(float(np.std(values, ddof=1)))
        output[str(metric)] = max(group_sigmas)
    return output


def _electrochemical_effects(
    rows: Sequence[Mapping[str, Any]], sigma: Mapping[str, float]
) -> dict[str, Any]:
    keyed = _grid(rows)
    axis_a = _best_axis_effect(
        keyed,
        axis="a",
        metrics=("selective_product_yield", "faradaic_efficiency", "ohmic_efficiency"),
        sigma=sigma,
    )
    axis_b = _best_axis_effect(
        keyed,
        axis="b",
        metrics=(
            "transport_efficiency",
            "faradaic_efficiency",
            "selective_product_yield",
        ),
        sigma=sigma,
    )
    signatures: list[dict[str, Any]] = []
    for metric in ("transport_efficiency", "faradaic_efficiency"):
        values = [
            max(
                float(keyed[(i, 0)]["metrics"][metric])
                - float(keyed[(i, 2)]["metrics"][metric]),
                0.0,
            )
            for i in GRID_LEVELS
        ]
        signatures.append(_signature(metric, "high_current_efficiency_loss", max(values), sigma))
    yield_curvature = max(
        (
            float(keyed[(i, 1)]["metrics"]["selective_product_yield"])
            - float(keyed[(i, 0)]["metrics"]["selective_product_yield"])
        )
        - (
            float(keyed[(i, 2)]["metrics"]["selective_product_yield"])
            - float(keyed[(i, 1)]["metrics"]["selective_product_yield"])
        )
        for i in GRID_LEVELS
    )
    signatures.append(
        _signature(
            "selective_product_yield",
            "diminishing_high_current_yield_gain",
            max(yield_curvature, 0.0),
            sigma,
        )
    )
    topology = max(signatures, key=lambda item: float(item["margin"]))
    return {
        "axis_a": axis_a,
        "axis_b": axis_b,
        "topology_signature": topology,
        "topology_candidates": signatures,
    }


def _crystallization_effects(
    rows: Sequence[Mapping[str, Any]], sigma: Mapping[str, float]
) -> dict[str, Any]:
    keyed = _grid(rows)
    axis_a = _best_axis_effect(
        keyed,
        axis="a",
        metrics=("crystal_csd_quality", "crystal_fines_fraction", "crystal_yield"),
        sigma=sigma,
    )
    axis_b = _best_axis_effect(
        keyed,
        axis="b",
        metrics=("crystal_yield", "crystal_csd_quality", "crystal_fines_fraction"),
        sigma=sigma,
    )
    signatures: list[dict[str, Any]] = []
    for metric in ("crystal_csd_quality", "crystal_fines_fraction"):
        seed_at_mild = (
            float(keyed[(2, 0)]["metrics"][metric])
            - float(keyed[(0, 0)]["metrics"][metric])
        )
        seed_at_severe = (
            float(keyed[(2, 2)]["metrics"][metric])
            - float(keyed[(0, 2)]["metrics"][metric])
        )
        signatures.append(
            _signature(
                metric,
                "seed_by_cooling_interaction",
                abs(seed_at_severe - seed_at_mild),
                sigma,
            )
        )
        contrast = max(
            abs(
                float(keyed[(2, j)]["metrics"][metric])
                - float(keyed[(0, j)]["metrics"][metric])
            )
            for j in GRID_LEVELS
        )
        signatures.append(
            _signature(metric, "seed_driven_csd_or_fines_contrast", contrast, sigma)
        )
    topology = max(signatures, key=lambda item: float(item["margin"]))
    return {
        "axis_a": axis_a,
        "axis_b": axis_b,
        "topology_signature": topology,
        "topology_candidates": signatures,
    }


def _best_axis_effect(
    keyed: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    axis: str,
    metrics: Sequence[str],
    sigma: Mapping[str, float],
) -> dict[str, Any]:
    candidates = []
    for metric in metrics:
        ranges = []
        for fixed in GRID_LEVELS:
            values = [
                float(keyed[(level, fixed)]["metrics"][metric])
                if axis == "a"
                else float(keyed[(fixed, level)]["metrics"][metric])
                for level in GRID_LEVELS
            ]
            ranges.append(max(values) - min(values))
        candidates.append(_signature(metric, f"axis_{axis}_range", max(ranges), sigma))
    return max(candidates, key=lambda item: float(item["margin"]))


def _signature(
    metric: str,
    signature: str,
    value: float,
    sigma: Mapping[str, float],
) -> dict[str, Any]:
    threshold = max(EFFECT_FLOOR, NOISE_MULTIPLIER * float(sigma.get(metric, 0.0)))
    return {
        "metric": metric,
        "signature": signature,
        "value": float(value),
        "sigma_observed": float(sigma.get(metric, 0.0)),
        "threshold": threshold,
        "margin": float(value) - threshold,
        "passed": float(value) >= threshold,
    }


def _model_qualification(
    main: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    sigma: Mapping[str, float],
    metrics: Sequence[str],
    aligned_features: Any,
    misspecified_features: Any,
    validation_groups: Sequence[tuple[int, int]],
    target_axis: str,
    candidate_id: str,
) -> dict[str, Any]:
    aligned_models = {
        metric: _fit_model(main, metric, aligned_features) for metric in metrics
    }
    misspecified_models = {
        metric: _fit_model(main, metric, misspecified_features) for metric in metrics
    }
    baseline = (1, 1)
    for metric in metrics:
        aligned_baseline = _predict(
            aligned_models[metric], metric, baseline, aligned_features
        )
        misspecified_baseline = _predict(
            misspecified_models[metric], metric, baseline, misspecified_features
        )
        misspecified_models[metric]["baseline_offset"] = (
            aligned_baseline - misspecified_baseline
        )

    group_means = _validation_group_means(
        validation,
        metrics,
        groups=validation_groups,
    )
    comparisons: list[dict[str, Any]] = []
    aligned_errors: list[float] = []
    misspecified_errors: list[float] = []
    for (axis_a, axis_b), observed in group_means.items():
        for metric in metrics:
            aligned_prediction = _predict(
                aligned_models[metric], metric, (axis_a, axis_b), aligned_features
            )
            misspecified_prediction = _predict(
                misspecified_models[metric],
                metric,
                (axis_a, axis_b),
                misspecified_features,
            )
            gate = max(EFFECT_FLOOR, NOISE_MULTIPLIER * float(sigma[metric]))
            aligned_error = abs(aligned_prediction - float(observed[metric]))
            misspecified_error = abs(misspecified_prediction - float(observed[metric]))
            aligned_errors.append(aligned_error)
            misspecified_errors.append(misspecified_error)
            comparisons.append(
                {
                    "axis_a_index": axis_a,
                    "axis_b_index": axis_b,
                    "metric": metric,
                    "observed_validation_mean": float(observed[metric]),
                    "aligned_prediction": aligned_prediction,
                    "misspecified_prediction": misspecified_prediction,
                    "prediction_difference": abs(aligned_prediction - misspecified_prediction),
                    "disagreement_gate": gate,
                    "disagrees": abs(aligned_prediction - misspecified_prediction) >= gate,
                    "aligned_error": aligned_error,
                    "misspecified_error": misspecified_error,
                }
            )
    baseline_rows = [
        row for row in comparisons if row["axis_a_index"] == 1 and row["axis_b_index"] == 1
    ]
    baseline_error_gap = max(
        abs(float(row["aligned_error"]) - float(row["misspecified_error"]))
        for row in baseline_rows
    )
    disagreement = [row for row in comparisons if row["disagrees"]]
    disagreement_fraction = len(disagreement) / len(comparisons)
    target_index = "axis_a_index" if target_axis == "a" else "axis_b_index"
    low_support = sum(row[target_index] == 0 for row in disagreement)
    high_support = sum(row[target_index] == 2 for row in disagreement)
    aligned_mae = float(np.mean(aligned_errors))
    misspecified_mae = float(np.mean(misspecified_errors))
    priors = build_prior_arms(candidate_id)
    aligned_prior = priors["aligned_nominal"]
    misspecified_prior = priors["misindexed_nominal"]
    schema_matched = set(aligned_prior) == set(misspecified_prior)
    word_counts = {
        "aligned": len(str(aligned_prior["claim"]).split()),
        "misspecified": len(str(misspecified_prior["claim"]).split()),
    }
    checks = {
        "baseline_error_matched": baseline_error_gap <= 1.0e-12,
        "held_out_disagreement": disagreement_fraction >= MINIMUM_DISAGREEMENT_FRACTION,
        "low_counterexample_region": low_support > 0,
        "high_counterexample_region": high_support > 0,
        "blind_identification": aligned_mae < misspecified_mae,
        "prior_schema_matched": schema_matched,
        "prior_word_count_matched": abs(word_counts["aligned"] - word_counts["misspecified"])
        <= 2,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_error_gap": baseline_error_gap,
        "comparison_count": len(comparisons),
        "disagreement_count": len(disagreement),
        "disagreement_fraction": disagreement_fraction,
        "low_counterexample_support": low_support,
        "high_counterexample_support": high_support,
        "aligned_validation_mae": aligned_mae,
        "misspecified_validation_mae": misspecified_mae,
        "blind_identified_aligned_model": aligned_mae < misspecified_mae,
        "prior_word_counts": word_counts,
        "comparisons": comparisons,
        "model_coefficients": {
            "aligned": aligned_models,
            "misspecified": misspecified_models,
        },
    }


def _fit_model(
    rows: Sequence[Mapping[str, Any]], metric: str, feature_function: Any
) -> dict[str, Any]:
    design = np.asarray(
        [
            feature_function(
                metric,
                int(row["axis_a_index"]),
                int(row["axis_b_index"]),
            )
            for row in rows
        ],
        dtype=float,
    )
    target = np.asarray([float(row["metrics"][metric]) for row in rows], dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return {"coefficients": coefficients.tolist(), "baseline_offset": 0.0}


def _predict(
    model: Mapping[str, Any],
    metric: str,
    point: tuple[int, int],
    feature_function: Any,
) -> float:
    value = float(
        np.asarray(model["coefficients"], dtype=float)
        @ np.asarray(feature_function(metric, *point), dtype=float)
    )
    return value + float(model.get("baseline_offset", 0.0))


def _electrochemical_aligned_features(
    metric: str, axis_a: int, axis_b: int
) -> list[float]:
    del metric
    p, current = float(axis_a - 1), float(axis_b - 1)
    return [1.0, p, current, p * current, current * current]


def _electrochemical_misspecified_features(
    metric: str, axis_a: int, axis_b: int
) -> list[float]:
    p, current = float(axis_a - 1), float(axis_b - 1)
    if metric in {"faradaic_efficiency", "transport_efficiency"}:
        return [1.0, p]
    return [1.0, p, current, p * current]


def _crystallization_aligned_features(
    metric: str, axis_a: int, axis_b: int
) -> list[float]:
    del metric
    seed, cooling = float(axis_a - 1), float(axis_b - 1)
    return [1.0, cooling, cooling * cooling, seed, seed * cooling]


def _crystallization_misspecified_features(
    metric: str, axis_a: int, axis_b: int
) -> list[float]:
    del metric
    del axis_a
    cooling = float(axis_b - 1)
    return [1.0, cooling, cooling * cooling]


def _validation_group_means(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    groups: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, float]]:
    output = {}
    for group_index, point in enumerate(groups):
        group = [row for row in rows if int(row["validation_group"]) == group_index]
        output[point] = {
            metric: float(np.mean([float(row["metrics"][metric]) for row in group]))
            for metric in metrics
        }
    return output


def _grid(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    output = {
        (int(row["axis_a_index"]), int(row["axis_b_index"])): row for row in rows
    }
    if set(output) != {(i, j) for i in GRID_LEVELS for j in GRID_LEVELS}:
        raise ValueError("main structural grid is incomplete or duplicated")
    return output


def _denominators(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "attempted": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "physical_failures": sum(row.get("status") == "physical_failure" for row in rows),
        "platform_failures": sum(row.get("status") == "platform_failure" for row in rows),
        "unsafe_completed": sum(
            row.get("status") == "completed" and row.get("safe") is False for row in rows
        ),
        "exact_replay": sum(row.get("exact_replay") is True for row in rows),
    }


def _early_result(
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "passed": False,
        "checks": dict(checks),
        "failures": sorted(key for key, passed in checks.items() if not passed),
        "denominators": _denominators(rows),
        "validation_sigma": None,
        "effects": None,
        "model_qualification": None,
        "prior_arms": build_prior_arms(candidate_id),
    }


def finite_metrics(metrics: Mapping[str, Any], metric_ids: Sequence[str]) -> dict[str, float]:
    output = {}
    for metric_id in metric_ids:
        value = metrics.get(metric_id)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"missing numeric structural metric: {metric_id}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite structural metric: {metric_id}")
        output[str(metric_id)] = number
    return output


__all__ = [
    "STRUCTURAL_QUALIFICATION_VERSION",
    "WORLD_SEEDS",
    "analyze_candidate_world",
    "build_prior_arms",
    "candidate_specs",
    "finite_metrics",
    "registered_queries",
    "validation_groups",
]
