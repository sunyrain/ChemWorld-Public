"""Executable sequential challenge probes for Experiment 1 v1.2.

Unlike the v1.0 development attempt, this module derives default, per-action,
and stopping decisions from public execution observations.  It never assigns a
fixed stopping cost or a fixed default outcome.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, variance
from typing import Any

from chemworld.eval import experiment_1_challenge as v1
from chemworld.eval import work_ii_structural_candidate_qualification as structural
from chemworld.eval.experiment_1_ec_qualification_repair import (
    _structural_model_qualification_repair,
)
from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_electrochemical_matched_prior_qualification import (
    predict_quadratic as predict_ec,
)
from chemworld.eval.work_ii_matched_prior_qualification import (
    predict_quadratic as predict_rx,
)

SCHEMA_VERSION = "chemworld-experiment-1-challenge-probes-1.2"
EXPECTED_BLOCKS = v1.EXPECTED_BLOCKS


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    if not values or not all(_finite(value) for value in values):
        raise ValueError("observation group is empty or non-finite")
    mean = fmean(float(value) for value in values)
    se = math.sqrt(variance(values) / len(values)) if len(values) > 1 else math.inf
    return mean, se


def _observation_digest(payload: Any) -> str:
    return canonical_json_sha256(payload)


def _step(
    *,
    index: int,
    action_id: str,
    observation: Any,
    statistic_before: float,
    statistic_after: float,
    reliable: bool,
    information_gain_over_default: float,
    stop_reason: str,
) -> dict[str, Any]:
    return {
        "step": index,
        "unique_condition_cost": index,
        "action_id": action_id,
        "observation_sha256": _observation_digest(observation),
        "statistic_before": statistic_before,
        "statistic_after": statistic_after,
        "reliable_falsification": reliable,
        "information_gain_over_default": information_gain_over_default,
        "stop_reason": stop_reason,
    }


def _finish_trace(
    steps: list[dict[str, Any]], budget: int, minimum_information_gain: float
) -> dict[str, Any]:
    if not steps:
        raise ValueError("sequential probe produced no steps")
    stopping_row = next((row for row in steps if row["reliable_falsification"]), None)
    stopping = int(stopping_row["unique_condition_cost"]) if stopping_row is not None else None
    default_success = bool(steps[0]["reliable_falsification"])
    stopping_gain = (
        float(stopping_row["information_gain_over_default"]) if stopping_row is not None else 0.0
    )
    return {
        "default_action_id": steps[0]["action_id"],
        "default_one_shot_reliably_discriminates": default_success,
        "ordered_policy_steps": steps,
        "minimum_reliable_unique_condition_cost": stopping,
        "information_choice_gain_over_default_at_stop": stopping_gain,
        "information_gain_estimand": (
            "realized evidence-statistic gain of the selected next unique condition "
            "relative to the frozen default condition"
        ),
        "minimum_information_gain": minimum_information_gain,
        "active_information_passed": bool(
            stopping_gain >= minimum_information_gain and stopping is not None
        ),
        "budget_window_passed": bool(stopping is not None and 1 < stopping <= budget),
    }


def _transposed_pair(report: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[int, int]:
    permutation = report.get("descriptor_permutation")
    if not isinstance(permutation, list):
        prior = _mapping(report.get("prior_audit"))
        permutation = _mapping(prior.get("misspecified")).get("descriptor_permutation")
    if not isinstance(permutation, list):
        permutation = policy.get("descriptor_permutation")
    if not isinstance(permutation, list):
        raise ValueError("entity report lacks a descriptor permutation")
    moved = [index for index, value in enumerate(permutation) if index != int(value)]
    if len(moved) != 2 or int(permutation[moved[0]]) != moved[1]:
        raise ValueError("entity challenge requires exactly one transposition")
    return moved[0], moved[1]


def descriptor_permutation_from_bound_contracts(
    root: Path, bindings: Sequence[Mapping[str, Any]]
) -> list[int]:
    permutations: list[list[int]] = []
    for binding in bindings:
        path = root / str(binding.get("path", ""))
        if not path.is_file():
            raise ValueError("bound source contract is missing")
        if file_sha256(path) != binding.get("sha256"):
            raise ValueError("bound source contract digest mismatch")
        document = _load_json(path)
        if not isinstance(document, dict):
            raise ValueError("bound source contract must be an object")
        entity = _mapping(_mapping(document.get("loci")).get("entity"))
        permutation = entity.get("descriptor_permutation")
        if isinstance(permutation, list):
            permutations.append([int(value) for value in permutation])
    if len(permutations) != 1:
        raise ValueError("exactly one bound entity permutation is required")
    return permutations[0]


def _entity_trace(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    first, second = _transposed_pair(report, policy)
    target_field = str(policy["target_field"])
    anchor_field = str(policy["anchor_field"])
    anchor_value = policy["anchor_value"]
    metric_container = str(policy["metric_container"])
    metric = str(policy["metric"])
    grouped: dict[int, list[float]] = defaultdict(list)
    row_digests: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed" or row.get(anchor_field) != anchor_value:
            continue
        target = int(row[target_field])
        value = _mapping(row.get(metric_container)).get(metric)
        if _finite(value):
            grouped[target].append(float(value))
            row_digests[target].append(str(row.get("receipt_sha256", row.get("query_id"))))
    mean_a, se_a = _mean_se(grouped[first])
    mean_b, se_b = _mean_se(grouped[second])
    effect = abs(mean_b - mean_a)
    joint_se = math.sqrt(se_a**2 + se_b**2)
    snr = effect / max(joint_se, 1.0e-12)
    threshold_snr = float(policy["reliable_snr"])
    threshold_effect = float(policy["minimum_effect"])
    steps = [
        _step(
            index=1,
            action_id=f"{anchor_field}={anchor_value};{target_field}={first}",
            observation={"row_digests": row_digests[first]},
            statistic_before=0.0,
            statistic_after=0.0,
            reliable=False,
            information_gain_over_default=0.0,
            stop_reason="paired label-exchange contrast not yet observed",
        ),
        _step(
            index=2,
            action_id=f"{anchor_field}={anchor_value};{target_field}={second}",
            observation={"row_digests": row_digests[second]},
            statistic_before=0.0,
            statistic_after=snr,
            reliable=bool(snr >= threshold_snr and effect >= threshold_effect),
            information_gain_over_default=snr,
            stop_reason=(
                "paired public-response contrast crossed the frozen SNR/effect thresholds"
                if snr >= threshold_snr and effect >= threshold_effect
                else "paired public-response contrast remained below threshold"
            ),
        ),
    ]
    result = _finish_trace(
        steps,
        int(policy["participant_budget"]),
        float(policy["minimum_information_gain"]),
    )
    result["measured_effect"] = effect
    result["measured_snr"] = snr
    return result


def _reflection_trace(
    block: str,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {str(row.get("query_id")): row for row in rows}
    action_ids = [str(value) for value in policy["ordered_action_ids"]]
    minimum_margin = float(policy["minimum_evidence_margin"])
    calibration = _mapping(policy.get("noise_calibration"))
    noise_sigma = calibration.get("sigma")
    repeated_noise_supported = bool(
        _finite(noise_sigma)
        and int(calibration.get("complete_group_count", 0))
        == int(calibration.get("planned_group_count", -1))
        and int(calibration.get("complete_group_count", 0)) >= 2
    )
    evidence_sum = 0.0
    default_information = 0.0
    steps: list[dict[str, Any]] = []
    if block == "EC-P":
        model = _mapping(_mapping(report["legacy_analysis"])["quadratic_model"])["score"]
        finite_rows = [row for row in rows if _finite(row.get("score"))]
        scale = max(float(row["score"]) for row in finite_rows) - min(
            float(row["score"]) for row in finite_rows
        )
        for index, action_id in enumerate(action_ids, 1):
            row = by_id[action_id]
            p = float(row["potential_coordinate"])
            c = float(row["current_coordinate"])
            aligned = predict_ec(model, p, c)
            misspecified = aligned if index == 1 else predict_ec(model, 1.0 - p, c)
            margin = (
                abs(misspecified - float(row["score"])) - abs(aligned - float(row["score"]))
            ) / scale
            information = abs(misspecified - aligned) / scale
            before = evidence_sum
            evidence_sum += margin
            raw_prediction_gap = abs(misspecified - aligned)
            reliable = bool(
                index > 1
                and evidence_sum >= minimum_margin
                and repeated_noise_supported
                and raw_prediction_gap >= float(policy["noise_multiplier"]) * float(noise_sigma)
            )
            steps.append(
                _step(
                    index=index,
                    action_id=action_id,
                    observation={"score": row["score"], "exact_replay": row["exact_replay"]},
                    statistic_before=before,
                    statistic_after=evidence_sum,
                    reliable=reliable,
                    information_gain_over_default=information - default_information,
                    stop_reason=(
                        "held-out reflection evidence and repeated-noise margin crossed thresholds"
                        if reliable
                        else "continue"
                    ),
                )
            )
            if index == 1:
                default_information = information
    else:
        models = _mapping(_mapping(report["legacy_analysis"])["quadratic_models"])
        score_model = models["score"]
        risk_model = models["safety_risk"]
        finite_rows = [
            row for row in rows if _finite(row.get("score")) and _finite(row.get("safety_risk"))
        ]
        score_scale = max(float(row["score"]) for row in finite_rows) - min(
            float(row["score"]) for row in finite_rows
        )
        risk_scale = max(float(row["safety_risk"]) for row in finite_rows) - min(
            float(row["safety_risk"]) for row in finite_rows
        )
        for index, action_id in enumerate(action_ids, 1):
            row = by_id[action_id]
            temperature = float(row["temperature_K"])
            duration = float(row["duration_s"])
            aligned_score = predict_rx(score_model, temperature, duration)
            aligned_risk = predict_rx(risk_model, temperature, duration)
            reflected_temperature = 840.0 - temperature
            if index == 1:
                reflected_temperature = temperature
            false_score = predict_rx(score_model, reflected_temperature, duration)
            false_risk = predict_rx(risk_model, reflected_temperature, duration)
            aligned_error = 0.5 * (
                abs(aligned_score - float(row["score"])) / score_scale
                + abs(aligned_risk - float(row["safety_risk"])) / risk_scale
            )
            false_error = 0.5 * (
                abs(false_score - float(row["score"])) / score_scale
                + abs(false_risk - float(row["safety_risk"])) / risk_scale
            )
            information = 0.5 * (
                abs(false_score - aligned_score) / score_scale
                + abs(false_risk - aligned_risk) / risk_scale
            )
            before = evidence_sum
            evidence_sum += false_error - aligned_error
            raw_prediction_gap = max(
                abs(false_score - aligned_score), abs(false_risk - aligned_risk)
            )
            reliable = bool(
                index > 1
                and evidence_sum >= minimum_margin
                and repeated_noise_supported
                and raw_prediction_gap >= float(policy["noise_multiplier"]) * float(noise_sigma)
            )
            steps.append(
                _step(
                    index=index,
                    action_id=action_id,
                    observation={
                        "score": row["score"],
                        "safety_risk": row["safety_risk"],
                        "exact_replay": row["exact_replay"],
                    },
                    statistic_before=before,
                    statistic_after=evidence_sum,
                    reliable=reliable,
                    information_gain_over_default=information - default_information,
                    stop_reason=(
                        "held-out reflection evidence and repeated-noise margin crossed thresholds"
                        if reliable
                        else "continue"
                    ),
                )
            )
            if index == 1:
                default_information = information
    result = _finish_trace(
        steps,
        int(policy["participant_budget"]),
        float(policy["minimum_information_gain"]),
    )
    result["repeated_noise_support"] = dict(calibration)
    return result


def _ec_structural_trace(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    del report
    spec = structural.candidate_specs()["electrochemical_transport"]
    main = [row for row in rows if row.get("phase") == "main_grid"]
    validation = [row for row in rows if row.get("phase") == "noisy_validation"]
    groups = ((0, 0), (0, 1), (0, 2))
    sigma = structural._validation_sigma(validation, spec["metrics"], groups=groups)
    recomputed = _structural_model_qualification_repair(
        main,
        validation,
        sigma=sigma,
        metrics=spec["model_metrics"],
        aligned_features=structural._electrochemical_aligned_features,
        misspecified_features=structural._electrochemical_misspecified_features,
        validation_groups=groups,
        target_axis="b",
        candidate_id="electrochemical_transport",
        effect_floor=float(policy["effect_floor"]),
        noise_multiplier=float(policy["model_noise_multiplier"]),
        minimum_disagreement_fraction=float(policy["minimum_disagreement_fraction"]),
    )
    comparisons = [_mapping(row) for row in _sequence(recomputed["comparisons"])]
    action_levels = [int(value) for value in policy["ordered_axis_b_indices"]]
    steps: list[dict[str, Any]] = []
    best = 0.0
    default_information = 0.0
    for index, level in enumerate(action_levels, 1):
        subset = [
            row
            for row in comparisons
            if int(row["axis_a_index"]) == 0 and int(row["axis_b_index"]) == level
        ]
        if len(subset) != 3:
            raise ValueError("EC-S sequential condition lacks three registered metrics")
        evidence = max(
            (float(row["misspecified_error"]) - float(row["aligned_error"]))
            / max(float(row["disagreement_gate"]), 1.0e-12)
            for row in subset
        )
        information = max(
            abs(float(row["prediction_difference"])) / max(float(row["disagreement_gate"]), 1.0e-12)
            for row in subset
        )
        before = best
        best = max(best, evidence)
        validation_rows = [
            row
            for row in rows
            if row.get("phase") == "noisy_validation"
            and int(row.get("axis_a_index", -1)) == 0
            and int(row.get("axis_b_index", -1)) == level
        ]
        noise_coordinates = {
            str(row.get("observation_coordinate_sha256")) for row in validation_rows
        }
        repeated_noise_supported = bool(
            len(validation_rows) >= int(policy["minimum_noise_replicates"])
            and len(noise_coordinates) == len(validation_rows)
            and "None" not in noise_coordinates
        )
        reliable = bool(
            index > 1
            and best >= float(policy["minimum_standardized_evidence"])
            and repeated_noise_supported
        )
        observations = [
            {
                "metric": row["metric"],
                "observed_validation_mean": row["observed_validation_mean"],
            }
            for row in subset
        ]
        steps.append(
            _step(
                index=index,
                action_id=f"axis_a=0;axis_b={level}",
                observation=observations,
                statistic_before=before,
                statistic_after=best,
                reliable=reliable,
                information_gain_over_default=information - default_information,
                stop_reason="model-error evidence crossed the frozen gate"
                if reliable
                else "continue",
            )
        )
        if index == 1:
            default_information = information
    return _finish_trace(
        steps,
        int(policy["participant_budget"]),
        float(policy["minimum_information_gain"]),
    )


def _rx_structural_trace(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    del report
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            grouped[(str(row.get("cell_id")), str(row.get("law_id")))].append(row)
    metrics = ("yield", "conversion", "selectivity")
    action_ids = [str(value) for value in policy["ordered_action_ids"]]
    gaps: list[dict[str, float]] = []
    steps: list[dict[str, Any]] = []
    default_information = 0.0
    repeated_noise_support: dict[str, int] = {}
    offline_separability: list[dict[str, Any]] = []
    for index, action_id in enumerate(action_ids, 1):
        baseline_rows = grouped[(action_id, "deactivating_baseline")]
        target_rows = grouped[(action_id, "reversible_target_pathway")]
        if not baseline_rows or not target_rows:
            raise ValueError("RX-S action lacks one of the frozen candidate families")
        target_noise_keys = {str(row.get("direct_noise_key_sha256")) for row in target_rows}
        if "None" in target_noise_keys or len(target_noise_keys) != len(target_rows):
            raise ValueError("RX-S target observations lack unique raw noise identities")
        repeated_noise_support[action_id] = len(target_noise_keys)
        target_means = {
            metric: fmean(float(row["direct_metrics"][metric]) for row in target_rows)
            for metric in metrics
        }
        baseline_predictions = {
            metric: fmean(float(row["direct_metrics"][metric]) for row in baseline_rows)
            for metric in metrics
        }
        gap = {metric: target_means[metric] - baseline_predictions[metric] for metric in metrics}
        gaps.append(gap)
        information = sum(abs(value) for value in gap.values())
        accumulation = (
            min(gap[metric] - gaps[0][metric] for metric in ("yield", "conversion"))
            if index > 1
            else 0.0
        )
        repeated_noise_supported = len(target_noise_keys) >= int(policy["minimum_noise_replicates"])
        reliable = bool(
            index > 1
            and accumulation >= float(policy["minimum_accumulation"])
            and repeated_noise_supported
        )
        offline_separability.append(
            {
                "action_id": action_id,
                "candidate_prediction_family": "deactivating_baseline",
                "hidden_observation_family": "reversible_target_pathway",
                "baseline_prediction": baseline_predictions,
                "target_mean": target_means,
                "paired_gap": gap,
            }
        )
        steps.append(
            _step(
                index=index,
                action_id=action_id,
                observation={
                    "hidden_world_law": "reversible_target_pathway",
                    "target_noise_keys": sorted(target_noise_keys),
                    "target_observed_metrics": target_means,
                },
                statistic_before=0.0
                if index == 1
                else min(gaps[-2][metric] - gaps[0][metric] for metric in ("yield", "conversion")),
                statistic_after=accumulation,
                reliable=reliable,
                information_gain_over_default=information - default_information,
                stop_reason="two-condition accumulation signature crossed threshold"
                if reliable
                else "continue",
            )
        )
        if index == 1:
            default_information = information
    result = _finish_trace(
        steps,
        int(policy["participant_budget"]),
        float(policy["minimum_information_gain"]),
    )
    result["repeated_noise_support"] = {
        "observed_replicates_by_condition": repeated_noise_support,
        "required": policy["minimum_noise_replicates"],
        "derivation": "unique direct_noise_key_sha256 values in hidden-world receipts",
    }
    result["participant_policy_semantics"] = (
        "fixed action order; each observation digest contains only the single hidden "
        "reversible-target-pathway outcome"
    )
    result["offline_paired_family_separability"] = offline_separability
    return result


def _pa_parametric_trace(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    aligned_band = _sequence(_mapping(report.get("aligned_prior")).get("k_star_band"))
    false_band = _sequence(_mapping(report.get("misspecified_prior")).get("k_star_band"))
    if len(aligned_band) != 2 or len(false_band) != 2:
        raise ValueError("PA-P public prior bands are missing")
    aligned_k = fmean(float(value) for value in aligned_band)
    false_k = fmean(float(value) for value in false_band)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            grouped[str(row.get("point_id"))].append(row)
    action_ids = [str(value) for value in policy["ordered_action_ids"]]
    steps: list[dict[str, Any]] = []
    default_gap = 0.0
    for index, action_id in enumerate(action_ids, 1):
        action_rows = grouped[action_id]
        if len(action_rows) < int(policy["minimum_noise_replicates"]):
            raise ValueError("PA-P action lacks preregistered repeated-noise support")
        noise_coordinates = {
            (str(row.get("observation_noise_namespace")), int(row["observation_seed"]))
            for row in action_rows
        }
        if len(noise_coordinates) != len(action_rows):
            raise ValueError("PA-P repeated observations do not use unique noise coordinates")
        allocations = []
        for row in action_rows:
            measurement = _mapping(row.get("measurement"))
            organic = float(measurement["product_in_organic"])
            aqueous = float(measurement["product_in_aqueous"])
            allocations.append(organic / max(organic + aqueous, 1.0e-12))
        observed_mean, standard_error = _mean_se(allocations)
        exemplar = action_rows[0]
        organic_volume = float(exemplar["extractant_volume_L"])
        aqueous_volume = float(exemplar["solvent_volume_L"]) + float(exemplar["aqueous_volume_L"])
        aligned_prediction = (
            aligned_k * organic_volume / (aligned_k * organic_volume + aqueous_volume)
        )
        false_prediction = false_k * organic_volume / (false_k * organic_volume + aqueous_volume)
        prediction_gap = abs(aligned_prediction - false_prediction)
        evidence_margin = abs(false_prediction - observed_mean) - abs(
            aligned_prediction - observed_mean
        )
        signal_to_noise = prediction_gap / max(standard_error, 1.0e-12)
        reliable = bool(
            evidence_margin >= float(policy["minimum_prediction_gap"])
            and prediction_gap >= float(policy["minimum_prediction_gap"])
            and signal_to_noise >= float(policy["minimum_signal_to_noise_ratio"])
        )
        if index == 1:
            default_gap = prediction_gap
        steps.append(
            _step(
                index=index,
                action_id=action_id,
                observation={
                    "receipt_sha256": [str(row.get("receipt_sha256")) for row in action_rows],
                    "replicate_count": len(action_rows),
                    "observed_mean_organic_allocation": observed_mean,
                    "standard_error": standard_error,
                },
                statistic_before=float(steps[-1]["statistic_after"]) if steps else 0.0,
                statistic_after=evidence_margin,
                reliable=reliable,
                information_gain_over_default=prediction_gap - default_gap,
                stop_reason=(
                    "raw replicated observations reject the false public prior"
                    if reliable
                    else "continue"
                ),
            )
        )
    return _finish_trace(
        steps,
        int(policy["participant_budget"]),
        float(policy["minimum_information_gain"]),
    )


def _pa_structural_trace(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    del report
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            grouped[(str(row.get("pair_id")), str(row.get("law_id")))].append(row)
    action_ids = [str(value) for value in policy["ordered_action_ids"]]
    points: list[tuple[float, float]] = []
    steps: list[dict[str, Any]] = []
    deviation = 0.0
    repeated_noise_support: dict[str, int] = {}
    offline_separability: list[dict[str, Any]] = []
    for index, action_id in enumerate(action_ids, 1):
        linear_rows = grouped[(action_id, "linear_response")]
        power_rows = grouped[(action_id, "power_response")]
        if not linear_rows or not power_rows:
            raise ValueError("PA-S action lacks one of the frozen candidate families")
        target_noise_keys = {
            (
                str(row.get("observation_noise_namespace")),
                int(row["observation_seed"]),
            )
            for row in power_rows
        }
        if len(target_noise_keys) != len(power_rows):
            raise ValueError("PA-S target observations lack unique raw noise identities")
        repeated_noise_support[action_id] = len(target_noise_keys)

        def mean_log_ratio(action_rows: Sequence[Mapping[str, Any]]) -> float:
            values = []
            for row in action_rows:
                measurement = _mapping(row.get("measurement"))
                values.append(
                    math.log(
                        float(measurement["product_in_organic"])
                        / float(measurement["product_in_aqueous"])
                    )
                )
            return fmean(values)

        x = mean_log_ratio(linear_rows)
        y = mean_log_ratio(power_rows)
        points.append((x, y))
        before = deviation
        if len(points) >= 2:
            x_mean = fmean(row[0] for row in points)
            y_mean = fmean(row[1] for row in points)
            denominator = sum((row[0] - x_mean) ** 2 for row in points)
            if denominator > 0.0:
                slope = sum((row[0] - x_mean) * (row[1] - y_mean) for row in points) / denominator
                deviation = abs(slope - 1.0)
        repeated_noise_supported = len(target_noise_keys) >= int(policy["minimum_noise_replicates"])
        reliable = bool(
            len(points) >= 2
            and deviation >= float(policy["minimum_slope_deviation"])
            and repeated_noise_supported
        )
        steps.append(
            _step(
                index=index,
                action_id=action_id,
                observation={
                    "hidden_world_law": "power_response",
                    "target_receipts": sorted(str(row.get("receipt_sha256")) for row in power_rows),
                    "target_log_partition_ratio": y,
                },
                statistic_before=before,
                statistic_after=deviation,
                reliable=reliable,
                information_gain_over_default=deviation,
                stop_reason="two-condition log-slope deviation crossed threshold"
                if reliable
                else "continue",
            )
        )
        offline_separability.append(
            {
                "action_id": action_id,
                "candidate_prediction_family": "linear_response",
                "hidden_observation_family": "power_response",
                "linear_candidate_log_partition_ratio": x,
                "power_target_log_partition_ratio": y,
                "absolute_gap": abs(y - x),
            }
        )
    result = _finish_trace(
        steps,
        int(policy["participant_budget"]),
        float(policy["minimum_information_gain"]),
    )
    result["repeated_noise_support"] = {
        "observed_replicates_by_condition": repeated_noise_support,
        "required": policy["minimum_noise_replicates"],
        "derivation": (
            "unique (observation_noise_namespace, observation_seed) values in hidden-world receipts"
        ),
    }
    result["participant_policy_semantics"] = (
        "fixed action order; each observation digest contains only the single hidden "
        "power-response outcome"
    )
    result["offline_paired_family_separability"] = offline_separability
    return result


def _c_parametric_trace(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    metric = str(policy["metric"])
    grouped: dict[float, list[float]] = defaultdict(list)
    digests: dict[float, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed" and _finite(_mapping(row.get("metrics")).get(metric)):
            temperature = float(row["temperature_K"])
            grouped[temperature].append(float(row["metrics"][metric]))
            digests[temperature].append(str(row.get("receipt_sha256")))
    action_ids = [float(value) for value in policy["ordered_temperatures_K"]]
    aligned = [float(value) for value in report["aligned_effect_band"]]
    false = [float(value) for value in report["misspecified_effect_band"]]
    seen: set[float] = set()
    steps: list[dict[str, Any]] = []
    statistic = 0.0
    for index, temperature in enumerate(action_ids, 1):
        seen.add(temperature)
        before = statistic
        reliable = False
        reason = "both frozen endpoint conditions have not yet been observed"
        if {270.0, 310.0} <= seen:
            low_mean, low_se = _mean_se(grouped[270.0])
            high_mean, high_se = _mean_se(grouped[310.0])
            effect = abs(low_mean - high_mean)
            snr = effect / max(math.sqrt(low_se**2 + high_se**2), 1.0e-12)
            statistic = snr
            reliable = bool(
                aligned[0] <= effect <= aligned[1]
                and not false[0] <= effect <= false[1]
                and snr >= float(policy["minimum_snr"])
            )
            reason = (
                "endpoint gain selected aligned band and rejected false band"
                if reliable
                else "endpoint evidence remained inconclusive"
            )
        steps.append(
            _step(
                index=index,
                action_id=f"temperature_K={temperature:g}",
                observation={"row_digests": digests[temperature]},
                statistic_before=before,
                statistic_after=statistic,
                reliable=reliable,
                information_gain_over_default=statistic,
                stop_reason=reason,
            )
        )
    return _finish_trace(
        steps,
        int(policy["participant_budget"]),
        float(policy["minimum_information_gain"]),
    )


TRACE_FAMILIES = {
    "entity_sequential_contrast": _entity_trace,
    "ec_rx_reflection_sequence": _reflection_trace,
    "ec_structural_sequence": _ec_structural_trace,
    "rx_structural_sequence": _rx_structural_trace,
    "pa_parametric_sequence": _pa_parametric_trace,
    "pa_structural_sequence": _pa_structural_trace,
    "c_parametric_sequence": _c_parametric_trace,
}


def validate_contract(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if contract.get("schema_version") not in {
        "chemworld-experiment-1-challenge-probes-contract-1.1",
        "chemworld-experiment-1-challenge-probes-contract-1.1.1",
        "chemworld-experiment-1-challenge-probes-contract-1.2",
    }:
        raise ValueError("unexpected executable challenge contract schema")
    if contract.get("schema_version") == "chemworld-experiment-1-challenge-probes-contract-1.1.1":
        adapter_binding = _mapping(contract.get("adapter_parent_contract"))
        adapter_path = root / str(adapter_binding.get("path", ""))
        if not adapter_path.is_file() or file_sha256(adapter_path) != adapter_binding.get("sha256"):
            raise ValueError("v1.1 adapter parent binding mismatch")
        adapter_parent = _load_json(adapter_path)
        if not isinstance(adapter_parent, dict) or adapter_parent.get("loci") != contract.get(
            "loci"
        ):
            raise ValueError("v1.1.1 changed a frozen threshold or sequential policy")
        deviation_binding = _mapping(contract.get("adapter_deviation_receipt"))
        deviation_path = root / str(deviation_binding.get("path", ""))
        if not deviation_path.is_file() or file_sha256(deviation_path) != deviation_binding.get(
            "sha256"
        ):
            raise ValueError("v1.1.1 deviation receipt binding mismatch")
    if contract.get("schema_version") == "chemworld-experiment-1-challenge-probes-contract-1.2":
        repair_binding = _mapping(contract.get("repair_parent_contract"))
        repair_path = root / str(repair_binding.get("path", ""))
        if not repair_path.is_file() or file_sha256(repair_path) != repair_binding.get("sha256"):
            raise ValueError("v1.2 repair parent binding mismatch")
        repair_parent = _load_json(repair_path)
        if not isinstance(repair_parent, dict):
            raise ValueError("v1.2 repair parent must be an object")
        expected_loci = json.loads(json.dumps(contract.get("loci")))
        if not isinstance(expected_loci, dict):
            raise ValueError("v1.2 loci must be an object")
        for block in ("RX-S", "PA-S"):
            _mapping(expected_loci.get(block))
            expected_loci[block]["observed_noise_replicates_per_condition"] = 1
        for key in (
            "effect_floor",
            "model_noise_multiplier",
            "minimum_disagreement_fraction",
        ):
            expected_loci["EC-S"].pop(key, None)
        for key in (
            "minimum_prediction_gap",
            "minimum_signal_to_noise_ratio",
            "minimum_noise_replicates",
        ):
            expected_loci["PA-P"].pop(key, None)
        if repair_parent.get("loci") != expected_loci:
            raise ValueError("v1.2 changed more than the declared repair adapter fields")
        deviation_binding = _mapping(contract.get("repair_deviation_receipt"))
        deviation_path = root / str(deviation_binding.get("path", ""))
        if not deviation_path.is_file() or file_sha256(deviation_path) != deviation_binding.get(
            "sha256"
        ):
            raise ValueError("v1.2 deviation receipt binding mismatch")
    parent_binding = _mapping(contract.get("parent_contract"))
    parent_path = root / str(parent_binding.get("path", ""))
    if not parent_path.is_file() or file_sha256(parent_path) != parent_binding.get("sha256"):
        raise ValueError("v1.0 parent contract binding mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    v1.validate_contract(parent, root)
    for name in ("source_registry", "protocol", "experiment_note"):
        binding = _mapping(contract.get(name))
        path = root / str(binding.get("path", ""))
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"{name} binding mismatch")
    manifest_binding = _mapping(contract.get("raw_evidence_manifest"))
    manifest_path = root / str(manifest_binding.get("path", ""))
    if not manifest_path.is_file() or file_sha256(manifest_path) != manifest_binding.get("sha256"):
        raise ValueError("raw evidence manifest binding mismatch")
    policies = _mapping(contract.get("loci"))
    if tuple(policies) != EXPECTED_BLOCKS:
        raise ValueError("v1.1 challenge must bind the ordered ten-locus denominator")
    for block, policy_any in policies.items():
        policy = _mapping(policy_any)
        if policy.get("sequential_family") not in TRACE_FAMILIES:
            raise ValueError(f"unknown sequential family for {block}")
        if not policy.get("raw_evidence_filename"):
            raise ValueError(f"raw evidence filename missing for {block}")
        calibration = policy.get("noise_calibration")
        if calibration is not None:
            binding = _mapping(calibration)
            path = root / str(binding.get("path", ""))
            if not path.is_file() or file_sha256(path) != binding.get("sha256"):
                raise ValueError(f"noise calibration binding mismatch for {block}")
    return parent


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must contain a JSON object or array")
    return value


def verify_canonical_self_hash(payload: Mapping[str, Any], key: str) -> str:
    declared = payload.get(key)
    if not isinstance(declared, str):
        raise ValueError(f"{key} is missing")
    unhashed = dict(payload)
    unhashed.pop(key, None)
    computed = canonical_json_sha256(unhashed)
    if declared != computed:
        raise ValueError(f"{key} canonical self-hash mismatch")
    return declared


def load_bound_registry(
    contract: Mapping[str, Any], root: Path, registry_path: Path
) -> dict[str, Any]:
    binding = _mapping(contract.get("source_registry"))
    expected = (root / str(binding.get("path", ""))).resolve()
    actual = registry_path.resolve()
    if actual != expected:
        raise ValueError("registry path does not match the exact contract binding")
    if file_sha256(actual) != binding.get("sha256"):
        raise ValueError("registry file digest does not match the contract binding")
    registry = _load_json(actual)
    if not isinstance(registry, dict):
        raise ValueError("registry must be an object")
    declared = verify_canonical_self_hash(registry, "registry_sha256")
    if declared != binding.get("registry_sha256"):
        raise ValueError("registry self-hash does not match the contract binding")
    return registry


def validate_raw_artifact(
    raw_path: Path,
    manifest_row: Mapping[str, Any],
    *,
    report_path: str,
) -> list[Mapping[str, Any]]:
    if manifest_row.get("report_path") != report_path:
        raise ValueError("raw manifest report path mismatch")
    if manifest_row.get("raw_filename") != raw_path.name:
        raise ValueError("raw manifest filename mismatch")
    if file_sha256(raw_path) != manifest_row.get("raw_sha256"):
        raise ValueError("raw evidence digest mismatch")
    raw = _load_json(raw_path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("raw public execution evidence must be a non-empty array")
    raw_rows = [_mapping(row) for row in raw]
    if any(not row for row in raw_rows):
        raise ValueError("raw execution evidence contains a malformed row")
    identities = []
    for raw_row in raw_rows:
        identity = raw_row.get("receipt_sha256") or raw_row.get("evaluation_id")
        if identity is None and raw_row.get("query_id") is not None:
            identity = f"{raw_row.get('query_id')}:{raw_row.get('law_id', '')}"
        if identity is None and raw_row.get("cell_id") is not None:
            identity = f"{raw_row.get('cell_id')}:{raw_row.get('law_id', '')}"
        identities.append(str(identity))
    if "None" in identities or len(set(identities)) != len(identities):
        raise ValueError("raw execution evidence has missing or duplicate identities")
    if len(raw_rows) != int(manifest_row.get("raw_rows", -1)):
        raise ValueError("raw evidence row count mismatch")
    return raw_rows


def validate_report_artifact(
    report_path: Path,
    manifest_row: Mapping[str, Any],
    registry_evidence: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, Any]:
    if manifest_row.get("report_path") != relative_path:
        raise ValueError("raw manifest report path mismatch")
    actual_report_sha = file_sha256(report_path)
    if manifest_row.get("report_sha256") != registry_evidence.get("sha256"):
        raise ValueError("manifest report digest disagrees with registry evidence")
    if actual_report_sha != registry_evidence.get("sha256"):
        raise ValueError("source report file digest mismatch")
    if actual_report_sha != manifest_row.get("report_sha256"):
        raise ValueError("source report digest mismatch against raw manifest")
    report = _load_json(report_path)
    if not isinstance(report, dict):
        raise ValueError("world report must be an object")
    unhashed = dict(report)
    self_hash = unhashed.pop("report_sha256", None)
    if self_hash != registry_evidence.get("report_sha256") or self_hash != canonical_json_sha256(
        unhashed
    ):
        raise ValueError("source report self hash mismatch")
    return report


def build_probe_summary(
    contract: Mapping[str, Any],
    registry_path: Path,
    *,
    root: Path,
    evidence_roots: Sequence[Path],
    source_commit: str,
) -> dict[str, Any]:
    parent = validate_contract(contract, root)
    registry = load_bound_registry(contract, root, registry_path)
    rows = registry.get("rows")
    if not isinstance(rows, list) or len(rows) != 105:
        raise ValueError("v1.1 challenge requires the complete 105-row registry")
    policies = _mapping(contract["loci"])
    parent_rules = _mapping(parent["loci"])
    manifest_path = root / str(_mapping(contract["raw_evidence_manifest"])["path"])
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("raw evidence manifest must be an object")
    verify_canonical_self_hash(manifest, "manifest_sha256")
    if manifest.get("source_registry_sha256") != registry.get("registry_sha256"):
        raise ValueError("raw evidence manifest is stale for this registry")
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != 50:
        raise ValueError("raw evidence manifest must bind 50 rows")
    manifest_by_unit = {
        str(row.get("unit_id")): row for row in manifest_rows if isinstance(row, Mapping)
    }
    if len(manifest_by_unit) != 50:
        raise ValueError("raw evidence manifest contains missing or duplicate unit IDs")
    selected: list[tuple[str, Mapping[str, Any]]] = []
    for row_any in rows:
        row = _mapping(row_any)
        block = f"{row.get('system_id', '')}-{str(row.get('prior_locus', ''))[:1].upper()}"
        if block in policies:
            selected.append((block, row))
    if len(selected) != 50 or len({str(row["unit_id"]) for _, row in selected}) != 50:
        raise ValueError("candidate denominator is missing or duplicated")

    probe_rows: list[dict[str, Any]] = []
    for block, registry_row in selected:
        failures: list[str] = []
        payload: dict[str, Any] = {}
        evidence = _mapping(_mapping(registry_row.get("qualification_run")).get("evidence"))
        relative = str(evidence.get("path", ""))
        try:
            if registry_row.get("current_status") != "qualified-development":
                raise ValueError("source unit is not qualified-development")
            report_path = v1._resolve_report(relative, evidence_roots)
            manifest_row = _mapping(manifest_by_unit.get(str(registry_row["unit_id"])))
            report = validate_report_artifact(
                report_path,
                manifest_row,
                evidence,
                relative_path=relative,
            )
            policy = _mapping(policies[block])
            policy = dict(policy)
            calibration_binding = policy.get("noise_calibration")
            if calibration_binding is not None:
                binding = _mapping(calibration_binding)
                calibration_doc = _load_json(root / str(binding["path"]))
                if not isinstance(calibration_doc, dict):
                    raise ValueError("noise calibration must be an object")
                world_seed = int(registry_row["world_seed"])
                calibration_worlds = [
                    row
                    for row in calibration_doc.get("worlds", [])
                    if isinstance(row, Mapping) and int(row.get("world_seed", -1)) == world_seed
                ]
                if len(calibration_worlds) != 1:
                    raise ValueError("noise calibration World binding is missing or duplicated")
                policy["noise_calibration"] = dict(
                    _mapping(_mapping(calibration_worlds[0]).get("analysis")).get(
                        "validation_noise"
                    )
                )
            raw_path = report_path.parent / str(policy["raw_evidence_filename"])
            raw_rows = validate_raw_artifact(raw_path, manifest_row, report_path=relative)
            old_rule = _mapping(parent_rules[block])
            if str(policy["sequential_family"]) == "entity_sequential_contrast" and not isinstance(
                report.get("descriptor_permutation"), list
            ):
                policy["descriptor_permutation"] = descriptor_permutation_from_bound_contracts(
                    root,
                    [_mapping(row) for row in _sequence(old_rule.get("source_contracts"))],
                )
            old_probe = v1.PROBE_FAMILIES[str(old_rule["probe_family"])](report, old_rule)
            family = str(policy["sequential_family"])
            if family == "ec_rx_reflection_sequence":
                trace = _reflection_trace(block, report, raw_rows, policy)
            else:
                trace = TRACE_FAMILIES[family](report, raw_rows, policy)
            payload = {
                "plausibility_passed": old_probe["plausibility_passed"],
                "plausibility_evidence": old_probe["plausibility_evidence"],
                "raw_evidence": {
                    "filename": raw_path.name,
                    "sha256": file_sha256(raw_path),
                    "rows": len(raw_rows),
                },
                **trace,
            }
            for name, passed in {
                "plausibility": payload["plausibility_passed"],
                "information_choice": payload["active_information_passed"],
                "budget_window": payload["budget_window_passed"],
            }.items():
                if not passed:
                    failures.append(name)
        except Exception as exc:
            failures.append(f"probe_error:{type(exc).__name__}:{exc}")
        probe_rows.append(
            {
                "block": block,
                "unit_id": registry_row.get("unit_id"),
                "world_id": registry_row.get("world_id"),
                "source_evidence_path": relative,
                "source_evidence_sha256": evidence.get("sha256"),
                "probe": payload,
                "failures": failures,
                "completed": bool(payload),
            }
        )

    locus_rows: list[dict[str, Any]] = []
    for block in EXPECTED_BLOCKS:
        subset = [row for row in probe_rows if row["block"] == block]
        defaults = sum(
            _mapping(row.get("probe")).get("default_one_shot_reliably_discriminates") is True
            for row in subset
        )
        checks = {
            "plausibility": len(subset) == 5
            and all(_mapping(row["probe"]).get("plausibility_passed") is True for row in subset),
            "non_triviality": len(subset) == 5 and defaults < 5,
            "information_choice": len(subset) == 5
            and all(
                _mapping(row["probe"]).get("active_information_passed") is True for row in subset
            ),
            "budget_window": len(subset) == 5
            and all(_mapping(row["probe"]).get("budget_window_passed") is True for row in subset),
        }
        locus_rows.append(
            {
                "block": block,
                "world_probe_rows": len(subset),
                "default_one_shot_success_worlds": defaults,
                "measured_stopping_costs": [
                    _mapping(row["probe"]).get("minimum_reliable_unique_condition_cost")
                    for row in subset
                ],
                "checks": checks,
                "challenge_probe_passed": all(checks.values()),
                "failures": [name for name, passed in checks.items() if not passed],
            }
        )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "formal_result": False,
        "evidence_semantics": "challenge-development-attempt-3",
        "provider_call_count": 0,
        "participant_execution_authorized": False,
        "confirmation_execution_authorized": False,
        "source_commit": source_commit,
        "contract_sha256": canonical_json_sha256(contract),
        "source_registry_file_sha256": file_sha256(registry_path.resolve()),
        "source_registry_sha256": registry.get("registry_sha256"),
        "source_registry": {
            "path": _mapping(contract["source_registry"])["path"],
            "file_sha256": file_sha256(registry_path.resolve()),
            "registry_sha256": registry.get("registry_sha256"),
        },
        "raw_evidence_manifest": {
            "path": _mapping(contract["raw_evidence_manifest"])["path"],
            "file_sha256": file_sha256(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "denominators": {
            "planned_world_probe_rows": 50,
            "attempted_world_probe_rows": len(probe_rows),
            "completed_world_probe_rows": sum(row["completed"] for row in probe_rows),
            "failed_world_probe_rows": sum(bool(row["failures"]) for row in probe_rows),
            "candidate_loci": 10,
        },
        "world_probe_rows": probe_rows,
        "loci": locus_rows,
        "challenge_passed_loci": sum(row["challenge_probe_passed"] for row in locus_rows),
        "challenge_failed_loci": sum(not row["challenge_probe_passed"] for row in locus_rows),
    }
    summary["challenge_probe_sha256"] = canonical_json_sha256(summary)
    return summary
