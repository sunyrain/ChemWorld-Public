"""Core analysis primitives for the mechanism-adaptation benchmark.

The module deliberately separates declared model reports from calibrated environment
posteriors.  Helpers that operate on declared distributions never call their outputs
Bayesian beliefs or information gain.  The small Gaussian oracle is an executable
reference implementation for candidate generators that can provide public-observation
samples without revealing the true candidate or nuisance realization.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

from chemworld.eval.mechanism_gate_decision import gate_a_certificate_decision

MECHANISM_ADAPTATION_ANALYSIS_VERSION = "chemworld-mechanism-adaptation-analysis-0.3"

REQUIRED_GATE_IDS = ("gate_0", "gate_a", "gate_b", "gate_c", "gate_d", "gate_e")
REQUIRED_EVALUATION_TRACK_IDS = (
    "static_world_identification",
    "calibrated_online_change",
    "uncalibrated_nonstationarity",
    "adaptive_recovery",
)
PRIMARY_CHANGE_TRACK_ID = "calibrated_online_change"

AutonomyStatus = Literal[
    "fully_autonomous_campaign",
    "autonomous_current_experiment_with_assisted_history",
    "assisted_campaign",
    "incomplete_campaign",
]


def load_mechanism_adaptation_protocol(path: str | Path) -> dict[str, Any]:
    """Load a full protocol or one hash-pinned versioned composition.

    The composition form lets a new scientific design reuse stable task and data
    contracts without copying hundreds of lines.  It is deliberately constrained:
    one relative, hash-pinned base; explicit dotted-path removals; and one recursive
    override object.
    """

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"mechanism-adaptation protocol must be an object: {source}")
    if (
        raw.get("schema_version")
        != "chemworld-mechanism-adaptation-protocol-composition-0.1"
    ):
        return raw
    extends = raw.get("extends")
    expected_base_sha = str(raw.get("base_sha256", "")).lower()
    overrides = raw.get("overrides")
    remove_paths = raw.get("remove_paths", [])
    if (
        not isinstance(extends, str)
        or not extends
        or not isinstance(overrides, Mapping)
        or not isinstance(remove_paths, list)
    ):
        raise ValueError("invalid mechanism-adaptation protocol composition")
    base_path = (source.parent / extends).resolve()
    if source.parent.resolve() not in base_path.parents:
        raise ValueError("composed protocol base must stay inside the protocol directory")
    observed_base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if observed_base_sha != expected_base_sha:
        raise ValueError(
            "composed protocol base hash mismatch: "
            f"expected {expected_base_sha}, observed {observed_base_sha}"
        )
    base = load_mechanism_adaptation_protocol(base_path)
    resolved = _deep_merge_protocol(base, overrides)
    for raw_path in remove_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("protocol remove_paths must contain dotted strings")
        _remove_protocol_path(resolved, raw_path)
    resolved["protocol_composition"] = {
        "schema_version": raw["schema_version"],
        "source": source.name,
        "extends": base_path.name,
        "base_sha256": observed_base_sha,
        "remove_paths": list(remove_paths),
    }
    return resolved


def _deep_merge_protocol(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[str(key)] = _deep_merge_protocol(
                result[str(key)],
                value,
            )
        else:
            result[str(key)] = value
    return result


def _remove_protocol_path(protocol: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    target: dict[str, Any] = protocol
    for part in parts[:-1]:
        value = target.get(part)
        if not isinstance(value, dict):
            raise ValueError(f"protocol removal path does not exist: {dotted_path}")
        target = value
    if parts[-1] not in target:
        raise ValueError(f"protocol removal path does not exist: {dotted_path}")
    del target[parts[-1]]


@dataclass(frozen=True)
class OutcomeLayers:
    """Keep intervention targets separate from evaluation truth."""

    environment_outcome: Mapping[str, Any]
    agent_visible_observation: Mapping[str, Any]
    evaluation_outcome: Mapping[str, Any]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "environment_outcome": dict(self.environment_outcome),
            "agent_visible_observation": dict(self.agent_visible_observation),
            "evaluation_outcome": dict(self.evaluation_outcome),
        }


def normalized_distribution(values: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize a finite categorical distribution."""

    if len(values) < 2:
        raise ValueError("a mechanism distribution requires at least two candidates")
    normalized: dict[str, float] = {}
    for key, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError("mechanism probabilities must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("mechanism probabilities must be finite and non-negative")
        normalized[str(key)] = value
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("mechanism probabilities must have positive total mass")
    return {key: value / total for key, value in normalized.items()}


def declared_change_probability(distribution: Mapping[str, float]) -> float:
    """Derive change probability from the declared no-change mass."""

    values = normalized_distribution(distribution)
    if "no_change" not in values:
        raise ValueError("declared mechanism distribution must include no_change")
    return 1.0 - values["no_change"]


def conditional_changed_family_distribution(
    distribution: Mapping[str, float],
) -> dict[str, float] | None:
    """Return q(family | change), or ``None`` when no change mass is one."""

    values = normalized_distribution(distribution)
    changed = {key: value for key, value in values.items() if key != "no_change"}
    mass = sum(changed.values())
    if mass <= 0.0:
        return None
    return {key: value / mass for key, value in changed.items()}


def evaluation_track(
    protocol: Mapping[str, Any],
    track_id: str = PRIMARY_CHANGE_TRACK_ID,
) -> Mapping[str, Any]:
    """Return one declared evaluation track from a v0.3 protocol."""

    tracks = protocol.get("evaluation_tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("protocol must declare evaluation_tracks")
    track = tracks.get(track_id)
    if not isinstance(track, Mapping):
        raise ValueError(f"protocol does not declare evaluation track {track_id!r}")
    return track


def reference_sufficiency_certificate(
    *,
    pre_change_action_ids: Sequence[str],
    required_reference_action_ids: Sequence[str],
    declared_relation_groups: Mapping[str, Sequence[Sequence[str]]],
    minimum_reference_experiments: int,
    predictive_information_non_saturated: bool,
    predictive_adequacy: Mapping[str, Any],
    universal_candidate_family_scope: bool,
    maximum_reference_age_experiments: int,
    observed_reference_ages: Mapping[str, int],
) -> dict[str, Any]:
    """Certify that an online history contains a usable old-world reference.

    The certificate is deliberately task-agnostic.  Mechanism-specific requirements
    enter only through declarative public action groups.  A trial is not treated as a
    mechanism-attribution failure when this prerequisite is absent; instead reference
    acquisition fails closed as its own Gate A3 component.
    """

    observed = [str(item) for item in pre_change_action_ids]
    reference_policy_actions = [
        str(item) for item in required_reference_action_ids
    ]
    if minimum_reference_experiments <= 0:
        raise ValueError("minimum reference experiments must be positive")
    if (
        not reference_policy_actions
        or len(reference_policy_actions) != len(set(reference_policy_actions))
    ):
        raise ValueError("reference-policy actions must be non-empty and unique")
    policy_action_set = set(reference_policy_actions)
    observed_set = set(observed)
    relation_coverage: dict[str, Any] = {}
    for declaration_id, raw_groups in sorted(declared_relation_groups.items()):
        groups = [
            tuple(str(action_id) for action_id in group)
            for group in raw_groups
        ]
        if not groups or any(len(group) < 2 for group in groups):
            raise ValueError(
                "declared diagnostic relations must contain groups of at least two actions"
            )
        covered_groups = [
            list(group) for group in groups if set(group).issubset(observed_set)
        ]
        relation_coverage[str(declaration_id)] = {
            "covered": bool(covered_groups),
            "covered_groups": covered_groups,
            "declared_groups": [list(group) for group in groups],
        }
    structural_checks = {
        "minimum_reference_experiments_met": (
            len(observed) >= minimum_reference_experiments
        ),
        "declared_relations_observed": all(
            item["covered"] for item in relation_coverage.values()
        ),
        "universal_candidate_family_scope": bool(
            universal_candidate_family_scope
        ),
        "predictive_information_non_saturated": bool(
            predictive_information_non_saturated
        ),
    }
    diagnostics = {
        "canonical_witness_action_set_completed": (
            policy_action_set.issubset(observed_set)
        ),
        "interpretation": (
            "The frozen policy's canonical actions are a reproducible witness set, "
            "not a certificate requirement. Pass/fail is controlled by relation "
            "closure, predictive adequacy, reference age, and minimum experiment count."
        ),
    }
    predictive_pass = predictive_adequacy.get("pass")
    if not isinstance(predictive_pass, bool):
        raise ValueError("predictive adequacy must provide a boolean pass field")
    ages = {str(key): int(value) for key, value in observed_reference_ages.items()}
    if not set(ages).issubset(policy_action_set) or any(
        value < 0 for value in ages.values()
    ):
        raise ValueError(
            "reference ages must be non-negative required-action entries"
        )
    age_pass = (
        maximum_reference_age_experiments >= 0
        and observed_set.issubset(ages)
        and max(ages.values(), default=0) <= maximum_reference_age_experiments
    )
    predictive_checks = {
        "held_out_old_world_predictive_adequacy": predictive_pass,
        "reference_age_within_frozen_limit": age_pass,
    }
    structural_pass = all(structural_checks.values())
    passed = structural_pass and all(predictive_checks.values())
    return {
        "schema_version": "chemworld-reference-sufficiency-certificate-0.3",
        "status": "passed" if passed else "failed",
        "pass": passed,
        "scope": "universal_all_declared_candidate_families_without_truth_conditioning",
        "minimum_reference_experiments": minimum_reference_experiments,
        "actual_reference_experiments": len(observed),
        "reference_policy_action_ids": reference_policy_actions,
        "observed_reference_action_ids": list(dict.fromkeys(observed)),
        "structural_sufficiency": {
            "pass": structural_pass,
            "certificate_basis": "relation_closure_not_recipe_id_closure",
            "relation_coverage": relation_coverage,
            "checks": structural_checks,
            "diagnostics": diagnostics,
        },
        "predictive_sufficiency": {
            **dict(predictive_adequacy),
            "pass": predictive_pass and age_pass,
            "maximum_reference_age_experiments": (
                maximum_reference_age_experiments
            ),
            "observed_reference_ages": ages,
            "checks": predictive_checks,
        },
    }


def categorical_entropy(distribution: Mapping[str, float], *, normalized: bool = False) -> float:
    """Return categorical entropy in nats, optionally divided by log(cardinality)."""

    values = normalized_distribution(distribution)
    entropy = -sum(value * math.log(value) for value in values.values() if value > 0.0)
    if normalized:
        entropy /= math.log(len(values))
    return entropy


def kl_divergence(
    following: Mapping[str, float],
    current: Mapping[str, float],
    *,
    floor: float = 1e-12,
) -> float:
    """Return KL(following || current) with a documented numerical floor."""

    left = normalized_distribution(following)
    right = normalized_distribution(current)
    if set(left) != set(right):
        raise ValueError("distributions must contain identical candidate IDs")
    return sum(
        value * math.log(value / max(right[key], floor))
        for key, value in left.items()
        if value > 0.0
    )


def js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return Jensen-Shannon divergence in nats."""

    first = normalized_distribution(left)
    second = normalized_distribution(right)
    if set(first) != set(second):
        raise ValueError("distributions must contain identical candidate IDs")
    midpoint = {key: 0.5 * (first[key] + second[key]) for key in first}
    return 0.5 * kl_divergence(first, midpoint) + 0.5 * kl_divergence(second, midpoint)


def multiclass_brier(distribution: Mapping[str, float], truth: str) -> float:
    """Return the multiclass Brier score for a declared distribution."""

    values = normalized_distribution(distribution)
    if truth not in values:
        raise ValueError("truth must be one of the declared candidates")
    return sum((value - (1.0 if key == truth else 0.0)) ** 2 for key, value in values.items())


def change_detection_summary(
    *,
    changed: Sequence[bool],
    probabilities: Sequence[float],
    detection_delays: Sequence[int | None],
    detections: Sequence[bool] | None = None,
    threshold: float = 0.5,
    right_censor_time: int | None = None,
    bootstrap_seed: int = 0,
    bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    """Score time-indexed change events and checkpoint probabilities.

    ``probabilities`` are the frozen-checkpoint scores used for AUROC and Brier.
    ``detections`` may separately declare whether the threshold was crossed at
    any time up to that checkpoint.  This distinction prevents a transient
    false alarm from disappearing merely because the terminal posterior later
    falls below the decision threshold.
    """

    aligned = len(changed) == len(probabilities) == len(detection_delays)
    if not changed or not aligned:
        raise ValueError("change labels, probabilities, and delays must be non-empty and aligned")
    if detections is not None and len(detections) != len(changed):
        raise ValueError("detection events must align with change labels")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if right_censor_time is not None and right_censor_time <= 0:
        raise ValueError("right-censor time must be positive")
    labels = [bool(item) for item in changed]
    scores = [float(item) for item in probabilities]
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in scores):
        raise ValueError("change probabilities must be finite and in [0, 1]")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("change detection requires changed and no-change campaigns")
    predicted = (
        [bool(item) for item in detections]
        if detections is not None
        else [score >= threshold for score in scores]
    )
    true_positive = sum(
        label and estimate for label, estimate in zip(labels, predicted, strict=True)
    )
    false_positive = sum(
        not label and estimate for label, estimate in zip(labels, predicted, strict=True)
    )
    observed_delays = [
        int(delay)
        for label, estimate, delay in zip(labels, predicted, detection_delays, strict=True)
        if label and estimate and delay is not None
    ]
    if any(delay <= 0 for delay in observed_delays):
        raise ValueError("observed detection delays must be positive")
    if right_censor_time is None:
        right_censor_time = max(observed_delays, default=None)
    changed_time_to_event = [
        {
            "duration": (
                int(delay)
                if estimate and delay is not None
                else right_censor_time
            ),
            "event_observed": bool(estimate and delay is not None),
        }
        for label, estimate, delay in zip(labels, predicted, detection_delays, strict=True)
        if label
    ]
    sensitivity_interval = wilson_interval(true_positive, positives)
    false_positive_interval = wilson_interval(false_positive, negatives)
    auroc = _binary_auroc(labels, scores)
    return {
        "campaign_count": len(labels),
        "changed_count": positives,
        "no_change_twin_count": negatives,
        "threshold": threshold,
        "sensitivity": true_positive / positives,
        "sensitivity_interval": list(sensitivity_interval),
        "false_positive_rate": false_positive / negatives,
        "false_positive_rate_interval": list(false_positive_interval),
        "auroc": auroc,
        "auroc_interval": list(
            binary_auroc_bootstrap_interval(
                changed=labels,
                probabilities=scores,
                seed=bootstrap_seed,
                draws=bootstrap_draws,
            )
        ),
        "brier_score": sum(
            (score - (1.0 if label else 0.0)) ** 2
            for label, score in zip(labels, scores, strict=True)
        )
        / len(labels),
        "detection_delay_event_definition": (
            "first checkpoint-relative experiment k with p(change)>=threshold"
        ),
        "detection_delay_handling": "right_censored",
        "right_censor_time": right_censor_time,
        "observed_event_mean_detection_delay": _mean(observed_delays),
        "detected_changed_count": len(observed_delays),
        "right_censored_changed_count": positives - len(observed_delays),
        "changed_time_to_event": changed_time_to_event,
    }


def binary_auroc_bootstrap_interval(
    *,
    changed: Sequence[bool],
    probabilities: Sequence[float],
    seed: int,
    draws: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic stratified percentile interval for binary AUROC."""

    labels = np.asarray([bool(item) for item in changed], dtype=bool)
    scores = np.asarray([float(item) for item in probabilities], dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0:
        raise ValueError("AUROC bootstrap labels and scores must be aligned")
    if not labels.any() or labels.all():
        raise ValueError("AUROC bootstrap requires both classes")
    if draws <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("AUROC bootstrap draws and confidence are invalid")
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(draws, dtype=float)
    for index in range(draws):
        selected_positive = rng.choice(positive, size=positive.size, replace=True)
        selected_negative = rng.choice(negative, size=negative.size, replace=True)
        selected = np.concatenate((selected_positive, selected_negative))
        estimates[index] = _binary_auroc(
            labels[selected].tolist(),
            scores[selected].tolist(),
        )
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )


def declared_distribution_update(
    current: Mapping[str, float],
    following: Mapping[str, float],
    *,
    truth: str,
) -> dict[str, float]:
    """Describe an adjacent public-distribution update without Bayesian claims."""

    first = normalized_distribution(current)
    second = normalized_distribution(following)
    if truth not in first or truth not in second:
        raise ValueError("truth must occur in both declared distributions")
    floor = 1e-12
    return {
        "declared_distribution_js_shift": js_divergence(first, second),
        "declared_distribution_kl_shift": kl_divergence(second, first),
        "declared_normalized_entropy_change": (
            categorical_entropy(second, normalized=True)
            - categorical_entropy(first, normalized=True)
        ),
        "truth_log_probability_change": math.log(max(second[truth], floor))
        - math.log(max(first[truth], floor)),
        "brier_improvement": multiclass_brier(first, truth) - multiclass_brier(second, truth),
    }


def feedback_effect_summary(
    *,
    within_condition_distances: Sequence[float],
    between_condition_distances: Sequence[float],
) -> dict[str, float | int | None]:
    """Separate feedback-condition differences from provider repeat noise."""

    within = _finite_values(within_condition_distances)
    between = _finite_values(between_condition_distances)
    within_mean = _mean(within)
    between_mean = _mean(between)
    return {
        "within_repeat_count": len(within),
        "between_condition_count": len(between),
        "mean_within_provider_variation": within_mean,
        "mean_between_feedback_variation": between_mean,
        "net_feedback_effect": (
            between_mean - within_mean
            if within_mean is not None and between_mean is not None
            else None
        ),
    }


def operation_aware_action_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compare actions without treating absent operation parameters as zero."""

    left_operation = str(left.get("operation") or "")
    right_operation = str(right.get("operation") or "")
    if left_operation != right_operation:
        return {
            "same_operation": False,
            "operation_distance": 1.0,
            "parameter_distance": None,
            "compared_parameters": [],
        }
    bounds = parameter_bounds or {}
    shared = sorted((set(left) & set(right)) - {"operation", "instrument"})
    squared: list[float] = []
    compared: list[str] = []
    for key in shared:
        first = left[key]
        second = right[key]
        if isinstance(first, bool) or isinstance(second, bool):
            continue
        if isinstance(first, int | float) and isinstance(second, int | float):
            low, high = bounds.get(key, (0.0, max(abs(float(first)), abs(float(second)), 1.0)))
            scale = max(float(high) - float(low), 1e-12)
            squared.append(((float(first) - float(second)) / scale) ** 2)
            compared.append(key)
        elif first != second:
            squared.append(1.0)
            compared.append(key)
        else:
            squared.append(0.0)
            compared.append(key)
    return {
        "same_operation": True,
        "operation_distance": 0.0,
        "parameter_distance": (math.sqrt(sum(squared) / len(squared)) if squared else 0.0),
        "compared_parameters": compared,
    }


def recovery_decomposition(
    *,
    iid_replay_iid_world: float,
    iid_replay_shifted_world: float,
    frozen_policy_shifted_world: float,
    adaptive_policy_shifted_world: float,
    oracle_shifted_world: float,
) -> dict[str, float | None]:
    """Separate world, behavioural-adaptation, and normalized-recovery effects."""

    world_effect = iid_replay_shifted_world - iid_replay_iid_world
    adaptation_gain = adaptive_policy_shifted_world - frozen_policy_shifted_world
    denominator = oracle_shifted_world - frozen_policy_shifted_world
    normalized_recovery = adaptation_gain / denominator if abs(denominator) > 1e-12 else None
    return {
        "world_effect": world_effect,
        "adaptation_gain": adaptation_gain,
        "normalized_recovery": normalized_recovery,
        "normalization_denominator": denominator,
    }


def campaign_autonomy_status(
    *,
    current_experiment_autonomous: bool,
    assisted_history: bool,
    campaign_complete: bool,
) -> AutonomyStatus:
    """Return the frozen three-level campaign autonomy classification."""

    if not campaign_complete:
        return "incomplete_campaign"
    if not current_experiment_autonomous:
        return "assisted_campaign"
    if assisted_history:
        return "autonomous_current_experiment_with_assisted_history"
    return "fully_autonomous_campaign"


@dataclass(frozen=True)
class GaussianCandidatePredictive:
    """Diagonal-Gaussian approximation for one candidate and public action."""

    candidate_id: str
    action_id: str
    mean: np.ndarray
    variance: np.ndarray
    likelihood_scale: float = 1.0

    def log_likelihood(self, observation: Sequence[float]) -> float:
        values = np.asarray(observation, dtype=float).reshape(-1)
        if values.shape != self.mean.shape:
            raise ValueError("observation dimension does not match predictive model")
        if not math.isfinite(self.likelihood_scale) or self.likelihood_scale <= 0.0:
            raise ValueError("predictive likelihood scale must be finite and positive")
        variance = np.maximum(self.variance, 1e-12)
        return float(
            -0.5
            * self.likelihood_scale
            * np.sum(np.log(2.0 * math.pi * variance) + ((values - self.mean) ** 2) / variance)
        )


class GaussianMechanismOracle:
    """Executable Monte-Carlo diagnosis oracle over candidate public generators.

    ``sample_public_observation`` must not reveal the true candidate.  It receives a
    candidate ID, a public action ID, a nuisance/noise seed, and should return the
    public numeric feature vector that an Agent could observe.  A caller may close over
    cloned candidate worlds and the current public history.
    """

    def __init__(
        self,
        *,
        candidate_ids: Sequence[str],
        action_ids: Sequence[str],
        sample_public_observation: Callable[[str, str, int], Sequence[float]],
        samples_per_candidate: int = 128,
        variance_floor: float = 1e-6,
        likelihood_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        if len(set(candidate_ids)) < 2:
            raise ValueError("oracle requires at least two candidate IDs")
        if not action_ids:
            raise ValueError("oracle requires at least one public action")
        if samples_per_candidate < 4:
            raise ValueError("samples_per_candidate must be at least four")
        self.candidate_ids = tuple(str(item) for item in candidate_ids)
        self.action_ids = tuple(str(item) for item in action_ids)
        self._sample = sample_public_observation
        self.samples_per_candidate = int(samples_per_candidate)
        self.variance_floor = float(variance_floor)
        self.likelihood_scale = float(likelihood_scale)
        if not math.isfinite(self.likelihood_scale) or self.likelihood_scale <= 0.0:
            raise ValueError("oracle likelihood scale must be finite and positive")
        self.seed = int(seed)
        uniform = 1.0 / len(self.candidate_ids)
        self.posterior = dict.fromkeys(self.candidate_ids, uniform)
        self._predictives: dict[tuple[str, str], GaussianCandidatePredictive] = {}

    def fit_predictives(self) -> None:
        """Fit held-out-ready predictive approximations for every action/candidate."""

        rng = np.random.default_rng(self.seed)
        sample_groups: dict[tuple[str, str], list[Sequence[float]]] = {}
        for action_id in self.action_ids:
            for candidate_id in self.candidate_ids:
                seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.samples_per_candidate)
                sample_groups[(candidate_id, action_id)] = [
                    self._sample(candidate_id, action_id, int(sample_seed)) for sample_seed in seeds
                ]
        self.fit_predictives_from_samples(sample_groups)

    def fit_predictives_from_samples(
        self,
        samples: Mapping[tuple[str, str], Sequence[Sequence[float]]],
    ) -> None:
        """Fit predictives from externally scheduled candidate-generator samples."""

        expected = {
            (candidate_id, action_id)
            for action_id in self.action_ids
            for candidate_id in self.candidate_ids
        }
        if set(samples) != expected:
            raise ValueError("predictive samples must exactly cover candidate/action pairs")
        fitted: dict[tuple[str, str], GaussianCandidatePredictive] = {}
        for candidate_id, action_id in sorted(samples):
            values = np.asarray(samples[(candidate_id, action_id)], dtype=float)
            if values.ndim != 2 or values.shape[0] < 4:
                raise ValueError("each candidate/action predictive requires four samples")
            fitted[(candidate_id, action_id)] = GaussianCandidatePredictive(
                candidate_id=candidate_id,
                action_id=action_id,
                mean=np.mean(values, axis=0),
                variance=np.maximum(np.var(values, axis=0, ddof=1), self.variance_floor),
                likelihood_scale=self.likelihood_scale,
            )
        self._predictives = fitted

    def export_predictives(self) -> dict[str, Any]:
        """Return a process-safe representation of fitted predictive models."""

        self._require_fitted()
        return {
            f"{candidate_id}\u241f{action_id}": {
                "candidate_id": candidate_id,
                "action_id": action_id,
                "mean": predictive.mean.tolist(),
                "variance": predictive.variance.tolist(),
                "likelihood_scale": predictive.likelihood_scale,
            }
            for (candidate_id, action_id), predictive in self._predictives.items()
        }

    def load_predictives(self, payload: Mapping[str, Mapping[str, Any]]) -> None:
        """Load a representation produced by :meth:`export_predictives`."""

        fitted: dict[tuple[str, str], GaussianCandidatePredictive] = {}
        for item in payload.values():
            candidate_id = str(item["candidate_id"])
            action_id = str(item["action_id"])
            fitted[(candidate_id, action_id)] = GaussianCandidatePredictive(
                candidate_id=candidate_id,
                action_id=action_id,
                mean=np.asarray(item["mean"], dtype=float),
                variance=np.asarray(item["variance"], dtype=float),
                likelihood_scale=float(item.get("likelihood_scale", self.likelihood_scale)),
            )
        expected = {
            (candidate_id, action_id)
            for action_id in self.action_ids
            for candidate_id in self.candidate_ids
        }
        if set(fitted) != expected:
            raise ValueError("loaded predictives must exactly cover candidate/action pairs")
        dimensions = {predictive.mean.shape for predictive in fitted.values()}
        if len(dimensions) != 1 or any(
            predictive.mean.shape != predictive.variance.shape for predictive in fitted.values()
        ):
            raise ValueError("loaded predictive dimensions are inconsistent")
        self._predictives = fitted

    def update(self, *, action_id: str, observation: Sequence[float]) -> dict[str, float]:
        """Update the calibrated-reference posterior for one public observation."""

        self._require_fitted()
        if action_id not in self.action_ids:
            raise ValueError("unknown public action ID")
        log_weights = {
            candidate: math.log(max(self.posterior[candidate], 1e-300))
            + self._predictives[(candidate, action_id)].log_likelihood(observation)
            for candidate in self.candidate_ids
        }
        maximum = max(log_weights.values())
        weights = {key: math.exp(value - maximum) for key, value in log_weights.items()}
        self.posterior = normalized_distribution(weights)
        return dict(self.posterior)

    def expected_information_by_action(self, *, draws: int = 256) -> dict[str, float]:
        """Estimate mutual information in nats under the current posterior."""

        self._require_fitted()
        if draws <= 0:
            raise ValueError("draws must be positive")
        rng = np.random.default_rng(self.seed + 1)
        prior_entropy = categorical_entropy(self.posterior)
        candidates = np.asarray(self.candidate_ids)
        probabilities = np.asarray([self.posterior[item] for item in self.candidate_ids])
        result: dict[str, float] = {}
        for action_id in self.action_ids:
            posterior_entropies: list[float] = []
            for _ in range(draws):
                candidate = str(rng.choice(candidates, p=probabilities))
                predictive = self._predictives[(candidate, action_id)]
                observation = rng.normal(predictive.mean, np.sqrt(predictive.variance))
                posterior_entropies.append(
                    categorical_entropy(
                        self._posterior_for_observation(action_id, observation.tolist())
                    )
                )
            result[action_id] = prior_entropy - float(np.mean(posterior_entropies))
        return result

    def select_action(self, *, draws: int = 256) -> str:
        """Select the public action with maximum estimated information value."""

        values = self.expected_information_by_action(draws=draws)
        return max(values, key=values.__getitem__)

    def reset_posterior(self, prior: Mapping[str, float] | None = None) -> None:
        """Reset state between independent world-seed trials."""

        if prior is None:
            uniform = 1.0 / len(self.candidate_ids)
            self.posterior = dict.fromkeys(self.candidate_ids, uniform)
            return
        values = normalized_distribution(prior)
        if set(values) != set(self.candidate_ids):
            raise ValueError("oracle prior must contain exactly the candidate IDs")
        self.posterior = values

    def _posterior_for_observation(
        self,
        action_id: str,
        observation: Sequence[float],
    ) -> dict[str, float]:
        log_weights = {
            candidate: math.log(max(self.posterior[candidate], 1e-300))
            + self._predictives[(candidate, action_id)].log_likelihood(observation)
            for candidate in self.candidate_ids
        }
        maximum = max(log_weights.values())
        return normalized_distribution(
            {key: math.exp(value - maximum) for key, value in log_weights.items()}
        )

    def _require_fitted(self) -> None:
        if not self._predictives:
            raise RuntimeError("fit_predictives must be called before diagnosis")


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the Wilson binomial confidence interval."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes and total must define a non-empty binomial sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    )
    return max(center - radius, 0.0), min(center + radius, 1.0)


def identifiability_certificate(
    *,
    truths: Sequence[str],
    predictions: Sequence[str],
    candidate_ids: Sequence[str],
    overall_lower_bound_threshold: float = 0.80,
    family_recall_lower_bound_threshold: float = 0.70,
) -> dict[str, Any]:
    """Build the frozen family-level active/decoder certificate."""

    if len(truths) != len(predictions) or not truths:
        raise ValueError("truths and predictions must be non-empty and aligned")
    candidates = tuple(str(item) for item in candidate_ids)
    if set(truths) - set(candidates) or set(predictions) - set(candidates):
        raise ValueError("truths and predictions must use declared candidate IDs")
    correct = sum(left == right for left, right in zip(truths, predictions, strict=True))
    overall_interval = wilson_interval(correct, len(truths))
    family: dict[str, Any] = {}
    for candidate in candidates:
        indices = [index for index, truth in enumerate(truths) if truth == candidate]
        if not indices:
            family[candidate] = {"count": 0, "recall": None, "recall_interval": None}
            continue
        hits = sum(predictions[index] == candidate for index in indices)
        interval = wilson_interval(hits, len(indices))
        family[candidate] = {
            "count": len(indices),
            "recall": hits / len(indices),
            "recall_interval": list(interval),
        }
    family_pass = all(
        item["recall_interval"] is not None
        and float(item["recall_interval"][0]) >= family_recall_lower_bound_threshold
        for item in family.values()
    )
    return {
        "sample_count": len(truths),
        "top1_accuracy": correct / len(truths),
        "top1_accuracy_interval": list(overall_interval),
        "family_recall": family,
        "overall_lower_bound_threshold": overall_lower_bound_threshold,
        "family_recall_lower_bound_threshold": family_recall_lower_bound_threshold,
        "overall_pass": overall_interval[0] >= overall_lower_bound_threshold,
        "family_pass": family_pass,
        "gate_pass": (overall_interval[0] >= overall_lower_bound_threshold and family_pass),
    }


def fixed_trajectory_decode(
    oracle: GaussianMechanismOracle,
    *,
    action_ids: Sequence[str],
    observations: Sequence[Sequence[float]],
    prior: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Decode an Agent-generated public trajectory without selecting its actions."""

    if len(action_ids) != len(observations) or not action_ids:
        raise ValueError("actions and observations must be non-empty and aligned")
    oracle.reset_posterior(prior)
    updates: list[dict[str, float]] = []
    for action_id, observation in zip(action_ids, observations, strict=True):
        updates.append(oracle.update(action_id=action_id, observation=observation))
    return {
        "mode": "fixed_trajectory_decoder",
        "action_count": len(action_ids),
        "posterior": dict(oracle.posterior),
        "prediction": max(oracle.posterior, key=oracle.posterior.__getitem__),
        "updates": updates,
    }


def active_oracle_diagnosis(
    oracle: GaussianMechanismOracle,
    *,
    observe: Callable[[str, int], Sequence[float]],
    budget: int,
    information_draws: int = 256,
    prior: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run one budget-matched active diagnosis trial on public observations."""

    if budget <= 0:
        raise ValueError("diagnosis budget must be positive")
    oracle.reset_posterior(prior)
    actions: list[str] = []
    updates: list[dict[str, float]] = []
    for index in range(budget):
        action_id = oracle.select_action(draws=information_draws)
        actions.append(action_id)
        observation = observe(action_id, index)
        updates.append(oracle.update(action_id=action_id, observation=observation))
    return {
        "mode": "active_budget_matched_diagnosis_oracle",
        "budget": budget,
        "actions": actions,
        "posterior": dict(oracle.posterior),
        "prediction": max(oracle.posterior, key=oracle.posterior.__getitem__),
        "updates": updates,
    }


def validate_mechanism_adaptation_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Return protocol-freeze errors for the six-Gate mechanism benchmark."""

    errors: list[str] = []
    gates = protocol.get("gates")
    if not isinstance(gates, Mapping):
        return ["gates must be an object containing gate_0 through gate_e"]
    for gate_id in REQUIRED_GATE_IDS:
        gate = gates.get(gate_id)
        if not isinstance(gate, Mapping):
            errors.append(f"missing {gate_id}")
        elif not gate.get("pass_rule"):
            errors.append(f"{gate_id} must freeze a pass_rule")

    schema_version = protocol.get("schema_version")
    design = protocol.get("design")
    if not isinstance(design, Mapping):
        errors.append("design must be an object")
    else:
        if schema_version == "chemworld-mechanism-adaptation-protocol-0.3.0":
            legacy_change_fields = {
                "pre_change_experiments",
                "change_after_experiments",
                "post_change_checkpoints",
                "total_experiment_horizon",
            } & set(design)
            if legacy_change_fields:
                errors.append(
                    "v0.3 change timing belongs to evaluation_tracks, not design: "
                    + ", ".join(sorted(legacy_change_fields))
                )
        else:
            if design.get("post_change_checkpoints") != [1, 2, 4, 8]:
                errors.append("post_change_checkpoints must equal [1, 2, 4, 8]")
            changepoints = design.get("change_after_experiments")
            if not isinstance(changepoints, list) or "never" not in changepoints:
                errors.append("change_after_experiments must include the no-change twin 'never'")
            if int(design.get("total_experiment_horizon", 0) or 0) < 8:
                errors.append("total_experiment_horizon must be at least eight")
        repeat_count = int(design.get("provider_repeats_per_paired_cell", 0) or 0)
        order_seeds = design.get("candidate_order_seeds")
        if not isinstance(order_seeds, list) or len(order_seeds) != repeat_count:
            errors.append("candidate_order_seeds must match provider repeat count")

    if schema_version == "chemworld-mechanism-adaptation-protocol-0.3.0":
        tracks = protocol.get("evaluation_tracks")
        if not isinstance(tracks, Mapping):
            errors.append("v0.3 requires evaluation_tracks")
        elif set(tracks) != set(REQUIRED_EVALUATION_TRACK_IDS):
            errors.append(
                "evaluation_tracks must exactly declare "
                + ", ".join(REQUIRED_EVALUATION_TRACK_IDS)
            )
        else:
            static = tracks["static_world_identification"]
            if (
                not isinstance(static, Mapping)
                or static.get("world_initialization") != "candidate_world_from_start"
                or static.get("change_detection_claim_allowed") is not False
            ):
                errors.append(
                    "static_world_identification must identify the initialized current "
                    "world without a change-detection claim"
                )

            calibrated = tracks[PRIMARY_CHANGE_TRACK_ID]
            if not isinstance(calibrated, Mapping):
                errors.append("calibrated_online_change must be an object")
            else:
                minimum_prefix = int(
                    calibrated.get("minimum_stable_prefix_experiments", 0) or 0
                )
                checkpoints = calibrated.get("post_change_checkpoints")
                changepoints = calibrated.get("change_after_experiments")
                numeric_changepoints = (
                    [int(item) for item in changepoints if item != "never"]
                    if isinstance(changepoints, list)
                    else []
                )
                if minimum_prefix <= 0:
                    errors.append(
                        "calibrated_online_change requires a positive stable prefix"
                    )
                if checkpoints != [1, 2, 4, 8]:
                    errors.append(
                        "calibrated_online_change post_change_checkpoints must equal "
                        "[1, 2, 4, 8]"
                    )
                if (
                    not isinstance(changepoints, list)
                    or "never" not in changepoints
                    or not numeric_changepoints
                ):
                    errors.append(
                        "calibrated_online_change must include hidden change times and never"
                    )
                if calibrated.get("truth_change_time_support") != changepoints:
                    errors.append(
                        "calibrated truth_change_time_support must exactly match "
                        "change_after_experiments"
                    )
                if (
                    calibrated.get("changepoint_semantics")
                    != "tau_is_completed_old_world_experiment_count; tau=6 means "
                    "experiments 1-6 use the old world and experiment 7 is the first "
                    "experiment eligible to use the changed world"
                ):
                    errors.append(
                        "calibrated changepoint semantics must freeze the completed "
                        "old-world experiment-count convention"
                    )
                elif min(numeric_changepoints) < minimum_prefix:
                    errors.append(
                        "calibrated change times cannot precede the stable reference prefix"
                    )
                horizon = int(calibrated.get("total_experiment_horizon", 0) or 0)
                maximum_checkpoint = max(checkpoints or [0])
                if (
                    numeric_changepoints
                    and horizon
                    < max(numeric_changepoints) + maximum_checkpoint
                ):
                    errors.append(
                        "calibrated horizon must cover the latest change plus final checkpoint"
                    )
                sufficiency = calibrated.get("reference_sufficiency")
                if not isinstance(sufficiency, Mapping):
                    errors.append(
                        "calibrated_online_change requires reference_sufficiency"
                    )
                else:
                    if (
                        sufficiency.get("basis")
                        != "universal_structural_relation_coverage_and_held_out_predictive_adequacy"
                    ):
                        errors.append(
                            "reference sufficiency must combine universal structural "
                            "coverage and held-out predictive adequacy"
                        )
                    if (
                        sufficiency.get("scope")
                        != "universal_all_declared_candidate_families_without_truth_conditioning"
                    ):
                        errors.append(
                            "reference sufficiency must be universal and independent "
                            "of the realized hidden family"
                        )
                    if (
                        sufficiency.get("relation_closure_controls_certificate")
                        is not True
                    ):
                        errors.append(
                            "reference sufficiency must be controlled by relation "
                            "closure rather than a canonical recipe-ID checklist"
                        )
                    if sufficiency.get("all_declared_relations_observed") is not True:
                        errors.append(
                            "reference sufficiency must require every declared relation"
                        )
                    if (
                        sufficiency.get("held_out_old_world_predictive_check_required")
                        is not True
                    ):
                        errors.append(
                            "reference sufficiency must require held-out old-world "
                            "predictive adequacy"
                        )
                timing_contract = calibrated.get("agent_visible_timing_contract")
                if not isinstance(timing_contract, Mapping):
                    errors.append(
                        "calibrated online change requires an Agent-visible timing contract"
                    )
                else:
                    hidden_timing = set(timing_contract.get("hidden", []))
                    required_hidden = {
                        "minimum_stable_prefix_experiments",
                        "change_after_experiments",
                        "truth_change_time",
                        "evaluator_pseudo_checkpoint",
                        "reference_sufficiency_certificate",
                        "post_change_checkpoint",
                    }
                    if not required_hidden.issubset(hidden_timing):
                        errors.append(
                            "Agent-visible timing contract leaks or omits private "
                            "calibrated timing fields"
                        )

            stress = tracks["uncalibrated_nonstationarity"]
            if not isinstance(stress, Mapping):
                errors.append("uncalibrated_nonstationarity must be an object")
            else:
                stress_times = stress.get("change_after_experiments")
                if (
                    not isinstance(stress_times, list)
                    or 0 not in stress_times
                    or stress.get("controls_publication_gate") is not False
                    or stress.get("ordinary_change_detection_claim_allowed") is not False
                ):
                    errors.append(
                        "uncalibrated_nonstationarity must retain time zero as a "
                        "non-gating current-world/joint-inference stress case"
                    )

            recovery = tracks["adaptive_recovery"]
            if (
                not isinstance(recovery, Mapping)
                or recovery.get("source_change_track") != PRIMARY_CHANGE_TRACK_ID
            ):
                errors.append(
                    "adaptive_recovery must consume the calibrated online change track"
                )

    reporting = protocol.get("reporting")
    if not isinstance(reporting, Mapping):
        errors.append("reporting must be an object")
    else:
        if reporting.get("statistical_unit") != "physical_cell_cluster_with_provider_repeat_nested":
            errors.append("statistical_unit must cluster physical cells and nest provider repeats")
        if reporting.get("provider_repeats") != "nested_technical_replicates":
            errors.append("provider repeats must be nested technical replicates")

    data_contract = protocol.get("data_contract")
    required_outcomes = {
        "environment_outcome",
        "agent_visible_observation",
        "evaluation_outcome",
    }
    if not isinstance(data_contract, Mapping):
        errors.append("data_contract must be an object")
    elif set(data_contract.get("outcome_layers", [])) != required_outcomes:
        errors.append("data_contract must separate all three outcome layers")

    diagnosis = protocol.get("diagnosis_contract")
    if not isinstance(diagnosis, Mapping):
        errors.append("diagnosis_contract must be an object")
    else:
        if diagnosis.get("change_probability") != "derived_as_1_minus_q_no_change":
            errors.append("change probability must be derived from q(no_change)")
        if not diagnosis.get("candidate_definitions_required"):
            errors.append("operational candidate definitions are required")
        if not diagnosis.get("candidate_order_randomized"):
            errors.append("candidate order randomization is required")

    task_contracts = protocol.get("task_mechanism_contracts")
    expected_tasks = set(design.get("tasks", [])) if isinstance(design, Mapping) else set()
    if not isinstance(task_contracts, Mapping) or set(task_contracts) != expected_tasks:
        errors.append("task_mechanism_contracts must exactly cover design tasks")
    elif isinstance(diagnosis, Mapping):
        definitions = diagnosis.get("candidate_definitions", {})
        for task_id, raw_contract in task_contracts.items():
            if not isinstance(raw_contract, Mapping):
                errors.append(f"task mechanism contract must be an object: {task_id}")
                continue
            candidates = raw_contract.get("candidate_ids")
            interventions = raw_contract.get("interventions")
            diagnostic_relations = raw_contract.get("diagnostic_relations", {})
            if not isinstance(candidates, list) or "no_change" not in candidates:
                errors.append(f"task candidates must include no_change: {task_id}")
                continue
            if not isinstance(interventions, Mapping) or set(interventions) != set(candidates):
                errors.append(f"task interventions must exactly cover candidates: {task_id}")
            if set(candidates) - set(definitions):
                errors.append(f"task candidates lack public definitions: {task_id}")
            if isinstance(interventions, Mapping) and interventions.get("no_change") != []:
                errors.append(f"no_change intervention must be empty: {task_id}")
            if isinstance(interventions, Mapping):
                for candidate_id in candidates:
                    if candidate_id == "no_change":
                        continue
                    candidate_interventions = interventions.get(candidate_id)
                    if (
                        not isinstance(candidate_interventions, list)
                        or len(candidate_interventions) != 1
                    ):
                        errors.append(
                            "changed candidates must declare exactly one intervention: "
                            f"{task_id}/{candidate_id}"
                        )
                        continue
                    intervention = candidate_interventions[0]
                    if not isinstance(intervention, Mapping):
                        errors.append(f"intervention must be an object: {task_id}/{candidate_id}")
                        continue
                    if intervention.get("kind") == "material_law_counterfactual":
                        material_field = intervention.get("material_field")
                        if material_field not in {
                            "catalyst",
                            "solvent",
                            "electrolyte_profile",
                        }:
                            errors.append(
                                "unsupported material counterfactual field: "
                                f"{task_id}/{candidate_id}/{material_field}"
                            )
                    elif schema_version == "chemworld-mechanism-adaptation-protocol-0.3.0":
                        relation = (
                            diagnostic_relations.get(candidate_id)
                            if isinstance(diagnostic_relations, Mapping)
                            else None
                        )
                        if not isinstance(relation, Mapping):
                            errors.append(
                                "v0.3 mechanism interventions require a diagnostic "
                                f"relation: {task_id}/{candidate_id}"
                            )
                        else:
                            varied_fields = relation.get("varied_fields")
                            if (
                                relation.get("relation_graph_version")
                                != "chemworld-diagnostic-relation-graph-0.1"
                                or not isinstance(varied_fields, list)
                                or not varied_fields
                                or int(
                                    relation.get("minimum_distinct_actions", 0)
                                    or 0
                                )
                                < 2
                                or relation.get("controlled_background")
                                != "all_remaining_recipe_operations_fields_and_order"
                                or not isinstance(
                                    relation.get("observable_channels"),
                                    list,
                                )
                                or not relation.get("observable_channels")
                                or not isinstance(
                                    relation.get("candidate_signatures"),
                                    Mapping,
                                )
                            ):
                                errors.append(
                                    "invalid diagnostic relation: "
                                    f"{task_id}/{candidate_id}"
                                )

    if protocol.get("schema_version") in {
        "chemworld-mechanism-adaptation-protocol-0.2.1",
        "chemworld-mechanism-adaptation-protocol-0.3.0",
    }:
        alignment = protocol.get("intervention_action_alignment")
        if not isinstance(alignment, Mapping) or not alignment.get("required"):
            errors.append("v0.2.1+ requires intervention_action_alignment")
        elif not alignment.get("moved_indices_must_be_covered_by_gate_a_action_library"):
            errors.append("v0.2.1+ must require Gate A coverage of moved material indices")

    return errors


def build_paired_campaign_matrix(
    protocol: Mapping[str, Any],
    *,
    world_seeds: Sequence[int] | None = None,
    track_id: str | None = None,
) -> list[dict[str, Any]]:
    """Expand the frozen changed/no-change twin confirmatory matrix."""

    errors = validate_mechanism_adaptation_protocol(protocol)
    if errors:
        raise ValueError("invalid mechanism-adaptation protocol: " + "; ".join(errors))
    design = protocol["design"]
    task_contracts = protocol["task_mechanism_contracts"]
    seeds = list(world_seeds or design["public_development_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("world seeds must be non-empty and unique")
    if protocol.get("schema_version") == "chemworld-mechanism-adaptation-protocol-0.3.0":
        selected_track_id = track_id or PRIMARY_CHANGE_TRACK_ID
        track = evaluation_track(protocol, selected_track_id)
        if track.get("kind") not in {
            "calibrated_hidden_change",
            "uncalibrated_hidden_nonstationarity",
        }:
            raise ValueError(
                "paired campaign matrices require a calibrated or uncalibrated "
                "change track"
            )
        timing = track
    else:
        if track_id is not None:
            raise ValueError("legacy protocols do not declare evaluation tracks")
        selected_track_id = "legacy_paired_change"
        timing = design
    change_times = [
        int(item) for item in timing["change_after_experiments"] if item != "never"
    ]
    label_modes = list(protocol["diagnosis_contract"]["candidate_label_modes"])
    order_seeds = [int(item) for item in design["candidate_order_seeds"]]
    rows: list[dict[str, Any]] = []
    for task_id, raw_contract in task_contracts.items():
        candidates = list(raw_contract["candidate_ids"])
        interventions = raw_contract["interventions"]
        for truth_id in candidates:
            if truth_id == "no_change":
                continue
            for change_time in change_times:
                for label_mode in label_modes:
                    for world_seed in seeds:
                        for repeat_id, order_seed in enumerate(order_seeds):
                            physical_cell_payload = {
                                "protocol_id": protocol["protocol_id"],
                                "evaluation_track_id": selected_track_id,
                                "task_id": task_id,
                                "phase_reset_after_experiment": change_time,
                                "candidate_label_mode": label_mode,
                                "world_seed": int(world_seed),
                            }
                            physical_cell_id = hashlib.sha256(
                                json.dumps(
                                    physical_cell_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()[:20]
                            pair_payload = {
                                "protocol_id": protocol["protocol_id"],
                                "evaluation_track_id": selected_track_id,
                                "task_id": task_id,
                                "changed_truth_id": truth_id,
                                "phase_reset_after_experiment": change_time,
                                "candidate_label_mode": label_mode,
                                "world_seed": int(world_seed),
                                "provider_repeat_id": repeat_id,
                                "candidate_order_seed": order_seed,
                            }
                            pair_id = hashlib.sha256(
                                json.dumps(
                                    pair_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()[:20]
                            for arm in ("changed", "no_change_twin"):
                                arm_truth = truth_id if arm == "changed" else "no_change"
                                rows.append(
                                    {
                                        **pair_payload,
                                        "pair_id": pair_id,
                                        "statistical_cluster_id": physical_cell_id,
                                        "technical_repeat_id": repeat_id,
                                        "arm": arm,
                                        "truth_id": arm_truth,
                                        "truth_change_time": (
                                            change_time
                                            if arm == "changed"
                                            else "never"
                                        ),
                                        "evaluator_pseudo_checkpoint": (
                                            None
                                            if arm == "changed"
                                            else change_time
                                        ),
                                        "candidate_ids": candidates,
                                        "world_interventions": (
                                            interventions[truth_id]
                                            if arm == "changed"
                                            else interventions["no_change"]
                                        ),
                                        "hidden_law_changes": arm == "changed",
                                        "agent_memory_preserved_across_phase_reset": True,
                                        "total_experiment_horizon": timing[
                                            "total_experiment_horizon"
                                        ],
                                        "post_change_checkpoints": timing[
                                            "post_change_checkpoints"
                                        ],
                                        "ordinary_change_detection_claim_allowed": (
                                            timing.get(
                                                "ordinary_change_detection_claim_allowed",
                                                True,
                                            )
                                            is True
                                        ),
                                    }
                                )
    return rows


def build_static_world_identification_matrix(
    protocol: Mapping[str, Any],
    *,
    world_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Expand worlds initialized in one candidate family without inventing a change."""

    errors = validate_mechanism_adaptation_protocol(protocol)
    if errors:
        raise ValueError("invalid mechanism-adaptation protocol: " + "; ".join(errors))
    if protocol.get("schema_version") != "chemworld-mechanism-adaptation-protocol-0.3.0":
        raise ValueError("static world identification requires a v0.3 protocol")
    track = evaluation_track(protocol, "static_world_identification")
    design = protocol["design"]
    seeds = list(world_seeds or design["public_development_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("world seeds must be non-empty and unique")
    checkpoints = [int(item) for item in track["experiment_budget_checkpoints"]]
    rows: list[dict[str, Any]] = []
    for task_id, contract in protocol["task_mechanism_contracts"].items():
        candidates = list(contract["candidate_ids"])
        for truth_id in candidates:
            for label_mode in protocol["diagnosis_contract"]["candidate_label_modes"]:
                for world_seed in seeds:
                    for repeat_id, order_seed in enumerate(
                        design["candidate_order_seeds"]
                    ):
                        payload = {
                            "protocol_id": protocol["protocol_id"],
                            "evaluation_track_id": "static_world_identification",
                            "task_id": task_id,
                            "truth_id": truth_id,
                            "candidate_label_mode": label_mode,
                            "world_seed": int(world_seed),
                            "provider_repeat_id": repeat_id,
                            "candidate_order_seed": int(order_seed),
                        }
                        row_id = hashlib.sha256(
                            json.dumps(
                                payload,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()[:20]
                        rows.append(
                            {
                                **payload,
                                "row_id": row_id,
                                "candidate_ids": candidates,
                                "world_interventions": contract["interventions"][
                                    truth_id
                                ],
                                "hidden_law_changes": False,
                                "change_probability_defined": False,
                                "current_world_identification_only": True,
                                "experiment_budget_checkpoints": checkpoints,
                            }
                        )
    return rows


def evaluate_protocol_gates(
    protocol: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    gate_a_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate protocol gates without independently redefining Gate A.

    Gate A is always composed by the authoritative two-certificate decision helper.
    The legacy active-oracle-plus-decoder shortcut is deliberately not accepted.
    Absent evidence can never pass.
    """

    protocol_errors = validate_mechanism_adaptation_protocol(protocol)
    if protocol_errors:
        raise ValueError("invalid mechanism-adaptation protocol: " + "; ".join(protocol_errors))
    gates = protocol["gates"]
    schema_version = protocol.get("schema_version")
    integrity = evidence.get("gate_0")
    gate_0 = bool(isinstance(integrity, Mapping) and integrity.get("all_required_checks_pass"))

    identifiability = evidence.get("gate_a")
    certificate_decision = (
        identifiability.get("certificate_decision")
        if isinstance(identifiability, Mapping)
        else None
    )
    controlled_gate_pass = bool(
        isinstance(certificate_decision, Mapping)
        and certificate_decision.get("controlled_matched_gate_pass") is True
    )
    controlled_certificate_present = bool(
        isinstance(certificate_decision, Mapping)
        and certificate_decision.get(
            "a2_controlled_matched_certificate_present",
            True,
        )
        is True
    )
    physical_intervention_validity_pass = bool(
        isinstance(certificate_decision, Mapping)
        and certificate_decision.get("a1_physical_intervention_validity_pass")
        is True
    )
    if isinstance(identifiability, Mapping) and not isinstance(certificate_decision, Mapping):
        controlled_gate_pass = identifiability.get("controlled_matched_gate_pass") is True
        controlled_certificate_present = (
            "controlled_matched_gate_pass" in identifiability
        )
        design_validity = identifiability.get("design_validity_audit")
        physical_intervention_validity_pass = bool(
            isinstance(design_validity, Mapping)
            and design_validity.get("pass") is True
        )
    online_attainability_certificate = (
        identifiability.get("online_attainability_certificate")
        if isinstance(identifiability, Mapping)
        else None
    )
    if not (
        isinstance(online_attainability_certificate, Mapping)
        and online_attainability_certificate.get("certificate_present", True) is True
    ):
        online_attainability_certificate = None
    gate_a_decision = gate_a_certificate_decision(
        protocol,
        gate_a_plan,
        physical_intervention_validity_pass=(
            physical_intervention_validity_pass
        ),
        controlled_gate_pass=controlled_gate_pass,
        controlled_certificate_present=controlled_certificate_present,
        online_attainability_certificate=online_attainability_certificate,
    )
    gate_a = gate_a_decision["gate_a_pass"] is True

    detection = evidence.get("gate_b")
    if (
        schema_version
        == "chemworld-mechanism-adaptation-protocol-0.3.0"
    ):
        task_detection = (
            detection.get("task_reports")
            if isinstance(detection, Mapping)
            else None
        )
        family_detection = (
            detection.get("family_reports")
            if isinstance(detection, Mapping)
            else None
        )
        expected_families = {
            f"{task_id}:{candidate_id}"
            for task_id, contract in protocol[
                "task_mechanism_contracts"
            ].items()
            for candidate_id in contract["candidate_ids"]
            if candidate_id != "no_change"
        }
        gate_b = bool(
            isinstance(detection, Mapping)
            and detection.get("evaluation_subject") == "participant_agent"
            and detection.get("checkpoints_experiments_after_change")
            == [1, 2, 4, 8]
            and detection.get("delay_handling") == "right_censored_at_k_8"
            and float(detection.get("fpr_horizon", math.inf))
            <= float(
                gates["gate_b"][
                    "maximum_no_change_false_positive_rate"
                ]
            )
            and _interval_lower(detection.get("auroc_interval"))
            >= float(
                gates["gate_b"][
                    "minimum_auroc_lower_confidence_bound"
                ]
            )
            and float(
                detection.get(
                    "integrated_mean_change_probability_brier_score",
                    math.inf,
                )
            )
            <= float(
                gates["gate_b"]["maximum_integrated_mean_brier_score"]
            )
            and isinstance(task_detection, Mapping)
            and set(task_detection) == set(protocol["design"]["tasks"])
            and all(
                isinstance(item, Mapping)
                and item.get("gate_pass") is True
                for item in task_detection.values()
            )
            and isinstance(family_detection, Mapping)
            and set(family_detection) == expected_families
            and all(
                isinstance(item, Mapping)
                and item.get("gate_pass") is True
                for item in family_detection.values()
            )
            and isinstance(detection.get("macro_average"), Mapping)
            and detection["macro_average"].get("gate_pass") is True
            and detection.get("pooled_micro_average_controls_gate") is False
        )
    else:
        gate_b = bool(
            isinstance(detection, Mapping)
            and int(detection.get("no_change_twin_count", 0)) > 0
            and float(detection.get("false_positive_rate", math.inf))
            <= float(
                gates["gate_b"]["maximum_no_change_false_positive_rate"]
            )
            and _interval_lower(detection.get("auroc_interval"))
            >= float(
                gates["gate_b"][
                    "minimum_auroc_lower_confidence_bound"
                ]
            )
        )

    feedback = evidence.get("gate_c")
    gate_c = bool(
        isinstance(feedback, Mapping)
        and float(feedback.get("net_local_feedback_effect", -math.inf))
        > float(gates["gate_c"]["minimum_net_local_feedback_effect"])
        and _interval_lower(feedback.get("paired_utility_interval")) > 0.0
    )

    recovery = evidence.get("gate_d")
    gate_d = bool(
        isinstance(recovery, Mapping)
        and _interval_lower(recovery.get("adaptation_gain_interval")) > 0.0
        and _interval_lower(recovery.get("normalized_recovery_interval"))
        >= float(gates["gate_d"]["minimum_normalized_recovery_lower_confidence_bound"])
    )

    autonomy = evidence.get("gate_e")
    gate_e = bool(
        isinstance(autonomy, Mapping)
        and bool(autonomy.get("both_scores_reported"))
        and _interval_upper(autonomy.get("protocol_failure_rate_interval"))
        <= float(gates["gate_e"]["maximum_protocol_failure_rate"])
    )
    results = {
        "gate_0": gate_0,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "gate_d": gate_d,
        "gate_e": gate_e,
    }
    return {
        "gates": results,
        "all_gates_pass": all(results.values()),
        "gate_a_certificate_decision": gate_a_decision,
        "missing_or_failed_gates": [key for key, passed in results.items() if not passed],
    }


def _finite_values(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    wins = 0.0
    total = 0
    for positive_index, positive in enumerate(labels):
        if not positive:
            continue
        for negative_index, negative in enumerate(labels):
            if negative:
                continue
            total += 1
            if scores[positive_index] > scores[negative_index]:
                wins += 1.0
            elif scores[positive_index] == scores[negative_index]:
                wins += 0.5
    if total == 0:
        raise ValueError("AUROC requires both classes")
    return wins / total


def _interval_lower(value: Any) -> float:
    return float(value[0]) if isinstance(value, Sequence) and len(value) == 2 else -math.inf


def _interval_upper(value: Any) -> float:
    return float(value[1]) if isinstance(value, Sequence) and len(value) == 2 else math.inf


__all__ = [
    "MECHANISM_ADAPTATION_ANALYSIS_VERSION",
    "PRIMARY_CHANGE_TRACK_ID",
    "REQUIRED_EVALUATION_TRACK_IDS",
    "AutonomyStatus",
    "GaussianCandidatePredictive",
    "GaussianMechanismOracle",
    "OutcomeLayers",
    "active_oracle_diagnosis",
    "binary_auroc_bootstrap_interval",
    "build_paired_campaign_matrix",
    "build_static_world_identification_matrix",
    "campaign_autonomy_status",
    "categorical_entropy",
    "change_detection_summary",
    "conditional_changed_family_distribution",
    "declared_change_probability",
    "declared_distribution_update",
    "evaluate_protocol_gates",
    "evaluation_track",
    "feedback_effect_summary",
    "fixed_trajectory_decode",
    "identifiability_certificate",
    "js_divergence",
    "kl_divergence",
    "load_mechanism_adaptation_protocol",
    "multiclass_brier",
    "normalized_distribution",
    "operation_aware_action_distance",
    "recovery_decomposition",
    "reference_sufficiency_certificate",
    "validate_mechanism_adaptation_protocol",
    "wilson_interval",
]
