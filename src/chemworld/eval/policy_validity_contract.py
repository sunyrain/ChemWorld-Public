"""Frozen experimental-agency construct and profile-record contract.

The contract deliberately defines a vector of observable policy properties.  It
does not collapse them into a scalar intelligence score, and it keeps endpoint
quality outside the construct axes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

PROFILE_SCHEMA_ID = "chemworld.experimental_agency_profile"
PROFILE_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    axis_id: str
    label: str
    operational_definition: str
    value_kind: str
    unit: str
    denominator: str
    nullable: bool
    null_when: str | None
    lower_bound: float | None
    upper_bound: float | None
    aggregation_unit: str
    known_policy_role: str
    source_events: tuple[str, ...]
    interpretation: str = "descriptive; no globally better direction"


AXES: tuple[dict[str, Any], ...] = (
    {
        "axis_id": "terminal_commitment",
        "label": "Lifecycle closure and terminal commitment",
        "construct_role": (
            "How often a started experimental lifecycle is closed, and whether the "
            "agent commits it to final assay or discards it."
        ),
    },
    {
        "axis_id": "evidence_acquisition",
        "label": "Active evidence acquisition",
        "construct_role": (
            "Whether, how often, and how early the agent requests non-final "
            "instrument evidence before a terminal decision."
        ),
    },
    {
        "axis_id": "evidence_conditioned_action",
        "label": "Action after evidence",
        "construct_role": (
            "Whether observed evidence is followed by further physical investment and "
            "whether a preregistered evidence rule predicts the terminal choice."
        ),
    },
    {
        "axis_id": "resource_deployment",
        "label": "Experimental resource deployment",
        "construct_role": (
            "How operation attempts, committed physical operations, monetary cost, and "
            "risk budget are deployed across closed lifecycles."
        ),
    },
    {
        "axis_id": "outcome_trajectory",
        "label": "Outcome-trajectory organization",
        "construct_role": (
            "When high-scoring assayed conditions are found and whether later assayed "
            "conditions retain, lose, or recover the running incumbent."
        ),
    },
)


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "closed_lifecycle_fraction",
        "terminal_commitment",
        "Closed lifecycle fraction",
        "Committed final assays plus committed discards divided by planned lifecycles.",
        "proportion",
        "fraction",
        "planned_lifecycle_count",
        False,
        None,
        0.0,
        1.0,
        "campaign",
        "completion gate",
        ("final_assay", "discard"),
    ),
    MetricSpec(
        "assay_fraction",
        "terminal_commitment",
        "Assay commitment fraction",
        "Committed final assays divided by closed lifecycles.",
        "proportion",
        "fraction",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        1.0,
        "campaign",
        "primary",
        ("final_assay",),
    ),
    MetricSpec(
        "discard_fraction",
        "terminal_commitment",
        "Discard fraction",
        "Committed discards divided by closed lifecycles.",
        "proportion",
        "fraction",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        1.0,
        "campaign",
        "primary",
        ("discard",),
    ),
    MetricSpec(
        "measured_lifecycle_fraction",
        "evidence_acquisition",
        "Measured lifecycle fraction",
        (
            "Closed lifecycles containing at least one committed non-final instrument "
            "measurement divided by closed lifecycles."
        ),
        "proportion",
        "fraction",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        1.0,
        "campaign",
        "primary",
        ("measure:nonfinal",),
    ),
    MetricSpec(
        "nonfinal_instrument_uses_per_closed_lifecycle",
        "evidence_acquisition",
        "Instrument uses per closed lifecycle",
        "Committed non-final measurements divided by closed lifecycles.",
        "nonnegative_rate",
        "uses/lifecycle",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        None,
        "campaign",
        "primary",
        ("measure:nonfinal",),
    ),
    MetricSpec(
        "mean_first_measurement_operation_fraction",
        "evidence_acquisition",
        "First-measurement timing",
        (
            "For measured lifecycles, the zero-based number of preceding attempted "
            "operations divided by the number of attempted operations before and "
            "including the terminal action, averaged across measured lifecycles."
        ),
        "proportion",
        "fraction of lifecycle operations",
        "measured_lifecycle_count",
        True,
        "no closed lifecycle contains a committed non-final measurement",
        0.0,
        1.0,
        "campaign",
        "secondary",
        ("operation_attempt", "measure:nonfinal", "final_assay", "discard"),
        "lower values indicate earlier evidence acquisition; not a quality ranking",
    ),
    MetricSpec(
        "continued_after_measurement_fraction",
        "evidence_conditioned_action",
        "Continued investment after evidence",
        (
            "Closed lifecycles with at least one committed physical process operation "
            "after their first committed non-final measurement and before termination, "
            "divided by all closed lifecycles."
        ),
        "proportion",
        "fraction",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        1.0,
        "campaign",
        "primary",
        ("measure:nonfinal", "committed_process_operation"),
    ),
    MetricSpec(
        "post_measure_process_operations_per_closed_lifecycle",
        "evidence_conditioned_action",
        "Post-measure process operations",
        (
            "Committed physical process operations after the first non-final measurement "
            "and before termination, divided by closed lifecycles."
        ),
        "nonnegative_rate",
        "operations/lifecycle",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        None,
        "campaign",
        "secondary",
        ("measure:nonfinal", "committed_process_operation"),
    ),
    MetricSpec(
        "threshold_eligible_fraction",
        "evidence_conditioned_action",
        "Threshold-eligible lifecycle fraction",
        (
            "Closed lifecycles with the preregistered diagnostic instrument and a finite "
            "decision signal divided by closed lifecycles."
        ),
        "proportion",
        "fraction",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        1.0,
        "campaign",
        "eligibility gate",
        ("measure:preregistered_diagnostic",),
    ),
    MetricSpec(
        "threshold_decision_concordance",
        "evidence_conditioned_action",
        "Evidence-to-terminal concordance",
        (
            "Threshold-eligible lifecycles whose committed terminal action equals the "
            "preregistered mapping from diagnostic signal to assay or discard, divided "
            "by threshold-eligible lifecycles."
        ),
        "proportion",
        "fraction",
        "threshold_eligible_lifecycle_count",
        True,
        "no closed lifecycle has a finite preregistered diagnostic signal",
        0.0,
        1.0,
        "campaign",
        "primary",
        ("measure:preregistered_diagnostic", "final_assay", "discard"),
    ),
    MetricSpec(
        "attempted_operations_per_closed_lifecycle",
        "resource_deployment",
        "Attempted operations per closed lifecycle",
        "All charged operation attempts divided by closed lifecycles.",
        "nonnegative_rate",
        "attempts/lifecycle",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        None,
        "campaign",
        "primary",
        ("operation_attempt",),
    ),
    MetricSpec(
        "committed_operations_per_closed_lifecycle",
        "resource_deployment",
        "Committed operations per closed lifecycle",
        "Committed typed operations divided by closed lifecycles.",
        "nonnegative_rate",
        "operations/lifecycle",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        None,
        "campaign",
        "secondary",
        ("operation_committed",),
    ),
    MetricSpec(
        "total_cost_per_closed_lifecycle",
        "resource_deployment",
        "Cost per closed lifecycle",
        "Campaign cost-ledger debit divided by closed lifecycles.",
        "nonnegative_rate",
        "cost units/lifecycle",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        None,
        "campaign",
        "primary",
        ("cost_ledger_delta",),
    ),
    MetricSpec(
        "total_risk_per_closed_lifecycle",
        "resource_deployment",
        "Risk debit per closed lifecycle",
        "Campaign risk-ledger debit divided by closed lifecycles.",
        "nonnegative_rate",
        "risk units/lifecycle",
        "closed_lifecycle_count",
        True,
        "no lifecycle is closed",
        0.0,
        None,
        "campaign",
        "secondary",
        ("risk_ledger_delta",),
    ),
    MetricSpec(
        "global_best_discovery_fraction",
        "outcome_trajectory",
        "Global-best discovery fraction",
        (
            "Zero-based ordinal of the first observed campaign-best final assay divided "
            "by one fewer than the number of final assays; defined as zero for one assay."
        ),
        "proportion",
        "fraction of assayed sequence",
        "final_assay_count",
        True,
        "no committed final assay exists",
        0.0,
        1.0,
        "campaign",
        "descriptive extension",
        ("final_assay:score",),
        "lower values mean earlier discovery of the observed campaign best",
    ),
    MetricSpec(
        "online_incumbent_retention_rate",
        "outcome_trajectory",
        "Online incumbent retention",
        (
            "Final assays after the first scoring at least the frozen retention fraction "
            "of their pre-assay incumbent, divided by retention opportunities."
        ),
        "proportion",
        "fraction",
        "final_assay_count_minus_one",
        True,
        "fewer than two committed final assays exist",
        0.0,
        1.0,
        "campaign",
        "descriptive extension",
        ("final_assay:score",),
    ),
    MetricSpec(
        "maximum_absolute_incumbent_drawdown",
        "outcome_trajectory",
        "Maximum incumbent drawdown",
        (
            "Largest positive difference between a pre-assay running-best score and the "
            "next assayed score."
        ),
        "nonnegative_rate",
        "score units",
        "final_assay_count_minus_one",
        True,
        "fewer than two committed final assays exist",
        0.0,
        None,
        "campaign",
        "descriptive extension",
        ("final_assay:score",),
        "lower values indicate smaller observed loss from the incumbent",
    ),
    MetricSpec(
        "loss_episode_recovery_rate",
        "outcome_trajectory",
        "Loss-episode recovery rate",
        (
            "Loss episodes recovered before campaign termination divided by all observed "
            "loss episodes; unresolved terminal episodes remain right-censored failures."
        ),
        "proportion",
        "fraction",
        "loss_episode_count",
        True,
        "no loss episode is observed",
        0.0,
        1.0,
        "campaign",
        "descriptive extension",
        ("final_assay:score",),
    ),
    MetricSpec(
        "terminal_to_global_best_ratio",
        "outcome_trajectory",
        "Terminal-to-best retention",
        "Terminal assayed score divided by the observed campaign-best assayed score.",
        "proportion",
        "ratio",
        "final_assay_count",
        True,
        "no positive committed final-assay score exists",
        0.0,
        1.0,
        "campaign",
        "descriptive extension",
        ("final_assay:score",),
        "higher values indicate closer terminal retention of the observed best",
    ),
)

ENDPOINT_CONTEXT: tuple[MetricSpec, ...] = (
    MetricSpec(
        "mean_assayed_score",
        "endpoint_context",
        "Mean assayed score",
        "Arithmetic mean of committed final-assay scores.",
        "score",
        "score units",
        "final_assay_count",
        True,
        "no committed final assay exists",
        None,
        None,
        "campaign",
        "context only",
        ("final_assay:score",),
        "reported beside, never combined into, the experimental-agency profile",
    ),
    MetricSpec(
        "best_assayed_score",
        "endpoint_context",
        "Best assayed score",
        "Maximum committed final-assay score in the campaign.",
        "score",
        "score units",
        "final_assay_count",
        True,
        "no committed final assay exists",
        None,
        None,
        "campaign",
        "context only",
        ("final_assay:score",),
        "reported beside, never combined into, the experimental-agency profile",
    ),
)

COUNT_FIELDS = (
    "planned_lifecycle_count",
    "closed_lifecycle_count",
    "final_assay_count",
    "discard_count",
    "measured_lifecycle_count",
    "threshold_eligible_lifecycle_count",
)


def build_profile_contract() -> dict[str, Any]:
    """Return the outcome-independent frozen contract payload."""

    return {
        "schema_id": PROFILE_SCHEMA_ID,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "construct": {
            "name": "experimental agency",
            "operational_definition": (
                "The observable organization of resource-constrained experimental choices "
                "over typed operations, active evidence acquisition, post-evidence action, "
                "and lifecycle termination in a hidden stateful chemical world."
            ),
            "unit_of_measurement": "one campaign in one fixed world and information arm",
            "representation": "multidimensional profile; no composite score",
            "explicit_non_claims": [
                "a unitary intelligence, reasoning, or chemical-knowledge score",
                "equivalence with endpoint optimization performance",
                "real-laboratory executability or safety",
                "direct measurement of private beliefs or internal cognition",
                "comparability across resource cards without an explicit contrast",
            ],
        },
        "axes": [dict(axis) for axis in AXES],
        "metrics": [asdict(metric) for metric in METRICS],
        "endpoint_context": [asdict(metric) for metric in ENDPOINT_CONTEXT],
        "profile_record": {
            "required_identity_fields": [
                "campaign_id",
                "world_id",
                "information_arm",
                "policy_id",
                "resource_card_sha256",
                "trajectory_manifest_sha256",
            ],
            "required_count_fields": list(COUNT_FIELDS),
            "construct_axis_container": "construct_axes",
            "endpoint_container": "endpoint_context",
            "reliability_fields": [
                "trajectory_exact_replay_match",
                "profile_exact_rebuild_match",
                "provider_call_count",
            ],
        },
        "counting_rules": {
            "operation_attempt": (
                "Every environment step admitted to campaign resource preflight, including "
                "validation failures and transactional rollbacks, is an attempted operation."
            ),
            "committed_operation": "An attempt with transaction_status=committed.",
            "measurement": (
                "A committed measure operation whose instrument is not final_assay; cached "
                "observations and failed measurements do not count."
            ),
            "closed_lifecycle": (
                "Exactly one committed terminal action, final_assay or discard, closes a "
                "started vessel lifecycle."
            ),
            "post_measure_process_operation": (
                "A committed non-measure, non-terminal physical operation strictly after "
                "the first committed non-final measurement in the same lifecycle."
            ),
            "cost_and_risk": (
                "Use campaign ledger deltas; penalties from charged failed attempts remain "
                "included, while rejected candidate-state physical changes remain excluded."
            ),
        },
        "aggregation": {
            "primary_unit": "campaign",
            "formal_cell": "world_id x information_arm x policy_id",
            "lifecycle_count_per_formal_cell": 6,
            "comparison_unit": "paired world x information arm",
            "pooling_rule": (
                "Compute each campaign profile first. Report policy summaries across the ten "
                "matched world-arm cells; do not pool lifecycle rows before profile creation."
            ),
            "information_contrast": (
                "Nominal-minus-opaque contrasts are paired within physical world and policy."
            ),
            "undefined_rule": (
                "Return null when a declared denominator is zero; never coerce undefined "
                "conditional metrics to zero."
            ),
        },
        "invariants": [
            "closed_lifecycle_count = final_assay_count + discard_count",
            "closed_lifecycle_count <= planned_lifecycle_count",
            "measured_lifecycle_count <= closed_lifecycle_count",
            "threshold_eligible_lifecycle_count <= measured_lifecycle_count",
            "assay_fraction + discard_fraction = 1 when closed_lifecycle_count > 0",
            "endpoint_context is null when final_assay_count = 0",
            "profile values are reconstructed from immutable events and resource ledgers",
        ],
        "reliability": {
            "exact_replay": (
                "Original and replayed trajectories must match event, state, resource, and "
                "profile hashes exactly."
            ),
            "test_retest": (
                "Deterministic known policies must yield identical profiles when rerun from "
                "the same world identity, seed, arm, and resource card."
            ),
            "provider_calls_for_known_policy_controls": 0,
        },
        "freeze_boundary": {
            "outcome_independent": True,
            "frozen_before_formal_policy_execution": True,
            "threshold_values_defined_elsewhere": "W1-V03 qualification worlds only",
            "retention_fraction": 0.9,
            "retention_fraction_provenance": (
                "pre-existing frozen G2 process-profile definition; not tuned on "
                "known-policy outcomes"
            ),
        },
        "compatibility_bindings": {
            "global_best_discovery_fraction": (
                "trajectory_learning.discovery_retention_recovery."
                "global_best_discovery_fraction"
            ),
            "online_incumbent_retention_rate": (
                "trajectory_learning.discovery_retention_recovery.online_retention_rate"
            ),
            "maximum_absolute_incumbent_drawdown": (
                "trajectory_learning.discovery_retention_recovery."
                "maximum_absolute_drawdown_from_prior_incumbent"
            ),
            "loss_episode_recovery_rate": (
                "trajectory_learning.discovery_retention_recovery.recovery_rate"
            ),
            "terminal_to_global_best_ratio": (
                "trajectory_learning.discovery_retention_recovery."
                "terminal_to_global_best_ratio"
            ),
        },
    }


def profile_contract_sha256(contract: Mapping[str, Any] | None = None) -> str:
    payload = build_profile_contract() if contract is None else dict(contract)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_specs_by_axis() -> dict[str, dict[str, MetricSpec]]:
    result: dict[str, dict[str, MetricSpec]] = {axis["axis_id"]: {} for axis in AXES}
    for metric in METRICS:
        result[metric.axis_id][metric.metric_id] = metric
    return result


def _validate_value(metric: MetricSpec, value: Any) -> list[str]:
    if value is None:
        return [] if metric.nullable else [f"{metric.metric_id} may not be null"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"{metric.metric_id} must be numeric or null"]
    numeric = float(value)
    if not math.isfinite(numeric):
        return [f"{metric.metric_id} must be finite"]
    if metric.lower_bound is not None and numeric < metric.lower_bound:
        return [f"{metric.metric_id} is below {metric.lower_bound}"]
    if metric.upper_bound is not None and numeric > metric.upper_bound:
        return [f"{metric.metric_id} is above {metric.upper_bound}"]
    return []


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def validate_profile_record(record: Mapping[str, Any]) -> list[str]:
    """Return deterministic validation errors for a profile record."""

    errors: list[str] = []
    if record.get("schema_id") != PROFILE_SCHEMA_ID:
        errors.append("schema_id does not match the frozen contract")
    if record.get("schema_version") != PROFILE_SCHEMA_VERSION:
        errors.append("schema_version does not match the frozen contract")
    if record.get("contract_sha256") != profile_contract_sha256():
        errors.append("contract_sha256 does not match the frozen contract")

    identity = record.get("identity")
    required_identity = build_profile_contract()["profile_record"][
        "required_identity_fields"
    ]
    if not isinstance(identity, Mapping):
        errors.append("identity must be a mapping")
    else:
        for field in required_identity:
            if not isinstance(identity.get(field), str) or not identity[field]:
                errors.append(f"identity.{field} must be a non-empty string")

    raw_counts = record.get("counts")
    counts: dict[str, int] = {}
    if not isinstance(raw_counts, Mapping):
        errors.append("counts must be a mapping")
    else:
        for field in COUNT_FIELDS:
            value = raw_counts.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"counts.{field} must be a non-negative integer")
            else:
                counts[field] = value

    expected_axes = _metric_specs_by_axis()
    raw_axes = record.get("construct_axes")
    if not isinstance(raw_axes, Mapping):
        errors.append("construct_axes must be a mapping")
    else:
        if set(raw_axes) != set(expected_axes):
            errors.append("construct_axes keys do not match the frozen axes")
        for axis_id, metric_specs in expected_axes.items():
            raw_axis = raw_axes.get(axis_id)
            if not isinstance(raw_axis, Mapping):
                errors.append(f"construct_axes.{axis_id} must be a mapping")
                continue
            if set(raw_axis) != set(metric_specs):
                errors.append(f"construct_axes.{axis_id} metric keys do not match")
            for metric_id, metric in metric_specs.items():
                errors.extend(_validate_value(metric, raw_axis.get(metric_id)))

    raw_endpoint = record.get("endpoint_context")
    endpoint_specs = {metric.metric_id: metric for metric in ENDPOINT_CONTEXT}
    if not isinstance(raw_endpoint, Mapping):
        errors.append("endpoint_context must be a mapping")
    else:
        if set(raw_endpoint) != set(endpoint_specs):
            errors.append("endpoint_context metric keys do not match")
        for metric_id, metric in endpoint_specs.items():
            errors.extend(_validate_value(metric, raw_endpoint.get(metric_id)))

    reliability = record.get("reliability")
    if not isinstance(reliability, Mapping):
        errors.append("reliability must be a mapping")
    else:
        for field in ("trajectory_exact_replay_match", "profile_exact_rebuild_match"):
            if not isinstance(reliability.get(field), bool):
                errors.append(f"reliability.{field} must be boolean")
        provider_calls = reliability.get("provider_call_count")
        if (
            isinstance(provider_calls, bool)
            or not isinstance(provider_calls, int)
            or provider_calls < 0
        ):
            errors.append("reliability.provider_call_count must be a non-negative integer")

    if len(counts) == len(COUNT_FIELDS):
        planned = counts["planned_lifecycle_count"]
        closed = counts["closed_lifecycle_count"]
        assays = counts["final_assay_count"]
        discards = counts["discard_count"]
        measured = counts["measured_lifecycle_count"]
        eligible = counts["threshold_eligible_lifecycle_count"]
        if closed != assays + discards:
            errors.append("closed lifecycle count must equal assays plus discards")
        if closed > planned:
            errors.append("closed lifecycle count exceeds planned count")
        if measured > closed:
            errors.append("measured lifecycle count exceeds closed count")
        if eligible > measured:
            errors.append("threshold-eligible count exceeds measured count")

        if isinstance(raw_axes, Mapping):
            terminal = raw_axes.get("terminal_commitment")
            evidence = raw_axes.get("evidence_acquisition")
            action = raw_axes.get("evidence_conditioned_action")
            if isinstance(terminal, Mapping):
                expected = closed / planned if planned else 0.0
                closed_fraction = _finite_float(
                    terminal.get("closed_lifecycle_fraction")
                )
                assay_fraction = _finite_float(terminal.get("assay_fraction"))
                discard_fraction = _finite_float(terminal.get("discard_fraction"))
                if closed_fraction is None or not math.isclose(
                    closed_fraction, expected, abs_tol=1e-12
                ):
                    errors.append("closed_lifecycle_fraction disagrees with counts")
                if closed:
                    if assay_fraction is None or not math.isclose(
                        assay_fraction, assays / closed, abs_tol=1e-12
                    ):
                        errors.append("assay_fraction disagrees with counts")
                    if discard_fraction is None or not math.isclose(
                        discard_fraction, discards / closed, abs_tol=1e-12
                    ):
                        errors.append("discard_fraction disagrees with counts")
                elif assay_fraction is not None or discard_fraction is not None:
                    errors.append("terminal fractions must be null with no closed lifecycle")
            measured_fraction = (
                _finite_float(evidence.get("measured_lifecycle_fraction"))
                if isinstance(evidence, Mapping)
                else None
            )
            if (
                isinstance(evidence, Mapping)
                and closed
                and (
                    measured_fraction is None
                    or not math.isclose(
                        measured_fraction, measured / closed, abs_tol=1e-12
                    )
                )
            ):
                errors.append("measured_lifecycle_fraction disagrees with counts")
            threshold_fraction = (
                _finite_float(action.get("threshold_eligible_fraction"))
                if isinstance(action, Mapping)
                else None
            )
            if (
                isinstance(action, Mapping)
                and closed
                and (
                    threshold_fraction is None
                    or not math.isclose(
                        threshold_fraction, eligible / closed, abs_tol=1e-12
                    )
                )
            ):
                errors.append("threshold_eligible_fraction disagrees with counts")
            if (
                eligible == 0
                and isinstance(action, Mapping)
                and action.get("threshold_decision_concordance") is not None
            ):
                errors.append(
                    "threshold_decision_concordance must be null with no eligible lifecycle"
                )
            if (
                measured == 0
                and isinstance(evidence, Mapping)
                and evidence.get("mean_first_measurement_operation_fraction") is not None
            ):
                errors.append(
                    "first-measurement timing must be null with no measured lifecycle"
                )
        if (
            assays == 0
            and isinstance(raw_endpoint, Mapping)
            and any(
                raw_endpoint.get(metric.metric_id) is not None
                for metric in ENDPOINT_CONTEXT
            )
        ):
            errors.append("endpoint_context must be null with no final assay")
        if assays == 0 and isinstance(raw_axes, Mapping):
            trajectory = raw_axes.get("outcome_trajectory")
            if isinstance(trajectory, Mapping) and any(
                trajectory.get(metric.metric_id) is not None
                for metric in METRICS
                if metric.axis_id == "outcome_trajectory"
            ):
                errors.append("outcome-trajectory metrics must be null with no final assay")

    return errors


__all__ = [
    "AXES",
    "COUNT_FIELDS",
    "ENDPOINT_CONTEXT",
    "METRICS",
    "PROFILE_SCHEMA_ID",
    "PROFILE_SCHEMA_VERSION",
    "MetricSpec",
    "build_profile_contract",
    "profile_contract_sha256",
    "validate_profile_record",
]
