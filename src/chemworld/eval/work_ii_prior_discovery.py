"""Typed contracts for Work II prior revision and law discovery.

The contracts in this module are provider independent.  They separate belief
snapshots, executable law summaries, held-out predictions, experiment phases,
and blind validation so endpoint optimization cannot stand in for discovery.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

WORK_II_LAW_SUMMARY_SCHEMA_VERSION = "chemworld-work-ii-law-summary-0.1"
WORK_II_SNAPSHOT_SCHEMA_VERSION = "chemworld-work-ii-belief-snapshot-0.1"
WORK_II_HELD_OUT_QUERY_SCHEMA_VERSION = "chemworld-work-ii-held-out-query-0.1"
WORK_II_EVIDENCE_ID_MAX_ITEMS = 128
WORK_II_SNAPSHOT_STAGES = (
    "pre_evidence",
    "after_experiment_1",
    "after_experiment_2",
    "final",
)
WORK_II_LEGACY_SNAPSHOT_STAGES = (
    "pre_evidence",
    "post_neutral",
    "post_discriminating",
    "final",
)
WORK_II_LAW_BASES = frozenset(
    {
        "linear",
        "quadratic",
        "cubic",
        "interaction",
        "categorical_level",
        "conditional_linear",
        "conditional_quadratic",
        "conditional_cubic",
    }
)
WORK_II_LAW_LINKS = frozenset({"identity", "logistic"})


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{field} fields do not match the contract: missing={missing}, extra={extra}"
        )


def _text(value: object, field: str, *, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field} exceeds its length limit")
    return normalized


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def _probability(value: object, field: str) -> float:
    normalized = _finite(value, field)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return normalized


def _positive_int(value: object, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return int(value)


def _string_tuple(
    value: object,
    field: str,
    *,
    allow_empty: bool = False,
    maximum_items: int = 64,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a string list")
    normalized = tuple(_text(item, field, maximum=200) for item in value)
    if not normalized and not allow_empty:
        raise ValueError(f"{field} cannot be empty")
    if len(normalized) > maximum_items:
        raise ValueError(f"{field} exceeds its item limit")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class WorkIILawTerm:
    term_id: str
    basis: str
    input_ids: tuple[str, ...]
    coefficient: float
    category_value: str | int | float | None = None

    def evaluate(self, features: Mapping[str, str | int | float]) -> float:
        missing = [feature_id for feature_id in self.input_ids if feature_id not in features]
        if missing:
            raise ValueError(f"law term {self.term_id} lacks feature values: {missing}")
        conditional_powers = {
            "conditional_linear": 1,
            "conditional_quadratic": 2,
            "conditional_cubic": 3,
        }
        if self.basis == "categorical_level":
            return self.coefficient * float(features[self.input_ids[0]] == self.category_value)
        if self.basis in conditional_powers:
            active = float(features[self.input_ids[0]] == self.category_value)
            numeric = _finite(features[self.input_ids[1]], f"features.{self.input_ids[1]}")
            return self.coefficient * active * numeric ** conditional_powers[self.basis]
        values = [
            _finite(features[feature_id], f"features.{feature_id}")
            for feature_id in self.input_ids
        ]
        if self.basis == "linear":
            basis_value = values[0]
        elif self.basis == "quadratic":
            basis_value = values[0] ** 2
        elif self.basis == "cubic":
            basis_value = values[0] ** 3
        elif self.basis == "interaction":
            basis_value = values[0] * values[1]
        else:  # pragma: no cover - construction validates this invariant
            raise ValueError(f"unsupported law basis: {self.basis}")
        return self.coefficient * basis_value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "term_id": self.term_id,
            "basis": self.basis,
            "input_ids": list(self.input_ids),
            "coefficient": self.coefficient,
        }
        if self.basis == "categorical_level" or self.basis.startswith("conditional_"):
            payload["category_value"] = self.category_value
        return payload


@dataclass(frozen=True)
class WorkIIMetricLaw:
    metric_id: str
    intercept: float
    link: str
    lower_bound: float
    upper_bound: float
    terms: tuple[WorkIILawTerm, ...]

    def evaluate(self, features: Mapping[str, str | int | float]) -> float:
        raw = self.intercept + sum(term.evaluate(features) for term in self.terms)
        if self.link == "logistic":
            if raw >= 0:
                scaled = 1.0 / (1.0 + math.exp(-raw))
            else:
                exp_raw = math.exp(raw)
                scaled = exp_raw / (1.0 + exp_raw)
            return self.lower_bound + (self.upper_bound - self.lower_bound) * scaled
        return min(max(raw, self.lower_bound), self.upper_bound)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "intercept": self.intercept,
            "link": self.link,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "terms": [term.to_dict() for term in self.terms],
        }


@dataclass(frozen=True)
class WorkIIExecutableLawSummary:
    summary_id: str
    feature_ids: tuple[str, ...]
    metric_laws: tuple[WorkIIMetricLaw, ...]
    evidence_ids: tuple[str, ...]
    applicability: str
    limitations: tuple[str, ...]
    confidence: float

    def predict(self, features: Mapping[str, str | int | float]) -> dict[str, float]:
        return {law.metric_id: law.evaluate(features) for law in self.metric_laws}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
            "summary_id": self.summary_id,
            "feature_ids": list(self.feature_ids),
            "metric_laws": [law.to_dict() for law in self.metric_laws],
            "evidence_ids": list(self.evidence_ids),
            "applicability": self.applicability,
            "limitations": list(self.limitations),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class WorkIIHeldOutQuery:
    query_id: str
    task_id: str
    feature_values: dict[str, str | int | float]
    metric_ids: tuple[str, ...]
    replicate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORK_II_HELD_OUT_QUERY_SCHEMA_VERSION,
            "query_id": self.query_id,
            "task_id": self.task_id,
            "feature_values": dict(self.feature_values),
            "metric_ids": list(self.metric_ids),
            "replicate_count": self.replicate_count,
        }


@dataclass(frozen=True)
class WorkIIMetricPrediction:
    metric_id: str
    mean: float
    interval_lower: float
    interval_upper: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class WorkIIHeldOutPrediction:
    query_id: str
    metrics: tuple[WorkIIMetricPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


@dataclass(frozen=True)
class WorkIIPriorAssessment:
    nominal_information_available: bool
    reliability_probability: float | None
    suspected_misindexed_fields: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_information_available": self.nominal_information_available,
            "reliability_probability": self.reliability_probability,
            "suspected_misindexed_fields": list(self.suspected_misindexed_fields),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class WorkIIBeliefSnapshot:
    snapshot_id: str
    stage: str
    prior_assessment: WorkIIPriorAssessment
    predictions: tuple[WorkIIHeldOutPrediction, ...]
    law_summary: WorkIIExecutableLawSummary
    evidence_ids: tuple[str, ...]
    next_experiment_intent: str
    overall_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORK_II_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "stage": self.stage,
            "prior_assessment": self.prior_assessment.to_dict(),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "law_summary": self.law_summary.to_dict(),
            "evidence_ids": list(self.evidence_ids),
            "next_experiment_intent": self.next_experiment_intent,
            "overall_confidence": self.overall_confidence,
        }


@dataclass(frozen=True)
class WorkIIDiscoverySchedule:
    neutral_prefix_experiments: int
    discriminating_prefix_experiments: int
    autonomous_suffix_experiments: int
    held_out_query_count: int
    held_out_replicates_per_query: int
    blind_recommendation_replicates: int
    max_provider_attempts_per_decision: int
    executor_guard_margin_operations: int
    snapshot_stages: tuple[str, ...] = WORK_II_SNAPSHOT_STAGES

    @property
    def exploration_experiments(self) -> int:
        return (
            self.neutral_prefix_experiments
            + self.discriminating_prefix_experiments
            + self.autonomous_suffix_experiments
        )

    @property
    def provider_decisions_per_cell(self) -> int:
        return len(self.snapshot_stages) + self.autonomous_suffix_experiments

    @property
    def provider_attempt_cap_per_cell(self) -> int:
        return self.provider_decisions_per_cell * self.max_provider_attempts_per_decision

    @property
    def physical_experiments_per_cell(self) -> int:
        return (
            self.exploration_experiments
            + self.held_out_query_count * self.held_out_replicates_per_query
            + self.blind_recommendation_replicates
        )

    def phase_for_experiment(self, experiment_number: int) -> str:
        if not 1 <= experiment_number <= self.exploration_experiments:
            raise ValueError("experiment_number is outside the discovery schedule")
        if experiment_number <= self.neutral_prefix_experiments:
            return "neutral_prefix"
        if experiment_number <= (
            self.neutral_prefix_experiments + self.discriminating_prefix_experiments
        ):
            return "discriminating_prefix"
        return "autonomous_suffix"


def parse_work_ii_law_summary(
    payload: object,
    *,
    allowed_feature_ids: Sequence[str],
    allowed_metric_ids: Sequence[str],
    evidence_catalog: Sequence[str],
    required_metric_ids: Sequence[str] = (),
) -> WorkIIExecutableLawSummary:
    value = _mapping(payload, "law_summary")
    _exact_fields(
        value,
        {
            "schema_version",
            "summary_id",
            "feature_ids",
            "metric_laws",
            "evidence_ids",
            "applicability",
            "limitations",
            "confidence",
        },
        "law_summary",
    )
    if value["schema_version"] != WORK_II_LAW_SUMMARY_SCHEMA_VERSION:
        raise ValueError("law_summary.schema_version does not match the frozen contract")
    feature_ids = _string_tuple(value["feature_ids"], "law_summary.feature_ids")
    allowed_features = set(allowed_feature_ids)
    if not set(feature_ids).issubset(allowed_features):
        raise ValueError("law_summary contains an unknown feature ID")
    evidence_ids = _string_tuple(
        value["evidence_ids"],
        "law_summary.evidence_ids",
        allow_empty=True,
        maximum_items=WORK_II_EVIDENCE_ID_MAX_ITEMS,
    )
    if not set(evidence_ids).issubset(set(evidence_catalog)):
        raise ValueError("law_summary cites an unknown evidence ID")
    raw_metric_laws = value["metric_laws"]
    if isinstance(raw_metric_laws, (str, bytes)) or not isinstance(raw_metric_laws, Sequence):
        raise ValueError("law_summary.metric_laws must be a list")
    if not raw_metric_laws or len(raw_metric_laws) > 32:
        raise ValueError("law_summary.metric_laws must contain between 1 and 32 items")
    allowed_metrics = set(allowed_metric_ids)
    metric_laws: list[WorkIIMetricLaw] = []
    seen_metrics: set[str] = set()
    for metric_index, raw_metric in enumerate(raw_metric_laws):
        metric = _mapping(raw_metric, f"law_summary.metric_laws[{metric_index}]")
        _exact_fields(
            metric,
            {"metric_id", "intercept", "link", "lower_bound", "upper_bound", "terms"},
            f"law_summary.metric_laws[{metric_index}]",
        )
        metric_id = _text(metric["metric_id"], f"metric_laws[{metric_index}].metric_id")
        if metric_id not in allowed_metrics:
            raise ValueError("law_summary contains an unknown metric ID")
        if metric_id in seen_metrics:
            raise ValueError("law_summary metric IDs must be unique")
        seen_metrics.add(metric_id)
        link = _text(metric["link"], f"metric_laws[{metric_index}].link")
        if link not in WORK_II_LAW_LINKS:
            raise ValueError("law_summary contains an unsupported link")
        lower = _finite(metric["lower_bound"], f"metric_laws[{metric_index}].lower_bound")
        upper = _finite(metric["upper_bound"], f"metric_laws[{metric_index}].upper_bound")
        if not lower < upper:
            raise ValueError("law_summary metric bounds must be strictly increasing")
        raw_terms = metric["terms"]
        if isinstance(raw_terms, (str, bytes)) or not isinstance(raw_terms, Sequence):
            raise ValueError("law_summary metric terms must be a list")
        if len(raw_terms) > 64:
            raise ValueError("law_summary metric terms exceed the item limit")
        terms: list[WorkIILawTerm] = []
        seen_term_ids: set[str] = set()
        for term_index, raw_term in enumerate(raw_terms):
            term = _mapping(raw_term, f"metric_laws[{metric_index}].terms[{term_index}]")
            basis = _text(term.get("basis"), f"terms[{term_index}].basis")
            expected_fields = {"term_id", "basis", "input_ids", "coefficient"}
            conditional_bases = {
                "conditional_linear",
                "conditional_quadratic",
                "conditional_cubic",
            }
            if basis == "categorical_level" or basis in conditional_bases:
                expected_fields.add("category_value")
            _exact_fields(term, expected_fields, f"metric_laws[{metric_index}].terms[{term_index}]")
            if basis not in WORK_II_LAW_BASES:
                raise ValueError("law_summary contains an unsupported basis")
            term_id = _text(term["term_id"], f"terms[{term_index}].term_id")
            if term_id in seen_term_ids:
                raise ValueError("law_summary term IDs must be unique within a metric")
            seen_term_ids.add(term_id)
            input_ids = _string_tuple(term["input_ids"], f"terms[{term_index}].input_ids")
            required_inputs = 2 if basis == "interaction" or basis in conditional_bases else 1
            if len(input_ids) != required_inputs:
                raise ValueError(f"{basis} law terms require {required_inputs} input IDs")
            if not set(input_ids).issubset(set(feature_ids)):
                raise ValueError("law_summary term references an undeclared feature")
            category_value: str | int | float | None = None
            if basis == "categorical_level" or basis in conditional_bases:
                candidate = term["category_value"]
                if isinstance(candidate, bool) or not isinstance(candidate, str | int | float):
                    raise ValueError("categorical law category_value has an unsupported type")
                if isinstance(candidate, float) and not math.isfinite(candidate):
                    raise ValueError("categorical law category_value must be finite")
                category_value = candidate
            terms.append(
                WorkIILawTerm(
                    term_id=term_id,
                    basis=basis,
                    input_ids=input_ids,
                    coefficient=_finite(term["coefficient"], f"terms[{term_index}].coefficient"),
                    category_value=category_value,
                )
            )
        metric_laws.append(
            WorkIIMetricLaw(
                metric_id=metric_id,
                intercept=_finite(metric["intercept"], f"metric_laws[{metric_index}].intercept"),
                link=link,
                lower_bound=lower,
                upper_bound=upper,
                terms=tuple(terms),
            )
        )
    if required_metric_ids and seen_metrics != set(required_metric_ids):
        raise ValueError("law_summary does not cover the exact held-out metric set")
    return WorkIIExecutableLawSummary(
        summary_id=_text(value["summary_id"], "law_summary.summary_id", maximum=200),
        feature_ids=feature_ids,
        metric_laws=tuple(metric_laws),
        evidence_ids=evidence_ids,
        applicability=_text(value["applicability"], "law_summary.applicability"),
        limitations=_string_tuple(
            value["limitations"], "law_summary.limitations", allow_empty=True, maximum_items=16
        ),
        confidence=_probability(value["confidence"], "law_summary.confidence"),
    )


def parse_work_ii_belief_snapshot_header(
    payload: object,
    *,
    expected_stage: str,
    allowed_feature_ids: Sequence[str],
    allowed_prior_fields: Sequence[str],
    evidence_catalog: Sequence[str],
    nominal_information_available: bool,
) -> dict[str, Any]:
    """Validate the participant-owned non-paged portion of one snapshot."""

    value = _mapping(payload, "belief_snapshot.snapshot_header")
    _exact_fields(
        value,
        {
            "snapshot_id",
            "stage",
            "prior_assessment",
            "law_summary",
            "evidence_ids",
            "next_experiment_intent",
            "overall_confidence",
        },
        "belief_snapshot.snapshot_header",
    )
    stage = _text(value["stage"], "belief_snapshot.stage")
    if stage != expected_stage:
        raise ValueError("belief_snapshot stage does not match the requested snapshot")
    prior = _mapping(value["prior_assessment"], "belief_snapshot.prior_assessment")
    _exact_fields(
        prior,
        {
            "nominal_information_available",
            "reliability_probability",
            "suspected_misindexed_fields",
            "rationale",
        },
        "belief_snapshot.prior_assessment",
    )
    if prior["nominal_information_available"] is not nominal_information_available:
        raise ValueError("prior assessment availability does not match the public dossier")
    reliability: float | None
    if nominal_information_available:
        reliability = _probability(
            prior["reliability_probability"], "prior_assessment.reliability_probability"
        )
    else:
        if prior["reliability_probability"] is not None:
            raise ValueError("opaque cells must report null prior reliability")
        reliability = None
    suspected = _string_tuple(
        prior["suspected_misindexed_fields"],
        "prior_assessment.suspected_misindexed_fields",
        allow_empty=True,
        maximum_items=16,
    )
    if not nominal_information_available and suspected:
        raise ValueError("opaque cells cannot name suspected dossier fields")
    if not set(suspected).issubset(set(allowed_prior_fields)):
        raise ValueError("prior assessment contains an unknown dossier field")
    evidence_ids = _string_tuple(
        value["evidence_ids"],
        "belief_snapshot.evidence_ids",
        allow_empty=True,
        maximum_items=WORK_II_EVIDENCE_ID_MAX_ITEMS,
    )
    if not set(evidence_ids).issubset(set(evidence_catalog)):
        raise ValueError("belief_snapshot cites an unknown evidence ID")
    if stage == "pre_evidence" and evidence_ids:
        raise ValueError("pre-evidence snapshot cannot cite experimental evidence")

    law = _mapping(value["law_summary"], "belief_snapshot.law_summary")
    _exact_fields(
        law,
        {
            "schema_version",
            "summary_id",
            "feature_ids",
            "evidence_ids",
            "applicability",
            "limitations",
            "confidence",
        },
        "belief_snapshot.law_summary",
    )
    if law["schema_version"] != WORK_II_LAW_SUMMARY_SCHEMA_VERSION:
        raise ValueError("law_summary.schema_version does not match the frozen contract")
    feature_ids = _string_tuple(law["feature_ids"], "law_summary.feature_ids")
    if not set(feature_ids).issubset(set(allowed_feature_ids)):
        raise ValueError("law_summary contains an unknown feature ID")
    law_evidence_ids = _string_tuple(
        law["evidence_ids"],
        "law_summary.evidence_ids",
        allow_empty=True,
        maximum_items=WORK_II_EVIDENCE_ID_MAX_ITEMS,
    )
    if not set(law_evidence_ids).issubset(set(evidence_catalog)):
        raise ValueError("law_summary cites an unknown evidence ID")
    return {
        "snapshot_id": _text(value["snapshot_id"], "belief_snapshot.snapshot_id", maximum=200),
        "stage": stage,
        "prior_assessment": {
            "nominal_information_available": nominal_information_available,
            "reliability_probability": reliability,
            "suspected_misindexed_fields": list(suspected),
            "rationale": _text(prior["rationale"], "prior_assessment.rationale"),
        },
        "law_summary": {
            "schema_version": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
            "summary_id": _text(law["summary_id"], "law_summary.summary_id", maximum=200),
            "feature_ids": list(feature_ids),
            "evidence_ids": list(law_evidence_ids),
            "applicability": _text(law["applicability"], "law_summary.applicability"),
            "limitations": list(
                _string_tuple(
                    law["limitations"],
                    "law_summary.limitations",
                    allow_empty=True,
                    maximum_items=16,
                )
            ),
            "confidence": _probability(law["confidence"], "law_summary.confidence"),
        },
        "evidence_ids": list(evidence_ids),
        "next_experiment_intent": _text(
            value["next_experiment_intent"], "belief_snapshot.next_experiment_intent"
        ),
        "overall_confidence": _probability(
            value["overall_confidence"], "belief_snapshot.overall_confidence"
        ),
    }


def parse_work_ii_prediction_page(
    payload: object,
    *,
    query_metric_contract: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Validate one exact ordered page of held-out predictions."""

    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ValueError("predictions must be a list")
    expected_queries = list(query_metric_contract)
    if len(payload) != len(expected_queries):
        raise ValueError("prediction page denominator does not match its fixed page plan")
    parsed: list[dict[str, Any]] = []
    for query_index, raw_prediction in enumerate(payload):
        prediction = _mapping(raw_prediction, f"predictions[{query_index}]")
        _exact_fields(prediction, {"query_id", "metrics"}, f"predictions[{query_index}]")
        query_id = _text(prediction["query_id"], f"predictions[{query_index}].query_id")
        if query_id != expected_queries[query_index]:
            raise ValueError("prediction page does not contain its exact ordered query IDs")
        raw_metrics = prediction["metrics"]
        if isinstance(raw_metrics, (str, bytes)) or not isinstance(raw_metrics, Sequence):
            raise ValueError("prediction metrics must be a list")
        expected_metrics = list(query_metric_contract[query_id])
        if len(raw_metrics) != len(expected_metrics):
            raise ValueError("prediction metrics do not match the fixed page plan")
        metrics: list[dict[str, Any]] = []
        for metric_index, raw_metric in enumerate(raw_metrics):
            metric = _mapping(raw_metric, f"predictions[{query_index}].metrics[{metric_index}]")
            _exact_fields(
                metric,
                {"metric_id", "mean", "interval_lower", "interval_upper", "confidence"},
                f"predictions[{query_index}].metrics[{metric_index}]",
            )
            metric_id = _text(metric["metric_id"], "prediction.metric_id")
            if metric_id != expected_metrics[metric_index]:
                raise ValueError("prediction metrics do not match the fixed page plan")
            mean = _finite(metric["mean"], "prediction.mean")
            lower = _finite(metric["interval_lower"], "prediction.interval_lower")
            upper = _finite(metric["interval_upper"], "prediction.interval_upper")
            if not lower <= mean <= upper:
                raise ValueError("prediction interval must contain its mean")
            metrics.append(
                {
                    "metric_id": metric_id,
                    "mean": mean,
                    "interval_lower": lower,
                    "interval_upper": upper,
                    "confidence": _probability(metric["confidence"], "prediction.confidence"),
                }
            )
        parsed.append({"query_id": query_id, "metrics": metrics})
    return parsed


def parse_work_ii_held_out_query(
    payload: object,
    *,
    expected_task_id: str,
    allowed_feature_ids: Sequence[str],
    allowed_metric_ids: Sequence[str],
) -> WorkIIHeldOutQuery:
    value = _mapping(payload, "held_out_query")
    _exact_fields(
        value,
        {
            "schema_version",
            "query_id",
            "task_id",
            "feature_values",
            "metric_ids",
            "replicate_count",
        },
        "held_out_query",
    )
    if value["schema_version"] != WORK_II_HELD_OUT_QUERY_SCHEMA_VERSION:
        raise ValueError("held_out_query.schema_version does not match the frozen contract")
    task_id = _text(value["task_id"], "held_out_query.task_id")
    if task_id != expected_task_id:
        raise ValueError("held_out_query task does not match its cell")
    feature_values = _mapping(value["feature_values"], "held_out_query.feature_values")
    if set(feature_values) != set(allowed_feature_ids):
        raise ValueError("held_out_query must bind the exact public feature set")
    normalized_features: dict[str, str | int | float] = {}
    for feature_id, feature_value in feature_values.items():
        if isinstance(feature_value, bool) or not isinstance(feature_value, str | int | float):
            raise ValueError("held_out_query feature values must be strings or finite numbers")
        if isinstance(feature_value, float) and not math.isfinite(feature_value):
            raise ValueError("held_out_query feature values must be finite")
        normalized_features[str(feature_id)] = feature_value
    metric_ids = _string_tuple(value["metric_ids"], "held_out_query.metric_ids")
    if not set(metric_ids).issubset(set(allowed_metric_ids)):
        raise ValueError("held_out_query contains an unknown metric ID")
    return WorkIIHeldOutQuery(
        query_id=_text(value["query_id"], "held_out_query.query_id", maximum=200),
        task_id=task_id,
        feature_values=normalized_features,
        metric_ids=metric_ids,
        replicate_count=_positive_int(value["replicate_count"], "replicate_count"),
    )


def parse_work_ii_belief_snapshot(
    payload: object,
    *,
    expected_stage: str,
    query_metric_contract: Mapping[str, Sequence[str]],
    allowed_feature_ids: Sequence[str],
    allowed_metric_ids: Sequence[str],
    allowed_prior_fields: Sequence[str],
    evidence_catalog: Sequence[str],
    nominal_information_available: bool,
) -> WorkIIBeliefSnapshot:
    value = _mapping(payload, "belief_snapshot")
    _exact_fields(
        value,
        {
            "schema_version",
            "snapshot_id",
            "stage",
            "prior_assessment",
            "predictions",
            "law_summary",
            "evidence_ids",
            "next_experiment_intent",
            "overall_confidence",
        },
        "belief_snapshot",
    )
    if value["schema_version"] != WORK_II_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("belief_snapshot.schema_version does not match the frozen contract")
    stage = _text(value["stage"], "belief_snapshot.stage")
    if stage != expected_stage:
        raise ValueError("belief_snapshot stage does not match the requested snapshot")
    prior = _mapping(value["prior_assessment"], "belief_snapshot.prior_assessment")
    _exact_fields(
        prior,
        {
            "nominal_information_available",
            "reliability_probability",
            "suspected_misindexed_fields",
            "rationale",
        },
        "belief_snapshot.prior_assessment",
    )
    if prior["nominal_information_available"] is not nominal_information_available:
        raise ValueError("prior assessment availability does not match the public dossier")
    reliability: float | None
    if nominal_information_available:
        reliability = _probability(
            prior["reliability_probability"], "prior_assessment.reliability_probability"
        )
    else:
        if prior["reliability_probability"] is not None:
            raise ValueError("opaque cells must report null prior reliability")
        reliability = None
    suspected = _string_tuple(
        prior["suspected_misindexed_fields"],
        "prior_assessment.suspected_misindexed_fields",
        allow_empty=True,
        maximum_items=16,
    )
    if not nominal_information_available and suspected:
        raise ValueError("opaque cells cannot name suspected dossier fields")
    if not set(suspected).issubset(set(allowed_prior_fields)):
        raise ValueError("prior assessment contains an unknown dossier field")
    evidence_ids = _string_tuple(
        value["evidence_ids"],
        "belief_snapshot.evidence_ids",
        allow_empty=True,
        maximum_items=WORK_II_EVIDENCE_ID_MAX_ITEMS,
    )
    if not set(evidence_ids).issubset(set(evidence_catalog)):
        raise ValueError("belief_snapshot cites an unknown evidence ID")
    if stage == "pre_evidence" and evidence_ids:
        raise ValueError("pre-evidence snapshot cannot cite experimental evidence")
    raw_predictions = value["predictions"]
    if isinstance(raw_predictions, (str, bytes)) or not isinstance(raw_predictions, Sequence):
        raise ValueError("belief_snapshot.predictions must be a list")
    if len(raw_predictions) != len(query_metric_contract):
        raise ValueError("belief_snapshot prediction denominator does not match the query set")
    predictions: list[WorkIIHeldOutPrediction] = []
    seen_queries: set[str] = set()
    for query_index, raw_prediction in enumerate(raw_predictions):
        prediction = _mapping(raw_prediction, f"predictions[{query_index}]")
        _exact_fields(prediction, {"query_id", "metrics"}, f"predictions[{query_index}]")
        query_id = _text(prediction["query_id"], f"predictions[{query_index}].query_id")
        if query_id not in query_metric_contract or query_id in seen_queries:
            raise ValueError("belief_snapshot contains an unknown or duplicate query ID")
        seen_queries.add(query_id)
        raw_metrics = prediction["metrics"]
        if isinstance(raw_metrics, (str, bytes)) or not isinstance(raw_metrics, Sequence):
            raise ValueError("prediction metrics must be a list")
        metric_predictions: list[WorkIIMetricPrediction] = []
        seen_metrics: set[str] = set()
        for metric_index, raw_metric in enumerate(raw_metrics):
            metric = _mapping(raw_metric, f"predictions[{query_index}].metrics[{metric_index}]")
            _exact_fields(
                metric,
                {"metric_id", "mean", "interval_lower", "interval_upper", "confidence"},
                f"predictions[{query_index}].metrics[{metric_index}]",
            )
            metric_id = _text(metric["metric_id"], "prediction.metric_id")
            if metric_id in seen_metrics:
                raise ValueError("prediction metric IDs must be unique per query")
            seen_metrics.add(metric_id)
            mean = _finite(metric["mean"], "prediction.mean")
            lower = _finite(metric["interval_lower"], "prediction.interval_lower")
            upper = _finite(metric["interval_upper"], "prediction.interval_upper")
            if not lower <= mean <= upper:
                raise ValueError("prediction interval must contain its mean")
            metric_predictions.append(
                WorkIIMetricPrediction(
                    metric_id=metric_id,
                    mean=mean,
                    interval_lower=lower,
                    interval_upper=upper,
                    confidence=_probability(metric["confidence"], "prediction.confidence"),
                )
            )
        if seen_metrics != set(query_metric_contract[query_id]):
            raise ValueError("prediction metrics do not match the frozen query contract")
        predictions.append(
            WorkIIHeldOutPrediction(query_id=query_id, metrics=tuple(metric_predictions))
        )
    if seen_queries != set(query_metric_contract):
        raise ValueError("belief_snapshot does not cover the exact held-out query set")
    required_metrics = sorted(
        {metric_id for metric_ids in query_metric_contract.values() for metric_id in metric_ids}
    )
    law_summary = parse_work_ii_law_summary(
        value["law_summary"],
        allowed_feature_ids=allowed_feature_ids,
        allowed_metric_ids=allowed_metric_ids,
        evidence_catalog=evidence_catalog,
        required_metric_ids=required_metrics,
    )
    return WorkIIBeliefSnapshot(
        snapshot_id=_text(value["snapshot_id"], "belief_snapshot.snapshot_id", maximum=200),
        stage=stage,
        prior_assessment=WorkIIPriorAssessment(
            nominal_information_available=nominal_information_available,
            reliability_probability=reliability,
            suspected_misindexed_fields=suspected,
            rationale=_text(prior["rationale"], "prior_assessment.rationale"),
        ),
        predictions=tuple(predictions),
        law_summary=law_summary,
        evidence_ids=evidence_ids,
        next_experiment_intent=_text(
            value["next_experiment_intent"], "belief_snapshot.next_experiment_intent"
        ),
        overall_confidence=_probability(
            value["overall_confidence"], "belief_snapshot.overall_confidence"
        ),
    )


def validate_work_ii_snapshot_sequence(
    snapshots: Sequence[WorkIIBeliefSnapshot],
    *,
    expected_stages: Sequence[str] | None = None,
) -> None:
    observed_stages = tuple(snapshot.stage for snapshot in snapshots)
    if expected_stages is None:
        valid = observed_stages in {
            WORK_II_SNAPSHOT_STAGES,
            WORK_II_LEGACY_SNAPSHOT_STAGES,
        }
    else:
        frozen = tuple(str(stage) for stage in expected_stages)
        valid = bool(frozen) and observed_stages == frozen and len(set(frozen)) == len(frozen)
    if not valid:
        raise ValueError("belief snapshots must follow the frozen checkpoint order")
    snapshot_ids = [snapshot.snapshot_id for snapshot in snapshots]
    if len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError("belief snapshot IDs must be unique")
    summary_ids = [snapshot.law_summary.summary_id for snapshot in snapshots]
    if len(set(summary_ids)) != len(summary_ids):
        raise ValueError("law summary IDs must be unique across snapshots")
    query_contract = {
        prediction.query_id: tuple(metric.metric_id for metric in prediction.metrics)
        for prediction in snapshots[0].predictions
    }
    for snapshot in snapshots[1:]:
        candidate = {
            prediction.query_id: tuple(metric.metric_id for metric in prediction.metrics)
            for prediction in snapshot.predictions
        }
        if candidate != query_contract:
            raise ValueError("held-out query contracts must remain unchanged across snapshots")
        if (
            snapshot.prior_assessment.nominal_information_available
            != snapshots[0].prior_assessment.nominal_information_available
        ):
            raise ValueError("prior dossier availability changed within one trajectory")


def parse_work_ii_discovery_schedule(payload: object) -> WorkIIDiscoverySchedule:
    value = _mapping(payload, "discovery_schedule")
    _exact_fields(
        value,
        {
            "snapshot_stages",
            "neutral_prefix_experiments",
            "discriminating_prefix_experiments",
            "autonomous_suffix_experiments",
            "held_out_query_count",
            "held_out_replicates_per_query",
            "blind_recommendation_replicates",
            "max_provider_attempts_per_decision",
            "executor_guard_margin_operations",
        },
        "discovery_schedule",
    )
    stages = _string_tuple(value["snapshot_stages"], "discovery_schedule.snapshot_stages")
    if stages not in {WORK_II_SNAPSHOT_STAGES, WORK_II_LEGACY_SNAPSHOT_STAGES}:
        raise ValueError("discovery schedule must use the frozen four snapshot stages")
    return WorkIIDiscoverySchedule(
        neutral_prefix_experiments=_positive_int(
            value["neutral_prefix_experiments"], "neutral_prefix_experiments"
        ),
        discriminating_prefix_experiments=_positive_int(
            value["discriminating_prefix_experiments"],
            "discriminating_prefix_experiments",
        ),
        autonomous_suffix_experiments=_positive_int(
            value["autonomous_suffix_experiments"], "autonomous_suffix_experiments"
        ),
        held_out_query_count=_positive_int(value["held_out_query_count"], "held_out_query_count"),
        held_out_replicates_per_query=_positive_int(
            value["held_out_replicates_per_query"], "held_out_replicates_per_query"
        ),
        blind_recommendation_replicates=_positive_int(
            value["blind_recommendation_replicates"], "blind_recommendation_replicates"
        ),
        max_provider_attempts_per_decision=_positive_int(
            value["max_provider_attempts_per_decision"], "max_provider_attempts_per_decision"
        ),
        executor_guard_margin_operations=_positive_int(
            value["executor_guard_margin_operations"],
            "executor_guard_margin_operations",
            allow_zero=True,
        ),
        snapshot_stages=stages,
    )


def score_work_ii_snapshot_predictions(
    snapshot: WorkIIBeliefSnapshot,
    observed: Mapping[str, Mapping[str, float]],
) -> dict[str, float | int]:
    errors: list[float] = []
    squared_errors: list[float] = []
    covered: list[float] = []
    confidences: list[float] = []
    for prediction in snapshot.predictions:
        if prediction.query_id not in observed:
            raise ValueError(f"missing observed held-out query: {prediction.query_id}")
        query_observed = observed[prediction.query_id]
        for metric in prediction.metrics:
            if metric.metric_id not in query_observed:
                raise ValueError(
                    f"missing observed metric {metric.metric_id} for {prediction.query_id}"
                )
            actual = _finite(query_observed[metric.metric_id], "observed metric")
            error = abs(metric.mean - actual)
            errors.append(error)
            squared_errors.append(error**2)
            covered.append(float(metric.interval_lower <= actual <= metric.interval_upper))
            confidences.append(metric.confidence)
    if not errors:
        raise ValueError("snapshot has no scoreable held-out predictions")
    interval_coverage = sum(covered) / len(covered)
    mean_confidence = sum(confidences) / len(confidences)
    return {
        "prediction_count": len(errors),
        "mean_absolute_error": sum(errors) / len(errors),
        "root_mean_squared_error": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "interval_coverage": interval_coverage,
        "mean_confidence": mean_confidence,
        "coverage_confidence_gap": abs(interval_coverage - mean_confidence),
    }


__all__ = [
    "WORK_II_EVIDENCE_ID_MAX_ITEMS",
    "WORK_II_HELD_OUT_QUERY_SCHEMA_VERSION",
    "WORK_II_LAW_BASES",
    "WORK_II_LAW_LINKS",
    "WORK_II_LAW_SUMMARY_SCHEMA_VERSION",
    "WORK_II_LEGACY_SNAPSHOT_STAGES",
    "WORK_II_SNAPSHOT_SCHEMA_VERSION",
    "WORK_II_SNAPSHOT_STAGES",
    "WorkIIBeliefSnapshot",
    "WorkIIDiscoverySchedule",
    "WorkIIExecutableLawSummary",
    "WorkIIHeldOutPrediction",
    "WorkIIHeldOutQuery",
    "WorkIILawTerm",
    "WorkIIMetricLaw",
    "WorkIIMetricPrediction",
    "WorkIIPriorAssessment",
    "parse_work_ii_belief_snapshot",
    "parse_work_ii_belief_snapshot_header",
    "parse_work_ii_discovery_schedule",
    "parse_work_ii_held_out_query",
    "parse_work_ii_law_summary",
    "parse_work_ii_prediction_page",
    "score_work_ii_snapshot_predictions",
    "validate_work_ii_snapshot_sequence",
]
