"""Structured, evidence-grounded scoring for scientific world understanding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

WORLD_UNDERSTANDING_SCHEMA_VERSION = "chemworld-world-understanding-claim-0.1"
RELATIONS = frozenset(
    {"positive", "negative", "nonmonotonic", "conditional", "no_direct_effect"}
)


@dataclass(frozen=True)
class WorldUnderstandingClaim:
    claim_id: str
    cause_variables: tuple[str, ...]
    effect_variable: str
    relation: str
    mechanism_tags: tuple[str, ...]
    scope: str
    evidence_ids: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "cause_variables": list(self.cause_variables),
            "effect_variable": self.effect_variable,
            "relation": self.relation,
            "mechanism_tags": list(self.mechanism_tags),
            "scope": self.scope,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ReferenceWorldClaim:
    claim_id: str
    cause_variables: tuple[str, ...]
    effect_variable: str
    accepted_relations: tuple[str, ...]
    mechanism_tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReferenceWorldClaim:
        relations = _string_tuple(payload.get("accepted_relations"), "accepted_relations")
        if not set(relations).issubset(RELATIONS):
            raise ValueError("reference accepted_relations contains an unknown relation")
        return cls(
            claim_id=_text(payload.get("claim_id"), "claim_id"),
            cause_variables=_string_tuple(payload.get("cause_variables"), "cause_variables"),
            effect_variable=_text(payload.get("effect_variable"), "effect_variable"),
            accepted_relations=relations,
            mechanism_tags=_string_tuple(payload.get("mechanism_tags"), "mechanism_tags"),
        )


@dataclass(frozen=True)
class WorldUnderstandingScore:
    structural_edge_precision: float
    structural_edge_recall: float
    structural_edge_f1: float
    directional_accuracy: float
    mechanism_tag_precision: float
    mechanism_tag_recall: float
    mechanism_tag_f1: float
    unsupported_claim_rate: float
    confidence_brier_score: float
    predicted_claim_count: int
    reference_claim_count: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AtomicWorldUnderstandingScore:
    structural_edge_precision: float
    structural_edge_recall: float
    structural_edge_f1: float
    matched_edge_directional_accuracy: float
    mechanism_tag_precision: float
    mechanism_tag_recall: float
    mechanism_tag_f1: float
    out_of_reference_edge_rate: float
    confidence_brier_score: float
    predicted_claim_count: int
    predicted_atomic_edge_count: int
    reference_atomic_edge_count: int
    matched_atomic_edge_prediction_count: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def parse_world_understanding_claims(
    payload: object,
    *,
    evidence_catalog: Sequence[str] = (),
    allowed_cause_variables: Sequence[str] = (),
    allowed_effect_variables: Sequence[str] = (),
    allowed_mechanism_tags: Sequence[str] = (),
) -> tuple[WorldUnderstandingClaim, ...]:
    """Validate participant claims without exposing the hidden reference graph."""

    if not isinstance(payload, list):
        raise ValueError("structured_claims must be a list")
    evidence = set(evidence_catalog)
    allowed_causes = set(allowed_cause_variables)
    allowed_effects = set(allowed_effect_variables)
    allowed_tags = set(allowed_mechanism_tags)
    claims: list[WorldUnderstandingClaim] = []
    seen_ids: set[str] = set()
    required = {
        "claim_id",
        "cause_variables",
        "effect_variable",
        "relation",
        "mechanism_tags",
        "scope",
        "evidence_ids",
        "confidence",
    }
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(f"structured_claims[{index}] fields do not match the contract")
        claim_id = _text(item["claim_id"], f"structured_claims[{index}].claim_id")
        if claim_id in seen_ids:
            raise ValueError("structured claim IDs must be unique")
        seen_ids.add(claim_id)
        causes = _string_tuple(
            item["cause_variables"], f"structured_claims[{index}].cause_variables"
        )
        if len(set(causes)) != len(causes):
            raise ValueError("structured claim cause_variables must be unique")
        if allowed_causes and not set(causes).issubset(allowed_causes):
            raise ValueError("structured claim contains an unknown cause variable")
        effect = _text(
            item["effect_variable"], f"structured_claims[{index}].effect_variable"
        )
        if allowed_effects and effect not in allowed_effects:
            raise ValueError("structured claim contains an unknown effect variable")
        relation = _text(item["relation"], f"structured_claims[{index}].relation")
        if relation not in RELATIONS:
            raise ValueError("structured claim contains an unknown relation")
        tags = _string_tuple(
            item["mechanism_tags"], f"structured_claims[{index}].mechanism_tags"
        )
        if allowed_tags and not set(tags).issubset(allowed_tags):
            raise ValueError("structured claim contains an unknown mechanism tag")
        evidence_ids = _string_tuple(
            item["evidence_ids"],
            f"structured_claims[{index}].evidence_ids",
            allow_empty=True,
        )
        if evidence and not set(evidence_ids).issubset(evidence):
            raise ValueError("structured claim cites an unknown evidence ID")
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError("structured claim confidence must be numeric")
        confidence_float = float(confidence)
        if not isfinite(confidence_float) or not 0.0 <= confidence_float <= 1.0:
            raise ValueError("structured claim confidence must be finite and in [0,1]")
        claims.append(
            WorldUnderstandingClaim(
                claim_id=claim_id,
                cause_variables=causes,
                effect_variable=effect,
                relation=relation,
                mechanism_tags=tags,
                scope=_text(item["scope"], f"structured_claims[{index}].scope"),
                evidence_ids=evidence_ids,
                confidence=confidence_float,
            )
        )
    return tuple(claims)


def parse_world_understanding_claims_tolerant(
    payload: object,
    *,
    evidence_catalog: Sequence[str] = (),
    allowed_cause_variables: Sequence[str] = (),
    allowed_effect_variables: Sequence[str] = (),
    allowed_mechanism_tags: Sequence[str] = (),
) -> tuple[tuple[WorldUnderstandingClaim, ...], dict[str, Any]]:
    """Keep scoreable claims and report claim-local validation failures.

    This policy is intended for secondary Declared diagnostics. A malformed or
    out-of-vocabulary claim must not invalidate an otherwise usable optimization
    recommendation, blind validation, or separately frozen Predictive evaluation.
    """

    submitted = payload if isinstance(payload, list) else []
    accepted: list[WorldUnderstandingClaim] = []
    rejected: list[dict[str, int | str | list[str] | None]] = []
    seen_ids: set[str] = set()
    if not isinstance(payload, list):
        rejected.append(
            {
                "claim_index": None,
                "claim_id": None,
                "reason_code": "invalid_claim_container",
                "message": "structured_claims must be a list",
            }
        )
    for index, item in enumerate(submitted):
        claim_id = (
            str(item.get("claim_id")).strip()
            if isinstance(item, Mapping)
            and isinstance(item.get("claim_id"), str)
            and str(item.get("claim_id")).strip()
            else None
        )
        try:
            parsed = parse_world_understanding_claims(
                [item],
                evidence_catalog=evidence_catalog,
                allowed_cause_variables=allowed_cause_variables,
                allowed_effect_variables=allowed_effect_variables,
                allowed_mechanism_tags=allowed_mechanism_tags,
            )[0]
            if parsed.claim_id in seen_ids:
                raise ValueError("structured claim IDs must be unique")
        except ValueError as error:
            message = str(error)
            reason_code = _claim_rejection_reason(message)
            rejection: dict[str, int | str | list[str] | None] = {
                "claim_index": index,
                "claim_id": claim_id,
                "reason_code": reason_code,
                "message": message,
            }
            unknown_terms = _claim_rejection_unknown_terms(
                item,
                reason_code=reason_code,
                evidence_catalog=evidence_catalog,
                allowed_cause_variables=allowed_cause_variables,
                allowed_effect_variables=allowed_effect_variables,
                allowed_mechanism_tags=allowed_mechanism_tags,
            )
            if unknown_terms:
                rejection["unknown_terms"] = unknown_terms
            rejected.append(rejection)
            continue
        accepted.append(parsed)
        seen_ids.add(parsed.claim_id)
    diagnostics = {
        "schema_version": "chemworld-world-understanding-claim-diagnostics-0.2",
        "validation_policy": "unscored_unknown_terms",
        "submitted_claim_count": len(submitted),
        "scored_claim_count": len(accepted),
        "unscored_claim_count": len(rejected),
        "unscored_claims": rejected,
    }
    return tuple(accepted), diagnostics


def _claim_rejection_unknown_terms(
    item: object,
    *,
    reason_code: str,
    evidence_catalog: Sequence[str],
    allowed_cause_variables: Sequence[str],
    allowed_effect_variables: Sequence[str],
    allowed_mechanism_tags: Sequence[str],
) -> list[str]:
    if not isinstance(item, Mapping):
        return []
    field_and_allowed = {
        "unknown_cause_variable": ("cause_variables", allowed_cause_variables),
        "unknown_effect_variable": ("effect_variable", allowed_effect_variables),
        "unknown_relation": ("relation", tuple(RELATIONS)),
        "unknown_mechanism_tag": ("mechanism_tags", allowed_mechanism_tags),
        "unknown_evidence_id": ("evidence_ids", evidence_catalog),
    }.get(reason_code)
    if field_and_allowed is None:
        return []
    field, allowed_values = field_and_allowed
    raw = item.get(field)
    submitted = raw if isinstance(raw, list | tuple) else [raw]
    allowed = {str(value) for value in allowed_values}
    return sorted(
        {
            str(value).strip()
            for value in submitted
            if isinstance(value, str)
            and str(value).strip()
            and str(value).strip() not in allowed
        }
    )


def _claim_rejection_reason(message: str) -> str:
    mappings = (
        ("fields do not match", "invalid_claim_fields"),
        ("IDs must be unique", "duplicate_claim_id"),
        ("unknown cause variable", "unknown_cause_variable"),
        ("unknown effect variable", "unknown_effect_variable"),
        ("unknown relation", "unknown_relation"),
        ("unknown mechanism tag", "unknown_mechanism_tag"),
        ("unknown evidence ID", "unknown_evidence_id"),
        ("confidence", "invalid_confidence"),
        ("cause_variables", "invalid_cause_variables"),
        ("must be a non-empty string", "invalid_text"),
    )
    for fragment, reason in mappings:
        if fragment in message:
            return reason
    return "invalid_structured_claim"


def score_world_understanding(
    predicted_claims: Sequence[WorldUnderstandingClaim],
    reference_claims: Sequence[ReferenceWorldClaim],
) -> WorldUnderstandingScore:
    """Score observable causal equivalence classes, not hidden species names."""

    predicted_edges = {
        (cause, claim.effect_variable)
        for claim in predicted_claims
        for cause in claim.cause_variables
    }
    reference_edges = {
        (cause, claim.effect_variable)
        for claim in reference_claims
        for cause in claim.cause_variables
    }
    edge_tp = len(predicted_edges & reference_edges)
    edge_precision, edge_recall, edge_f1 = _prf(
        edge_tp, len(predicted_edges), len(reference_edges)
    )
    references_by_structure = {
        (frozenset(claim.cause_variables), claim.effect_variable): claim
        for claim in reference_claims
    }
    matched: list[tuple[WorldUnderstandingClaim, ReferenceWorldClaim]] = []
    unsupported = 0
    for claim in predicted_claims:
        reference = references_by_structure.get(
            (frozenset(claim.cause_variables), claim.effect_variable)
        )
        if reference is None:
            unsupported += 1
        else:
            matched.append((claim, reference))
    direction_correct = sum(
        predicted.relation in reference.accepted_relations
        for predicted, reference in matched
    )
    directional_accuracy = direction_correct / len(matched) if matched else 0.0

    predicted_tags = {
        (reference.claim_id, tag)
        for predicted, reference in matched
        for tag in predicted.mechanism_tags
    }
    reference_tags = {
        (reference.claim_id, tag)
        for reference in reference_claims
        for tag in reference.mechanism_tags
    }
    tag_tp = len(predicted_tags & reference_tags)
    tag_precision, tag_recall, tag_f1 = _prf(
        tag_tp, len(predicted_tags), len(reference_tags)
    )

    brier_terms: list[float] = []
    structurally_matched_ids = set()
    for predicted, reference in matched:
        structurally_matched_ids.add(reference.claim_id)
        outcome = float(predicted.relation in reference.accepted_relations)
        brier_terms.append((predicted.confidence - outcome) ** 2)
    brier_terms.extend(
        1.0 for reference in reference_claims if reference.claim_id not in structurally_matched_ids
    )
    for predicted in predicted_claims:
        if not any(predicted is pair[0] for pair in matched):
            brier_terms.append(predicted.confidence**2)

    return WorldUnderstandingScore(
        structural_edge_precision=edge_precision,
        structural_edge_recall=edge_recall,
        structural_edge_f1=edge_f1,
        directional_accuracy=directional_accuracy,
        mechanism_tag_precision=tag_precision,
        mechanism_tag_recall=tag_recall,
        mechanism_tag_f1=tag_f1,
        unsupported_claim_rate=(unsupported / len(predicted_claims) if predicted_claims else 0.0),
        confidence_brier_score=(sum(brier_terms) / len(brier_terms) if brier_terms else 0.0),
        predicted_claim_count=len(predicted_claims),
        reference_claim_count=len(reference_claims),
    )


def score_world_understanding_atomic_edges(
    predicted_claims: Sequence[WorldUnderstandingClaim],
    reference_claims: Sequence[ReferenceWorldClaim],
) -> AtomicWorldUnderstandingScore:
    """Score multivariate declarations as atomic observable cause-effect edges."""

    predicted_atomic = [
        (claim, (cause, claim.effect_variable))
        for claim in predicted_claims
        for cause in claim.cause_variables
    ]
    references_by_edge: dict[tuple[str, str], ReferenceWorldClaim] = {}
    for reference in reference_claims:
        for cause in reference.cause_variables:
            edge = (cause, reference.effect_variable)
            if edge in references_by_edge:
                raise ValueError("reference world understanding contains a duplicate edge")
            references_by_edge[edge] = reference

    predicted_edges = {edge for _, edge in predicted_atomic}
    reference_edges = set(references_by_edge)
    edge_tp = len(predicted_edges & reference_edges)
    edge_precision, edge_recall, edge_f1 = _prf(
        edge_tp, len(predicted_edges), len(reference_edges)
    )

    matched = [
        (claim, edge, references_by_edge[edge])
        for claim, edge in predicted_atomic
        if edge in references_by_edge
    ]
    unsupported = [
        (claim, edge)
        for claim, edge in predicted_atomic
        if edge not in references_by_edge
    ]
    direction_correct = sum(
        claim.relation in reference.accepted_relations
        for claim, _, reference in matched
    )
    directional_accuracy = direction_correct / len(matched) if matched else 0.0

    predicted_tags = {
        (edge, tag)
        for claim, edge, _ in matched
        for tag in claim.mechanism_tags
    }
    reference_tags = {
        ((cause, reference.effect_variable), tag)
        for reference in reference_claims
        for cause in reference.cause_variables
        for tag in reference.mechanism_tags
    }
    tag_tp = len(predicted_tags & reference_tags)
    tag_precision, tag_recall, tag_f1 = _prf(
        tag_tp, len(predicted_tags), len(reference_tags)
    )

    brier_terms = [
        (
            claim.confidence
            - float(claim.relation in reference.accepted_relations)
        )
        ** 2
        for claim, _, reference in matched
    ]
    brier_terms.extend(claim.confidence**2 for claim, _ in unsupported)
    brier_terms.extend(1.0 for edge in reference_edges - predicted_edges)

    return AtomicWorldUnderstandingScore(
        structural_edge_precision=edge_precision,
        structural_edge_recall=edge_recall,
        structural_edge_f1=edge_f1,
        matched_edge_directional_accuracy=directional_accuracy,
        mechanism_tag_precision=tag_precision,
        mechanism_tag_recall=tag_recall,
        mechanism_tag_f1=tag_f1,
        out_of_reference_edge_rate=(
            len(unsupported) / len(predicted_atomic) if predicted_atomic else 0.0
        ),
        confidence_brier_score=(
            sum(brier_terms) / len(brier_terms) if brier_terms else 0.0
        ),
        predicted_claim_count=len(predicted_claims),
        predicted_atomic_edge_count=len(predicted_atomic),
        reference_atomic_edge_count=len(reference_edges),
        matched_atomic_edge_prediction_count=len(matched),
    )


def _prf(true_positive: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(
    value: object, field: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a string list")
    result = tuple(_text(item, field) for item in value)
    if not result and not allow_empty:
        raise ValueError(f"{field} cannot be empty")
    return result


__all__ = [
    "RELATIONS",
    "WORLD_UNDERSTANDING_SCHEMA_VERSION",
    "AtomicWorldUnderstandingScore",
    "ReferenceWorldClaim",
    "WorldUnderstandingClaim",
    "WorldUnderstandingScore",
    "parse_world_understanding_claims",
    "parse_world_understanding_claims_tolerant",
    "score_world_understanding",
    "score_world_understanding_atomic_edges",
]
