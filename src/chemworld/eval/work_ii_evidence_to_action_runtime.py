"""Provider-independent recipient runtime for the five evidence-to-action conditions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_evidence_to_action import (
    CONDITION_STAGES,
    CONDITIONS,
    DONOR_CONDITION,
    DONOR_DERIVED_CONDITIONS,
    build_learned_law_artifact,
    build_yoked_evidence_packet,
)
from chemworld.eval.work_ii_prior_discovery import (
    WORK_II_EVIDENCE_ID_MAX_ITEMS,
    WORK_II_LAW_BASES,
    WORK_II_LAW_LINKS,
    WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
    WORK_II_SNAPSHOT_SCHEMA_VERSION,
    parse_work_ii_belief_snapshot,
)

RECIPIENT_CONTEXT_SCHEMA = "chemworld-work-ii-evidence-to-action-recipient-context-0.1"
TERMINAL_SUBMISSION_SCHEMA = "chemworld-work-ii-evidence-to-action-terminal-submission-0.1"
RECIPIENT_SYSTEM_PROMPT = """You are making a blinded scientific decision in ChemWorld.
Use only the public task contract, prior information, and evidence included in this turn. Candidate
outcomes and evaluator ranks are unavailable. Return exactly the requested JSON object. Rank all
candidate ActionPlans from best to worst and select the first-ranked candidate. Do not invent
experiments, observations, or hidden simulator state."""

_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "agent_trace",
        "candidate_pool_ranks",
        "candidate_truth",
        "donor_reasoning",
        "hidden_evaluator_fields",
        "presented_candidate_ranks",
    }
)


class JsonRecipientClient(Protocol):
    model: str

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        output_schema: Mapping[str, Any] | None = None,
    ) -> Any: ...


class AutonomousStratumExecutor(Protocol):
    def __call__(self, cell: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _assert_public(value: Any, *, path: str = "context") -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_PUBLIC_KEYS & {str(key) for key in value}
        if forbidden:
            raise ValueError(f"{path} contains forbidden fields: {sorted(forbidden)}")
        for key, item in value.items():
            _assert_public(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_public(item, path=f"{path}[{index}]")


def _public_candidates(candidate_packet: Mapping[str, Any] | Sequence[Any]) -> list[dict[str, Any]]:
    if isinstance(candidate_packet, Mapping):
        if candidate_packet.get("candidate_outcomes_included") is not False:
            raise ValueError("candidate packet does not prove outcome blindness")
        raw = candidate_packet.get("candidates")
    else:
        raw = candidate_packet
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("candidate packet must contain a candidate list")
    candidates = [deepcopy(dict(row)) for row in raw if isinstance(row, Mapping)]
    query_ids = [str(row.get("query_id")) for row in candidates]
    if (
        len(candidates) != 8
        or len(set(query_ids)) != 8
        or any(query_id in {"", "None"} for query_id in query_ids)
    ):
        raise ValueError("candidate packet must contain eight unique query IDs")
    _assert_public(candidates, path="candidate_packet")
    return candidates


def _yoked_evidence_at_stage(
    packet: Mapping[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    if packet.get("schema_version") != "chemworld-work-ii-yoked-evidence-packet-0.1":
        raise ValueError("yoked evidence packet schema is invalid")
    rounds = packet.get("checkpoint_rounds")
    if not isinstance(rounds, list) or len(rounds) != 12:
        raise ValueError("yoked evidence packet must contain 12 rounds")
    visible_count = {
        "pre_evidence": 0,
        "after_experiment_3": 3,
        "after_experiment_6": 6,
        "after_experiment_9": 9,
        "final": 12,
        "terminal_ranking": 12,
    }.get(stage)
    if visible_count is None:
        raise ValueError("stage is invalid for yoked evidence")
    visible = [deepcopy(dict(row)) for row in rounds[:visible_count]]
    _assert_public(visible, path="yoked_evidence")
    return visible


def build_recipient_context(
    *,
    condition: str,
    stage: str,
    task_contract: Mapping[str, Any],
    initial_world_model: Mapping[str, Any],
    candidate_packet: Mapping[str, Any] | Sequence[Any],
    yoked_evidence_packet: Mapping[str, Any] | None = None,
    law_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one condition-specific public turn without crossing reveal gates."""

    if condition not in CONDITIONS:
        raise ValueError("unknown evidence-to-action condition")
    if stage not in CONDITION_STAGES[condition]:
        raise ValueError("recipient stage differs from its condition contract")
    if condition == "autonomous_exploration" and stage != "terminal_ranking":
        raise ValueError("autonomous evidence turns belong to the campaign runtime")
    candidates = _public_candidates(candidate_packet)
    candidate_visible = stage == "terminal_ranking"
    evidence: list[dict[str, Any]] = []
    artifact: dict[str, Any] | None = None

    if condition == "yoked_evidence":
        if not isinstance(yoked_evidence_packet, Mapping):
            raise ValueError("yoked condition requires its donor evidence packet")
        evidence = _yoked_evidence_at_stage(yoked_evidence_packet, stage)
    elif yoked_evidence_packet is not None:
        raise ValueError("non-yoked condition may not receive donor evidence")

    if condition in {"learned_law_only", "oracle_law"}:
        if not isinstance(law_artifact, Mapping):
            raise ValueError("artifact-only condition requires one law artifact")
        expected_type = (
            "participant_final_typed_law"
            if condition == "learned_law_only"
            else "provider_free_disjoint_grid_fitted_predictive_law"
        )
        if law_artifact.get("artifact_type") != expected_type:
            raise ValueError("law artifact type differs from the recipient condition")
        if law_artifact.get("candidate_information_included") is not False:
            raise ValueError("law artifact does not prove candidate blindness")
        artifact = deepcopy(dict(law_artifact))
        candidate_visible = True
    elif law_artifact is not None:
        raise ValueError("condition may not receive an external law artifact")

    context: dict[str, Any] = {
        "schema_version": RECIPIENT_CONTEXT_SCHEMA,
        "condition": condition,
        "stage": stage,
        "task_contract": deepcopy(dict(task_contract)),
        "initial_world_model": deepcopy(dict(initial_world_model)),
        "visible_yoked_evidence_rounds": evidence,
        "law_artifact": artifact,
        "candidate_packet": candidates if candidate_visible else None,
        "candidate_outcomes_included": False,
        "candidate_ranks_included": False,
        "physical_experiment_authority": condition == "autonomous_exploration",
    }
    _assert_public(context)
    context["context_sha256"] = canonical_json_sha256(context)
    return context


def terminal_output_schema(candidate_query_ids: Sequence[str]) -> dict[str, Any]:
    query_ids = [str(query_id) for query_id in candidate_query_ids]
    if len(query_ids) != 8 or len(set(query_ids)) != 8:
        raise ValueError("terminal schema requires eight unique candidate IDs")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "ranking",
            "selected_query_id",
            "decision_rationale",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": TERMINAL_SUBMISSION_SCHEMA},
            "ranking": {
                "type": "array",
                "minItems": 8,
                "maxItems": 8,
                "uniqueItems": True,
                "items": {"type": "string", "enum": query_ids},
            },
            "selected_query_id": {"type": "string", "enum": query_ids},
            "decision_rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
    }


def yoked_snapshot_output_schema(
    *,
    stage: str,
    query_metric_contract: Mapping[str, Sequence[str]],
    allowed_feature_ids: Sequence[str],
    allowed_metric_ids: Sequence[str],
    allowed_prior_fields: Sequence[str],
    evidence_catalog: Sequence[str],
    nominal_information_available: bool,
) -> dict[str, Any]:
    """Build the strict provider grammar for one cumulative yoked checkpoint."""

    query_ids = [str(query_id) for query_id in query_metric_contract]
    feature_ids = [str(feature_id) for feature_id in allowed_feature_ids]
    metric_ids = [str(metric_id) for metric_id in allowed_metric_ids]
    prior_fields = [str(field_id) for field_id in allowed_prior_fields]
    evidence_ids = [str(evidence_id) for evidence_id in evidence_catalog]
    if not query_ids or not feature_ids or not metric_ids:
        raise ValueError(
            "yoked snapshot schema requires non-empty query, feature and metric scopes"
        )
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("yoked snapshot schema query IDs must be unique")
    probability = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    common_term_properties = {
        "term_id": {"type": "string", "minLength": 1, "maxLength": 200},
        "input_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
            "items": {"type": "string", "enum": feature_ids},
        },
        "coefficient": {"type": "number"},
    }
    law_term = {
        "type": "object",
        "additionalProperties": False,
        "required": ["term_id", "basis", "input_ids", "coefficient", "category_value"],
        "properties": {
            **common_term_properties,
            "basis": {"type": "string", "enum": sorted(WORK_II_LAW_BASES)},
            "category_value": {"type": ["string", "number", "null"]},
        },
    }
    prediction_metric_counts: list[int] = []
    for raw_metrics in query_metric_contract.values():
        required_metrics = [str(metric_id) for metric_id in raw_metrics]
        if not required_metrics or not set(required_metrics).issubset(set(metric_ids)):
            raise ValueError("yoked snapshot query metric scope is invalid")
        prediction_metric_counts.append(len(required_metrics))
    prediction_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query_id", "metrics"],
        "properties": {
            "query_id": {"type": "string", "enum": query_ids},
            "metrics": {
                "type": "array",
                "minItems": min(prediction_metric_counts),
                "maxItems": max(prediction_metric_counts),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "metric_id",
                        "mean",
                        "interval_lower",
                        "interval_upper",
                        "confidence",
                    ],
                    "properties": {
                        "metric_id": {"type": "string", "enum": metric_ids},
                        "mean": {"type": "number"},
                        "interval_lower": {"type": "number"},
                        "interval_upper": {"type": "number"},
                        "confidence": probability,
                    },
                },
            },
        },
    }
    metric_law = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "metric_id",
            "intercept",
            "link",
            "lower_bound",
            "upper_bound",
            "terms",
        ],
        "properties": {
            "metric_id": {"type": "string", "enum": metric_ids},
            "intercept": {"type": "number"},
            "link": {"type": "string", "enum": sorted(WORK_II_LAW_LINKS)},
            "lower_bound": {"type": "number"},
            "upper_bound": {"type": "number"},
            "terms": {"type": "array", "maxItems": 64, "items": law_term},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "snapshot_id",
            "stage",
            "prior_assessment",
            "predictions",
            "law_summary",
            "evidence_ids",
            "next_experiment_intent",
            "overall_confidence",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": WORK_II_SNAPSHOT_SCHEMA_VERSION},
            "snapshot_id": {"type": "string", "minLength": 1, "maxLength": 200},
            "stage": {"type": "string", "const": stage},
            "prior_assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "nominal_information_available",
                    "reliability_probability",
                    "suspected_misindexed_fields",
                    "rationale",
                ],
                "properties": {
                    "nominal_information_available": {
                        "type": "boolean",
                        "const": nominal_information_available,
                    },
                    "reliability_probability": (
                        probability if nominal_information_available else {"type": "null"}
                    ),
                    "suspected_misindexed_fields": {
                        "type": "array",
                        "maxItems": len(prior_fields) if nominal_information_available else 0,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": prior_fields},
                    },
                    "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
                },
            },
            "predictions": {
                "type": "array",
                "minItems": len(query_ids),
                "maxItems": len(query_ids),
                "items": prediction_item,
            },
            "law_summary": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "schema_version",
                    "summary_id",
                    "feature_ids",
                    "metric_laws",
                    "evidence_ids",
                    "applicability",
                    "limitations",
                    "confidence",
                ],
                "properties": {
                    "schema_version": {
                        "type": "string",
                        "const": WORK_II_LAW_SUMMARY_SCHEMA_VERSION,
                    },
                    "summary_id": {"type": "string", "minLength": 1, "maxLength": 200},
                    "feature_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": feature_ids},
                    },
                    "metric_laws": {
                        "type": "array",
                        "minItems": len(metric_ids),
                        "maxItems": len(metric_ids),
                        "items": metric_law,
                    },
                    "evidence_ids": {
                        "type": "array",
                        "maxItems": WORK_II_EVIDENCE_ID_MAX_ITEMS,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": evidence_ids},
                    },
                    "applicability": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "limitations": {
                        "type": "array",
                        "maxItems": 16,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1, "maxLength": 200},
                    },
                    "confidence": probability,
                },
            },
            "evidence_ids": {
                "type": "array",
                "maxItems": WORK_II_EVIDENCE_ID_MAX_ITEMS,
                "uniqueItems": True,
                "items": {"type": "string", "enum": evidence_ids},
            },
            "next_experiment_intent": {"type": "string", "minLength": 1, "maxLength": 2000},
            "overall_confidence": probability,
        },
    }


def validate_terminal_submission(
    payload: Mapping[str, Any],
    *,
    candidate_query_ids: Sequence[str],
) -> dict[str, Any]:
    expected = {"schema_version", "ranking", "selected_query_id", "decision_rationale"}
    if set(payload) != expected:
        raise ValueError("terminal submission fields differ from the contract")
    if payload.get("schema_version") != TERMINAL_SUBMISSION_SCHEMA:
        raise ValueError("terminal submission schema is invalid")
    ranking = payload.get("ranking")
    if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
        raise ValueError("terminal ranking must be a list")
    normalized = [str(query_id) for query_id in ranking]
    expected_ids = [str(query_id) for query_id in candidate_query_ids]
    if len(normalized) != 8 or len(set(normalized)) != 8 or set(normalized) != set(expected_ids):
        raise ValueError("terminal ranking must be a permutation of all candidates")
    if payload.get("selected_query_id") != normalized[0]:
        raise ValueError("selected query must equal the first-ranked candidate")
    rationale = payload.get("decision_rationale")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 2000:
        raise ValueError("terminal decision rationale is invalid")
    return {
        "schema_version": TERMINAL_SUBMISSION_SCHEMA,
        "ranking": normalized,
        "selected_query_id": normalized[0],
        "decision_rationale": rationale.strip(),
    }


def execute_terminal_recipient(
    client: JsonRecipientClient,
    context: Mapping[str, Any],
    *,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Execute one nonphysical terminal recipient turn through an injected provider client."""

    if context.get("schema_version") != RECIPIENT_CONTEXT_SCHEMA:
        raise ValueError("recipient context schema is invalid")
    if context.get("stage") != "terminal_ranking":
        raise ValueError("terminal recipient requires the terminal-ranking stage")
    candidates = context.get("candidate_packet")
    if not isinstance(candidates, list):
        raise ValueError("terminal recipient lacks its candidate packet")
    query_ids = [str(row["query_id"]) for row in candidates]
    completion = client.complete_json(
        system_prompt=RECIPIENT_SYSTEM_PROMPT,
        user_prompt=json.dumps(context, ensure_ascii=False, sort_keys=True),
        max_tokens=max_tokens,
        output_schema=terminal_output_schema(query_ids),
    )
    submission = validate_terminal_submission(
        completion.payload,
        candidate_query_ids=query_ids,
    )
    return {
        "status": "completed",
        "condition": context["condition"],
        "stage": "terminal_ranking",
        "context_sha256": context["context_sha256"],
        "submission": submission,
        "provider_model": str(completion.model),
        "provider_request_id": getattr(completion, "request_id", None),
        "provider_attempts": int(completion.attempts),
        "provider_usage": deepcopy(dict(completion.usage)),
    }


def _snapshot_required_shape(
    stage: str,
    query_metric_contract: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "schema_version": "chemworld-work-ii-belief-snapshot-0.1",
        "snapshot_id": "unique string",
        "stage": stage,
        "prior_assessment": {
            "nominal_information_available": "boolean matching the supplied prior",
            "reliability_probability": "number in [0,1], or null for opaque prior",
            "suspected_misindexed_fields": ["public prior field ID"],
            "rationale": "string",
        },
        "predictions": [
            {
                "query_id": query_id,
                "metrics": [
                    {
                        "metric_id": metric_id,
                        "mean": "number",
                        "interval_lower": "number <= mean",
                        "interval_upper": "number >= mean",
                        "confidence": "number in [0,1]",
                    }
                    for metric_id in metric_ids
                ],
            }
            for query_id, metric_ids in query_metric_contract.items()
        ],
        "law_summary": {
            "schema_version": "chemworld-work-ii-law-summary-0.1",
            "summary_id": "unique string",
            "feature_ids": ["public feature ID"],
            "metric_laws": ["typed executable metric law"],
            "evidence_ids": ["visible evidence ID"],
            "applicability": "string",
            "limitations": ["string"],
            "confidence": "number in [0,1]",
        },
        "evidence_ids": ["visible evidence ID"],
        "next_experiment_intent": "string",
        "overall_confidence": "number in [0,1]",
    }


def execute_yoked_recipient(
    client: JsonRecipientClient,
    *,
    task_contract: Mapping[str, Any],
    initial_world_model: Mapping[str, Any],
    candidate_packet: Mapping[str, Any] | Sequence[Any],
    yoked_evidence_packet: Mapping[str, Any],
    query_metric_contract: Mapping[str, Sequence[str]],
    allowed_feature_ids: Sequence[str],
    allowed_metric_ids: Sequence[str],
    allowed_prior_fields: Sequence[str],
    nominal_information_available: bool,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """Run the five matched yoked checkpoints and one terminal ranking turn."""

    snapshot_stages = (
        "pre_evidence",
        "after_experiment_3",
        "after_experiment_6",
        "after_experiment_9",
        "final",
    )
    snapshots: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for stage in snapshot_stages:
        context = build_recipient_context(
            condition="yoked_evidence",
            stage=stage,
            task_contract=task_contract,
            initial_world_model=initial_world_model,
            candidate_packet=candidate_packet,
            yoked_evidence_packet=yoked_evidence_packet,
        )
        context_without_hash = {
            key: deepcopy(value) for key, value in context.items() if key != "context_sha256"
        }
        context_without_hash["previous_belief_snapshots"] = deepcopy(snapshots)
        context_without_hash["required_json_shape"] = _snapshot_required_shape(
            stage,
            query_metric_contract,
        )
        _assert_public(context_without_hash)
        context_without_hash["context_sha256"] = canonical_json_sha256(context_without_hash)
        completion = client.complete_json(
            system_prompt=RECIPIENT_SYSTEM_PROMPT,
            user_prompt=json.dumps(context_without_hash, ensure_ascii=False, sort_keys=True),
            max_tokens=max_tokens,
            output_schema=None,
        )
        visible_ids = [
            str(event["evidence_id"])
            for round_row in context["visible_yoked_evidence_rounds"]
            for event in round_row["events"]
        ]
        snapshot = validate_yoked_snapshot_submission(
            completion.payload,
            stage=stage,
            query_metric_contract=query_metric_contract,
            allowed_feature_ids=allowed_feature_ids,
            allowed_metric_ids=allowed_metric_ids,
            allowed_prior_fields=allowed_prior_fields,
            evidence_catalog=visible_ids,
            nominal_information_available=nominal_information_available,
        )
        snapshots.append(snapshot)
        receipts.append(
            {
                "stage": stage,
                "context_sha256": context_without_hash["context_sha256"],
                "provider_model": str(completion.model),
                "provider_request_id": getattr(completion, "request_id", None),
                "provider_attempts": int(completion.attempts),
                "provider_usage": deepcopy(dict(completion.usage)),
            }
        )

    terminal_context = build_recipient_context(
        condition="yoked_evidence",
        stage="terminal_ranking",
        task_contract=task_contract,
        initial_world_model=initial_world_model,
        candidate_packet=candidate_packet,
        yoked_evidence_packet=yoked_evidence_packet,
    )
    terminal_without_hash = {
        key: deepcopy(value) for key, value in terminal_context.items() if key != "context_sha256"
    }
    terminal_without_hash["previous_belief_snapshots"] = deepcopy(snapshots)
    _assert_public(terminal_without_hash)
    terminal_without_hash["context_sha256"] = canonical_json_sha256(terminal_without_hash)
    terminal = execute_terminal_recipient(client, terminal_without_hash, max_tokens=max_tokens)
    return {
        "status": "completed",
        "condition": "yoked_evidence",
        "snapshot_count": len(snapshots),
        "belief_snapshots": snapshots,
        "snapshot_provider_receipts": receipts,
        "terminal_result": terminal,
        "provider_call_count": len(receipts) + 1,
        "physical_experiment_count": 0,
    }


def validate_yoked_snapshot_submission(
    payload: Mapping[str, Any],
    *,
    stage: str,
    query_metric_contract: Mapping[str, Sequence[str]],
    allowed_feature_ids: Sequence[str],
    allowed_metric_ids: Sequence[str],
    allowed_prior_fields: Sequence[str],
    evidence_catalog: Sequence[str],
    nominal_information_available: bool,
) -> dict[str, Any]:
    """Apply the existing typed belief schema to a yoked recipient checkpoint."""

    normalized = deepcopy(dict(payload))
    law = normalized.get("law_summary")
    law = law if isinstance(law, Mapping) else {}
    metric_laws = law.get("metric_laws")
    metric_laws = metric_laws if isinstance(metric_laws, list) else []
    unconditional_bases = WORK_II_LAW_BASES - {
        "categorical_level",
        "conditional_linear",
        "conditional_quadratic",
        "conditional_cubic",
    }
    for metric_law in metric_laws:
        if not isinstance(metric_law, Mapping):
            continue
        terms = metric_law.get("terms")
        if not isinstance(terms, list):
            continue
        for term in terms:
            if (
                isinstance(term, dict)
                and term.get("basis") in unconditional_bases
                and term.get("category_value") is None
            ):
                term.pop("category_value")
    parsed = parse_work_ii_belief_snapshot(
        normalized,
        expected_stage=stage,
        query_metric_contract=query_metric_contract,
        allowed_feature_ids=allowed_feature_ids,
        allowed_metric_ids=allowed_metric_ids,
        allowed_prior_fields=allowed_prior_fields,
        evidence_catalog=evidence_catalog,
        nominal_information_available=nominal_information_available,
    )
    return parsed.to_dict()


def resolve_dependency_status(
    cell: Mapping[str, Any],
    completed_results: Mapping[str, Mapping[str, Any]],
) -> str:
    """Resolve a scheduled cell without replacing a failed autonomous donor."""

    dependencies = cell.get("dependency_cell_ids")
    if not isinstance(dependencies, list):
        raise ValueError("cell dependency list is invalid")
    if not dependencies:
        return "ready"
    if len(dependencies) != 1:
        raise ValueError("recipient cell must have exactly one autonomous donor")
    donor_id = str(dependencies[0])
    donor = completed_results.get(donor_id)
    if donor is None:
        return "waiting_for_donor"
    campaign_summary = donor.get("campaign_summary")
    campaign_summary = campaign_summary if isinstance(campaign_summary, Mapping) else {}
    completed = donor.get("status") == "completed_uncontaminated" or (
        donor.get("completed") is True or campaign_summary.get("completed") is True
    )
    return "ready" if completed else "not_started_due_to_missing_donor"


def build_donor_derivatives(
    *,
    donor_cell_id: str,
    donor_result: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
    candidate_query_ids: Sequence[str],
) -> dict[str, Any]:
    """Create the two allowed donor products after an eligible autonomous terminal."""

    dependency_status = resolve_dependency_status(
        {"dependency_cell_ids": [donor_cell_id]},
        {donor_cell_id: donor_result},
    )
    if dependency_status != "ready":
        raise ValueError("failed autonomous donor may not produce recipient artifacts")
    campaign_summary = donor_result.get("campaign_summary")
    campaign_summary = campaign_summary if isinstance(campaign_summary, Mapping) else donor_result
    yoked = build_yoked_evidence_packet(
        trajectory_rows,
        donor_cell_id=donor_cell_id,
    )
    learned = build_learned_law_artifact(
        campaign_summary,
        donor_cell_id=donor_cell_id,
        candidate_query_ids=candidate_query_ids,
    )
    return {
        "schema_version": "chemworld-work-ii-evidence-to-action-donor-derivatives-0.1",
        "donor_cell_id": donor_cell_id,
        "yoked_evidence_packet": yoked,
        "learned_law_artifact": learned,
        "donor_reasoning_transferred": False,
        "candidate_information_transferred": False,
    }


def execute_stratum(
    client: JsonRecipientClient,
    *,
    cells: Sequence[Mapping[str, Any]],
    autonomous_executor: AutonomousStratumExecutor,
    task_contract: Mapping[str, Any],
    initial_world_model: Mapping[str, Any],
    candidate_packet: Mapping[str, Any] | Sequence[Any],
    oracle_law_artifact: Mapping[str, Any],
    query_metric_contract: Mapping[str, Sequence[str]],
    allowed_feature_ids: Sequence[str],
    allowed_metric_ids: Sequence[str],
    allowed_prior_fields: Sequence[str],
    nominal_information_available: bool,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """Execute one five-condition stratum while preserving its donor dependency graph."""

    if len(cells) != len(CONDITIONS):
        raise ValueError("one execution stratum must contain exactly five cells")
    by_condition = {str(cell.get("condition")): cell for cell in cells}
    if set(by_condition) != set(CONDITIONS):
        raise ValueError("execution stratum differs from the five-condition contract")
    stratum_ids = {str(cell.get("stratum_id")) for cell in cells}
    if len(stratum_ids) != 1 or stratum_ids == {"None"}:
        raise ValueError("all execution cells must belong to one explicit stratum")
    candidates = _public_candidates(candidate_packet)
    candidate_ids = [str(row["query_id"]) for row in candidates]
    results: dict[str, dict[str, Any]] = {}

    def observed_calls(before: int | None, *, default: int) -> int:
        after = getattr(client, "total_provider_call_count", None)
        if isinstance(before, int) and isinstance(after, int) and after >= before:
            return after - before
        return default

    def failed_recipient(
        cell: Mapping[str, Any],
        condition: str,
        error: Exception,
        *,
        before_calls: int | None,
        default_calls: int,
    ) -> dict[str, Any]:
        classification = getattr(error, "classification", None)
        if not isinstance(classification, str):
            classification = (
                "participant_schema" if isinstance(error, ValueError) else "runner_infrastructure"
            )
        row: dict[str, Any] = {
            "cell_id": str(cell["cell_id"]),
            "condition": condition,
            "status": "failed_retained",
            "failure": {
                "type": type(error).__name__,
                "classification": classification,
                "message": str(error)[:2000],
            },
            "provider_call_count": observed_calls(before_calls, default=default_calls),
            "physical_experiment_count": 0,
        }
        receipt = getattr(error, "receipt", None)
        if isinstance(receipt, Mapping):
            row["failed_provider_receipt"] = deepcopy(dict(receipt))
        return row

    def terminal_condition(
        condition: str,
        *,
        law_artifact: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        cell = by_condition[condition]
        context = build_recipient_context(
            condition=condition,
            stage="terminal_ranking",
            task_contract=task_contract,
            initial_world_model=initial_world_model,
            candidate_packet=candidate_packet,
            law_artifact=law_artifact,
        )
        before = getattr(client, "total_provider_call_count", None)
        try:
            terminal = execute_terminal_recipient(client, context, max_tokens=max_tokens)
        except Exception as error:  # retained participant/provider failure
            return failed_recipient(
                cell,
                condition,
                error,
                before_calls=before if isinstance(before, int) else None,
                default_calls=1,
            )
        return {
            "cell_id": str(cell["cell_id"]),
            "condition": condition,
            "status": "completed",
            "terminal_result": terminal,
            "submission": deepcopy(dict(terminal["submission"])),
            "provider_call_count": 1,
            "physical_experiment_count": 0,
        }

    no_evidence = terminal_condition("no_evidence")
    results[no_evidence["cell_id"]] = no_evidence

    oracle = terminal_condition("oracle_law", law_artifact=oracle_law_artifact)
    results[oracle["cell_id"]] = oracle

    donor_cell = by_condition[DONOR_CONDITION]
    donor_payload = dict(autonomous_executor(donor_cell))
    donor_payload.setdefault("cell_id", str(donor_cell["cell_id"]))
    donor_payload.setdefault("condition", DONOR_CONDITION)
    donor_id = str(donor_payload["cell_id"])
    if donor_id != str(donor_cell["cell_id"]):
        raise ValueError("autonomous executor returned a different donor identity")
    if "submission" not in donor_payload:
        donor_ranking = donor_payload.get("participant_ranking")
        if isinstance(donor_ranking, list):
            donor_payload["submission"] = {"ranking": deepcopy(donor_ranking)}
    results[donor_id] = donor_payload

    dependency_status = resolve_dependency_status(
        {"dependency_cell_ids": [donor_id]},
        results,
    )
    physical_count = donor_payload.get("physical_experiment_count")
    if (
        not isinstance(physical_count, int)
        or isinstance(physical_count, bool)
        or not 0 <= physical_count <= 12
    ):
        raise ValueError("autonomous donor physical-experiment count is invalid")
    if dependency_status == "ready" and physical_count != 12:
        raise ValueError("eligible autonomous donor must complete all 12 experiments")
    donor_submission = donor_payload.get("submission")
    donor_ranking = (
        donor_submission.get("ranking") if isinstance(donor_submission, Mapping) else None
    )
    if dependency_status == "ready" and (
        not isinstance(donor_ranking, list)
        or len(donor_ranking) != 8
        or len(set(map(str, donor_ranking))) != 8
        or set(map(str, donor_ranking)) != set(candidate_ids)
    ):
        raise ValueError("eligible autonomous donor lacks its complete terminal ranking")
    if dependency_status != "ready":
        blocked = []
        for condition in DONOR_DERIVED_CONDITIONS:
            cell = by_condition[condition]
            row = {
                "cell_id": str(cell["cell_id"]),
                "condition": condition,
                "status": "not_started_due_to_missing_donor",
                "dependency_cell_id": donor_id,
                "provider_call_count": 0,
                "physical_experiment_count": 0,
            }
            results[row["cell_id"]] = row
            blocked.append(row["cell_id"])
        return {
            "schema_version": "chemworld-work-ii-evidence-to-action-stratum-result-0.1",
            "stratum_id": next(iter(stratum_ids)),
            "status": "completed_with_failed_donor_dependencies_retained",
            "cell_results": results,
            "blocked_cell_ids": blocked,
            "provider_call_count": sum(
                int(row.get("provider_call_count", 0)) for row in results.values()
            ),
            "participant_physical_experiment_count": physical_count,
        }

    trajectory_rows = donor_payload.get("trajectory_rows")
    if not isinstance(trajectory_rows, list):
        raise ValueError("eligible autonomous donor result lacks trajectory rows")
    derivatives = build_donor_derivatives(
        donor_cell_id=donor_id,
        donor_result=donor_payload,
        trajectory_rows=trajectory_rows,
        candidate_query_ids=candidate_ids,
    )

    yoked_cell = by_condition["yoked_evidence"]
    before_yoked = getattr(client, "total_provider_call_count", None)
    try:
        yoked = execute_yoked_recipient(
            client,
            task_contract=task_contract,
            initial_world_model=initial_world_model,
            candidate_packet=candidate_packet,
            yoked_evidence_packet=derivatives["yoked_evidence_packet"],
            query_metric_contract=query_metric_contract,
            allowed_feature_ids=allowed_feature_ids,
            allowed_metric_ids=allowed_metric_ids,
            allowed_prior_fields=allowed_prior_fields,
            nominal_information_available=nominal_information_available,
            max_tokens=max_tokens,
        )
        yoked["cell_id"] = str(yoked_cell["cell_id"])
        yoked["submission"] = deepcopy(dict(yoked["terminal_result"]["submission"]))
    except Exception as error:  # retained cumulative-recipient failure
        yoked = failed_recipient(
            yoked_cell,
            "yoked_evidence",
            error,
            before_calls=before_yoked if isinstance(before_yoked, int) else None,
            default_calls=1,
        )
    results[yoked["cell_id"]] = yoked

    learned = terminal_condition(
        "learned_law_only",
        law_artifact=derivatives["learned_law_artifact"],
    )
    results[learned["cell_id"]] = learned
    retained_failures = [
        cell_id for cell_id, row in results.items() if row.get("status") == "failed_retained"
    ]
    return {
        "schema_version": "chemworld-work-ii-evidence-to-action-stratum-result-0.1",
        "stratum_id": next(iter(stratum_ids)),
        "status": "completed" if not retained_failures else "completed_with_retained_failures",
        "cell_results": results,
        "blocked_cell_ids": [],
        "failed_cell_ids": retained_failures,
        "provider_call_count": sum(
            int(row.get("provider_call_count", 0)) for row in results.values()
        ),
        "participant_physical_experiment_count": physical_count,
    }


__all__ = [
    "RECIPIENT_CONTEXT_SCHEMA",
    "RECIPIENT_SYSTEM_PROMPT",
    "TERMINAL_SUBMISSION_SCHEMA",
    "AutonomousStratumExecutor",
    "JsonRecipientClient",
    "build_donor_derivatives",
    "build_recipient_context",
    "execute_stratum",
    "execute_terminal_recipient",
    "execute_yoked_recipient",
    "resolve_dependency_status",
    "terminal_output_schema",
    "validate_terminal_submission",
    "validate_yoked_snapshot_submission",
    "yoked_snapshot_output_schema",
]
