"""Stateful scientific scaffold for development mechanism-adaptation studies."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from chemworld.agents.interaction import AgentDecisionContext
from chemworld.agents.mechanism_adaptation_live_llm import (
    MechanismAdaptationLiveLLMAgent,
)
from chemworld.eval.mechanism_adaptation import normalized_distribution

STATEFUL_SCIENTIFIC_SCAFFOLD_VERSION = "chemworld-stateful-scientific-agent-0.4-dev"
STATEFUL_SCIENTIFIC_PROMPT_STATE_VERSION = (
    "chemworld-stateful-scientific-public-prompt-state-0.4-dev"
)
_PLAN_STATUSES = {"pending", "active", "completed", "abandoned"}
_MAX_PLAN_ITEMS = 2
_MAX_EVIDENCE_ITEMS = 2
MAX_SCIENTIFIC_STATE_JSON_CHARACTERS = 1_400


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field} exceeds its development character limit")
    return normalized


def _label_list(
    value: Any,
    *,
    field: str,
    labels: set[str],
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of public candidate labels")
    normalized = [str(item) for item in value]
    if len(set(normalized)) != len(normalized) or not set(normalized).issubset(labels):
        raise ValueError(f"{field} contains duplicate or unknown candidate labels")
    return normalized


class StatefulScientificMechanismAgent(MechanismAdaptationLiveLLMAgent):
    """Add bounded, Agent-authored scientific state without oracle knowledge.

    The public observation adapter, operation contract, provider call count, and
    mechanism candidates remain those of :class:`MechanismAdaptationLiveLLMAgent`.
    The scaffold only persists a validated state returned by the model itself.
    """

    name = "stateful_scientific_mechanism_agent"

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._scientific_state: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload.update(
            {
                "agent_family": type(self).__name__,
                "scaffold_id": "stateful_scientific",
                "stateful_scientific_scaffold_version": (
                    STATEFUL_SCIENTIFIC_SCAFFOLD_VERSION
                ),
                "state_contract_sha256": _canonical_sha256(
                    self._state_shape()
                ),
                "provider_calls_per_logical_decision": 1,
                "scientific_state_origin": "agent_generated_public_evidence_only",
                "gate_a_oracle_knowledge_supplied": False,
                "harness_generated_scientific_content": False,
            }
        )
        return payload

    def export_prompt_state(self) -> dict[str, Any]:
        state = super().export_prompt_state()
        state["base_schema_version"] = state["schema_version"]
        state["schema_version"] = STATEFUL_SCIENTIFIC_PROMPT_STATE_VERSION
        state["scientific_state"] = copy.deepcopy(self._scientific_state)
        state["scientific_state_sha256"] = (
            _canonical_sha256(self._scientific_state)
            if self._scientific_state is not None
            else None
        )
        return state

    def restore_prompt_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != STATEFUL_SCIENTIFIC_PROMPT_STATE_VERSION:
            raise ValueError("unsupported stateful-scientific prompt-state schema")
        base_schema = state.get("base_schema_version")
        if not isinstance(base_schema, str) or not base_schema:
            raise ValueError("stateful-scientific snapshot is missing its base schema")
        raw_scientific_state = state.get("scientific_state")
        expected_hash = state.get("scientific_state_sha256")
        if raw_scientific_state is None:
            if expected_hash is not None:
                raise ValueError("empty scientific state cannot declare a state hash")
            validated_state = None
        else:
            validated_state = self._validate_scientific_state(raw_scientific_state)
            if expected_hash != _canonical_sha256(validated_state):
                raise ValueError("scientific state hash mismatch")
        base_state = dict(state)
        base_state["schema_version"] = base_schema
        for key in (
            "base_schema_version",
            "scientific_state",
            "scientific_state_sha256",
        ):
            base_state.pop(key, None)
        super().restore_prompt_state(base_state)
        self._scientific_state = copy.deepcopy(validated_state)

    def scientific_state(self) -> dict[str, Any] | None:
        """Return a defensive copy for development audits and tests."""

        return copy.deepcopy(self._scientific_state)

    def decision_audit(self) -> dict[str, Any] | None:
        payload = super().decision_audit()
        if payload is None:
            return None
        payload.update(
            {
                "scaffold_id": "stateful_scientific",
                "scientific_state_sha256": (
                    _canonical_sha256(self._scientific_state)
                    if self._scientific_state is not None
                    else None
                ),
                "scientific_state_update_source": "agent_generated",
            }
        )
        return payload

    def _build_prompt(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> str:
        payload = self._build_mechanism_adaptation_payload(context, public_view)
        payload["instruction"] = (
            str(payload["instruction"])
            + " In the same response, return concise scientific_state using public "
            "evidence only."
        )
        payload["prior_scientific_state"] = copy.deepcopy(self._scientific_state)
        payload["stateful_scientific_contract"] = {
            "source": "public evidence only; hidden truth and oracle data forbidden",
            "limits": (
                "plan<=2; evidence<=2; state JSON<=1400 chars"
            ),
            "predictions": "use expected_effect and belief_update_rule",
        }
        shape = payload["required_json_shape"]
        shape.pop("mechanism_report", None)
        shape["scientific_state"] = self._state_shape()
        return self._serialize_extended_prompt(payload)

    def _normalize_decision(
        self,
        payload: dict[str, Any],
        *,
        context: AgentDecisionContext,
    ) -> dict[str, Any]:
        validated_state = self._validate_scientific_state(
            payload.get("scientific_state")
        )
        delegated_payload = copy.deepcopy(payload)
        delegated_payload["mechanism_report"] = {
            "mechanism_distribution": validated_state[
                "mechanism_distribution"
            ]
        }
        decision = super()._normalize_decision(
            delegated_payload,
            context=context,
        )
        self._scientific_state = copy.deepcopy(validated_state)
        decision.update(
            {
                "scientific_state": copy.deepcopy(validated_state),
                "scientific_state_sha256": _canonical_sha256(validated_state),
                "scientific_state_update_source": "agent_generated",
                "scaffold_id": "stateful_scientific",
            }
        )
        return decision

    def _failure_decision(
        self,
        context: AgentDecisionContext,
        error: Exception,
    ) -> dict[str, Any]:
        decision = super()._failure_decision(context, error)
        decision.update(
            {
                "scientific_state": copy.deepcopy(self._scientific_state),
                "scientific_state_sha256": (
                    _canonical_sha256(self._scientific_state)
                    if self._scientific_state is not None
                    else None
                ),
                "scientific_state_update_source": "retained_after_invalid_response",
                "scaffold_id": "stateful_scientific",
            }
        )
        return decision

    def _state_shape(self) -> dict[str, Any]:
        labels = list(self._public_to_internal)
        return {
            "current_question": "str",
            "campaign_plan": [
                {
                    "step": "str",
                    "purpose": "str",
                    "status": "pending|active|completed|abandoned",
                }
            ],
            "mechanism_distribution": dict.fromkeys(labels, 0.25),
            "evidence_ledger": [
                {
                    "evidence_id": "str",
                    "supports": ["candidate"],
                    "contradicts": ["candidate"],
                    "interpretation": "str",
                }
            ],
            "replan_trigger": "str",
            "uncertainty": 0.5,
        }

    def _validate_scientific_state(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("scientific_state must be an object")
        required = {
            "current_question",
            "campaign_plan",
            "mechanism_distribution",
            "evidence_ledger",
            "replan_trigger",
            "uncertainty",
        }
        if set(value) != required:
            raise ValueError("scientific_state must contain exactly the frozen fields")
        public_labels = set(self._public_to_internal)
        distribution = value["mechanism_distribution"]
        if not isinstance(distribution, Mapping) or set(distribution) != public_labels:
            raise ValueError(
                "scientific_state mechanism_distribution must contain public labels"
            )
        normalized = normalized_distribution(
            {str(key): raw for key, raw in distribution.items()}
        )
        plan = value["campaign_plan"]
        if (
            not isinstance(plan, list)
            or not 1 <= len(plan) <= _MAX_PLAN_ITEMS
            or not all(isinstance(item, Mapping) for item in plan)
        ):
            raise ValueError("campaign_plan must contain one or two plan objects")
        validated_plan: list[dict[str, str]] = []
        for index, item in enumerate(plan):
            if set(item) != {"step", "purpose", "status"}:
                raise ValueError("campaign_plan entries have unexpected fields")
            status = str(item["status"])
            if status not in _PLAN_STATUSES:
                raise ValueError("campaign_plan status is invalid")
            validated_plan.append(
                {
                    "step": _text(
                        item["step"],
                        field=f"campaign_plan[{index}].step",
                        maximum=160,
                    ),
                    "purpose": _text(
                        item["purpose"],
                        field=f"campaign_plan[{index}].purpose",
                        maximum=200,
                    ),
                    "status": status,
                }
            )
        raw_evidence = value["evidence_ledger"]
        if (
            not isinstance(raw_evidence, list)
            or len(raw_evidence) > _MAX_EVIDENCE_ITEMS
            or not all(isinstance(item, Mapping) for item in raw_evidence)
        ):
            raise ValueError("evidence_ledger exceeds its development contract")
        validated_evidence: list[dict[str, Any]] = []
        seen_evidence_ids: set[str] = set()
        for index, item in enumerate(raw_evidence):
            if set(item) != {
                "evidence_id",
                "supports",
                "contradicts",
                "interpretation",
            }:
                raise ValueError("evidence_ledger entry has unexpected fields")
            evidence_id = _text(
                item["evidence_id"],
                field=f"evidence_ledger[{index}].evidence_id",
                maximum=160,
            )
            if evidence_id in seen_evidence_ids:
                raise ValueError("evidence_ledger IDs must be unique")
            seen_evidence_ids.add(evidence_id)
            supports = _label_list(
                item["supports"],
                field=f"evidence_ledger[{index}].supports",
                labels=public_labels,
            )
            contradicts = _label_list(
                item["contradicts"],
                field=f"evidence_ledger[{index}].contradicts",
                labels=public_labels,
            )
            if set(supports) & set(contradicts):
                raise ValueError("one evidence item cannot support and contradict a label")
            validated_evidence.append(
                {
                    "evidence_id": evidence_id,
                    "supports": supports,
                    "contradicts": contradicts,
                    "interpretation": _text(
                        item["interpretation"],
                        field=f"evidence_ledger[{index}].interpretation",
                        maximum=220,
                    ),
                }
            )
        uncertainty = value["uncertainty"]
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, int | float):
            raise ValueError("scientific_state uncertainty must be numeric")
        uncertainty_value = float(uncertainty)
        if not math.isfinite(uncertainty_value) or not 0.0 <= uncertainty_value <= 1.0:
            raise ValueError("scientific_state uncertainty must be in [0, 1]")
        validated = {
            "current_question": _text(
                value["current_question"],
                field="current_question",
                maximum=140,
            ),
            "campaign_plan": validated_plan,
            "mechanism_distribution": normalized,
            "evidence_ledger": validated_evidence,
            "replan_trigger": _text(
                value["replan_trigger"],
                field="replan_trigger",
                maximum=120,
            ),
            "uncertainty": uncertainty_value,
        }
        if len(
            json.dumps(
                validated,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ) > MAX_SCIENTIFIC_STATE_JSON_CHARACTERS:
            raise ValueError(
                "scientific_state exceeds its development size limit "
                f"({MAX_SCIENTIFIC_STATE_JSON_CHARACTERS} characters)"
            )
        return validated


__all__ = [
    "MAX_SCIENTIFIC_STATE_JSON_CHARACTERS",
    "STATEFUL_SCIENTIFIC_PROMPT_STATE_VERSION",
    "STATEFUL_SCIENTIFIC_SCAFFOLD_VERSION",
    "StatefulScientificMechanismAgent",
]
