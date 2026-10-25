"""Leakage-resistant live-LLM contract for mechanism-adaptation studies."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from chemworld.agents.interaction import AgentDecisionContext
from chemworld.eval.mechanism_adaptation import (
    declared_change_probability,
    normalized_distribution,
)
from chemworld.legacy.agents.diagnostic_live_llm import MechanismDiagnosticLiveLLMAgent

MECHANISM_ADAPTATION_PROMPT_VERSION = "chemworld-mechanism-adaptation-prompt-0.5"

CandidateLabelMode = Literal["semantic", "anonymous"]


@dataclass(frozen=True)
class MechanismCandidateSpec:
    """Public operational definition for one mutually exclusive candidate."""

    candidate_id: str
    public_definition: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.public_definition.strip():
            raise ValueError("candidate ID and public definition must be non-empty")


class MechanismAdaptationLiveLLMAgent(MechanismDiagnosticLiveLLMAgent):
    """Collect one declared mechanism distribution without derived-field leakage.

    The Agent reports only a categorical distribution and a public, unitless
    information-value forecast. Change probability and the top-1 candidate are derived
    by the evaluator and are deliberately absent from subsequent prompt memory.
    """

    name = "mechanism_adaptation_live_llm"

    def __init__(
        self,
        *args: Any,
        candidate_specs: Sequence[MechanismCandidateSpec],
        candidate_label_mode: CandidateLabelMode = "semantic",
        candidate_order_seed: int,
        randomize_candidate_order: bool = True,
        **kwargs: Any,
    ) -> None:
        specs = tuple(candidate_specs)
        identifiers = tuple(spec.candidate_id for spec in specs)
        if len(specs) < 2 or len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate_specs must contain at least two unique IDs")
        if "no_change" not in identifiers:
            raise ValueError("candidate_specs must include no_change")
        if candidate_label_mode not in {"semantic", "anonymous"}:
            raise ValueError("candidate_label_mode must be semantic or anonymous")
        self.candidate_specs = specs
        self.candidate_label_mode = candidate_label_mode
        self.candidate_order_seed = int(candidate_order_seed)
        self.randomize_candidate_order = bool(randomize_candidate_order)
        self._public_to_internal = {item: item for item in identifiers}
        self._internal_to_public = {item: item for item in identifiers}
        super().__init__(*args, mechanism_candidates=identifiers, **kwargs)

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        ordered = list(self.candidate_specs)
        if self.randomize_candidate_order:
            material = f"{self.role_id}|{task_info.get('task_id')}|{self.candidate_order_seed}"
            digest = hashlib.sha256(material.encode("utf-8")).digest()
            random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
        if self.candidate_label_mode == "anonymous":
            public_labels = [f"H{index}" for index in range(1, len(ordered) + 1)]
        else:
            public_labels = [spec.candidate_id for spec in ordered]
        self._public_to_internal = {
            label: spec.candidate_id for label, spec in zip(public_labels, ordered, strict=True)
        }
        self._internal_to_public = {
            internal: public for public, internal in self._public_to_internal.items()
        }

    def act_with_public_view(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]:
        action = super().act_with_public_view(context, public_view)
        if self._last_decision is not None and self._recent_decisions:
            for key in ("change_probability", "mechanism_prediction"):
                self._last_decision.pop(key, None)
                self._recent_decisions[-1].pop(key, None)
            for key in ("mechanism_distribution", "declared_information_value"):
                if key in self._last_decision:
                    self._recent_decisions[-1][key] = self._last_decision[key]
        return action

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        public_candidates = self._public_candidates()
        contract = {
            "version": MECHANISM_ADAPTATION_PROMPT_VERSION,
            "candidate_label_mode": self.candidate_label_mode,
            "randomize_candidate_order": self.randomize_candidate_order,
            "candidate_order_seed": self.candidate_order_seed,
            "public_candidates": public_candidates,
            "change_probability": "derived_as_one_minus_no_change",
        }
        payload.update(
            {
                "agent_family": type(self).__name__,
                "mechanism_adaptation_prompt_version": (MECHANISM_ADAPTATION_PROMPT_VERSION),
                "mechanism_adaptation_prompt_sha256": hashlib.sha256(
                    json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "candidate_label_mode": self.candidate_label_mode,
                "candidate_order_randomized": self.randomize_candidate_order,
                "candidate_order_seed": self.candidate_order_seed,
                "candidate_definitions_supplied": True,
                "change_probability_source": "derived_from_mechanism_distribution",
                "derived_diagnostics_returned_to_agent": False,
            }
        )
        return payload

    def _build_prompt(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> str:
        return self._serialize_extended_prompt(
            self._build_mechanism_adaptation_payload(context, public_view)
        )

    def _build_mechanism_adaptation_payload(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]:
        """Compose the final public mechanism payload before one cap check."""

        payload = self._build_diagnostic_payload(context, public_view)
        payload["instruction"] = (
            "Choose one legal operation from released evidence. Maintain a normalized "
            "distribution over the mutually exclusive candidates; count or stage is not "
            "evidence. Report a unitless, uncalibrated information-value forecast. "
            "Complete the experiment without harness-selected scientific actions."
        )
        payload["mechanism_diagnostic_contract"] = {
            "candidates": self._public_candidates(),
            "probabilities": "non-negative and sum to one",
            "ground_truth": "withheld",
        }
        shape = payload["required_json_shape"]
        shape.pop("diagnostic_report", None)
        shape["mechanism_report"] = {
            "mechanism_distribution": dict.fromkeys(
                self._public_to_internal, "probability from 0 to 1"
            ),
        }
        payload["recent_decisions"] = [
            self._sanitized_memory_item(item) for item in payload.get("recent_decisions", [])
        ]
        return payload

    def _normalize_decision(
        self,
        payload: dict[str, Any],
        *,
        context: AgentDecisionContext,
    ) -> dict[str, Any]:
        # Bypass the v0.1 diagnostic parser while retaining the base operation parser.
        decision = super(MechanismDiagnosticLiveLLMAgent, self)._normalize_decision(
            payload,
            context=context,
        )
        report = payload.get("mechanism_report")
        if not isinstance(report, Mapping):
            raise ValueError("model decision is missing mechanism_report")
        raw_distribution = report.get("mechanism_distribution")
        if not isinstance(raw_distribution, Mapping):
            raise ValueError("mechanism_distribution must be an object")
        if set(raw_distribution) != set(self._public_to_internal):
            raise ValueError("mechanism_distribution must contain exactly the public labels")
        public_distribution = normalized_distribution(
            {str(key): value for key, value in raw_distribution.items()}
        )
        internal_distribution = {
            self._public_to_internal[public]: value for public, value in public_distribution.items()
        }
        decision.update(
            {
                "mechanism_distribution": internal_distribution,
                "public_mechanism_distribution": public_distribution,
                "declared_information_value": decision[
                    "expected_information_gain"
                ],
                "diagnostic_measurement_rationale": decision[
                    "diagnostic_target"
                ],
            }
        )
        return decision

    def _failure_decision(
        self,
        context: AgentDecisionContext,
        error: Exception,
    ) -> dict[str, Any]:
        decision = super(MechanismDiagnosticLiveLLMAgent, self)._failure_decision(
            context,
            error,
        )
        uniform = 1.0 / len(self.mechanism_candidates)
        decision.update(
            {
                "mechanism_distribution": dict.fromkeys(
                    self.mechanism_candidates,
                    uniform,
                ),
                "public_mechanism_distribution": dict.fromkeys(
                    self._public_to_internal,
                    uniform,
                ),
                "declared_information_value": 0.0,
                "diagnostic_measurement_rationale": (
                    "No forecast was available because no valid model decision returned."
                ),
            }
        )
        return decision

    def derived_diagnostics(self) -> dict[str, Any] | None:
        """Return evaluator-side fields; callers must not place them in prompt memory."""

        if self._last_decision is None:
            return None
        distribution = self._last_decision.get("mechanism_distribution")
        if not isinstance(distribution, Mapping):
            return None
        values = normalized_distribution({str(key): value for key, value in distribution.items()})
        return {
            "change_probability": declared_change_probability(values),
            "mechanism_prediction": max(values, key=values.__getitem__),
        }

    def _public_candidates(self) -> list[dict[str, str]]:
        by_id = {spec.candidate_id: spec for spec in self.candidate_specs}
        return [
            {
                "label": public,
                "definition": by_id[internal].public_definition,
            }
            for public, internal in self._public_to_internal.items()
        ]

    @staticmethod
    def _sanitized_memory_item(item: Any) -> dict[str, Any]:
        if not isinstance(item, Mapping):
            return {}
        blocked = {
            "adaptation_source",
            "change_probability",
            "mechanism_prediction",
            "provider_attempts",
            "provider_model",
            "provider_request_id",
            "provider_usage",
            "status",
            "system_fingerprint",
        }
        return {str(key): value for key, value in item.items() if key not in blocked}

__all__ = [
    "MECHANISM_ADAPTATION_PROMPT_VERSION",
    "CandidateLabelMode",
    "MechanismAdaptationLiveLLMAgent",
    "MechanismCandidateSpec",
]
