"""Lightweight fail-closed G2 controller with one provider call per operation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

from chemworld.agents.decision_schema import build_decision_output_schema
from chemworld.agents.interaction import AgentDecisionContext
from chemworld.legacy.agents.live_llm import JsonCompletionLike, LiveLLMAgent

STRUCTURED_G2_PROMPT_CONTRACT_VERSION = "chemworld-structured-g2-operation-0.4"

STRUCTURED_G2_SYSTEM_PROMPT = """You are the operation-level experimental agent in ChemWorld.
Choose exactly one currently legal primitive operation from the compact public decision state.
The environment, lifecycle gates, campaign resource ledger, and legal action signatures are
authoritative. Manage the shared campaign pool across all planned batches; preserve enough
operation attempts, vessel starts, stock, and final-assay capacity to close every batch unless you
explicitly choose a legal discard. Material information supplied in the task is public for this
benchmark condition and must not be replaced with guessed real-world identities.

Closing the requested number of batches is the primary lifecycle obligation: an open batch at the
operation ceiling has no final assay. For each batch, establish materials and controls, use bounded
diagnostics and electrolysis, then explicitly terminate and measure with final_assay. Preserve
enough attempts to repeat that lifecycle for every unopened vessel. Repeating a legal operation
without a new observation is not by itself evidence that another repetition is valuable.

Return only the JSON object required by the strict output schema. Put every action parameter
directly beside operation. The host will not repair an invalid action, terminate, discard, or run a
final assay on your behalf. Do not claim hidden simulator state or provide private chain-of-thought;
declare only the requested concise expectation, diagnostic target, belief-update rule, uncertainty,
and action. If the transport exposes a single chemworld_decision function, call it exactly once and
do not answer with plain text; otherwise return the same decision as the required JSON object.
"""


class StructuredG2InterfaceError(RuntimeError):
    """A provider or structured-decision failure that must not reach the world."""


class StructuredG2Agent(LiveLLMAgent):
    """Select every primitive operation through a small strict-schema provider call."""

    name = "structured_g2"

    def __init__(self, client: Any, *, role_id: str, **kwargs: Any) -> None:
        super().__init__(client, role_id=role_id, **kwargs)
        self._last_output_schema_sha256: str | None = None
        self._last_output_schema_action_variant_count = 0

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._last_output_schema_sha256 = None
        self._last_output_schema_action_variant_count = 0

    def act_with_public_view(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]:
        action = super().act_with_public_view(context, public_view)
        if self._last_decision is not None:
            provider_enforced = bool(getattr(self.client, "strict_tool_calls", False))
            self._last_decision["structured_output"] = {
                "schema_sha256": self._last_output_schema_sha256,
                "action_variant_count": self._last_output_schema_action_variant_count,
                "provider_enforced": provider_enforced,
                "locally_validated": True,
            }
        return action

    def _complete_decision(
        self,
        *,
        prompt: str,
        public_view: dict[str, Any],
    ) -> JsonCompletionLike:
        tool_json = public_view.get("tool_json")
        if not isinstance(tool_json, Mapping):
            tool_json = {}
        schema = build_decision_output_schema(tool_json)
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._last_output_schema_sha256 = hashlib.sha256(encoded).hexdigest()
        action_schema = schema.get("properties", {}).get("action", {})
        variants = action_schema.get("anyOf", []) if isinstance(action_schema, Mapping) else []
        self._last_output_schema_action_variant_count = (
            len(variants) if isinstance(variants, list) else 0
        )
        client = cast(Any, self.client)
        return client.complete_json(
            system_prompt=STRUCTURED_G2_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=self.response_max_tokens,
            output_schema=schema,
        )

    def _failure_decision(
        self,
        context: AgentDecisionContext,
        error: Exception,
    ) -> dict[str, Any]:
        del context
        raise StructuredG2InterfaceError(
            "G2 decision failed before environment execution "
            f"({type(error).__name__}); no fallback action was emitted"
        ) from error

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        attempt_limit = max(int(getattr(self.client, "max_attempts", 1)), 1)
        provider_enforced = bool(getattr(self.client, "strict_tool_calls", False))
        payload.update(
            {
                "prompt_contract_version": STRUCTURED_G2_PROMPT_CONTRACT_VERSION,
                "decision_transport": "one_logical_provider_decision_per_primitive_operation",
                "provider_attempt_limit_per_operation": attempt_limit,
                "structured_output_policy": (
                    "provider_enforced_plus_locally_validated_dynamic_json_schema"
                    if provider_enforced
                    else "json_object_plus_locally_validated_dynamic_json_schema"
                ),
                "shell_tools_enabled": False,
                "lab_tool_used": False,
                "failure_policy": "fail_closed_before_environment_step",
            }
        )
        return payload

    def method_resource_usage(self) -> dict[str, Any]:
        usage = super().method_resource_usage()
        attempt_limit = max(int(getattr(self.client, "max_attempts", 1)), 1)
        provider_enforced = bool(getattr(self.client, "strict_tool_calls", False))
        provenance = usage.get("model_provenance")
        if isinstance(provenance, dict):
            parameters = provenance.get("request_parameters")
            if isinstance(parameters, dict):
                parameters.update(
                    {
                        "response_format": (
                            "dynamic_strict_json_schema"
                            if provider_enforced
                            else "json_object_plus_local_dynamic_schema_validation"
                        ),
                        "shell_tools": False,
                        "one_logical_provider_decision_per_primitive_operation": True,
                        "provider_attempt_limit_per_operation": attempt_limit,
                    }
                )
        return usage


__all__ = [
    "STRUCTURED_G2_PROMPT_CONTRACT_VERSION",
    "STRUCTURED_G2_SYSTEM_PROMPT",
    "StructuredG2Agent",
    "StructuredG2InterfaceError",
]
