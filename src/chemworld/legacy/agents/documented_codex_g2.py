"""Fail-closed G2 controller with isolated, persistent experiment documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

from chemworld.agents.decision_schema import build_decision_output_schema
from chemworld.agents.experiment_documents import ExperimentDocumentWorkspace
from chemworld.agents.interaction import AgentDecisionContext
from chemworld.data.logging import to_builtin
from chemworld.legacy.agents.live_llm import JsonCompletionLike, LiveLLMAgent

DOCUMENTED_G2_PROMPT_CONTRACT_VERSION = "chemworld-documented-codex-g2-0.1"

DOCUMENTED_G2_SYSTEM_PROMPT = """You are the operation-level experimental agent in ChemWorld.
Choose exactly one currently legal operation from the compact public decision state. Return only
the JSON object required by the supplied output schema; keep every action parameter directly beside
operation. Never claim hidden identities or simulator state.

Your current working directory is an isolated per-run document workspace. The prompt supplies only
document paths and fingerprints, never the experiment ledger text. The environment-owned JSONL
ledger is authoritative and read-only: inspect it on demand with targeted searches or tail reads,
and never modify, replace, rename, or delete it. The Markdown notebook is yours: read or update it
when useful, keeping concise hypotheses, evidence summaries, comparisons, and next tests rather
than copying the ledger. Do not inspect files outside this workspace.

Do not provide private chain-of-thought. In the final JSON declare only the expected effect,
diagnostic target, information-value forecast, conditional belief-update rule, uncertainty, and
exact action.
"""


class DocumentedCodexG2InterfaceError(RuntimeError):
    """A provider or structured-decision failure that must not reach the world."""


class DocumentedCodexG2Agent(LiveLLMAgent):
    """Run G2 with bounded prompts and a persistent, ownership-separated lab book."""

    name = "documented_codex_g2"

    def __init__(
        self,
        client: Any,
        *,
        documents: ExperimentDocumentWorkspace,
        role_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(client, role_id=role_id, **kwargs)
        self.documents = documents
        self._last_output_schema_sha256: str | None = None
        self._last_output_schema_action_variant_count = 0

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._initial_document_manifest = self.documents.initialize()
        self._last_output_schema_sha256 = None
        self._last_output_schema_action_variant_count = 0

    def act_with_public_view(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]:
        before = self.documents.manifest()
        try:
            action = super().act_with_public_view(context, public_view)
        finally:
            # A model turn is never authorized to change the host ledger.
            self.documents.verify_authoritative_integrity()
        after = self.documents.manifest()
        if self._last_decision is not None:
            self._last_decision["document_memory"] = {
                "before_decision": before,
                "after_decision": after,
                "model_notebook_updated": (
                    before["model_notebook"]["sha256"]
                    != after["model_notebook"]["sha256"]
                ),
                "authoritative_ledger_unchanged_during_model_turn": True,
                "output_schema_sha256": self._last_output_schema_sha256,
                "output_schema_action_variant_count": (
                    self._last_output_schema_action_variant_count
                ),
            }
        return action

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        super().update(action, observation, reward, info)
        if self._last_decision is None:
            return
        event = _authoritative_operation_event(
            step=self._logical_decision_count,
            action=action,
            observation=observation,
            reward=reward,
            info=info,
            decision=self._last_decision,
        )
        after_outcome = self.documents.append_operation(event)
        document_memory = self._last_decision.setdefault("document_memory", {})
        if isinstance(document_memory, dict):
            document_memory["after_environment_outcome"] = after_outcome

    def _build_prompt(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> str:
        payload = json.loads(super()._build_prompt(context, public_view))
        manifest = self.documents.manifest()
        payload["document_memory"] = {
            "workspace_root": ".",
            "ledger_contents_in_prompt": False,
            "authoritative_ledger": manifest["authoritative_ledger"],
            "model_notebook": manifest["model_notebook"],
            "access_contract": {
                "authoritative_ledger": (
                    "environment-owned, append-only, agent read-only; use targeted "
                    "tail/search reads when history can change the decision"
                ),
                "model_notebook": (
                    "agent-owned and writable; keep a concise scientific state, not a "
                    "duplicate operation log"
                ),
                "persistence": "both documents persist across otherwise-ephemeral decisions",
            },
        }
        return self._serialize_extended_prompt(payload)

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
        variants = action_schema.get("anyOf", []) if isinstance(action_schema, dict) else []
        self._last_output_schema_action_variant_count = (
            len(variants) if isinstance(variants, list) else 0
        )
        client = cast(Any, self.client)
        return client.complete_json(
            system_prompt=DOCUMENTED_G2_SYSTEM_PROMPT,
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
        raise DocumentedCodexG2InterfaceError(
            "G2 decision failed before environment execution "
            f"({type(error).__name__}); no fallback action was emitted"
        ) from error

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload.update(
            {
                "prompt_contract_version": DOCUMENTED_G2_PROMPT_CONTRACT_VERSION,
                "prompt_hash": _documented_prompt_hash(),
                "prompt_context_policy": (
                    "bounded_public_state_plus_document_fingerprints_no_ledger_text"
                ),
                "document_memory": {
                    "workspace_isolated": True,
                    "persistent_across_decisions": True,
                    "authoritative_ledger_owner": "environment_host",
                    "authoritative_ledger_agent_writable": False,
                    "model_notebook_owner": "model",
                    "model_notebook_agent_writable": True,
                    "ledger_contents_in_prompt": False,
                },
                "structured_output_policy": (
                    "dynamic_strict_schema_from_current_public_action_affordances"
                ),
                "failure_policy": "fail_closed_before_environment_step",
            }
        )
        return payload

    def method_resource_usage(self) -> dict[str, Any]:
        usage = super().method_resource_usage()
        provenance = usage.get("model_provenance")
        if isinstance(provenance, dict):
            provenance["prompt_hash"] = _documented_prompt_hash()
            parameters = provenance.get("request_parameters")
            if isinstance(parameters, dict):
                parameters.update(
                    {
                        "response_format": "dynamic_strict_json_schema",
                        "document_tools": True,
                        "document_workspace_persistent": True,
                        "ledger_contents_in_prompt": False,
                    }
                )
        return usage

    def document_manifest(self) -> dict[str, Any]:
        """Expose only document fingerprints and relative paths for run summaries."""

        return self.documents.manifest()


def _authoritative_operation_event(
    *,
    step: int,
    action: Mapping[str, Any],
    observation: Mapping[str, Any],
    reward: float,
    info: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    compact_observation = {
        str(key): to_builtin(value)
        for key, value in observation.items()
        if value is not None
    }
    flags = info.get("constraint_flags")
    active_flags = (
        {
            str(key): to_builtin(value)
            for key, value in flags.items()
            if value not in (None, False, 0, "", [], {})
        }
        if isinstance(flags, Mapping)
        else {}
    )
    final_assay = (
        action.get("operation") == "measure"
        and action.get("instrument") == "final_assay"
    )
    experiment_ended = bool(info.get("experiment_ended", False) or final_assay)
    if experiment_ended:
        event_type = "experiment_end"
    elif action.get("operation") == "measure":
        event_type = "measurement_result"
    else:
        event_type = "operation_result"
    outcome = {
        "event_type": event_type,
        "transaction_status": info.get("transaction_status"),
        "operation_type": info.get("operation_type", action.get("operation")),
        "reward": float(reward),
        "observed_keys": to_builtin(info.get("observed_keys", [])),
        "constraint_flags": active_flags,
        "error_message": info.get("error_message"),
        "leaderboard_score": info.get("leaderboard_score"),
        "measurement_cost": info.get("measurement_cost"),
        "sample_consumed": info.get("sample_consumed"),
        "experiment_ended": experiment_ended,
        "state_delta_summary": to_builtin(info.get("state_delta_summary", {})),
        "processed_estimate": to_builtin(info.get("processed_estimate", {})),
        "observation": compact_observation,
    }
    return {
        "schema_version": "chemworld-authoritative-operation-event-0.1",
        "event_id": f"operation-{step:04d}",
        "step": int(step),
        "action": to_builtin(dict(action)),
        "declared_decision": {
            key: to_builtin(decision[key])
            for key in (
                "expected_effect",
                "diagnostic_target",
                "expected_information_gain",
                "belief_update_rule",
                "uncertainty",
                "request_historical_spectrum_id",
            )
            if key in decision
        },
        "outcome": outcome,
    }


def _documented_prompt_hash() -> str:
    material = (
        DOCUMENTED_G2_SYSTEM_PROMPT + "|" + DOCUMENTED_G2_PROMPT_CONTRACT_VERSION
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


__all__ = [
    "DOCUMENTED_G2_PROMPT_CONTRACT_VERSION",
    "DOCUMENTED_G2_SYSTEM_PROMPT",
    "DocumentedCodexG2Agent",
    "DocumentedCodexG2InterfaceError",
]
