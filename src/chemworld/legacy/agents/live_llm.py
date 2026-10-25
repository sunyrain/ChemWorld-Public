"""Operation-level live-LLM adapter for the official benchmark runner.

The adapter intentionally owns no provider SDK.  A small JSON client is injected so
the official runner, fake-client tests, and provider-specific launchers all share the
same interaction semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast

from chemworld.agent_interface import experiment_lifecycle_contract
from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.interaction import AgentDecisionContext, InteractionCapabilities
from chemworld.agents.prompt_context import (
    DEFAULT_PROMPT_TOKEN_ESTIMATE_CAP,
    PROMPT_CONTEXT_VERSION,
    PromptBudgetExceededError,
    build_decision_prompt,
    estimate_prompt_segments,
    serialize_prompt_payload,
)
from chemworld.data.logging import to_builtin

SpectrumDisclosure = Literal["assigned", "unassigned", "masked"]

SYSTEM_PROMPT = """You are an operation-level agent in the ChemWorld causal world-model environment.
Use only the compact public decision state and legal action signatures. Choose exactly one
next operation and return one JSON object. The action is flat: put every required field
directly beside operation and never use a nested parameters object. Never claim hidden
identities or simulator state.
Do not provide private chain-of-thought. Declare only the expected effect, diagnostic target,
information-value forecast, conditional belief-update rule, uncertainty, and exact action.
"""

PROMPT_CONTRACT_VERSION = "chemworld-live-llm-operation-json-0.8"
PROMPT_STATE_VERSION = "chemworld-live-llm-public-prompt-state-0.1"

_PURE_SPECTRAL_PACKET_KINDS = {
    "gc_chromatogram",
    "hplc_chromatogram",
    "ir_spectrum",
    "nmr_1h_spectrum",
    "uvvis_spectrum",
}
_SPECTRAL_FIELD_MARKERS = (
    "assignment",
    "channel",
    "chemical_shift",
    "chromatogram",
    "peak",
    "retention",
    "spectra",
    "spectrum",
    "wavelength",
    "wavenumber",
)


class JsonCompletionLike(Protocol):
    payload: dict[str, Any]
    model: str
    usage: dict[str, Any]
    attempts: int


class JsonPlannerClientLike(Protocol):
    model: str

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> JsonCompletionLike: ...


class LiveLLMProviderUnavailableError(OSError):
    """Fail a formal cell without publishing a method-performance terminal."""

    def __init__(self, cause: Exception) -> None:
        super().__init__("live-LLM provider was unavailable before a billable response")
        self.provider_error_type = type(cause).__name__
        self.status_code = getattr(cause, "status_code", None)
        self.retryable = bool(getattr(cause, "retryable", False))
        self.attempts = max(int(getattr(cause, "attempts", 1)), 1)


class LiveLLMAgent(BaseAgent):
    """Use one provider decision per operation in the official runner.

    Provider and output failures are normally retained as invalid ``model_failure``
    actions.  Formal runs may instead fail fast when every provider attempt was rejected
    before billing, so an external outage cannot masquerade as method performance.  The
    harness never repairs an action or performs a terminal assay on the model's behalf.
    """

    name = "live_llm"

    def __init__(
        self,
        client: JsonPlannerClientLike,
        *,
        role_id: str,
        spectrum_disclosure: SpectrumDisclosure = "assigned",
        recent_decision_limit: int = 4,
        experiment_memory_limit: int = 4,
        response_max_tokens: int | None = None,
        prompt_token_estimate_cap: int = DEFAULT_PROMPT_TOKEN_ESTIMATE_CAP,
        environment_view_token_estimate_cap: int | None = None,
        agent_memory_token_estimate_cap: int | None = None,
        fail_fast_on_unbillable_provider_failure: bool = False,
    ) -> None:
        if spectrum_disclosure not in {"assigned", "unassigned", "masked"}:
            raise ValueError(
                "spectrum_disclosure must be assigned, unassigned, or masked"
            )
        if recent_decision_limit <= 0 or experiment_memory_limit <= 0:
            raise ValueError("memory limits must be positive")
        if response_max_tokens is not None and response_max_tokens <= 0:
            raise ValueError("response_max_tokens must be positive")
        if prompt_token_estimate_cap < 500:
            raise ValueError("prompt_token_estimate_cap must be at least 500")
        if (
            environment_view_token_estimate_cap is not None
            and environment_view_token_estimate_cap < 500
        ):
            raise ValueError(
                "environment_view_token_estimate_cap must be at least 500"
            )
        if (
            agent_memory_token_estimate_cap is not None
            and agent_memory_token_estimate_cap < 100
        ):
            raise ValueError("agent_memory_token_estimate_cap must be at least 100")
        self.client = client
        self.role_id = role_id
        self.spectrum_disclosure = spectrum_disclosure
        self.recent_decision_limit = int(recent_decision_limit)
        self.experiment_memory_limit = int(experiment_memory_limit)
        self.response_max_tokens = (
            int(response_max_tokens)
            if response_max_tokens is not None
            else (8000 if bool(getattr(client, "thinking", False)) else 2000)
        )
        self.prompt_token_estimate_cap = int(prompt_token_estimate_cap)
        self.environment_view_token_estimate_cap = (
            None
            if environment_view_token_estimate_cap is None
            else int(environment_view_token_estimate_cap)
        )
        self.agent_memory_token_estimate_cap = (
            None
            if agent_memory_token_estimate_cap is None
            else int(agent_memory_token_estimate_cap)
        )
        self.fail_fast_on_unbillable_provider_failure = bool(
            fail_fast_on_unbillable_provider_failure
        )

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._usage = _empty_usage()
        self._model_call_count = 0
        self._provider_call_accounting_complete = True
        self._provider_token_accounting_complete = True
        self._provider_cache_accounting_complete = True
        self._recent_decisions: list[dict[str, Any]] = []
        self._experiment_memory: list[dict[str, Any]] = []
        self._current_experiment_operations: list[dict[str, Any]] = []
        self._completed_experiment_count = 0
        self._last_decision: dict[str, Any] | None = None
        self._last_context: dict[str, Any] = {}
        self._last_public_view: dict[str, Any] = {}
        self._logical_decision_count = 0
        self._pending_historical_spectrum_id: str | None = None
        self._provider_failure_count = 0
        self._retry_count = 0
        self._system_fingerprints: set[str] = set()
        self._provider_attempt_records: list[dict[str, Any]] = []
        self._last_prompt_estimated_tokens = 0
        self._maximum_prompt_estimated_tokens = 0
        self._last_prompt_segment_estimates: dict[str, int] = {}
        self._maximum_prompt_segment_estimates: dict[str, int] = {}
        self._last_prompt_reduction_steps: tuple[str, ...] = ()
        self._prompt_reduction_decision_count = 0

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        raise RuntimeError("LiveLLMAgent requires the official public-view runner")

    def export_prompt_state(self) -> dict[str, Any]:
        """Export only public prompt memory for controlled same-prefix branching.

        Provider receipts, usage counters, private reasoning, and hidden environment state
        are deliberately excluded. The snapshot is bound to the current compact public
        task contract and can only be restored after ``reset`` on that same contract.
        """

        return {
            "schema_version": PROMPT_STATE_VERSION,
            "task_contract_sha256": _compact_task_contract_sha256(self.task_info),
            "recent_decisions": copy.deepcopy(to_builtin(self._recent_decisions)),
            "completed_experiment_memory": copy.deepcopy(
                to_builtin(self._experiment_memory)
            ),
            "current_experiment_operations": copy.deepcopy(
                to_builtin(self._current_experiment_operations)
            ),
            "completed_experiment_count": self._completed_experiment_count,
            "pending_historical_spectrum_id": self._pending_historical_spectrum_id,
        }

    def restore_prompt_state(self, state: Mapping[str, Any]) -> None:
        """Restore a validated public-memory snapshot for a local causal audit branch."""

        if state.get("schema_version") != PROMPT_STATE_VERSION:
            raise ValueError("unsupported live-LLM prompt-state schema")
        expected = _compact_task_contract_sha256(self.task_info)
        if state.get("task_contract_sha256") != expected:
            raise ValueError("prompt state does not match the active public task contract")
        recent = state.get("recent_decisions")
        experiments = state.get("completed_experiment_memory")
        operations = state.get("current_experiment_operations")
        if (
            not isinstance(recent, list)
            or not isinstance(experiments, list)
            or not isinstance(operations, list)
        ):
            raise ValueError("prompt-state memory fields must be lists")
        if not all(
            isinstance(record, dict)
            for memory in (recent, experiments, operations)
            for record in memory
        ):
            raise ValueError("prompt-state memory entries must be objects")
        completed = state.get("completed_experiment_count")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("completed_experiment_count must be a non-negative integer")
        pending = state.get("pending_historical_spectrum_id")
        if pending is not None and not isinstance(pending, str):
            raise ValueError("pending historical spectrum ID must be a string or null")
        recent_records = cast(list[dict[str, Any]], recent)
        experiment_records = cast(list[dict[str, Any]], experiments)
        operation_records = cast(list[dict[str, Any]], operations)
        self._recent_decisions = copy.deepcopy(recent_records)[-self.recent_decision_limit :]
        self._experiment_memory = copy.deepcopy(experiment_records)[
            -self.experiment_memory_limit :
        ]
        self._current_experiment_operations = copy.deepcopy(operation_records)
        self._completed_experiment_count = completed
        self._pending_historical_spectrum_id = pending
        self._last_decision = None
        self._last_context = {}
        self._last_public_view = {}

    def act_with_public_view(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]:
        self._logical_decision_count += 1
        self._last_context = context.to_dict()
        self._last_public_view = to_builtin(public_view)
        prompt = self._build_prompt(context, public_view)
        if self._last_prompt_reduction_steps:
            self._prompt_reduction_decision_count += 1
        completion: JsonCompletionLike | None = None
        try:
            completion = self._complete_decision(
                prompt=prompt,
                public_view=public_view,
            )
        except Exception as exc:
            attempts = max(int(getattr(exc, "attempts", 1)), 1)
            usage = getattr(exc, "usage", {})
            self._record_provider_usage(attempts, usage if isinstance(usage, dict) else {})
            self._record_provider_attempts(
                getattr(exc, "attempt_records", ()),
                fallback_status="failed",
                fallback_attempts=attempts,
                fallback_usage=usage if isinstance(usage, dict) else {},
            )
            self._provider_failure_count += 1
            self._retry_count += max(attempts - 1, 0)
            if (
                self.fail_fast_on_unbillable_provider_failure
                and _all_provider_attempts_failed_unbillable(exc)
            ):
                raise LiveLLMProviderUnavailableError(exc) from exc
            decision = self._failure_decision(context, exc)
        else:
            self._record_provider_usage(completion.attempts, completion.usage)
            self._record_provider_attempts(
                getattr(completion, "attempt_records", ()),
                fallback_status="succeeded",
                fallback_attempts=completion.attempts,
                fallback_usage=completion.usage,
                fallback_request_id=getattr(completion, "request_id", None),
                fallback_model=completion.model,
            )
            try:
                decision = self._normalize_decision(completion.payload, context=context)
            except Exception as exc:
                self._provider_failure_count += 1
                decision = self._failure_decision(context, exc)
                decision["normalization_error"] = " ".join(str(exc).split())[:300]
            else:
                decision["status"] = "model_decision"
            decision["provider_model"] = str(completion.model)
            decision["provider_attempts"] = int(completion.attempts)
            decision["provider_usage"] = to_builtin(completion.usage)
            decision["provider_request_id"] = getattr(completion, "request_id", None)
            decision["system_fingerprint"] = getattr(
                completion, "system_fingerprint", None
            )
            decision["finish_reason"] = getattr(completion, "finish_reason", None)
            decision["reasoning_content_present"] = bool(
                getattr(completion, "reasoning_content_present", False)
            )
            decision["reasoning_character_count"] = int(
                getattr(completion, "reasoning_character_count", 0)
            )
            fingerprint = decision["system_fingerprint"]
            if isinstance(fingerprint, str) and fingerprint:
                self._system_fingerprints.add(fingerprint)
            self._retry_count += max(int(completion.attempts) - 1, 0)
        self._last_decision = decision
        self._recent_decisions.append(_prompt_memory_decision(decision))
        self._recent_decisions = self._recent_decisions[-self.recent_decision_limit :]
        return dict(decision["action"])

    def _complete_decision(
        self,
        *,
        prompt: str,
        public_view: dict[str, Any],
    ) -> JsonCompletionLike:
        """Provider hook for agents with stricter schemas or isolated document tools."""

        del public_view
        return self.client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=self.response_max_tokens,
        )

    def consume_historical_spectrum_request(self) -> str | None:
        """Consume the previous decision's explicit request at the next operation."""

        spectrum_id = self._pending_historical_spectrum_id
        self._pending_historical_spectrum_id = None
        return spectrum_id

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        if self._last_decision is None:
            return
        outcome = {
            "reward": float(reward),
            "observed_keys": list(info.get("observed_keys", [])),
            "constraint_flags": to_builtin(info.get("constraint_flags", {})),
            "error_message": info.get("error_message"),
            "leaderboard_score": info.get("leaderboard_score"),
            "experiment_ended": bool(info.get("experiment_ended", False)),
            "observation": _compact_observation(observation),
        }
        self._last_decision["outcome"] = outcome
        if self._recent_decisions:
            self._recent_decisions[-1]["outcome"] = outcome
        self._current_experiment_operations.append(
            {
                "action": to_builtin(action),
                "observation": outcome["observation"],
                "constraint_flags": {
                    str(key): bool(value)
                    for key, value in outcome["constraint_flags"].items()
                    if value
                },
                "error_message": outcome["error_message"],
            }
        )
        final_assay = action.get("operation") == "measure" and action.get("instrument") == (
            "final_assay"
        )
        if outcome["experiment_ended"] or final_assay:
            self._completed_experiment_count += 1
            measurement_results = [
                item
                for item in self._current_experiment_operations
                if item["action"].get("operation") == "measure"
            ]
            self._experiment_memory.append(
                {
                    "experiment_index": self._completed_experiment_count,
                    "operation_count": len(self._current_experiment_operations),
                    "operation_sequence": [
                        item["action"] for item in self._current_experiment_operations
                    ],
                    "terminal_action": to_builtin(action),
                    "score": info.get("leaderboard_score"),
                    "visible_metrics": to_builtin(
                        self._last_context.get("visible_metrics", {})
                    ),
                    "constraint_flags": {
                        str(key): bool(value)
                        for key, value in outcome["constraint_flags"].items()
                        if value
                    },
                    "terminal_observation": outcome["observation"],
                    "measurement_results": measurement_results,
                }
            )
            self._experiment_memory = self._experiment_memory[
                -self.experiment_memory_limit :
            ]
            self._recent_decisions = []
            self._current_experiment_operations = []

    def decision_audit(self) -> dict[str, Any] | None:
        if self._last_decision is None:
            return None
        return {
            "action": dict(self._last_decision["action"]),
            "expected_effect": str(self._last_decision["expected_effect"]),
            "diagnostic_target": str(self._last_decision["diagnostic_target"]),
            "expected_information_gain": float(
                self._last_decision["expected_information_gain"]
            ),
            "belief_update_rule": dict(
                self._last_decision["belief_update_rule"]
            ),
            "uncertainty": float(self._last_decision["uncertainty"]),
            "request_historical_spectrum_id": self._last_decision.get(
                "request_historical_spectrum_id"
            ),
            "adaptation_source": str(self._last_decision["adaptation_source"]),
        }

    def agent_trace(self) -> list[dict[str, Any]]:
        """Return only the current decision so JSONL logging remains linear in steps."""

        if self._last_decision is None:
            return []
        return [to_builtin(self._last_decision)]

    def interaction_capabilities(self) -> InteractionCapabilities:
        return InteractionCapabilities(
            decision_scope="operation",
            consumes_intermediate_observations=True,
            consumes_spectra=self.spectrum_disclosure != "masked",
            adapts_within_experiment=True,
            adapts_across_experiments=True,
            emits_structured_decision_audit=True,
        )

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload.update(
            {
                "role_id": self.role_id,
                "requires_online_model": True,
                "provider_model": self.client.model,
                "prompt_contract_version": PROMPT_CONTRACT_VERSION,
                "prompt_context_version": PROMPT_CONTEXT_VERSION,
                "prompt_hash": _prompt_hash(),
                "prompt_token_estimate_cap": self.prompt_token_estimate_cap,
                "environment_view_token_estimate_cap": (
                    self.environment_view_token_estimate_cap
                ),
                "agent_memory_token_estimate_cap": (
                    self.agent_memory_token_estimate_cap
                ),
                "prompt_context_policy": (
                    "decision_first_no_raw_arrays_with_explicit_hard_cap"
                ),
                "spectrum_disclosure": self.spectrum_disclosure,
                "historical_spectrum_access": (
                    "explicit_request_by_public_spectrum_id_delivered_next_decision"
                ),
                "failure_policy": "retain_as_invalid_operation_without_harness_closeout",
                "formal_unbillable_provider_failure_policy": (
                    "raise_resumable_infrastructure_interruption"
                    if self.fail_fast_on_unbillable_provider_failure
                    else "retain_as_invalid_operation"
                ),
                "private_reasoning_retained": False,
            }
        )
        return payload

    def method_resource_usage(self) -> dict[str, Any]:
        pricing_factory = getattr(self.client, "pricing_snapshot", None)
        cost_factory = getattr(self.client, "estimate_cost_usd", None)
        pricing = pricing_factory() if callable(pricing_factory) else None
        pricing_requires_cache_accounting = _pricing_requires_cache_accounting(
            pricing
        )
        monetary_accounting_complete = bool(
            isinstance(pricing, dict)
            and pricing.get("accounting_complete", True) is True
            and callable(cost_factory)
            and (
                not pricing_requires_cache_accounting
                or self._provider_cache_accounting_complete
            )
        )
        provider_usage_accounting_complete = bool(
            self._provider_call_accounting_complete
            and self._provider_token_accounting_complete
        )
        accounting_complete = bool(
            provider_usage_accounting_complete and monetary_accounting_complete
        )
        cost = 0.0
        if monetary_accounting_complete and callable(cost_factory):
            cost = float(cost_factory(self._usage))
        provider = _provider_name(self.client, pricing)
        model_access_date = _model_access_date(pricing)
        return {
            "schema_version": "chemworld-method-resource-usage-0.1",
            "accounting_complete": accounting_complete,
            "provider_usage_accounting_complete": (
                provider_usage_accounting_complete
            ),
            "provider_call_accounting_complete": (
                self._provider_call_accounting_complete
            ),
            "provider_token_accounting_complete": (
                self._provider_token_accounting_complete
            ),
            "provider_cache_accounting_complete": (
                self._provider_cache_accounting_complete
            ),
            "monetary_accounting_complete": monetary_accounting_complete,
            "usage_source": (
                "provider_usage_and_frozen_price_snapshot"
                if accounting_complete
                else "provider_usage_with_pricing_unavailable"
            ),
            "model_call_count": self._model_call_count,
            "input_token_count": int(self._usage["prompt_tokens"]),
            "cached_input_token_count": int(
                self._usage["prompt_cache_hit_tokens"]
            ),
            "uncached_input_token_count": int(
                self._usage["prompt_cache_miss_tokens"]
            ),
            "output_token_count": int(self._usage["completion_tokens"]),
            "monetary_cost_usd": cost,
            "training_environment_step_count": 0,
            "cpu_time_s": 0.0,
            "gpu_time_s": 0.0,
            "model_provenance": {
                "provider": provider,
                "model_id": self.client.model,
                "model_snapshot_or_access_date": model_access_date,
                "prompt_hash": _prompt_hash(),
                "request_parameters": {
                    "response_format": "json_object",
                    "thinking": bool(getattr(self.client, "thinking", False)),
                    "reasoning_effort": (
                        getattr(self.client, "reasoning_effort", None)
                        if bool(getattr(self.client, "thinking", False))
                        else None
                    ),
                    "max_tokens": self.response_max_tokens,
                    "prompt_token_estimate_cap": self.prompt_token_estimate_cap,
                    "environment_view_token_estimate_cap": (
                        self.environment_view_token_estimate_cap
                    ),
                    "agent_memory_token_estimate_cap": (
                        self.agent_memory_token_estimate_cap
                    ),
                    "maximum_prompt_estimated_tokens": (
                        self._maximum_prompt_estimated_tokens
                    ),
                    "maximum_prompt_segment_estimates": dict(
                        self._maximum_prompt_segment_estimates
                    ),
                    "prompt_reduction_decision_count": (
                        self._prompt_reduction_decision_count
                    ),
                    "logical_decisions": self._logical_decision_count,
                    "spectrum_disclosure": self.spectrum_disclosure,
                },
                "tokenizer_or_provider_usage_source": (
                    f"{provider} response.usage"
                ),
                "pricing": pricing,
                "private_reasoning_retained": False,
                "provider_failure_count": self._provider_failure_count,
                "retry_count": self._retry_count,
                "observed_system_fingerprints": sorted(self._system_fingerprints),
            },
        }

    def provider_receipts(self) -> list[dict[str, Any]]:
        """Return attempt-level provider evidence without prompts or private reasoning."""

        pricing_factory = getattr(self.client, "pricing_snapshot", None)
        cost_factory = getattr(self.client, "estimate_cost_usd", None)
        pricing = pricing_factory() if callable(pricing_factory) else {}
        pricing_requires_cache_accounting = _pricing_requires_cache_accounting(
            pricing
        )
        monetary_pricing_complete = bool(
            isinstance(pricing, dict)
            and pricing.get("accounting_complete", True) is True
            and callable(cost_factory)
        )
        pricing_digest = (
            pricing.get("pricing_version_sha256") if isinstance(pricing, dict) else None
        )
        provider = _provider_name(self.client, pricing)
        receipts: list[dict[str, Any]] = []
        for raw in self._provider_attempt_records:
            usage = raw.get("usage")
            normalized = usage if isinstance(usage, dict) else {}
            usage_complete = bool(raw.get("usage_complete", False))
            billable = bool(raw.get("billable", False))
            token_accounting_complete = _provider_token_usage_complete(
                normalized
            )
            cache_accounting_complete = _provider_cache_usage_complete(
                normalized
            )
            monetary_accounting_complete = bool(
                not billable
                or (
                    monetary_pricing_complete
                    and (
                        not pricing_requires_cache_accounting
                        or cache_accounting_complete
                    )
                )
            )
            billed_cost: float | None = None
            if not billable:
                billed_cost = 0.0
            elif (
                monetary_accounting_complete
                and token_accounting_complete
                and callable(cost_factory)
            ):
                billed_cost = float(cost_factory(normalized))
            receipts.append(
                {
                    "schema_version": "chemworld-provider-receipt-0.5",
                    "request_id": raw.get("request_id"),
                    "logical_decision_index": raw["logical_decision_index"],
                    "attempt_index": raw["attempt_index"],
                    "status": raw["status"],
                    "provider": provider,
                    "model_id": raw.get("model_id", self.client.model),
                    "pricing_version_sha256": pricing_digest,
                    "usage_source": raw.get("usage_source", "unavailable"),
                    "usage_complete": usage_complete,
                    "provider_token_accounting_complete": (
                        token_accounting_complete
                    ),
                    "provider_cache_accounting_complete": (
                        cache_accounting_complete
                    ),
                    "billable": billable,
                    "monetary_accounting_complete": (
                        monetary_accounting_complete
                    ),
                    "input_token_count": int(normalized.get("prompt_tokens", 0) or 0),
                    "output_token_count": int(normalized.get("completion_tokens", 0) or 0),
                    "input_cache_hit_token_count": int(
                        normalized.get("prompt_cache_hit_tokens", 0) or 0
                    ),
                    "input_cache_miss_token_count": int(
                        normalized.get("prompt_cache_miss_tokens", 0) or 0
                    ),
                    "billed_cost_usd": billed_cost,
                    "failure_type": raw.get("failure_type"),
                    "failure_detail_type": raw.get("parse_error_type"),
                    "finish_reason": raw.get("finish_reason"),
                }
            )
        return receipts

    def _build_prompt(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> str:
        tool_json = public_view.get("tool_json", {})
        if not isinstance(tool_json, dict):
            tool_json = {}
        supplied_context = context.to_dict()
        supplied_context, tool_json = _condition_spectrum_inputs(
            supplied_context,
            tool_json,
            condition=self.spectrum_disclosure,
        )
        packet = build_decision_prompt(
            task_contract=_compact_task_contract(self.task_info),
            decision_context=supplied_context,
            tool_json=tool_json,
            experiment_memory=self._experiment_memory,
            recent_decisions=self._recent_decisions,
            max_estimated_tokens=self.prompt_token_estimate_cap,
        )
        self._record_prompt_packet(packet)
        return packet.text

    def _serialize_extended_prompt(self, payload: Mapping[str, Any]) -> str:
        """Recheck the hard cap after a diagnostic subclass extends the contract."""

        packet = serialize_prompt_payload(
            payload,
            max_estimated_tokens=self.prompt_token_estimate_cap,
        )
        self._record_prompt_packet(packet)
        return packet.text

    def _record_prompt_packet(self, packet: Any) -> None:
        segments = estimate_prompt_segments(packet.payload)
        environment_cap = self.environment_view_token_estimate_cap
        memory_cap = self.agent_memory_token_estimate_cap
        if (
            environment_cap is not None
            and segments["environment_view_estimated_tokens"] > environment_cap
        ):
            raise PromptBudgetExceededError(
                "environment view estimate "
                f"{segments['environment_view_estimated_tokens']} exceeds cap "
                f"{environment_cap}"
            )
        if (
            memory_cap is not None
            and segments["agent_memory_estimated_tokens"] > memory_cap
        ):
            raise PromptBudgetExceededError(
                "agent memory estimate "
                f"{segments['agent_memory_estimated_tokens']} exceeds cap {memory_cap}"
            )
        self._last_prompt_estimated_tokens = packet.estimated_tokens
        self._maximum_prompt_estimated_tokens = max(
            self._maximum_prompt_estimated_tokens,
            packet.estimated_tokens,
        )
        self._last_prompt_segment_estimates = segments
        for key, value in segments.items():
            self._maximum_prompt_segment_estimates[key] = max(
                self._maximum_prompt_segment_estimates.get(key, 0),
                value,
            )
        self._last_prompt_reduction_steps = tuple(packet.reduction_steps)

    def _normalize_decision(
        self,
        payload: dict[str, Any],
        *,
        context: AgentDecisionContext,
    ) -> dict[str, Any]:
        raw_action = payload.get("action")
        action = raw_action if isinstance(raw_action, dict) else None
        if not action or not action.get("operation"):
            raise ValueError("model decision is missing action.operation")
        expected_effect = str(
            payload.get("expected_effect") or payload.get("hypothesis") or ""
        ).strip()
        diagnostic_target = str(
            payload.get("diagnostic_target") or payload.get("rationale") or ""
        ).strip()
        if not expected_effect or not diagnostic_target:
            raise ValueError(
                "model decision is missing expected_effect or diagnostic_target"
            )
        raw_information_gain = payload.get("expected_information_gain", 0.0)
        if isinstance(raw_information_gain, bool) or not isinstance(
            raw_information_gain,
            int | float,
        ):
            raise ValueError("expected_information_gain must be numeric")
        information_gain = float(raw_information_gain)
        if not 0.0 <= information_gain <= 1.0:
            raise ValueError("expected_information_gain must be in [0, 1]")
        raw_update_rule = payload.get("belief_update_rule")
        if raw_update_rule is None and (
            payload.get("hypothesis") is not None or payload.get("rationale") is not None
        ):
            update_rule = {
                "if_supported": "increase support for the stated hypothesis",
                "if_not_supported": "decrease support and choose a discriminating follow-up",
            }
        elif isinstance(raw_update_rule, Mapping):
            update_rule = {
                "if_supported": str(raw_update_rule.get("if_supported") or "").strip(),
                "if_not_supported": str(
                    raw_update_rule.get("if_not_supported") or ""
                ).strip(),
            }
        else:
            raise ValueError("belief_update_rule must be an object")
        if not all(update_rule.values()):
            raise ValueError("belief_update_rule requires both conditional branches")
        raw_uncertainty = payload.get("uncertainty")
        if isinstance(raw_uncertainty, bool) or not isinstance(raw_uncertainty, int | float):
            raise ValueError("model decision uncertainty must be numeric")
        uncertainty = float(raw_uncertainty)
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("model decision uncertainty must be in [0, 1]")
        raw_request = payload.get("request_historical_spectrum_id")
        if raw_request is None:
            spectrum_request = None
        elif isinstance(raw_request, str) and raw_request.strip():
            spectrum_request = raw_request.strip()
        else:
            raise ValueError(
                "request_historical_spectrum_id must be a non-empty string or null"
            )
        self._pending_historical_spectrum_id = spectrum_request
        return {
            "action": to_builtin(action),
            "expected_effect": expected_effect,
            "diagnostic_target": diagnostic_target,
            "expected_information_gain": information_gain,
            "belief_update_rule": update_rule,
            "uncertainty": uncertainty,
            "request_historical_spectrum_id": spectrum_request,
            "adaptation_source": self._adaptation_source(context),
            "prompt_context_version": PROMPT_CONTEXT_VERSION,
            "prompt_estimated_tokens": int(
                getattr(self, "_last_prompt_estimated_tokens", 0)
            ),
        }

    def _failure_decision(
        self,
        context: AgentDecisionContext,
        error: Exception,
    ) -> dict[str, Any]:
        error_kind = type(error).__name__
        self._pending_historical_spectrum_id = None
        return {
            "action": {"operation": "model_failure"},
            "expected_effect": "No executable expectation was produced.",
            "diagnostic_target": (
                f"Provider or structured-output failure: {error_kind}."
            ),
            "expected_information_gain": 0.0,
            "belief_update_rule": {
                "if_supported": "not available",
                "if_not_supported": "retain failure as an invalid operation",
            },
            "uncertainty": 1.0,
            "request_historical_spectrum_id": None,
            "adaptation_source": self._adaptation_source(context),
            "prompt_context_version": PROMPT_CONTEXT_VERSION,
            "prompt_estimated_tokens": self._last_prompt_estimated_tokens,
            "provider_attempts": max(int(getattr(error, "attempts", 1)), 1),
            "status": "model_failure",
            "error_type": error_kind,
        }

    def _adaptation_source(self, context: AgentDecisionContext) -> str:
        spectra = context.latest_spectra
        requested = context.requested_historical_spectrum
        has_spectrum = bool(
            spectra.get("has_spectral_packet")
            or requested.get("raw_signal")
        )
        if self.spectrum_disclosure != "masked" and has_spectrum:
            return "spectrum"
        if self._experiment_memory:
            return "experiment_memory"
        if context.previous_event_type == "measurement_result":
            return "measurement"
        if context.constraint_flags:
            return "validator"
        return "none"

    def _record_provider_usage(self, attempts: int, usage: dict[str, Any]) -> None:
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            self._provider_call_accounting_complete = False
        self._model_call_count += max(int(attempts), 1)
        if not _provider_token_usage_complete(usage):
            self._provider_token_accounting_complete = False
        if not _provider_cache_usage_complete(usage):
            self._provider_cache_accounting_complete = False
        for key in self._usage:
            value = usage.get(key, 0)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self._usage[key] += value

    def _record_provider_attempts(
        self,
        records: Any,
        *,
        fallback_status: str,
        fallback_attempts: int,
        fallback_usage: dict[str, Any],
        fallback_request_id: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        supplied = (
            [dict(item) for item in records if isinstance(item, dict)]
            if isinstance(records, list | tuple)
            else []
        )
        if not supplied:
            supplied = [
                {
                    "attempt_index": 1,
                    "status": fallback_status,
                    "request_id": fallback_request_id,
                    "model_id": fallback_model or self.client.model,
                    "usage": to_builtin(fallback_usage),
                    "usage_complete": False,
                    "billable": False,
                    "usage_source": "unavailable",
                    "reported_attempt_count": int(fallback_attempts),
                }
            ]
        for raw in supplied:
            raw["logical_decision_index"] = self._logical_decision_count
            self._provider_attempt_records.append(to_builtin(raw))


def _prompt_hash() -> str:
    return hashlib.sha256(
        (SYSTEM_PROMPT + "|" + PROMPT_CONTRACT_VERSION).encode("utf-8")
    ).hexdigest()


def _condition_spectrum_inputs(
    context: dict[str, Any],
    tool_view: dict[str, Any],
    *,
    condition: SpectrumDisclosure,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = to_builtin(context)
    tool = to_builtin(tool_view)
    if condition == "masked":
        masked_tool = _mask_spectral_tool_view(tool)
        provenance = supplied.get("observation_provenance")
        if isinstance(provenance, dict):
            provenance["current_spectral_packet"] = False
        masked_latest: dict[str, Any] = {
            "spectrum_condition": "masked",
            "available": False,
        }
        for key in ("raw_signal", "processed_estimate"):
            value = masked_tool.get(key)
            if isinstance(value, dict) and value:
                masked_latest[key] = value
        supplied["latest_spectra"] = masked_latest
        if supplied.get("requested_historical_spectrum"):
            request = supplied["requested_historical_spectrum"]
            supplied["requested_historical_spectrum"] = {
                "spectrum_id": request.get("spectrum_id"),
                "status": request.get("status"),
                "spectrum_condition": "masked",
                "available": False,
            }
        return supplied, masked_tool
    if condition == "unassigned":
        supplied["latest_spectra"] = _unassign_spectral_fields(
            supplied.get("latest_spectra", {})
        )
        supplied["requested_historical_spectrum"] = _unassign_spectral_fields(
            supplied.get("requested_historical_spectrum", {})
        )
        return supplied, _unassign_spectral_tool_view(tool)
    for key in ("latest_spectra", "requested_historical_spectrum"):
        packet = supplied.get(key)
        if isinstance(packet, dict) and packet:
            packet["spectrum_condition"] = "assigned"
    return supplied, tool


def _mask_spectral_tool_view(tool_json: dict[str, Any]) -> dict[str, Any]:
    """Remove spectral evidence while preserving every non-spectral public field."""

    masked = dict(tool_json)
    raw_signal = masked.get("raw_signal")
    if isinstance(raw_signal, dict):
        kind = str(raw_signal.get("kind", ""))
        if kind in _PURE_SPECTRAL_PACKET_KINDS:
            masked.pop("raw_signal", None)
        else:
            masked["raw_signal"] = _redact_spectral_fields(raw_signal)
    processed = masked.get("processed_estimate")
    if isinstance(processed, dict):
        masked["processed_estimate"] = _redact_spectral_fields(processed)
    lab_report = masked.get("lab_report")
    if isinstance(lab_report, dict):
        public_report = dict(lab_report)
        public_report.pop("spectra_summary", None)
        masked["lab_report"] = public_report
    requested = masked.get("requested_historical_spectrum")
    if isinstance(requested, dict) and requested:
        masked["requested_historical_spectrum"] = {
            "spectrum_id": requested.get("spectrum_id"),
            "status": requested.get("status"),
            "spectrum_condition": "masked",
            "available": False,
        }
    return masked


def _unassign_spectral_tool_view(tool_json: dict[str, Any]) -> dict[str, Any]:
    unassigned = to_builtin(tool_json)
    for key in ("raw_signal", "processed_estimate", "requested_historical_spectrum"):
        if key in unassigned:
            unassigned[key] = _unassign_spectral_fields(unassigned[key])
    lab_report = unassigned.get("lab_report")
    if isinstance(lab_report, dict) and "spectra_summary" in lab_report:
        lab_report["spectra_summary"] = _unassign_spectral_fields(
            lab_report["spectra_summary"]
        )
    return unassigned


def _unassign_spectral_fields(payload: Any) -> Any:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for raw_key, value in payload.items():
            key = str(raw_key)
            normalized = key.lower()
            if normalized in {
                "species_id",
                "analyte_id",
                "group",
                "metadata",
                "identity",
            }:
                continue
            if normalized == "assignments":
                result[key] = []
            elif normalized == "assignment":
                result[key] = "unassigned"
            else:
                result[key] = _unassign_spectral_fields(value)
        if result and (
            "raw_signal" in result
            or "peaks" in result
            or "bands" in result
            or "spectrum_id" in result
        ):
            result["spectrum_condition"] = "unassigned"
        return result
    if isinstance(payload, list):
        return [_unassign_spectral_fields(item) for item in payload]
    return to_builtin(payload)


def _redact_spectral_fields(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            str(key): _redact_spectral_fields(value)
            for key, value in payload.items()
            if not _is_spectral_field(str(key))
        }
    if isinstance(payload, list):
        return [_redact_spectral_fields(item) for item in payload]
    return to_builtin(payload)


def _is_spectral_field(key: str) -> bool:
    normalized = key.lower()
    return any(marker in normalized for marker in _SPECTRAL_FIELD_MARKERS)


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }


def _provider_token_usage_complete(usage: Mapping[str, Any]) -> bool:
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not all(
        isinstance(usage.get(key), int)
        and not isinstance(usage.get(key), bool)
        and int(usage[key]) >= 0
        for key in required
    ):
        return False
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    return bool(
        prompt_tokens > 0
        and int(usage["total_tokens"]) == prompt_tokens + completion_tokens
    )


def _provider_cache_usage_complete(usage: Mapping[str, Any]) -> bool:
    required = (
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    if not all(
        isinstance(usage.get(key), int)
        and not isinstance(usage.get(key), bool)
        and int(usage[key]) >= 0
        for key in required
    ):
        return False
    prompt_tokens = int(usage["prompt_tokens"])
    return bool(
        prompt_tokens > 0
        and int(usage["prompt_cache_hit_tokens"])
        + int(usage["prompt_cache_miss_tokens"])
        == prompt_tokens
    )


def _pricing_requires_cache_accounting(
    pricing: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(pricing, Mapping):
        return False
    return any(
        key in pricing
        for key in (
            "input_cache_hit_per_million_usd",
            "input_cache_miss_per_million_usd",
        )
    )


def _provider_name(
    client: JsonPlannerClientLike,
    pricing: Mapping[str, Any] | None,
) -> str:
    if isinstance(pricing, Mapping):
        provider = pricing.get("provider")
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
    provider = getattr(client, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    name = type(client).__name__
    return name[:-6] if name.endswith("Client") else name


def _model_access_date(pricing: Mapping[str, Any] | None) -> Any:
    if not isinstance(pricing, Mapping):
        return None
    return pricing.get("access_date") or pricing.get("model_access_date")


def _all_provider_attempts_failed_unbillable(error: Exception) -> bool:
    raw_records = getattr(error, "attempt_records", ())
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, str | bytes):
        return False
    records = tuple(raw_records)
    return bool(records) and all(
        isinstance(record, Mapping)
        and record.get("status") == "failed"
        and record.get("billable") is False
        for record in records
    )


def _prompt_memory_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        key: to_builtin(decision[key])
        for key in (
            "action",
            "expected_effect",
            "diagnostic_target",
            "expected_information_gain",
            "belief_update_rule",
            "uncertainty",
            "request_historical_spectrum_id",
            "adaptation_source",
            "status",
        )
        if key in decision
    }


def _compact_task_contract_sha256(task_info: dict[str, Any]) -> str:
    payload = json.dumps(
        _compact_task_contract(task_info),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact_task_contract(task_info: dict[str, Any]) -> dict[str, Any]:
    """Keep user-facing decision facts without resending backend internals."""

    keys = (
        "task_goal",
        "description",
        "task_id",
        "objective",
        "budget",
        "episode_mode",
        "safety_limit",
        "success_metrics",
        "constraints",
        "termination_policy",
        "measurement_policy",
        "experiment_lifecycle",
        "observation_policy",
        "allowed_operations",
        "allowed_instruments",
        "material_information",
        "material_catalog",
        "electrochemical_workflow_mode",
        "scoring_contract",
        "scoring_contract_id",
        "method_budget_contract",
        "observation_keys",
        "scenario_id",
    )
    compact = {
        key: to_builtin(task_info[key])
        for key in keys
        if key in task_info and task_info[key] is not None
    }
    compact.setdefault(
        "experiment_lifecycle",
        experiment_lifecycle_contract(task_info.get("episode_mode")),
    )
    if "task_goal" not in compact and compact.get("description"):
        compact["task_goal"] = compact["description"]
    return compact


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): to_builtin(value)
        for key, value in observation.items()
        if value is not None
    }


__all__ = [
    "PROMPT_CONTRACT_VERSION",
    "PROMPT_STATE_VERSION",
    "SYSTEM_PROMPT",
    "JsonPlannerClientLike",
    "LiveLLMAgent",
    "LiveLLMProviderUnavailableError",
    "SpectrumDisclosure",
]
