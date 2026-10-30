"""Shared two-phase campaign and causal-feedback diagnostic utilities."""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from chemworld.agents.base import Agent, HistoryRecord
from chemworld.agents.interaction import AgentDecisionContext, InteractionCapabilities
from chemworld.agents.task_recipes import task_recipe_event_count
from chemworld.eval.artifact_paths import repository_relative_reference
from chemworld.eval.runner import run_agent
from chemworld.tasks import get_task

FeedbackCondition = Literal[
    "true_feedback",
    "permuted_feedback",
    "delayed_feedback",
    "critical_measurement_deleted",
]


class PublicViewAgent(Protocol):
    def act_with_public_view(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FeedbackOutcomePacket:
    instrument: str
    experiment_index: int
    observation: dict[str, float | None]
    reward: float
    info: dict[str, Any]


@dataclass(frozen=True)
class FeedbackViewPacket:
    instrument: str
    experiment_index: int
    context: AgentDecisionContext
    public_view: dict[str, Any]


class _ContinuingAgentBase:
    """Preserve agent memory across two environment instances."""

    def __init__(self, delegate: Agent, *, method_id: str) -> None:
        self.delegate = delegate
        self.name = method_id
        self.method_id = method_id
        self._initialized = False
        self._phase = "iid"
        self._experiment_offset = 0
        self._operation_offset = 0
        self._usage_baseline: dict[str, Any] = {}
        self._per_experiment_action_limit: int | None = None
        self._lifecycle_assistance_enabled = False
        self._experiment_action_count = 0
        self._pending_guardrail_audit: dict[str, Any] | None = None
        self._pending_guardrail_trace: list[dict[str, Any]] | None = None
        self.lifecycle_guardrail_log: list[dict[str, Any]] = []

    def configure_lifecycle_guardrail(self, per_experiment_action_limit: int) -> None:
        """Compatibility helper for explicitly assisted diagnostic runs."""

        self.configure_experiment_action_contract(
            per_experiment_action_limit,
            lifecycle_assistance=True,
        )

    def configure_experiment_action_contract(
        self,
        per_experiment_action_limit: int,
        *,
        lifecycle_assistance: bool = False,
    ) -> None:
        """Expose the public limit without silently enabling harness closeout."""

        if per_experiment_action_limit < 2:
            raise ValueError("per_experiment_action_limit must be at least two")
        self._per_experiment_action_limit = int(per_experiment_action_limit)
        self._lifecycle_assistance_enabled = bool(lifecycle_assistance)

    def begin_phase(
        self,
        phase: Literal["iid", "shifted"],
        *,
        experiment_offset: int,
        operation_offset: int,
    ) -> None:
        self._phase = phase
        self._experiment_offset = int(experiment_offset)
        self._operation_offset = int(operation_offset)
        self._experiment_action_count = 0
        self._pending_guardrail_audit = None
        self._pending_guardrail_trace = None

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        if not self._initialized:
            self.delegate.reset(task_info, seed)
            self._initialized = True
        else:
            # Preserve learned state and public experiment memory while updating
            # only the current task contract reference used by official agents.
            if hasattr(self.delegate, "task_info"):
                self.delegate.task_info = task_info
        usage_factory = getattr(self.delegate, "method_resource_usage", None)
        self._usage_baseline = copy.deepcopy(usage_factory()) if callable(usage_factory) else {}

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        self.delegate.update(action, observation, reward, info)
        self._experiment_action_count += 1
        if bool(info.get("experiment_ended")):
            self._experiment_action_count = 0

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        raise RuntimeError("this diagnostic adapter requires a specialized act method")

    def manifest(self) -> dict[str, Any]:
        payload = dict(self.delegate.manifest())
        payload.update(
            {
                "diagnostic_adapter": "two_phase_memory_preserving",
                "diagnostic_method_id": self.method_id,
                "diagnostic_phase": self._phase,
                "environment_instance_reset_between_phases": True,
                "agent_memory_reset_between_phases": False,
                "benchmark_claim_allowed": False,
                "diagnostic_per_experiment_action_limit": (self._per_experiment_action_limit),
                "lifecycle_assistance_enabled": self._lifecycle_assistance_enabled,
                "lifecycle_guardrail": (
                    (
                        "force only currently legal terminate and final_assay actions in "
                        "the final two per-experiment action slots"
                    )
                    if self._lifecycle_assistance_enabled
                    else "disabled; delegate must terminate and request final_assay"
                ),
            }
        )
        return payload

    def method_resource_usage(self) -> dict[str, Any]:
        usage_factory = getattr(self.delegate, "method_resource_usage", None)
        if not callable(usage_factory):
            return {}
        current = copy.deepcopy(usage_factory())
        if not current.get("model_call_count") and not current.get("input_token_count"):
            return current
        for key in (
            "model_call_count",
            "input_token_count",
            "output_token_count",
            "monetary_cost_usd",
            "cpu_time_s",
            "gpu_time_s",
        ):
            if key in current:
                current[key] = max(
                    float(current[key]) - float(self._usage_baseline.get(key, 0.0)),
                    0.0,
                )
                if key not in {
                    "monetary_cost_usd",
                    "cpu_time_s",
                    "gpu_time_s",
                }:
                    current[key] = int(current[key])
        return current

    def interaction_capabilities(self) -> InteractionCapabilities:
        factory = getattr(self.delegate, "interaction_capabilities", None)
        return factory() if callable(factory) else InteractionCapabilities()

    def decision_audit(self) -> dict[str, Any] | None:
        if self._pending_guardrail_audit is not None:
            return copy.deepcopy(self._pending_guardrail_audit)
        factory = getattr(self.delegate, "decision_audit", None)
        return factory() if callable(factory) else None

    def agent_trace(self) -> list[dict[str, Any]]:
        if self._pending_guardrail_trace is not None:
            trace = copy.deepcopy(self._pending_guardrail_trace)
            self._pending_guardrail_trace = None
            return trace
        factory = getattr(self.delegate, "agent_trace", None)
        trace = factory() if callable(factory) else []
        return [dict(item) for item in trace]

    def _guardrail_action(
        self,
        context: AgentDecisionContext,
    ) -> dict[str, Any] | None:
        self._pending_guardrail_audit = None
        self._pending_guardrail_trace = None
        if not self._lifecycle_assistance_enabled:
            return None
        limit = self._per_experiment_action_limit
        if limit is None:
            return None
        action: dict[str, Any] | None = None
        reason = ""
        previous_action = getattr(self, "_previous_action", None)
        closeout_ready = context.decision_stage == "experiment_closeout" or (
            isinstance(previous_action, Mapping)
            and previous_action.get("operation") == "terminate"
            and bool(getattr(self, "_previous_action_committed", False))
        )
        available = set(context.available_operations)
        if (
            closeout_ready
            and "measure" in available
            and self._experiment_action_count >= limit - 2
        ):
            action = {"operation": "measure", "instrument": "final_assay"}
            reason = "final_assay_reserved_slot"
        elif (
            "terminate" in available
            and self._experiment_action_count >= limit - 2
        ):
            action = {"operation": "terminate"}
            reason = "terminate_reserved_slot"
        if action is None:
            return None
        event = {
            "phase": self._phase,
            "experiment_index": int(context.campaign_state.get("experiment_index", 0))
            + self._experiment_offset,
            "actions_used_before_override": self._experiment_action_count,
            "per_experiment_action_limit": limit,
            "decision_stage": context.decision_stage,
            "selected_action": copy.deepcopy(action),
            "reason": reason,
        }
        self.lifecycle_guardrail_log.append(event)
        self._pending_guardrail_audit = {
            "action": copy.deepcopy(action),
            "evidence": [
                "The public per-experiment action limit reached its reserved closeout slots."
            ],
            "hypothesis": "Lifecycle closeout is required for a scoreable experiment.",
            "uncertainty": None,
            "rationale": (
                "The protocol-reserved closeout action replaces an optional model decision."
            ),
            "adaptation_source": "validator",
            "spectrum_interpretation": "",
            "request_historical_spectrum_id": None,
            "status": "provided",
        }
        self._pending_guardrail_trace = [
            {
                "status": "lifecycle_guardrail",
                "action": copy.deepcopy(action),
                "reason": reason,
                "per_experiment_action_limit": limit,
                "actions_used_before_override": self._experiment_action_count,
                "private_reasoning_retained": False,
            }
        ]
        return action

    def consume_historical_spectrum_request(self) -> str | None:
        factory = getattr(self.delegate, "consume_historical_spectrum_request", None)
        return factory() if callable(factory) else None

    def full_method_resource_usage(self) -> dict[str, Any]:
        factory = getattr(self.delegate, "method_resource_usage", None)
        return copy.deepcopy(factory()) if callable(factory) else {}

    def provider_receipts(self) -> list[dict[str, Any]]:
        factory = getattr(self.delegate, "provider_receipts", None)
        receipts = factory() if callable(factory) else []
        return [dict(item) for item in receipts if isinstance(item, Mapping)]


class ContinuingPublicViewAgent(_ContinuingAgentBase):
    """Memory-preserving adapter with optional causal feedback intervention."""

    def __init__(
        self,
        delegate: Agent,
        *,
        method_id: str,
        feedback_condition: FeedbackCondition = "true_feedback",
        critical_instrument: str,
    ) -> None:
        super().__init__(delegate, method_id=method_id)
        if feedback_condition not in {
            "true_feedback",
            "permuted_feedback",
            "delayed_feedback",
            "critical_measurement_deleted",
        }:
            raise ValueError(f"unsupported feedback condition: {feedback_condition}")
        self.feedback_condition = feedback_condition
        self.critical_instrument = critical_instrument
        self._previous_action: dict[str, Any] | None = None
        self._iid_outcomes: dict[str, list[FeedbackOutcomePacket]] = defaultdict(list)
        self._iid_views: dict[str, list[FeedbackViewPacket]] = defaultdict(list)
        self._shifted_outcomes: dict[str, list[FeedbackOutcomePacket]] = defaultdict(list)
        self._shifted_views: dict[str, list[FeedbackViewPacket]] = defaultdict(list)
        self._outcome_cursors: dict[str, int] = defaultdict(int)
        self._view_cursors: dict[str, int] = defaultdict(int)
        self.feedback_intervention_log: list[dict[str, Any]] = []
        self._previous_action_committed = False

    def act_with_public_view(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
    ) -> dict[str, Any]:
        guarded = self._guardrail_action(context)
        if guarded is not None:
            return guarded
        method = getattr(self.delegate, "act_with_public_view", None)
        if not callable(method):
            raise TypeError("delegate does not implement act_with_public_view")
        current_context = self._offset_context(context)
        current_view = copy.deepcopy(public_view)
        instrument = _measurement_instrument(self._previous_action)
        if self._phase == "iid":
            if instrument is not None:
                self._iid_views[instrument].append(
                    FeedbackViewPacket(
                        instrument=instrument,
                        experiment_index=max(
                            int(context.campaign_state.get("experiment_index", 0)) - 1,
                            0,
                        ),
                        context=current_context,
                        public_view=current_view,
                    )
                )
        elif instrument is not None:
            donor = self._view_donor(instrument)
            transformed = self._transform_view_feedback(
                current_context,
                current_view,
                instrument=instrument,
                donor=donor,
            )
            current_context, current_view = transformed
            self._shifted_views[instrument].append(
                FeedbackViewPacket(
                    instrument=instrument,
                    experiment_index=int(context.campaign_state.get("experiment_index", 0)),
                    context=self._offset_context(context),
                    public_view=copy.deepcopy(public_view),
                )
            )
        return method(current_context, current_view)

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        instrument = _measurement_instrument(action)
        packet: FeedbackOutcomePacket | None = None
        if instrument is not None:
            packet = FeedbackOutcomePacket(
                instrument=instrument,
                experiment_index=int(info.get("experiment_index", 0)),
                observation=copy.deepcopy(observation),
                reward=float(reward),
                info=copy.deepcopy(info),
            )
        delivered_observation = observation
        delivered_reward = reward
        delivered_info = info
        if packet is not None and self._phase == "iid":
            self._iid_outcomes[packet.instrument].append(packet)
        elif packet is not None and self._phase == "shifted":
            donor = self._outcome_donor(packet.instrument)
            (
                delivered_observation,
                delivered_reward,
                delivered_info,
            ) = self._transform_outcome_feedback(packet, donor=donor)
            self._shifted_outcomes[packet.instrument].append(packet)
        self.delegate.update(
            action,
            delivered_observation,
            float(delivered_reward),
            delivered_info,
        )
        self._previous_action = dict(action)
        self._previous_action_committed = info.get("transaction_status") == "committed"
        self._experiment_action_count += 1
        if bool(info.get("experiment_ended")):
            self._experiment_action_count = 0

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload.update(
            {
                "feedback_condition": self.feedback_condition,
                "critical_instrument": self.critical_instrument,
                "feedback_intervention_scope": "agent-visible evidence only",
                "environment_state_and_scoring_intervened": False,
            }
        )
        return payload

    def _offset_context(self, context: AgentDecisionContext) -> AgentDecisionContext:
        campaign = copy.deepcopy(context.campaign_state)
        campaign["diagnostic_actions_used_current_experiment"] = self._experiment_action_count
        campaign["diagnostic_per_experiment_action_limit"] = self._per_experiment_action_limit
        for key in ("experiment_index", "final_assay_count"):
            if isinstance(campaign.get(key), int) and not isinstance(campaign.get(key), bool):
                campaign[key] = int(campaign[key]) + self._experiment_offset
        if isinstance(campaign.get("operation_count"), int) and not isinstance(
            campaign.get("operation_count"), bool
        ):
            campaign["operation_count"] = int(campaign["operation_count"]) + self._operation_offset
        return replace(context, campaign_state=campaign)

    def _view_donor(self, instrument: str) -> FeedbackViewPacket | None:
        condition = self.feedback_condition
        if condition == "true_feedback":
            return None
        if condition == "critical_measurement_deleted":
            return None
        if condition == "delayed_feedback":
            candidates = self._shifted_views[instrument]
            return candidates[-1] if candidates else None
        candidates = self._iid_views[instrument]
        if not candidates:
            return None
        cursor = self._view_cursors[instrument]
        self._view_cursors[instrument] += 1
        return candidates[(cursor + 1) % len(candidates)]

    def _outcome_donor(self, instrument: str) -> FeedbackOutcomePacket | None:
        condition = self.feedback_condition
        if condition == "true_feedback":
            return None
        if condition == "critical_measurement_deleted":
            return None
        if condition == "delayed_feedback":
            candidates = self._shifted_outcomes[instrument]
            return candidates[-1] if candidates else None
        candidates = self._iid_outcomes[instrument]
        if not candidates:
            return None
        cursor = self._outcome_cursors[instrument]
        self._outcome_cursors[instrument] += 1
        return candidates[(cursor + 1) % len(candidates)]

    def _transform_view_feedback(
        self,
        context: AgentDecisionContext,
        public_view: dict[str, Any],
        *,
        instrument: str,
        donor: FeedbackViewPacket | None,
    ) -> tuple[AgentDecisionContext, dict[str, Any]]:
        transformed_context, transformed_view, source, donor_experiment = (
            transform_feedback_view(
                condition=self.feedback_condition,
                critical_instrument=self.critical_instrument,
                instrument=instrument,
                context=context,
                public_view=public_view,
                donor=donor,
            )
        )
        if source == "true_feedback":
            return transformed_context, transformed_view
        self.feedback_intervention_log.append(
            {
                "surface": "decision_view",
                "condition": self.feedback_condition,
                "instrument": instrument,
                "source": source,
                "donor_experiment_index": donor_experiment,
            }
        )
        return transformed_context, transformed_view

    def _transform_outcome_feedback(
        self,
        packet: FeedbackOutcomePacket,
        *,
        donor: FeedbackOutcomePacket | None,
    ) -> tuple[dict[str, float | None], float, dict[str, Any]]:
        observation, reward, info, source, donor_experiment = transform_feedback_outcome(
            condition=self.feedback_condition,
            critical_instrument=self.critical_instrument,
            packet=packet,
            donor=donor,
        )
        if source == "true_feedback":
            return observation, reward, info
        self.feedback_intervention_log.append(
            {
                "surface": "agent_update",
                "condition": self.feedback_condition,
                "instrument": packet.instrument,
                "source": source,
                "donor_experiment_index": donor_experiment,
                "true_environment_score_retained_for_evaluation": packet.info.get(
                    "leaderboard_score"
                ),
            }
        )
        return observation, reward, info


def transform_feedback_view(
    *,
    condition: FeedbackCondition,
    critical_instrument: str,
    instrument: str,
    context: AgentDecisionContext,
    public_view: dict[str, Any],
    donor: FeedbackViewPacket | None,
) -> tuple[AgentDecisionContext, dict[str, Any], str, int | None]:
    """Apply one declared Agent-visible view intervention without changing the world."""

    if condition == "true_feedback" or (
        condition == "critical_measurement_deleted" and instrument != critical_instrument
    ):
        return context, public_view, "true_feedback", None
    if donor is None:
        return (
            replace(
                context,
                visible_metrics={},
                latest_spectra={"has_spectral_packet": False},
                uncertainty={},
            ),
            _mask_public_feedback(public_view),
            "masked_no_eligible_donor",
            None,
        )
    return (
        replace(
            context,
            visible_metrics=copy.deepcopy(donor.context.visible_metrics),
            latest_spectra=copy.deepcopy(donor.context.latest_spectra),
            uncertainty=copy.deepcopy(donor.context.uncertainty),
        ),
        _replace_public_feedback(public_view, donor.public_view),
        "iid_permuted" if condition == "permuted_feedback" else "one_event_delayed",
        donor.experiment_index,
    )


def transform_feedback_outcome(
    *,
    condition: FeedbackCondition,
    critical_instrument: str,
    packet: FeedbackOutcomePacket,
    donor: FeedbackOutcomePacket | None,
) -> tuple[dict[str, float | None], float, dict[str, Any], str, int | None]:
    """Apply the matching memory-feedback intervention while retaining lifecycle truth."""

    if condition == "true_feedback" or (
        condition == "critical_measurement_deleted"
        and packet.instrument != critical_instrument
    ):
        return packet.observation, packet.reward, packet.info, "true_feedback", None
    if donor is None:
        observation: dict[str, float | None] = {
            key: None if value is not None else value for key, value in packet.observation.items()
        }
        info = copy.deepcopy(packet.info)
        info["observed_keys"] = []
        info["observed_mask"] = dict.fromkeys(info.get("observed_mask", {}), False)
        info["leaderboard_score"] = None
        return observation, 0.0, info, "masked_no_eligible_donor", None
    return (
        copy.deepcopy(donor.observation),
        donor.reward,
        _replace_outcome_feedback(packet.info, donor.info),
        "iid_permuted" if condition == "permuted_feedback" else "one_event_delayed",
        donor.experiment_index,
    )


def stable_phase_observation_seed(
    *,
    task_id: str,
    world_seed: int,
    observation_pair_id: str,
    phase: str,
) -> int:
    """Derive paired-within-phase and independent-between-phase observation noise."""

    if phase not in {"iid", "shifted"}:
        raise ValueError("phase must be 'iid' or 'shifted'")
    digest = hashlib.sha256(
        (
            f"chemworld-observation:{task_id}:{world_seed}:"
            f"{observation_pair_id}:{phase}"
        ).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def run_two_phase_campaign(
    *,
    task_id: str,
    adapter: _ContinuingAgentBase,
    seed: int,
    pre_change_experiments: int,
    post_change_experiments: int,
    shifted_interventions: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    campaign_id: str,
    observation_pair_id: str | None = None,
    closeout_headroom_per_experiment: int = 6,
    enable_lifecycle_assistance: bool = False,
    minimum_reference_experiments: int | None = None,
    progress_callback: (
        Callable[[str, HistoryRecord, list[dict[str, Any]]], None] | None
    ) = None,
) -> dict[str, Any]:
    """Run IID and shifted phases while preserving only the agent's memory."""

    if closeout_headroom_per_experiment < 0:
        raise ValueError("closeout_headroom_per_experiment must be non-negative")
    if minimum_reference_experiments is not None and minimum_reference_experiments <= 0:
        raise ValueError("minimum_reference_experiments must be positive")
    task = get_task(task_id)
    per_experiment = task_recipe_event_count(task.to_dict())
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    iid_path = root / f"{campaign_id}--iid.jsonl"
    shifted_path = root / f"{campaign_id}--shifted.jsonl"
    per_experiment_limit = per_experiment + closeout_headroom_per_experiment
    iid_limit = per_experiment_limit * pre_change_experiments
    shifted_limit = per_experiment_limit * post_change_experiments
    prompt_cap = getattr(adapter.delegate, "prompt_token_estimate_cap", None)
    response_cap = getattr(adapter.delegate, "response_max_tokens", None)
    adapter.configure_experiment_action_contract(
        per_experiment_limit,
        lifecycle_assistance=enable_lifecycle_assistance,
    )
    noise_pair = campaign_id if observation_pair_id is None else observation_pair_id

    phase_observation_seeds = {
        phase: stable_phase_observation_seed(
            task_id=task_id,
            world_seed=seed,
            observation_pair_id=noise_pair,
            phase=phase,
        )
        for phase in ("iid", "shifted")
    }

    adapter.begin_phase("iid", experiment_offset=0, operation_offset=0)
    iid_history = run_agent(
        env_id=task.env_id,
        agent=adapter,
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=seed,
        observation_seed=phase_observation_seeds["iid"],
        task_id=task.task_id,
        output_path=iid_path,
        budget_override=iid_limit,
        episode_mode_override="campaign",
        method_resource_limits=_diagnostic_resource_limits(
            iid_limit,
            pre_change_experiments,
        ),
        evaluation_policy="vnext_risk_cost",
        step_callback=(
            None
            if progress_callback is None
            else lambda record, trace: progress_callback("iid", record, trace)
        ),
    )
    iid_complete = sum(record.event_type == "experiment_end" for record in iid_history)
    adapter.begin_phase(
        "shifted",
        experiment_offset=iid_complete,
        operation_offset=len(iid_history),
    )
    shifted_history = run_agent(
        env_id=task.env_id,
        agent=adapter,
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=seed,
        observation_seed=phase_observation_seeds["shifted"],
        task_id=task.task_id,
        output_path=shifted_path,
        budget_override=shifted_limit,
        episode_mode_override="campaign",
        method_resource_limits=_diagnostic_resource_limits(
            shifted_limit,
            post_change_experiments,
        ),
        evaluation_policy="vnext_risk_cost",
        world_interventions=tuple(dict(item) for item in shifted_interventions),
        step_callback=(
            None
            if progress_callback is None
            else lambda record, trace: progress_callback("shifted", record, trace)
        ),
    )
    reference_acquisition = assess_reference_acquisition_execution(
        iid_history,
        requested_pre_change_experiments=pre_change_experiments,
        minimum_reference_experiments=(
            pre_change_experiments
            if minimum_reference_experiments is None
            else minimum_reference_experiments
        ),
        lifecycle_assistance_event_count=sum(
            item.get("phase") == "iid" for item in adapter.lifecycle_guardrail_log
        ),
    )
    return {
        "campaign_id": campaign_id,
        "task_id": task_id,
        "method_id": adapter.method_id,
        "seed": seed,
        "iid": summarize_phase(iid_history, path=iid_path, artifact_root=root),
        "shifted": summarize_phase(shifted_history, path=shifted_path, artifact_root=root),
        "shifted_interventions": [dict(item) for item in shifted_interventions],
        "agent_memory_reset_between_phases": False,
        "environment_instance_reset_between_phases": True,
        "phase_observation_seeds": phase_observation_seeds,
        "recipe_event_count": per_experiment,
        "closeout_headroom_per_experiment": closeout_headroom_per_experiment,
        "phase_operation_limit_per_experiment": per_experiment_limit,
        "lifecycle_assistance_enabled": enable_lifecycle_assistance,
        "campaign_token_ceiling": {
            "estimate_kind": "tokenizer_independent_utf8_bytes_divided_by_four",
            "per_decision_prompt_estimated_tokens": prompt_cap,
            "per_decision_output_tokens": response_cap,
            "iid_prompt_estimated_token_ceiling": (
                iid_limit * int(prompt_cap)
                if isinstance(prompt_cap, int)
                else None
            ),
            "shifted_prompt_estimated_token_ceiling": (
                shifted_limit * int(prompt_cap)
                if isinstance(prompt_cap, int)
                else None
            ),
            "total_prompt_estimated_token_ceiling": (
                (iid_limit + shifted_limit) * int(prompt_cap)
                if isinstance(prompt_cap, int)
                else None
            ),
            "interpretation": (
                "Planning ceiling only; provider-reported input/output tokens remain "
                "the accounting source of truth."
            ),
        },
        "reference_acquisition": reference_acquisition,
        "full_method_resources": adapter.full_method_resource_usage(),
        "provider_receipts": adapter.provider_receipts(),
        "feedback_intervention_log": list(getattr(adapter, "feedback_intervention_log", ())),
        "lifecycle_guardrail_log": list(adapter.lifecycle_guardrail_log),
    }


def assess_reference_acquisition_execution(
    history: Sequence[HistoryRecord],
    *,
    requested_pre_change_experiments: int,
    minimum_reference_experiments: int,
    lifecycle_assistance_event_count: int = 0,
) -> dict[str, Any]:
    """Classify reference-phase execution without pretending to certify science.

    This evaluator is intentionally outside the Agent-visible state. Completing the
    old-world phase is necessary but not sufficient for the frozen relation-coverage
    and predictive-reference certificate.
    """

    if requested_pre_change_experiments <= 0:
        raise ValueError("requested_pre_change_experiments must be positive")
    if minimum_reference_experiments <= 0:
        raise ValueError("minimum_reference_experiments must be positive")
    terminal = [record for record in history if record.event_type == "experiment_end"]
    committed_measurements = [
        record
        for record in history
        if record.action.get("operation") == "measure"
        and record.info.get("transaction_status") == "committed"
    ]
    committed_final_assays = [
        record
        for record in committed_measurements
        if record.action.get("instrument") == "final_assay"
    ]
    failed_operations = [
        record
        for record in history
        if record.info.get("transaction_status") != "committed"
    ]
    reasons: list[str] = []
    execution_failed = len(terminal) < requested_pre_change_experiments
    if execution_failed:
        reasons.append("incomplete_prechange_experiment")
    if len(committed_final_assays) < len(terminal):
        reasons.append("invalid_or_missing_final_assay")
    if failed_operations:
        reasons.append("provider_or_action_failure_during_reference")
    if lifecycle_assistance_event_count:
        reasons.append("harness_lifecycle_assistance_used")
    horizon_sufficient = (
        requested_pre_change_experiments >= minimum_reference_experiments
    )
    if not horizon_sufficient:
        reasons.append("development_horizon_below_frozen_reference_minimum")
    if not execution_failed and horizon_sufficient:
        reasons.extend(
            (
                "relation_coverage_not_yet_evaluated",
                "predictive_reference_not_yet_evaluated",
            )
        )

    if execution_failed:
        status = "reference_execution_failed"
        reference_acquisition_failed: bool | None = True
    elif not horizon_sufficient:
        status = "development_horizon_insufficient_for_reference_evaluation"
        reference_acquisition_failed = None
    else:
        status = "pending_relation_and_predictive_reference_evaluation"
        reference_acquisition_failed = None
    return {
        "schema_version": "chemworld-reference-acquisition-execution-0.1",
        "evaluator_visibility": "evaluator_only_not_agent_visible",
        "status": status,
        "reference_acquisition_failed": reference_acquisition_failed,
        "conditional_attribution_eligible": False,
        "requested_pre_change_experiments": requested_pre_change_experiments,
        "minimum_reference_experiments": minimum_reference_experiments,
        "development_horizon_sufficient": horizon_sufficient,
        "complete_prechange_experiment_count": len(terminal),
        "committed_measurement_count": len(committed_measurements),
        "committed_final_assay_count": len(committed_final_assays),
        "failed_operation_count": len(failed_operations),
        "lifecycle_assistance_event_count": lifecycle_assistance_event_count,
        "reason_codes": reasons,
        "interpretation": (
            "Execution completion is only a prerequisite. Relation coverage and "
            "cross-fitted predictive sufficiency remain separate frozen evaluators."
        ),
    }


def summarize_phase(
    history: Sequence[HistoryRecord],
    *,
    path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    terminal = [record for record in history if record.event_type == "experiment_end"]
    scores = [
        float(record.info["leaderboard_score"])
        for record in terminal
        if record.info.get("leaderboard_score") is not None
    ]
    trajectory_reference = repository_relative_reference(path, repository_root=artifact_root)
    trajectory_reference_kind = "artifact_relative"
    return {
        "trajectory_path": trajectory_reference,
        "trajectory_reference_kind": trajectory_reference_kind,
        "trajectory_sha256": _file_sha256(path),
        "operation_count": len(history),
        "complete_experiment_count": len(terminal),
        "scores": scores,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "best_score": max(scores) if scores else None,
        "failure_count": sum(
            record.info.get("transaction_status") != "committed" for record in history
        ),
        "operation_counts": _counts(
            str(record.action.get("operation") or "unknown") for record in history
        ),
    }


def _diagnostic_resource_limits(
    operation_limit: int,
    complete_experiment_limit: int,
) -> dict[str, Any]:
    """Bound live calls generously while retaining exact phase accounting."""

    return {
        "operation_limit": operation_limit,
        "complete_experiment_limit": complete_experiment_limit,
        "wall_time_limit_s": float(operation_limit * 150),
        "model_call_limit": operation_limit * 2,
        "input_token_limit": operation_limit * 12_000,
        "output_token_limit": operation_limit * 1_500,
        "monetary_cost_limit_usd": operation_limit * 0.01,
        "training_environment_step_limit": 0,
        "checkpoint_complete_experiments": tuple(range(1, complete_experiment_limit + 1)),
    }


def _measurement_instrument(action: Mapping[str, Any] | None) -> str | None:
    if not isinstance(action, Mapping) or action.get("operation") != "measure":
        return None
    instrument = action.get("instrument")
    return str(instrument) if isinstance(instrument, str) and instrument else None


def _replace_public_feedback(
    current: Mapping[str, Any],
    donor: Mapping[str, Any],
) -> dict[str, Any]:
    replaced = copy.deepcopy(dict(current))
    donor_copy = copy.deepcopy(dict(donor))
    for key in ("observation", "raw_signal", "processed_estimate", "uncertainty", "lab_report"):
        if key in donor_copy:
            replaced[key] = donor_copy[key]
    current_tool = replaced.get("tool_json")
    donor_tool = donor_copy.get("tool_json")
    if isinstance(current_tool, dict) and isinstance(donor_tool, dict):
        for key in ("observation", "raw_signal", "processed_estimate", "uncertainty", "lab_report"):
            if key in donor_tool:
                current_tool[key] = copy.deepcopy(donor_tool[key])
    return replaced


def _mask_public_feedback(current: Mapping[str, Any]) -> dict[str, Any]:
    masked = copy.deepcopy(dict(current))
    for container in (masked, masked.get("tool_json")):
        if not isinstance(container, dict):
            continue
        observation = container.get("observation")
        if isinstance(observation, dict):
            container["observation"] = dict.fromkeys(observation)
        container["raw_signal"] = {}
        container["processed_estimate"] = {}
        container["uncertainty"] = {}
        report = container.get("lab_report")
        if isinstance(report, dict):
            report["visible_metrics"] = {}
            report["spectra_summary"] = {"has_spectral_packet": False}
    return masked


def _replace_outcome_feedback(
    current: Mapping[str, Any],
    donor: Mapping[str, Any],
) -> dict[str, Any]:
    replaced = copy.deepcopy(dict(current))
    for key in (
        "leaderboard_score",
        "observed_keys",
        "observed_mask",
        "processed_estimate",
        "raw_signal",
        "uncertainty",
        "measurement_cost",
    ):
        if key in donor:
            replaced[key] = copy.deepcopy(donor[key])
    # Lifecycle and transaction validity always come from the true current world.
    for key in (
        "experiment_ended",
        "next_experiment_ready",
        "operation_type",
        "transaction_status",
        "preconditions",
        "constraint_flags",
    ):
        if key in current:
            replaced[key] = copy.deepcopy(current[key])
    return replaced


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "ContinuingPublicViewAgent",
    "FeedbackCondition",
    "FeedbackOutcomePacket",
    "FeedbackViewPacket",
    "assess_reference_acquisition_execution",
    "run_two_phase_campaign",
    "stable_phase_observation_seed",
    "summarize_phase",
    "transform_feedback_outcome",
    "transform_feedback_view",
]
