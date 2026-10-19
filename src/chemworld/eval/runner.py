"""Official runner for benchmark agents."""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.agent_interface import agent_view_bundle
from chemworld.agents import (
    CodexSubagentReplayAgent,
    GaussianProcessBOAgent,
    GaussianProcessPIAgent,
    GaussianProcessUCBAgent,
    GreedyLocalAgent,
    LatinHypercubeAgent,
    LLMReplayAgent,
    RandomAgent,
    RandomForestEIAgent,
    RandomRecipeAgent,
    SafetyConstrainedBOAgent,
    ScriptedChemistryAgent,
    StructuredGaussianProcessBOAgent,
    StructuredGaussianProcessPIAgent,
    StructuredGaussianProcessUCBAgent,
    StructuredRandomForestEIAgent,
    StructuredSafetyConstrainedBOAgent,
    ToolUsingLLMStubAgent,
)
from chemworld.agents.base import Agent, HistoryRecord
from chemworld.agents.interaction import (
    INTERACTION_CONTRACT_VERSION,
    DecisionAuditRecord,
    build_decision_context,
)
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.logging import TrajectoryLogger, action_payload, observation_to_json
from chemworld.data.submission import git_commit
from chemworld.eval.observation_identifiability import (
    ObservationIdentifiabilityError,
    PublicSpectrumArchive,
)
from chemworld.eval.resource_accounting import (
    MethodResourceLedger,
    MethodResourceLimitError,
    MethodResourceLimits,
    evaluation_resource_limits,
    load_resource_protocol,
)
from chemworld.eval.risk_policy import (
    RiskCostTaskPolicy,
    load_risk_cost_protocol,
)

EvaluationPolicy = Literal["task_contract", "vnext_risk_cost"]

AGENT_REGISTRY: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "lhs": LatinHypercubeAgent,
    "latin_hypercube": LatinHypercubeAgent,
    "greedy": GreedyLocalAgent,
    "greedy_local": GreedyLocalAgent,
    "gp_bo": GaussianProcessBOAgent,
    "gp_pi": GaussianProcessPIAgent,
    "gp_ucb": GaussianProcessUCBAgent,
    "rf_ei": RandomForestEIAgent,
    "safe_gp_bo": SafetyConstrainedBOAgent,
    "structured_gp_bo": StructuredGaussianProcessBOAgent,
    "structured_gp_pi": StructuredGaussianProcessPIAgent,
    "structured_gp_ucb": StructuredGaussianProcessUCBAgent,
    "structured_rf_ei": StructuredRandomForestEIAgent,
    "structured_safe_gp_bo": StructuredSafetyConstrainedBOAgent,
    "random_recipe": RandomRecipeAgent,
    "scripted_chemistry": ScriptedChemistryAgent,
    "scripted_reaction_to_purification": ScriptedChemistryAgent,
    "partition_discovery_heuristic": ScriptedChemistryAgent,
    "heuristic": ScriptedChemistryAgent,
    "tool_using_llm_stub": ToolUsingLLMStubAgent,
    "llm_replay": LLMReplayAgent,
    "codex_subagent_replay": CodexSubagentReplayAgent,
}


def make_agent(name: str) -> Agent:
    if name not in AGENT_REGISTRY:
        allowed = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"Unknown agent={name!r}. Allowed: {allowed}")
    return AGENT_REGISTRY[name]()


def run_agent(
    *,
    env_id: str,
    agent: Agent,
    world_split: str,
    budget: int,
    objective: str,
    seed: int,
    agent_seed: int | None = None,
    observation_seed: int | None = None,
    task_id: str | None = None,
    composition: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    budget_override: int | None = None,
    episode_mode_override: str | None = None,
    step_callback: Callable[[HistoryRecord, list[dict[str, Any]]], None] | None = None,
    env_wrapper: Callable[[gym.Env[Any, Any]], gym.Env[Any, Any]] | None = None,
    method_resource_limits: dict[str, Any] | None = None,
    evaluation_policy: EvaluationPolicy = "task_contract",
    world_interventions: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    safety_limit_override: float | None = None,
    material_information: Mapping[str, Any] | None = None,
    research_brief: Mapping[str, Any] | None = None,
    campaign_resource_card: (
        Mapping[str, Any] | CampaignResourceCard | None
    ) = None,
    electrochemical_material_family_id: str | None = None,
    crystallization_material_family_id: str | None = None,
    electrochemical_workflow_mode: str | None = None,
    scoring_contract_id: str | None = None,
    observation_noise_mode: str | None = None,
    observation_noise_namespace: str | None = None,
    full_process_contract_id: str | None = None,
) -> list[HistoryRecord]:
    """Run one benchmark episode and optionally write a JSONL trajectory."""

    if task_id is not None and composition is not None:
        raise ValueError("task_id and composition are mutually exclusive")
    if evaluation_policy not in {"task_contract", "vnext_risk_cost"}:
        raise ValueError("evaluation_policy must be task_contract or vnext_risk_cost")
    if safety_limit_override is not None:
        if evaluation_policy != "task_contract":
            raise ValueError(
                "safety_limit_override cannot be combined with a named evaluation policy"
            )
        if not 0.0 < float(safety_limit_override) < 1.0:
            raise ValueError("safety_limit_override must be in (0, 1)")
    risk_policy: RiskCostTaskPolicy | None = None
    if evaluation_policy == "vnext_risk_cost":
        if task_id is None:
            raise ValueError("vnext_risk_cost requires a registered serious task_id")
        risk_policy = RiskCostTaskPolicy.from_protocol(
            task_id,
            load_risk_cost_protocol(),
        )

    env_kwargs: dict[str, Any] = {
        "world_split": world_split,
        "budget": budget,
        "objective": objective,
        "seed": seed,
    }
    if observation_seed is not None:
        env_kwargs["observation_seed_override"] = int(observation_seed)
    if observation_noise_mode is not None:
        env_kwargs["observation_noise_mode"] = str(observation_noise_mode)
    if observation_noise_namespace is not None:
        env_kwargs["observation_noise_namespace"] = str(
            observation_noise_namespace
        )
    if full_process_contract_id is not None:
        env_kwargs["full_process_contract_id"] = full_process_contract_id
    if task_id is not None:
        env_kwargs["task_id"] = task_id
    if composition is not None:
        env_kwargs["composition"] = copy.deepcopy(dict(composition))
    if material_information is not None:
        env_kwargs["material_information"] = dict(material_information)
    if research_brief is not None:
        env_kwargs["research_brief"] = copy.deepcopy(dict(research_brief))
    if campaign_resource_card is not None:
        env_kwargs["campaign_resource_card"] = (
            campaign_resource_card
            if isinstance(campaign_resource_card, CampaignResourceCard)
            else copy.deepcopy(dict(campaign_resource_card))
        )
    if electrochemical_material_family_id is not None:
        env_kwargs["electrochemical_material_family_id"] = str(
            electrochemical_material_family_id
        )
    if crystallization_material_family_id is not None:
        env_kwargs["crystallization_material_family_id"] = str(
            crystallization_material_family_id
        )
    if electrochemical_workflow_mode is not None:
        env_kwargs["electrochemical_workflow_mode"] = str(
            electrochemical_workflow_mode
        )
    if scoring_contract_id is not None:
        env_kwargs["scoring_contract_id"] = str(scoring_contract_id)
    if world_interventions:
        env_kwargs["world_interventions"] = list(world_interventions)
    if risk_policy is not None:
        env_kwargs["safety_limit_override"] = risk_policy.risk_limit
    elif safety_limit_override is not None:
        env_kwargs["safety_limit_override"] = float(safety_limit_override)
    if budget_override is not None:
        env_kwargs["budget_override"] = budget_override
    if episode_mode_override is not None:
        env_kwargs["episode_mode_override"] = episode_mode_override
    effective_budget = budget_override if budget_override is not None else budget
    env = gym.make(
        env_id,
        **env_kwargs,
    )
    if env_wrapper is not None:
        env = env_wrapper(env)
    initial_obs, task_info = env.reset(seed=seed)
    if not hasattr(env.unwrapped, "task_info"):
        raise RuntimeError(f"{env_id} does not expose task_info()")
    base_env: Any = env.unwrapped
    task_info = base_env.task_info()
    evaluator_provenance = base_env.evaluator_provenance()
    if risk_policy is not None:
        task_info.update(risk_policy.task_info_overlay())
        task_info["risk_policy_hash"] = risk_policy.policy_hash
    if method_resource_limits is not None:
        task_info["method_budget_contract"] = {
            key: (
                list(method_resource_limits[key])
                if key == "checkpoint_complete_experiments"
                else method_resource_limits[key]
            )
            for key in (
                "operation_limit",
                "complete_experiment_limit",
                "checkpoint_complete_experiments",
            )
            if key in method_resource_limits
        }
    logging_task_info = {**task_info, **evaluator_provenance}

    # The default policy RNG is public and deliberately unrelated to hidden
    # world generation. Formal methods may still supply an explicitly bound
    # private method seed.
    resolved_agent_seed = 0 if agent_seed is None else int(agent_seed)
    agent.reset(task_info, resolved_agent_seed)
    agent_metadata = agent.manifest()
    if agent_seed is not None:
        # Formal method RNG streams are committed privately and must not be copied
        # into public trajectory metadata. The world seed remains in task_info for
        # replay binding, while the policy receives only resolved_agent_seed.
        agent_metadata.pop("seed", None)
        agent_metadata["agent_seed_disclosure"] = "private_committed"
    agent_metadata["git_commit"] = git_commit()
    agent_metadata["evaluation_policy"] = evaluation_policy
    agent_metadata["risk_policy_hash"] = (
        risk_policy.policy_hash if risk_policy is not None else None
    )
    agent_metadata["safety_limit_override"] = (
        float(safety_limit_override) if safety_limit_override is not None else None
    )
    agent_metadata["official_runner_policy"] = {
        "one_agent_decision_per_operation": True,
        "automatic_action_repair": False,
        "automatic_terminate": False,
        "automatic_final_assay": False,
        "failed_or_invalid_actions_retained": True,
    }
    requires_online_model = bool(agent_metadata.get("requires_online_model", False))
    resource_limits = (
        MethodResourceLimits.from_payload(
            method_resource_limits,
            operation_limit=effective_budget,
        )
        if method_resource_limits is not None
        else evaluation_resource_limits(
            load_resource_protocol(),
            operation_limit=effective_budget,
            requires_online_model=requires_online_model,
        )
    )
    resource_ledger = MethodResourceLedger(
        limits=resource_limits,
        requires_online_model=requires_online_model,
    )
    agent_metadata["method_resource_contract_version"] = resource_ledger.snapshot()[
        "schema_version"
    ]

    history: list[HistoryRecord] = []
    spectrum_archive = PublicSpectrumArchive(retrieval_cost=0.0)
    latest_spectrum_id: str | None = None
    experiment_index = 1
    current_observation = initial_obs
    current_info: dict[str, Any] = {}
    previous_event_type: str | None = None
    logger_context = TrajectoryLogger(output_path) if output_path is not None else None
    try:
        logger = logger_context.__enter__() if logger_context is not None else None
        for step in range(1, effective_budget + 1):
            pre_decision_view = agent_view_bundle(env, current_observation, current_info)
            spectrum_request_factory = getattr(agent, "consume_historical_spectrum_request", None)
            requested_spectrum_id = (
                spectrum_request_factory() if callable(spectrum_request_factory) else None
            )
            pre_decision_view, retrieval_event = _attach_spectrum_archive(
                pre_decision_view,
                archive=spectrum_archive,
                latest_spectrum_id=latest_spectrum_id,
                requested_spectrum_id=(
                    str(requested_spectrum_id)
                    if isinstance(requested_spectrum_id, str) and requested_spectrum_id
                    else None
                ),
            )
            decision_context = build_decision_context(
                step=step,
                task_info=task_info,
                campaign_state=base_env.campaign_state(),
                public_view=pre_decision_view,
                previous_event_type=previous_event_type,
            )
            public_view_act = getattr(agent, "act_with_public_view", None)
            context_act = getattr(agent, "act_with_context", None)
            decision_started = perf_counter()
            if callable(public_view_act):
                action = public_view_act(decision_context, pre_decision_view)
            elif callable(context_act):
                action = context_act(decision_context)
            else:
                action = agent.act(history)
            decision_elapsed_s = perf_counter() - decision_started
            normalized_action = action_payload(action)
            if not isinstance(normalized_action, dict):
                raise TypeError("agent action must normalize to a JSON object")
            action = normalized_action
            usage_factory = getattr(agent, "method_resource_usage", None)
            agent_usage = usage_factory() if callable(usage_factory) else {}
            decision_audit_factory = getattr(agent, "decision_audit", None)
            decision_audit = DecisionAuditRecord.from_payload(
                decision_audit_factory() if callable(decision_audit_factory) else None,
                action=action,
            )
            try:
                resource_ledger.record_decision(
                    elapsed_s=decision_elapsed_s,
                    agent_usage=agent_usage,
                )
            except MethodResourceLimitError:
                # The provider decision is real and billable even though its lab
                # action must not execute after the frozen resource cap is crossed.
                # Retain it as a non-mutating trajectory event so receipts, tokens,
                # cost, and decision count remain exactly reconcilable.
                obs_json = observation_to_json(current_observation)
                method_resources = resource_ledger.snapshot()
                resource_limit_info: dict[str, Any] = {
                    "transaction_status": "not_executed_resource_limit",
                    "operation_type": action.get("operation"),
                    "experiment_ended": False,
                    "resource_limit_reached": True,
                    "observed_keys": [],
                    "constraint_flags": {},
                }
                interaction_evidence = {
                    "interaction_contract_version": INTERACTION_CONTRACT_VERSION,
                    "decision_context": decision_context.to_dict(),
                    "decision_audit": decision_audit.to_dict(),
                    "outcome": {
                        "event_type": "resource_limit",
                        "observed_keys": [],
                        "has_spectral_packet": False,
                        "experiment_ended": False,
                        "constraint_flags": {},
                    },
                    "method_resources": method_resources,
                    "historical_spectrum_retrieval": retrieval_event,
                }
                record = HistoryRecord(
                    step=step,
                    action=dict(action),
                    observation=obs_json,
                    reward=0.0,
                    info=resource_limit_info,
                    public_view=pre_decision_view,
                    decision_context=decision_context.to_dict(),
                    decision_audit=decision_audit.to_dict(),
                    event_type="resource_limit",
                    method_resources=method_resources,
                )
                history.append(record)
                agent_trace_factory = getattr(agent, "agent_trace", None)
                agent_trace = agent_trace_factory() if callable(agent_trace_factory) else []
                if logger is not None:
                    logger.log(
                        task_info=logging_task_info,
                        step=step,
                        action=action,
                        observation=obs_json,
                        reward=0.0,
                        terminated=False,
                        truncated=True,
                        info=resource_limit_info,
                        agent_metadata=agent_metadata,
                        explanation=interaction_evidence,
                        agent_view=pre_decision_view,
                        agent_trace=agent_trace,
                        method_resources=method_resources,
                    )
                if step_callback is not None:
                    step_callback(record, agent_trace)
                raise
            observation, reward, terminated, truncated, info = env.step(action)
            obs_json = observation_to_json(observation)
            update_started = perf_counter()
            agent.update(action, obs_json, float(reward), info)
            update_elapsed_s = perf_counter() - update_started
            outcome_resource_error: MethodResourceLimitError | None = None
            post_update_usage = (
                usage_factory() if callable(usage_factory) else {}
            )
            try:
                resource_ledger.reconcile_agent_usage(post_update_usage)
            except MethodResourceLimitError as exc:
                # Some experiment-level provider sessions learn exact token
                # usage only when agent.update closes the session. The physical
                # operation is already real, so retain and log its outcome.
                outcome_resource_error = exc
                info = {**info, "resource_limit_reached": True}
            public_view = agent_view_bundle(env, observation, info)
            packet = _public_spectrum_packet(public_view)
            if packet is not None:
                latest_spectrum_id = f"spectrum-e{experiment_index:03d}-s{step:04d}"
                spectrum_archive.record(
                    latest_spectrum_id,
                    packet,
                    experiment_index=experiment_index,
                    measurement_step=step,
                    measurement_cost=float(info.get("measurement_cost", 0.0) or 0.0),
                )
                public_view, _ = _attach_spectrum_archive(
                    public_view,
                    archive=spectrum_archive,
                    latest_spectrum_id=latest_spectrum_id,
                    requested_spectrum_id=None,
                )
            final_assay_ended = bool(
                terminated
                and action.get("operation") == "measure"
                and action.get("instrument") == "final_assay"
            )
            batch_discarded = bool(info.get("batch_discarded", False))
            experiment_completed = bool(
                info.get("experiment_completed", not batch_discarded)
            )
            if (info.get("experiment_ended") and experiment_completed) or final_assay_ended:
                event_type = "experiment_end"
            elif info.get("experiment_ended") and batch_discarded:
                event_type = "batch_discard"
            elif info.get("operation_type") == "measure":
                event_type = "measurement_result"
            else:
                event_type = "operation_result"
            try:
                resource_ledger.record_outcome(
                    experiment_ended=event_type == "experiment_end",
                    update_elapsed_s=update_elapsed_s,
                )
            except MethodResourceLimitError as exc:
                # The physical transaction already occurred, so log its exact
                # outcome before propagating the resource-budget failure.
                if outcome_resource_error is None:
                    outcome_resource_error = exc
                info = {**info, "resource_limit_reached": True}
            method_resources = resource_ledger.snapshot()
            interaction_evidence = {
                "interaction_contract_version": INTERACTION_CONTRACT_VERSION,
                "decision_context": decision_context.to_dict(),
                "decision_audit": decision_audit.to_dict(),
                "outcome": {
                    "event_type": event_type,
                    "observed_keys": list(info.get("observed_keys", [])),
                    "has_spectral_packet": bool(
                        public_view.get("lab_report", {})
                        .get("spectra_summary", {})
                        .get("has_spectral_packet", False)
                    ),
                    "experiment_ended": bool(
                        info.get("experiment_ended", False) or final_assay_ended
                    ),
                    "experiment_completed": experiment_completed,
                    "batch_discarded": batch_discarded,
                    "constraint_flags": info.get("constraint_flags", {}),
                },
                "method_resources": method_resources,
                "historical_spectrum_retrieval": retrieval_event,
            }
            record = HistoryRecord(
                step=step,
                action=dict(action),
                observation=obs_json,
                reward=float(reward),
                info=info,
                public_view=public_view,
                decision_context=decision_context.to_dict(),
                decision_audit=decision_audit.to_dict(),
                event_type=event_type,
                method_resources=method_resources,
            )
            history.append(record)
            agent_trace_factory = getattr(agent, "agent_trace", None)
            agent_trace = agent_trace_factory() if callable(agent_trace_factory) else []
            if logger is not None:
                logger.log(
                    task_info=logging_task_info,
                    step=step,
                    action=action,
                    observation=obs_json,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata=agent_metadata,
                    explanation=interaction_evidence,
                    agent_view=public_view,
                    agent_trace=agent_trace,
                    method_resources=method_resources,
                )
            if step_callback is not None:
                step_callback(record, agent_trace)
            if outcome_resource_error is not None:
                raise outcome_resource_error
            current_observation = observation
            current_info = info
            previous_event_type = event_type
            if event_type in {"experiment_end", "batch_discard"}:
                experiment_index += 1
            complete_experiment_budget_reached = (
                resource_ledger.limits.complete_experiment_limit is not None
                and resource_ledger.complete_experiment_count
                >= resource_ledger.limits.complete_experiment_limit
            )
            if terminated or truncated or complete_experiment_budget_reached:
                break
    finally:
        primary_error = sys.exc_info()[1]
        agent_close_error: BaseException | None = None
        close_agent = getattr(agent, "close", None)
        if callable(close_agent):
            try:
                close_agent()
            except BaseException as exc:
                # Cleanup must not replace the exception that caused the runner
                # to unwind. On a normal exit, however, close failures remain
                # visible because they can signal an incomplete provider session.
                agent_close_error = exc
        try:
            if logger_context is not None:
                logger_context.__exit__(None, None, None)
        finally:
            env.close()
        if primary_error is None and agent_close_error is not None:
            raise agent_close_error
    return history


def _attach_spectrum_archive(
    public_view: dict[str, Any],
    *,
    archive: PublicSpectrumArchive,
    latest_spectrum_id: str | None,
    requested_spectrum_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach catalog metadata and at most one explicitly requested packet."""

    attached = copy.deepcopy(public_view)
    tool_view = attached.get("tool_json")
    if not isinstance(tool_view, dict):
        tool_view = {}
        attached["tool_json"] = tool_view
    tool_view["historical_spectrum_catalog"] = archive.catalog()
    retrieval_event: dict[str, Any] | None = None
    if requested_spectrum_id is not None:
        try:
            requested = archive.retrieve(requested_spectrum_id)
            tool_view["requested_historical_spectrum"] = {
                "spectrum_id": requested_spectrum_id,
                "status": "retrieved",
                **requested,
            }
        except ObservationIdentifiabilityError:
            tool_view["requested_historical_spectrum"] = {
                "spectrum_id": requested_spectrum_id,
                "status": "unknown_id",
            }
        retrieval_event = archive.ledger()[-1]
    if latest_spectrum_id is not None:
        for report in (
            attached.get("lab_report"),
            tool_view.get("lab_report"),
        ):
            if not isinstance(report, dict):
                continue
            summary = report.get("spectra_summary")
            if isinstance(summary, dict) and summary.get("has_spectral_packet"):
                summary["spectrum_id"] = latest_spectrum_id
    return attached, retrieval_event


def _public_spectrum_packet(public_view: dict[str, Any]) -> dict[str, Any] | None:
    tool_view = public_view.get("tool_json")
    report = public_view.get("lab_report")
    if not isinstance(tool_view, dict) or not isinstance(report, dict):
        return None
    summary = report.get("spectra_summary")
    if not isinstance(summary, dict) or not summary.get("has_spectral_packet"):
        return None
    raw_signal = tool_view.get("raw_signal")
    if not isinstance(raw_signal, dict) or not raw_signal:
        return None
    instrument = report.get("instrument_summary")
    return {
        "schema_version": "chemworld-public-spectrum-packet-0.1",
        "instrument_id": (instrument.get("instrument") if isinstance(instrument, dict) else None),
        "kind": raw_signal.get("kind"),
        "raw_signal": copy.deepcopy(raw_signal),
        "processed_estimate": copy.deepcopy(tool_view.get("processed_estimate", {})),
        "uncertainty": copy.deepcopy(tool_view.get("uncertainty", {})),
    }
