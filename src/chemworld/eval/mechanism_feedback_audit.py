"""Same-prefix local feedback reaction audit for mechanism-adaptation Agents."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.agents.interaction import AgentDecisionContext
from chemworld.data.records import trajectory_value
from chemworld.eval.flagship_diagnostics import (
    FeedbackCondition,
    FeedbackOutcomePacket,
    FeedbackViewPacket,
    transform_feedback_outcome,
    transform_feedback_view,
)
from chemworld.eval.mechanism_adaptation import (
    declared_change_probability,
    feedback_effect_summary,
    js_divergence,
    normalized_distribution,
    operation_aware_action_distance,
)
from chemworld.eval.mechanism_adaptation_execution import build_mechanism_agent
from chemworld.eval.provenance import canonical_json_sha256, git_source_commit
from chemworld.eval.risk_policy import RiskCostTaskPolicy, load_risk_cost_protocol
from chemworld.tasks import get_task

LOCAL_FEEDBACK_REPORT_VERSION = "chemworld-mechanism-local-feedback-audit-0.1"
LOCAL_FEEDBACK_CONDITIONS: tuple[FeedbackCondition, ...] = (
    "true_feedback",
    "permuted_feedback",
    "delayed_feedback",
    "critical_measurement_deleted",
)

_PROMPT_DECISION_KEYS = (
    "action",
    "evidence",
    "spectrum_interpretation",
    "hypothesis",
    "uncertainty",
    "rationale",
    "request_historical_spectrum_id",
    "adaptation_source",
    "status",
)


class _PromptHashingClient:
    """Record prompt digests while forwarding all provider/accounting behavior."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.model = delegate.model
        self.thinking = bool(getattr(delegate, "thinking", False))
        self.reasoning_effort = getattr(delegate, "reasoning_effort", None)
        self.prompt_sha256: list[str] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> Any:
        self.prompt_sha256.append(
            canonical_json_sha256(
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
            )
        )
        return self.delegate.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

    def pricing_snapshot(self) -> dict[str, Any]:
        return dict(self.delegate.pricing_snapshot())

    def estimate_cost_usd(self, usage: dict[str, Any]) -> float:
        return float(self.delegate.estimate_cost_usd(usage))


def build_local_feedback_case(
    iid_records: Sequence[Mapping[str, Any]],
    shifted_records: Sequence[Mapping[str, Any]],
    *,
    critical_instrument: str,
    target_shifted_experiment: int = 2,
) -> dict[str, Any]:
    """Build four branches sharing one exact public prefix before the last feedback."""

    if target_shifted_experiment < 1:
        raise ValueError("target shifted experiment must be positive")
    iid = [copy.deepcopy(dict(item)) for item in iid_records]
    shifted = [copy.deepcopy(dict(item)) for item in shifted_records]
    if not iid or not shifted:
        raise ValueError("local feedback audit requires non-empty IID and shifted traces")
    target_index = _select_target_index(
        shifted,
        critical_instrument=critical_instrument,
        target_shifted_experiment=target_shifted_experiment,
    )
    iid_donor_index = _select_donor_index(iid, critical_instrument=critical_instrument)
    delayed_donor_index = _select_donor_index(
        shifted[:target_index],
        critical_instrument=critical_instrument,
    )
    iid_offset = _completed_experiment_count(iid)
    operation_offset = len(iid)
    action_limit = int(
        shifted[target_index]
        .get("agent_metadata", {})
        .get("diagnostic_per_experiment_action_limit", 0)
    )
    if action_limit <= 0:
        raise ValueError("trajectory lacks the diagnostic per-experiment action limit")

    prefix_state = reconstruct_prompt_memory((*iid, *shifted[: target_index + 1]))
    target_record = shifted[target_index]
    target_packet = _outcome_packet(target_record, critical_instrument)
    current_context = _context_after_measurement(
        shifted[target_index + 1],
        experiment_offset=iid_offset,
        operation_offset=operation_offset,
        experiment_action_count=_actions_in_current_experiment(
            shifted, target_index
        ),
        action_limit=action_limit,
    )
    current_view = copy.deepcopy(dict(trajectory_value(target_record, "agent_view")))
    iid_donor = _feedback_donor(
        iid,
        iid_donor_index,
        critical_instrument=critical_instrument,
        experiment_offset=0,
        operation_offset=0,
        experiment_action_count=_actions_in_current_experiment(iid, iid_donor_index),
        action_limit=action_limit,
    )
    delayed_donor = _feedback_donor(
        shifted,
        delayed_donor_index,
        critical_instrument=critical_instrument,
        experiment_offset=iid_offset,
        operation_offset=operation_offset,
        experiment_action_count=_actions_in_current_experiment(
            shifted, delayed_donor_index
        ),
        action_limit=action_limit,
    )

    branches: dict[str, Any] = {}
    for condition in LOCAL_FEEDBACK_CONDITIONS:
        donor = (
            iid_donor
            if condition == "permuted_feedback"
            else delayed_donor if condition == "delayed_feedback" else None
        )
        donor_view = donor["view"] if donor is not None else None
        donor_outcome = donor["outcome"] if donor is not None else None
        context, view, view_source, view_donor_experiment = transform_feedback_view(
            condition=condition,
            critical_instrument=critical_instrument,
            instrument=critical_instrument,
            context=current_context,
            public_view=current_view,
            donor=donor_view,
        )
        observation, reward, info, outcome_source, outcome_donor_experiment = (
            transform_feedback_outcome(
                condition=condition,
                critical_instrument=critical_instrument,
                packet=target_packet,
                donor=donor_outcome,
            )
        )
        state = copy.deepcopy(prefix_state["after_decision_before_feedback"])
        _apply_outcome_to_prompt_memory(
            state,
            action=dict(target_record["action"]),
            observation=observation,
            reward=reward,
            info=info,
        )
        branches[condition] = {
            "prompt_memory": state,
            "decision_context": context,
            "public_view": view,
            "feedback_source": outcome_source,
            "view_source": view_source,
            "donor_experiment_index": outcome_donor_experiment,
            "view_donor_experiment_index": view_donor_experiment,
        }

    return {
        "pre_feedback_prefix_sha256": canonical_json_sha256(
            prefix_state["after_decision_before_feedback"]
        ),
        "target": {
            "shifted_record_index": target_index,
            "trajectory_step": target_record.get("step"),
            "shifted_experiment": target_shifted_experiment,
            "instrument": critical_instrument,
            "action": copy.deepcopy(target_record["action"]),
        },
        "donors": {
            "permuted_iid_record_index": iid_donor_index,
            "delayed_shifted_record_index": delayed_donor_index,
        },
        "branches": branches,
    }


def run_local_feedback_audit(
    protocol: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    iid_records: Sequence[Mapping[str, Any]],
    shifted_records: Sequence[Mapping[str, Any]],
    llm_methods: Mapping[str, Any],
    repository_root: str | Path,
    method_id: str = "live_llm_b",
    spectrum_disclosure: str = "assigned",
    provider_repeats: int = 3,
    target_shifted_experiment: int = 2,
) -> dict[str, Any]:
    """Execute local branches and subtract within-condition provider variation."""

    if provider_repeats < 2:
        raise ValueError("local feedback audit requires at least two provider repeats")
    task_id = str(row["task_id"])
    critical = "hplc" if task_id == "reaction-to-crystallization" else "uvvis"
    case = build_local_feedback_case(
        iid_records,
        shifted_records,
        critical_instrument=critical,
        target_shifted_experiment=target_shifted_experiment,
    )
    task_info = _campaign_task_info(row, shifted_records[0])
    results: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in LOCAL_FEEDBACK_CONDITIONS
    }
    receipts: list[dict[str, Any]] = []
    for condition in LOCAL_FEEDBACK_CONDITIONS:
        branch = case["branches"][condition]
        for repeat_id in range(provider_repeats):
            agent = build_mechanism_agent(
                protocol,
                row,
                llm_methods=llm_methods,
                method_id=method_id,
                spectrum_disclosure=spectrum_disclosure,
            )
            recording_client = _PromptHashingClient(agent.client)
            agent.client = recording_client
            agent.reset(task_info, int(row["world_seed"]))
            snapshot = agent.export_prompt_state()
            snapshot.update(copy.deepcopy(branch["prompt_memory"]))
            agent.restore_prompt_state(snapshot)
            action = agent.act_with_public_view(
                branch["decision_context"],
                branch["public_view"],
            )
            trace = agent.agent_trace()
            if not trace:
                raise RuntimeError("local feedback branch produced no decision trace")
            decision = trace[-1]
            distribution = normalized_distribution(
                {
                    str(key): value
                    for key, value in decision["mechanism_distribution"].items()
                }
            )
            result = {
                "condition": condition,
                "provider_repeat_id": repeat_id,
                "prompt_sha256": recording_client.prompt_sha256[-1],
                "status": decision.get("status"),
                "mechanism_distribution": distribution,
                "change_probability": declared_change_probability(distribution),
                "action": copy.deepcopy(action),
                "declared_information_value": decision.get(
                    "declared_information_value"
                ),
                "provider_request_id": decision.get("provider_request_id"),
                "provider_model": decision.get("provider_model"),
                "system_fingerprint": decision.get("system_fingerprint"),
            }
            results[condition].append(result)
            receipts.extend(agent.provider_receipts())

    metrics = _local_metrics(results, critical_instrument=critical)
    prompt_hashes = {
        condition: sorted({item["prompt_sha256"] for item in rows})
        for condition, rows in results.items()
    }
    return {
        "schema_version": LOCAL_FEEDBACK_REPORT_VERSION,
        "status": "local_feedback_pilot_complete",
        "formal_benchmark_result": False,
        "confirmatory_gate_pass": False,
        "benchmark_claim_allowed": False,
        "publication_ready": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_json_sha256(protocol),
        "report_source_commit": git_source_commit(Path(repository_root).resolve()),
        "pair_id": row["pair_id"],
        "task_id": task_id,
        "truth_id": row["truth_id"],
        "method_id": method_id,
        "agent_weight_updates_performed": False,
        "provider_repeat_count": provider_repeats,
        "same_prefix_contract": {
            "same_public_history_prefix": True,
            "only_last_feedback_packet_changes": True,
            "pre_feedback_prefix_sha256": case["pre_feedback_prefix_sha256"],
            "target": case["target"],
            "donors": case["donors"],
            "condition_prompt_sha256": prompt_hashes,
            "prompt_is_identical_within_each_condition": all(
                len(values) == 1 for values in prompt_hashes.values()
            ),
            "raw_prompts_retained": False,
        },
        "branch_feedback_provenance": {
            condition: {
                key: value
                for key, value in branch.items()
                if key
                in {
                    "feedback_source",
                    "view_source",
                    "donor_experiment_index",
                    "view_donor_experiment_index",
                }
            }
            for condition, branch in case["branches"].items()
        },
        "decisions": results,
        "metrics": metrics,
        "provider_receipts": receipts,
        "resources": {
            "provider_receipt_count": len(receipts),
            "provider_billed_cost_usd": sum(
                float(item.get("billed_cost_usd") or 0.0) for item in receipts
            ),
        },
        "interpretation": (
            "This local pilot measures whether changing only the last feedback packet changes "
            "the Agent's next declared distribution or action beyond within-condition provider "
            "variation. It does not measure full-campaign utility or pass Gate C."
        ),
    }


def reconstruct_prompt_memory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild the public memory fields produced by LiveLLMAgent from JSONL records."""

    state: dict[str, Any] = {
        "recent_decisions": [],
        "completed_experiment_memory": [],
        "current_experiment_operations": [],
        "completed_experiment_count": 0,
        "pending_historical_spectrum_id": None,
    }
    last_context: dict[str, Any] = {}
    for index, raw in enumerate(records):
        record = dict(raw)
        decision = _model_decision(record)
        if decision is not None:
            memory = {
                key: copy.deepcopy(decision[key])
                for key in _PROMPT_DECISION_KEYS
                if key in decision
            }
            for key in ("mechanism_distribution", "declared_information_value"):
                if key in decision:
                    memory[key] = copy.deepcopy(decision[key])
            state["recent_decisions"].append(memory)
            state["recent_decisions"] = state["recent_decisions"][-4:]
            last_context = copy.deepcopy(
                dict(record.get("explanation", {}).get("decision_context", {}))
            )
            state["pending_historical_spectrum_id"] = decision.get(
                "request_historical_spectrum_id"
            )
        if index == len(records) - 1:
            return {
                "after_decision_before_feedback": copy.deepcopy(state),
                "last_context": last_context,
            }
        _apply_outcome_to_prompt_memory(
            state,
            action=dict(record["action"]),
            observation=_agent_visible_observation(record),
            reward=_agent_visible_reward(record),
            info=_record_info(record),
            last_context=last_context,
        )
    raise ValueError("cannot reconstruct prompt memory from an empty record sequence")


def _apply_outcome_to_prompt_memory(
    state: dict[str, Any],
    *,
    action: dict[str, Any],
    observation: Mapping[str, Any],
    reward: float,
    info: Mapping[str, Any],
    last_context: Mapping[str, Any] | None = None,
) -> None:
    outcome: dict[str, Any] = {
        "reward": float(reward),
        "observed_keys": list(info.get("observed_keys", [])),
        "constraint_flags": copy.deepcopy(dict(info.get("constraint_flags", {}))),
        "error_message": info.get("error_message"),
        "leaderboard_score": info.get("leaderboard_score"),
        "experiment_ended": bool(info.get("experiment_ended", False)),
        "observation": {
            str(key): copy.deepcopy(value)
            for key, value in observation.items()
            if value is not None
        },
    }
    if state["recent_decisions"]:
        state["recent_decisions"][-1]["outcome"] = copy.deepcopy(outcome)
    operation = {
        "action": copy.deepcopy(action),
        "observation": copy.deepcopy(outcome["observation"]),
        "constraint_flags": {
            str(key): bool(value)
            for key, value in outcome["constraint_flags"].items()
            if value
        },
        "error_message": outcome["error_message"],
    }
    state["current_experiment_operations"].append(operation)
    final_assay = action.get("operation") == "measure" and action.get("instrument") == (
        "final_assay"
    )
    if outcome["experiment_ended"] or final_assay:
        state["completed_experiment_count"] += 1
        context = dict(last_context or {})
        state["completed_experiment_memory"].append(
            {
                "experiment_index": state["completed_experiment_count"],
                "operation_count": len(state["current_experiment_operations"]),
                "operation_sequence": [
                    copy.deepcopy(item["action"])
                    for item in state["current_experiment_operations"]
                ],
                "terminal_action": copy.deepcopy(action),
                "score": info.get("leaderboard_score"),
                "visible_metrics": copy.deepcopy(context.get("visible_metrics", {})),
                "constraint_flags": operation["constraint_flags"],
                "terminal_observation": copy.deepcopy(outcome["observation"]),
                "measurement_results": [
                    copy.deepcopy(item)
                    for item in state["current_experiment_operations"]
                    if item["action"].get("operation") == "measure"
                ],
            }
        )
        state["completed_experiment_memory"] = state[
            "completed_experiment_memory"
        ][-4:]
        state["recent_decisions"] = []
        state["current_experiment_operations"] = []


def _local_metrics(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    critical_instrument: str,
) -> dict[str, Any]:
    within: list[float] = []
    between: list[float] = []
    for rows in results.values():
        for left, right in itertools.combinations(rows, 2):
            within.append(
                js_divergence(
                    left["mechanism_distribution"], right["mechanism_distribution"]
                )
            )
    true_rows = results["true_feedback"]
    pair_metrics: dict[str, Any] = {}
    for condition in LOCAL_FEEDBACK_CONDITIONS[1:]:
        rows = results[condition]
        js_values: list[float] = []
        change_shifts: list[float] = []
        operation_changes: list[float] = []
        action_distances: list[float] = []
        for true_row, altered in itertools.product(true_rows, rows):
            distance = js_divergence(
                true_row["mechanism_distribution"],
                altered["mechanism_distribution"],
            )
            js_values.append(distance)
            between.append(distance)
            change_shifts.append(
                float(altered["change_probability"])
                - float(true_row["change_probability"])
            )
            action_distance = operation_aware_action_distance(
                true_row["action"], altered["action"]
            )
            operation_changes.append(float(not action_distance["same_operation"]))
            action_distances.append(
                float(action_distance["operation_distance"])
                if not action_distance["same_operation"]
                else float(action_distance["parameter_distance"] or 0.0)
            )
        pair_metrics[condition] = {
            "mean_declared_distribution_js_shift": _mean(js_values),
            "mean_change_probability_shift_from_true": _mean(change_shifts),
            "next_operation_change_rate": _mean(operation_changes),
            "mean_operation_aware_action_distance": _mean(action_distances),
            "diagnostic_measurement_rate": _diagnostic_rate(
                rows, critical_instrument=critical_instrument
            ),
        }
    return {
        "feedback_effect": feedback_effect_summary(
            within_condition_distances=within,
            between_condition_distances=between,
        ),
        "by_intervention_vs_true": pair_metrics,
        "diagnostic_measurement_rate_by_condition": {
            condition: _diagnostic_rate(rows, critical_instrument=critical_instrument)
            for condition, rows in results.items()
        },
    }


def _campaign_task_info(
    row: Mapping[str, Any],
    shifted_record: Mapping[str, Any],
) -> dict[str, Any]:
    task = get_task(str(row["task_id"]))
    limits = dict(shifted_record.get("method_resources", {}).get("limits", {}))
    operation_limit = int(limits["operation_limit"])
    risk_policy = RiskCostTaskPolicy.from_protocol(
        task.task_id,
        load_risk_cost_protocol(),
    )
    env = gym.make(
        task.env_id,
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=int(row["world_seed"]),
        task_id=task.task_id,
        world_interventions=[dict(item) for item in row["world_interventions"]],
        safety_limit_override=risk_policy.risk_limit,
        budget_override=operation_limit,
        episode_mode_override="campaign",
    )
    try:
        env.reset(seed=int(row["world_seed"]))
        base_env: Any = env.unwrapped
        task_info = dict(base_env.task_info())
    finally:
        env.close()
    task_info.update(risk_policy.task_info_overlay())
    task_info["risk_policy_hash"] = risk_policy.policy_hash
    task_info["method_budget_contract"] = {
        key: copy.deepcopy(limits[key])
        for key in (
            "operation_limit",
            "complete_experiment_limit",
            "checkpoint_complete_experiments",
        )
        if key in limits
    }
    return task_info


def _select_target_index(
    records: Sequence[Mapping[str, Any]],
    *,
    critical_instrument: str,
    target_shifted_experiment: int,
) -> int:
    expected_completed = target_shifted_experiment - 1
    for index in range(len(records) - 1):
        record = records[index]
        decision = _model_decision(record)
        if (
            _measurement_instrument(record) == critical_instrument
            and int(record.get("method_resources", {}).get("complete_experiment_count", 0))
            == expected_completed
            and decision is not None
            and _model_decision(records[index + 1]) is not None
            and not decision.get("request_historical_spectrum_id")
        ):
            return index
    raise ValueError("no eligible same-prefix target feedback packet was found")


def _select_donor_index(
    records: Sequence[Mapping[str, Any]],
    *,
    critical_instrument: str,
) -> int:
    for index in range(len(records) - 1, -1, -1):
        decision = _model_decision(records[index])
        if (
            _measurement_instrument(records[index]) == critical_instrument
            and decision is not None
            and not decision.get("request_historical_spectrum_id")
            and index + 1 < len(records)
        ):
            return index
    raise ValueError(f"no eligible {critical_instrument} feedback donor was found")


def _feedback_donor(
    records: Sequence[Mapping[str, Any]],
    index: int,
    *,
    critical_instrument: str,
    experiment_offset: int,
    operation_offset: int,
    experiment_action_count: int,
    action_limit: int,
) -> dict[str, Any]:
    outcome = _outcome_packet(records[index], critical_instrument)
    context = _context_after_measurement(
        records[index + 1],
        experiment_offset=experiment_offset,
        operation_offset=operation_offset,
        experiment_action_count=experiment_action_count,
        action_limit=action_limit,
    )
    return {
        "outcome": outcome,
        "view": FeedbackViewPacket(
            instrument=critical_instrument,
            experiment_index=outcome.experiment_index,
            context=context,
            public_view=copy.deepcopy(dict(trajectory_value(records[index], "agent_view"))),
        ),
    }


def _outcome_packet(
    record: Mapping[str, Any],
    instrument: str,
) -> FeedbackOutcomePacket:
    return FeedbackOutcomePacket(
        instrument=instrument,
        experiment_index=int(record.get("experiment_index", 0)),
        observation=_agent_visible_observation(record),
        reward=_agent_visible_reward(record),
        info=_record_info(record),
    )


def _record_info(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "leaderboard_score",
        "observed_keys",
        "observed_mask",
        "processed_estimate",
        "raw_signal",
        "uncertainty",
        "measurement_cost",
        "operation_type",
        "transaction_status",
        "preconditions",
        "constraint_flags",
        "error_message",
    )
    info = {key: copy.deepcopy(record[key]) for key in keys if key in record}
    if "evaluation_outcome" in record:
        info["leaderboard_score"] = trajectory_value(record, "leaderboard_score", None)
    outcome = record.get("explanation", {}).get("outcome", {})
    info["experiment_ended"] = outcome.get("event_type") == "experiment_end"
    return info


def _agent_visible_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("schema_version") == "chemworld-trajectory-0.3":
        layer = record.get("agent_visible_observation")
        if not isinstance(layer, Mapping) or not isinstance(layer.get("observation"), Mapping):
            raise ValueError("trajectory v0.3 lacks an Agent-visible observation")
        return copy.deepcopy(dict(layer["observation"]))
    raw = (
        record["agent_visible_observation"]
        if "agent_visible_observation" in record
        else trajectory_value(record, "observation", {})
    )
    if not isinstance(raw, Mapping):
        raise ValueError("trajectory record lacks an Agent-visible observation")
    nested = raw.get("observation")
    if isinstance(nested, Mapping):
        raw = nested
    return copy.deepcopy(dict(raw))


def _agent_visible_reward(record: Mapping[str, Any]) -> float:
    layer = record.get("agent_visible_observation")
    if isinstance(layer, Mapping) and layer.get("observed_reward") is not None:
        return float(layer["observed_reward"])
    if record.get("schema_version") == "chemworld-trajectory-0.3":
        raise ValueError("trajectory v0.3 lacks an Agent-visible reward")
    reward = (
        record["observed_reward"]
        if "observed_reward" in record
        else trajectory_value(record, "reward", 0.0)
    )
    return float(reward or 0.0)


def _model_decision(record: Mapping[str, Any]) -> dict[str, Any] | None:
    trace = record.get("agent_trace")
    if not isinstance(trace, list) or not trace or not isinstance(trace[-1], Mapping):
        return None
    decision = dict(trace[-1])
    return decision if decision.get("status") in {"model_decision", "model_failure"} else None


def _measurement_instrument(record: Mapping[str, Any]) -> str | None:
    action = record.get("action")
    if not isinstance(action, Mapping) or action.get("operation") != "measure":
        return None
    instrument = action.get("instrument")
    return str(instrument) if instrument is not None else None


def _context_after_measurement(
    next_record: Mapping[str, Any],
    *,
    experiment_offset: int,
    operation_offset: int,
    experiment_action_count: int,
    action_limit: int,
) -> AgentDecisionContext:
    payload = next_record.get("explanation", {}).get("decision_context")
    if not isinstance(payload, Mapping):
        raise ValueError("next trajectory record lacks a decision context")
    campaign = copy.deepcopy(dict(payload.get("campaign_state", {})))
    campaign["diagnostic_actions_used_current_experiment"] = experiment_action_count
    campaign["diagnostic_per_experiment_action_limit"] = action_limit
    for key in ("experiment_index", "final_assay_count"):
        if isinstance(campaign.get(key), int) and not isinstance(campaign.get(key), bool):
            campaign[key] = int(campaign[key]) + experiment_offset
    if isinstance(campaign.get("operation_count"), int) and not isinstance(
        campaign.get("operation_count"), bool
    ):
        campaign["operation_count"] = int(campaign["operation_count"]) + operation_offset
    return AgentDecisionContext(
        step=int(payload["step"]),
        task_id=None if payload.get("task_id") is None else str(payload["task_id"]),
        decision_stage=str(payload["decision_stage"]),
        campaign_state=campaign,
        visible_metrics=copy.deepcopy(dict(payload.get("visible_metrics", {}))),
        latest_spectra=copy.deepcopy(dict(payload.get("latest_spectra", {}))),
        uncertainty=copy.deepcopy(dict(payload.get("uncertainty", {}))),
        constraint_flags=copy.deepcopy(dict(payload.get("constraint_flags", {}))),
        available_operations=tuple(str(item) for item in payload["available_operations"]),
        previous_event_type=(
            None
            if payload.get("previous_event_type") is None
            else str(payload["previous_event_type"])
        ),
        historical_spectrum_catalog=tuple(
            copy.deepcopy(dict(item))
            for item in payload.get("historical_spectrum_catalog", [])
            if isinstance(item, Mapping)
        ),
        requested_historical_spectrum=copy.deepcopy(
            dict(payload.get("requested_historical_spectrum", {}))
        ),
    )


def _completed_experiment_count(records: Sequence[Mapping[str, Any]]) -> int:
    return max(
        int(item.get("method_resources", {}).get("complete_experiment_count", 0))
        for item in records
    )


def _actions_in_current_experiment(
    records: Sequence[Mapping[str, Any]],
    inclusive_index: int,
) -> int:
    count = 0
    for record in reversed(records[: inclusive_index + 1]):
        count += 1
        outcome = record.get("explanation", {}).get("outcome", {})
        if outcome.get("event_type") == "experiment_end" and count > 1:
            return count - 1
    return count


def _diagnostic_rate(
    rows: Sequence[Mapping[str, Any]],
    *,
    critical_instrument: str,
) -> float:
    return sum(
        item["action"].get("operation") == "measure"
        and item["action"].get("instrument") == critical_instrument
        for item in rows
    ) / len(rows)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


__all__ = [
    "LOCAL_FEEDBACK_CONDITIONS",
    "LOCAL_FEEDBACK_REPORT_VERSION",
    "build_local_feedback_case",
    "reconstruct_prompt_memory",
    "run_local_feedback_audit",
]
