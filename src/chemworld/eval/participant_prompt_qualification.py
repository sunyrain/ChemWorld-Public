"""Offline prompt-envelope qualification for mechanism-adaptation participants."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from chemworld.agent_interface import agent_view_bundle, campaign_state
from chemworld.agents.interaction import AgentDecisionContext, build_decision_context
from chemworld.agents.mechanism_adaptation_live_llm import (
    MechanismAdaptationLiveLLMAgent,
    MechanismCandidateSpec,
)
from chemworld.agents.stateful_scientific import (
    MAX_SCIENTIFIC_STATE_JSON_CHARACTERS,
    StatefulScientificMechanismAgent,
)
from chemworld.agents.task_recipes import (
    task_recipe_dimension,
    task_recipe_event_count,
    task_recipe_from_unit_vector,
)
from chemworld.tasks import CONFIRMATORY_BENCHMARK_TASK_IDS, get_task

ScaffoldId = Literal["direct_reactive", "stateful_scientific"]
_QUALIFICATION_CAP = 8_000
_MARGIN = 0.15


class _OfflineClient:
    model = "offline-prompt-qualification"
    thinking = False
    reasoning_effort = None

    def complete_json(self, **_: Any) -> Any:
        raise AssertionError("offline prompt qualification must not call a provider")

    def pricing_snapshot(self) -> dict[str, Any]:
        return {}


def qualify_participant_prompt_envelopes(
    protocol: Mapping[str, Any],
    *,
    task_ids: Sequence[str] = CONFIRMATORY_BENCHMARK_TASK_IDS,
) -> dict[str, Any]:
    """Exercise every legal midpoint stage with saturated public prompt memory."""

    rows: list[dict[str, Any]] = []
    public_view_hashes: dict[tuple[str, str], dict[str, str]] = {}
    for task_id in task_ids:
        contexts = _legal_lifecycle_contexts(str(task_id))
        for scaffold_id in ("direct_reactive", "stateful_scientific"):
            agent = _build_offline_agent(
                protocol,
                task_id=str(task_id),
                scaffold_id=scaffold_id,
            )
            for stage_id, context, public_view in contexts:
                prompt = agent._build_prompt(context, public_view)
                payload = json.loads(prompt)
                checks = _qualification_checks(payload)
                view_hash = _public_environment_view_hash(payload)
                public_view_hashes.setdefault((str(task_id), stage_id), {})[
                    scaffold_id
                ] = view_hash
                rows.append(
                    {
                        "task_id": str(task_id),
                        "scaffold_id": scaffold_id,
                        "stage_id": stage_id,
                        "decision_stage": context.decision_stage,
                        "available_operations": list(context.available_operations),
                        "estimated_tokens": agent._last_prompt_estimated_tokens,
                        "segment_estimates": dict(
                            agent._last_prompt_segment_estimates
                        ),
                        "reduction_steps": list(
                            agent._last_prompt_reduction_steps
                        ),
                        "environment_view_sha256": view_hash,
                        "checks": checks,
                        "passed": all(checks.values())
                        and not agent._last_prompt_reduction_steps,
                    }
                )

    mismatches = [
        {
            "task_id": task_id,
            "stage_id": stage_id,
            "hashes": hashes,
        }
        for (task_id, stage_id), hashes in public_view_hashes.items()
        if len(set(hashes.values())) != 1
    ]
    maxima = _maxima(rows)
    suggested = _suggested_budgets(maxima)
    all_rows_passed = all(row["passed"] for row in rows)
    return {
        "schema_version": "chemworld-participant-prompt-qualification-0.1",
        "qualification_kind": "offline_worst_legal_prompt_fixture",
        "provider_calls": 0,
        "task_ids": [str(item) for item in task_ids],
        "scaffold_ids": ["direct_reactive", "stateful_scientific"],
        "qualification_cap_estimated_tokens": _QUALIFICATION_CAP,
        "headroom_fraction": _MARGIN,
        "fixture_count": len(rows),
        "all_rows_passed": all_rows_passed,
        "same_environment_view_across_scaffolds": not mismatches,
        "environment_view_mismatches": mismatches,
        "observed_maxima": maxima,
        "suggested_development_budgets": suggested,
        "fixtures": rows,
        "interpretation": (
            "Suggested budgets are ceilings derived from the largest unreduced legal "
            "fixture plus 15% headroom; they are execution qualification limits, not "
            "scientific performance thresholds."
        ),
    }


def _legal_lifecycle_contexts(
    task_id: str,
) -> list[tuple[str, AgentDecisionContext, dict[str, Any]]]:
    task = get_task(task_id)
    task_info = task.to_dict()
    recipe = task_recipe_from_unit_vector(
        task_info,
        np.full(task_recipe_dimension(task_info), 0.5),
    )
    actions = [dict(item) for item in recipe["steps"]]
    per_experiment_limit = task_recipe_event_count(task_info) + 6
    env = gym.make(
        "ChemWorld",
        task_id=task_id,
        budget_override=len(actions) + 2,
        episode_mode_override="campaign",
    )
    contexts: list[tuple[str, AgentDecisionContext, dict[str, Any]]] = []
    try:
        observation, _ = env.reset(seed=0)
        current_info: dict[str, Any] = {}
        previous_event_type: str | None = None
        for index in range(len(actions) + 1):
            public_view = agent_view_bundle(env, observation, current_info)
            current_campaign_state = campaign_state(env)
            action_count = 0 if previous_event_type == "experiment_end" else index
            current_campaign_state.update(
                {
                    "diagnostic_actions_used_current_experiment": action_count,
                    "diagnostic_per_experiment_action_limit": per_experiment_limit,
                }
            )
            context = build_decision_context(
                step=index + 1,
                task_info=task_info,
                campaign_state=current_campaign_state,
                public_view=public_view,
                previous_event_type=previous_event_type,
            )
            context = _saturate_public_retrieval(context, task_id=task_id)
            next_operation = (
                str(actions[index]["operation"]) if index < len(actions) else "next_experiment"
            )
            previous_operation = (
                str(actions[index - 1]["operation"]) if index else "reset"
            )
            stage_id = (
                f"{index:02d}-after-{previous_operation}-before-{next_operation}"
            )
            contexts.append((stage_id, context, copy.deepcopy(public_view)))
            if index == len(actions):
                break
            observation, _, _, _, current_info = env.step(actions[index])
            final_assay_ended = (
                actions[index].get("operation") == "measure"
                and actions[index].get("instrument") == "final_assay"
                and current_info.get("leaderboard_score") is not None
            )
            if current_info.get("experiment_ended") or final_assay_ended:
                previous_event_type = "experiment_end"
            elif actions[index].get("operation") == "measure":
                previous_event_type = "measurement_result"
            else:
                previous_event_type = "operation_result"
    finally:
        env.close()
    return contexts


def _build_offline_agent(
    protocol: Mapping[str, Any],
    *,
    task_id: str,
    scaffold_id: ScaffoldId,
) -> MechanismAdaptationLiveLLMAgent:
    task_contract = protocol["task_mechanism_contracts"][task_id]
    definitions = protocol["diagnosis_contract"]["candidate_definitions"]
    specs = tuple(
        MechanismCandidateSpec(
            candidate_id=str(candidate_id),
            public_definition=str(definitions[candidate_id]),
        )
        for candidate_id in task_contract["candidate_ids"]
    )
    agent_type = (
        MechanismAdaptationLiveLLMAgent
        if scaffold_id == "direct_reactive"
        else StatefulScientificMechanismAgent
    )
    agent = agent_type(
        _OfflineClient(),
        role_id=f"offline_{scaffold_id}",
        spectrum_disclosure="assigned",
        response_max_tokens=1_000,
        prompt_token_estimate_cap=_QUALIFICATION_CAP,
        candidate_specs=specs,
        candidate_label_mode="semantic",
        candidate_order_seed=0,
        randomize_candidate_order=False,
    )
    agent.reset(get_task(task_id).to_dict(), 0)
    agent._experiment_memory = _saturated_experiment_memory(task_id)
    agent._recent_decisions = _saturated_recent_decisions(
        tuple(agent._public_to_internal)
    )
    if isinstance(agent, StatefulScientificMechanismAgent):
        agent._scientific_state = agent._validate_scientific_state(
            _saturated_scientific_state(tuple(agent._public_to_internal))
        )
    return agent


def _saturate_public_retrieval(
    context: AgentDecisionContext,
    *,
    task_id: str,
) -> AgentDecisionContext:
    instrument = "hplc" if task_id == "reaction-to-crystallization" else "uvvis"
    catalog = tuple(
        {
            "spectrum_id": f"spectrum-e{index:03d}-s{index * 3:04d}",
            "instrument": instrument,
            "kind": f"{instrument}_spectrum",
            "measurement_step": index * 3,
            "experiment_index": index,
        }
        for index in range(1, 9)
    )
    requested = {
        "has_spectral_packet": True,
        "spectrum_id": catalog[-1]["spectrum_id"],
        "instrument": instrument,
        "kind": f"{instrument}_spectrum",
        "measurement_step": catalog[-1]["measurement_step"],
        "peaks": [
            {
                "assignment": f"public_peak_{index}",
                "center": round(0.5 + index * 0.37, 3),
                "area": float(10_000 - index * 317),
                "fraction": round(0.05 + index * 0.01, 3),
            }
            for index in range(8)
        ],
        "processed_estimate": {
            f"public_metric_{index}": round(0.05 * index, 4)
            for index in range(12)
        },
    }
    return replace(
        context,
        historical_spectrum_catalog=catalog,
        requested_historical_spectrum=requested,
    )


def _saturated_experiment_memory(task_id: str) -> list[dict[str, Any]]:
    operation = (
        "cool_crystallize"
        if task_id == "reaction-to-crystallization"
        else "electrolyze"
    )
    return [
        {
            "experiment_index": index,
            "score": round(0.11 * index, 4),
            "visible_metrics": {
                f"metric_{metric}": round(index * 0.01 + metric * 0.001, 4)
                for metric in range(12)
            },
            "constraint_flags": {
                "high_cost": index == 4,
                "low_selectivity": index == 3,
            },
            "operation_sequence": [
                {"operation": name}
                for name in (
                    "add_solvent",
                    "add_reagent",
                    operation,
                    "measure",
                    operation,
                    "measure",
                    "terminate",
                    "measure",
                )
            ],
        }
        for index in range(1, 5)
    ]


def _saturated_recent_decisions(labels: Sequence[str]) -> list[dict[str, Any]]:
    distribution = {
        label: round(1.0 / len(labels), 8)
        for label in labels
    }
    return [
        {
            "action": {"operation": "measure", "instrument": "hplc"},
            "expected_effect": "e" * 400,
            "diagnostic_target": "d" * 400,
            "uncertainty": 0.65,
            "request_historical_spectrum_id": f"spectrum-e00{index}-s0003",
            "status": "model_decision",
            "mechanism_distribution": distribution,
            "declared_information_value": 0.5,
        }
        for index in range(1, 5)
    ]


def _saturated_scientific_state(labels: Sequence[str]) -> dict[str, Any]:
    distribution = dict.fromkeys(labels, 1.0 / len(labels))
    state: dict[str, Any] = {
        "current_question": "best probe?",
        "campaign_plan": [
            {
                "step": "probe one",
                "purpose": "separate candidates",
                "status": "active",
            },
            {
                "step": "probe two",
                "purpose": "test stability",
                "status": "pending",
            },
        ],
        "mechanism_distribution": distribution,
        "evidence_ledger": [
            {
                "evidence_id": "e1",
                "supports": [],
                "contradicts": [],
                "interpretation": "ambiguous response",
            },
            {
                "evidence_id": "e2",
                "supports": [],
                "contradicts": [],
                "interpretation": "pending replication",
            },
        ],
        "replan_trigger": "r",
        "uncertainty": 0.7,
    }
    padding_targets = (
        (state, "current_question", 140),
        (state["campaign_plan"][0], "step", 160),
        (state["campaign_plan"][0], "purpose", 200),
        (state["campaign_plan"][1], "step", 160),
        (state["campaign_plan"][1], "purpose", 200),
        (state["evidence_ledger"][0], "interpretation", 220),
        (state["evidence_ledger"][1], "interpretation", 220),
        (state, "replan_trigger", 120),
    )
    target_length = MAX_SCIENTIFIC_STATE_JSON_CHARACTERS - 10
    for mapping, key, maximum in padding_targets:
        encoded = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        remaining = max(target_length - len(encoded), 0)
        available = maximum - len(str(mapping[key]))
        mapping[key] = str(mapping[key]) + "x" * min(remaining, available)
    return state


def _qualification_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    decision_state = payload.get("decision_state")
    lifecycle = (
        decision_state.get("lifecycle")
        if isinstance(decision_state, Mapping)
        else None
    )
    required_lifecycle = {
        "experiment_action_count",
        "experiment_action_limit",
        "ordinary_action_slots_remaining",
        "reserved_closeout_slots",
        "experiment_terminated",
        "final_assay_available",
        "closeout_status",
    }
    legal_actions = payload.get("legal_actions")
    return {
        "lifecycle_projection_complete": (
            isinstance(lifecycle, Mapping)
            and required_lifecycle.issubset(lifecycle)
        ),
        "legal_actions_present": (
            isinstance(legal_actions, list) and bool(legal_actions)
        ),
        "candidate_contract_present": isinstance(
            payload.get("mechanism_diagnostic_contract"),
            Mapping,
        ),
        "required_output_shape_present": isinstance(
            payload.get("required_json_shape"),
            Mapping,
        ),
        "raw_arrays_absent": not _contains_raw_array_key(payload),
    }


def _contains_raw_array_key(value: Any) -> bool:
    forbidden = {
        "axis",
        "axes",
        "intensity",
        "intensities",
        "replicate_signals",
        "signal",
        "signals",
        "values",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_raw_array_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_raw_array_key(item) for item in value)
    return False


def _public_environment_view_hash(payload: Mapping[str, Any]) -> str:
    projection = {
        key: payload.get(key)
        for key in ("task", "decision_state", "legal_actions", "on_demand_detail")
    }
    return hashlib.sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _maxima(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scaffold: dict[str, dict[str, int]] = {}
    for scaffold_id in ("direct_reactive", "stateful_scientific"):
        selected = [row for row in rows if row["scaffold_id"] == scaffold_id]
        by_scaffold[scaffold_id] = {
            "environment_view_estimated_tokens": max(
                int(row["segment_estimates"]["environment_view_estimated_tokens"])
                for row in selected
            ),
            "agent_memory_estimated_tokens": max(
                int(row["segment_estimates"]["agent_memory_estimated_tokens"])
                for row in selected
            ),
            "method_contract_estimated_tokens": max(
                int(row["segment_estimates"]["method_contract_estimated_tokens"])
                for row in selected
            ),
            "total_estimated_tokens": max(
                int(row["estimated_tokens"]) for row in selected
            ),
        }
    return {
        "common_environment_view_estimated_tokens": max(
            item["environment_view_estimated_tokens"]
            for item in by_scaffold.values()
        ),
        "by_scaffold": by_scaffold,
    }


def _suggested_budgets(maxima: Mapping[str, Any]) -> dict[str, Any]:
    common_environment = _with_headroom(
        int(maxima["common_environment_view_estimated_tokens"])
    )
    by_scaffold: dict[str, dict[str, int]] = {}
    for scaffold_id, observed in maxima["by_scaffold"].items():
        by_scaffold[str(scaffold_id)] = {
            "environment_view_max_estimated_tokens": common_environment,
            "agent_memory_max_estimated_tokens": _with_headroom(
                int(observed["agent_memory_estimated_tokens"])
            ),
            "per_decision_max_estimated_tokens": _with_headroom(
                int(observed["total_estimated_tokens"])
            ),
        }
    return {
        "same_environment_view_budget_across_scaffolds": True,
        "by_scaffold": by_scaffold,
        "campaign_estimated_token_ceiling_rule": (
            "per_decision_max_estimated_tokens * frozen_operation_limit"
        ),
    }


def _with_headroom(value: int) -> int:
    return int(math.ceil((value * (1.0 + _MARGIN)) / 50.0) * 50)


__all__ = ["qualify_participant_prompt_envelopes"]
