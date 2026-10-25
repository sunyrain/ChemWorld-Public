"""Deterministic execution for fixed-world scientific optimization."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from chemworld.agents.electrochemical_single_stage import (
    ELECTROCHEMICAL_SINGLE_STAGE_EVENT_COUNT,
)
from chemworld.agents.static_optimization import (
    STATIC_OPTIMIZATION_INTERFACE_VERSION,
    StaticOptimizationAgent,
    StaticOptimizationPlan,
    compile_static_optimization_plan,
)
from chemworld.agents.task_recipes import task_recipe_event_count
from chemworld.data.logging import observation_to_json, to_builtin
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.eval.static_optimization_protocol import (
    PREDICTIVE_CALL_INTEGRATED,
    exploration_experiment_count,
    static_optimization_crystallization_material_family_id,
    static_optimization_material_family_id,
    static_optimization_predictive_call_policy,
    static_optimization_scoring_contract_id,
    static_optimization_workflow_mode,
)
from chemworld.legacy.agents.live_llm import JsonPlannerClientLike
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE,
    normalize_electrochemical_workflow_mode,
)
from chemworld.providers.deepseek import DeepSeekClient
from chemworld.tasks import get_task
from chemworld.world.scoring import TASK_DERIVED_SCORING_CONTRACT, TaskScoringContract

STATIC_OPTIMIZATION_RESULT_VERSION = "chemworld-static-optimization-result-0.1-s0-dev"

_PUBLIC_STEP_INFO_KEYS = (
    "step",
    "remaining_budget",
    "experiment_index",
    "operation_type",
    "instrument",
    "observed_keys",
    "processed_estimate",
    "raw_signal",
    "uncertainty",
    "constraint_flags",
    "cost",
    "measurement_cost",
    "sample_consumed",
    "transaction_status",
    "experiment_ended",
    "leaderboard_score",
)


@dataclass(frozen=True)
class StaticOptimizationResult:
    task_id: str
    experiment_index: int
    plan: StaticOptimizationPlan
    executed_steps: tuple[dict[str, Any], ...]
    measurement_evidence: tuple[dict[str, Any], ...]
    terminal_summary: dict[str, Any]
    completed: bool
    peak_safety_risk: float
    compiled_operation_count: int
    runtime_operation_cap: int
    runtime_margin_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATIC_OPTIMIZATION_RESULT_VERSION,
            "interface_version": STATIC_OPTIMIZATION_INTERFACE_VERSION,
            "task_id": self.task_id,
            "experiment_index": self.experiment_index,
            "plan": self.plan.to_dict(),
            "executed_steps": copy.deepcopy(list(self.executed_steps)),
            "measurement_evidence": copy.deepcopy(list(self.measurement_evidence)),
            "terminal_summary": copy.deepcopy(self.terminal_summary),
            "completed": self.completed,
            "operation_count": len(self.executed_steps),
            "compiled_operation_count": self.compiled_operation_count,
            "runtime_operation_cap": self.runtime_operation_cap,
            "runtime_margin_used": self.runtime_margin_used,
            "peak_safety_risk": self.peak_safety_risk,
        }

    def public_record(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("executed_steps", None)
        return payload


class StaticOptimizationExperimentSession:
    """Execute a static plan in one fixed world with no intervention input."""

    def __init__(
        self,
        *,
        task_id: str,
        seed: int,
        experiment_horizon: int,
        experiment_index_offset: int = 0,
        observation_seed: int | None = None,
        observation_noise_namespace: str = "static-optimization-s0",
        electrochemical_workflow_mode: str = (
            ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
        ),
        electrochemical_material_family_id: str | None = None,
        crystallization_material_family_id: str | None = None,
        scoring_contract_id: str = TASK_DERIVED_SCORING_CONTRACT,
    ) -> None:
        if experiment_horizon <= 0:
            raise ValueError("experiment_horizon must be positive")
        if experiment_index_offset < 0:
            raise ValueError("experiment_index_offset must be non-negative")
        self.task_id = str(task_id)
        self.task_info = get_task(self.task_id).to_dict()
        self.experiment_horizon = int(experiment_horizon)
        self.experiment_index_offset = int(experiment_index_offset)
        self.electrochemical_workflow_mode = normalize_electrochemical_workflow_mode(
            electrochemical_workflow_mode
        )
        self.electrochemical_material_family_id = electrochemical_material_family_id
        self.crystallization_material_family_id = crystallization_material_family_id
        self.scoring_contract_id = str(scoring_contract_id)
        if (
            self.task_id == "electrochemical-conversion"
            and self.electrochemical_workflow_mode
            == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
        ):
            per_experiment = ELECTROCHEMICAL_SINGLE_STAGE_EVENT_COUNT
        else:
            per_experiment = task_recipe_event_count(self.task_info)
        self.maximum_compiled_operations = per_experiment
        self.runtime_operation_cap = (per_experiment + 1) * self.experiment_horizon
        self.environment = ChemWorldEnv(
            task_id=self.task_id,
            seed=int(seed),
            episode_mode_override="campaign",
            budget_override=self.runtime_operation_cap,
            observation_seed_override=observation_seed,
            observation_noise_mode="keyed",
            observation_noise_namespace=observation_noise_namespace,
            world_interventions=(),
            electrochemical_workflow_mode=self.electrochemical_workflow_mode,
            electrochemical_material_family_id=(
                self.electrochemical_material_family_id
            ),
            crystallization_material_family_id=(
                self.crystallization_material_family_id
            ),
            scoring_contract_id=self.scoring_contract_id,
        )
        self.environment.reset(seed=int(seed))
        self._completed_experiments = 0

    def execute(self, plan: StaticOptimizationPlan) -> StaticOptimizationResult:
        if self._completed_experiments >= self.experiment_horizon:
            raise RuntimeError("static optimization session horizon is exhausted")
        recipe = compile_static_optimization_plan(
            self.task_info,
            plan,
            electrochemical_workflow_mode=self.electrochemical_workflow_mode,
        )
        compiled_operation_count = len(recipe["steps"])
        if compiled_operation_count > self.runtime_operation_cap:
            raise RuntimeError(
                "compiled static experiment exceeds its runtime operation cap"
            )
        slots_by_step = recipe["metadata"]["measurement_slots_by_step"]
        executed_steps: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        peak_safety_risk = 0.0
        final_info: dict[str, Any] = {}
        experiment_index = self.experiment_index_offset + self._completed_experiments
        for step_index, action in enumerate(recipe["steps"]):
            observation, reward, terminated, truncated, info = self.environment.step(action)
            if info.get("transaction_status") != "committed":
                raise RuntimeError(
                    "static optimization step was not committed: "
                    f"step={step_index}, operation={action.get('operation')}"
                )
            public_observation = {
                key: value
                for key, value in observation_to_json(observation).items()
                if value is not None
            }
            public_info = {
                key: copy.deepcopy(to_builtin(info[key]))
                for key in _PUBLIC_STEP_INFO_KEYS
                if key in info
            }
            executed_steps.append(
                {
                    "step_index": step_index,
                    "action": copy.deepcopy(to_builtin(action)),
                    "observation": public_observation,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": public_info,
                }
            )
            safety_risk = public_observation.get("safety_risk")
            if isinstance(safety_risk, int | float):
                peak_safety_risk = max(peak_safety_risk, float(safety_risk))
            slot_id = slots_by_step.get(str(step_index))
            if slot_id is not None:
                evidence.append(
                    {
                        "evidence_id": f"s0e{experiment_index:03d}-{slot_id}",
                        "measurement_slot_id": slot_id,
                        "instrument": str(action["instrument"]),
                        "observation": public_observation,
                        "processed_estimate": copy.deepcopy(
                            to_builtin(info.get("processed_estimate", {}))
                        ),
                        "uncertainty": copy.deepcopy(to_builtin(info.get("uncertainty", {}))),
                        "reward": float(reward),
                    }
                )
            final_info = info
        completed = bool(final_info.get("experiment_ended")) and bool(
            recipe["steps"][-1].get("operation") == "measure"
            and recipe["steps"][-1].get("instrument") == "final_assay"
        )
        if not completed:
            raise RuntimeError("static optimization plan did not reach final assay")
        if len(executed_steps) != compiled_operation_count:
            raise RuntimeError("static optimization execution was not atomic")
        terminal_summary = copy.deepcopy(to_builtin(final_info.get("last_terminal_summary", {})))
        terminal_summary["environment_experiment_index"] = terminal_summary.get("experiment_index")
        terminal_summary["experiment_index"] = experiment_index
        self._completed_experiments += 1
        return StaticOptimizationResult(
            task_id=self.task_id,
            experiment_index=experiment_index,
            plan=plan,
            executed_steps=tuple(executed_steps),
            measurement_evidence=tuple(evidence),
            terminal_summary=terminal_summary,
            completed=True,
            peak_safety_risk=peak_safety_risk,
            compiled_operation_count=compiled_operation_count,
            runtime_operation_cap=self.runtime_operation_cap,
            runtime_margin_used=len(executed_steps) > compiled_operation_count,
        )

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> StaticOptimizationExperimentSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def build_static_optimization_agent(
    protocol: Mapping[str, Any],
    task_id: str,
    *,
    llm_methods: Mapping[str, Any],
    method_id: str,
    client: Any | None = None,
) -> StaticOptimizationAgent:
    task_spec = get_task(str(task_id))
    scoring_contract = TaskScoringContract.from_success_metrics(
        objective=task_spec.objective,
        success_metrics=task_spec.success_metrics,
        contract_id=static_optimization_scoring_contract_id(protocol),
    )
    method = llm_methods["methods"][method_id]
    request = method["request_configuration"]
    prompt_budget = method.get("static_optimization_prompt_budget_contract")
    if not isinstance(prompt_budget, Mapping):
        raise ValueError("S0 method lacks static optimization prompt budget contract")
    planner = cast(
        JsonPlannerClientLike,
        client
        if client is not None
        else DeepSeekClient(
            model=str(method["model_id"]),
            thinking=bool(request["thinking"]),
            reasoning_effort=cast(Any, str(request.get("reasoning_effort") or "max")),
            timeout_s=float(request["timeout_s"]),
            max_attempts=int(request["max_attempts"]),
            retry_backoff_s=float(request["retry_backoff_s"]),
        ),
    )
    agent = StaticOptimizationAgent(
        planner,
        role_id=f"static_optimization_{method_id}",
        response_max_tokens=int(request["max_tokens"]),
        history_limit=int(prompt_budget["history_limit"]),
        prompt_token_estimate_cap=int(prompt_budget["per_decision_max_estimated_tokens"]),
        experiment_horizon=exploration_experiment_count(protocol),
        horizon_visible=bool(
            protocol.get("scientific_campaign_budget", {}).get(
                "horizon_visible", False
            )
        ),
        final_synthesis_enabled=bool(
            protocol.get("final_synthesis", {}).get("enabled", False)
        ),
        final_synthesis_prompt_token_estimate_cap=int(
            prompt_budget.get(
                "final_synthesis_max_estimated_tokens",
                prompt_budget["per_decision_max_estimated_tokens"],
            )
        ),
        predictive_synthesis_prompt_token_estimate_cap=int(
            prompt_budget.get(
                "predictive_synthesis_max_estimated_tokens",
                prompt_budget.get(
                    "final_synthesis_max_estimated_tokens",
                    prompt_budget["per_decision_max_estimated_tokens"],
                ),
            )
        ),
        include_task_operation_budget=bool(
            protocol.get("executor_contract", {}).get(
                "show_task_operation_budget_to_agent", True
            )
        ),
        predictive_world_understanding_enabled=bool(
            protocol.get("world_understanding", {}).get(
                "predictive_score_enabled", False
            )
        ),
        predictive_queries_in_final_synthesis=(
            static_optimization_predictive_call_policy(protocol)
            == PREDICTIVE_CALL_INTEGRATED
        ),
        declared_claim_validation_policy=str(
            method.get("declared_claim_validation_policy", "strict")
        ),
        material_information=protocol.get("material_information"),
        electrochemical_material_family_id=(
            static_optimization_material_family_id(protocol)
        ),
        crystallization_material_family_id=(
            static_optimization_crystallization_material_family_id(protocol)
        ),
        scoring_contract=scoring_contract.to_dict(),
        electrochemical_workflow_mode=static_optimization_workflow_mode(protocol),
        optimization_scaffold_id=str(
            method.get(
                "static_optimization_scaffold_id",
                "direct_known_horizon_five_task_public_contract_full_history_v15",
            )
        ),
    )
    agent.reset(task_spec.to_dict(), int(protocol["candidate_order_seed"]))
    return agent

__all__ = [
    "STATIC_OPTIMIZATION_RESULT_VERSION",
    "StaticOptimizationExperimentSession",
    "StaticOptimizationResult",
    "build_static_optimization_agent",
    "static_optimization_workflow_mode",
]
