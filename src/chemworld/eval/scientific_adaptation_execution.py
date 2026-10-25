"""Deterministic execution for experiment-level scientific adaptation plans."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from chemworld.agents.scientific_adaptation import (
    SCIENTIFIC_ADAPTATION_INTERFACE_VERSION,
    BoundedScientificMemory,
    DirectScaffoldPolicy,
    NullScientificMemory,
    ScaffoldPolicy,
    ScientificAdaptationAgent,
    ScientificExperimentPlan,
    ScientificMemoryStore,
    StatefulScientificScaffoldPolicy,
    compile_scientific_experiment_plan,
)
from chemworld.agents.task_recipes import task_recipe_event_count
from chemworld.data.logging import observation_to_json, to_builtin
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.legacy.agents.live_llm import JsonPlannerClientLike
from chemworld.providers.deepseek import DeepSeekClient
from chemworld.tasks import get_task

SCIENTIFIC_EXPERIMENT_RESULT_VERSION = "chemworld-scientific-experiment-result-0.1-dev"

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
class ScientificExperimentResult:
    """Standard public terminal result for one complete experiment."""

    task_id: str
    experiment_index: int
    plan: ScientificExperimentPlan
    executed_steps: tuple[dict[str, Any], ...]
    measurement_evidence: tuple[dict[str, Any], ...]
    terminal_summary: dict[str, Any]
    completed: bool
    peak_safety_risk: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCIENTIFIC_EXPERIMENT_RESULT_VERSION,
            "interface_version": SCIENTIFIC_ADAPTATION_INTERFACE_VERSION,
            "task_id": self.task_id,
            "experiment_index": self.experiment_index,
            "plan": self.plan.to_dict(),
            "executed_steps": copy.deepcopy(list(self.executed_steps)),
            "measurement_evidence": copy.deepcopy(list(self.measurement_evidence)),
            "terminal_summary": copy.deepcopy(self.terminal_summary),
            "completed": self.completed,
            "operation_count": len(self.executed_steps),
            "peak_safety_risk": self.peak_safety_risk,
        }

    def public_record(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["plan"] = self.plan.to_dict(include_scientific_state=False)
        payload.pop("executed_steps", None)
        return payload


class ScientificAdaptationExperimentSession:
    """Run complete plans in one nuisance-consistent campaign environment."""

    def __init__(
        self,
        *,
        task_id: str,
        seed: int,
        experiment_horizon: int,
        experiment_index_offset: int = 0,
        interventions: Sequence[Mapping[str, Any]] = (),
        observation_seed: int | None = None,
        observation_noise_mode: str = "keyed",
        observation_noise_namespace: str = "scientific-adaptation-development",
    ) -> None:
        if experiment_horizon <= 0:
            raise ValueError("experiment_horizon must be positive")
        if experiment_index_offset < 0:
            raise ValueError("experiment_index_offset must be non-negative")
        self.task_id = str(task_id)
        self.task_info = get_task(self.task_id).to_dict()
        self.experiment_horizon = int(experiment_horizon)
        self.experiment_index_offset = int(experiment_index_offset)
        per_experiment = task_recipe_event_count(self.task_info)
        self.environment = ChemWorldEnv(
            task_id=self.task_id,
            seed=int(seed),
            episode_mode_override="campaign",
            budget_override=(per_experiment + 1) * self.experiment_horizon,
            observation_seed_override=observation_seed,
            observation_noise_mode=observation_noise_mode,
            observation_noise_namespace=observation_noise_namespace,
            world_interventions=tuple(dict(item) for item in interventions),
        )
        self.environment.reset(seed=int(seed))
        self._completed_experiments = 0

    def execute(self, plan: ScientificExperimentPlan) -> ScientificExperimentResult:
        if self._completed_experiments >= self.experiment_horizon:
            raise RuntimeError("scientific adaptation session horizon is exhausted")
        recipe = compile_scientific_experiment_plan(self.task_info, plan)
        slot_by_step = recipe["metadata"]["measurement_slots_by_step"]
        executed_steps: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        peak_safety_risk = 0.0
        final_info: dict[str, Any] = {}
        experiment_index = self.experiment_index_offset + self._completed_experiments

        for step_index, action in enumerate(recipe["steps"]):
            observation, reward, terminated, truncated, info = self.environment.step(action)
            if info.get("transaction_status") != "committed":
                raise RuntimeError(
                    "deterministic scientific experiment step was not committed: "
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
            slot_id = slot_by_step.get(str(step_index))
            if slot_id is not None:
                evidence.append(
                    {
                        "evidence_id": f"e{experiment_index:03d}-{slot_id}",
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
            raise RuntimeError("compiled scientific experiment did not reach final assay")
        terminal_summary = copy.deepcopy(to_builtin(final_info.get("last_terminal_summary", {})))
        terminal_summary["environment_experiment_index"] = terminal_summary.get("experiment_index")
        terminal_summary["experiment_index"] = experiment_index
        self._completed_experiments += 1
        return ScientificExperimentResult(
            task_id=self.task_id,
            experiment_index=experiment_index,
            plan=plan,
            executed_steps=tuple(executed_steps),
            measurement_evidence=tuple(evidence),
            terminal_summary=terminal_summary,
            completed=True,
            peak_safety_risk=peak_safety_risk,
        )

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> ScientificAdaptationExperimentSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def execute_scientific_experiment(
    *,
    task_id: str,
    seed: int,
    plan: ScientificExperimentPlan,
    interventions: Sequence[Mapping[str, Any]] = (),
    observation_seed: int | None = None,
) -> ScientificExperimentResult:
    """Execute one development plan and close its environment."""

    with ScientificAdaptationExperimentSession(
        task_id=task_id,
        seed=seed,
        experiment_horizon=1,
        interventions=interventions,
        observation_seed=observation_seed,
    ) as session:
        return session.execute(plan)


def build_scientific_adaptation_agent(
    protocol: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    llm_methods: Mapping[str, Any],
    method_id: str,
    client: Any | None = None,
) -> ScientificAdaptationAgent:
    """Build one development experiment-level Participant method cell."""

    method = llm_methods["methods"][method_id]
    request = method["request_configuration"]
    prompt_budget = method.get("scientific_adaptation_prompt_budget_contract")
    if not isinstance(prompt_budget, Mapping):
        raise ValueError(
            "scientific adaptation method lacks its track-specific prompt budget contract"
        )
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
    definitions = protocol["diagnosis_contract"]["candidate_definitions"]
    candidate_definitions = {
        str(candidate_id): str(definitions[candidate_id]) for candidate_id in row["candidate_ids"]
    }
    scaffold_id = str(
        method.get("scientific_adaptation_scaffold_id") or method.get("scaffold_id") or "direct"
    )
    scaffold_policy: ScaffoldPolicy
    memory_store: ScientificMemoryStore
    if scaffold_id in {"direct", "direct_reactive"}:
        scaffold_policy = DirectScaffoldPolicy()
        memory_store = NullScientificMemory()
    elif scaffold_id == "stateful_scientific":
        scaffold_policy = StatefulScientificScaffoldPolicy()
        memory_store = BoundedScientificMemory(tuple(candidate_definitions))
    else:
        raise ValueError(f"unsupported scientific adaptation scaffold: {scaffold_id}")
    agent = ScientificAdaptationAgent(
        planner,
        role_id=f"scientific_adaptation_{method_id}",
        candidate_definitions=candidate_definitions,
        scaffold_policy=scaffold_policy,
        memory_store=memory_store,
        response_max_tokens=int(request["max_tokens"]),
        history_limit=int(prompt_budget["history_limit"]),
        prompt_token_estimate_cap=int(prompt_budget["per_decision_max_estimated_tokens"]),
    )
    agent.reset(get_task(str(row["task_id"])).to_dict(), int(row["candidate_order_seed"]))
    return agent


__all__ = [
    "SCIENTIFIC_EXPERIMENT_RESULT_VERSION",
    "ScientificAdaptationExperimentSession",
    "ScientificExperimentResult",
    "build_scientific_adaptation_agent",
    "execute_scientific_experiment",
]
