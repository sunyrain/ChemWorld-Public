"""Optional Gymnasium wrappers and event-action validators."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.utils import RecordConstructorArgs

from chemworld.agent_interface import (
    action_schema,
    observation_view,
    rl_observation_spec,
    rl_observation_view,
    validate_action,
)
from chemworld.data.logging import to_builtin
from chemworld.rl.hybrid_actions import (
    conditional_hybrid_action_contract,
    decode_conditional_hybrid_action,
)
from chemworld.rl.rewards import (
    REWARD_COMPONENTS,
    REWARD_SCHEMA_VERSION,
    PublicBehaviorTracker,
    reward_contract,
)
from chemworld.world.operations import OPERATION_TYPES, operation_contracts
from chemworld.world.scoring import safety_cost_from_flags


def _base_env(env: gym.Env[Any, Any]) -> Any:
    return getattr(env, "unwrapped", env)


def valid_operations(env: gym.Env[Any, Any]) -> list[str]:
    """Return operation types whose current preconditions are satisfied."""

    base = _base_env(env)
    return list(base.operation_validator.valid_operations(base._state))


def action_mask(env: gym.Env[Any, Any]) -> list[bool]:
    base = _base_env(env)
    return list(base.operation_validator.action_mask(base._state))


def validate_event_action(action: dict[str, Any], env: gym.Env[Any, Any]) -> dict[str, Any]:
    """Validate one event action against the current environment state."""

    base = _base_env(env)
    return base.operation_validator.validate(action, base._state).to_dict()


def validate_operation_affordance(operation: str, env: gym.Env[Any, Any]) -> dict[str, Any]:
    """Validate current operation-level affordance without requiring payload fields."""

    base = _base_env(env)
    return base.operation_validator.operation_affordance(operation, base._state).to_dict()


def decode_continuous_event_action(
    action: Any,
    *,
    event_action_space: gym.spaces.Dict,
    operation_mask: list[bool] | np.ndarray,
) -> dict[str, Any]:
    """Decode the frozen RL vector contract using public operation affordances.

    This function is deliberately independent of an environment instance.  It
    lets frozen-policy evaluators use exactly the same mapping as the Gym
    wrapper while supplying only the public affordance mask available to every
    benchmark agent.
    """

    action_keys = tuple(event_action_space.spaces)
    parameter_keys = tuple(key for key in action_keys if key != "operation")
    operation_logit_count = len(OPERATION_TYPES)
    expected_shape = (operation_logit_count + len(parameter_keys),)
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != expected_shape or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"continuous event action must be a finite vector with shape {expected_shape}"
        )
    public_mask = np.asarray(operation_mask, dtype=bool)
    if public_mask.shape != (operation_logit_count,):
        raise ValueError("public operation mask does not match the frozen operation registry")
    operation_logits = vector[:operation_logit_count]
    if np.any(public_mask):
        operation_index = int(np.argmax(np.where(public_mask, operation_logits, -np.inf)))
    else:
        operation_index = int(np.argmax(operation_logits))
    operation = OPERATION_TYPES[operation_index]
    required_fields = frozenset(operation_contracts()[operation].required_fields)
    payload: dict[str, Any] = {"operation": operation_index}
    unit = np.clip((vector[operation_logit_count:] + 1.0) / 2.0, 0.0, 1.0)
    for index, key in enumerate(parameter_keys):
        if key not in required_fields:
            continue
        space = event_action_space[key]
        coordinate = float(unit[index])
        if isinstance(space, gym.spaces.Discrete):
            payload[key] = min(int(coordinate * space.n), space.n - 1)
        elif isinstance(space, gym.spaces.Box):
            value = space.low + coordinate * (space.high - space.low)
            payload[key] = to_builtin(np.asarray(value, dtype=space.dtype))
        else:
            raise TypeError(f"unsupported event action component: {key}={type(space).__name__}")
    return payload


class ActionMaskWrapper(gym.Wrapper[Any, Any, Any, Any], RecordConstructorArgs):
    """Add operation validity signals to reset/step info."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_mask(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward), terminated, truncated, self._with_mask(info)

    def _with_mask(self, info: dict[str, Any]) -> dict[str, Any]:
        payload = dict(info)
        mask = action_mask(self.env)
        payload["valid_operations"] = [
            operation_type
            for operation_type, is_valid in zip(OPERATION_TYPES, mask, strict=True)
            if is_valid
        ]
        payload["action_mask"] = np.asarray(mask, dtype=bool)
        payload["operation_types"] = list(OPERATION_TYPES)
        payload["invalid_reasons"] = {
            operation_type: validate_operation_affordance(operation_type, self.env)[
                "invalid_reasons"
            ]
            for operation_type in OPERATION_TYPES
            if operation_type not in payload["valid_operations"]
        }
        return payload


class SafetyCostWrapper(gym.Wrapper[Any, Any, Any, Any], RecordConstructorArgs):
    """Expose safe-RL style cost signals without changing Gymnasium returns."""

    def __init__(self, env: gym.Env[Any, Any], *, constraint_budget: float = 1.0) -> None:
        RecordConstructorArgs.__init__(self, constraint_budget=constraint_budget)
        super().__init__(env)
        self.constraint_budget = float(constraint_budget)
        self._spent = 0.0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._spent = 0.0
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_cost(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = self._with_cost(info)
        return observation, float(reward), terminated, truncated, info

    def _with_cost(self, info: dict[str, Any]) -> dict[str, Any]:
        payload = dict(info)
        if "cost" in payload and "cost_components" in payload:
            cost_signal = float(payload["cost"])
            self._spent += cost_signal
            payload["cost_signal"] = cost_signal
            payload["constraint_budget_remaining"] = max(self.constraint_budget - self._spent, 0.0)
            return payload
        flags = payload.get("constraint_flags", {})
        cost_signal, components = safety_cost_from_flags(flags)
        self._spent += cost_signal
        payload["cost_signal"] = cost_signal
        payload["cost_components"] = components
        payload["constraint_budget_remaining"] = max(self.constraint_budget - self._spent, 0.0)
        return payload


class NaNObservationWrapper(gym.ObservationWrapper[Any, Any, Any], RecordConstructorArgs):
    """Convert dict observations with NaN values into RL-friendly vectors."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        sentinel: float = -1.0,
        include_mask: bool = True,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            sentinel=sentinel,
            include_mask=include_mask,
        )
        super().__init__(env)
        observation_space = env.observation_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("NaNObservationWrapper requires a Dict observation space.")
        self.sentinel = float(sentinel)
        self.include_mask = include_mask
        self.observation_keys = list(observation_space.spaces.keys())
        width = len(self.observation_keys) * (2 if include_mask else 1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(width,),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        values: list[float] = []
        mask: list[float] = []
        for key in self.observation_keys:
            value = float(np.asarray(observation[key]).reshape(-1)[0])
            observed = np.isfinite(value)
            values.append(value if observed else self.sentinel)
            mask.append(1.0 if observed else 0.0)
        if self.include_mask:
            values.extend(mask)
        return np.asarray(values, dtype=np.float32)


class AgentInfoWrapper(gym.Wrapper[Any, Any, Any, Any], RecordConstructorArgs):
    """Attach task prompt, campaign state, and current action affordances to info."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_agent_info(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward), terminated, truncated, self._with_agent_info(info)

    def _with_agent_info(self, info: dict[str, Any]) -> dict[str, Any]:
        base = _base_env(self.env)
        payload = dict(info)
        payload["task_prompt"] = base.task_prompt()
        payload["campaign_state"] = base.campaign_state()
        payload["available_actions"] = base.available_actions()
        return payload


class LLMObservationWrapper(gym.Wrapper[Any, Any, Any, Any], RecordConstructorArgs):
    """Attach deterministic LLM/student-readable observation summaries to info."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_views(observation, info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            self._with_views(observation, info),
        )

    def _with_views(self, observation: Any, info: dict[str, Any]) -> dict[str, Any]:
        payload = dict(info)
        payload["lab_report"] = observation_view(
            self.env,
            "lab_report",
            observation,
            info,
        )
        payload["tool_json"] = observation_view(
            self.env,
            "tool_json",
            observation,
            info,
        )
        return payload


class RLObservationWrapper(gym.Wrapper[Any, Any, Any, Any], RecordConstructorArgs):
    """Return finite observations and mask/cost info compatible with Gym's checker.

    Unique episode identity is retained in ``last_audit_metadata`` and the base
    environment, outside seed-deterministic training info. Raw observations keep
    their missing-value semantics; the vector always has an explicit mask by default.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        include_mask: bool = True,
        include_cost: bool = True,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            include_mask=include_mask,
            include_cost=include_cost,
        )
        super().__init__(env)
        self.include_mask = include_mask
        self.include_cost = include_cost
        self.last_audit_metadata: dict[str, Any] = {}
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("RLObservationWrapper requires a Dict observation space.")
        spec = rl_observation_spec(include_cost=include_cost)
        self.observation_keys = list(spec["keys"])
        value_low = list(spec["value_bounds"]["low"])
        value_high = list(spec["value_bounds"]["high"])
        mask_low = list(spec["mask_bounds"]["low"])
        mask_high = list(spec["mask_bounds"]["high"])
        low = [*value_low, *mask_low] if include_mask else value_low
        high = [*value_high, *mask_high] if include_mask else value_high
        self.observation_space = gym.spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            shape=(len(low),),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        vector, payload = self._vectorize(observation, info)
        return vector, payload

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        vector, payload = self._vectorize(observation, info)
        return vector, float(reward), terminated, truncated, payload

    def _vectorize(
        self,
        observation: Any,
        info: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        view = rl_observation_view(observation, info, include_cost=self.include_cost)
        values = list(view["vector"])
        mask = list(view["mask"])
        vector_values = [*values, *mask] if self.include_mask else values
        payload = dict(info)
        self.last_audit_metadata = {"campaign_id": payload.pop("campaign_id", None)}
        payload["rl_view"] = view
        payload["observation_mask"] = np.asarray(mask, dtype=np.float32)
        payload["cost_signal"] = float(view["cost"])
        return np.asarray(vector_values, dtype=np.float32), payload


class ContinuousEventActionWrapper(gym.ActionWrapper[Any, Any, Any], RecordConstructorArgs):
    """Map masked operation logits plus parameters onto a typed event action.

    Operation coordinates retain fixed global semantics and are filtered by the
    same public affordance mask exposed in the observation. This avoids imposing
    a false ordinal geometry on operation categories while keeping a Box action
    space shared by PPO and SAC. Parameter coordinates remain stationary.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Dict):
            raise TypeError("ContinuousEventActionWrapper requires a Dict action space")
        self.event_action_space = env.action_space
        self.action_keys = tuple(self.event_action_space.spaces)
        self.parameter_keys = tuple(key for key in self.action_keys if key != "operation")
        self.operation_types = tuple(OPERATION_TYPES)
        self.operation_logit_count = len(self.operation_types)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.operation_logit_count + len(self.parameter_keys),),
            dtype=np.float32,
        )

    def action_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "chemworld-continuous-event-action-0.4",
            "action_keys": list(self.action_keys),
            "operation_types": list(self.operation_types),
            "operation_logit_count": self.operation_logit_count,
            "parameter_keys": list(self.parameter_keys),
            "shape": list(self.action_space.shape or ()),
            "low": -1.0,
            "high": 1.0,
            "operation_mapping": (
                "fixed global logits decoded by argmax after public affordance mask"
            ),
            "parameter_discrete_mapping": "fixed global index; floor(unit_coordinate * n)",
            "empty_operation_mask_policy": "retain global argmax and record the resulting failure",
            "inactive_parameter_policy": "excluded_from_execution_and_trajectory",
            "invalid_payload_policy": "retain environment precondition or domain failure",
            "execution_numeric_policy": (
                "normalize numpy values to their JSON trajectory representation before execution"
            ),
        }

    def action(self, action: Any) -> dict[str, Any]:
        return decode_continuous_event_action(
            action,
            event_action_space=self.event_action_space,
            operation_mask=action_mask(self.env),
        )


class ConditionalHybridActionWrapper(gym.ActionWrapper[Any, Any, Any], RecordConstructorArgs):
    """SB3 adapter for the categorical, operation-conditional action contract.

    The exposed Box is a framework compatibility latent, not the semantic
    benchmark action. Only fields required by the chosen operation are decoded.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Dict):
            raise TypeError("ConditionalHybridActionWrapper requires a Dict action space")
        self.event_action_space = env.action_space
        self.parameter_keys = tuple(
            key for key in self.event_action_space.spaces if key != "operation"
        )
        self.operation_types = tuple(OPERATION_TYPES)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.operation_types) + len(self.parameter_keys),),
            dtype=np.float32,
        )

    def action_contract(self) -> dict[str, Any]:
        return conditional_hybrid_action_contract(self.event_action_space)

    def action(self, action: Any) -> dict[str, Any]:
        operation_mask = action_mask(self.env)
        static = decode_conditional_hybrid_action(
            action,
            event_action_space=self.event_action_space,
            operation_mask=operation_mask,
        )
        operation = self.operation_types[int(static["operation"])]
        public_schema = action_schema(self.env, operation)
        decoded = decode_conditional_hybrid_action(
            action,
            event_action_space=self.event_action_space,
            operation_mask=operation_mask,
            operation_schema=public_schema,
        )
        return self._project_public_choices(action, decoded, public_schema)

    def _project_public_choices(
        self,
        latent: Any,
        decoded: dict[str, Any],
        public_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Map categorical coordinates only across choices valid under the public validator."""

        if validate_action(self.env, decoded).get("valid") is True:
            return decoded
        vector = np.asarray(latent, dtype=np.float32).reshape(-1)
        parameter_unit = np.clip(
            (vector[len(self.operation_types) :] + 1.0) / 2.0,
            0.0,
            1.0,
        )
        for field_schema in public_schema.get("fields", []):
            if not isinstance(field_schema, dict):
                continue
            field = field_schema.get("field")
            choices = field_schema.get("choices")
            if field not in self.parameter_keys or not isinstance(choices, list):
                continue
            valid_choices = []
            for choice in choices:
                candidate = {**decoded, str(field): choice}
                if validate_action(self.env, candidate).get("valid") is True:
                    valid_choices.append(choice)
            if valid_choices:
                coordinate = float(parameter_unit[self.parameter_keys.index(str(field))])
                decoded[str(field)] = valid_choices[
                    min(int(coordinate * len(valid_choices)), len(valid_choices) - 1)
                ]
        return decoded


class RLControlObservationWrapper(gym.ObservationWrapper[Any, Any, Any], RecordConstructorArgs):
    """Append public procedural state, affordances, and campaign progress.

    Training reward depends on whether the current experiment has already
    executed each public core-operation group. Exposing the same public ledger
    here keeps that reward Markov: identical observations no longer hide
    different progress-dependent rewards.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("RLControlObservationWrapper requires a Box observation space")
        base_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        base = _base_env(env)
        self._task_id = getattr(base, "task_id", None)
        self._behavior = PublicBehaviorTracker(
            base.allowed_operations,
            task_id=self._task_id,
        )
        self._core_requirements = self._behavior.requirements
        extra_count = len(self._core_requirements) + len(OPERATION_TYPES) + 3
        extra_low = np.zeros(extra_count, dtype=np.float32)
        extra_high = np.ones(extra_count, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([base_low, extra_low]),
            high=np.concatenate([base_high, extra_high]),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._behavior.reset()
        observation, info = self.env.reset(**kwargs)
        payload = dict(info)
        payload["rl_core_progress"] = self._progress_payload()
        return self.observation(observation), payload

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._behavior.observe(info)
        if bool(info.get("experiment_ended", False)):
            # Campaign mode has already reset the physical experiment in the
            # returned observation, so its public procedural ledger resets too.
            self._behavior.reset()
        payload = dict(info)
        payload["rl_core_progress"] = self._progress_payload()
        return self.observation(observation), float(reward), terminated, truncated, payload

    def observation(self, observation: Any) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        base = _base_env(self.env)
        budget = max(int(getattr(base, "budget", 1)), 1)
        step = max(int(getattr(base, "_step_count", 0)), 0)
        experiment = max(int(getattr(base, "_experiment_index", 0)), 0)
        summaries = list(getattr(base, "_experiment_summaries", []))
        progress = np.asarray(
            [
                min(step / budget, 1.0),
                min(max(budget - step, 0) / budget, 1.0),
                min(len(summaries) / max(experiment + 1, 1), 1.0),
            ],
            dtype=np.float32,
        )
        core_progress = np.asarray(
            [float(value) for value in self._behavior.satisfied],
            dtype=np.float32,
        )
        affordances = np.asarray(action_mask(self.env), dtype=np.float32)
        # Keep affordances immediately before the final three campaign fields;
        # the conditional PPO policy reads that stable trailing layout.
        combined = np.concatenate([vector, core_progress, affordances, progress]).astype(np.float32)
        if not self.observation_space.contains(combined):
            raise ValueError("RL control observation escaped its declared finite bounds")
        return combined

    def _progress_payload(self) -> dict[str, Any]:
        return {
            "requirements": [list(group) for group in self._core_requirements],
            "satisfied": self._behavior.satisfied,
            "behavior_tokens": sorted(self._behavior.tokens),
            "public_operation_history_only": True,
        }


class RLTrainingRewardWrapper(gym.Wrapper[Any, Any, Any, Any], RecordConstructorArgs):
    """Add auditable public-signal shaping for operation-level RL training.

    The wrapper never changes observations, actions, termination, or the raw
    benchmark score. Frozen-policy evaluation must omit this wrapper and use
    replayed task metrics. Shaping exists only to make invalid preconditions and
    completed experiments distinguishable to a learner.
    """

    SCHEMA_VERSION = REWARD_SCHEMA_VERSION

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)
        self._previous_valid_operations: set[str] = set()
        base = _base_env(env)
        self._task_id = getattr(base, "task_id", None)
        self._allowed_operations = tuple(sorted(base.allowed_operations))
        self._behavior = PublicBehaviorTracker(
            self._allowed_operations,
            task_id=self._task_id,
        )
        self._core_requirements = self._behavior.requirements
        self._satisfied_requirement_count = 0
        self._diagnostics = self._empty_diagnostics()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._previous_valid_operations = self._valid_operations()
        self._behavior.reset()
        self._satisfied_requirement_count = 0
        self._diagnostics["episode_count"] += 1
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        flags = dict(info.get("constraint_flags", {}))
        invalid_precondition = bool(flags.get("precondition_failed", False))
        transaction_rolled_back = info.get("transaction_status") == "rolled_back"
        constitution_failed = bool(flags.get("constitution_failed", False))
        invalid = invalid_precondition or transaction_rolled_back or constitution_failed
        current_valid = self._valid_operations()
        newly_unlocked = len(current_valid.difference(self._previous_valid_operations))
        operation = str(info.get("operation_type", ""))
        experiment_ended = bool(info.get("experiment_ended", False))
        # Campaign-mode final assay resets the next experiment before returning.
        # Those post-reset affordances are not consequences worth rewarding on
        # the terminal measurement step.
        if experiment_ended:
            newly_unlocked = 0
        newly_satisfied_tokens = self._behavior.observe(info)
        satisfied_count = self._behavior.satisfied_count
        newly_satisfied = satisfied_count - self._satisfied_requirement_count
        behavior_complete = self._behavior.complete
        newly_behavior_complete = behavior_complete and (
            self._satisfied_requirement_count < len(self._core_requirements)
        )
        quick_close = experiment_ended and not behavior_complete
        shaped_reward = float(raw_reward)
        if invalid_precondition:
            shaped_reward += REWARD_COMPONENTS["invalid_precondition"]
        elif transaction_rolled_back:
            shaped_reward += REWARD_COMPONENTS["transaction_rollback"]
        elif operation not in {"measure", "terminate"}:
            shaped_reward += REWARD_COMPONENTS["valid_nonterminal_operation"]
        shaped_reward += REWARD_COMPONENTS["newly_unlocked_operation"] * newly_unlocked
        shaped_reward += REWARD_COMPONENTS["newly_satisfied_core_requirement"] * newly_satisfied
        if newly_behavior_complete:
            shaped_reward += REWARD_COMPONENTS["behavioral_core_completion"]
        if experiment_ended and behavior_complete:
            shaped_reward += REWARD_COMPONENTS["experiment_ended"]
        if quick_close:
            shaped_reward += REWARD_COMPONENTS["quick_close_incomplete"]
        if bool(flags.get("unsafe_by_task_limit", False)):
            shaped_reward += REWARD_COMPONENTS["unsafe_step"]
        if bool(flags.get("high_cost", False)):
            shaped_reward += REWARD_COMPONENTS["high_cost_step"]

        self._previous_valid_operations = current_valid
        self._satisfied_requirement_count = satisfied_count
        self._diagnostics["step_count"] += 1
        self._diagnostics["invalid_action_count"] += int(invalid)
        self._diagnostics["transaction_rollback_count"] += int(transaction_rolled_back)
        self._diagnostics["constitution_failure_count"] += int(constitution_failed)
        self._diagnostics["runtime_domain_failure_count"] += int(
            info.get("preconditions", {}).get("runtime_domain_valid") is False
        )
        self._diagnostics["observation_domain_failure_count"] += int(
            info.get("preconditions", {}).get("observation_domain_valid") is False
        )
        self._diagnostics["measurement_count"] += int(not invalid and operation == "measure")
        self._diagnostics["completed_experiment_count"] += int(experiment_ended)
        self._diagnostics["behavior_complete_experiment_count"] += int(
            experiment_ended and behavior_complete
        )
        self._diagnostics["quick_close_count"] += int(quick_close)
        self._diagnostics["core_requirement_satisfied_count"] += newly_satisfied
        for token in newly_satisfied_tokens:
            core_counts = self._diagnostics["core_operation_counts"]
            core_counts[token] = int(core_counts.get(token, 0)) + 1
        self._diagnostics["newly_unlocked_operation_count"] += newly_unlocked
        self._diagnostics["unsafe_step_count"] += int(
            bool(flags.get("unsafe_by_task_limit", False))
        )
        self._diagnostics["high_cost_step_count"] += int(bool(flags.get("high_cost", False)))
        self._diagnostics["raw_reward_sum"] += float(raw_reward)
        self._diagnostics["shaped_reward_sum"] += shaped_reward
        payload = dict(info)
        payload["rl_training_reward"] = {
            "schema_version": self.SCHEMA_VERSION,
            "raw_reward": float(raw_reward),
            "shaped_reward": shaped_reward,
            "invalid_action": invalid,
            "invalid_precondition": invalid_precondition,
            "transaction_rolled_back": transaction_rolled_back,
            "constitution_failed": constitution_failed,
            "newly_unlocked_operations": newly_unlocked,
            "newly_satisfied_core_requirements": newly_satisfied,
            "core_operation_requirements": [list(group) for group in self._core_requirements],
            "executed_core_operations": sorted(self._behavior.tokens),
            "behavior_complete": behavior_complete,
            "newly_behavior_complete": newly_behavior_complete,
            "quick_close_incomplete": quick_close,
            "experiment_ended": experiment_ended,
        }
        if experiment_ended:
            self._behavior.reset()
            self._satisfied_requirement_count = 0
        return observation, shaped_reward, terminated, truncated, payload

    def reward_contract(self) -> dict[str, Any]:
        return reward_contract(self._allowed_operations, task_id=self._task_id)

    def training_diagnostics(self) -> dict[str, Any]:
        payload = dict(self._diagnostics)
        steps = max(int(payload["step_count"]), 1)
        payload["invalid_action_rate"] = payload["invalid_action_count"] / steps
        payload["completed_experiments_per_1000_steps"] = (
            1000.0 * payload["completed_experiment_count"] / steps
        )
        payload["quick_close_rate"] = payload["quick_close_count"] / max(
            int(payload["completed_experiment_count"]), 1
        )
        return payload

    def _valid_operations(self) -> set[str]:
        base = _base_env(self.env)
        return set(base.operation_validator.valid_operations(base._state))

    @staticmethod
    def _empty_diagnostics() -> dict[str, Any]:
        return {
            "step_count": 0,
            "episode_count": 0,
            "invalid_action_count": 0,
            "transaction_rollback_count": 0,
            "constitution_failure_count": 0,
            "runtime_domain_failure_count": 0,
            "observation_domain_failure_count": 0,
            "measurement_count": 0,
            "completed_experiment_count": 0,
            "behavior_complete_experiment_count": 0,
            "quick_close_count": 0,
            "core_requirement_satisfied_count": 0,
            "core_operation_counts": {},
            "newly_unlocked_operation_count": 0,
            "unsafe_step_count": 0,
            "high_cost_step_count": 0,
            "raw_reward_sum": 0.0,
            "shaped_reward_sum": 0.0,
        }


class ActionSuggestionWrapper(gym.Wrapper[Any, Any, Any, Any], RecordConstructorArgs):
    """Expose legal next actions without auto-correcting the submitted action."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        RecordConstructorArgs.__init__(self)
        super().__init__(env)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return observation, self._with_suggestions(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward), terminated, truncated, self._with_suggestions(info)

    def _with_suggestions(self, info: dict[str, Any]) -> dict[str, Any]:
        base = _base_env(self.env)
        payload = dict(info)
        suggestions = base.available_actions()
        payload["action_suggestions"] = suggestions
        if payload.get("constraint_flags", {}).get("precondition_failed", False):
            operation_names = [entry["operation"] for entry in suggestions[:4]]
            payload["recovery_suggestion"] = "Try one of the currently valid operations: " + (
                ", ".join(operation_names) if operation_names else "none"
            )
        return payload
