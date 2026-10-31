from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from tests.fixtures.rl_worlds import example_allocation as _allocation

import chemworld  # noqa: F401
from chemworld.agent_interface import agent_view_bundle
from chemworld.agents.interaction import build_decision_context
from chemworld.data.logging import to_builtin
from chemworld.legacy.agents.rl import (
    RL_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    FrozenSB3Agent,
    build_frozen_rl_observation,
)
from chemworld.rl.environment import (
    build_rl_environment,
)
from chemworld.rl.evaluation import evaluate_replay_verified_sb3_checkpoint
from chemworld.rl.hybrid_actions import (
    conditional_hybrid_action_contract,
    policy_distribution_contract,
)
from chemworld.rl.observation_contract import rl_observation_contract
from chemworld.rl.rewards import reward_contract
from chemworld.tasks import get_task
from chemworld.world.operations import OPERATION_TYPES
from chemworld.wrappers import (
    ContinuousEventActionWrapper,
    action_mask,
    decode_continuous_event_action,
)


def _require_sb3_backend() -> Any:
    pytest.importorskip(
        "torch",
        reason="this replay-evaluation test requires the optional torch backend",
    )
    return pytest.importorskip(
        "stable_baselines3",
        reason="this replay-evaluation test requires the optional stable-baselines3 backend",
    )


def test_public_view_rebuilds_exact_training_observation() -> None:
    task_id = "flow-reaction-optimization"
    env = build_rl_environment(
        task_id=task_id,
        allocation=_allocation(task_id),
        sampler_seed=7,
        operation_budget=5,
    )
    try:
        wrapped_observation, _ = env.reset()
        base = cast(Any, env.unwrapped)
        raw_observation = base._last_observation
        raw_info = base._last_info
        public_view = agent_view_bundle(base, raw_observation, raw_info)
        context = build_decision_context(
            step=1,
            task_info=base.task_info(),
            campaign_state=base.campaign_state(),
            public_view=public_view,
            previous_event_type=None,
        )
        rebuilt, operation_mask = build_frozen_rl_observation(public_view, context)
        assert rebuilt == pytest.approx(wrapped_observation)
        assert sum(operation_mask) == len(context.available_operations)
    finally:
        env.close()


def test_public_view_rejects_coordinate_order_drift() -> None:
    task_id = "flow-reaction-optimization"
    env = build_rl_environment(
        task_id=task_id,
        allocation=_allocation(task_id),
        sampler_seed=7,
        operation_budget=5,
    )
    try:
        env.reset()
        base = cast(Any, env.unwrapped)
        public_view = agent_view_bundle(base, base._last_observation, base._last_info)
        rl_view = public_view["rl"]
        rl_view["keys"] = list(reversed(rl_view["keys"]))
        context = build_decision_context(
            step=1,
            task_info=base.task_info(),
            campaign_state=base.campaign_state(),
            public_view=public_view,
            previous_event_type=None,
        )
        with pytest.raises(ValueError, match="coordinate keys"):
            build_frozen_rl_observation(public_view, context)
    finally:
        env.close()


def test_public_view_rebuilds_nonzero_core_progress_after_valid_operation() -> None:
    task_id = "flow-reaction-optimization"
    env = build_rl_environment(
        task_id=task_id,
        allocation=_allocation(task_id),
        sampler_seed=7,
        operation_budget=5,
    )
    try:
        env.reset()
        action_shape = env.action_space.shape
        assert action_shape is not None
        action = np.zeros(action_shape, dtype=np.float32)
        action[OPERATION_TYPES.index("set_flow_rate")] = 1.0
        wrapped_observation, _, _, _, _ = env.step(action)
        base = cast(Any, env.unwrapped)
        public_view = agent_view_bundle(base, base._last_observation, base._last_info)
        context = build_decision_context(
            step=2,
            task_info=base.task_info(),
            campaign_state=base.campaign_state(),
            public_view=public_view,
            previous_event_type="operation_result",
        )
        rebuilt, _ = build_frozen_rl_observation(
            public_view,
            context,
            executed_operations={"set_flow_rate"},
        )
        assert rebuilt == pytest.approx(wrapped_observation)
        assert rebuilt[-(len(OPERATION_TYPES) + 5)] == 1.0
    finally:
        env.close()


def test_frozen_agent_public_operation_ledger_tracks_success_and_resets() -> None:
    agent = object.__new__(FrozenSB3Agent)
    agent._executed_operations = set()

    agent.update(
        {"operation": "set_flow_rate"},
        {},
        0.0,
        {
            "operation_type": "set_flow_rate",
            "constraint_flags": {"precondition_failed": False},
        },
    )
    assert agent._executed_operations == {"set_flow_rate"}

    agent.update(
        {"operation": "run_flow"},
        {},
        0.0,
        {
            "operation_type": "run_flow",
            "constraint_flags": {"precondition_failed": True},
        },
    )
    assert agent._executed_operations == {"set_flow_rate"}

    agent.update(
        {"operation": "measure"},
        {},
        0.0,
        {
            "operation_type": "measure",
            "constraint_flags": {"precondition_failed": False},
            "experiment_ended": True,
        },
    )
    assert agent._executed_operations == set()


def test_pure_decoder_matches_gym_action_wrapper() -> None:
    env = ContinuousEventActionWrapper(
        gym.make("ChemWorld", task_id="flow-reaction-optimization")
    )
    try:
        env.reset(seed=3)
        vector = np.linspace(
            -1.0,
            1.0,
            int(np.prod(env.action_space.shape)),
            dtype=np.float32,
        )
        mask = action_mask(env)
        decoded = decode_continuous_event_action(
            vector,
            event_action_space=env.event_action_space,
            operation_mask=mask,
        )
        assert to_builtin(decoded) == to_builtin(env.action(vector))
    finally:
        env.close()


def test_checkpoint_digest_mismatch_fails_before_policy_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not-a-checkpoint")
    manifest = {
        "schema_version": RL_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "algorithm": "ppo",
        "task_id": "flow-reaction-optimization",
        "checkpoint_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="digest"):
        FrozenSB3Agent(
            algorithm="ppo",
            checkpoint=checkpoint,
            checkpoint_manifest=manifest,
            task_id="flow-reaction-optimization",
        )


def test_checkpoint_contract_mismatch_fails_before_policy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sb3 = _require_sb3_backend()
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not-loaded")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": RL_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "algorithm": "ppo",
        "task_id": "flow-reaction-optimization",
        "checkpoint_sha256": digest,
        "observation_contract_hash": rl_observation_contract(
            "flow-reaction-optimization"
        )["contract_hash"],
        "action_contract_hash": "0" * 64,
        "training_reward_contract_hash": "0" * 64,
    }

    def unexpected_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("incompatible checkpoint reached policy deserialization")

    monkeypatch.setattr(sb3.PPO, "load", unexpected_load)
    with pytest.raises(ValueError, match="action contract hash"):
        FrozenSB3Agent(
            algorithm="ppo",
            checkpoint=checkpoint,
            checkpoint_manifest=manifest,
            task_id="flow-reaction-optimization",
        )


def test_observation_contract_mismatch_fails_before_policy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sb3 = _require_sb3_backend()
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not-loaded")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": RL_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "algorithm": "ppo",
        "task_id": "flow-reaction-optimization",
        "checkpoint_sha256": digest,
        "observation_contract_hash": "0" * 64,
    }

    def unexpected_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("incompatible checkpoint reached policy deserialization")

    monkeypatch.setattr(sb3.PPO, "load", unexpected_load)
    with pytest.raises(ValueError, match="observation contract hash"):
        FrozenSB3Agent(
            algorithm="ppo",
            checkpoint=checkpoint,
            checkpoint_manifest=manifest,
            task_id="flow-reaction-optimization",
        )


def test_frozen_policy_produces_official_replay_verified_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sb3 = _require_sb3_backend()
    task_id = "flow-reaction-optimization"
    shape_env = build_rl_environment(
        task_id=task_id,
        allocation=_allocation(task_id),
        sampler_seed=1,
        operation_budget=3,
    )
    observation_space = shape_env.observation_space
    action_space = shape_env.action_space
    event_action_space = cast(gym.spaces.Dict, shape_env.unwrapped.action_space)
    action_contract = conditional_hybrid_action_contract(event_action_space)
    shape_env.close()

    class FakeModel:
        def __init__(self) -> None:
            self.observation_space = observation_space
            self.action_space = action_space

        def set_random_seed(self, seed: int) -> None:
            self.seed = seed

        def predict(
            self, observation: np.ndarray, *, deterministic: bool
        ) -> tuple[np.ndarray, None]:
            del observation, deterministic
            return np.zeros(self.action_space.shape, dtype=np.float32), None

    monkeypatch.setattr(sb3.PPO, "load", lambda *args, **kwargs: FakeModel())
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"frozen-policy-fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": RL_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "algorithm": "ppo",
        "task_id": task_id,
        "checkpoint_sha256": digest,
        "training_environment_step_count": 8,
        "cpu_time_s": 0.1,
        "gpu_time_s": 0.0,
        "allocation": {"name": "train"},
        "versions": {"stable_baselines3": sb3.__version__},
        "observation_contract_hash": rl_observation_contract(task_id)["contract_hash"],
        "action_contract_hash": action_contract["contract_hash"],
        "training_reward_contract_hash": reward_contract(
            get_task(task_id).allowed_operations
        )["contract_hash"],
        "policy_distribution_contract_hash": policy_distribution_contract(
            tuple(action_contract["training_adapter"]["parameter_coordinate_keys"])
        )["contract_hash"],
    }
    report = evaluate_replay_verified_sb3_checkpoint(
        algorithm="ppo",
        checkpoint=checkpoint,
        checkpoint_manifest=manifest,
        task_id=task_id,
        seeds=[1180],
        operation_budget=3,
        output_dir=tmp_path / "evaluation",
        policy_seed=50,
    )
    assert report["all_replay_verified"] is True
    assert report["formal_evidence"] is False
    result = report["results"][0]
    assert result["resource_usage"]["training_environment_step_count"] == 8
    assert result["rl_evaluation"]["world_family_allocation"] is None
    layered = result["score_replay"]["layered_evaluation"]
    assert layered["status"] == "failed_no_successful_final_assay"
    assert layered["objective"]["best"] == 0.0
    assert layered["resources"]["incomplete_experiment_count"] == 1
