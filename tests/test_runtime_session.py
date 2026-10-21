"""Native, Gym and Agent consumers share one physical and resource history."""

import json
from copy import deepcopy

import gymnasium as gym
import pytest

from chemworld.agents.base import BaseAgent
from chemworld.campaign_resources import CampaignResourceCard
from chemworld.data.logging import load_jsonl, observation_to_json
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.envs.spaces import to_observation
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.runtime import ChemWorldRuntime, SessionSnapshot
from chemworld.world.surface_world import surface_world_spec

FEEDS = [
    {"operation": "add_component", "component": "CO-feed", "amount_mol": 8e-5},
    {"operation": "add_component", "component": "O2-feed", "amount_mol": 4e-5},
]
MEASUREMENT = {"operation": "measure", "instrument": "gas_composition"}


def public_result(result):
    """Episode IDs are intentionally random; they do not encode world seeds."""
    info = deepcopy(result)
    info.pop("campaign_id", None)
    return info


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
@pytest.mark.parametrize("terminal_assay", [False, True])
def test_native_and_gym_have_identical_committed_histories(noise, terminal_assay):
    kwargs = {
        "composition": surface_world_spec(terminal_assay=terminal_assay),
        "observation_noise_mode": noise,
    }
    session, env = ChemWorldRuntime(**kwargs), ChemWorldEnv(**kwargs)
    reading, _ = session.reset(seed=12)
    obs, _ = env.reset(seed=12)
    assert observation_to_json(obs) == observation_to_json(to_observation(reading.values))
    assert not isinstance(session, gym.Env)
    assert not hasattr(session, "action_space")
    actions = [
        {"operation": "unknown-action"},
        *FEEDS,
        {"operation": "heat", "target_temperature_K": 330, "duration_s": 2,
         "stirring_speed_rpm": 600},
        MEASUREMENT,
        {"operation": "route_material", "connection": "reactor-to-receiver",
         "transfer_fraction": .2, "mixing": "allow"},
        {"operation": "select_vessel", "vessel": "receiver"},
        {"operation": "measure", "instrument": "gas_pressure"},
        {"operation": "terminate"},
    ]
    if terminal_assay:
        actions.append({"operation": "measure", "instrument": "gas_final_assay"})
    for action in actions:
        result = session.execute(action)
        if action["operation"] != "unknown-action":
            assert result.info["transaction_status"] == "committed", result.info
        obs, reward, ended, truncated, info = env.step(action)
        assert session._state.to_dict(include_hidden=True) == (
            env._state.to_dict(include_hidden=True)
        )
        assert observation_to_json(obs) == (
            observation_to_json(to_observation(result.observation.values))
        )
        assert (reward, ended, truncated) == (result.reward, result.terminated, result.truncated)
        assert public_result(info) == public_result(result.info)
    assert result.terminated and not result.truncated
    assert result.info["transaction_status"] == "committed"


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_snapshot_restores_noisy_physical_continuation_after_reset(noise):
    session = ChemWorldRuntime(composition=surface_world_spec(), observation_noise_mode=noise)
    session.reset(seed=7)
    for action in [*FEEDS, MEASUREMENT]:
        assert session.execute(action).info["transaction_status"] == "committed"
    snapshot = session.snapshot()
    expected = session.execute(MEASUREMENT)
    expected_state = session._state.to_dict(include_hidden=True)
    expected_noise = session.observation_noise_provenance()
    session.reset(seed=23)
    session.restore(snapshot)
    actual = session.execute(MEASUREMENT)
    assert actual == expected
    assert session._state.to_dict(include_hidden=True) == expected_state
    assert session.observation_noise_provenance() == expected_noise
    assert session.seed == 7
    session.restore(snapshot)
    assert session.execute(MEASUREMENT) == expected


def test_reading_public_views_and_snapshot_do_not_sample_or_charge():
    session = ChemWorldRuntime(composition=surface_world_spec())
    session.reset(seed=12)
    for action in [*FEEDS, MEASUREMENT]:
        session.execute(action)
    before = session._state.to_dict(include_hidden=True)
    rng = deepcopy(session._rng.bit_generator.state)
    count = session._operation_id
    reading = session.observe()
    for _ in range(3):
        assert session.observe() == reading
        session.snapshot()
        session.observation_view()
        session.task_prompt()
    assert session._operation_id == count
    assert session._rng.bit_generator.state == rng
    assert session._state.to_dict(include_hidden=True) == before
    assert reading.channel("CO_fraction").unit == "dimensionless"


def test_snapshot_preserves_campaign_events_and_rejected_attempts():
    card = CampaignResourceCard(
        card_id="session-test", operation_attempt_limit=15, vessel_start_limit=2,
        final_assay_limit=2, nonfinal_instrument_use_limit=0,
        stock_limits={"solvent_L": .04, "reagent_mol": .04},
    )
    session = ChemWorldRuntime(
        task_id="reaction-to-assay", episode_mode_override="campaign",
        campaign_resource_card=card, budget_override=15,
    )
    session.reset(seed=4)
    for action in [
        {"operation": "add_solvent", "solvent": 0, "volume_L": .02},
        {"operation": "add_reagent", "amount_mol": .01},
    ]:
        assert session.execute(action).info["transaction_status"] == "committed"
    snapshot = session.snapshot()
    actions = (
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    )
    results = [session.execute(action) for action in actions]
    ledger = session.campaign_resource_snapshot()
    state = session._state.to_dict(include_hidden=True)
    assert results[0].info["transaction_status"] == "campaign_resource_rejected"
    assert results[-1].info["experiment_ended"]
    session.restore(snapshot)
    assert [session.execute(action) for action in actions] == results
    assert session.campaign_resource_snapshot() == ledger
    assert session._state.to_dict(include_hidden=True) == state


def test_snapshot_cannot_silently_replace_another_sessions_world():
    left = ChemWorldRuntime(task_id="reaction-to-assay")
    right = ChemWorldRuntime(task_id="reaction-to-crystallization")
    with pytest.raises(ValueError, match="different experimental session"):
        right.restore(left.snapshot())


class SequenceAgent(BaseAgent):
    actions = (
        {"operation": "add_solvent", "solvent": 0, "volume_L": .02},
        {"operation": "add_reagent", "amount_mol": .01},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    )

    def act(self, history):
        return self.actions[len(history)]


def test_agent_runs_without_constructing_gym_and_replays_exactly(tmp_path, monkeypatch):
    def no_gym(*args, **kwargs):
        raise AssertionError("native Agent execution must not create a Gym environment")

    with monkeypatch.context() as patch:
        patch.setattr(gym, "make", no_gym)
        path = tmp_path / "native-agent.jsonl"
        records = run_agent(
            env_id="ChemWorld", agent=SequenceAgent(), task_id="reaction-to-assay",
            world_split="public-dev", budget=10, objective="balanced", seed=5,
            output_path=path,
        )
    assert len(records) == len(SequenceAgent.actions)
    report = verify_records(load_jsonl(path), tolerance=0.0)
    assert report.verified, report
    reference = ChemWorldRuntime(task_id="reaction-to-assay")
    reference.reset(seed=5)
    for record, action in zip(records, SequenceAgent.actions, strict=True):
        result = reference.execute(action)
        assert record.observation == observation_to_json(to_observation(result.observation.values))
        assert record.info["native_observation"] == result.info["native_observation"]
        assert record.info["state_delta_summary"] == result.info["state_delta_summary"]


def test_cli_render_executes_native_session(tmp_path, monkeypatch, capsys):
    from chemworld.cli import main

    def no_gym(*args, **kwargs):
        raise AssertionError("CLI render must use the native session")

    monkeypatch.setattr(gym, "make", no_gym)
    paths = []
    for index, action in enumerate(SequenceAgent.actions[:2]):
        path = tmp_path / f"action-{index}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        paths.append(str(path))
    main(["render", "--task", "reaction-to-assay", "--seed", "5", "--actions", *paths])
    assert "step: 2/" in capsys.readouterr().out


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_four_entry_points_preserve_readouts_outcomes_and_final_physics(
    tmp_path, monkeypatch, capsys, noise,
):
    from chemworld.cli import main

    actions = (
        {"operation": "unknown-action"},
        *SequenceAgent.actions[:3],
        {"operation": "wait", "duration_s": 1, "stirring_speed_rpm": 600},
        *SequenceAgent.actions[3:],
    )
    session = ChemWorldRuntime(task_id="reaction-to-assay", observation_noise_mode=noise)
    env = ChemWorldEnv(task_id="reaction-to-assay", observation_noise_mode=noise)
    session.reset(seed=5)
    env.reset(seed=5)
    results = []
    for action in actions:
        result = session.execute(action)
        _, reward, terminated, truncated, info = env.step(action)
        results.append(result)
        assert (reward, terminated, truncated) == (
            result.reward, result.terminated, result.truncated,
        )
        assert public_result(info) == public_result(result.info)
        assert session._state.to_dict() == env._state.to_dict()

    action_paths = []
    for index, action in enumerate(actions):
        path = tmp_path / f"action-{index}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        action_paths.append(str(path))
    cli_log, cli_snapshot = tmp_path / "cli.jsonl", tmp_path / "cli-snapshot.json"
    main([
        "render", "--task", "reaction-to-assay", "--seed", "5", "--noise-mode", noise,
        "--format", "json", "--trajectory", str(cli_log), "--snapshot", str(cli_snapshot),
        "--actions", *action_paths,
    ])
    assert len(json.loads(capsys.readouterr().out)["steps"]) == len(actions)
    cli = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(cli_snapshot))
    assert cli._state.to_dict() == session._state.to_dict()

    agent = SequenceAgent()
    agent.actions = actions
    agent_log = tmp_path / "agent.jsonl"
    final_states = []
    original_close = ChemWorldRuntime.close

    def capture_close(runtime):
        final_states.append(runtime._state.to_dict())
        original_close(runtime)

    with monkeypatch.context() as patch:
        patch.setattr(ChemWorldRuntime, "close", capture_close)
        agent_records = run_agent(
            env_id="ChemWorld", agent=agent, task_id="reaction-to-assay",
            world_split="public-dev", budget=30, objective="balanced", seed=5,
            observation_noise_mode=noise,
            output_path=agent_log,
        )
    assert final_states == [session._state.to_dict()]
    assert len(agent_records) == len(results)
    for records in (load_jsonl(cli_log), load_jsonl(agent_log)):
        assert len(records) == len(results)
        for record, result in zip(records, results, strict=True):
            outcome = record["environment_outcome"]
            assert (
                record["evaluation_outcome"]["online_transition_reward"],
                outcome["terminated"], outcome["truncated"],
            ) == (
                result.reward, result.terminated, result.truncated,
            )
            for key in (
                "native_observation", "state_delta_summary", "operation_outcome",
                "transaction_status", "actual_duration_s", "measurement_status",
                "measurement_cost", "sample_consumed",
            ):
                assert outcome[key] == result.info[key]
