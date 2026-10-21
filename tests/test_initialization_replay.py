"""Evaluator boundaries survive ordinary logs without entering Agent payloads."""

from copy import deepcopy

import pytest

from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.envs.spaces import to_observation
from chemworld.eval.verify import verify_records
from chemworld.examples.multi_device_lab import multi_device_lab_world
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.reports import sanitize_agent_info


@pytest.mark.parametrize("payload", (
    [], {True: {"scope": "material"}}, {-1: {"scope": "material"}},
    {"01": {"scope": "material"}}, {0: {"scope": "world"}},
    {0: {"scope": "material", "seed": 12}},
    {0: {"scope": "material"}, "0": {"scope": "vessel"}},
))
def test_evaluator_boundary_schedule_rejects_ambiguous_or_world_resets(payload):
    from chemworld.runtime.lifecycle import initialization_boundaries

    with pytest.raises(ValueError):
        initialization_boundaries(payload)


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_cli_and_agent_execute_the_same_explicit_evaluator_boundary(noise, tmp_path, capsys):
    from chemworld.agents.base import BaseAgent
    from chemworld.cli import main
    from chemworld.eval.runner import run_agent

    actions = [{"operation": "wait", "duration_s": 1},
               {"operation": "wait", "duration_s": 1}, {"operation": "terminate"}]
    boundaries = {1: {"scope": "material", "vessel_ids": ["reactor"]}}

    class Sequence(BaseAgent):
        def act(self, history):
            return deepcopy(actions[len(history)])

    import json

    spec = multi_device_lab_world()
    composition, initialization = tmp_path / "world.json", tmp_path / "boundaries.json"
    composition.write_text(json.dumps(spec.to_dict()))
    initialization.write_text(json.dumps(boundaries))
    paths = []
    for index, action in enumerate(actions):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(action))
        paths.append(str(path))
    cli_path, agent_path = tmp_path / "cli.jsonl", tmp_path / "agent.jsonl"
    main(["render", "--composition", str(composition), "--seed", "7",
          "--observation-seed", "12", "--noise-mode", noise, "--actions", *paths,
          "--initializations", str(initialization), "--trajectory", str(cli_path),
          "--format", "json"])
    assert len(json.loads(capsys.readouterr().out)["steps"]) == 3
    run_agent(env_id="ChemWorld", agent=Sequence(), world_split="public-dev", budget=5,
              objective="balanced", seed=7, composition=spec.to_dict(),
              observation_seed=12, observation_noise_mode=noise, output_path=agent_path,
              initialization_after_action=boundaries)
    cli, agent = load_jsonl(cli_path), load_jsonl(agent_path)
    assert len(cli) == len(agent) == 3
    assert cli[0]["initialization_replay"] is None
    assert len(cli[1]["initialization_replay"]) == 1
    assert cli[1]["initialization_replay"] == agent[1]["initialization_replay"]
    for records in (cli, agent):
        replay = verify_records(records, tolerance=0)
        assert replay.verified, replay.mismatches


@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("scope", ("material", "vessel", "equipment"))
def test_private_initialization_replay_and_receipt_tampering(scope, noise, tmp_path):
    session = ChemWorldRuntime(composition=multi_device_lab_world(),
                               observation_noise_mode=noise, observation_seed_override=12)
    session.reset(seed=7)
    path = tmp_path / "boundary.jsonl"
    try:
        with TrajectoryLogger(path) as logger:
            for index, action in enumerate((
                {"operation": "queue_lab_heat", "device": "heater-a",
                 "target_temperature_K": 320, "duration_s": 2},
                {"operation": "wait", "duration_s": 2},
                {"operation": "wait", "duration_s": 1},
                {"operation": "terminate"},
            ), 1):
                if index == 3:
                    options = {"scope": scope, "vessel_ids": ["reactor"]}
                    if scope == "equipment":
                        record = session._state.equipment.equipment["heater-a"].to_dict()
                        record["equipment_id"] = "new-heater"
                        options = {"scope": scope, "equipment_id": "heater-a",
                                   "replacement": record}
                    session.reset(options=options)
                result = session.execute(action)
                assert result.info["transaction_status"] == "committed"
                provenance = session.evaluator_provenance()
                assert "initialization_replay" not in result.info
                assert "initialization_replay" not in sanitize_agent_info(provenance)
                logger.log(task_info={**session.task_info(), **provenance}, step=index,
                           action=action, observation=to_observation(result.observation.values),
                           reward=result.reward, terminated=result.terminated,
                           truncated=result.truncated, info=result.info,
                           agent_metadata={"agent_id": "fixed-boundary-check"})
        records = load_jsonl(path)
        verified = verify_records(records, tolerance=0)
        assert verified.verified, verified.mismatches
        assert verified.checked_steps == 4
        altered = deepcopy(records)
        altered[2]["initialization_replay"][0]["receipt_sha256"] = "0" * 64
        rejected = verify_records(altered, tolerance=0)
        assert not rejected.verified
        assert rejected.checked_steps == 2
        assert rejected.mismatches[-1]["field"] == "initialization_replay"
    finally:
        session.close()
