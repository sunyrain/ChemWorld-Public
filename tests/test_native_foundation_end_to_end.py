"""Fixed P1 workflows across real public entry points.

Numerical/process/readout error injections are interface fixtures. The stopped
case uses a real thermal event; none of these tests adds a failure mechanism.
"""

import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.agents.base import BaseAgent
from chemworld.cli import main as cli_main
from chemworld.data.logging import TrajectoryLogger, load_jsonl, to_builtin
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.envs.spaces import to_observation
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.foundation import Observation, ObservationChannel
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.coupled_thermal import CoupledThermalRuntime
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.observation_services import ChemWorldObservationKernel
from chemworld.world.surface_world import surface_world_spec


def actions(assay):
    result = [
        {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5},
        {"operation": "add_component", "component": "O2-feed", "amount_mol": 3e-5},
        {"operation": "heat", "target_temperature_K": 380, "duration_s": 20,
         "stirring_speed_rpm": 600},
        {"operation": "measure", "instrument": "gas_pressure"},
        {"operation": "route_material", "connection": "reactor-to-receiver",
         "transfer_fraction": .2, "mixing": 1},
        # Valid syntax, physically rejected: the destination already contains gas.
        {"operation": "route_material", "connection": "reactor-to-receiver",
         "transfer_fraction": .2, "mixing": 0},
        {"operation": "wait", "duration_s": 2, "stirring_speed_rpm": 600},
        {"operation": "measure", "instrument": "gas_composition"},
        {"operation": "select_vessel", "vessel": "receiver"},
        {"operation": "wait", "duration_s": 3, "stirring_speed_rpm": 600},
        {"operation": "measure", "instrument": "gas_composition"},
        {"operation": "terminate"},
    ]
    if assay:
        result.append({"operation": "measure", "instrument": "gas_final_assay"})
    return result


def trace(session, info):
    return to_builtin({
        "state": session._state.to_dict(), "reading": info["native_observation"],
        "outcome": info["operation_outcome"], "duration_s": info["actual_duration_s"],
        "measurement": info["measurement_status"], "transaction": info["transaction_status"],
        "material_balance": info["material_balance"],
    })


def install_contract_fixture(monkeypatch, case):
    if case in {"numerical_failure", "process_failure"}:
        original = CoupledThermalRuntime.advance
        def advance(self, state, action, **kwargs):
            if action.get("duration_s") != 2:
                return original(self, state, action, **kwargs)
            if case == "numerical_failure":
                return ModelProviderResult(
                    outputs={}, diagnostics={"contract_fixture": case}, success=False,
                    outcome=case, failure_reason="declared_contract_fixture",
                )
            result = original(self, state, action, **kwargs)
            return replace(result, success=False, outcome=case,
                           failure_reason="declared_contract_fixture",
                           diagnostics={**result.diagnostics, "contract_fixture": case})
        monkeypatch.setattr(CoupledThermalRuntime, "advance", advance)
    elif case == "readout_invalid":
        original = ChemWorldObservationKernel.observe
        def observe(self, state, action, rng):
            if action.get("instrument") == "gas_pressure":
                return Observation(channels=(ObservationChannel("pressure", "Pa", float("inf")),),
                                   instrument_id="gas_pressure", cost=.02)
            return original(self, state, action, rng)
        monkeypatch.setattr(ChemWorldObservationKernel, "observe", observe)


class SequenceAgent(BaseAgent):
    name = "native-p1-fixed-sequence"

    def __init__(self, sequence):
        self.sequence = sequence

    def act(self, history):
        return deepcopy(self.sequence[len(history)])


CASES = [
    (case, noise, assay, False)
    for case in ("normal", "numerical_failure", "process_failure", "readout_invalid", "stopped")
    for noise in ("sequential", "keyed")
    for assay in ((False, True) if case == "normal" else (True,))
] + [pytest.param("normal", noise, False, True, id=f"multisite-{noise}")
     for noise in ("sequential", "keyed")]


@pytest.mark.parametrize("case,noise,assay,multisite", CASES)
def test_fixed_histories_match_native_gym_cli_agent_and_replay(
    tmp_path, monkeypatch, case, noise, assay, multisite,
):
    install_contract_fixture(monkeypatch, case)
    spec = surface_world_spec(terminal_assay=assay)
    if multisite:
        from chemworld.physchem.surface_specs import MultisiteSurfaceSpec
        spec = surface_world_spec(surface_spec=MultisiteSurfaceSpec(
            interaction_matrix_J_mol=(
                (5000, 0, 2500, 0), (0, 5000, 0, 2500),
                (2500, 0, 5000, 0), (0, 2500, 0, 5000),
            )))
    if case == "stopped":
        models = dict(spec.physics.thermal_models)
        models["reactor"] = replace(models["reactor"], stop_temperature_K=355)
        spec = replace(spec, physics=replace(spec.physics, thermal_models=models))
    sequence = actions(assay)
    kwargs = {"composition": spec, "observation_noise_mode": noise, "seed": 12}
    expected = []
    paths = []
    for kind in ("native", "gym"):
        session = ChemWorldRuntime(**kwargs) if kind == "native" else ChemWorldEnv(**kwargs)
        session.reset(seed=12)
        history = []
        path = tmp_path / f"{kind}.jsonl"
        with TrajectoryLogger(path) as logger:
            for index, action in enumerate(sequence, 1):
                if kind == "native":
                    result = session.execute(action)
                    observation = to_observation(result.observation.values)
                    reward, ended, truncated, info = (
                        result.reward, result.terminated, result.truncated, result.info,
                    )
                else:
                    observation, reward, ended, truncated, info = session.step(action)
                history.append(trace(session, info))
                logger.log(task_info={**session.task_info(), **session.evaluator_provenance()},
                           step=index, action=action, observation=observation, reward=reward,
                           terminated=ended, truncated=truncated, info=info,
                           agent_metadata={"agent_id": f"p1-{kind}"})
        if kind == "native":
            expected = history
        else:
            assert history == expected
        assert ended and not truncated
        paths.append(path)
        session.close()

    # Observe the actual session constructed by each adapter, without replacing
    # its execute implementation or introducing a second physical step loop.
    def capture_factory(output):
        def factory(*args, **kw):
            session = ChemWorldRuntime(*args, **kw)
            execute = session.execute
            def captured(action):
                result = execute(action)
                output.append(trace(session, result.info))
                return result
            session.execute = captured
            return session
        return factory

    import chemworld.runtime as runtime_module
    cli_history = []
    with monkeypatch.context() as patch:
        patch.setattr(runtime_module, "ChemWorldRuntime", capture_factory(cli_history))
        composition_path = tmp_path / "composition.json"
        composition_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
        action_paths = []
        for index, action in enumerate(sequence):
            action_path = tmp_path / f"action-{index}.json"
            action_path.write_text(json.dumps(action), encoding="utf-8")
            action_paths.append(str(action_path))
        path = tmp_path / "cli.jsonl"
        with redirect_stdout(io.StringIO()) as output:
            cli_main(["render", "--composition", str(composition_path), "--seed", "12",
                      "--noise-mode", noise, "--actions", *action_paths, "--format", "json",
                      "--trajectory", str(path), "--snapshot", str(tmp_path / "cli-state.json")])
        assert len(json.loads(output.getvalue())["steps"]) == len(sequence)
        assert cli_history == expected
        paths.append(path)

    import chemworld.eval.runner as runner
    agent_history = []
    with monkeypatch.context() as patch:
        patch.setattr(runner, "ChemWorldRuntime", capture_factory(agent_history))
        path = tmp_path / "agent.jsonl"
        run_agent(env_id="ChemWorld", agent=SequenceAgent(sequence), composition=spec.to_dict(),
                  world_split="public-dev", budget=40, objective="balanced", seed=12,
                  observation_noise_mode=noise, output_path=path)
        assert agent_history == expected
        paths.append(path)

    assert any(row["outcome"] == "rejected" for row in expected)
    if case in {"numerical_failure", "process_failure", "stopped"}:
        assert any(row["outcome"] == case for row in expected)
    if case == "readout_invalid":
        assert any(row["measurement"] == "invalid" for row in expected)
    for path in paths:
        # Sorted JSON keys can change floating point summation order when the
        # composition is rebuilt. Keep the physical replay tolerance much
        # tighter than the CLI default, while entry histories above stay exact.
        report = verify_records(load_jsonl(path), tolerance=1e-8)
        assert report.verified, (path.name, report.mismatches)


def test_installed_style_examples_reuse_device_and_change_only_task_ending():
    from chemworld.examples.reuse_native_equipment import run_example
    report = run_example()
    assert len(report["configurations"]) == 4
    assert all(row["same_heater_after_material_initialization"] for row in report["configurations"])
    assert all(value for key, value in report["task_binding"].items() if key != "assay_readings")
