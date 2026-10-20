"""Fixed item-41 histories across actual public adapters, not surrogate solvers."""

import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from types import SimpleNamespace

import pytest

from chemworld.agents.base import BaseAgent
from chemworld.cli import main as cli_main
from chemworld.data.logging import TrajectoryLogger, load_jsonl, to_builtin
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.envs.spaces import to_observation
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.examples.native_domains import WORKFLOWS, workflow
from chemworld.runtime import ChemWorldRuntime


def trace(session, result):
    return to_builtin(
        {
            "state": session._state.to_dict(),
            "reading": result.info["native_observation"],
            "queue": result.info["lab_clock"],
            "outcome": result.info["operation_outcome"],
            "duration_s": result.info["actual_duration_s"],
            "balance": result.info["material_balance"],
        }
    )


class FixedSequence(BaseAgent):
    name = "native-domain-integration-no-provider"

    def __init__(self, actions):
        self.actions = actions

    def act(self, history):
        return deepcopy(self.actions[len(history)])


def check_domain_transition(name, spec, before, after, action):
    """Independent ownership/ledger checks on the same actual adapter history."""
    if name == "membrane":
        from chemworld.runtime.membranes import unpack_membrane

        _, model, left = unpack_membrane(before)
        _, _, right = unpack_membrane(after)
        for species in ("water", "nacl", "foulant"):
            assert sum(n[species] for n in right.amounts_mol) == pytest.approx(
                sum(n[species] for n in left.amounts_mol), rel=1e-8, abs=1e-12
            )
        if action["operation"] == "run_flow":
            assert after.ledger.flow_pump_work_J > before.ledger.flow_pump_work_J
            assert after.ledger.time_s - before.ledger.time_s == action["duration_s"]
            path = before.equipment.equipment["membrane"].settings["flow_control"]["flow_path"]
            if path == "rinse":
                assert right.amounts_mol[4]["foulant"] < left.amounts_mol[4]["foulant"]
                assert (
                    after.inactive_vessels[model.fouling.waste_vessel_id].species_amounts["foulant"]
                    > before.inactive_vessels[model.fouling.waste_vessel_id].species_amounts[
                        "foulant"
                    ]
                )
            else:
                assert right.amounts_mol[4]["foulant"] > left.amounts_mol[4]["foulant"]
    elif name == "mineral":
        from chemworld.runtime.mineral_kinetics import mineral_geometry

        assert after.temperature_K == 298.15
        if action["operation"] == "route_material":
            assert after.species_amounts["Calcite"] == pytest.approx(
                before.species_amounts["Calcite"] * 0.75
            )
            assert mineral_geometry(after).reference_area_m2 == pytest.approx(
                mineral_geometry(before).reference_area_m2 * 0.75
            )
        if action["operation"] == "sample":
            assert after.species_amounts["Calcite"] == before.species_amounts["Calcite"]
            assert after.ledger.sample_consumed_L - before.ledger.sample_consumed_L == 0.001
        if action["operation"] in {"measure", "terminate"}:
            assert after.samples.to_dict() == before.samples.to_dict()
    elif name == "distillation" and action.get("full_reflux"):
        from chemworld.runtime.dynamic_column import read_column

        model = spec.physics.column_models[before.vessel_id]
        left, right = (read_column(state, model) for state in (before, after))
        assert left.amounts_mol[-1] == right.amounts_mol[-1]
        assert left.withdrawn_product_mol == right.withdrawn_product_mol
        for species in left.amounts_mol[0]:
            assert sum(n[species] for n in right.amounts_mol) == pytest.approx(
                sum(n[species] for n in left.amounts_mol), rel=1e-8, abs=1e-12
            )
        energy_change = sum(right.enthalpies_J) - sum(left.enthalpies_J)
        supplied = (
            after.ledger.energy_jacket_J
            - before.ledger.energy_jacket_J
            - after.ledger.heat_loss_J
            + before.ledger.heat_loss_J
        )
        assert energy_change == pytest.approx(supplied, rel=1e-6, abs=1e-6)
    elif name == "crystal" and action["operation"] in {"heat", "wait", "cool_crystallize"}:
        from chemworld.runtime.crystal_thermal import (
            crystal_inventory_energy,
            population_from_state,
        )

        model, thermal = (
            spec.physics.crystal_models["reactor"],
            spec.physics.thermal_models["reactor"],
        )
        population_from_state(after, model)
        change = crystal_inventory_energy(after, model, thermal) - crystal_inventory_energy(
            before, model, thermal
        )
        work = (
            after.ledger.energy_jacket_J
            - before.ledger.energy_jacket_J
            - after.ledger.heat_loss_J
            + before.ledger.heat_loss_J
        )
        assert change == pytest.approx(work, abs=1e-6, rel=1e-6)
        assert after.samples.to_dict() == before.samples.to_dict()
        assert after.species_amounts == pytest.approx(before.species_amounts, rel=1e-8, abs=1e-12)
    elif name == "shared-clock":
        assert len(set(after.vessel_elapsed_s.values())) == 1
        assert all(t == after.ledger.time_s for t in after.vessel_elapsed_s.values())
        if action["operation"] == "terminate":
            jobs = after.lab_clock.completed
            assert [job["sequence"] for job in jobs] == [0, 1]
            assert [job["started_at"] for job in jobs] == [2, 3]
            assert [job["available_at"] for job in jobs] == [3, 4]
            assert after.ledger.analysis_time_s == 2


@pytest.mark.parametrize("name", WORKFLOWS)
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_native_gym_cli_agent_and_all_logged_replays(name, noise, tmp_path, monkeypatch):
    spec, actions = workflow(name)
    histories, paths = {}, []

    def capture(kind):
        def factory(*args, **kwargs):
            session = ChemWorldRuntime(*args, **kwargs)
            original = session.execute

            def execute(action):
                result = original(action)
                assert result.info["transaction_status"] == "committed", {
                    key: result.info.get(key)
                    for key in ("error_message", "rollback_reason", "outcome_diagnostics")
                }
                assert result.info["measurement_status"] not in {"invalid", "failed"}
                histories[kind].append(trace(session, result))
                return result

            session.execute = execute
            return session

        histories[kind] = []
        return factory

    import chemworld.runtime as runtime_module

    # Native and Gym produce their own logs. CLI and Agent use their real logging.
    for kind in ("native", "gym"):
        factory = capture(kind)
        with monkeypatch.context() as patch:
            patch.setattr(runtime_module, "ChemWorldRuntime", factory)
            session = (
                factory(
                    composition=spec, observation_noise_mode=noise, observation_seed_override=12
                )
                if kind == "native"
                else ChemWorldEnv(
                    composition=spec, observation_noise_mode=noise, observation_seed_override=12
                )
            )
            session.reset(seed=7)
            path = tmp_path / f"{name}-{kind}.jsonl"
            with TrajectoryLogger(path) as logger:
                for step, action in enumerate(actions, 1):
                    if kind == "native":
                        before = session._state
                        result = session.execute(action)
                        check_domain_transition(name, spec, before, session._state, action)
                        observation = to_observation(result.observation.values)
                        reward, ended, truncated, info = (
                            result.reward,
                            result.terminated,
                            result.truncated,
                            result.info,
                        )
                    else:
                        observation, reward, ended, truncated, info = session.step(action)
                        histories[kind].append(trace(session, SimpleNamespace(info=info)))
                    logger.log(
                        task_info={**session.task_info(), **session.evaluator_provenance()},
                        step=step,
                        action=action,
                        observation=observation,
                        reward=reward,
                        terminated=ended,
                        truncated=truncated,
                        info=info,
                        agent_metadata={"agent_id": "fixed-integration"},
                    )
            assert ended and not truncated
            paths.append(path)
            session.close()
    expected = histories["native"]

    spec_path = tmp_path / "composition.json"
    spec_path.write_text(json.dumps(spec.to_dict()))
    action_paths = []
    for index, action in enumerate(actions):
        path = tmp_path / f"action-{index}.json"
        path.write_text(json.dumps(action))
        action_paths.append(str(path))
    cli_log = tmp_path / "cli.jsonl"
    with monkeypatch.context() as patch:
        patch.setattr(runtime_module, "ChemWorldRuntime", capture("cli"))
        with redirect_stdout(io.StringIO()):
            cli_main(
                [
                    "render",
                    "--composition",
                    str(spec_path),
                    "--seed",
                    "7",
                    "--noise-mode",
                    noise,
                    "--observation-seed",
                    "12",
                    "--actions",
                    *action_paths,
                    "--format",
                    "json",
                    "--trajectory",
                    str(cli_log),
                ]
            )
    paths.append(cli_log)

    import chemworld.eval.runner as runner

    agent_log = tmp_path / "agent.jsonl"
    with monkeypatch.context() as patch:
        patch.setattr(runner, "ChemWorldRuntime", capture("agent"))
        run_agent(
            env_id="ChemWorld",
            agent=FixedSequence(actions),
            composition=spec.to_dict(),
            world_split="public-dev",
            budget=80,
            objective="balanced",
            seed=7,
            observation_noise_mode=noise,
            observation_seed=12,
            output_path=agent_log,
        )
    paths.append(agent_log)
    for kind, history in histories.items():
        assert history == expected, kind
    for path in paths:
        replay = verify_records(load_jsonl(path), tolerance=1e-8)
        assert replay.verified, (path.name, replay.mismatches)
        assert replay.checked_steps == len(actions)


@pytest.mark.parametrize("name", WORKFLOWS)
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
def test_snapshot_resume_in_middle_of_real_workflow(name, noise, tmp_path):
    import subprocess
    import sys

    spec, actions = workflow(name)
    original = ChemWorldRuntime(
        composition=spec, observation_noise_mode=noise, observation_seed_override=12
    )
    original.reset(seed=7)
    middle = 4 if name == "shared-clock" else len(actions) // 2
    for action in actions[:middle]:
        assert original.execute(action).info["transaction_status"] == "committed"
    snapshot = original.snapshot().to_dict()
    if name == "shared-clock":
        assert original._state.lab_clock.queued and original._state.lab_clock.running
        assert original._state.process.stochastic_continuation
        assert all("observation" not in job for job in original.lab_status()["running"])
    source, destination = tmp_path / "resume.json", tmp_path / "continued.json"
    source.write_text(json.dumps({"snapshot": snapshot, "actions": actions[middle:]}))
    code = """
import json, sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
from chemworld.data.logging import to_builtin
payload = json.loads(Path(sys.argv[1]).read_text())
session = ChemWorldRuntime.from_snapshot(payload['snapshot'])
history = []
for action in payload['actions']:
    result = session.execute(action)
    assert result.info['transaction_status'] == 'committed'
    history.append(to_builtin({'state': session._state.to_dict(),
        'reading': result.info['native_observation'], 'queue': result.info['lab_clock'],
        'outcome': result.info['operation_outcome'], 'duration_s': result.info['actual_duration_s'],
        'balance': result.info['material_balance']}))
Path(sys.argv[2]).write_text(json.dumps(history, allow_nan=False))
session.close()
"""
    try:
        expected = []
        for action in actions[middle:]:
            result = original.execute(action)
            assert result.info["transaction_status"] == "committed"
            expected.append(trace(original, result))
        subprocess.run(
            [sys.executable, "-c", code, str(source), str(destination)], check=True, timeout=180
        )
        assert json.loads(destination.read_text()) == json.loads(json.dumps(expected))
    finally:
        original.close()


def test_heater_unique_identity_calibration_and_domain_histories():
    spec, actions = workflow("heater")
    r = ChemWorldRuntime(composition=spec)
    r.reset(seed=41)
    initial = r._state.to_dict()
    try:
        for action in actions:
            before = r._state
            result = r.execute(action)
            assert result.info["transaction_status"] == "committed"
            states = {r._state.vessel_id: r._state, **r._state.inactive_vessels}
            heaters = [
                e
                for s in states.values()
                for e in s.equipment.equipment.values()
                if e.equipment_type == "thermal_device"
            ]
            assert len(heaters) == 1 and heaters[0].equipment_id == "reactor:heater"
            assert heaters[0].settings["thermal_device"]["calibration_factor"] == 1.07
            if action["operation"] == "reconnect_heater":
                old = {before.vessel_id: before, **before.inactive_vessels}
                assert r._state.ledger == before.ledger
                assert all(states[k].species_amounts == s.species_amounts for k, s in old.items())
                assert all(states[k].temperature_K == s.temperature_K for k, s in old.items())
                for key, state in old.items():
                    assert states[key].state_slots == state.state_slots
                    assert states[key].phases == state.phases
                    assert states[key].process == state.process
                    for equipment_id, equipment in state.equipment.equipment.items():
                        if equipment.equipment_type != "thermal_device":
                            assert states[key].equipment.equipment[equipment_id] == equipment
        assert heaters[0].attached_vessel_id == "cell"
        assert states["reactor"].species_amounts["CO2"] > 0
        assert (
            states["adsorber"].species_amounts["A"]
            < initial["inactive_vessels"]["adsorber"]["species_amounts"]["A"]
        )
        assert states["cell"].temperature_K > 298.15
        assert r._state.ledger.energy_jacket_J > 0
        assert r._state.vessel_elapsed_s["reactor"] == 15
        assert r._state.vessel_elapsed_s["adsorber"] == 21
        assert r._state.vessel_elapsed_s["cell"] == 21
    finally:
        r.close()


@pytest.mark.parametrize("case", ("active-heater", "missing-field", "no-domain"))
def test_invalid_heater_reconnection_preserves_physics(case):
    spec, _ = workflow("heater")
    r = ChemWorldRuntime(composition=spec)
    r.reset(seed=41)
    if case != "active-heater":
        assert r.execute({"operation": "pause_control"}).info["transaction_status"] == "committed"
    before = r._state
    action = {"operation": "reconnect_heater", "vessel": "adsorber"}
    if case == "missing-field":
        action.pop("vessel")
    if case == "no-domain":
        action["vessel"] = "co_supply"
    result = r.execute(action)
    assert result.info["transaction_status"] != "committed"
    assert before.species_amounts == r._state.species_amounts
    assert before.equipment == r._state.equipment
    assert before.inactive_vessels == r._state.inactive_vessels
    assert before.ledger.time_s == r._state.ledger.time_s
    r.close()


@pytest.mark.parametrize(
    "case",
    (
        "wrong-unit",
        "undeclared-species",
        "missing-local-species",
        "missing-caloric-parameter",
        "unsupported-coupling",
    ),
)
def test_invalid_compositions_do_not_silently_fallback(case):
    spec, _ = workflow("shared-clock" if case == "unsupported-coupling" else "heater")
    payload = spec.to_dict()
    state = payload["physics"]["initial_states"]["reactor"]
    if case == "wrong-unit":
        state["units"]["amount"] = "mmol"
    elif case == "undeclared-species":
        state["species_amounts"]["not-declared"] = 1
    elif case == "missing-local-species":
        state["species_amounts"].pop("CO2")
    elif case == "missing-caloric-parameter":
        payload["physics"]["thermal_models"]["reactor"]["thermodynamics"]["species"].pop("CO")
    else:
        payload["components"].append(
            {"id": "bad-heat", "kind": "thermal", "vessel": "reactor", "parameters": {}}
        )
        payload["physics"]["thermal_models"]["reactor"] = payload["physics"]["thermal_models"][
            "continuous"
        ]
    with pytest.raises(ValueError):
        ChemWorldRuntime(composition=payload)


def test_same_world_two_task_bindings_preserve_physics_and_equipment_history():
    from chemworld.world.composition import compile_world_composition

    spec, actions = workflow("heater")
    compiled = compile_world_composition(spec)
    task = spec.task.to_dict()
    task["objective"] = "independent observation task on the same apparatus"
    rebound = compiled.bind_task(task)
    assert rebound.spec.physics.to_dict() == compiled.spec.physics.to_dict()
    sessions = [ChemWorldRuntime(composition=c) for c in (compiled, rebound)]
    try:
        for r in sessions:
            r.reset(seed=41)
            for action in actions[:10]:
                assert r.execute(action).info["transaction_status"] == "committed"
        assert sessions[0]._state.to_dict() == sessions[1]._state.to_dict()
        r = sessions[0]
        old = r._state.equipment
        r.reset(options={"scope": "material", "vessel_ids": ["adsorber"]})
        assert r._state.equipment.equipment["reactor:heater"] == old.equipment["reactor:heater"]
        assert r._state.equipment.equipment["adsorber:bed"] == old.equipment["adsorber:bed"]
        assert all(
            r._state.equipment.equipment[k].settings == record.settings
            for k, record in old.equipment.items()
        )
    finally:
        for r in sessions:
            r.close()


def test_capability_export_follows_installed_declarations():
    from pathlib import Path

    from chemworld.examples.native_capabilities import capability_rows, markdown
    from chemworld.world.domain_modules import DomainModuleRegistry

    registry = DomainModuleRegistry.default()
    rows = capability_rows(registry)
    assert {r["module"] for r in rows} == {m.module_id for m in registry.modules}
    for row, module in zip(rows, registry.modules, strict=True):
        assert row["scope"] == list(module.scope_notes)
        assert set(row["operations"]) == {op.operation.id for op in module.operations}
        assert set(row["components"]) == {key for key, _ in module.components}
    document = Path(__file__).parents[1] / "docs/native_capabilities.md"
    assert document.read_text() == markdown()


@pytest.mark.parametrize("name", ("heater", "shared-clock", "solid"))
def test_bound_capability_export_uses_public_ranges_without_mutation(
    name, tmp_path, capsys, monkeypatch
):
    from chemworld.examples.native_capabilities import bound_capabilities, main

    spec, _ = workflow(name)
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=7)
        before = session._state.to_dict()
        exported = bound_capabilities(session)
        assert exported["actions"] == session.available_actions(include_invalid=True)
        assert session._state.to_dict() == before
        path = tmp_path / "world.json"
        path.write_text(json.dumps(spec.to_dict()))
        monkeypatch.setattr("sys.argv", ["native_capabilities", "--composition", str(path)])
        main()
        assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(exported))
    finally:
        session.close()


def test_wrong_connection_unit_is_rejected_by_actual_port_contract():
    spec, _ = workflow("mineral")
    payload = spec.to_dict()
    payload["connections"][0]["unit"] = "kg"
    with pytest.raises(ValueError, match="unit"):
        ChemWorldRuntime(composition=payload)
