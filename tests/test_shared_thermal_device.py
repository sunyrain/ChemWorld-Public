"""Actual heat/wait transactions share finite-power equipment and coupled physics."""

from dataclasses import replace
from math import exp

import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.coupled_system import ConstantCvThermodynamics
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.surface_rates import gas_amounts_from_pressure
from chemworld.physchem.surface_specs import (
    bulk_co_oxidation_network,
    make_surface_slot,
    surface_species,
    surface_species_energies,
)
from chemworld.world.physical_world import CompositionPhysics


def make_env(*, bulk=False, surface=False, device=None, temperature=350.0, budget=10):
    network = replace(
        bulk_co_oxidation_network(),
        species=surface_species(),
        reactions=bulk_co_oxidation_network().reactions if bulk else (),
    )
    device = device or ThermalDeviceSpec()
    equipment = {
        "heater": EquipmentRecord(
            "heater",
            "thermal_device",
            attached_vessel_id="reactor",
            settings={"thermal_device": device.to_dict()},
        )
    }
    slots = ()
    if surface:
        equipment["catalyst"] = EquipmentRecord(
            "catalyst", "fixed_catalyst", attached_vessel_id="reactor"
        )
        slots = (make_surface_slot("catalyst", "reactor"),)
    amounts = gas_amounts_from_pressure(
        {"CO": 0.2, "O2": 0.1, "N2": 0.7},
        pressure_Pa=100000,
        temperature_K=temperature,
        volume_L=0.01,
    )
    state = WorldState(
        amounts,
        0.01,
        temperature,
        100000,
        "gas",
        "reactor",
        units={
            "amount": "mol",
            "volume": "L",
            "temperature": "K",
            "pressure": "Pa",
            "time": "s",
            "cost": "currency",
            "risk": "risk",
        },
        equipment=EquipmentLedger(equipment),
        state_slots=StateSlots(slots),
    )
    model = ThermalSystemSpec(ConstantCvThermodynamics(surface_species_energies(), 20.0))
    kinds = ["thermal"] + (["reaction"] if bulk else []) + (["surface_reaction"] if surface else [])
    payload = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "shared-heater-test",
        "components": [{"kind": kind, "parameters": {}} for kind in kinds],
        "physics": CompositionPhysics(
            network, {"reactor": state}, thermal_models={"reactor": model}
        ).to_dict(),
        "task": {
            "objective": "observe thermal response",
            "budget": budget,
            "operations": ["heat", "wait", "terminate"],
            "instruments": [],
        },
    }
    env = ChemWorldEnv(composition=payload)
    env.reset(seed=7)
    return env


def advance(env, seconds, *, target=None):
    action = {
        "operation": "wait" if target is None else "heat",
        "duration_s": seconds,
        "stirring_speed_rpm": 600.0,
    }
    if target is not None:
        action["target_temperature_K"] = target
    result = env.step(action)
    assert result[4]["transaction_status"] == "committed", result[4]
    return result


@pytest.mark.parametrize(
    "bulk,surface", [(False, False), (True, False), (False, True), (True, True)]
)
def test_same_heater_and_operations_use_one_clock_and_energy_balance(bulk, surface):
    env = make_env(bulk=bulk, surface=surface)
    before = env._state
    result = advance(env, 10, target=380)
    assert before.temperature_K < env._state.temperature_K < 380
    assert env._state.ledger.time_s == pytest.approx(10)
    assert 0 < env._state.ledger.energy_jacket_J <= 50 + 1e-8
    assert abs(result[4]["outcome_diagnostics"]["energy_balance_residual_J"]) < 1e-6
    assert all(check["passed"] for check in result[4]["constitution_checks"])
    advance(env, 10)
    assert env._state.ledger.time_s == pytest.approx(20)
    assert env._state.equipment.equipment["heater"].settings["thermal_device"]["setpoint_K"] == 380
    if bulk or surface:
        assert env._state.species_amounts["CO2"] > before.species_amounts["CO2"]


def test_inert_response_matches_analytic_finite_conductance_solution():
    env = make_env(device=ThermalDeviceSpec(setpoint_K=360))
    initial = env._state
    capacity = 20 + sum(
        initial.species_amounts[key] * surface_species_energies()[key].cv_J_mol_K
        for key in initial.species_amounts
    )
    equilibrium = (0.2 * 360 + 0.01 * 300) / 0.21
    expected = equilibrium + (350 - equilibrium) * exp(-0.21 * 60 / capacity)
    advance(env, 60)
    assert env._state.temperature_K == pytest.approx(expected, abs=2e-6)


def test_isothermal_chemistry_records_compensating_heat():
    env = make_env(surface=True, device=ThermalDeviceSpec(mode="isothermal"))
    advance(env, 30)
    assert env._state.temperature_K == pytest.approx(350)
    assert env._state.species_amounts["CO2"] > 0
    assert env._state.ledger.energy_jacket_J == pytest.approx(
        env._state.ledger.heat_loss_J - env._state.ledger.heat_reaction_J,
        abs=1e-6,
    )


def test_disabled_heater_keeps_reactions_and_ambient_heat_exchange():
    env = make_env(surface=True, device=ThermalDeviceSpec(enabled=False))
    advance(env, 30)
    assert env._state.ledger.energy_jacket_J == 0
    assert env._state.ledger.heat_loss_J > 0
    assert env._state.species_amounts["CO2"] > 0


def test_pause_and_resume_existing_control_actions_only_change_device_enable():
    original = make_env(surface=True)
    compiled = original.compiled_composition
    task = compiled.spec.task.to_dict()
    task["operations"] += ["pause_control", "resume_control"]
    env = ChemWorldEnv(composition=compiled.bind_task(task))
    env.reset(seed=7)
    advance(env, 10, target=380)
    before_heat = env._state.ledger.energy_jacket_J
    assert env.step({"operation": "pause_control"})[4]["transaction_status"] == "committed"
    before_co2 = env._state.species_amounts["CO2"]
    advance(env, 10)
    assert env._state.ledger.energy_jacket_J == before_heat
    assert env._state.species_amounts["CO2"] > before_co2
    assert env.step({"operation": "resume_control"})[4]["transaction_status"] == "committed"
    advance(env, 10)
    assert env._state.ledger.energy_jacket_J > before_heat


def test_event_stop_keeps_actual_time_energy_and_inventory():
    from chemworld.world.composition import compile_world_composition

    env = make_env(surface=True)
    spec = env.compiled_composition.spec
    models = {"reactor": replace(spec.physics.thermal_models["reactor"], stop_temperature_K=350.1)}
    env = ChemWorldEnv(
        composition=compile_world_composition(
            replace(
                spec,
                physics=replace(spec.physics, thermal_models=models),
            )
        )
    )
    env.reset(seed=7)
    result = advance(env, 30, target=380)
    assert result[4]["operation_outcome"] == "stopped"
    assert 0 < env._state.ledger.time_s < 30
    assert env._state.temperature_K == pytest.approx(350.1, abs=1e-6)
    assert env._state.ledger.energy_jacket_J <= 5 * env._state.ledger.time_s + 1e-8


def test_device_reconnection_preserves_calibration_and_leaves_material_in_place():
    from chemworld.runtime.coupled_thermal import reconnect_thermal_device

    env = make_env(device=ThermalDeviceSpec(calibration_factor=1.2))
    source = env._state
    destination = source.replace(
        vessel_id="receiver",
        vessels=None,
        phases=None,
        thermal=None,
        species_amounts=dict.fromkeys(source.species_amounts, 0.0),
        pressure_Pa=0.0,
        equipment=EquipmentLedger({}),
    )
    before = source.replace(inactive_vessels={"receiver": destination})
    after = reconnect_thermal_device(before, "heater", "receiver")
    assert after.species_amounts == before.species_amounts
    assert after.inactive_vessels["receiver"].species_amounts == destination.species_amounts
    assert after.ledger == before.ledger
    assert "heater" not in after.equipment.equipment
    heater = after.inactive_vessels["receiver"].equipment.equipment["heater"]
    assert heater.settings == source.equipment.equipment["heater"].settings
    assert heater.attached_vessel_id == "receiver"
    from chemworld.runtime.transactions import StatePatch

    committed = env.runtime.transaction_manager.commit(
        state=before,
        operation_type="reconnect_thermal_device",
        events=(),
        patches=(StatePatch("replace_state", ("equipment",), after),),
        actual_duration_s=0,
    )
    assert committed.transaction_status == "committed", committed.constitution_checks
    # Bind the reconnected apparatus and exercise the existing public heat action.
    # This does not claim an agent-facing hot-plug action or a full session restore.
    payload = env.compiled_composition.spec.to_dict()
    payload["physics"]["initial_states"] = {
        "reactor": after.replace(inactive_vessels={}).to_dict(),
        "receiver": after.inactive_vessels["receiver"].to_dict(),
    }
    payload["physics"]["thermal_models"]["receiver"] = payload["physics"]["thermal_models"][
        "reactor"
    ]
    payload["vessels"].append({**payload["vessels"][0], "id": "receiver"})
    payload["components"].append(
        {"id": "receiver-thermal", "kind": "thermal", "vessel": "receiver", "parameters": {}}
    )
    payload["task"]["operations"].append("select_vessel")
    rebuilt = ChemWorldEnv(composition=payload)
    rebuilt.reset(seed=7)
    assert (
        rebuilt.step({"operation": "select_vessel", "vessel": "receiver"})[4]["transaction_status"]
        == "committed"
    )
    advance(rebuilt, 10, target=380)
    assert rebuilt._state.temperature_K > destination.temperature_K
    assert (
        rebuilt._state.equipment.equipment["heater"].settings["thermal_device"][
            "calibration_factor"
        ]
        == 1.2
    )


def test_heat_wait_partition_matches_single_shared_advance():
    split = make_env(bulk=True, surface=True)
    whole = make_env(bulk=True, surface=True)
    advance(split, 10, target=380)
    advance(split, 10)
    advance(whole, 20, target=380)
    assert split._state.temperature_K == pytest.approx(whole._state.temperature_K, abs=1e-5)
    assert split._state.species_amounts == pytest.approx(whole._state.species_amounts, abs=1e-10)
    assert split._state.ledger.energy_jacket_J == pytest.approx(
        whole._state.ledger.energy_jacket_J, abs=1e-5
    )


def test_empty_thermal_body_does_not_need_fictitious_reagent():
    env = make_env()
    compiled = env.compiled_composition
    initial = compiled.spec.physics.initial_states["reactor"]
    initial = initial.replace(
        species_amounts=dict.fromkeys(initial.species_amounts, 0.0), pressure_Pa=0.0
    )
    physics = replace(compiled.spec.physics, initial_states={"reactor": initial})
    from chemworld.world.composition import compile_world_composition

    env = ChemWorldEnv(
        composition=compile_world_composition(replace(compiled.spec, physics=physics))
    )
    env.reset(seed=7)
    advance(env, 10, target=380)
    assert sum(env._state.species_amounts.values()) == 0
    assert env._state.temperature_K > 350
    result = env.step({"operation": "terminate"})
    assert result[2] and result[4]["transaction_status"] == "committed"


@pytest.mark.parametrize("mode,target", [("finite_power", 440), ("isothermal", 380)])
def test_invalid_control_is_rejected_without_altering_committed_physics(mode, target):
    env = make_env(surface=True, device=ThermalDeviceSpec(mode=mode))
    advance(env, 5)
    before = env._state
    info = env.step(
        {
            "operation": "heat",
            "duration_s": 10,
            "target_temperature_K": target,
            "stirring_speed_rpm": 600,
        }
    )[4]
    assert info["operation_outcome"] == "rejected"
    assert info["actual_duration_s"] == 0
    assert env._state.temperature_K == before.temperature_K
    assert env._state.species_amounts == before.species_amounts
    assert env._state.state_slots == before.state_slots
    assert env._state.equipment == before.equipment
    assert env._state.ledger.time_s == before.ledger.time_s
    assert env._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J


def test_numerical_failure_retains_previous_commits_and_solver_diagnostic(monkeypatch):
    from chemworld.physchem.coupled_system import CoupledSystem

    env = make_env(bulk=True, surface=True)
    advance(env, 10, target=380)
    before = env._state

    def failed_solve(*args, **kwargs):
        raise RuntimeError("controlled nonconvergence")

    monkeypatch.setattr(CoupledSystem, "advance", failed_solve)
    info = env.step({"operation": "wait", "duration_s": 10, "stirring_speed_rpm": 600})[4]
    assert info["operation_outcome"] == "numerical_failure"
    assert info["actual_duration_s"] == 0
    assert info["outcome_diagnostics"]["solver_error"] == "controlled nonconvergence"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.state_slots == before.state_slots
    assert env._state.temperature_K == before.temperature_K
    assert env._state.ledger.time_s == before.ledger.time_s
    assert env._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J


def test_thermal_capability_is_physical_and_passive_body_can_still_evolve():
    from chemworld.world.composition import WorldCompositionError, compile_world_composition

    spec = make_env().compiled_composition.spec
    physics = spec.physics
    initial = physics.initial_states["reactor"]
    passive = replace(
        physics, initial_states={"reactor": initial.replace(equipment=EquipmentLedger({}))}
    )
    with pytest.raises(WorldCompositionError, match="heat"):
        compile_world_composition(replace(spec, physics=passive))
    task = replace(spec.task, operations=None)
    compiled = compile_world_composition(replace(spec, physics=passive, task=task))
    assert set(compiled.task_spec.allowed_operations) == {"wait", "terminate"}
    env = ChemWorldEnv(composition=compiled)
    env.reset(seed=7)
    advance(env, 10)
    assert env._state.temperature_K < initial.temperature_K
    assert env._state.ledger.energy_jacket_J == 0
    assert env._state.ledger.heat_loss_J > 0
    unmodelled = compile_world_composition(
        replace(spec, physics=replace(physics, thermal_models={}), task=task)
    )
    assert set(unmodelled.task_spec.allowed_operations) == {"terminate"}
    with pytest.raises(WorldCompositionError, match="configure_control"):
        make_env().compiled_composition.bind_task(
            {
                "objective": "control",
                "operations": ["configure_control", "terminate"],
            }
        )


def test_native_heat_pause_wait_has_exact_logged_replay(tmp_path):
    from chemworld.agent_interface import agent_view_bundle
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.eval.verify import verify_records

    compiled = make_env(bulk=True, surface=True).compiled_composition
    task = compiled.spec.task.to_dict()
    task["operations"] += ["pause_control", "resume_control"]
    env = ChemWorldEnv(composition=compiled.bind_task(task))
    env.reset(seed=7)
    task_info = {**env.task_info(), **env.evaluator_provenance()}
    path = tmp_path / "thermal.jsonl"
    with TrajectoryLogger(path) as logger:
        for step, action in enumerate(
            (
                {
                    "operation": "heat",
                    "target_temperature_K": 380,
                    "duration_s": 10,
                    "stirring_speed_rpm": 600,
                },
                {"operation": "pause_control"},
                {"operation": "wait", "duration_s": 10, "stirring_speed_rpm": 600},
                {"operation": "resume_control"},
                {"operation": "wait", "duration_s": 10, "stirring_speed_rpm": 600},
                {"operation": "terminate"},
            ),
            start=1,
        ):
            observation, reward, terminated, truncated, info = env.step(action)
            assert info["transaction_status"] == "committed"
            logger.log(
                task_info=task_info,
                step=step,
                action=action,
                observation=observation_to_json(observation),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"agent_id": "thermal-test"},
                agent_view=agent_view_bundle(env, observation, info),
            )
    report = verify_records(load_jsonl(path))
    assert report.verified, report


def test_hidden_snapshot_retains_control_coverage_and_continued_response():
    env = make_env(bulk=True, surface=True)
    advance(env, 10, target=380)
    restored = WorldState.from_dict(env._state.to_dict())
    coupled = env.runtime.domain_services.reaction_thermal.coupled
    action = {"operation": "wait", "duration_s": 10, "stirring_speed_rpm": 600}
    first = coupled.advance(env._state, action, heat=False)
    second = coupled.advance(restored, action, heat=False)
    assert first.success and second.success
    assert first.state.to_dict() == second.state.to_dict()
