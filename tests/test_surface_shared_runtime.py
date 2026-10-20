"""Public finite-feed and material-transfer paths into shared surface dynamics."""

from dataclasses import replace

import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.world.surface_world import surface_world_spec


def make_env(**kwargs):
    env = ChemWorldEnv(composition=surface_world_spec(**kwargs))
    env.reset(seed=12)
    return env


def step(env, **action):
    result = env.step(action)
    assert result[4]["transaction_status"] == "committed", result[4]
    return result


def dose(env):
    step(env, operation="add_component", component="CO-feed", amount_mol=8e-5)
    step(env, operation="add_component", component="O2-feed", amount_mol=4e-5)


def wait(env, duration=10):
    return step(env, operation="wait", duration_s=duration, stirring_speed_rpm=600)


@pytest.mark.parametrize(
    "mobile,bulk", [(False, False), (True, False), (False, True), (True, True)]
)
def test_finite_feed_heat_route_and_continued_reaction(mobile, bulk):
    env = make_env(mobile=mobile, include_bulk=bulk)
    before = env._state
    dose(env)
    assert env._state.inactive_vessels["co_supply"].species_amounts["CO"] == pytest.approx(2e-5)
    assert env._state.volume_L == before.volume_L
    assert env._state.temperature_K > 350  # Flow enthalpy includes pV, not just internal energy.
    step(env, operation="heat", target_temperature_K=380, duration_s=10, stirring_speed_rpm=600)
    wait(env)
    reacted = env._state
    assert reacted.species_amounts["CO2"] > 0
    slots = reacted.state_slots
    step(
        env,
        operation="route_material",
        connection="reactor-to-receiver",
        transfer_fraction=0.5,
        mixing="allow",
    )
    assert env._state.volume_L == before.volume_L
    assert env._state.inactive_vessels["receiver"].volume_L == 0.01
    if mobile:
        assert env._state.state_slots.species_amounts() == pytest.approx(
            {key: value / 2 for key, value in slots.species_amounts().items()}
        )
        assert env._state.inactive_vessels[
            "receiver"
        ].state_slots.species_amounts() == pytest.approx(env._state.state_slots.species_amounts())
    else:
        assert env._state.state_slots == slots
    receipt = env._state.metadata["last_material_packet"]
    assert abs(receipt["destination_energy_residual_J"]) < 1e-7
    assert abs(receipt["source_energy_residual_J"]) < 1e-7
    step(env, operation="select_vessel", vessel="receiver")
    before_product = env._state.species_amounts["CO2"]
    wait(env)
    assert env._state.species_amounts["CO2"] > before_product
    assert env._state.ledger.time_s == pytest.approx(30)
    assert all(check["passed"] for check in env._last_operation_record.constitution_checks)


def test_finite_feed_uses_closed_element_boundary_and_actual_source_debit():
    env = make_env()
    before = env._state
    info = step(env, operation="add_component", component="CO-feed", amount_mol=5e-5)[4]
    record = env._last_operation_record
    assert not record.material_balance.exchanges
    assert dict(record.material_balance.stock_consumption)["reagent_mol"] == pytest.approx(5e-5)
    assert env._state.inactive_vessels["co_supply"].species_amounts["CO"] == pytest.approx(5e-5)
    assert env._state.species_amounts["CO"] - before.species_amounts["CO"] == pytest.approx(5e-5)
    receipt = info["outcome_diagnostics"]["material_packet"]
    assert receipt["gas_enthalpy_J"] == pytest.approx(5e-5 * (-110000 + 8.31446261815324 * 350))
    assert receipt["source_energy_residual_J"] is None
    assert abs(receipt["destination_energy_residual_J"]) < 1e-7
    step(env, operation="add_component", component="CO-feed", amount_mol=5e-5)
    assert env._state.inactive_vessels["co_supply"].species_amounts["CO"] == 0
    exhausted = env._state
    info = env.step({"operation": "add_component", "component": "CO-feed", "amount_mol": 1e-6})[4]
    assert info["operation_outcome"] == "rejected"
    assert env._state.species_amounts == exhausted.species_amounts
    assert env._state.inactive_vessels == exhausted.inactive_vessels


@pytest.mark.parametrize(
    "component,amount",
    [("CO-feed", 0.00011), ("CO-feed", -1e-5), ("missing", 1e-5), ("CO-feed", float("nan"))],
)
def test_invalid_feed_preserves_real_inventory_and_thermal_state(component, amount):
    env = make_env()
    before = env._state
    info = env.step({"operation": "add_component", "component": component, "amount_mol": amount})[4]
    assert info["operation_outcome"] == "rejected"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.state_slots == before.state_slots
    assert env._state.inactive_vessels == before.inactive_vessels
    assert env._state.temperature_K == before.temperature_K
    assert env._state.ledger.time_s == before.ledger.time_s


def test_pressure_capacity_rejects_feed_and_stops_actual_thermal_advance():
    from chemworld.physchem.reaction_network import R_J_PER_MOL_K

    spec = surface_world_spec(include_surface=False)
    original = spec.physics.initial_states["reactor"]
    from chemworld.foundation.state_ledgers import PhaseLedger

    amounts = {
        **dict.fromkeys(original.species_amounts, 0.0),
        "N2": 495000 * 0.00001 / (R_J_PER_MOL_K * 350),
    }
    initial = original.replace(
        species_amounts=amounts,
        pressure_Pa=495000,
        phases=PhaseLedger(
            {
                key: replace(phase, species_amounts_mol=amounts)
                for key, phase in original.phases.phases.items()
            }
        ),
    )
    physics = replace(
        spec.physics, initial_states={**spec.physics.initial_states, "reactor": initial}
    )
    env = ChemWorldEnv(composition=replace(spec, physics=physics))
    env.reset(seed=12)
    info = env.step({"operation": "add_component", "component": "CO-feed", "amount_mol": 1e-4})[4]
    assert info["operation_outcome"] == "rejected"
    assert env._state.pressure_Pa == 495000
    info = step(
        env, operation="heat", target_temperature_K=380, duration_s=60, stirring_speed_rpm=600
    )[4]
    assert info["operation_outcome"] == "stopped"
    assert info["outcome_diagnostics"]["stopping_event"] == "pressure_limit"
    assert 0 < info["actual_duration_s"] < 60
    assert env._state.pressure_Pa == pytest.approx(500000, abs=1e-6)


def test_native_feed_ids_and_legacy_vector_share_one_source_selection():
    from chemworld.agent_interface import action_schema

    env = make_env()
    schema = action_schema(env, "add_component")
    assert next(field for field in schema["fields"] if field["field"] == "component")[
        "choices"
    ] == ["CO-feed", "O2-feed"]
    action = {"operation": "add_component", "component": "O2-feed", "amount_mol": 1e-5}
    codec = env.action_codec
    decoded = codec.canonicalize(codec.decode_vector(codec.encode_vector(action)))
    assert decoded["component"] == "O2-feed"
    step(env, **decoded)
    assert env._state.inactive_vessels["oxygen_supply"].species_amounts["O2"] == pytest.approx(9e-5)
    assert env._state.inactive_vessels["co_supply"].species_amounts["CO"] == 1e-4


def test_fixed_carrier_survives_complete_gas_replacement():
    env = make_env()
    dose(env)
    wait(env)
    slots = env._state.state_slots
    equipment = env._state.equipment
    step(
        env,
        operation="route_material",
        connection="reactor-to-receiver",
        transfer_fraction=1.0,
        mixing="allow",
    )
    assert not any(env._state.species_amounts.values())
    assert env._state.state_slots == slots
    assert env._state.equipment == equipment
    step(env, operation="add_component", component="O2-feed", amount_mol=2e-5)
    wait(env)
    assert env._state.species_amounts["CO2"] > 0  # Retained adsorbed carbon is used.


def test_packet_heat_transfer_and_surface_history_have_exact_replay(tmp_path):
    from chemworld.agent_interface import agent_view_bundle
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.eval.verify import verify_records

    env = make_env(mobile=True, include_bulk=True)
    task_info = {**env.task_info(), **env.evaluator_provenance()}
    path = tmp_path / "surface.jsonl"
    actions = [
        {"operation": "add_component", "component": "CO-feed", "amount_mol": 8e-5},
        {"operation": "add_component", "component": "O2-feed", "amount_mol": 4e-5},
        {
            "operation": "heat",
            "target_temperature_K": 380,
            "duration_s": 10,
            "stirring_speed_rpm": 600,
        },
        {
            "operation": "route_material",
            "connection": "reactor-to-receiver",
            "transfer_fraction": 0.5,
            "mixing": "allow",
        },
        {"operation": "select_vessel", "vessel": "receiver"},
        {"operation": "wait", "duration_s": 10, "stirring_speed_rpm": 600},
        {"operation": "terminate"},
    ]
    with TrajectoryLogger(path) as logger:
        for index, action in enumerate(actions, start=1):
            observation, reward, terminated, truncated, info = step(env, **action)
            logger.log(
                task_info=task_info,
                step=index,
                action=action,
                observation=observation_to_json(observation),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={},
                agent_view=agent_view_bundle(env, observation, info),
            )
    report = verify_records(load_jsonl(path))
    assert report.verified, report
