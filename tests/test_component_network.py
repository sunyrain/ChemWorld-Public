from __future__ import annotations

from copy import deepcopy

import pytest

from chemworld.agent_interface import action_schema, agent_view_bundle
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.eval.verify import verify_records
from chemworld.runtime.component_network import public_network
from chemworld.runtime.solvent_transport import total_solvents
from chemworld.world.composition import WorldCompositionError, compile_world_composition


def request(kind="dual"):
    vessels = ("a", "b", "c") if kind == "series" else ("a", "b")
    components = [
        {"id": v + "-" + k, "kind": k, "vessel": v}
        for v in vessels
        for k in ("reaction", "thermal", "observation")
    ]
    source, port = "a-reaction", "out"
    if kind == "distillation-crystal":
        components += [
            {"id": "column", "kind": "distillation", "vessel": "a"},
            {"id": "crystals", "kind": "crystallization", "vessel": "b"},
        ]
        source, port = "column", "distillate"
    connections = [
        {
            "id": "a-to-b",
            "source": {"component": source, "port": port},
            "target": {"component": "b-reaction", "port": "in"},
            "unit": "mol",
        }
    ]
    if kind == "series":
        connections.append(
            {
                "id": "b-to-c",
                "source": {"component": "b-reaction", "port": "out"},
                "target": {"component": "c-reaction", "port": "in"},
                "unit": "mol",
            }
        )
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": kind,
        "components": components,
        "connections": connections,
        "task": {"budget": 80},
    }


def actions(kind="dual"):
    result = [
        {"operation": "add_solvent", "solvent": 0, "volume_L": 0.03},
        {"operation": "add_reagent", "amount_mol": 0.03},
        {"operation": "add_catalyst", "catalyst": 1, "catalyst_amount_mol": 0.0003},
        {
            "operation": "heat",
            "target_temperature_K": 365,
            "duration_s": 600,
            "stirring_speed_rpm": 600,
        },
    ]
    if kind == "distillation-crystal":
        result.append(
            {
                "operation": "distill",
                "target_temperature_K": 380,
                "duration_s": 300,
                "reflux_ratio": 1.5,
            }
        )
    result += [
        {
            "operation": "route_material",
            "connection": "a-to-b",
            "transfer_fraction": 0.7,
            "mixing": "empty_only",
        },
        {"operation": "select_vessel", "vessel": "b"},
    ]
    if kind == "distillation-crystal":
        result += [
            {"operation": "seed_crystals", "seed_mass_g": 0.002},
            {"operation": "cool_crystallize", "target_temperature_K": 275, "duration_s": 600},
            {"operation": "filter_crystals"},
        ]
    else:
        result.append(
            {
                "operation": "heat",
                "target_temperature_K": 345,
                "duration_s": 120,
                "stirring_speed_rpm": 600,
            }
        )
    if kind == "series":
        result += [
            {
                "operation": "route_material",
                "connection": "b-to-c",
                "transfer_fraction": 0.5,
                "mixing": 0,
            },
            {"operation": "select_vessel", "vessel": "c"},
            {"operation": "wait", "duration_s": 30, "stirring_speed_rpm": 600},
        ]
    result += [{"operation": "terminate"}, {"operation": "measure", "instrument": "final_assay"}]
    return result


def step(env, action):
    result = env.step(action)
    info = result[-1]
    assert info["transaction_status"] == "committed", (
        action,
        info.get("rollback_reason"),
        {k: v for k, v in info.get("preconditions", {}).items() if not v},
    )
    assert env.constitution.check_state(env._state).passed
    state = env._state
    account = state.solvent_accounting
    assert tuple(
        a + b - c
        for a, b, c in zip(
            account.initial.volumes_L,
            account.added.volumes_L,
            account.removed.volumes_L,
            strict=True,
        )
    ) == pytest.approx(total_solvents(state).volumes_L, abs=1e-10), action
    return result


@pytest.mark.parametrize("kind", ["dual", "distillation-crystal", "series"])
@pytest.mark.parametrize("seed", [0, 1])
def test_complete_network_path_and_replay(kind, seed, tmp_path):
    env = ChemWorldEnv(composition=request(kind), seed=seed)
    env.reset(seed=seed)
    path = tmp_path / "trajectory.jsonl"
    task = {**env.task_info(), **env.evaluator_provenance()}
    with TrajectoryLogger(path) as logger:
        for i, action in enumerate(actions(kind), 1):
            observation, reward, terminated, truncated, info = step(env, action)
            logger.log(
                task_info=task,
                step=i,
                action=action,
                observation=observation_to_json(observation),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"agent_id": "network-test"},
                agent_view=agent_view_bundle(env, observation, info),
            )
    assert terminated and not truncated
    assert sum(env._state.vessel_elapsed_s.values()) == pytest.approx(env._state.ledger.time_s)
    for key in ("energy_jacket_J", "heat_reaction_J", "heat_loss_J"):
        energy = sum(
            getattr(s.thermal.vessels[s.vessel_id], key)
            for s in (env._state, *env._state.inactive_vessels.values())
        )
        assert energy == pytest.approx(getattr(env._state.ledger, key), abs=1e-8)
    assert verify_records(load_jsonl(path), tolerance=0).verified


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("unit", "unit_mismatch"),
        ("port", "invalid_port"),
        ("dependency", "missing_dependency"),
        ("id", "conflicting_state_owner"),
        ("vessel", "unknown_vessel"),
    ],
)
def test_graph_rejects_invalid_authoring(mutation, code):
    value = request()
    if mutation == "unit":
        value["connections"][0]["unit"] = "kg"
    elif mutation == "port":
        value["connections"][0]["source"]["port"] = "in"
    elif mutation == "dependency":
        value["components"] = [c for c in value["components"] if c["id"] != "b-thermal"]
    elif mutation == "id":
        value["components"][-1]["id"] = value["components"][0]["id"]
    elif mutation == "vessel":
        value["vessels"] = [{"id": "a"}]
    with pytest.raises(WorldCompositionError) as caught:
        compile_world_composition(value)
    assert code in {d.code for d in caught.value.diagnostics}


def test_local_capabilities_and_bounds():
    value = request("distillation-crystal")
    for c in value["components"]:
        if c["kind"] == "thermal":
            c["parameters"] = {
                "temperature_range_K": [300, 340] if c["vessel"] == "a" else [350, 400]
            }
    env = ChemWorldEnv(composition=value)
    env.reset()
    for op in actions()[:2]:
        step(env, op)
    schema = action_schema(env, "heat")
    assert schema is not None
    before = env._state
    info = env.step(
        {
            "operation": "heat",
            "target_temperature_K": 380,
            "duration_s": 60,
            "stirring_speed_rpm": 600,
        }
    )[-1]
    assert info["transaction_status"] != "committed"
    assert env._state.species_amounts == before.species_amounts
    step(env, {"operation": "select_vessel", "vessel": "b"})
    for op in actions()[:2]:
        step(env, op)
    step(
        env,
        {
            "operation": "heat",
            "target_temperature_K": 380,
            "duration_s": 60,
            "stirring_speed_rpm": 600,
        },
    )
    info = env.step(
        {"operation": "distill", "target_temperature_K": 380, "duration_s": 10, "reflux_ratio": 1.5}
    )[-1]
    assert info["transaction_status"] != "committed"


def test_routing_conservation_rollback_and_explicit_mixing():
    env = ChemWorldEnv(composition=request())
    env.reset()
    for op in actions()[:4]:
        step(env, op)
    original = env._state
    route = {"operation": "route_material", "connection": 0, "transfer_fraction": 0.5, "mixing": 0}
    step(env, route)
    assert env.constitution.check_material_conservation(original, env._state).passed
    before = env._state
    info = env.step(route)[-1]
    assert info["transaction_status"] != "committed"
    after = env._state
    assert after.species_amounts == before.species_amounts
    assert after.inactive_vessels["b"].to_dict() == before.inactive_vessels["b"].to_dict()
    step(env, {**route, "mixing": 1})
    assert env.constitution.check_material_conservation(original, env._state).passed
    selected = deepcopy(env._state.inactive_vessels["b"].species_amounts)
    step(env, {"operation": "select_vessel", "vessel": 1})
    assert env._state.species_amounts == selected
    assert public_network(env._state)["active_vessel"] == "b"


def test_capacity_rejection_and_gym_transport():
    value = request()
    value["vessels"] = [{"id": "a", "capacity_L": 0.1}, {"id": "b", "capacity_L": 0.001}]
    env = ChemWorldEnv(composition=value)
    env.reset()
    for op in actions()[:2]:
        step(env, op)
    route = {
        "operation": "route_material",
        "connection": "a-to-b",
        "transfer_fraction": 0.5,
        "mixing": "allow",
    }
    codec = env.action_codec
    assert codec.decode_vector(codec.encode_vector(route)) == codec.canonicalize(route)
    before = env._state
    info = env.step(route)[-1]
    assert info["transaction_status"] != "committed"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.inactive_vessels["b"].volume_L == 0


def test_mix_temperature_and_independent_charge_provenance():
    from chemworld.foundation.solvents import HEAT_CAPACITY_RATIOS
    from chemworld.world.mixtures import working_solvents

    env = ChemWorldEnv(composition=request())
    env.reset()
    assert action_schema(env, "route_material")["fields"][0]["choices"] == []
    step(env, {"operation": "add_solvent", "volume_L": 0.02, "solvent": 0})
    step(env, {"operation": "add_reagent", "amount_mol": 0.01})
    step(
        env,
        {
            "operation": "heat",
            "target_temperature_K": 360,
            "duration_s": 60,
            "stirring_speed_rpm": 600,
        },
    )
    source = env._state
    step(env, {"operation": "select_vessel", "vessel": "b"})
    step(env, {"operation": "add_solvent", "volume_L": 0.01, "solvent": 1})
    step(env, {"operation": "add_reagent", "amount_mol": 0.02})
    destination = env._state
    step(env, {"operation": "select_vessel", "vessel": "a"})
    assert env._state.species_amounts == source.species_amounts
    step(
        env, {"operation": "route_material", "connection": 0, "transfer_fraction": 0.5, "mixing": 1}
    )
    mixed = env._state.inactive_vessels["b"]
    capacities = [
        working_solvents(s).linear_property(HEAT_CAPACITY_RATIOS) * s.volume_L * f
        for s, f in ((source, 0.5), (destination, 1))
    ]
    expected = (
        source.temperature_K * capacities[0] + destination.temperature_K * capacities[1]
    ) / sum(capacities)
    assert mixed.temperature_K == pytest.approx(expected, abs=1e-9)
    assert len(mixed.samples.active_reference_sources) == 2
    assert mixed.species.initial_amounts_mol == pytest.approx(
        {
            k: v * 0.5 + destination.species.initial_amounts_mol.get(k, 0)
            for k, v in source.species.initial_amounts_mol.items()
        }
    )


def test_new_capability_union_without_registered_pattern():
    value = request()
    value["components"] += [{"id": "cell", "kind": "electrochemistry", "vessel": "a"}]
    compiled = compile_world_composition(value)
    assert {"electrolyze", "heat", "route_material"} <= set(compiled.task_spec.allowed_operations)


def test_distillation_outlet_keeps_species_equipment_and_other_vessels_consistent():
    from chemworld.foundation import equipment_settings

    value = request("distillation-crystal")
    value["components"] += [
        {"id": "c-" + k, "kind": k, "vessel": "c"} for k in ("reaction", "thermal", "observation")
    ]
    env = ChemWorldEnv(composition=value)
    env.reset()
    for action in actions("distillation-crystal")[:5]:
        step(env, action)
    untouched = env._state.inactive_vessels["c"].to_dict()
    before = env._state.phases.phases["distillate"]
    step(env, actions("distillation-crystal")[5])
    dest = env._state.inactive_vessels["b"]
    assert dest.species_amounts == pytest.approx(
        {k: v * 0.7 for k, v in before.species_amounts_mol.items()}
    )
    catalyst = sum(
        dest.species_amounts.get(k, 0)
        for k, roles in dest.species.species_roles.items()
        if "catalyst" in roles
    )
    assert equipment_settings(dest.equipment, "batch_reactor")[
        "catalyst_amount_mol"
    ] == pytest.approx(catalyst)
    assert env._state.inactive_vessels["c"].to_dict() == untouched


def test_campaign_closes_all_vessels_and_advances_lineage():
    env = ChemWorldEnv(composition=request(), episode_mode_override="campaign")
    env.reset()
    for action in actions():
        step(env, action)
    assert env._state.samples.batch_generation == 1
    assert env._state.volume_L == 0
    assert env._state.inactive_vessels["b"].volume_L == 0
    assert env._state.inactive_vessels["b"].samples.active_lineage == ("batch-0001-b",)
    assert total_solvents(env._state).volume_L == 0


def test_single_authored_vessel_capacity_is_effective():
    value = request()
    value["components"] = [c for c in value["components"] if c["vessel"] == "a"]
    value["connections"] = []
    value["vessels"] = [{"id": "a", "capacity_L": 0.01}]
    env = ChemWorldEnv(composition=value)
    env.reset()
    assert env._state.vessel_id == "a"
    info = env.step({"operation": "add_solvent", "volume_L": 0.02, "solvent": 0})[-1]
    assert info["transaction_status"] != "committed"


def test_inactive_check_order_does_not_depend_on_species_insertion_order():
    env = ChemWorldEnv(composition=request())
    env.reset()
    for action in actions()[:5]:
        step(env, action)
    state = env._state
    local = state.inactive_vessels["b"]
    reordered = local.replace(species_amounts=dict(reversed(tuple(local.species_amounts.items()))))
    changed = state.replace(inactive_vessels={"b": reordered})
    first = [(c.name, c.passed) for c in env.constitution.check_state(state).checks]
    second = [(c.name, c.passed) for c in env.constitution.check_state(changed).checks]
    assert first == second
