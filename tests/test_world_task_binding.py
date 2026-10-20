"""Physical composition is invariant under task rebinding."""

from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.agent_interface import agent_view_bundle
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.eval.verify import verify_records
from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import SpeciesSpec
from chemworld.runtime.mechanisms import compile_mechanism
from chemworld.world.composition import (
    WorldCompositionError,
    WorldCompositionSpec,
    compile_world_composition,
)
from chemworld.world.physical_world import CompositionPhysics
from chemworld.world.state_factory import initial_chemworld_state


def request(*, inert=False):
    if inert:
        network = ReactionNetworkSpec("inert-gas", (SpeciesSpec("N2", "N2", phase="gas"),), ())
        state = WorldState(
            species_amounts={"N2": 100000 * 0.00001 / (8.314462618 * 350)},
            volume_L=0.01,
            temperature_K=350,
            pressure_Pa=100000,
            phase="gas",
            vessel_id="reactor",
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
                "cost": "currency",
                "risk": "risk",
            },
            equipment=EquipmentLedger(
                {
                    "heater": EquipmentRecord(
                        "heater",
                        "thermal_device",
                        attached_vessel_id="reactor",
                        settings={"heat_capacity_J_K": 12.0},
                    )
                }
            ),
        )
        kinds = ["thermal"]
        instruments = []
        operations = ["terminate"]
    else:
        mechanism = compile_mechanism("simple_batch_reaction")
        network = replace(
            mechanism.network,
            species=tuple(
                replace(
                    species,
                    observable_aliases=tuple(
                        role
                        for role, names in mechanism.observable_mapping.items()
                        if species.species_id in names
                    ),
                )
                for species in mechanism.network.species
            ),
        )
        state = initial_chemworld_state(
            species_ids=network.species_ids, species_roles=mechanism.species_roles
        )
        state = state.replace(
            vessel_id="reactor",
            vessels=None,
            phases=None,
            thermal=None,
            equipment=EquipmentLedger(
                {
                    k: replace(v, attached_vessel_id="reactor")
                    for k, v in state.equipment.equipment.items()
                }
            ),
            volume_L=0.03,
            temperature_K=360,
            species_amounts={
                name: 0.001 * mechanism.initial_amount_policy.get(name, 0.0)
                for name in network.species_ids
            },
        )
        kinds = ["reaction", "thermal", "observation"]
        instruments = ["hplc", "final_assay"]
        operations = ["heat", "wait", "measure", "terminate"]
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "native-test",
        "components": [
            {
                "kind": kind,
                "parameters": {"instruments": instruments} if kind == "observation" else {},
            }
            for kind in kinds
        ],
        "physics": CompositionPhysics(network, {"reactor": state}, 7).to_dict(),
        "task": {
            "objective": "collect response curve",
            "budget": 8,
            "operations": operations,
            "instruments": ["hplc"] if instruments else [],
            "termination": "explicit-stop-or-budget",
        },
    }


def test_native_gas_is_not_a_liquid_reaction_preset():
    payload = request(inert=True)
    compiled = compile_world_composition(payload)
    assert compiled.runtime_task_profile_id is None
    assert compiled.task_spec.allowed_instruments == ()
    env = ChemWorldEnv(composition=compiled)
    env.reset(seed=19)
    assert env._state.phase == "gas"
    assert (
        env._state.species_amounts
        == payload["physics"]["initial_states"]["reactor"]["species_amounts"]
    )
    assert not env.scenario_instance.compiled_mechanism.network.reactions
    assert env._state.solvent_accounting.initial.volume_L == 0
    assert env._state.equipment.equipment["heater"].settings["heat_capacity_J_K"] == 12
    _, _, terminated, truncated, info = env.step({"operation": "terminate"})
    assert info["transaction_status"] == "committed"
    assert terminated and not truncated


def test_rebinding_changes_task_ending_not_physical_response():
    curve = request()
    assay = deepcopy(curve)
    assay["task"].update(
        objective="yield",
        instruments=["hplc", "final_assay"],
        termination="final-assay-or-budget",
        evaluation={"metrics": ["final_assay_score"]},
    )
    curve_env, assay_env = (ChemWorldEnv(composition=p) for p in (curve, assay))
    for env in (curve_env, assay_env):
        env.reset(seed=4)
        _, _, _, _, info = env.step(
            {"operation": "wait", "duration_s": 60.0, "stirring_speed_rpm": 600.0}
        )
        assert info["transaction_status"] == "committed"
    assert curve_env._state.to_dict() == assay_env._state.to_dict()
    assert curve_env.scoring_contract.score_family == "unscored"
    assert assay_env.scoring_contract.score_family == "reaction"
    _, reward, done, _, info = curve_env.step({"operation": "measure", "instrument": "hplc"})
    assert info["transaction_status"] == "committed" and not done
    assert reward == 0
    assert curve_env._state.process.last_observed_mask["yield"]
    assert not curve_env._state.process.last_observed_mask["score"]
    assert curve_env.step({"operation": "terminate"})[2]
    assert not assay_env.step({"operation": "terminate"})[2]
    result = assay_env.step({"operation": "measure", "instrument": "final_assay"})
    assert result[2] and result[4]["transaction_status"] == "committed"


def test_task_rebinding_preserves_authored_physics_and_permissions():
    compiled = compile_world_composition(request())
    rebound = compiled.bind_task(
        {
            "budget": 1,
            "objective": "stop immediately",
            "operations": ["terminate"],
            "instruments": [],
        }
    )
    assert rebound.spec.physics is compiled.spec.physics
    env = ChemWorldEnv(composition=rebound)
    env.reset(seed=4)
    assert env.allowed_operations == {"terminate"}
    result = env.step({"operation": "terminate"})
    assert result[2] and not result[3]


def test_native_vessels_keep_distinct_inventory_and_equipment():
    payload = request(inert=True)
    first = WorldState.from_dict(payload["physics"]["initial_states"]["reactor"])
    second = first.replace(
        vessel_id="receiver",
        phases=None,
        thermal=None,
        vessels=None,
        species_amounts={"N2": 0.0},
        pressure_Pa=0.0,
        equipment=EquipmentLedger({}),
    )
    payload["physics"]["initial_states"]["receiver"] = second.to_dict()
    payload["components"].append(
        {"id": "second-heater", "kind": "thermal", "vessel": "receiver", "parameters": {}}
    )
    payload["task"]["operations"] = ["select_vessel", "terminate"]
    env = ChemWorldEnv(composition=payload)
    env.reset(seed=1)
    states = {env._state.vessel_id: env._state, **env._state.inactive_vessels}
    assert states["reactor"].species_amounts["N2"] > 0
    assert states["receiver"].species_amounts["N2"] == 0
    assert not states["receiver"].equipment.equipment
    assert states["reactor"].equipment.equipment["heater"].attached_vessel_id == "reactor"


@pytest.mark.parametrize("mutation", ["instrument", "process", "state_owner", "units"])
def test_invalid_binding_fails_before_execution(mutation):
    payload = request(inert=True)
    if mutation == "instrument":
        payload["task"]["instruments"] = ["final_assay"]
    elif mutation == "process":
        payload["components"].append({"kind": "reaction", "parameters": {}})
    elif mutation == "state_owner":
        payload["physics"]["initial_states"]["reactor"]["vessel_id"] = "missing"
    else:
        payload["physics"]["network"]["units"]["amount"] = "kg"
    with pytest.raises((ValueError, WorldCompositionError)):
        compile_world_composition(payload)


def test_physical_payload_roundtrips_and_stays_evaluator_only():
    payload = request(inert=True)
    spec = WorldCompositionSpec.from_dict(payload)
    assert WorldCompositionSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    env = ChemWorldEnv(composition=spec)
    env.reset(seed=99)
    assert "composition_request" not in env.task_info()
    assert env.evaluator_provenance()["composition_request"] == spec.to_dict()
    assert env.world.world_id == ChemWorldEnv(composition=spec, seed=1).world.world_id


def test_native_observation_and_explicit_stop_replay(tmp_path):
    env = ChemWorldEnv(composition=request(), seed=4)
    env.reset(seed=4)
    task_info = {**env.task_info(), **env.evaluator_provenance()}
    path = tmp_path / "native.jsonl"
    with TrajectoryLogger(path) as logger:
        for step, action in enumerate(
            (
                {"operation": "measure", "instrument": "hplc"},
                {"operation": "terminate"},
            ),
            start=1,
        ):
            observation, reward, terminated, truncated, info = env.step(action)
            logger.log(
                task_info=task_info,
                step=step,
                action=action,
                observation=observation_to_json(observation),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"agent_id": "native-binding-test"},
                agent_view=agent_view_bundle(env, observation, info),
            )
    report = verify_records(load_jsonl(path))
    assert report.verified, report
