"""Finite gas kinetics, import boundaries and actual shared heat/wait paths."""

import copy
import json
import math
from pathlib import Path

import pytest
import yaml

from chemworld.physchem.mechanism_import import convert_mechanism, import_mechanism
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_rate_contracts import audit_reaction_rate_contract
from chemworld.physchem.reaction_rate_laws import arrhenius_k, pressure_rate_constant

FIXTURES = Path(__file__).parent / "fixtures/mechanisms"


@pytest.mark.parametrize("name", ["pressure_log", "chebyshev"])
def test_pressure_contract_grid_and_roundtrip(name):
    network = import_mechanism(FIXTURES / f"{name}.yaml")
    reaction = network.reactions[0]
    assert audit_reaction_rate_contract(reaction).passed
    for temperature in (600, 900, 1200, 1500, 1800):
        for pressure in (1e4, 1e5, 1e6):
            assert (
                pressure_rate_constant(
                    reaction.rate_law.parameters,
                    reaction.rate_law.equation_id,
                    temperature,
                    pressure,
                )
                > 0
            )
    restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
    assert restored.to_dict() == network.to_dict()
    assert restored.reaction_rates(
        {"A": 0.01}, volume_L=1, temperature_K=900
    ) == network.reaction_rates({"A": 0.01}, volume_L=1, temperature_K=900)


def test_plog_same_pressure_sum_then_interpolate():
    law = import_mechanism(FIXTURES / "pressure_log.yaml").reactions[0].rate_law
    params = law.parameters
    low = sum(arrhenius_k(term, 1000) for term in params["rate_constants"][:2])
    high = arrhenius_k(params["rate_constants"][2], 1000)
    assert pressure_rate_constant(params, law.equation_id, 1000, 1e4) == pytest.approx(low)
    assert pressure_rate_constant(
        params, law.equation_id, 1000, math.sqrt(1e4 * 1e5)
    ) == pytest.approx(math.sqrt(low * high))


@pytest.mark.parametrize("name", ["pressure_log", "chebyshev"])
@pytest.mark.parametrize(
    "temperature,pressure",
    [
        (599, 1e5),
        (1801, 1e5),
        (900, 9999),
        (900, 1000001),
        (float("nan"), 1e5),
        (900, float("inf")),
    ],
)
def test_outside_domain_rejected_even_with_empty_reactant(name, temperature, pressure):
    law = import_mechanism(FIXTURES / f"{name}.yaml").reactions[0].rate_law
    with pytest.raises(ValueError, match="validity"):
        pressure_rate_constant(law.parameters, law.equation_id, temperature, pressure)


@pytest.mark.parametrize("kind", ["elementary", "pressure-dependent-Arrhenius", "Chebyshev"])
@pytest.mark.parametrize(
    "units,factor",
    [
        ({"length": "m", "quantity": "kmol"}, 1),
        ({"length": "m", "quantity": "mol"}, 1e3),
        ({"length": "cm", "quantity": "mol"}, 1e-3),
        ({"volume": "L", "quantity": "mol"}, 1),
    ],
)
def test_second_order_unit_conversion(kind, units, factor):
    payload = yaml.safe_load((FIXTURES / "chebyshev.yaml").read_text())
    payload["species"][1]["composition"] = {"H": 4}
    if "volume" in units:
        payload["units"].pop("length")
    payload["units"].update(units)
    node = {"equation": "2 A => B", "type": kind}
    if kind == "elementary":
        node["rate-constant"] = {"A": 1, "b": 0, "Ea": "1 cal/mol"}
    elif kind == "pressure-dependent-Arrhenius":
        node.update(
            {
                "temperature-range": [600, 1800],
                "rate-constants": [{"P": p, "A": 1, "b": 0, "Ea": "1 cal/mol"} for p in (1e4, 1e6)],
            }
        )
    else:
        node.update({"temperature-range": [600, 1800], "pressure-range": [1e4, 1e6], "data": [[0]]})
    payload["reactions"] = [node]
    law = convert_mechanism(payload).reactions[0].rate_law
    if kind == "elementary":
        assert law.parameters["A"] == pytest.approx(factor)
        assert law.parameters["Ea_J_per_mol"] == 4.184
    elif kind == "Chebyshev":
        assert pressure_rate_constant(law.parameters, law.equation_id, 900, 1e5) == pytest.approx(
            factor
        )
    else:
        assert law.parameters["rate_constants"][0]["A"] == pytest.approx(factor)
        assert law.parameters["rate_constants"][0]["Ea_J_per_mol"] == 4.184


@pytest.mark.parametrize(
    "mutation",
    [
        "unbalanced",
        "unknown",
        "SRI",
        "orders",
        "negative_A",
        "external_species",
        "bad_thermo",
        "ragged",
        "unknown_field",
    ],
)
def test_import_rejects_whole_invalid_mechanism(mutation):
    payload = yaml.safe_load((FIXTURES / "chebyshev.yaml").read_text())
    node = payload["reactions"][0]
    if mutation == "unbalanced":
        node["equation"] = "A => 2 B"
    elif mutation == "unknown":
        node["equation"] = "A => C"
    elif mutation in {"SRI", "orders", "unknown_field"}:
        node[mutation] = {"A": 1}
    elif mutation == "negative_A":
        node.clear()
        node.update(equation="A => B", **{"rate-constant": {"A": -1, "b": 0, "Ea": 0}})
    elif mutation == "external_species":
        payload["phases"][0]["species"] = [{"external.yaml/species": "all"}]
    elif mutation == "bad_thermo":
        payload["species"][0]["thermo"]["model"] = "NASA9"
    elif mutation == "ragged":
        node["data"] = [[1, 2], [3]]
    with pytest.raises(ValueError):
        convert_mechanism(payload)


def test_json_and_executable_input(tmp_path):
    payload = yaml.safe_load((FIXTURES / "pressure_log.yaml").read_text())
    target = tmp_path / "mechanism.json"
    target.write_text(json.dumps(payload))
    assert len(import_mechanism(target).reactions) == 1
    with pytest.raises(ValueError, match="declarative"):
        import_mechanism(tmp_path / "mechanism.py")
    target = tmp_path / "unsafe.yaml"
    target.write_text("!!python/object/apply:builtins.eval ['1+1']")
    with pytest.raises(yaml.YAMLError):
        import_mechanism(target)


def test_h2o2_all_channels_preserved_and_reverse_without_forward_reactants():
    path = FIXTURES / "cantera-3.1.0/h2o2.yaml"
    with pytest.raises(ValueError, match="explicit phase"):
        import_mechanism(path)
    network = import_mechanism(path, phase_name="ohmech")
    assert len(network.species) == 10 and len(network.reactions) == 29
    assert all(audit_reaction_rate_contract(r).passed for r in network.reactions)
    assert network.reactions[6].forward_orders == {"H": 1, "O2": 2}
    assert network.reactions[6].reverse_orders == {"HO2": 1, "O2": 1}
    rates = network.reaction_rates({"O2": 0.001}, volume_L=1, temperature_K=1500)
    assert rates["reaction-001"] < 0


@pytest.mark.parametrize("name", ["pressure_log", "chebyshev"])
def test_only_arithmetic_endpoint_tolerance_not_extrapolation(name):
    law = import_mechanism(FIXTURES / f"{name}.yaml").reactions[0].rate_law
    expected = pressure_rate_constant(law.parameters, law.equation_id, 900, 1e4)
    assert (
        pressure_rate_constant(law.parameters, law.equation_id, 900, math.nextafter(1e4, 0))
        == expected
    )
    with pytest.raises(ValueError, match="validity"):
        pressure_rate_constant(law.parameters, law.equation_id, 900, 1e4 - 1e-6)
    with pytest.raises(ValueError, match="validity"):
        from chemworld.physchem.reaction_network import evaluate_rate_law

        evaluate_rate_law(
            import_mechanism(FIXTURES / f"{name}.yaml").reactions[0],
            concentrations_mol_L={"A": 0, "B": 0},
            temperature_K=900,
        )


def test_actual_log_replay_and_snapshot_continuation(tmp_path):
    from chemworld.agent_interface import agent_view_bundle
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.eval.verify import verify_records
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.gas_mechanism_world import gas_mechanism_world

    network = import_mechanism(FIXTURES / "chebyshev.yaml")
    env = ChemWorldEnv(composition=gas_mechanism_world(network, {"A": 0.0001}))
    env.reset(seed=7)
    task_info = {**env.task_info(), **env.evaluator_provenance()}
    path = tmp_path / "gas.jsonl"
    with TrajectoryLogger(path) as logger:
        for step, action in enumerate(
            (
                {"operation": "heat", "target_temperature_K": 750, "duration_s": 1},
                {"operation": "measure", "instrument": "gas_pressure"},
                {"operation": "wait", "duration_s": 1},
            ),
            1,
        ):
            observation, reward, terminated, truncated, info = env.step(action)
            assert info["transaction_status"] == "committed", info
            logger.log(
                task_info=task_info,
                step=step,
                action=action,
                observation=observation_to_json(observation),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"agent_id": "gas-test"},
                agent_view=agent_view_bundle(env, observation, info),
            )
    report = verify_records(load_jsonl(path))
    assert report.verified, report
    payload = json.loads(json.dumps(env.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(payload)
    action = {"operation": "wait", "duration_s": 1}
    env.step(action)
    restored.execute(action)
    assert env._state.to_dict() == restored._state.to_dict()
    env.close()
    restored.close()


def test_native_temperature_domain_does_not_allow_invalid_controller():
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.world.gas_mechanism_world import gas_mechanism_world

    network = import_mechanism(FIXTURES / "chebyshev.yaml")
    env = ChemWorldEnv(composition=gas_mechanism_world(network, {"A": 0.0001}))
    env.reset(seed=7)
    before = dict(env._state.species_amounts)
    info = env.step({"operation": "heat", "target_temperature_K": 1900, "duration_s": 1})[4]
    assert info["operation_outcome"] == "rejected"
    assert env._state.species_amounts == before
    assert env._state.ledger.time_s == 0
    assert env.compiled_composition.spec.task.instruments == ("gas_pressure",)
    env.close()


@pytest.mark.parametrize("name", ["pressure_log", "chebyshev", "cantera-3.1.0/h2o2"])
def test_real_heat_wait_transactions_and_exact_replay(name):
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.world.gas_mechanism_world import gas_mechanism_world

    hydrogen = "h2o2" in name
    network = import_mechanism(FIXTURES / f"{name}.yaml", phase_name="ohmech" if hydrogen else None)
    amounts = {"H2": 0.00002, "O2": 0.00001, "AR": 0.00007} if hydrogen else {"A": 0.0001}
    payload = gas_mechanism_world(network, amounts).to_dict()
    finals = []
    for _ in range(2):
        env = ChemWorldEnv(composition=copy.deepcopy(payload))
        env.reset(seed=7)
        initial = dict(env._state.species_amounts)
        for action in (
            {
                "operation": "heat",
                "target_temperature_K": 750,
                "duration_s": 1,
                "stirring_speed_rpm": 600,
            },
            {"operation": "wait", "duration_s": 1, "stirring_speed_rpm": 600},
        ):
            info = env.step(action)[4]
            assert info["transaction_status"] == "committed", info
            assert abs(info["outcome_diagnostics"]["energy_balance_residual_J"]) < 1e-5
        assert env._state.temperature_K > 700
        assert env._state.ledger.time_s == pytest.approx(2)
        assert env._state.ledger.energy_jacket_J <= 10 + 1e-8
        assert env._state.species_amounts != initial
        finals.append(
            (
                dict(env._state.species_amounts),
                env._state.temperature_K,
                env._state.ledger.to_dict(),
            )
        )
        env.close()
    assert finals[0] == finals[1]
