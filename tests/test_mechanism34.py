"""Expanded gas mechanisms use the same identity, derivatives, heat and codec."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from chemworld.physchem.coupled_system import CoupledState, HeatFlux
from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics, gas_batch_system
from chemworld.physchem.mechanism_import import (
    convert_mechanism,
    import_mechanism,
    mechanism_import_capabilities,
)
from chemworld.physchem.mechanism_rates import mechanism_forward_rate_constant
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_rate_contracts import audit_reaction_rate_contract
from chemworld.physchem.reaction_rate_laws import R_J_PER_MOL_K
from chemworld.physchem.thermochemistry import (
    species_thermo_from_cantera_yaml,
    species_thermo_from_dict,
)

spec = importlib.util.spec_from_file_location(
    "mechanism34_cases", Path(__file__).parent / "reference/mechanism34_cases.py"
)
cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases)


def test_global_caloric_and_activation_units_share_source_energy_quantity():
    document = cases.payload(
        {"equation": "A => B", "rate-constant": {"A": 2, "Ea": 3}}, thermo="constant-cp"
    )
    document["units"] = {"energy": "cal", "quantity": "mol", "length": "cm"}
    for species in document["species"]:
        species["thermo"].update(h0=-100, s0=20, cp0=30)
    network = convert_mechanism(document)
    assert network.reactions[0].rate_law.parameters["Ea_J_per_mol"] == pytest.approx(12.552)
    state = network.imported_gas_thermochemistry["A"].evaluate(298.15)
    assert state.enthalpy_J_mol == pytest.approx(-418.4)
    assert state.entropy_J_mol_K == pytest.approx(83.68)
    assert state.cp_J_mol_K == pytest.approx(125.52)


def test_coverage_is_machine_readable_and_names_unimplemented_source_families():
    capabilities = mechanism_import_capabilities()
    assert set(capabilities["thermo_models"]) == set(cases.THERMOS)
    assert "linear-Burke" in capabilities["unsupported"]
    assert "transport" in capabilities["metadata_only"]
    assert json.loads(json.dumps(capabilities)) == capabilities


@pytest.mark.parametrize("model", cases.THERMOS)
def test_thermo_and_full_network_json_replay(model):
    network = convert_mechanism(cases.payload(thermo=model))
    restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
    assert restored.to_dict() == network.to_dict()
    assert restored.reaction_rates(
        {"A": 0.2, "B": 0.3, "C": 0.5}, volume_L=1, temperature_K=700
    ) == {"reaction-001": 0.4}
    thermo = network.imported_gas_thermochemistry["A"]
    roundtrip = species_thermo_from_dict(json.loads(json.dumps(thermo.to_dict())))
    assert roundtrip.to_dict() == thermo.to_dict()
    assert roundtrip.evaluate(700) == thermo.evaluate(700)
    assert thermo.min_temperature_K == 200
    assert thermo.max_temperature_K == 3000
    adapter = IdealGasThermodynamics(network.imported_gas_thermochemistry)
    assert (
        IdealGasThermodynamics.from_dict(json.loads(json.dumps(adapter.to_dict()))).to_dict()
        == adapter.to_dict()
    )
    assert gas_batch_system(network).thermodynamics.to_dict() == adapter.to_dict()


@pytest.mark.parametrize("name,node", cases.rate_cases().items())
def test_all_rate_declarations_have_contract_and_native_derivatives(name, node):
    network = convert_mechanism(cases.payload(node))
    assert audit_reaction_rate_contract(network.reactions[0]).passed
    concentrations = {"A": 0.2, "B": 0.3, "C": 0.5}
    rates = network.reaction_rates(concentrations, volume_L=1, temperature_K=700)
    derivatives = network.amount_derivatives(concentrations, volume_L=1, temperature_K=700)
    assert derivatives["A"] == -rates["reaction-001"]
    assert derivatives["B"] == rates["reaction-001"]
    assert derivatives["C"] == 0
    assert sum(derivatives.values()) == 0


@pytest.mark.parametrize("order", [-1, 0, 0.5, 2])
def test_custom_orders_are_explicit_and_detached_from_stoichiometry(order):
    node = {
        "equation": "A => B",
        "rate-constant": {"A": 2},
        "orders": {"A": order},
        "negative-orders": True,
    }
    network = convert_mechanism(cases.payload(node))
    concentration = 0 if order == 0 else 0.25
    assert network.reaction_rates({"A": concentration}, volume_L=1, temperature_K=700)[
        "reaction-001"
    ] == pytest.approx(2 * concentration**order)
    if order < 0:
        with pytest.raises(ValueError, match="singular"):
            network.reaction_rates({}, volume_L=1, temperature_K=700)


def test_nonreactant_orders_require_permission_and_known_species():
    node = {
        "equation": "A => B",
        "rate-constant": {"A": 2},
        "orders": {"C": 0.5},
        "nonreactant-orders": True,
    }
    network = convert_mechanism(cases.payload(node))
    assert network.reaction_rates({"A": 0.2, "C": 0.25}, volume_L=1, temperature_K=700)[
        "reaction-001"
    ] == pytest.approx(0.2)
    node["nonreactant-orders"] = False
    with pytest.raises(ValueError, match="permission"):
        convert_mechanism(cases.payload(node))


def test_signed_duplicates_and_explicit_reverse_are_retained():
    document = cases.payload()
    document["reactions"] = [
        {"equation": "A => B", "rate-constant": {"A": value}, "duplicate": True, "negative-A": True}
        for value in (3, -1)
    ]
    document["reactions"].append({"equation": "B => A", "rate-constant": {"A": 0.5}})
    network = convert_mechanism(document)
    assert len(network.reactions) == 3
    assert network.reaction_rates({"A": 2, "B": 1}, volume_L=1, temperature_K=700) == {
        "reaction-001": 6,
        "reaction-002": -2,
        "reaction-003": 0.5,
    }
    assert network.amount_derivatives({"A": 2, "B": 1}, volume_L=1, temperature_K=700)["B"] == 3.5


@pytest.mark.parametrize("kind", ["three-body", "falloff", "chemically-activated"])
def test_named_collider_depends_only_on_declared_species(kind):
    if kind == "three-body":
        node = {"equation": "A + C => B + C", "type": kind, "rate-constant": [2, 0, 0]}
    else:
        node = {
            "equation": "A (+C) => B (+C)",
            "type": kind,
            "low-P-rate-constant": [4, 0, 0],
            "high-P-rate-constant": [2, 0, 0],
        }
    reaction = convert_mechanism(cases.payload(node)).reactions[0]
    k = mechanism_forward_rate_constant(reaction, {"A": 0.2, "B": 0.1, "C": 0.5}, 700)
    assert mechanism_forward_rate_constant(reaction, {"A": 0.2, "B": 10, "C": 0.5}, 700) == k
    assert k == pytest.approx(2 if kind == "chemically-activated" else 1)


@pytest.mark.parametrize(
    "source_units,a,expected",
    [
        ({"length": "cm", "quantity": "mol"}, 2, 0.002),
        ({}, "2 cm^3/mol/s", 0.002),
        ({}, "2 m^3/kmol/s", 2),
        ({}, "120 L/mol/min", 2),
    ],
)
def test_second_order_explicit_and_context_units(source_units, a, expected):
    document = cases.payload({"equation": "A => B", "rate-constant": [a, 0, 0], "orders": {"A": 2}})
    document["units"].update(source_units)
    network = convert_mechanism(document)
    assert network.reaction_rates({"A": 0.5}, volume_L=1, temperature_K=700)[
        "reaction-001"
    ] == pytest.approx(expected * 0.25)


def test_plog_signed_groups_and_endpoint_clamping():
    node = {
        "equation": "A => B",
        "type": "pressure-dependent-Arrhenius",
        "rate-constants": [
            {"P": p, "A": a, "b": 0, "Ea": 0} for p, a in ((1e4, 3), (1e4, -1), (1e6, 8))
        ],
    }
    reaction = convert_mechanism(cases.payload(node)).reactions[0]
    for pressure, expected in ((1, 2), (1e5, 4), (1e9, 8)):
        k = mechanism_forward_rate_constant(
            reaction, {"A": pressure / (1000 * R_J_PER_MOL_K * 700)}, 700
        )
        assert k == pytest.approx(expected)
    reaction.rate_law.parameters["rate_constants"][0]["A"] = 0.5
    with pytest.raises(ValueError, match="grouped"):
        mechanism_forward_rate_constant(reaction, {"A": 0.1}, 700)


@pytest.mark.parametrize(
    "model,expected", [("NASA7", 3), ("NASA9", 4), ("Shomate", 3 / R_J_PER_MOL_K)]
)
def test_source_boundary_ownership(model, expected):
    data = copy.deepcopy(cases.THERMOS[model])
    zero = [0] * (9 if model == "NASA9" else 7)
    data["data"] = [zero.copy(), zero.copy()]
    index = 2 if model == "NASA9" else 0
    data["data"][0][index], data["data"][1][index] = 3, 4
    thermo = species_thermo_from_cantera_yaml(species_id="A", thermo=data)
    assert thermo.evaluate(1000).cp_over_R == pytest.approx(expected)
    with pytest.raises(ValueError):
        thermo.evaluate(100)


def test_piecewise_gibbs_dimensionless_data_and_caloric_derivatives():
    node = copy.deepcopy(cases.THERMOS["piecewise-Gibbs"])
    node["dimensionless"] = True
    node["data"] = {
        t: float(g.split()[0]) * 1000 / (R_J_PER_MOL_K * t) for t, g in node["data"].items()
    }
    thermo = species_thermo_from_cantera_yaml(species_id="A", thermo=node)
    for t in (250, 500, 1500):
        state = thermo.evaluate(t)
        left, right = thermo.evaluate(t - 0.001), thermo.evaluate(t + 0.001)
        assert (right.enthalpy_J_mol - left.enthalpy_J_mol) / 0.002 == pytest.approx(
            state.cp_J_mol_K
        )
        assert (right.entropy_J_mol_K - left.entropy_J_mol_K) / 0.002 == pytest.approx(
            state.cp_J_mol_K / t
        )


def test_yaml_no_species_and_safe_loader(tmp_path):
    file = tmp_path / "mechanism.yaml"
    import yaml

    document = cases.payload()
    document["species"][2]["name"] = "NO"
    document["phases"][0]["species"][2] = "NO"
    # Force the unquoted YAML 1.2 spelling that PyYAML 1.1 otherwise treats as false.
    file.write_text(yaml.safe_dump(document).replace("'NO'", "NO"), encoding="utf-8")
    assert "NO" in import_mechanism(file).species_ids
    file.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        import_mechanism(file)


def test_complete_local_mechanism_can_exceed_previous_arbitrary_caps():
    document = cases.payload()
    for i in range(64):
        document["species"].append({**copy.deepcopy(document["species"][0]), "name": f"X{i}"})
        document["phases"][0]["species"].append(f"X{i}")
    document["reactions"] *= 300
    network = convert_mechanism(document)
    assert len(network.species) == 67
    assert len(network.reactions) == 300
    assert all(
        math.isfinite(v)
        for v in network.reaction_rates({"A": 0.1}, volume_L=1, temperature_K=700).values()
    )


@pytest.mark.parametrize("model", cases.THERMOS)
def test_native_thermal_advance_and_continuation(model):
    network = convert_mechanism(cases.payload(thermo=model))
    system = gas_batch_system(network, heat_flux=lambda _: HeatFlux(heater_W=5))
    initial = CoupledState({"A": 0.2, "B": 0.3, "C": 0.5}, 700, 1)
    result = system.advance(initial, 0.1)
    assert result.final_state.time_s == pytest.approx(0.1)
    assert result.final_state.amounts_mol["A"] == pytest.approx(0.2 * math.exp(-0.2), rel=2e-7)
    assert result.final_state.temperature_K > initial.temperature_K
    assert result.final_state.boundary_heat_J == pytest.approx(0.5)
    assert abs(result.energy_balance_residual_J) < 1e-5
    restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
    continued = gas_batch_system(restored, heat_flux=lambda _: HeatFlux(heater_W=5)).advance(
        result.final_state, 0.1
    )
    assert continued.final_state.time_s == pytest.approx(0.2)
    assert continued.final_state.amounts_mol["A"] == pytest.approx(0.2 * math.exp(-0.4), rel=5e-7)
