"""Finite energy states use original gas material, thermal and replay ownership."""

import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chemworld.physchem.energy_grain_binding import compile_energy_grain_network
from chemworld.physchem.energy_grain_master_equation import (
    boltzmann_grain_fractions,
    collision_transition_probabilities,
    energy_grain_generator,
    grain_population_report,
    well_collision_generator,
)
from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_rate_laws import evaluate_rate_law
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.world.energy_grain import energy_grain_world


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f"reference/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = load("energy_grain38_cases")
reference = load("energy_grain38_reference")


def setup(topology="branch", pressure=1e4, temperature=600):
    card = cases.declaration(topology=topology)
    model = EnergyGrainMasterEquationSpec.from_dict(card)
    network = compile_energy_grain_network(model)
    reactor = GasReactorNetworkSpec((GasReactorNodeSpec("cell", 1),), (), temperature, "cell")
    amounts = cases.initial_amounts(card, temperature=temperature, pressure=pressure)
    return card, model, network, reactor, amounts


@pytest.mark.parametrize("temperature", (300, 600, 1200))
def test_collision_probability_detailed_balance_and_boltzmann_nullspace(temperature):
    _, model, _, _, amounts = setup(temperature=temperature)
    well = model.wells[0]
    equilibrium = np.array(list(boltzmann_grain_fractions(well, temperature).values()))
    for bath in well.baths:
        probability = np.array(collision_transition_probabilities(well, bath, temperature))
        assert probability.min() >= 0
        assert probability.sum(axis=0) == pytest.approx(np.ones(3), abs=2e-15)
        flux = probability * equilibrium
        assert flux == pytest.approx(flux.T, abs=2e-15)
    generator = np.array(
        well_collision_generator(well, temperature_K=temperature, concentrations_mol_L=amounts)
    )
    assert generator @ equilibrium == pytest.approx(np.zeros(3), abs=2e-14)
    assert generator.sum(axis=0) == pytest.approx(np.zeros(3), abs=2e-14)


def test_bath_frequencies_are_additive_and_follow_actual_concentrations():
    _, model, _, _, amounts = setup()
    well = model.wells[0]
    together = np.array(
        well_collision_generator(well, temperature_K=600, concentrations_mol_L=amounts)
    )
    separate = sum(
        np.array(
            well_collision_generator(
                well, temperature_K=600, concentrations_mol_L={name: amounts[name]}
            )
        )
        for name in ("He", "Ar")
    )
    assert together == pytest.approx(separate, abs=1e-14)
    doubled = {name: 2 * value for name, value in amounts.items()}
    assert well_collision_generator(well, temperature_K=600, concentrations_mol_L=doubled) == (
        pytest.approx(2 * together)
    )


def test_thermo_excludes_internal_modes_then_adds_state_energy_and_entropy_once():
    _, model, network, _, _ = setup()
    thermo = network.imported_gas_thermochemistry
    for t in (300, 600, 1200):
        ground, excited = (thermo[key].evaluate(t) for key in ("A0", "A2"))
        assert excited.cp_J_mol_K == ground.cp_J_mol_K == 30
        assert excited.enthalpy_J_mol - ground.enthalpy_J_mol == pytest.approx(10000)
        assert excited.entropy_J_mol_K - ground.entropy_J_mol_K == pytest.approx(
            cases.R * np.log(5)
        )
    assert model.baseline.excluded_internal_modes
    assert "A" not in network.species_ids
    assert all(s.phase == "gas" for s in network.species)
    assert network.check_element_balance()


def test_json_network_and_declaration_restore_identical_temperature_dependent_rates():
    _, model, network, _, amounts = setup()
    assert EnergyGrainMasterEquationSpec.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
    assert restored.to_dict() == network.to_dict()
    for temperature in (300, 1200):
        assert restored.reaction_rates(amounts, volume_L=1, temperature_K=temperature) == (
            network.reaction_rates(amounts, volume_L=1, temperature_K=temperature)
        )


@pytest.mark.parametrize("topology", ("branch", "two-well", "six-grain", "two-level"))
def test_original_coupled_advance_matches_independent_matrix_exponential(topology):
    card, model, network, reactor, amounts = setup(topology)
    initial = pack_gas_reactor_network_state(reactor, network, {"cell": amounts})
    result, diagnostic = advance_gas_reactor_network(reactor, network, initial, 0.2)
    actual = diagnostic["nodes"]["cell"]["amounts_mol"]
    expected = reference.trajectory(card, 600, amounts, [0.2])[0]
    assert actual == pytest.approx(expected, rel=3e-6, abs=2e-10)
    report = grain_population_report(model, actual)
    assert report["molecular_amount_mol"] == pytest.approx(0.01, abs=2e-12)
    delta_energy = sum(
        (actual[g.species_id] - amounts[g.species_id]) * g.energy_J_mol
        for g in model.molecular_states
    )
    assert diagnostic["bath_heat_J"] == pytest.approx(delta_energy, abs=2e-6)
    assert result.final_state.boundary_heat_J == pytest.approx(delta_energy, abs=2e-6)
    assert actual["He"] == amounts["He"]
    assert actual["Ar"] == amounts["Ar"]


def flow_world():
    card, amounts, temperature, flow, _ = cases.flow_inputs()
    model = EnergyGrainMasterEquationSpec.from_dict(card)
    reactor = GasReactorNetworkSpec(
        (GasReactorNodeSpec("feed", 1, chemistry_enabled=False), GasReactorNodeSpec("reactor", 1)),
        (GasFlowConnectionSpec("bath-feed", "feed", "reactor", coefficient=flow),),
        temperature,
        "reactor",
    )
    return card, model, reactor, amounts


def test_actual_finite_bath_flow_changes_pressure_and_original_thermal_ledgers():
    card, model, reactor, amounts = flow_world()
    runtime = ChemWorldRuntime(composition=energy_grain_world(model, reactor, amounts))
    try:
        runtime.reset(seed=38)
        outcome = runtime.execute({"operation": "wait", "duration_s": 1})
        assert outcome.info["transaction_status"] == "committed", outcome.info
        expected = reference.finite_flow(card, amounts, 600, 0.003, 1)
        states = local_states(runtime._state)
        for name, state in states.items():
            assert state.species_amounts == pytest.approx(
                expected["amounts"][name], rel=3e-6, abs=2e-10
            )
            assert state.phases.phases["gas"].species_amounts_mol == state.species_amounts
            assert state.thermal.vessels[name].energy_jacket_J == pytest.approx(
                expected["bath_heat_J"][name], rel=5e-6, abs=2e-6
            )
        assert runtime._state.ledger.time_s == 1
        assert runtime._state.vessel_elapsed_s == {"reactor": 1, "feed": 1}
        assert states["reactor"].species_amounts["He"] > amounts["reactor"]["He"]
        assert runtime._state.ledger.energy_jacket_J == pytest.approx(
            sum(expected["bath_heat_J"].values()), rel=5e-6, abs=2e-6
        )
    finally:
        runtime.close()


def test_original_world_snapshot_continuation_is_exact():
    _, model, reactor, amounts = flow_world()
    runtime = ChemWorldRuntime(composition=energy_grain_world(model, reactor, amounts))
    restored = None
    try:
        runtime.reset(seed=38)
        assert (
            runtime.execute({"operation": "wait", "duration_s": 0.2}).info["transaction_status"]
            == "committed"
        )
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for instance in (runtime, restored):
            assert (
                instance.execute({"operation": "wait", "duration_s": 0.3}).info[
                    "transaction_status"
                ]
                == "committed"
            )
        assert restored._state.to_dict() == runtime._state.to_dict()
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def test_collision_failure_rolls_back_entire_shared_physical_slice(monkeypatch):
    _, model, reactor, amounts = flow_world()
    runtime = ChemWorldRuntime(composition=energy_grain_world(model, reactor, amounts))
    try:
        runtime.reset(seed=38)
        before = deepcopy(runtime._state.to_dict())

        def fail(*args, **kwargs):
            raise RuntimeError("injected collision failure")

        monkeypatch.setattr(
            "chemworld.physchem.energy_grain_master_equation.evaluate_energy_grain_collision_rate",
            fail,
        )
        result = runtime.execute({"operation": "wait", "duration_s": 0.2})
        assert result.info["operation_outcome"] == "numerical_failure"
        after = runtime._state.to_dict()
        for key in ("ledger", "process"):
            for field in ("cost", "risk"):
                before[key].pop(field)
                after[key].pop(field)
        assert after == before
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "defect", ("temperature", "activities", "negative-bath", "negative-grain", "binding")
)
def test_rate_family_rejects_unphysical_or_mismatched_inputs(defect):
    _, _, network, _, amounts = setup()
    reaction = next(
        r for r in network.reactions if r.rate_law.equation_id == "energy_grain_collision"
    )
    temperature, activities = 600, None
    if defect == "temperature":
        temperature = 1700
    elif defect == "activities":
        activities = {}
    elif defect == "negative-bath":
        amounts["He"] = -1
    elif defect == "negative-grain":
        amounts[reaction.rate_law.parameters["source_grain_id"]] = -1
    else:
        reaction = replace(reaction, forward_orders={"He": 1})
    with pytest.raises(ValueError):
        evaluate_rate_law(
            reaction,
            concentrations_mol_L=amounts,
            temperature_K=temperature,
            activity_coefficients=activities,
        )


@pytest.mark.parametrize(
    "defect", ("modes", "Cv", "weight", "duplicate", "bath", "channel-energy", "uphill-sink")
)
def test_declaration_rejects_ambiguous_physics(defect):
    card = cases.declaration(topology="two-well")
    if defect == "modes":
        card["baseline"]["excluded_internal_modes"] = []
    elif defect == "Cv":
        card["baseline"]["cp_J_mol_K"] = 2
    elif defect == "weight":
        card["wells"][0]["grains"][0]["statistical_weight"] = 0
    elif defect == "duplicate":
        card["sinks"][0]["species_id"] = "A0"
    elif defect == "bath":
        card["wells"][0]["baths"][0]["bath_species_id"] = "missing"
    elif defect == "channel-energy":
        card["wells"][1]["grains"][1]["energy_J_mol"] = 10001
    else:
        card["sinks"][0]["energy_J_mol"] = 100000
    with pytest.raises(ValueError):
        EnergyGrainMasterEquationSpec.from_dict(card)


def test_zero_bath_has_only_authored_chemical_channels_and_no_hidden_pressure():
    card, model, network, _, amounts = setup()
    amounts.update({"He": 0, "Ar": 0})
    rates = network.reaction_rates(amounts, volume_L=1, temperature_K=600)
    assert all(rate == 0 for key, rate in rates.items() if key.startswith("grain-collision"))
    assert energy_grain_generator(model, temperature_K=600, bath_concentrations_mol_L={})[
        "generator_s_inverse"
    ] == pytest.approx(reference.generator(card, 600, {})[1])


def test_grain_query_is_read_only_and_does_not_create_well_total_inventory():
    _, model, _, _, amounts = setup()
    before = dict(amounts)
    report = grain_population_report(model, amounts)
    assert report["well_amounts_mol"] == {"A": 0.01}
    assert report["grained_internal_energy_J"] == 100
    assert amounts == before
    with pytest.raises(ValueError, match="undeclared"):
        grain_population_report(model, {**amounts, "A": 0.01})
