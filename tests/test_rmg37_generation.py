"""Explicit parameter binding and execution through the original network engine."""

import json
from dataclasses import replace
from math import exp

import pytest

from chemworld.physchem.reaction_family import elementary_reaction_families
from chemworld.physchem.reaction_generation import (
    GenerationPolicy,
    bind_generated_network,
    generate_reactions,
)
from chemworld.physchem.reaction_network import BatchSolverOptions, ReactionNetworkSpec
from reference.rmg37_fixtures import bindings, structures


def make_network(family, seed, *, factor=1):
    report = generate_reactions(
        {seed: structures()[seed]}, elementary_reaction_families()[family : family + 1]
    )
    species, rates = bindings(report, factor=factor)
    network = bind_generated_network(
        report, species, rates, network_id="generated37", reference_temperature_K=500
    )
    return report, network


@pytest.mark.parametrize("family,seed", [(1, "methyl"), (2, "ethyl")])
def test_generated_native_network_matches_closed_form_trajectory(family, seed):
    report, network = make_network(family, seed)
    assert network.check_conservation()
    network = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
    times = [0, 0.5, 1, 2]
    result = network.integrate_batch(
        {seed: 0.1},
        volume_L=1,
        temperature_K=500,
        duration_s=2,
        evaluation_times_s=times,
        solver_options=BatchSolverOptions(method="DOP853", rtol=1e-10, atol_mol=1e-13),
    )
    for time, values in zip(result.times_s, zip(*result.amounts_mol, strict=True), strict=True):
        actual = dict(zip(result.species_ids, values, strict=True))
        expected = 0.1 / (1 + 2 * 0.2 * 0.1 * time) if family == 1 else 0.1 * exp(-0.2 * time)
        assert actual[seed] == pytest.approx(expected, rel=2e-7, abs=2e-10)
        for key, count in report.reactions[0].products.items():
            assert actual[key] == pytest.approx(
                count * (0.1 - expected) / (2 if family == 1 else 1), rel=2e-7, abs=2e-10
            )
    assert network.imported_gas_thermochemistry.keys() == set(network.species_ids)
    reaction = network.reactions[0]
    expected_h = sum(
        nu * network.imported_gas_thermochemistry[k].evaluate(500).enthalpy_J_mol
        for k, nu in reaction.stoichiometry.items()
    )
    assert reaction.delta_h_J_per_mol == pytest.approx(expected_h)


def test_explicit_degeneracy_is_separate_from_observed_path_count():
    report, base = make_network(2, "ethyl")
    assert len(report.reactions[0].paths) == 3
    _, changed = make_network(2, "ethyl", factor=7)
    assert base.reactions[0].rate_law.parameters["A"] == 0.2
    assert changed.reactions[0].rate_law.parameters["A"] == pytest.approx(1.4)


def test_missing_data_wrong_units_identity_and_truncation_refuse_export():
    graphs = structures()
    report = generate_reactions({"methyl": graphs["methyl"]}, elementary_reaction_families()[1:2])
    species, rates = bindings(report)
    kwargs = {"network_id": "x", "reference_temperature_K": 500}
    with pytest.raises(ValueError, match="exactly one"):
        bind_generated_network(report, species[:-1], rates, **kwargs)
    with pytest.raises(ValueError, match="kinetic"):
        bind_generated_network(report, species, (), **kwargs)
    with pytest.raises(ValueError, match="units"):
        bind_generated_network(
            report, species, (replace(rates[0], pre_exponential_unit="s^-1"),), **kwargs
        )
    with pytest.raises(ValueError, match="thermochemistry"):
        replace(
            species[0],
            thermo_declaration={**species[0].thermo_declaration, "composition": {"O": 1}},
        )
    limited = generate_reactions(
        {"ethyl": graphs["ethyl"]},
        elementary_reaction_families()[2:],
        policy=GenerationPolicy(maximum_matches=1),
    )
    with pytest.raises(ValueError, match="truncated"):
        bind_generated_network(limited, (), (), **kwargs)


def test_generated_network_runs_in_original_world_and_resumes_snapshot():
    from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    report, network = make_network(2, "ethyl")
    spec = GasReactorNetworkSpec(
        (
            GasReactorNodeSpec("reactor", 1),
            GasReactorNodeSpec("storage", 1, chemistry_enabled=False),
        ),
        (),
        500,
        "reactor",
    )
    original = ChemWorldRuntime(
        composition=gas_reactor_network_world(
            spec, network, {"reactor": {"ethyl": 0.1}, "storage": {"ethyl": 0.05}}
        )
    )
    resumed = None
    try:
        original.reset(seed=37)
        assert (
            original.execute({"operation": "wait", "duration_s": 0.5}).info["transaction_status"]
            == "committed"
        )
        resumed = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(original.snapshot().to_dict()))
        )
        for runtime in (original, resumed):
            outcome = runtime.execute({"operation": "wait", "duration_s": 0.5})
            assert outcome.info["transaction_status"] == "committed", outcome.info
            assert runtime._state.ledger.time_s == 1
            expected = 0.1 * exp(-0.2)
            assert runtime._state.species_amounts["ethyl"] == pytest.approx(expected, rel=2e-7)
            for key, count in report.reactions[0].products.items():
                assert runtime._state.species_amounts[key] == pytest.approx(
                    count * (0.1 - expected), rel=2e-7
                )
            assert (
                runtime._state.phases.phases["gas"].species_amounts_mol
                == runtime._state.species_amounts
            )
            assert runtime._state.inactive_vessels["storage"].species_amounts["ethyl"] == 0.05
        assert resumed._state.to_dict() == original._state.to_dict()
    finally:
        original.close()
        if resumed is not None:
            resumed.close()
