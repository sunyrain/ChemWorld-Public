"""Parameter estimation must actually feed existing mechanisms and replay contracts."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.native_parameter_estimation import (
    example_graphs,
    example_parameter_inputs,
    prepare_example,
    run_example,
)
from chemworld.physchem.kinetics_rate_rules import estimate_reaction_kinetics
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import GrowthCondition, MechanismGrowthLibrary
from chemworld.physchem.mechanism_parameter_estimation import prepare_parameterized_mechanism
from chemworld.physchem.molecular_graph import GraphAtom, GraphBond
from chemworld.physchem.reaction_generation import GenerationPolicy
from chemworld.physchem.reaction_rate_laws import arrhenius_k
from chemworld.physchem.thermo_group_additivity import estimate_group_thermochemistry
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml


def test_automatic_graph_closure_binds_five_species_and_three_rates():
    result = prepare_example()
    assert result.complete, result.reasons
    assert len(result.network.species) == 5 and len(result.network.reactions) == 3
    assert result.network.check_element_balance()
    assert {t["tier"] for t in result.kinetic_traces.values()} == {"library", "training", "rule"}
    copy = MechanismGrowthLibrary.from_dict(json.loads(json.dumps(result.growth_library.to_dict())))
    assert copy.to_dict() == result.growth_library.to_dict()
    assert result.generation_rounds == 2


def test_linear_cp_compilation_is_continuous_and_keeps_hbi_symmetry_trace():
    _, _, library, _ = example_parameter_inputs()
    result = estimate_group_thermochemistry(example_graphs()["methyl"], "methyl", library)
    assert result.success, result.failure_reason
    assert result.trace["symmetry_number"] == 6
    assert result.trace["hbi_counts"] == {"carbon-HBI": 1}
    model = species_thermo_from_cantera_yaml(
        species_id="methyl", **result.binding.thermo_declaration
    )
    for boundary in (400, 600, 1000):
        lo, hi = model.evaluate(boundary - 1e-7), model.evaluate(boundary + 1e-7)
        assert hi.enthalpy_J_mol - lo.enthalpy_J_mol == pytest.approx(
            2e-7 * model.evaluate(boundary).cp_J_mol_K, abs=2e-8
        )
        assert hi.entropy_J_mol_K - lo.entropy_J_mol_K == pytest.approx(
            2e-7 * model.evaluate(boundary).cp_J_mol_K / boundary, abs=2e-10
        )
    for outside in (298.14, 1500.01):
        with pytest.raises(ValueError):
            model.evaluate(outside)


def test_missing_parameters_and_enumeration_limits_retain_candidates():
    seeds, families, thermo, kinetics = example_parameter_inputs()
    for changed in (replace(thermo, contexts=()), replace(thermo, hbi_groups=())):
        result = prepare_parameterized_mechanism(
            seeds, families, changed, kinetics, network_id="missing", reference_temperature_K=500
        )
        assert not result.complete and result.network is None and result.reasons
        assert len(result.generation.species) == 5
    limited = prepare_parameterized_mechanism(
        seeds,
        families,
        thermo,
        kinetics,
        network_id="limited",
        reference_temperature_K=500,
        generation_policy=GenerationPolicy(maximum_reactions=1),
    )
    assert not limited.complete and any(
        r["kind"] == "generation_truncated" for r in limited.reasons
    )


def test_source_degeneracy_is_not_observed_path_count():
    result = prepare_example()
    _, _, _, kinetics = example_parameter_inputs()
    candidate = next(
        r
        for r in result.generation.reactions
        if {result.generation.species[k].formula for k in r.reactants} == {"CH3"}
    )
    changed = replace(candidate, paths=(*candidate.paths, *candidate.paths))
    first = estimate_reaction_kinetics(candidate, result.generation.species, kinetics)
    second = estimate_reaction_kinetics(changed, result.generation.species, kinetics)
    assert first.binding.degeneracy_factor == second.binding.degeneracy_factor == 1
    assert first.binding.rate_law.parameters["A"] == second.binding.rate_law.parameters["A"]


def test_outside_declared_temperature_rejects_even_zero_reactant_flux():
    network = prepare_example().network
    with pytest.raises(ValueError, match="kinetic validity"):
        network.reaction_rates({}, volume_L=1, temperature_K=200)
    with pytest.raises(ValueError, match="temperature range"):
        arrhenius_k({"A": 1, "b": 0, "Ea": 0, "valid_temperature_range_K": [500, 300]}, 400)
    assert arrhenius_k({"A": 1, "b": 0, "Ea": 0}, 200) == 1


def test_nonunit_declared_degeneracy_applies_once_in_both_consumers():
    seeds, families, thermo, kinetics = example_parameter_inputs()
    kinetics = replace(
        kinetics,
        rules=tuple(
            replace(
                entry,
                parameters=replace(
                    entry.parameters, degeneracy_basis="per_declared_path", degeneracy_factor=3
                ),
            )
            if entry.entry_id == "rule-CC"
            else entry
            for entry in kinetics.rules
        ),
    )
    prepared = prepare_parameterized_mechanism(
        seeds, families, thermo, kinetics, network_id="degeneracy", reference_temperature_K=500
    )
    assert prepared.complete, prepared.reasons
    growth = grow_mechanism(
        prepared.growth_library,
        families,
        ("H", "methyl"),
        (GrowthCondition("check", {"H": 0.1, "methyl": 0.1}, 500, 1, (0, 1)),),
    )
    assert growth.complete, growth.reasons
    for network in (prepared.network, growth.network):
        reaction = next(r for r in network.reactions if dict(r.reactants) == {"methyl": 2})
        assert arrhenius_k(reaction.rate_law.parameters, 500) == pytest.approx(3)
        rates = network.reaction_rates({"methyl": 0.1}, volume_L=1, temperature_K=500)
        assert rates[reaction.reaction_id] == pytest.approx(0.03)


def test_unknown_cyclic_or_multiple_radical_graph_is_not_silently_estimated():
    _, _, thermo, _ = example_parameter_inputs()
    cyclic = replace(
        example_graphs()["propane"],
        atoms=tuple(
            a for a in example_graphs()["propane"].atoms if a.atom_id not in {"0H2", "2H2"}
        ),
        bonds=(
            *tuple(
                b
                for b in example_graphs()["propane"].bonds
                if not {b.first, b.second} & {"0H2", "2H2"}
            ),
            GraphBond("0", "2", 1),
        ),
    )
    assert not estimate_group_thermochemistry(cyclic, "cyclic", thermo).success
    charged = replace(
        example_graphs()["methyl"],
        multiplicity=1,
        atoms=(GraphAtom("0", "C", 1, 0, 0), *example_graphs()["methyl"].atoms[1:]),
    )
    assert not estimate_group_thermochemistry(charged, "charged", thermo).success


def test_original_growth_world_and_snapshot_use_estimated_parameters():
    result = run_example()
    assert result["growth"]["complete"]
    assert result["time_s"] == 1
    assert result["snapshot_replay_equal"]
