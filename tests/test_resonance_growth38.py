"""Finite resonance identity reaches original growth without duplicate inventories."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.resonance_mechanism_growth import example_resonance_inputs, run_example
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    CoreEdgeGrowthPolicy,
    GrowthResourceLimits,
    MechanismGrowthCheckpoint,
    MechanismGrowthLibrary,
)
from chemworld.physchem.molecular_identity import molecular_identity_mapping
from chemworld.physchem.reaction_generation import (
    GenerationPolicy,
    bind_generated_network,
    generate_reactions,
)
from reference.resonance_growth38_reference import analytic_amounts, path_checks


def inputs(**kwargs):
    return example_resonance_inputs(**kwargs)


def run(*, library=None, resources=None, checkpoint=None, **kwargs):
    declared, families, conditions = inputs(**kwargs)
    return grow_mechanism(
        library or declared,
        families,
        ("H", "R"),
        conditions,
        policy=CoreEdgeGrowthPolicy(generation=GenerationPolicy(allylic_resonance=True)),
        resources=resources,
        checkpoint=checkpoint,
    )


def test_shared_identity_is_opt_in_complete_and_nuclear():
    a = inputs()[0].species_by_id["R"].graph
    b = inputs(representative=1)[0].species_by_id["R"].graph
    assert molecular_identity_mapping(a, b) is None
    mapping = molecular_identity_mapping(a, b, allylic_resonance=True)
    assert set(mapping) == {x.atom_id for x in a.atoms}
    assert set(mapping.values()) == {x.atom_id for x in b.atoms}
    with pytest.raises(ValueError, match="truncated"):
        molecular_identity_mapping(a, b, allylic_resonance=True, maximum_forms=1)


@pytest.mark.parametrize("changed_thermo", [False, True])
def test_ambiguous_resonance_library_fails_before_any_integration(monkeypatch, changed_thermo):
    from chemworld.physchem.reaction_network import ReactionNetworkSpec

    library, _, _ = inputs()
    alias = inputs(representative=1)[0].species_by_id["R"]
    declaration = json.loads(json.dumps(alias.thermo_declaration))
    if changed_thermo:
        declaration["thermo"]["data"][0][5] += 500
    alias = replace(
        alias,
        species=replace(alias.species, species_id="competing_R"),
        thermo_declaration=declaration,
    )
    conflict = replace(library, species=(*library.species, alias))
    monkeypatch.setattr(
        ReactionNetworkSpec,
        "integrate_batch",
        lambda *a, **kw: pytest.fail("invalid identity reached integration"),
    )
    with pytest.raises(ValueError, match=r"resonance_identity_conflict.*competing_R"):
        run(library=conflict)


def test_duplicate_ids_and_rates_keep_original_rejection():
    library, _, _ = inputs()
    with pytest.raises(ValueError, match="unique species"):
        replace(library, species=(*library.species, library.species[0]))
    with pytest.raises(ValueError, match="reaction keys"):
        replace(library, reactions=(*library.reactions, library.reactions[0]))


@pytest.mark.parametrize("representative", [0, 1])
def test_growth_is_real_and_bound_forms_replay_in_library_coordinates(representative):
    library, families, _ = inputs(representative=representative)
    result = run(representative=representative)
    assert result.complete and len(result.network.reactions) == 2
    assert set(result.network.species_ids) == {"R", "H", "butene1", "butene2"}
    assert result.checkpoint.state["history"][0]["core_ids"] == ["H", "R"]
    assert set(result.checkpoint.state["history"][0]["decisions"]["promoted_species"]) == {
        "butene1",
        "butene2",
    }
    checks = path_checks(result.network.to_dict(), [f.to_dict() for f in families])
    assert len(checks) == 2 and all(all(row.values()) for row in checks)
    for species in result.network.species:
        assert (
            species.metadata["generated_graph"]
            == library.species_by_id[species.species_id].graph.to_dict()
        )


@pytest.mark.parametrize("factor", [1.0, 3.0])
def test_only_explicit_degeneracy_changes_rates_and_analytic_trajectory(factor):
    result = run(per_path_factor=factor)
    values = sorted(r.rate_law.parameters["A"] for r in result.network.reactions)
    assert values == [2 * factor, 3 * factor]
    samples = result.checkpoint.state["history"][-1]["conditions"][0]["samples"]
    for sample in samples:
        assert sample["core_amounts_mol"] == pytest.approx(
            analytic_amounts(sample["time_s"], factor=factor), rel=2e-7, abs=2e-11
        )


def test_alias_seeds_do_not_duplicate_paths_or_erase_actual_isomers():
    library, families, _ = inputs()
    alias = inputs(representative=1)[0].species_by_id["R"].graph
    seeds = {"R": library.species_by_id["R"].graph, "H": library.species_by_id["H"].graph}
    baseline = generate_reactions(seeds, families, policy=GenerationPolicy(allylic_resonance=True))
    result = generate_reactions(
        {**seeds, "another_R": alias}, families, policy=GenerationPolicy(allylic_resonance=True)
    )
    assert result.seed_aliases["R"] == result.seed_aliases["another_R"]
    assert [r.to_dict() for r in result.reactions] == [r.to_dict() for r in baseline.reactions]
    assert len(result.reactions) == 2


def test_growth_resource_stop_json_continues_exactly():
    complete = run()
    stopped = run(resources=GrowthResourceLimits(maximum_rounds=1))
    assert stopped.status == "resource_limit"
    checkpoint = MechanismGrowthCheckpoint.from_dict(
        json.loads(json.dumps(stopped.checkpoint.to_dict()))
    )
    resumed = run(checkpoint=checkpoint)
    assert resumed.network.to_dict() == complete.network.to_dict()
    assert resumed.checkpoint.state["history"] == complete.checkpoint.state["history"]


def test_original_world_wait_and_complete_snapshot_replay():
    result = run_example()
    assert result["snapshot_replay_equal"] and result["world_time_s"] == 1.0
    assert result["world_amounts_mol"] == pytest.approx(analytic_amounts(1.0), rel=2e-7, abs=2e-11)


def test_library_codec_and_no_false_path_metadata():
    library, families, _ = inputs()
    assert (
        MechanismGrowthLibrary.from_dict(json.loads(json.dumps(library.to_dict()))).to_dict()
        == library.to_dict()
    )
    report = generate_reactions(
        {"R": library.species_by_id["R"].graph, "H": library.species_by_id["H"].graph}, families
    )
    assert len(report.reactions) == 1
    assert all(
        "reactant_forms" not in path and "product_forms" not in path
        for r in report.reactions
        for path in r.paths
    )


@pytest.mark.parametrize("kind", ["repeated-occurrence", "topology"])
def test_all_bound_occurrences_and_topology_labels_follow_library_nuclei(kind):
    from chemworld.examples.native_intramolecular_reactions import example_graphs, example_library
    from chemworld.physchem.reaction_family import (
        elementary_reaction_families,
        extended_radical_reaction_families,
    )
    from reference.rmg37_fixtures import bindings, structures

    if kind == "repeated-occurrence":
        family = elementary_reaction_families()[1]
        report = generate_reactions(
            {"methyl": structures()["methyl"]},
            (family,),
            policy=GenerationPolicy(allylic_resonance=True),
        )
        species, kinetics = bindings(report)
    else:
        family = extended_radical_reaction_families()[-1]
        report = generate_reactions(
            {"hexenyl": example_graphs()["hexenyl"]},
            (family,),
            policy=GenerationPolicy(allylic_resonance=True),
        )
        library = example_library()
        species = library.species
        kinetics = [
            replace(library.reactions[0].kinetics, reaction_id=report.reactions[0].reaction_id)
        ]
    renamed = []
    for binding in species:
        graph = binding.graph
        labels = {a.atom_id: "library:" + a.atom_id for a in graph.atoms}
        renamed.append(
            replace(
                binding,
                graph=replace(
                    graph,
                    atoms=tuple(replace(a, atom_id=labels[a.atom_id]) for a in graph.atoms),
                    bonds=tuple(
                        replace(b, first=labels[b.first], second=labels[b.second])
                        for b in graph.bonds
                    ),
                ),
            )
        )
    network = bind_generated_network(
        report, renamed, kinetics, network_id="mapped", reference_temperature_K=500.0
    )
    checks = path_checks(network.to_dict(), [family.to_dict()])
    assert checks and all(all(row.values()) for row in checks)
    if kind == "repeated-occurrence":
        rows = network.reactions[0].metadata["atom_paths"][0]["atom_mapping"]
        assert {r["reactant_index"] for r in rows} == {0, 1}
