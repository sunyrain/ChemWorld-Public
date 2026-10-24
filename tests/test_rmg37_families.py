"""Three actual graph-recipe families, atom maps and path multiplicity."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.molecular_graph import (
    AtomPattern,
    GraphAtom,
    MolecularGraph,
    MolecularPattern,
    enumerate_allylic_resonance,
    molecular_isomorphic,
)
from chemworld.physchem.reaction_family import (
    ReactionFamily,
    RecipeAction,
    elementary_reaction_families,
)
from chemworld.physchem.reaction_generation import GenerationPolicy, generate_reactions
from reference.rmg37_fixtures import SOURCE, structures


def product_names(report, reaction):
    known = structures()
    answer = {}
    for key, count in reaction.products.items():
        matches = [
            name
            for name, graph in known.items()
            if molecular_isomorphic(graph, report.species[key])
        ]
        assert len(matches) == 1
        answer[matches[0]] = count
    return answer


@pytest.mark.parametrize(
    "family,seeds,expected,paths",
    [
        (0, ("methane", "H"), [{"methyl": 1, "H2": 1}], [4]),
        (1, ("methyl",), [{"ethane": 1}], [1]),
        (2, ("ethyl",), [{"ethene": 1, "H": 1}], [3]),
        (2, ("propyl",), [{"ethene": 1, "methyl": 1}, {"propene": 1, "H": 1}], [1, 2]),
    ],
)
def test_complete_three_family_products_and_bijective_atom_mapping(family, seeds, expected, paths):
    graphs = structures()
    chosen = elementary_reaction_families()[family]
    restored = ReactionFamily.from_dict(json.loads(json.dumps(chosen.to_dict())))
    report = generate_reactions({k: graphs[k] for k in seeds}, (restored,))
    assert report.complete and not report.rejected
    actual = [product_names(report, r) for r in report.reactions]
    assert sorted(map(repr, map(sorted, (d.items() for d in actual)))) == sorted(
        map(repr, map(sorted, (d.items() for d in expected)))
    )
    assert sorted(len(r.paths) for r in report.reactions) == sorted(paths)
    for reaction in report.reactions:
        for path in reaction.paths:
            rows = path["atom_mapping"]
            assert len({(r["reactant_index"], r["reactant_atom"]) for r in rows}) == len(rows)
            assert len({(r["product_index"], r["product_atom"]) for r in rows}) == len(rows)
            for row in rows:
                left = {a.atom_id: a for a in report.species[row["reactant_species"]].atoms}[
                    row["reactant_atom"]
                ]
                right = {a.atom_id: a for a in report.species[row["product_species"]].atoms}[
                    row["product_atom"]
                ]
                assert (left.element, left.isotope) == (right.element, right.isotope)
    assert json.loads(json.dumps(report.to_dict(), allow_nan=False))["complete"]


def test_unordered_reactants_and_aliases_do_not_double_channels():
    graphs = structures()
    report = generate_reactions(
        {"methyl": graphs["methyl"], "alias": graphs["methyl"], "H": graphs["H"]},
        elementary_reaction_families()[1:2],
    )
    assert report.complete and len(report.reactions) == 3
    assert report.seed_aliases["alias"] == report.seed_aliases["methyl"]
    assert all(len(r.paths) == 1 for r in report.reactions)


def test_limits_and_bad_recipe_failures_are_retained():
    graphs = structures()
    family = elementary_reaction_families()[0]
    report = generate_reactions(
        {"methane": graphs["methane"], "H": graphs["H"]},
        (family,),
        policy=GenerationPolicy(maximum_matches=2),
    )
    assert not report.complete and report.truncated_reasons
    bad = replace(family, actions=family.actions[:2])
    rejected = generate_reactions({"methane": graphs["methane"], "H": graphs["H"]}, (bad,))
    assert rejected.complete and not rejected.reactions and len(rejected.rejected) == 4
    with pytest.raises(ValueError, match="bind"):
        replace(family, actions=(RecipeAction("CHANGE_RADICAL", ("missing",), 1),))
    with pytest.raises(ValueError, match="empty"):
        MolecularPattern.from_dict({"atoms": [{"label": "*1", "elements": []}], "bonds": []})


def test_resonance_aliasing_is_opt_in_and_bounded():
    a, b = enumerate_allylic_resonance(structures()["allyl"]).forms
    ordinary = generate_reactions({"a": a, "b": b}, ())
    assert ordinary.seed_aliases["a"] != ordinary.seed_aliases["b"]
    report = generate_reactions(
        {"a": a, "b": b}, (), policy=GenerationPolicy(allylic_resonance=True)
    )
    assert report.complete and report.seed_aliases["a"] == report.seed_aliases["b"]
    limited = generate_reactions(
        {"a": a, "b": b},
        (),
        policy=GenerationPolicy(allylic_resonance=True, maximum_resonance_forms=1),
    )
    assert not limited.complete


def test_resonance_forms_generate_distinct_addition_positions():
    graphs = structures()
    report = generate_reactions(
        {"allyl": graphs["allyl"], "H": graphs["H"]},
        elementary_reaction_families()[1:2],
        policy=GenerationPolicy(allylic_resonance=True),
    )
    assert report.complete
    channels = [r for r in report.reactions if r.reactants == {"H": 1, "allyl": 1}]
    assert len(channels) == 2
    products = [report.species[next(iter(r.products))] for r in channels]
    assert products[0].composition == products[1].composition == {"C": 4, "H": 8}
    assert not molecular_isomorphic(*products)


def test_declared_ionic_recipe_keeps_isotope_nuclei_and_total_charge():
    # A structural recipe fixture, not a physical gas-phase kinetic claim.
    cation = MolecularGraph("deuteron", (GraphAtom("D", "H", 1, 0, 0, 2),), (), 1, SOURCE)
    anion = MolecularGraph("hydride", (GraphAtom("H", "H", -1, 0, 1),), (), 1, SOURCE)
    recipe = ReactionFamily(
        "declared_ionic_bond",
        (
            MolecularPattern((AtomPattern("*1", elements=("H",), charges=(1,)),)),
            MolecularPattern((AtomPattern("*2", elements=("H",), charges=(-1,)),)),
        ),
        (
            RecipeAction("FORM_BOND", ("*1", "*2"), 1),
            RecipeAction("CHANGE_CHARGE", ("*1",), -1),
            RecipeAction("CHANGE_CHARGE", ("*2",), 1),
            RecipeAction("CHANGE_LONE_PAIR", ("*2",), -1),
        ),
        SOURCE,
    )
    report = generate_reactions({"D+": cation, "H-": anion}, (recipe,))
    assert report.complete and not report.rejected and len(report.reactions) == 1
    reaction = report.reactions[0]
    product = report.species[next(iter(reaction.products))]
    assert product.charge == 0
    assert product.composition == {"[2H]": 1, "H": 1}
    mapping = reaction.paths[0]["atom_mapping"]
    deuteron = next(row for row in mapping if row["reactant_species"] == "D+")
    assert next(a for a in product.atoms if a.atom_id == deuteron["product_atom"]).isotope == 2
