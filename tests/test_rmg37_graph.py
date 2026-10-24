"""Explicit atom identities, VF2 matching and bounded resonance semantics."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.molecular_graph import (
    GraphAtom,
    GraphBond,
    MolecularGraph,
    allylic_resonance_equivalent,
    enumerate_allylic_resonance,
    molecular_graph_from_adjacency,
    molecular_isomorphic,
)
from reference.rmg37_fixtures import SOURCE, molecule, structures


def relabel(graph):
    mapping = {a.atom_id: "x" + str(i) for i, a in enumerate(reversed(graph.atoms))}
    return replace(
        graph,
        atoms=tuple(replace(a, atom_id=mapping[a.atom_id]) for a in reversed(graph.atoms)),
        bonds=tuple(
            GraphBond(mapping[b.first], mapping[b.second], b.order) for b in reversed(graph.bonds)
        ),
    )


@pytest.mark.parametrize(
    "key", ["H", "H2", "methane", "methyl", "ethane", "ethene", "ethanol", "ether"]
)
def test_atom_order_does_not_change_identity_and_json(key):
    graph = structures()[key]
    restored = MolecularGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert molecular_isomorphic(graph, relabel(restored))
    assert restored.formula == graph.formula


def test_same_formula_isomers_isotopes_and_electrons_remain_distinct():
    graph = structures()
    assert graph["ethanol"].composition == graph["ether"].composition
    assert not molecular_isomorphic(graph["ethanol"], graph["ether"])
    labelled = replace(
        graph["methane"],
        atoms=tuple(
            replace(a, isotope=13) if a.element == "C" else a for a in graph["methane"].atoms
        ),
    )
    assert not molecular_isomorphic(graph["methane"], labelled)
    cation = molecule("cation", [("C", 1, 0, 0, 3)], [])
    assert cation.composition == graph["methyl"].composition
    assert not molecular_isomorphic(cation, graph["methyl"])
    oxygen = molecule("oxygen", [("O", 0, 2, 2, 0)], [], multiplicity=3)
    assert not molecular_isomorphic(oxygen, replace(oxygen, multiplicity=1))


def test_allylic_radical_shift_preserves_nuclei_and_reports_truncation():
    original = structures()["allyl"]
    result = enumerate_allylic_resonance(original)
    assert result.complete and len(result.forms) == 2
    assert result.forms[0].composition == result.forms[1].composition
    assert not molecular_isomorphic(*result.forms)
    assert allylic_resonance_equivalent(*result.forms)
    assert {a.atom_id for a in result.forms[1].atoms} == {a.atom_id for a in original.atoms}
    assert not enumerate_allylic_resonance(original, maximum_forms=1).complete
    with pytest.raises(ValueError, match="incomplete"):
        allylic_resonance_equivalent(original, result.forms[1], maximum_forms=1)


def test_strict_explicit_adjacency_and_unsupported_semantics():
    graph = molecular_graph_from_adjacency(
        "multiplicity 1\n1 H u0 p0 c0 {2,S}\n2 H u0 p0 c0 {1,S}", graph_id="H2", source=SOURCE
    )
    assert molecular_isomorphic(graph, structures()["H2"])
    for text in (
        "1 H u0 p0 c0",
        "multiplicity 1\n1 C u0 p0 c0",
        "multiplicity 1\n1 H u0 p0 c0 {2,S}\n2 H u0 p0 c0",
        "multiplicity 1\n1 C u0 p0 c0 {2,B}",
    ):
        with pytest.raises(ValueError):
            molecular_graph_from_adjacency(text, graph_id="invalid", source=SOURCE)
    with pytest.raises(ValueError, match="unknown graph"):
        MolecularGraph.from_dict({**graph.to_dict(), "stereochemistry": "R"})
    with pytest.raises(ValueError, match="supported main-group"):
        GraphAtom("Fe", "Fe", 0, 0, 0)
    with pytest.raises(ValueError, match="spin"):
        replace(structures()["methyl"], multiplicity=1)
