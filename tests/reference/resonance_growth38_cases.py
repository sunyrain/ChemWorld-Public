"""Prospectively fixed 16-unit / 90-comparison finite-resonance block."""

from dataclasses import replace

from chemworld.physchem.molecular_graph import GraphAtom, GraphBond, MolecularGraph
from reference.rmg37_fixtures import SOURCE

CASES = (
    ("identity-allyl", "identity", "allyl", 6),
    ("identity-butenyl", "identity", "butenyl", 6),
    ("identity-pentadienyl", "identity", "pentadienyl", 6),
    ("identity-negatives", "negatives", None, 6),
    ("conflict-same-id", "conflict", "same-id", 4),
    ("conflict-resonance-same-thermo", "conflict", "same-thermo", 4),
    ("conflict-resonance-different-thermo", "conflict", "different-thermo", 4),
    ("conflict-rate-source", "conflict", "rate", 4),
    ("truncated-closure", "truncation", None, 4),
    ("alias-paths-and-isomers", "aliases", None, 4),
    ("repeated-species-occurrences", "mapped", "occurrences", 4),
    ("original-37C-topology", "mapped", "topology", 4),
    ("growth-representative-0-effective", "growth", (0, 1.0), 10),
    ("growth-representative-1-effective", "growth", (1, 1.0), 10),
    ("growth-representative-1-per-path", "growth", (1, 3.0), 10),
    ("complete-original-false-objects", "compatibility", None, 4),
)


def declared_forms(name):
    """Full hand-declared closures, with the same fixed H nuclei in every form."""
    hs, states = {
        "allyl": ((2, 1, 2), ((0, (1, 2)), (2, (2, 1)))),
        "butenyl": ((3, 1, 1, 2), ((1, (1, 1, 2)), (3, (1, 2, 1)))),
        "pentadienyl": ((2, 1, 1, 1, 2), ((0, (1, 2, 1, 2)), (2, (2, 1, 1, 2)), (4, (2, 1, 2, 1)))),
    }[name]
    forms = []
    for radical, orders in states:
        atoms, bonds = [], []
        for i, count in enumerate(hs):
            atom = str(i)
            atoms.append(GraphAtom(atom, "C", 0, int(i == radical), 0))
            for h in range(count):
                key = f"H{i}.{h}"
                atoms.append(GraphAtom(key, "H", 0, 0, 0))
                bonds.append(GraphBond(atom, key, 1))
        bonds.extend(GraphBond(str(i), str(i + 1), order) for i, order in enumerate(orders))
        forms.append(MolecularGraph(name, tuple(atoms), tuple(bonds), 2, SOURCE))
    return tuple(forms)


def renamed(graph):
    labels = {a.atom_id: "bound:" + a.atom_id for a in graph.atoms}
    return replace(
        graph,
        atoms=tuple(replace(a, atom_id=labels[a.atom_id]) for a in graph.atoms),
        bonds=tuple(
            replace(b, first=labels[b.first], second=labels[b.second]) for b in graph.bonds
        ),
    )


def mapped_network(kind):
    from chemworld.examples.native_intramolecular_reactions import example_graphs, example_library
    from chemworld.physchem.reaction_family import (
        elementary_reaction_families,
        extended_radical_reaction_families,
    )
    from chemworld.physchem.reaction_generation import (
        GenerationPolicy,
        bind_generated_network,
        generate_reactions,
    )
    from reference.rmg37_fixtures import bindings, structures

    if kind == "occurrences":
        family = elementary_reaction_families()[1]
        seeds = {"methyl": structures()["methyl"]}
    else:
        family = extended_radical_reaction_families()[-1]
        seeds = {"hexenyl": example_graphs()["hexenyl"]}
    report = generate_reactions(seeds, (family,), policy=GenerationPolicy(allylic_resonance=True))
    if kind == "occurrences":
        species, rates = bindings(report)
    else:
        library = example_library()
        species = library.species
        rates = (
            replace(library.reactions[0].kinetics, reaction_id=report.reactions[0].reaction_id),
        )
    bound = bind_generated_network(
        report,
        tuple(replace(b, graph=renamed(b.graph)) for b in species),
        rates,
        network_id="38E-bound-map",
        reference_temperature_K=500.0,
    )
    return bound, (family,)
