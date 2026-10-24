"""Authored explicit structures and synthetic parameter bindings, not an RMG library."""

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_graph import GraphAtom, GraphBond, MolecularGraph
from chemworld.physchem.reaction_generation import KineticsBinding, SpeciesDataBinding
from chemworld.physchem.reaction_network_specs import RateLawSpec, SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import rate_coefficient_unit

SOURCE = DataSourceProvenance(
    "synthetic-rmg37",
    0,
    "Authored graph and numerical test declarations",
    version="1",
    license="CC0-1.0",
)


def molecule(name, atoms, edges, *, multiplicity=1):
    """Each row explicitly declares element, charge, radicals, lone pairs and H count."""
    result, bonds = [], [GraphBond(str(i), str(j), order) for i, j, order in edges]
    for i, (element, charge, radicals, pairs, hydrogens) in enumerate(atoms):
        result.append(GraphAtom(str(i), element, charge, radicals, pairs))
        for j in range(hydrogens):
            key = f"{i}H{j}"
            result.append(GraphAtom(key, "H", 0, 0, 0))
            bonds.append(GraphBond(str(i), key, 1))
    return MolecularGraph(name, tuple(result), tuple(bonds), multiplicity, SOURCE)


def structures():
    return {
        "H": molecule("H", [("H", 0, 1, 0, 0)], [], multiplicity=2),
        "H2": molecule("H2", [("H", 0, 0, 0, 1)], []),
        "methane": molecule("methane", [("C", 0, 0, 0, 4)], []),
        "methyl": molecule("methyl", [("C", 0, 1, 0, 3)], [], multiplicity=2),
        "ethane": molecule("ethane", [("C", 0, 0, 0, 3), ("C", 0, 0, 0, 3)], [(0, 1, 1)]),
        "ethyl": molecule(
            "ethyl", [("C", 0, 1, 0, 2), ("C", 0, 0, 0, 3)], [(0, 1, 1)], multiplicity=2
        ),
        "ethene": molecule("ethene", [("C", 0, 0, 0, 2), ("C", 0, 0, 0, 2)], [(0, 1, 2)]),
        "propyl": molecule(
            "propyl",
            [("C", 0, 1, 0, 2), ("C", 0, 0, 0, 2), ("C", 0, 0, 0, 3)],
            [(0, 1, 1), (1, 2, 1)],
            multiplicity=2,
        ),
        "propene": molecule(
            "propene",
            [("C", 0, 0, 0, 2), ("C", 0, 0, 0, 1), ("C", 0, 0, 0, 3)],
            [(0, 1, 2), (1, 2, 1)],
        ),
        "ethanol": molecule(
            "ethanol",
            [("C", 0, 0, 0, 3), ("C", 0, 0, 0, 2), ("O", 0, 0, 2, 1)],
            [(0, 1, 1), (1, 2, 1)],
        ),
        "ether": molecule(
            "ether",
            [("C", 0, 0, 0, 3), ("O", 0, 0, 2, 0), ("C", 0, 0, 0, 3)],
            [(0, 1, 1), (1, 2, 1)],
        ),
        "allyl": molecule(
            "allyl",
            [("C", 0, 0, 0, 3), ("C", 0, 1, 0, 1), ("C", 0, 0, 0, 1), ("C", 0, 0, 0, 2)],
            [(0, 1, 1), (1, 2, 1), (2, 3, 2)],
            multiplicity=2,
        ),
    }


def bindings(report, *, rate=0.2, factor=1.0):
    species = []
    for key, graph in report.species.items():
        node = {
            "thermo": {
                "model": "NASA7",
                "temperature-ranges": [200.0, 2000.0],
                "data": [[4.0, 0.0, 0.0, 0.0, 0.0, 100.0 * len(graph.atoms), 2.0]],
            },
            "composition": graph.composition,
            "reference_pressure_Pa": 101325.0,
        }
        species.append(
            SpeciesDataBinding(
                graph,
                SpeciesSpec(key, graph.formula, phase="gas", charge=graph.charge),
                node,
                SOURCE,
            )
        )
    kinetics = [
        KineticsBinding(
            r.reaction_id,
            RateLawSpec(r.reaction_id + "-rate", "arrhenius", {"A": rate, "b": 0.0, "Ea": 0.0}),
            rate_coefficient_unit(sum(r.reactants.values())),
            "J/mol",
            "per_declared_path" if factor != 1 else "effective_reaction",
            factor,
            SOURCE,
        )
        for r in report.reactions
    ]
    return species, kinetics
