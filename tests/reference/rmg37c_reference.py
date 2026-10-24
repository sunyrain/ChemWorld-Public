"""Independent finite graph signatures and scalar Cantera/SciPy reference."""

from collections import defaultdict
from itertools import pairwise, permutations, product
from types import SimpleNamespace

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp

from chemworld.physchem.molecular_graph import MolecularGraph

R = ct.gas_constant / 1000
TIMES = np.array([0, 0.25, 0.5, 0.75, 1.0])


def adjacency(graph):
    result = {a.atom_id: {} for a in graph.atoms}
    for edge in graph.bonds:
        result[edge.first][edge.second] = edge.order
        result[edge.second][edge.first] = edge.order
    return result


def signature(graph):
    """Enumerate heavy-atom permutations; terminal H decorations stay exact."""
    atoms = {a.atom_id: a for a in graph.atoms}
    adj = adjacency(graph)
    heavy = [k for k, a in atoms.items() if a.element != "H"]
    if not heavy:
        heavy = list(atoms)
    if any(k not in heavy and (len(adj[k]) != 1 or next(iter(adj[k])) not in heavy) for k in atoms):
        raise ValueError("reference supports only explicitly attached terminal H")

    def attributes(a):
        return (a.element, a.isotope or 0, a.charge, a.radical_electrons, a.lone_pairs)

    groups = defaultdict(list)
    for key in heavy:
        decoration = tuple(
            sorted((attributes(atoms[h]), order) for h, order in adj[key].items() if h not in heavy)
        )
        groups[(attributes(atoms[key]), decoration)].append(key)
    ordered = sorted(groups)
    matrices = []
    for assignment in product(*(permutations(groups[k]) for k in ordered)):
        order = sum(assignment, ())
        matrices.append(
            tuple(adj[a].get(b, 0) for i, a in enumerate(order) for b in order[i + 1 :])
        )
    return (graph.multiplicity, tuple((k, len(groups[k])) for k in ordered), min(matrices))


def product_signatures(report):
    return sorted(
        signature(report.species[k])
        for reaction in report.reactions
        for k, count in reaction.products.items()
        for _ in range(count)
    )


def maps_valid(report):
    for reaction in report.reactions:
        for path in reaction.paths:
            rows = path["atom_mapping"]
            for prefix, stoich in (
                ("reactant", reaction.reactants),
                ("product", reaction.products),
            ):
                pairs = [(r[prefix + "_index"], r[prefix + "_atom"]) for r in rows]
                if len(set(pairs)) != len(pairs):
                    return False
                occurrences = {(r[prefix + "_index"], r[prefix + "_species"]) for r in rows}
                if sorted(k for _, k in occurrences) != sorted(
                    k for k, n in stoich.items() for _ in range(n)
                ):
                    return False
                if set(pairs) != {
                    (i, a.atom_id) for i, k in occurrences for a in report.species[k].atoms
                }:
                    return False
            for row in rows:
                original = next(
                    a
                    for a in report.species[row["reactant_species"]].atoms
                    if a.atom_id == row["reactant_atom"]
                )
                final = next(
                    a
                    for a in report.species[row["product_species"]].atoms
                    if a.atom_id == row["product_atom"]
                )
                if (original.element, original.isotope, original.charge) != (
                    final.element,
                    final.isotope,
                    final.charge,
                ):
                    return False
    return True


def distance(adj, start, stop, banned=None):
    frontier, seen, length = {start}, {start}, 0
    while frontier:
        if stop in frontier:
            return length
        following = {
            b for a in frontier for b in adj[a] if b not in seen and frozenset((a, b)) != banned
        }
        seen.update(following)
        frontier, length = following, length + 1
    return None


def paths_valid(report, family_index):
    for reaction in report.reactions:
        for path in reaction.paths:
            records = path.get("topology", [])
            if family_index == 0:
                if records:
                    return False
                continue
            if len(records) != 1:
                return False
            record = records[0]
            source = next(
                row["reactant_species"]
                for row in path["atom_mapping"]
                if row["reactant_index"] == record["reactant_index"]
            )
            adj, route = adjacency(report.species[source]), record["atom_path"]
            expected = 3 if family_index == 1 else 4
            if len(route) != expected + 1 or len(set(route)) != len(route):
                return False
            if (
                distance(adj, route[0], route[-1]) != expected
                or record["shortest_bonds"] != expected
            ):
                return False
            if record["bond_orders"] != [1] * expected or not record["bridge_bonds_required"]:
                return False
            for a, b in pairwise(route):
                if adj[a].get(b) != 1 or distance(adj, a, b, frozenset((a, b))) is not None:
                    return False
            exo = record["exocyclic_double_bond"]
            if family_index == 3:
                if exo[0] != route[-1] or exo[1] in route or adj[exo[0]].get(exo[1]) != 2:
                    return False
            elif exo is not None:
                return False
    return True


def network_path_validation(network):
    """Check complete atom maps in the network's own stored graph coordinates."""
    species = {
        s.species_id: MolecularGraph.from_dict(s.metadata["generated_graph"])
        for s in network.species
    }
    reactions = [
        SimpleNamespace(
            reactants={k: int(-v) for k, v in r.stoichiometry.items() if v < 0},
            products={k: int(v) for k, v in r.stoichiometry.items() if v > 0},
            paths=r.metadata["atom_paths"],
        )
        for r in network.reactions
    ]
    report = SimpleNamespace(species=species, reactions=reactions)
    return {
        "nuclear_bijections": maps_valid(report),
        "original_paths": paths_valid(report, 3),
        "path_counts": [len(r.paths) for r in reactions],
        "identities": {k: signature(v) for k, v in species.items()},
    }


def reference_phase(temperature):
    # Public data construction, no upstream source and no native generated network.
    species = []
    for name, h in (("R", 50000), ("P", 45000)):
        s = ct.Species(name, {"C": 6, "H": 11})
        s.thermo = ct.ConstantCp(300, 1400, 101325, [298.15, h * 1000, 250000, 110000])
        species.append(s)
    reaction = ct.Reaction({"R": 1}, {"P": 1}, ct.ArrheniusRate(2, 0, 2000000))
    reaction.reversible = False
    gas = ct.Solution(thermo="ideal-gas", kinetics="gas", species=species, reactions=[reaction])
    gas.TPX = temperature, 101325, {"R": 1}
    return gas


def reference_history(temperature, isothermal):
    gas = reference_phase(temperature)
    capacity = 1 + 0.02 * (110 - R)

    def temp(extent):
        return temperature if isothermal else temperature + 5000 * extent / capacity

    def rhs(time, values):
        gas.TP = temp(values[0]), 101325
        return [gas.forward_rate_constants[0] * (0.02 - values[0])]

    result = solve_ivp(
        rhs, (0.0, 1.0), [0.0], method="DOP853", t_eval=TIMES, rtol=1e-12, atol=1e-14
    )
    if not result.success:
        raise ValueError(result.message)
    product_mol = result.y[0]
    if isothermal:
        exact = 0.02 * (1 - np.exp(-2 * np.exp(-2000 / (R * temperature)) * TIMES))
        if not np.allclose(product_mol, exact, rtol=1e-11, atol=1e-13):
            raise ValueError("reference ODE disagrees with analytic exponential")
    return np.column_stack((0.02 - product_mol, product_mol)), [temp(p) for p in product_mol]


def energy(amounts, temperature, *, support):
    return support * temperature + sum(
        n * (h + 110 * (temperature - 298.15) - R * temperature)
        for n, h in zip(amounts, (50000, 45000), strict=True)
    )
