"""Independent labelled-bond replay and analytic finite two-channel reference.

No production graph matching, resonance search, recipe executor or ODE solver.
"""

from collections import Counter, defaultdict
from itertools import permutations, product
from math import exp


def atom_attributes(atom):
    return tuple(
        atom.get(key) for key in ("element", "isotope", "charge", "radical_electrons", "lone_pairs")
    )


def labelled_state(graph):
    return (
        {a["atom_id"]: atom_attributes(a) for a in graph["atoms"]},
        {tuple(sorted((b["first"], b["second"]))): b["order"] for b in graph["bonds"]},
        graph["multiplicity"],
    )


def signature(graph):
    """Exhaust all compatible heavy-nucleus permutations; H is explicit decoration."""
    atoms = {a["atom_id"]: a for a in graph["atoms"]}
    adj = {key: {} for key in atoms}
    for edge in graph["bonds"]:
        a, b = edge["first"], edge["second"]
        adj[a][b] = adj[b][a] = edge["order"]
    heavy = [key for key, atom in atoms.items() if atom["element"] != "H"] or list(atoms)
    groups = defaultdict(list)
    for key in heavy:
        decorations = []
        for h, order in adj[key].items():
            if h not in heavy:
                if len(adj[h]) != 1:
                    raise ValueError("reference requires terminal explicit H decorations")
                decorations.append((atom_attributes(atoms[h]), order))
        attributes = atom_attributes(atoms[key])
        # JSON gives null isotopes. repr sorts equivalent typed declarations consistently.
        groups[repr((attributes, sorted(decorations, key=repr)))].append(key)
    ordered = sorted(groups)
    matrices = []
    for assignment in product(*(permutations(groups[key]) for key in ordered)):
        order = sum(assignment, ())
        matrices.append(
            tuple(adj[a].get(b, 0) for i, a in enumerate(order) for b in order[i + 1 :])
        )
    return (graph["multiplicity"], tuple((key, len(groups[key])) for key in ordered), min(matrices))


def path_checks(network, families):
    """Return nuclear, actual-form recipe and original topology verdicts per path."""
    species = {s["species_id"]: s["metadata"]["generated_graph"] for s in network["species"]}
    rules = {f["family_id"]: f for f in families}
    results = []
    for reaction in network["reactions"]:
        family = rules[reaction["metadata"]["family_id"]]
        for path in reaction["metadata"]["atom_paths"]:
            reactants = {r["occurrence"]: r for r in path["reactant_forms"]}
            products = {r["occurrence"]: r for r in path["product_forms"]}
            nuclear = True
            for forms, expected in (
                (reactants, reaction["forward_orders"]),
                (
                    products,
                    {
                        key: reaction["forward_orders"].get(key, 0) + value
                        for key, value in reaction["stoichiometry"].items()
                        if reaction["forward_orders"].get(key, 0) + value
                    },
                ),
            ):
                nuclear &= Counter(r["species_id"] for r in forms.values()) == Counter(expected)
                for record in forms.values():
                    declared = species[record["species_id"]]
                    actual = record["graph"]

                    def nuclei(g):
                        return {a["atom_id"]: (a["element"], a.get("isotope")) for a in g["atoms"]}

                    def skeleton(g):
                        return {tuple(sorted((e["first"], e["second"]))) for e in g["bonds"]}

                    nuclear &= nuclei(declared) == nuclei(actual)
                    nuclear &= skeleton(declared) == skeleton(actual)
                    nuclear &= declared["multiplicity"] == actual["multiplicity"]
            source_atoms, source_bonds = {}, {}
            target_atoms, target_bonds = {}, {}
            for forms, atoms, bonds in (
                (reactants, source_atoms, source_bonds),
                (products, target_atoms, target_bonds),
            ):
                for index, row in forms.items():
                    graph = row["graph"]
                    atoms.update(
                        {(index, a["atom_id"]): list(atom_attributes(a)) for a in graph["atoms"]}
                    )
                    bonds.update(
                        {
                            tuple(sorted(((index, b["first"]), (index, b["second"])))): b["order"]
                            for b in graph["bonds"]
                        }
                    )
            maps = {
                (row["reactant_index"], row["reactant_atom"]): (
                    row["product_index"],
                    row["product_atom"],
                )
                for row in path["atom_mapping"]
            }
            nuclear &= len(maps) == len(path["atom_mapping"]) == len(source_atoms)
            nuclear &= set(maps) == set(source_atoms) and set(maps.values()) == set(target_atoms)
            nuclear &= all(source_atoms[a][:2] == target_atoms[b][:2] for a, b in maps.items())
            labels = {
                label: (i, atom)
                for i, match in enumerate(path["template_matches"])
                for label, atom in match.items()
            }
            topology = True
            for record in path.get("topology", []):
                i = record["reactant_index"]
                route = record["atom_path"]
                topology &= all(
                    source_bonds.get(tuple(sorted(((i, a), (i, b))))) == order
                    for a, b, order in zip(
                        route[:-1], route[1:], record["bond_orders"], strict=True
                    )
                )
                exo = record["exocyclic_double_bond"]
                if exo:
                    topology &= source_bonds.get(tuple(sorted(((i, exo[0]), (i, exo[1]))))) == 2
            for action in family["actions"]:
                kind, amount = action["kind"], action["value"]
                keys = [labels[label] for label in action["labels"]]
                if kind in ("FORM_BOND", "BREAK_BOND", "CHANGE_BOND"):
                    pair = tuple(sorted(keys))
                    before = source_bonds.get(pair, 0)
                    if kind == "FORM_BOND":
                        assert before == 0
                        source_bonds[pair] = amount
                    elif kind == "BREAK_BOND":
                        assert before == amount
                        del source_bonds[pair]
                    else:
                        source_bonds[pair] += amount
                else:
                    field = {"CHANGE_CHARGE": 2, "CHANGE_RADICAL": 3, "CHANGE_LONE_PAIR": 4}[kind]
                    source_atoms[keys[0]][field] += amount
            replay = {maps[a]: value for a, value in source_atoms.items()} == target_atoms
            replay &= {
                tuple(sorted((maps[a], maps[b]))): value for (a, b), value in source_bonds.items()
            } == target_bonds
            results.append(
                {
                    "nuclear_bijection": bool(nuclear),
                    "recipe_replay": bool(replay),
                    "topology_coordinates": bool(topology),
                }
            )
    return results


def analytic_amounts(time_s, *, factor=1.0, initial_r=0.1, initial_h=0.2, volume_L=1.0):
    k = 5.0 * factor / volume_L
    difference = initial_h - initial_r
    if difference == 0:
        remaining = initial_r / (1 + k * initial_r * time_s)
    else:
        e = exp(-k * difference * time_s)
        remaining = difference * initial_r * e / (initial_h - initial_r * e)
    consumed = initial_r - remaining
    return {
        "R": remaining,
        "H": remaining + difference,
        "butene1": consumed * 0.4,
        "butene2": consumed * 0.6,
    }
