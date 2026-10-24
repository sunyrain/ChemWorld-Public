"""Authored full structures and expected products, independent of reaction recipes."""

from dataclasses import replace

from chemworld.examples.native_intramolecular_reactions import SOURCE
from chemworld.physchem.molecular_graph import GraphAtom, GraphBond, MolecularGraph


def graph(name, rows, edges, *, isotope_at=None):
    atoms, bonds = [], [GraphBond(f"C{a}", f"C{b}", order) for a, b, order in edges]
    for i, (hs, radical) in enumerate(rows, 1):
        atoms.append(GraphAtom(f"C{i}", "C", 0, radical, 0))
        for j in range(hs):
            key = f"H{i}_{j}"
            atoms.append(GraphAtom(key, "H", 0, 0, 0, 2 if isotope_at == (i, j) else None))
            bonds.append(GraphBond(f"C{i}", key, 1))
    return MolecularGraph(
        name, tuple(atoms), tuple(bonds), 1 + sum(a.radical_electrons for a in atoms), SOURCE
    )


def chain(name, rows, *, double=None, isotope_at=None):
    return graph(
        name,
        rows,
        [(i, i + 1, 2 if i == double else 1) for i in range(1, len(rows))],
        isotope_at=isotope_at,
    )


def fixture_data():
    h = MolecularGraph("H", (GraphAtom("H", "H", 0, 1, 0),), (), 2, SOURCE)
    methyl = chain("methyl", [(3, 1)])
    ethene = chain("ethene", [(2, 0), (2, 0)], double=1)
    propene = chain("propene", [(2, 0), (1, 0), (3, 0)], double=1)
    ethyl = chain("ethyl", [(3, 0), (2, 1)])
    propyl = chain("npropyl", [(2, 1), (2, 0), (3, 0)])
    isopropyl = chain("isopropyl", [(3, 0), (1, 1), (3, 0)])
    secbutyl = chain("secbutyl", [(3, 0), (2, 0), (1, 1), (3, 0)])
    isobutyl = graph(
        "isobutyl", [(2, 1), (1, 0), (3, 0), (3, 0)], [(1, 2, 1), (2, 3, 1), (2, 4, 1)]
    )
    hexyl = chain("hexyl", [(2, 1), (2, 0), (2, 0), (2, 0), (2, 0), (3, 0)])
    shift4 = chain("hexyl4", [(3, 0), (2, 0), (2, 0), (1, 1), (2, 0), (3, 0)])
    shift5 = chain("hexyl5", [(3, 0), (2, 0), (2, 0), (2, 0), (1, 1), (3, 0)])
    hexenyl = chain("hexenyl", [(2, 1), (2, 0), (2, 0), (2, 0), (1, 0), (2, 0)], double=5)
    heptenyl = chain("heptenyl", [(2, 1), (2, 0), (2, 0), (2, 0), (1, 0), (1, 0), (3, 0)], double=5)
    cycle_edges = [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 1, 1), (5, 6, 1)]
    cycle = graph(
        "cyclopentylmethyl", [(2, 0), (2, 0), (2, 0), (2, 0), (1, 0), (2, 1)], cycle_edges
    )
    cycle_ethyl = graph(
        "cyclopentylethyl",
        [(2, 0), (2, 0), (2, 0), (2, 0), (1, 0), (1, 1), (3, 0)],
        [*cycle_edges, (6, 7, 1)],
    )
    return locals()


def cases():
    f = fixture_data()
    result = []
    for index, (family, seeds, expected, paths) in enumerate(
        (
            (0, ("h", "ethene"), ("ethyl",), 2),
            (0, ("methyl", "ethene"), ("propyl",), 2),
            (0, ("h", "propene"), ("propyl", "isopropyl"), 2),
            (0, ("methyl", "propene"), ("secbutyl", "isobutyl"), 2),
            (1, ("hexyl",), ("shift4",), 2),
            (2, ("hexyl",), ("shift5",), 2),
            (3, ("hexenyl",), ("cycle",), 1),
            (3, ("heptenyl",), ("cycle_ethyl",), 1),
        )
    ):
        result.append(
            {
                "name": f"products-{index}",
                "kind": "product",
                "family": family,
                "seeds": {name: f[name] for name in seeds},
                "expected": [f[name] for name in expected],
                "paths": paths,
                "expected_comparisons": 5,
            }
        )
    blocked = graph(
        "no-donor-H",
        [(2, 1), (2, 0), (2, 0), (0, 0), (3, 0), (3, 0), (3, 0)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1)],
    )
    four = chain("4-exo", [(2, 1), (2, 0), (2, 0), (1, 0), (2, 0)], double=4)
    six = chain("6-exo", [(2, 1), (2, 0), (2, 0), (2, 0), (2, 0), (1, 0), (2, 0)], double=6)
    shortcut = graph(
        "shortcut",
        [(1, 1), (2, 0), (2, 0), (1, 0), (1, 0), (2, 0)],
        [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 2), (1, 4, 1)],
    )
    for index, (family, source) in enumerate(
        ((1, blocked), (1, f["propyl"]), (3, four), (3, six), (3, shortcut))
    ):
        result.append(
            {
                "name": f"topology-{index}",
                "kind": "topology",
                "family": family,
                "seeds": {"source": source},
                "expected": [],
                "expected_comparisons": 4,
                "shorter_route": index == 4,
            }
        )
    original = f["hexenyl"]
    rename = {a.atom_id: f"atom_{i}" for i, a in enumerate(reversed(original.atoms))}
    renamed = replace(
        original,
        atoms=tuple(replace(a, atom_id=rename[a.atom_id]) for a in reversed(original.atoms)),
        bonds=tuple(
            GraphBond(rename[b.second], rename[b.first], b.order) for b in reversed(original.bonds)
        ),
    )
    result.append(
        {
            "name": "topology-rename",
            "kind": "topology",
            "family": 3,
            "seeds": {"renamed": renamed},
            "expected": [f["cycle"]],
            "expected_comparisons": 4,
        }
    )
    for index, original_index in enumerate((4, 5, 0)):
        result.append({**result[original_index], "name": f"maps-{index}", "kind": "maps"})
    deuterated = replace(
        f["hexyl"],
        atoms=tuple(replace(a, isotope=2) if a.atom_id == "H4_0" else a for a in f["hexyl"].atoms),
    )
    isotopes = [
        chain("D-product", [(3, 0), (2, 0), (2, 0), (1, 1), (2, 0), (3, 0)], isotope_at=position)
        for position in ((1, 0), (4, 0))
    ]
    result.append(
        {
            "name": "maps-isotope",
            "kind": "maps",
            "family": 1,
            "seeds": {"deuterated": deuterated},
            "expected": isotopes,
            "paths": 2,
            "expected_comparisons": 5,
        }
    )
    for temperature in (500.0, 900.0):
        for isothermal in (True, False):
            result.append(
                {
                    "name": f"world-{temperature}-{isothermal}",
                    "kind": "world",
                    "temperature": temperature,
                    "isothermal": isothermal,
                    "expected_comparisons": 10,
                }
            )
    for temperature in (500.0, 900.0):
        result.append(
            {
                "name": f"growth-{temperature}",
                "kind": "growth",
                "temperature": temperature,
                "expected_comparisons": 6,
            }
        )
    return result
