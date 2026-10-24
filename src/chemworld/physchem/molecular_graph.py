"""Explicit bounded Lewis graphs and generic VF2 matching, without an identity parser.

Hydrogen, integer bonds, electrons and spin are declared. InChI/SMILES, aromatic
perception, stereochemistry and coordination chemistry are not inferred here.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.elements import get_element, get_isotope, hill_formula

_VALENCE_ELECTRONS = {"H": 1, "C": 4, "N": 5, "O": 6, "F": 7, "Cl": 7, "Br": 7, "I": 7}


def _integer(value: Any, name: str, *, minimum: int | None = None) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(name + " must be an explicit integer in its domain")


def _source(source: DataSourceProvenance) -> None:
    if not source.license.strip() or not source.version.strip():
        raise ValueError("graph/family data need an explicit source version and license")


@dataclass(frozen=True)
class GraphAtom:
    atom_id: str
    element: str
    charge: int
    radical_electrons: int
    lone_pairs: int
    isotope: int | None = None

    def __post_init__(self) -> None:
        if not self.atom_id or self.element not in _VALENCE_ELECTRONS:
            raise ValueError(
                "bounded Lewis graph requires atom identity and supported main-group element"
            )
        get_element(self.element)
        for key, value, minimum in (
            ("charge", self.charge, None),
            ("radicals", self.radical_electrons, 0),
            ("lone pairs", self.lone_pairs, 0),
        ):
            _integer(value, key, minimum=minimum)
        if self.isotope is not None:
            _integer(self.isotope, "isotope", minimum=1)
            get_isotope(self.element, self.isotope)

    @property
    def attributes(self) -> tuple[Any, ...]:
        return (self.element, self.isotope, self.charge, self.radical_electrons, self.lone_pairs)


@dataclass(frozen=True)
class GraphBond:
    first: str
    second: str
    order: int

    def __post_init__(self) -> None:
        if not self.first or not self.second or self.first == self.second:
            raise ValueError("bond needs two distinct declared atom identities")
        if type(self.order) is not int or self.order not in (1, 2, 3):
            raise ValueError("only explicit single/double/triple bonds are supported")


@dataclass(frozen=True)
class MolecularGraph:
    graph_id: str
    atoms: tuple[GraphAtom, ...]
    bonds: tuple[GraphBond, ...]
    multiplicity: int
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        _source(self.source)
        if (
            not self.graph_id
            or not self.atoms
            or len({a.atom_id for a in self.atoms}) != len(self.atoms)
        ):
            raise ValueError("molecule needs a graph identity and unique explicit atoms")
        _integer(self.multiplicity, "multiplicity", minimum=1)
        radicals = sum(a.radical_electrons for a in self.atoms)
        if self.multiplicity > radicals + 1 or (radicals + 1 - self.multiplicity) % 2:
            raise ValueError("declared spin multiplicity conflicts with unpaired electrons")
        orders = dict.fromkeys((a.atom_id for a in self.atoms), 0)
        edges = set()
        for bond in self.bonds:
            key = frozenset((bond.first, bond.second))
            if not key <= orders.keys() or key in edges:
                raise ValueError("duplicate bond or undeclared atom")
            edges.add(key)
            orders[bond.first] += bond.order
            orders[bond.second] += bond.order
        for atom in self.atoms:
            b = orders[atom.atom_id]
            charge = (
                _VALENCE_ELECTRONS[atom.element] - 2 * atom.lone_pairs - atom.radical_electrons - b
            )
            electrons = 2 * (atom.lone_pairs + b) + atom.radical_electrons
            if charge != atom.charge or electrons > (2 if atom.element == "H" else 8):
                raise ValueError(
                    "explicit Lewis electron/charge or octet contract fails: " + atom.atom_id
                )
        if not nx.is_connected(self.to_networkx()):
            raise ValueError("one MolecularGraph must be connected; declare separate reactants")

    @property
    def composition(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for atom in self.atoms:
            key = atom.element if atom.isotope is None else f"[{atom.isotope}{atom.element}]"
            result[key] = result.get(key, 0) + 1
        return result

    @property
    def formula(self) -> str:
        return hill_formula(self.composition)

    @property
    def charge(self) -> int:
        return sum(a.charge for a in self.atoms)

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from((a.atom_id, {"atom": a}) for a in self.atoms)
        graph.add_edges_from((b.first, b.second, {"order": b.order}) for b in self.bonds)
        return graph

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-explicit-lewis-graph-v1",
            "graph_id": self.graph_id,
            "atoms": [asdict(a) for a in self.atoms],
            "bonds": [asdict(b) for b in self.bonds],
            "multiplicity": self.multiplicity,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularGraph:
        if payload.get("model") != "chemworld-explicit-lewis-graph-v1":
            raise ValueError("unsupported molecular graph schema")
        if set(payload) - {"model", "graph_id", "atoms", "bonds", "multiplicity", "source"}:
            raise ValueError("unknown graph semantics")
        return cls(
            payload["graph_id"],
            tuple(GraphAtom(**a) for a in payload["atoms"]),
            tuple(GraphBond(**b) for b in payload["bonds"]),
            payload["multiplicity"],
            DataSourceProvenance.from_dict(dict(payload["source"])),
        )


def molecular_isomorphisms(left: MolecularGraph, right: MolecularGraph) -> Iterator[dict[str, str]]:
    """Exact identity ignores local atom labels, but keeps isotope/electrons/spin."""
    if left.multiplicity != right.multiplicity or left.composition != right.composition:
        return
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        left.to_networkx(),
        right.to_networkx(),
        node_match=lambda a, b: a["atom"].attributes == b["atom"].attributes,
        edge_match=lambda a, b: a["order"] == b["order"],
    )
    yield from matcher.isomorphisms_iter()


def molecular_isomorphic(left: MolecularGraph, right: MolecularGraph) -> bool:
    return next(molecular_isomorphisms(left, right), None) is not None


@dataclass(frozen=True)
class AtomPattern:
    label: str
    elements: tuple[str, ...] | None = None
    charges: tuple[int, ...] | None = None
    radicals: tuple[int, ...] | None = None
    lone_pairs: tuple[int, ...] | None = None
    isotopes: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("pattern atom needs a label")
        for name in ("elements", "charges", "radicals", "lone_pairs", "isotopes"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, tuple(values))
                if not values:
                    raise ValueError("empty atom constraint is not a wildcard")
        if self.elements is not None and set(self.elements) - _VALENCE_ELECTRONS.keys():
            raise ValueError("unsupported element/atom-type pattern")
        for name in ("charges", "radicals", "lone_pairs"):
            for value in getattr(self, name) or ():
                _integer(value, name, minimum=None if name == "charges" else 0)
        for value in self.isotopes or ():
            if value is not None:
                _integer(value, "pattern isotope", minimum=1)

    def matches(self, atom: GraphAtom) -> bool:
        return all(
            values is None or value in values
            for values, value in (
                (self.elements, atom.element),
                (self.charges, atom.charge),
                (self.radicals, atom.radical_electrons),
                (self.lone_pairs, atom.lone_pairs),
                (self.isotopes, atom.isotope),
            )
        )


@dataclass(frozen=True)
class BondPattern:
    first: str
    second: str
    orders: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", tuple(self.orders))
        if not self.orders or any(type(v) is not int or v not in (1, 2, 3) for v in self.orders):
            raise ValueError("pattern bond requires explicit integer bond-order alternatives")
        if not self.first or not self.second or self.first == self.second:
            raise ValueError("pattern bond needs distinct atom labels")


@dataclass(frozen=True)
class MolecularPattern:
    atoms: tuple[AtomPattern, ...]
    bonds: tuple[BondPattern, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        labels = {a.label for a in self.atoms}
        if not labels or len(labels) != len(self.atoms):
            raise ValueError("pattern labels must be nonempty and unique")
        edges = [frozenset((b.first, b.second)) for b in self.bonds]
        if len(set(edges)) != len(edges) or any(not e <= labels for e in edges):
            raise ValueError("duplicate pattern edge or undeclared label")
        if not nx.is_connected(self.to_networkx()):
            raise ValueError("each reactant pattern must be connected")

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from((a.label, {"pattern": a}) for a in self.atoms)
        graph.add_edges_from((b.first, b.second, {"orders": b.orders}) for b in self.bonds)
        return graph

    def to_dict(self) -> dict[str, Any]:
        return {"atoms": [asdict(a) for a in self.atoms], "bonds": [asdict(b) for b in self.bonds]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularPattern:
        if set(payload) != {"atoms", "bonds"}:
            raise ValueError("unsupported pattern fields")
        return cls(
            tuple(AtomPattern(**a) for a in payload["atoms"]),
            tuple(BondPattern(**b) for b in payload["bonds"]),
        )


def find_pattern_matches(
    graph: MolecularGraph, pattern: MolecularPattern
) -> Iterator[dict[str, str]]:
    """Yield pattern-label -> actual-atom mappings; unspecified bonds are permitted."""
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph.to_networkx(),
        pattern.to_networkx(),
        node_match=lambda a, b: b["pattern"].matches(a["atom"]),
        edge_match=lambda a, b: a["order"] in b["orders"],
    )
    for mapping in matcher.subgraph_monomorphisms_iter():
        yield {label: atom for atom, label in mapping.items()}


@dataclass(frozen=True)
class ResonanceResult:
    forms: tuple[MolecularGraph, ...]
    complete: bool
    transformations: tuple[tuple[str, str, str], ...]
    scope: str = "neutral_allylic_radical_1_3_shift"


def _labelled_key(graph: MolecularGraph) -> tuple[Any, ...]:
    return (
        tuple(sorted((a.atom_id, a.attributes) for a in graph.atoms)),
        tuple(
            sorted((min(b.first, b.second), max(b.first, b.second), b.order) for b in graph.bonds)
        ),
    )


def enumerate_allylic_resonance(
    graph: MolecularGraph, *, maximum_forms: int = 64
) -> ResonanceResult:
    """Bounded neutral 1,3 radical shifts; atom labels and all hydrogen nuclei stay fixed."""
    _integer(maximum_forms, "maximum_forms", minimum=1)
    forms, seen = [graph], {_labelled_key(graph)}
    steps: list[tuple[str, str, str]] = []
    for current in forms:
        network = current.to_networkx()
        for a in current.atoms:
            if a.charge or a.radical_electrons != 1 or a.element == "H":
                continue
            for b in network.neighbors(a.atom_id):
                middle = network.nodes[b]["atom"]
                if network[a.atom_id][b]["order"] != 1 or middle.charge or middle.radical_electrons:
                    continue
                for c in network.neighbors(b):
                    terminal = network.nodes[c]["atom"]
                    if (
                        c == a.atom_id
                        or terminal.charge
                        or terminal.radical_electrons
                        or terminal.element == "H"
                        or network[b][c]["order"] != 2
                    ):
                        continue
                    atoms = tuple(
                        replace(
                            x,
                            radical_electrons=x.radical_electrons
                            + (-1 if x.atom_id == a.atom_id else 1 if x.atom_id == c else 0),
                        )
                        for x in current.atoms
                    )
                    bonds = tuple(
                        replace(
                            edge,
                            order=edge.order
                            + (
                                1
                                if {edge.first, edge.second} == {a.atom_id, b}
                                else -1
                                if {edge.first, edge.second} == {b, c}
                                else 0
                            ),
                        )
                        for edge in current.bonds
                    )
                    try:
                        candidate = replace(current, atoms=atoms, bonds=bonds)
                    except ValueError:
                        continue
                    key = _labelled_key(candidate)
                    if key in seen:
                        continue
                    if len(forms) == maximum_forms:
                        return ResonanceResult(tuple(forms), False, tuple(steps))
                    seen.add(key)
                    forms.append(candidate)
                    steps.append((a.atom_id, b, c))
    return ResonanceResult(tuple(forms), True, tuple(steps))


def allylic_resonance_equivalent(
    left: MolecularGraph, right: MolecularGraph, *, maximum_forms: int = 64
) -> bool:
    forms = enumerate_allylic_resonance(left, maximum_forms=maximum_forms)
    if any(molecular_isomorphic(value, right) for value in forms.forms):
        return True
    if not forms.complete:
        raise ValueError("bounded resonance search incomplete; inequivalence is not established")
    return False


def molecular_graph_from_adjacency(
    text: str, *, graph_id: str, source: DataSourceProvenance
) -> MolecularGraph:
    """Explicit-H RMG-style molecule subset: multiplicity; index E uN pN cN [iN].

    Both directions of each S/D/T bond must be present. No atom types, implicit
    hydrogens, aromatic bonds, wildcard electrons or executable data are accepted.
    """
    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines or re.fullmatch(r"multiplicity\s+[1-9][0-9]*", lines[0]) is None:
        raise ValueError("explicit adjacency requires a multiplicity declaration")
    atoms, directed = [], {}
    for line in lines[1:]:
        bonds = re.findall(r"\{([1-9][0-9]*),([SDT])\}", line)
        header = re.sub(r"\{[1-9][0-9]*,[SDT]\}", "", line).strip()
        match = re.fullmatch(
            r"([1-9][0-9]*)\s+([A-Z][a-z]?)\s+u([0-9]+)\s+p([0-9]+)\s+c([+-]?[0-9]+)(?:\s+i([1-9][0-9]*))?",
            header,
        )
        if match is None:
            raise ValueError("unsupported or incomplete explicit adjacency atom")
        index, element, radical, pairs, charge, isotope = match.groups()
        atoms.append(
            GraphAtom(
                index,
                element,
                int(charge),
                int(radical),
                int(pairs),
                int(isotope) if isotope else None,
            )
        )
        for target, order in bonds:
            key = (index, target)
            if key in directed:
                raise ValueError("duplicate adjacency bond")
            directed[key] = {"S": 1, "D": 2, "T": 3}[order]
    if any(directed.get((b, a)) != value for (a, b), value in directed.items()):
        raise ValueError("adjacency bonds must be symmetric")
    edges = tuple(GraphBond(a, b, value) for (a, b), value in directed.items() if int(a) < int(b))
    return MolecularGraph(graph_id, tuple(atoms), edges, int(lines[0].split()[1]), source)
