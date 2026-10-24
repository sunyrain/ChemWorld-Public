"""Explicit Benson-style group sums compiled exactly to original gas thermo.

Local group counting, molecular symmetry/optical/nonlocal declarations and HBI
are distinct operations. No RMG implementation or parameter database is embedded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import isfinite, log
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_graph import (
    GraphAtom,
    GraphBond,
    MolecularGraph,
    MolecularPattern,
    find_pattern_matches,
    molecular_isomorphic,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_generation import SpeciesDataBinding
from chemworld.physchem.reaction_network_specs import SpeciesSpec


def _source(source: DataSourceProvenance) -> None:
    if not source.license or not source.version:
        raise ValueError("parameter data need explicit source version and license")


@dataclass(frozen=True)
class GroupThermoData:
    temperatures_K: tuple[float, ...]
    heat_capacities_J_mol_K: tuple[float, ...]
    enthalpy_298_J_mol: float
    entropy_298_J_mol_K: float
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        _source(self.source)
        object.__setattr__(self, "temperatures_K", tuple(self.temperatures_K))
        object.__setattr__(self, "heat_capacities_J_mol_K", tuple(self.heat_capacities_J_mol_K))
        t, cp = self.temperatures_K, self.heat_capacities_J_mol_K
        if (
            len(t) < 2
            or len(t) != len(cp)
            or t[0] != 298.15
            or any(a >= b for a, b in pairwise(t))
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (*t, *cp, self.enthalpy_298_J_mol, self.entropy_298_J_mol_K)
            )
        ):
            raise ValueError("group thermo needs finite increasing Cp nodes starting at 298.15 K")

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperatures_K": list(self.temperatures_K),
            "heat_capacities_J_mol_K": list(self.heat_capacities_J_mol_K),
            "enthalpy_298_J_mol": self.enthalpy_298_J_mol,
            "entropy_298_J_mol_K": self.entropy_298_J_mol_K,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GroupThermoData:
        data = dict(payload)
        data["source"] = DataSourceProvenance.from_dict(data["source"])
        return cls(**data)


@dataclass(frozen=True)
class RootedGroupNode:
    node_id: str
    parent_id: str | None
    pattern: MolecularPattern
    root_label: str
    data: GroupThermoData | None = None

    def __post_init__(self) -> None:
        if not self.node_id or self.root_label not in {a.label for a in self.pattern.atoms}:
            raise ValueError("group node needs identity and a bound root label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "pattern": self.pattern.to_dict(),
            "root_label": self.root_label,
            "data": self.data.to_dict() if self.data else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RootedGroupNode:
        return cls(
            payload["node_id"],
            payload["parent_id"],
            MolecularPattern.from_dict(payload["pattern"]),
            payload["root_label"],
            GroupThermoData.from_dict(payload["data"]) if payload["data"] else None,
        )


def rooted_group_ancestors(nodes: Sequence[RootedGroupNode]) -> dict[str, tuple[str, ...]]:
    """Validate a finite declared forest; each lineage starts at its own node."""
    by_id = {n.node_id: n for n in nodes}
    if not by_id or len(by_id) != len(nodes):
        raise ValueError("group hierarchy requires unique nonempty nodes")
    result: dict[str, tuple[str, ...]] = {}
    for node in nodes:
        lineage: list[str] = []
        current: str | None = node.node_id
        while current is not None:
            if current not in by_id or current in lineage:
                raise ValueError("group hierarchy has a missing parent or cycle")
            lineage.append(current)
            current = by_id[current].parent_id
        result[node.node_id] = tuple(lineage)
    return result


def match_rooted_group(
    graph: MolecularGraph, atom_id: str, nodes: Sequence[RootedGroupNode]
) -> RootedGroupNode:
    """One maximal matching branch, with every ancestor also matching this atom."""
    ancestry = rooted_group_ancestors(nodes)
    matched = {
        n.node_id
        for n in nodes
        if any(m[n.root_label] == atom_id for m in find_pattern_matches(graph, n.pattern))
    }
    valid = {key for key in matched if set(ancestry[key]) <= matched}
    leaves = valid - {parent for key in valid for parent in ancestry[key][1:]}
    if len(leaves) != 1:
        raise ValueError(f"{'missing' if not leaves else 'ambiguous'} rooted group at {atom_id}")
    return next(n for n in nodes if n.node_id in leaves)


@dataclass(frozen=True)
class MolecularThermoContext:
    graph: MolecularGraph
    symmetry_number: int
    optical_isomers: int
    nonlocal_increment: GroupThermoData
    source: DataSourceProvenance
    hbi_includes_electronic_entropy: bool

    def __post_init__(self) -> None:
        _source(self.source)
        if any(type(v) is not int or v < 1 for v in (self.symmetry_number, self.optical_isomers)):
            raise ValueError("symmetry and optical counts must be explicit positive integers")
        if type(self.hbi_includes_electronic_entropy) is not bool:
            raise ValueError("electronic entropy convention must be explicit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "symmetry_number": self.symmetry_number,
            "optical_isomers": self.optical_isomers,
            "nonlocal_increment": self.nonlocal_increment.to_dict(),
            "source": self.source.to_dict(),
            "hbi_includes_electronic_entropy": self.hbi_includes_electronic_entropy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularThermoContext:
        return cls(
            MolecularGraph.from_dict(payload["graph"]),
            payload["symmetry_number"],
            payload["optical_isomers"],
            GroupThermoData.from_dict(payload["nonlocal_increment"]),
            DataSourceProvenance.from_dict(payload["source"]),
            payload["hbi_includes_electronic_entropy"],
        )


@dataclass(frozen=True)
class GroupAdditivityLibrary:
    groups: tuple[RootedGroupNode, ...]
    hbi_groups: tuple[RootedGroupNode, ...]
    contexts: tuple[MolecularThermoContext, ...]
    elementary: tuple[SpeciesDataBinding, ...]
    reference_basis_id: str
    reference_pressure_Pa: float
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        for name in ("groups", "hbi_groups", "contexts", "elementary"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _source(self.source)
        rooted_group_ancestors(self.groups)
        if self.hbi_groups:
            rooted_group_ancestors(self.hbi_groups)
        if (
            not self.reference_basis_id
            or not isfinite(self.reference_pressure_Pa)
            or self.reference_pressure_Pa <= 0
        ):
            raise ValueError("group library needs a common explicit gas reference basis")
        for i, context in enumerate(self.contexts):
            if any(molecular_isomorphic(context.graph, old.graph) for old in self.contexts[:i]):
                raise ValueError("competing molecular thermo contexts")
        for entry in self.elementary:
            if any(a.element != "H" for a in entry.graph.atoms) or len(entry.graph.atoms) > 2:
                raise ValueError("elementary exceptions are limited to explicit H/H2 entries")
            if entry.thermo_declaration.get("reference_pressure_Pa") != self.reference_pressure_Pa:
                raise ValueError("elementary and group reference pressures disagree")


@dataclass(frozen=True)
class GroupThermoEstimate:
    binding: SpeciesDataBinding | None
    trace: dict[str, Any]
    failure_reason: str = ""

    @property
    def success(self) -> bool:
        return self.binding is not None


def _compile_thermo(
    graph: MolecularGraph,
    species_id: str,
    data: Sequence[GroupThermoData],
    entropy_shift: float,
    pressure: float,
) -> dict[str, Any]:
    grid = data[0].temperatures_K
    if any(item.temperatures_K != grid for item in data):
        raise ValueError("group and correction Cp grids must explicitly agree")
    cp = [sum(item.heat_capacities_J_mol_K[i] for item in data) for i in range(len(grid))]
    if any(value <= 0 for value in cp):
        raise ValueError("estimated total gas Cp must be positive on every linear segment")
    h = sum(item.enthalpy_298_J_mol for item in data)
    s = sum(item.entropy_298_J_mol_K for item in data) + entropy_shift
    rows = []
    for i, (lo, hi) in enumerate(pairwise(grid)):
        slope = (cp[i + 1] - cp[i]) / (hi - lo)
        intercept = cp[i] - slope * lo
        rows.append(
            [
                intercept / R,
                slope / R,
                0,
                0,
                0,
                (h - intercept * lo - slope * lo * lo / 2) / R,
                (s - intercept * log(lo) - slope * lo) / R,
            ]
        )
        h += intercept * (hi - lo) + slope * (hi * hi - lo * lo) / 2
        s += intercept * log(hi / lo) + slope * (hi - lo)
    return {
        "thermo": {
            "model": "NASA7",
            "temperature-ranges": list(grid),
            "data": rows,
            "note": f"Exact piecewise-linear-Cp group sum for {species_id}",
        },
        "composition": graph.composition,
        "reference_pressure_Pa": pressure,
    }


def estimate_group_thermochemistry(
    graph: MolecularGraph, species_id: str, library: GroupAdditivityLibrary
) -> GroupThermoEstimate:
    """Parameterize one declared graph; unresolved data remain a readable failure."""
    trace: dict[str, Any] = {
        "method": "explicit-group-additivity",
        "reference_basis_id": library.reference_basis_id,
    }
    try:
        if (
            graph.charge
            or any(
                a.charge or a.isotope is not None or a.element not in {"C", "H"}
                for a in graph.atoms
            )
            or not nx.is_tree(graph.to_networkx())
            or sum(a.radical_electrons for a in graph.atoms) > 1
        ):
            raise ValueError(
                "group estimation requires neutral nonisotopic acyclic C/H, <=1 radical"
            )
        elementary = [
            entry for entry in library.elementary if molecular_isomorphic(graph, entry.graph)
        ]
        if elementary:
            if len(elementary) != 1:
                raise ValueError("ambiguous elementary thermochemistry")
            entry = elementary[0]
            trace.update(method="explicit-elementary-entry", source=entry.source.to_dict())
            binding = replace(
                entry,
                graph=graph,
                species=replace(
                    entry.species,
                    species_id=species_id,
                    metadata={**entry.species.metadata, "parameter_estimation": trace},
                ),
            )
            return GroupThermoEstimate(binding, trace)
        if any(b.order != 1 for b in graph.bonds):
            raise ValueError("first group slice requires saturated C/H skeletons")
        contexts = [c for c in library.contexts if molecular_isomorphic(graph, c.graph)]
        if len(contexts) != 1:
            raise ValueError("missing or ambiguous molecular symmetry/nonlocal context")
        context = contexts[0]
        radical = [a for a in graph.atoms if a.radical_electrons]
        saturated = graph
        data, labels = [context.nonlocal_increment], []
        hbi_labels = []
        if radical:
            if radical[0].element != "C" or not context.hbi_includes_electronic_entropy:
                raise ValueError("carbon HBI including electronic entropy must be explicit")
            if not library.hbi_groups:
                raise ValueError("missing HBI group hierarchy")
            node = match_rooted_group(graph, radical[0].atom_id, library.hbi_groups)
            if node.data is None:
                raise ValueError("missing HBI group parameters: " + node.node_id)
            data.append(node.data)
            hbi_labels.append(node.node_id)
            key = "HBI_added_H"
            while key in {a.atom_id for a in graph.atoms}:
                key += "_"
            saturated = replace(
                graph,
                multiplicity=1,
                atoms=(
                    *tuple(
                        replace(a, radical_electrons=0) if a == radical[0] else a
                        for a in graph.atoms
                    ),
                    GraphAtom(key, "H", 0, 0, 0),
                ),
                bonds=(*graph.bonds, GraphBond(radical[0].atom_id, key, 1)),
            )
        for atom in saturated.atoms:
            if atom.element == "H":
                continue
            node = match_rooted_group(saturated, atom.atom_id, library.groups)
            if node.data is None:
                raise ValueError("missing thermochemical parameters: " + node.node_id)
            data.append(node.data)
            labels.append(node.node_id)
        if not labels:
            raise ValueError("H/H2 requires an explicit elementary entry")
        trace.update(
            group_counts=dict(sorted(Counter(labels).items())),
            hbi_counts=dict(sorted(Counter(hbi_labels).items())),
            symmetry_number=context.symmetry_number,
            optical_isomers=context.optical_isomers,
            nonlocal_source=context.nonlocal_increment.source.to_dict(),
            contribution_sources=[item.source.to_dict() for item in data],
        )
        declaration = _compile_thermo(
            graph,
            species_id,
            data,
            -R * log(context.symmetry_number / context.optical_isomers),
            library.reference_pressure_Pa,
        )
        binding = SpeciesDataBinding(
            graph,
            SpeciesSpec(
                species_id,
                graph.formula,
                phase="gas",
                charge=graph.charge,
                metadata={"parameter_estimation": trace},
            ),
            declaration,
            library.source,
        )
        return GroupThermoEstimate(binding, trace)
    except ValueError as exc:
        return GroupThermoEstimate(None, trace, str(exc))
