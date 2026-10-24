"""Declared original-graph paths for bounded intramolecular reaction families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from chemworld.physchem.molecular_graph import MolecularGraph, MolecularPattern


@dataclass(frozen=True)
class ReactionPathConstraint:
    """A full ordered path, optionally ending at an exocyclic double bond.

    Bridge bonds exclude pre-existing rings on the selected route. This is a
    connectivity eligibility rule, not a prediction of energetic selectivity.
    """

    labels: tuple[str, ...]
    bond_orders: tuple[int, ...]
    require_shortest: bool = True
    require_bridge_bonds: bool = True
    exocyclic_double_bond: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "bond_orders", tuple(self.bond_orders))
        if self.exocyclic_double_bond is not None:
            object.__setattr__(self, "exocyclic_double_bond", tuple(self.exocyclic_double_bond))
        if (
            len(self.labels) < 2
            or len(set(self.labels)) != len(self.labels)
            or any(not label for label in self.labels)
            or len(self.bond_orders) != len(self.labels) - 1
            or any(type(n) is not int or n not in (1, 2, 3) for n in self.bond_orders)
            or type(self.require_shortest) is not bool
            or type(self.require_bridge_bonds) is not bool
        ):
            raise ValueError("reaction path requires unique ordered labels and exact bond orders")
        exo = self.exocyclic_double_bond
        if exo is not None and (
            len(exo) != 2 or exo[0] != self.labels[-1] or not exo[1] or exo[1] in self.labels
        ):
            raise ValueError("exocyclic double bond must leave the final path atom")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_path_templates(
    constraints: Sequence[ReactionPathConstraint], patterns: Sequence[MolecularPattern]
) -> None:
    """Reject path declarations not represented in one original reactant template."""
    for constraint in constraints:
        labels = set(constraint.labels) | set(constraint.exocyclic_double_bond or ())
        candidates = [p for p in patterns if labels <= {a.label for a in p.atoms}]
        if len(candidates) != 1:
            raise ValueError("reaction path must bind within one reactant template")
        bonds = {frozenset((b.first, b.second)): b.orders for b in candidates[0].bonds}
        expected = [
            (*pair, order)
            for pair, order in zip(
                pairwise(constraint.labels), constraint.bond_orders, strict=True
            )
        ]
        if constraint.exocyclic_double_bond is not None:
            expected.append((*constraint.exocyclic_double_bond, 2))
        if any(bonds.get(frozenset((a, b))) != (order,) for a, b, order in expected):
            raise ValueError("reaction path and template must declare identical exact bonds")


def validate_reaction_paths(
    constraints: Sequence[ReactionPathConstraint],
    reactants: Sequence[MolecularGraph],
    matches: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], ...]:
    """Validate before editing; retain the actual original atom path as evidence."""
    records = []
    for constraint in constraints:
        needed = set(constraint.labels) | set(constraint.exocyclic_double_bond or ())
        owners = [i for i, match in enumerate(matches) if needed <= set(match)]
        if len(owners) != 1:
            raise ValueError("reaction path requires one matched original reactant")
        owner = owners[0]
        host, match = reactants[owner].to_networkx(), matches[owner]
        path = tuple(match[k] for k in constraint.labels)
        if len(set(path)) != len(path) or any(k not in host for k in path):
            raise ValueError("reaction path must be simple and contained in the original graph")
        edges = tuple(pairwise(path))
        if any(
            not host.has_edge(a, b) or host[a][b]["order"] != order
            for (a, b), order in zip(edges, constraint.bond_orders, strict=True)
        ):
            raise ValueError("reaction path original bond order mismatch")
        distance = nx.shortest_path_length(host, path[0], path[-1])
        if constraint.require_shortest and distance != len(path) - 1:
            raise ValueError("reaction path has a shorter original-graph route")
        if constraint.require_bridge_bonds:
            bridges = {frozenset(edge) for edge in nx.bridges(host)}
            if any(frozenset(edge) not in bridges for edge in edges):
                raise ValueError("reaction path contains a pre-existing ring bond")
        exo = constraint.exocyclic_double_bond
        external = tuple(match[k] for k in exo) if exo is not None else None
        if external is not None and (
            external[1] in path
            or not host.has_edge(*external)
            or host[external[0]][external[1]]["order"] != 2
        ):
            raise ValueError("reaction path requires its declared outside double-bond partner")
        records.append(
            {
                "reactant_index": owner,
                "atom_path": list(path),
                "bond_orders": list(constraint.bond_orders),
                "shortest_bonds": distance,
                "bridge_bonds_required": constraint.require_bridge_bonds,
                "exocyclic_double_bond": list(external) if external is not None else None,
            }
        )
    return tuple(records)
