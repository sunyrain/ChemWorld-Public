"""Declarative atom-labelled graph recipes, without rates or a second runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_graph import (
    AtomPattern,
    BondPattern,
    GraphBond,
    MolecularGraph,
    MolecularPattern,
)
from chemworld.physchem.reaction_family_topology import (
    ReactionPathConstraint,
    validate_path_templates,
    validate_reaction_paths,
)


@dataclass(frozen=True)
class RecipeAction:
    kind: str
    labels: tuple[str, ...]
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        bond = self.kind in ("FORM_BOND", "BREAK_BOND", "CHANGE_BOND")
        if not bond and self.kind not in ("CHANGE_RADICAL", "CHANGE_LONE_PAIR", "CHANGE_CHARGE"):
            raise ValueError("unsupported graph recipe action")
        if len(self.labels) != (2 if bond else 1) or len(set(self.labels)) != len(self.labels):
            raise ValueError("recipe action has incorrect atom-label arity")
        if type(self.value) is not int or not self.value:
            raise ValueError("recipe action needs an explicit nonzero integer change/order")
        if self.kind in ("FORM_BOND", "BREAK_BOND") and self.value not in (1, 2, 3):
            raise ValueError("bond creation/removal requires its exact integer order")


@dataclass(frozen=True)
class ReactionFamily:
    family_id: str
    reactant_patterns: tuple[MolecularPattern, ...]
    actions: tuple[RecipeAction, ...]
    source: DataSourceProvenance
    unordered_reactants: bool = False
    path_constraints: tuple[ReactionPathConstraint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reactant_patterns", tuple(self.reactant_patterns))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "path_constraints", tuple(self.path_constraints))
        if not self.family_id or not 1 <= len(self.reactant_patterns) <= 2 or not self.actions:
            raise ValueError("family requires identity, one/two reactant patterns and a recipe")
        if not self.source.version or not self.source.license:
            raise ValueError("family source version and data license are required")
        labels = [atom.label for p in self.reactant_patterns for atom in p.atoms]
        if len(set(labels)) != len(labels) or any(
            set(a.labels) - set(labels) for a in self.actions
        ):
            raise ValueError("family labels must be globally unique and all actions must bind")
        if type(self.unordered_reactants) is not bool:
            raise ValueError("unordered reactants must be explicitly boolean")
        validate_path_templates(self.path_constraints, self.reactant_patterns)
        if self.unordered_reactants and self.path_constraints:
            raise ValueError("path-constrained family profiles require ordered reactants")
        if self.unordered_reactants:
            if len(self.reactant_patterns) != 2:
                raise ValueError("unordered reactants need two equivalent templates")
            left, right = (p.to_networkx() for p in self.reactant_patterns)

            def node_match(a: dict, b: dict) -> bool:
                return {k: v for k, v in asdict(a["pattern"]).items() if k != "label"} == {
                    k: v for k, v in asdict(b["pattern"]).items() if k != "label"
                }

            if not nx.is_isomorphic(
                left,
                right,
                node_match=node_match,
                edge_match=lambda a, b: a["orders"] == b["orders"],
            ):
                raise ValueError("unordered reactants require equivalent patterns")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-reaction-family-v1",
            "family_id": self.family_id,
            "reactant_patterns": [p.to_dict() for p in self.reactant_patterns],
            "actions": [asdict(a) for a in self.actions],
            "source": self.source.to_dict(),
            "unordered_reactants": self.unordered_reactants,
            **(
                {"path_constraints": [p.to_dict() for p in self.path_constraints]}
                if self.path_constraints
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReactionFamily:
        if payload.get("model") != "chemworld-reaction-family-v1":
            raise ValueError("unsupported family schema")
        if set(payload) - {
            "model",
            "family_id",
            "reactant_patterns",
            "actions",
            "source",
            "unordered_reactants",
            "path_constraints",
        }:
            raise ValueError("unknown reaction-family semantics")
        return cls(
            payload["family_id"],
            tuple(MolecularPattern.from_dict(p) for p in payload["reactant_patterns"]),
            tuple(RecipeAction(**a) for a in payload["actions"]),
            DataSourceProvenance.from_dict(dict(payload["source"])),
            payload["unordered_reactants"],
            tuple(ReactionPathConstraint(**p) for p in payload.get("path_constraints", ())),
        )


@dataclass(frozen=True)
class RecipeProducts:
    products: tuple[MolecularGraph, ...]
    # Each product atom retains its occurrence-qualified original atom identity.
    original_atoms: dict[str, tuple[int, str]]
    topology_matches: tuple[dict[str, Any], ...] = ()


def apply_reaction_recipe(
    family: ReactionFamily,
    reactants: Sequence[MolecularGraph],
    matches: Sequence[Mapping[str, str]],
) -> RecipeProducts:
    """Apply exact edits to a temporary graph and validate every resulting Lewis graph."""
    if len(reactants) != len(family.reactant_patterns) or len(matches) != len(reactants):
        raise ValueError("recipe requires one graph and match for every reactant template")
    combined, labels, original = nx.Graph(), {}, {}
    for i, (graph, pattern, match) in enumerate(
        zip(reactants, family.reactant_patterns, matches, strict=True)
    ):
        if set(match) != {a.label for a in pattern.atoms} or len(set(match.values())) != len(match):
            raise ValueError("recipe matches must bind each pattern atom injectively")
        host = graph.to_networkx()
        for atom in pattern.atoms:
            target = match[atom.label]
            if target not in host or not atom.matches(host.nodes[target]["atom"]):
                raise ValueError("recipe atom match does not satisfy its pattern")
        for bond in pattern.bonds:
            a, b = match[bond.first], match[bond.second]
            if not host.has_edge(a, b) or host[a][b]["order"] not in bond.orders:
                raise ValueError("recipe bond match does not satisfy its pattern")
        mapping = {a.atom_id: f"r{i}:{a.atom_id}" for a in graph.atoms}
        original.update({mapping[a.atom_id]: (i, a.atom_id) for a in graph.atoms})
        combined.add_nodes_from(
            (mapping[a.atom_id], {"atom": replace(a, atom_id=mapping[a.atom_id])})
            for a in graph.atoms
        )
        combined.add_edges_from(
            (mapping[b.first], mapping[b.second], {"order": b.order}) for b in graph.bonds
        )
        labels.update({key: mapping[value] for key, value in match.items()})
    topology = validate_reaction_paths(family.path_constraints, reactants, matches)
    for action in family.actions:
        atoms = [labels[label] for label in action.labels]
        if action.kind in ("FORM_BOND", "BREAK_BOND", "CHANGE_BOND"):
            a, b = atoms
            exists = combined.has_edge(a, b)
            if action.kind == "FORM_BOND":
                if exists:
                    raise ValueError("FORM_BOND requires an absent bond")
                combined.add_edge(a, b, order=action.value)
            elif action.kind == "BREAK_BOND":
                if not exists or combined[a][b]["order"] != action.value:
                    raise ValueError("BREAK_BOND requires the declared existing order")
                combined.remove_edge(a, b)
            else:
                if not exists or combined[a][b]["order"] + action.value not in (1, 2, 3):
                    raise ValueError("CHANGE_BOND must retain a supported positive order")
                combined[a][b]["order"] += action.value
        else:
            field = {
                "CHANGE_RADICAL": "radical_electrons",
                "CHANGE_LONE_PAIR": "lone_pairs",
                "CHANGE_CHARGE": "charge",
            }[action.kind]
            atom = combined.nodes[atoms[0]]["atom"]
            combined.nodes[atoms[0]]["atom"] = replace(
                atom, **{field: getattr(atom, field) + action.value}
            )
    products = []
    for i, component in enumerate(sorted(nx.connected_components(combined), key=lambda c: min(c))):
        product_atoms = tuple(combined.nodes[key]["atom"] for key in sorted(component))
        radicals = sum(a.radical_electrons for a in product_atoms)
        if radicals > 1:
            raise ValueError(
                "generated multiradical product spin is ambiguous in this family profile"
            )
        bonds = tuple(
            GraphBond(a, b, data["order"])
            for a, b, data in combined.edges(data=True)
            if a in component and b in component
        )
        products.append(
            MolecularGraph(
                family.family_id + ":product:" + str(i),
                product_atoms,
                bonds,
                radicals + 1,
                family.source,
            )
        )
    initial_charge = sum(g.charge for g in reactants)
    if sum(g.charge for g in products) != initial_charge:
        raise ValueError("recipe changes total formal charge")
    if {a.atom_id for g in products for a in g.atoms} != set(original):
        raise ValueError("recipe failed nuclear atom conservation")
    return RecipeProducts(tuple(products), original, topology)


def elementary_reaction_families() -> tuple[ReactionFamily, ...]:
    """Authored elementary recipes; no upstream RMG parameter database is embedded."""
    source = DataSourceProvenance(
        "chemworld-elementary-recipes",
        0,
        "Explicit elementary graph edits; no kinetic/material parameters",
        version="1",
        license="MIT",
    )
    donor = MolecularPattern(
        (
            AtomPattern("*1", radicals=(0,), charges=(0,)),
            AtomPattern("*2", elements=("H",), radicals=(0,), charges=(0,)),
        ),
        (BondPattern("*1", "*2", (1,)),),
    )
    acceptor = MolecularPattern((AtomPattern("*3", radicals=(1,), charges=(0,)),))
    abstraction = ReactionFamily(
        "H_abstraction",
        (donor, acceptor),
        (
            RecipeAction("BREAK_BOND", ("*1", "*2"), 1),
            RecipeAction("FORM_BOND", ("*2", "*3"), 1),
            RecipeAction("CHANGE_RADICAL", ("*1",), 1),
            RecipeAction("CHANGE_RADICAL", ("*3",), -1),
        ),
        source,
    )
    recombination = ReactionFamily(
        "radical_recombination",
        (
            MolecularPattern((AtomPattern("*1", radicals=(1,), charges=(0,)),)),
            MolecularPattern((AtomPattern("*2", radicals=(1,), charges=(0,)),)),
        ),
        (
            RecipeAction("FORM_BOND", ("*1", "*2"), 1),
            RecipeAction("CHANGE_RADICAL", ("*1",), -1),
            RecipeAction("CHANGE_RADICAL", ("*2",), -1),
        ),
        source,
        True,
    )
    scission = ReactionFamily(
        "beta_scission",
        (
            MolecularPattern(
                (
                    AtomPattern("*1", radicals=(1,), charges=(0,)),
                    AtomPattern("*2", radicals=(0,), charges=(0,)),
                    AtomPattern("*3", radicals=(0,), charges=(0,)),
                ),
                (BondPattern("*1", "*2", (1,)), BondPattern("*2", "*3", (1,))),
            ),
        ),
        (
            RecipeAction("CHANGE_BOND", ("*1", "*2"), 1),
            RecipeAction("BREAK_BOND", ("*2", "*3"), 1),
            RecipeAction("CHANGE_RADICAL", ("*1",), -1),
            RecipeAction("CHANGE_RADICAL", ("*3",), 1),
        ),
        source,
    )
    return abstraction, recombination, scission


def extended_radical_reaction_families() -> tuple[ReactionFamily, ...]:
    """Authored neutral C/H graph families; no rate or stereochemical inference.

    1,4 / 1,5 H migration means 3 / 4 bonds between the radical carbon and
    H-donor carbon, i.e. 5 / 6 atoms in the path including migrating hydrogen.
    5-exo closes C1-C5 of C1-C2-C3-C4-C5=C6 and leaves the radical at C6.
    """
    source = DataSourceProvenance(
        "chemworld-radical-topology-recipes",
        0,
        "Authored explicit radical graph edits and original-path constraints; no kinetic data",
        version="1",
        license="MIT",
    )

    def carbon(label: str, radical: int = 0) -> AtomPattern:
        return AtomPattern(
            label, elements=("C",), charges=(0,), radicals=(radical,), lone_pairs=(0,)
        )

    addition = ReactionFamily(
        "radical_addition_to_alkene",
        (
            MolecularPattern(
                (
                    AtomPattern(
                        "*R", elements=("H", "C"), charges=(0,), radicals=(1,), lone_pairs=(0,)
                    ),
                )
            ),
            MolecularPattern((carbon("*1"), carbon("*2")), (BondPattern("*1", "*2", (2,)),)),
        ),
        (
            RecipeAction("FORM_BOND", ("*R", "*1"), 1),
            RecipeAction("CHANGE_BOND", ("*1", "*2"), -1),
            RecipeAction("CHANGE_RADICAL", ("*R",), -1),
            RecipeAction("CHANGE_RADICAL", ("*2",), 1),
        ),
        source,
    )
    migrations = []
    for position in (4, 5):
        labels = tuple(f"*{i}" for i in range(1, position + 1))
        donor = labels[-1]
        pattern = MolecularPattern(
            (
                *tuple(carbon(k, int(i == 0)) for i, k in enumerate(labels)),
                AtomPattern("*H", elements=("H",), charges=(0,), radicals=(0,), lone_pairs=(0,)),
            ),
            (
                *tuple(BondPattern(a, b, (1,)) for a, b in pairwise(labels)),
                BondPattern(donor, "*H", (1,)),
            ),
        )
        migrations.append(
            ReactionFamily(
                f"H_migration_1_{position}",
                (pattern,),
                (
                    RecipeAction("BREAK_BOND", (donor, "*H"), 1),
                    RecipeAction("FORM_BOND", ("*1", "*H"), 1),
                    RecipeAction("CHANGE_RADICAL", ("*1",), -1),
                    RecipeAction("CHANGE_RADICAL", (donor,), 1),
                ),
                source,
                path_constraints=(ReactionPathConstraint(labels, (1,) * (position - 1)),),
            )
        )
    chain = tuple(f"*{i}" for i in range(1, 6))
    cyclization = ReactionFamily(
        "radical_cyclization_5_exo",
        (
            MolecularPattern(
                tuple(carbon(f"*{i}", int(i == 1)) for i in range(1, 7)),
                (
                    *tuple(BondPattern(a, b, (1,)) for a, b in pairwise(chain)),
                    BondPattern("*5", "*6", (2,)),
                ),
            ),
        ),
        (
            RecipeAction("FORM_BOND", ("*1", "*5"), 1),
            RecipeAction("CHANGE_BOND", ("*5", "*6"), -1),
            RecipeAction("CHANGE_RADICAL", ("*1",), -1),
            RecipeAction("CHANGE_RADICAL", ("*6",), 1),
        ),
        source,
        path_constraints=(
            ReactionPathConstraint(chain, (1, 1, 1, 1), exocyclic_double_bond=("*5", "*6")),
        ),
    )
    return (addition, *migrations, cyclization)
