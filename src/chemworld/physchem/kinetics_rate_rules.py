"""Source-declared exact kinetics and bounded ancestral rate-rule selection.

No upstream database, implicit parameter average or edit-path degeneracy is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from math import isfinite
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_graph import MolecularGraph, molecular_isomorphic
from chemworld.physchem.reaction_generation import GeneratedReaction, KineticsBinding
from chemworld.physchem.reaction_network_specs import RateLawSpec
from chemworld.physchem.reaction_rate_contracts import rate_coefficient_unit
from chemworld.physchem.thermo_group_additivity import (
    RootedGroupNode,
    match_rooted_group,
    rooted_group_ancestors,
)


@dataclass(frozen=True)
class DeclaredArrhenius:
    A: float
    b: float
    Ea_J_mol: float
    pre_exponential_unit: str
    temperature_bounds_K: tuple[float, float]
    degeneracy_basis: str
    degeneracy_factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_bounds_K", tuple(self.temperature_bounds_K))
        if (
            len(self.temperature_bounds_K) != 2
            or not 0 < self.temperature_bounds_K[0] < self.temperature_bounds_K[1]
            or self.A <= 0
            or self.degeneracy_factor <= 0
            or not self.pre_exponential_unit
            or self.degeneracy_basis not in {"effective_reaction", "per_declared_path"}
            or (self.degeneracy_basis == "effective_reaction" and self.degeneracy_factor != 1)
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (
                    self.A,
                    self.b,
                    self.Ea_J_mol,
                    self.degeneracy_factor,
                    *self.temperature_bounds_K,
                )
            )
        ):
            raise ValueError(
                "kinetics requires finite explicit A/b/Ea, units, validity and degeneracy"
            )


@dataclass(frozen=True)
class KineticRoleTree:
    family_label: str
    nodes: tuple[RootedGroupNode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if not self.family_label:
            raise ValueError("kinetic role needs the original family atom label")
        rooted_group_ancestors(self.nodes)


@dataclass(frozen=True)
class KineticRuleEntry:
    entry_id: str
    family_id: str
    node_ids: tuple[str, ...]
    parameters: DeclaredArrhenius
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        if (
            not self.entry_id
            or not self.family_id
            or not self.source.version
            or not self.source.license
        ):
            raise ValueError("rate rule needs identity, family and versioned licensed source")


@dataclass(frozen=True)
class KineticTrainingEntry:
    """Exact reaction graph multiset, direction and effective/per-path basis retained."""

    entry_id: str
    family_id: str
    reactants: tuple[MolecularGraph, ...]
    products: tuple[MolecularGraph, ...]
    parameters: DeclaredArrhenius
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        for name in ("reactants", "products"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not self.entry_id
            or not self.family_id
            or not self.reactants
            or not self.products
            or not self.source.version
            or not self.source.license
        ):
            raise ValueError("exact kinetic entry needs both graph sides and a licensed source")


@dataclass(frozen=True)
class KineticEstimationLibrary:
    family_trees: Mapping[str, tuple[KineticRoleTree, ...]]
    rules: tuple[KineticRuleEntry, ...]
    training: tuple[KineticTrainingEntry, ...]
    library: tuple[KineticTrainingEntry, ...]

    def __post_init__(self) -> None:
        for name in ("rules", "training", "library"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for entries in (self.rules, self.training, self.library):
            if len({e.entry_id for e in entries}) != len(entries):
                raise ValueError("kinetic entry identities must be unique within their tier")
        for family, roles in self.family_trees.items():
            if not family or not roles or len({r.family_label for r in roles}) != len(roles):
                raise ValueError("family roles must be nonempty and unique")
        for entry in self.rules:
            roles = self.family_trees.get(entry.family_id, ())
            if len(roles) != len(entry.node_ids) or any(
                key not in {n.node_id for n in role.nodes}
                for key, role in zip(entry.node_ids, roles, strict=True)
            ):
                raise ValueError("rate-rule tuple references unknown family roles/nodes")


@dataclass(frozen=True)
class KineticEstimate:
    binding: KineticsBinding | None
    trace: dict[str, Any]
    failure_reason: str = ""

    @property
    def success(self) -> bool:
        return self.binding is not None


def _same_side(left: Sequence[MolecularGraph], right: Sequence[MolecularGraph]) -> bool:
    return len(left) == len(right) and any(
        all(molecular_isomorphic(a, b) for a, b in zip(left, order, strict=True))
        for order in permutations(right)
    )


def _exact(
    entries: Sequence[KineticTrainingEntry],
    reaction: GeneratedReaction,
    species: Mapping[str, MolecularGraph],
) -> list[KineticTrainingEntry]:
    sides = [
        [species[k] for k, n in counts.items() for _ in range(n)]
        for counts in (reaction.reactants, reaction.products)
    ]
    return [
        e
        for e in entries
        if e.family_id == reaction.family_id
        and _same_side(e.reactants, sides[0])
        and _same_side(e.products, sides[1])
    ]


def _unique(entries: Sequence[Any]) -> Any:
    priority = min(e.source.priority for e in entries)
    choices = [e for e in entries if e.source.priority == priority]
    if len(choices) != 1:
        raise ValueError("ambiguous kinetic entries at equal priority; no implicit averaging")
    return choices[0]


def estimate_reaction_kinetics(
    reaction: GeneratedReaction,
    species: Mapping[str, MolecularGraph],
    library: KineticEstimationLibrary,
) -> KineticEstimate:
    trace: dict[str, Any] = {
        "family_id": reaction.family_id,
        "observed_edit_paths": len(reaction.paths),
    }
    try:
        selected, distance, tier = None, 0, ""
        for kind, entries in (("library", library.library), ("training", library.training)):
            choices = _exact(entries, reaction, species)
            if choices:
                selected, tier = _unique(choices), kind
                break
        if selected is None:
            roles = library.family_trees.get(reaction.family_id)
            if not roles:
                raise ValueError("missing kinetic family role tree")
            selected_paths = []
            path_traces = []
            for path in reaction.paths:
                lineages, targets = [], []
                for role in roles:
                    occurrences = [
                        (i, m[role.family_label])
                        for i, m in enumerate(path["template_matches"])
                        if role.family_label in m
                    ]
                    if len(occurrences) != 1:
                        raise ValueError("kinetic role cannot bind an original reaction atom")
                    index, atom = occurrences[0]
                    names = {
                        row["reactant_species"]
                        for row in path["atom_mapping"]
                        if row["reactant_index"] == index
                    }
                    if len(names) != 1:
                        raise ValueError("kinetic role has ambiguous reactant occurrence")
                    graph = species[next(iter(names))]
                    node = match_rooted_group(graph, atom, role.nodes)
                    targets.append(node.node_id)
                    lineages.append(rooted_group_ancestors(role.nodes)[node.node_id])
                scored = [
                    (
                        sum(
                            lineage.index(key)
                            for key, lineage in zip(e.node_ids, lineages, strict=True)
                        ),
                        e,
                    )
                    for e in library.rules
                    if e.family_id == reaction.family_id
                    and all(
                        key in lineage for key, lineage in zip(e.node_ids, lineages, strict=True)
                    )
                ]
                if not scored:
                    raise ValueError("missing kinetic rate rule and ancestral fallback")
                best = min(d for d, _ in scored)
                entry = _unique([e for d, e in scored if d == best])
                selected_paths.append((entry, best))
                path_traces.append(
                    {"target_nodes": targets, "entry_id": entry.entry_id, "ancestor_distance": best}
                )
            if not selected_paths:
                raise ValueError("reaction has no declared atom paths")
            if len({(e.entry_id, d) for e, d in selected_paths}) != 1:
                raise ValueError("different atom paths select incompatible kinetic rules")
            selected, distance = selected_paths[0]
            tier = "rule" if distance == 0 else "ancestor"
            trace["path_selections"] = path_traces
        params = selected.parameters
        order = sum(reaction.reactants.values())
        if params.pre_exponential_unit != rate_coefficient_unit(
            order, temperature_exponent=params.b
        ):
            raise ValueError("kinetic parameter units disagree with generated reaction order")
        trace.update(
            tier=tier,
            entry_id=selected.entry_id,
            ancestor_distance=distance,
            source=selected.source.to_dict(),
            degeneracy_basis=params.degeneracy_basis,
            degeneracy_factor=params.degeneracy_factor,
            temperature_bounds_K=list(params.temperature_bounds_K),
        )
        law = RateLawSpec(
            reaction.reaction_id + ":estimated-rate",
            "modified_arrhenius",
            {
                "A": params.A,
                "b": params.b,
                "Ea": params.Ea_J_mol,
                "valid_temperature_range_K": list(params.temperature_bounds_K),
                "parameter_estimation": trace,
            },
        )
        binding = KineticsBinding(
            reaction.reaction_id,
            law,
            params.pre_exponential_unit,
            "J/mol",
            params.degeneracy_basis,
            params.degeneracy_factor,
            selected.source,
        )
        return KineticEstimate(binding, trace)
    except ValueError as exc:
        return KineticEstimate(None, trace, str(exc))
