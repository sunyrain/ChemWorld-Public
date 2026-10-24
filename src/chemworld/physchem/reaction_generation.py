"""One static graph-generation pass and explicit bindings to ReactionNetworkSpec."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations_with_replacement, islice, product
from math import fsum, isfinite
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_graph import (
    MolecularGraph,
    enumerate_allylic_resonance,
    find_pattern_matches,
    molecular_isomorphisms,
)
from chemworld.physchem.reaction_family import ReactionFamily, apply_reaction_recipe
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import RateLawSpec, ReactionSpec, SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import (
    audit_reaction_rate_contract,
    rate_coefficient_unit,
)
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml


@dataclass(frozen=True)
class GenerationPolicy:
    maximum_matches: int = 10000
    maximum_reactions: int = 1000
    maximum_resonance_forms: int = 64
    allylic_resonance: bool = False

    def __post_init__(self) -> None:
        for value in (self.maximum_matches, self.maximum_reactions, self.maximum_resonance_forms):
            if type(value) is not int or value < 1:
                raise ValueError("generation bounds must be positive integers")
        if type(self.allylic_resonance) is not bool:
            raise ValueError("allylic resonance policy must be explicit boolean")


@dataclass(frozen=True)
class GeneratedReaction:
    reaction_id: str
    family_id: str
    reactants: dict[str, int]
    products: dict[str, int]
    paths: tuple[dict[str, Any], ...]
    source: DataSourceProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "family_id": self.family_id,
            "reactants": self.reactants,
            "products": self.products,
            "paths": list(self.paths),
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class GenerationReport:
    species: dict[str, MolecularGraph]
    reactions: tuple[GeneratedReaction, ...]
    seed_aliases: dict[str, str]
    complete: bool
    truncated_reasons: tuple[str, ...]
    raw_matches: int
    rejected: tuple[dict[str, Any], ...]
    policy: GenerationPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-static-reaction-generation-v1",
            "species": {k: g.to_dict() for k, g in self.species.items()},
            "reactions": [r.to_dict() for r in self.reactions],
            "seed_aliases": self.seed_aliases,
            "complete": self.complete,
            "truncated_reasons": list(self.truncated_reasons),
            "raw_matches": self.raw_matches,
            "rejected": list(self.rejected),
            "policy": asdict(self.policy),
        }


def _match(
    left: MolecularGraph, right: MolecularGraph, policy: GenerationPolicy
) -> dict[str, str] | None:
    variants: tuple[MolecularGraph, ...] = (left,)
    if policy.allylic_resonance:
        resonance = enumerate_allylic_resonance(left, maximum_forms=policy.maximum_resonance_forms)
        if not resonance.complete:
            raise ValueError("resonance_identity_search_truncated")
        variants = resonance.forms
    for variant in variants:
        mapping = next(molecular_isomorphisms(variant, right), None)
        if mapping is not None:
            return mapping
    return None


def _path_key(
    family: ReactionFamily, assignment: tuple[str, ...], matches: Sequence[Mapping[str, str]]
) -> tuple:
    labels = {label: (i, atom) for i, match in enumerate(matches) for label, atom in match.items()}

    def signature(swap: bool) -> tuple:
        values = []
        for action in family.actions:
            coordinates = tuple(
                (1 - labels[k][0] if swap else labels[k][0], labels[k][1]) for k in action.labels
            )
            if len(coordinates) == 2:
                coordinates = tuple(sorted(coordinates))
            values.append((action.kind, coordinates, action.value))
        return tuple(sorted(values))

    normal = signature(False)
    return (
        min(normal, signature(True))
        if len(assignment) == 2 and assignment[0] == assignment[1]
        else normal
    )


def generate_reactions(
    reactants: Mapping[str, MolecularGraph],
    families: Sequence[ReactionFamily],
    *,
    policy: GenerationPolicy | None = None,
) -> GenerationReport:
    """Enumerate only the supplied seed pool once; no automatic mechanism growth."""
    if policy is None:
        policy = GenerationPolicy()
    if not reactants or any(not key or any(c.isspace() for c in key) for key in reactants):
        raise ValueError("generation requires named seed graphs")
    families = tuple(families)
    if len({f.family_id for f in families}) != len(families):
        raise ValueError("family identities must be unique")
    registry: dict[str, MolecularGraph] = {}
    aliases: dict[str, str] = {}
    candidates: dict[tuple, GeneratedReaction] = {}
    rejected: list[dict[str, Any]] = []
    truncated: list[str] = []
    raw, next_species = 0, 1

    def register(graph: MolecularGraph, preferred: str | None = None) -> tuple[str, dict[str, str]]:
        nonlocal next_species
        for key, old in registry.items():
            mapping = _match(graph, old, policy)
            if mapping is not None:
                return key, mapping
        key = preferred or f"generated_{next_species:04d}"
        while key in registry:
            next_species += 1
            key = f"generated_{next_species:04d}"
        if preferred is None:
            next_species += 1
        registry[key] = graph
        return key, {a.atom_id: a.atom_id for a in graph.atoms}

    try:
        for key in sorted(reactants):
            aliases[key], _ = register(reactants[key], key)
    except ValueError as exc:
        return GenerationReport(registry, (), aliases, False, (str(exc),), 0, (), policy)
    seeds = tuple(dict.fromkeys(aliases.values()))
    seen_paths = set()
    for family in families:
        assignments = (
            combinations_with_replacement(seeds, 2)
            if family.unordered_reactants
            else product(seeds, repeat=len(family.reactant_patterns))
        )
        for assignment in assignments:
            variants = []
            for key in assignment:
                resonance = (
                    enumerate_allylic_resonance(
                        registry[key], maximum_forms=policy.maximum_resonance_forms
                    )
                    if policy.allylic_resonance
                    else None
                )
                if resonance is not None and not resonance.complete:
                    truncated.append("reactant_resonance_forms")
                variants.append(resonance.forms if resonance else (registry[key],))
            for graphs in product(*variants):
                choices = []
                for graph, pattern in zip(graphs, family.reactant_patterns, strict=True):
                    found = list(
                        islice(find_pattern_matches(graph, pattern), policy.maximum_matches + 1)
                    )
                    if len(found) > policy.maximum_matches:
                        truncated.append("template_matches")
                        found = found[: policy.maximum_matches]
                    choices.append(found)
                for matches in product(*choices):
                    if raw == policy.maximum_matches:
                        truncated.append("total_matches")
                        return GenerationReport(
                            registry,
                            tuple(candidates.values()),
                            aliases,
                            False,
                            tuple(sorted(set(truncated))),
                            raw,
                            tuple(rejected),
                            policy,
                        )
                    raw += 1
                    try:
                        generated = apply_reaction_recipe(family, graphs, matches)
                    except ValueError as exc:
                        rejected.append(
                            {
                                "family_id": family.family_id,
                                "reactants": list(assignment),
                                "matches": list(matches),
                                "reason": str(exc),
                            }
                        )
                        continue
                    products, mappings = [], []
                    try:
                        for i, graph in enumerate(generated.products):
                            key, mapping = register(graph)
                            products.append(key)
                            for atom in graph.atoms:
                                occurrence, original_atom = generated.original_atoms[atom.atom_id]
                                mappings.append(
                                    {
                                        "reactant_index": occurrence,
                                        "reactant_species": assignment[occurrence],
                                        "reactant_atom": original_atom,
                                        "product_index": i,
                                        "product_species": key,
                                        "product_atom": mapping[atom.atom_id],
                                    }
                                )
                    except ValueError as exc:
                        truncated.append(str(exc))
                        continue
                    left, right = dict(Counter(assignment)), dict(Counter(products))
                    if left == right:
                        rejected.append(
                            {
                                "family_id": family.family_id,
                                "reactants": list(assignment),
                                "matches": list(matches),
                                "reason": "identity_exchange_no_net_reaction",
                            }
                        )
                        continue
                    identity = (
                        family.family_id,
                        tuple(sorted(left.items())),
                        tuple(sorted(right.items())),
                    )
                    # The same edits on different resonance forms can lead to
                    # distinct constitutional products. Deduplicate only after
                    # their actual product identities have been established.
                    path_key = (identity, assignment, _path_key(family, assignment, matches))
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    path = {"template_matches": list(matches), "atom_mapping": mappings}
                    if generated.topology_matches:
                        path["topology"] = list(generated.topology_matches)
                    if identity in candidates:
                        old = candidates[identity]
                        candidates[identity] = replace(old, paths=(*old.paths, path))
                    else:
                        if len(candidates) == policy.maximum_reactions:
                            truncated.append("reaction_count")
                            return GenerationReport(
                                registry,
                                tuple(candidates.values()),
                                aliases,
                                False,
                                tuple(sorted(set(truncated))),
                                raw,
                                tuple(rejected),
                                policy,
                            )
                        candidates[identity] = GeneratedReaction(
                            f"reaction_{len(candidates) + 1:04d}",
                            family.family_id,
                            left,
                            right,
                            (path,),
                            family.source,
                        )
    return GenerationReport(
        registry,
        tuple(candidates.values()),
        aliases,
        not truncated,
        tuple(sorted(set(truncated))),
        raw,
        tuple(rejected),
        policy,
    )


@dataclass(frozen=True)
class SpeciesDataBinding:
    graph: MolecularGraph
    species: SpeciesSpec
    thermo_declaration: Mapping[str, Any]
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        if (
            self.species.phase != "gas"
            or self.species.composition != self.graph.composition
            or self.species.charge != self.graph.charge
        ):
            raise ValueError("generated gas species binding disagrees with its explicit graph")
        if not self.source.license or not self.source.version:
            raise ValueError("species parameter source needs version/license")
        if set(self.thermo_declaration) - {
            "thermo",
            "composition",
            "reference_pressure_Pa",
            "units",
        }:
            raise ValueError("unsupported thermochemistry binding fields")
        state = species_thermo_from_cantera_yaml(
            species_id=self.species.species_id, **self.thermo_declaration
        )
        if dict(state.composition) != self.graph.composition:
            raise ValueError("thermochemistry composition disagrees with generated species")


@dataclass(frozen=True)
class KineticsBinding:
    reaction_id: str
    rate_law: RateLawSpec
    pre_exponential_unit: str
    activation_energy_unit: str
    degeneracy_basis: str
    degeneracy_factor: float
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        if (
            not self.reaction_id
            or self.rate_law.equation_id not in ("arrhenius", "modified_arrhenius")
            or self.rate_law.uses_activities
        ):
            raise ValueError(
                "static graph binding currently requires explicit forward concentration Arrhenius"
            )
        if self.activation_energy_unit != "J/mol":
            raise ValueError("activation energy binding must explicitly use J/mol")
        if self.degeneracy_basis not in ("effective_reaction", "per_declared_path"):
            raise ValueError("declare effective-reaction or per-declared-path kinetics")
        if (
            isinstance(self.degeneracy_factor, bool)
            or not isfinite(self.degeneracy_factor)
            or self.degeneracy_factor <= 0
            or (self.degeneracy_basis == "effective_reaction" and self.degeneracy_factor != 1)
        ):
            raise ValueError("degeneracy factor must be explicit; effective rates use one")
        if not self.source.version or not self.source.license:
            raise ValueError("kinetic data need source version and license")


def bind_generated_network(
    report: GenerationReport,
    species_data: Sequence[SpeciesDataBinding],
    kinetics_bindings: Sequence[KineticsBinding],
    *,
    network_id: str,
    reference_temperature_K: float,
) -> ReactionNetworkSpec:
    """Require all data, then emit the existing gas-thermo/network contract.

    Edit-path counts are metadata only. Only the explicitly supplied and sourced
    degeneracy factor changes A; no guessed kinetics or caloric zeros are used.
    """
    if not report.complete:
        raise ValueError("cannot export a truncated static generation report as a complete network")
    if not report.reactions:
        raise ValueError("generated network requires at least one actual net reaction")
    if not isfinite(reference_temperature_K) or reference_temperature_K <= 0:
        raise ValueError("enthalpy reference temperature must be explicit and positive")
    bound, thermos, nodes = {}, {}, {}
    for key, graph in report.species.items():
        matches = [b for b in species_data if _match(graph, b.graph, report.policy) is not None]
        if len(matches) != 1:
            raise ValueError(
                f"species {key} requires exactly one graph-matched identity/thermochemistry binding"
            )
        binding = matches[0]
        bound[key] = replace(
            binding.species,
            metadata={
                **binding.species.metadata,
                "generated_graph": graph.to_dict(),
                "parameter_source": binding.source.to_dict(),
            },
        )
        thermos[key] = species_thermo_from_cantera_yaml(
            species_id=binding.species.species_id, **binding.thermo_declaration
        )
        nodes[binding.species.species_id] = dict(binding.thermo_declaration)
    if len({s.species_id for s in bound.values()}) != len(bound):
        raise ValueError("distinct generated graph species cannot share one network identity")
    if len({t.reference_pressure_Pa for t in thermos.values()}) != 1:
        raise ValueError("species thermochemistry reference pressures must agree")
    bindings = {b.reaction_id: b for b in kinetics_bindings}
    if len(bindings) != len(kinetics_bindings) or set(bindings) != {
        r.reaction_id for r in report.reactions
    }:
        raise ValueError("every generated reaction needs exactly one kinetic binding")
    reactions = []
    for generated in report.reactions:
        kinetic = bindings[generated.reaction_id]
        order = sum(generated.reactants.values())
        params = dict(kinetic.rate_law.parameters)
        if not {"A", "b", "Ea"} <= params.keys():
            raise ValueError("kinetic bindings must explicitly provide A, b and Ea")
        expected = rate_coefficient_unit(order, temperature_exponent=float(params["b"]))
        if kinetic.pre_exponential_unit != expected:
            raise ValueError(
                "Arrhenius pre-exponential units disagree with generated reaction order"
            )
        params["A"] = float(params["A"]) * kinetic.degeneracy_factor
        stoich = {
            bound[k].species_id: float(generated.products.get(k, 0) - generated.reactants.get(k, 0))
            for k in generated.reactants.keys() | generated.products.keys()
            if generated.products.get(k, 0) != generated.reactants.get(k, 0)
        }

        def side(values: Mapping[str, int]) -> str:
            return " + ".join(
                (f"{v} " if v != 1 else "") + bound[k].species_id for k, v in sorted(values.items())
            )

        enthalpy = fsum(
            (generated.products.get(k, 0) - generated.reactants.get(k, 0))
            * thermos[k].evaluate(reference_temperature_K).enthalpy_J_mol
            for k in bound
        )
        native_paths = [
            {
                **path,
                "atom_mapping": [
                    {
                        **row,
                        "reactant_species": bound[row["reactant_species"]].species_id,
                        "product_species": bound[row["product_species"]].species_id,
                    }
                    for row in path["atom_mapping"]
                ],
            }
            for path in generated.paths
        ]
        reaction = ReactionSpec(
            generated.reaction_id,
            side(generated.reactants) + " => " + side(generated.products),
            stoich,
            False,
            replace(kinetic.rate_law, parameters=params),
            enthalpy,
            metadata={
                "family_id": generated.family_id,
                "family_source": generated.source.to_dict(),
                "kinetics_source": kinetic.source.to_dict(),
                "degeneracy_basis": kinetic.degeneracy_basis,
                "declared_degeneracy_factor": kinetic.degeneracy_factor,
                "observed_edit_paths": len(generated.paths),
                "atom_paths": native_paths,
                "enthalpy_reference_temperature_K": reference_temperature_K,
            },
            forward_orders={bound[k].species_id: float(v) for k, v in generated.reactants.items()},
        )
        audit = audit_reaction_rate_contract(reaction)
        if not audit.passed:
            raise ValueError("generated rate contract: " + "; ".join(audit.violations))
        reactions.append(reaction)
    return ReactionNetworkSpec(
        network_id,
        tuple(bound.values()),
        tuple(reactions),
        metadata={
            "generation_mode": "single_static_seed_pass",
            "generation_complete": True,
            "gas_thermochemistry": nodes,
            "seed_aliases": report.seed_aliases,
            "generation_source_species": {k: s.species_id for k, s in bound.items()},
            "rejected_generation_paths": list(report.rejected),
        },
    )
