"""Bind only reactions actually discovered in a complete native graph pass."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from chemworld.physchem.mechanism_growth_specs import (
    CoreEdgeGrowthPolicy,
    MechanismGrowthLibrary,
    reaction_library_key,
)
from chemworld.physchem.molecular_graph import molecular_isomorphic
from chemworld.physchem.reaction_family import ReactionFamily
from chemworld.physchem.reaction_generation import (
    GenerationReport,
    bind_generated_network,
    generate_reactions,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import ReactionSpec


def enumerate_bound_candidates(
    core: Sequence[str],
    library: MechanismGrowthLibrary,
    families: Sequence[ReactionFamily],
    policy: CoreEdgeGrowthPolicy,
) -> tuple[GenerationReport, dict[str, ReactionSpec], list[dict[str, Any]]]:
    pool = library.species_by_id
    report = generate_reactions(
        {key: pool[key].graph for key in sorted(core)}, families, policy=policy.generation
    )
    if not report.complete:
        return report, {}, []
    names: dict[str, str] = {}
    missing: list[dict[str, Any]] = []
    for key, graph in report.species.items():
        matches = [
            name for name, binding in pool.items() if molecular_isomorphic(graph, binding.graph)
        ]
        if len(matches) != 1:
            missing.append(
                {
                    "kind": "species_binding",
                    "generated_id": key,
                    "graph": graph.to_dict(),
                    "matches": matches,
                }
            )
        else:
            names[key] = matches[0]
    rates = {entry.key: entry for entry in library.reactions}
    discovered: dict[str, ReactionSpec] = {}
    for candidate in report.reactions:
        used = candidate.reactants.keys() | candidate.products.keys()
        if used - names.keys():
            missing.append(
                {
                    "kind": "reaction_species_binding",
                    "candidate": candidate.to_dict(),
                    "missing_generated_ids": sorted(used - names.keys()),
                }
            )
            continue
        identity = reaction_library_key(
            candidate.family_id,
            {names[k]: n for k, n in candidate.reactants.items()},
            {names[k]: n for k, n in candidate.products.items()},
        )
        if identity not in rates:
            missing.append(
                {"kind": "kinetics_binding", "key": identity, "candidate": candidate.to_dict()}
            )
            continue
        # This is an explicitly selected reaction from an already complete pass,
        # not a fabricated completeness flag for a truncated enumeration.
        selected = replace(
            report,
            species={k: report.species[k] for k in sorted(used)},
            reactions=(candidate,),
            seed_aliases={k: v for k, v in report.seed_aliases.items() if v in used},
        )
        kinetics = replace(rates[identity].kinetics, reaction_id=candidate.reaction_id)
        try:
            bound = bind_generated_network(
                selected,
                library.species,
                (kinetics,),
                network_id="growth-candidate",
                reference_temperature_K=policy.reference_temperature_K,
            ).reactions[0]
        except (ValueError, KeyError, OverflowError) as error:
            missing.append(
                {"kind": "invalid_parameter_binding", "key": identity, "message": str(error)}
            )
            continue
        discovered[identity] = replace(
            bound,
            reaction_id=identity,
            rate_law=replace(bound.rate_law, rate_law_id=identity + ":rate"),
            metadata={**bound.metadata, "growth_library_key": identity},
        )
    return report, discovered, missing


def make_growth_network(
    names: Sequence[str],
    reactions: Sequence[ReactionSpec],
    library: MechanismGrowthLibrary,
    *,
    network_id: str,
) -> ReactionNetworkSpec:
    """Use original identities/thermo, including a legitimate zero-reaction core."""
    pool = library.species_by_id
    selected = [pool[key] for key in sorted(names)]
    return ReactionNetworkSpec(
        network_id,
        tuple(
            replace(
                b.species,
                metadata={
                    **b.species.metadata,
                    "generated_graph": b.graph.to_dict(),
                    "parameter_source": b.source.to_dict(),
                },
            )
            for b in selected
        ),
        tuple(sorted(reactions, key=lambda r: r.reaction_id)),
        metadata={
            "gas_thermochemistry": {
                b.species.species_id: dict(b.thermo_declaration) for b in selected
            },
            "generation_mode": "finite-library-core-edge",
        },
    )
