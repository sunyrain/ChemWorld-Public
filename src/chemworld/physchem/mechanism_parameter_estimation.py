"""Prepare complete finite parameter libraries for the original mechanism consumers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from chemworld.physchem.kinetics_rate_rules import (
    KineticEstimationLibrary,
    estimate_reaction_kinetics,
)
from chemworld.physchem.mechanism_growth_specs import MechanismGrowthLibrary, ReactionLibraryEntry
from chemworld.physchem.molecular_graph import MolecularGraph
from chemworld.physchem.reaction_family import ReactionFamily
from chemworld.physchem.reaction_generation import (
    GenerationPolicy,
    GenerationReport,
    bind_generated_network,
    generate_reactions,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.thermo_group_additivity import (
    GroupAdditivityLibrary,
    estimate_group_thermochemistry,
)


@dataclass(frozen=True)
class ParameterPreparationResult:
    generation: GenerationReport
    network: ReactionNetworkSpec | None
    growth_library: MechanismGrowthLibrary | None
    thermo_traces: dict[str, dict[str, Any]]
    kinetic_traces: dict[str, dict[str, Any]]
    reasons: tuple[dict[str, Any], ...]
    generation_rounds: int

    @property
    def complete(self) -> bool:
        return self.network is not None and self.growth_library is not None and not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "native-parameter-preparation-v1",
            "complete": self.complete,
            "generation": self.generation.to_dict(),
            "network": self.network.to_dict() if self.network else None,
            "growth_library": self.growth_library.to_dict() if self.growth_library else None,
            "thermo_traces": self.thermo_traces,
            "kinetic_traces": self.kinetic_traces,
            "reasons": list(self.reasons),
            "generation_rounds": self.generation_rounds,
        }


def prepare_parameterized_mechanism(
    seed_graphs: Mapping[str, MolecularGraph],
    families: Sequence[ReactionFamily],
    thermo_library: GroupAdditivityLibrary,
    kinetic_library: KineticEstimationLibrary,
    *,
    network_id: str,
    reference_temperature_K: float,
    generation_policy: GenerationPolicy | None = None,
    maximum_rounds: int = 4,
    maximum_species: int = 64,
) -> ParameterPreparationResult:
    """Generate a finite closure and estimate its data, then use the original binder.

    Preparation never promotes core species, integrates material or replaces a failed
    estimate. A bound only produces a readable incomplete candidate, never completeness.
    """
    if any(type(v) is not int or v < 1 for v in (maximum_rounds, maximum_species)):
        raise ValueError("preparation requires positive finite generation budgets")
    registry = dict(seed_graphs)
    reasons: list[dict[str, Any]] = []
    report = None
    for round_index in range(1, maximum_rounds + 1):  # noqa: B007 - retained in result after loop
        report = generate_reactions(registry, families, policy=generation_policy)
        if not report.complete:
            reasons.append(
                {"kind": "generation_truncated", "detail": list(report.truncated_reasons)}
            )
            break
        if len(report.species) > maximum_species:
            reasons.append({"kind": "species_budget", "observed": len(report.species)})
            break
        if len(report.species) == len(registry):
            break
        registry = report.species
    else:
        reasons.append({"kind": "closure_round_budget", "rounds": maximum_rounds})
    # Positive maximum_rounds above guarantees at least one generation pass.
    report = cast(GenerationReport, report)
    thermo_traces, kinetic_traces = {}, {}
    bindings, rates = [], []
    for key, graph in sorted(report.species.items()):
        estimated = estimate_group_thermochemistry(graph, key, thermo_library)
        thermo_traces[key] = estimated.trace
        if estimated.binding is None:
            reasons.append(
                {"kind": "thermochemistry", "species_id": key, "reason": estimated.failure_reason}
            )
        else:
            bindings.append(estimated.binding)
    for candidate in report.reactions:
        estimated_rate = estimate_reaction_kinetics(candidate, report.species, kinetic_library)
        kinetic_traces[candidate.reaction_id] = estimated_rate.trace
        if estimated_rate.binding is None:
            reasons.append(
                {
                    "kind": "kinetics",
                    "reaction_id": candidate.reaction_id,
                    "reason": estimated_rate.failure_reason,
                }
            )
        else:
            rates.append(estimated_rate.binding)
    network = library = None
    if not reasons:
        try:
            network = bind_generated_network(
                report,
                bindings,
                rates,
                network_id=network_id,
                reference_temperature_K=reference_temperature_K,
            )
            network = replace(
                network,
                metadata={
                    **network.metadata,
                    "parameter_estimation": {"thermo": thermo_traces, "kinetics": kinetic_traces},
                },
            )
            by_reaction = {r.reaction_id: r for r in rates}
            library = MechanismGrowthLibrary(
                tuple(bindings),
                tuple(
                    ReactionLibraryEntry(
                        r.family_id, r.reactants, r.products, by_reaction[r.reaction_id]
                    )
                    for r in report.reactions
                ),
            )
        except ValueError as exc:
            reasons.append({"kind": "binding", "reason": str(exc)})
            network = None
    return ParameterPreparationResult(
        report, network, library, thermo_traces, kinetic_traces, tuple(reasons), round_index
    )
