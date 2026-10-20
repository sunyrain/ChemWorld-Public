"""Mechanism compilation facade for the ChemWorld transactional runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256

from chemworld.physchem.mechanism_library import (
    MechanismScenarioCard,
    get_mechanism_card,
    list_mechanism_cards,
    load_library_mechanism,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.runtime.mechanism_manifest import (
    CompiledMechanism,
    MechanismManifest,
    MechanismValidationReport,
    ScoreSpec,
)
from chemworld.runtime.mechanism_validation import (
    build_observable_mapping,
    initial_limiting_species,
    mechanism_hash,
    validate_compiled_role_contract,
    validate_mechanism_file,
)
from chemworld.schemas import MECHANISM_SCHEMA_VERSION

SCENARIO_MECHANISM_DEFAULTS = {
    "reaction-optimization": "simple_batch_reaction",
    "reaction-safety": "catalyst_deactivation",
    "reaction-mechanism": "autocatalytic_reaction",
    "reaction-to-assay": "simple_batch_reaction",
    "reaction-to-purification": "reaction_extraction",
    "reaction-to-crystallization": "simple_batch_reaction",
    "reaction-to-distillation": "reactive_distillation_lite",
    "flow-reaction-optimization": "pfr_hotspot",
    "electrochemical-conversion": "electrochemical_conversion",
    "equilibrium-characterization": "simple_batch_reaction",
    "partition-discovery": "reaction_extraction",
    "purity-yield-tradeoff": "reaction_extraction",
    "generalization": "parallel_series_reaction",
    "low-budget-characterization": "autocatalytic_reaction",
    "tool-agent-planning": "reaction_extraction",
}


def compile_mechanism_for_scenario(scenario_id: str) -> CompiledMechanism:
    mechanism_id = mechanism_id_for_scenario(scenario_id)
    return compile_mechanism(mechanism_id, require_runtime_roles=True)


def mechanism_id_for_scenario(scenario_id: str) -> str:
    if scenario_id in SCENARIO_MECHANISM_DEFAULTS:
        return SCENARIO_MECHANISM_DEFAULTS[scenario_id]
    for card in list_mechanism_cards():
        if card.scenario_id == scenario_id:
            return card.mechanism_id
    return "simple_batch_reaction"


def compile_mechanism(
    card_or_mechanism_id: str | MechanismScenarioCard,
    *,
    require_runtime_roles: bool | None = None,
) -> CompiledMechanism:
    card = (
        card_or_mechanism_id
        if isinstance(card_or_mechanism_id, MechanismScenarioCard)
        else get_mechanism_card(card_or_mechanism_id)
    )
    validation_report = validate_mechanism_file(card.resolved_mechanism_path)
    network = load_library_mechanism(card)
    species_roles = {
        species.species_id: tuple(species.observable_aliases) for species in network.species
    }
    observable_mapping = build_observable_mapping(network, card)
    limiting_species = initial_limiting_species(card)
    validate_compiled_role_contract(
        network,
        card,
        observable_mapping=observable_mapping,
        initial_limiting_species=limiting_species,
        require_runtime_roles=(
            card.scenario_id in SCENARIO_MECHANISM_DEFAULTS
            if require_runtime_roles is None
            else require_runtime_roles
        ),
    )
    score_spec = ScoreSpec(
        target_species=card.target_species,
        impurity_species=card.impurity_species,
        initial_limiting_species=limiting_species,
    )
    return _assemble_compiled_network(
        network,
        species_roles,
        observable_mapping,
        score_spec,
        card.initial_amounts_mol,
        validation_report,
    )


def compile_native_network(
    network: ReactionNetworkSpec, *, initial_amounts: Mapping[str, float]
) -> CompiledMechanism:
    """Compile an explicitly authored network, including a nonreacting inventory.

    No scenario lookup or benchmark role defaults apply at this boundary. Roles
    are the aliases declared by the network; absent roles remain absent.
    """
    roles = {s.species_id: tuple(s.observable_aliases) for s in network.species}
    observable = {
        role: tuple(name for name, aliases in roles.items() if role in aliases)
        for role in sorted({r for aliases in roles.values() for r in aliases})
    }
    reactants = observable.get("reactant", ())
    score = ScoreSpec(
        observable.get("target", ()),
        observable.get("impurity", ()),
        reactants[0] if reactants else None,
    )
    payload = {
        "network_id": network.network_id,
        **({"stochastic": network.stochastic} if network.stochastic is not None else {}),
        "species": [s.to_dict() for s in network.species],
        "reactions": [r.to_dict() for r in network.reactions],
        "units": network.units,
        "metadata": network.metadata,
    }
    digest = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    equations = tuple(sorted({r.rate_law.equation_id for r in network.reactions}))
    report = MechanismValidationReport(
        network.network_id,
        MECHANISM_SCHEMA_VERSION,
        digest,
        "world-composition",
        True,
        (),
        (),
        len(network.species),
        len(network.reactions),
        equations,
    )
    return _assemble_compiled_network(network, roles, observable, score, initial_amounts, report)


def _assemble_compiled_network(
    network,
    species_roles,
    observable_mapping,
    score_spec,
    initial_amounts,
    validation_report,
) -> CompiledMechanism:
    manifest = MechanismManifest(
        mechanism_id=network.network_id,
        mechanism_version=MECHANISM_SCHEMA_VERSION,
        mechanism_hash=validation_report.mechanism_hash,
        source_path=validation_report.source_path,
        species_count=len(network.species),
        reaction_count=len(network.reactions),
        rate_law_equation_ids=validation_report.rate_law_equation_ids,
        species_roles=species_roles,
        observable_mapping=observable_mapping,
        score_spec=score_spec,
        initial_amount_policy=dict(initial_amounts),
        validation_report=validation_report,
    )
    return CompiledMechanism(
        mechanism_id=network.network_id,
        mechanism_version=MECHANISM_SCHEMA_VERSION,
        mechanism_hash=validation_report.mechanism_hash,
        network=network,
        species_index=network.species_index,
        stoichiometric_matrix=network.stoichiometric_matrix(),
        reaction_enthalpies={
            reaction.reaction_id: reaction.delta_h_J_per_mol for reaction in network.reactions
        },
        species_roles=species_roles,
        observable_mapping=observable_mapping,
        score_spec=score_spec,
        initial_amount_policy=dict(initial_amounts),
        manifest=manifest,
    )


__all__ = [
    "MECHANISM_SCHEMA_VERSION",
    "SCENARIO_MECHANISM_DEFAULTS",
    "CompiledMechanism",
    "MechanismManifest",
    "MechanismValidationReport",
    "ScoreSpec",
    "compile_mechanism",
    "compile_mechanism_for_scenario",
    "compile_native_network",
    "mechanism_hash",
    "mechanism_id_for_scenario",
    "validate_mechanism_file",
]
