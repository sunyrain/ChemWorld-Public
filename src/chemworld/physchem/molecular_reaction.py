"""Same-source elementary chemistry in an explicit homogeneous PR phase.

An explicit molecular thermal binding selects this process. The standalone
imported ideal mechanism, its rate constants, and its source thermo stay intact.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import exp
from typing import Any, cast

from chemworld.physchem.coupled_system import CoupledProcess, CoupledStateView, ProcessContribution
from chemworld.physchem.cpa import CPAModel
from chemworld.physchem.mechanism_ideal_reference import MechanismIdealReference
from chemworld.physchem.mechanism_rates import MECHANISM_SCHEMA, evaluate_mechanism_rate
from chemworld.physchem.molecular_eos import ResidualHelmholtzModel
from chemworld.physchem.molecular_mixture import (
    MolecularCaloricModel,
    MolecularIsochoricThermodynamics,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec


def mechanism_caloric_model(
    network: ReactionNetworkSpec, residual_model: ResidualHelmholtzModel
) -> MolecularCaloricModel:
    """Bind each original source declaration in the explicit EOS component order."""
    declarations = cast(
        Mapping[str, dict[str, Any]], network.metadata.get("gas_thermochemistry", {})
    )
    if set(residual_model.component_ids) != set(network.species_ids):
        raise ValueError("molecular phase requires the complete mechanism species inventory")
    if set(declarations) != set(network.species_ids):
        raise ValueError("complete imported gas thermochemistry is required")
    return MolecularCaloricModel(
        residual_model,
        tuple(
            MechanismIdealReference(key, declarations[key]) for key in residual_model.component_ids
        ),
    )


def _validate_binding(
    network: ReactionNetworkSpec, thermodynamics: MolecularIsochoricThermodynamics
) -> None:
    model = thermodynamics.model
    residual = model.residual_model
    if (
        type(residual) is not CPAModel
        or residual.cubic_model != "peng_robinson"
        or residual.association is not None
        or residual.ge_mixing is not None
    ):
        raise ValueError(
            "reacting molecular phase requires explicit nonassociating PR quadratic mixing"
        )
    if set(network.species_ids) != set(model.component_ids):
        raise ValueError("reacting molecular phase requires the complete network inventory")
    declarations = cast(
        Mapping[str, dict[str, Any]], network.metadata.get("gas_thermochemistry", {})
    )
    if set(declarations) != set(network.species_ids):
        raise ValueError("reacting molecular phase requires complete source thermochemistry")
    for ref in model.ideal_references:
        if type(ref) is not MechanismIdealReference or not ref.matches_source(
            declarations[ref.fluid_id]
        ):
            raise ValueError(
                "molecular ideal references must match original mechanism source exactly"
            )
    for species in network.species:
        if species.phase != "gas" or species.charge or species.catalyst:
            raise ValueError("reacting molecular phase requires neutral homogeneous gas species")
        if species.composition != declarations[species.species_id]["composition"]:
            raise ValueError("mechanism species and caloric source composition must agree")
    for reaction in network.reactions:
        p = reaction.rate_law.parameters
        if (
            reaction.rate_law.equation_id != "arrhenius"
            or p.get("mechanism_schema") != MECHANISM_SCHEMA
            or reaction.rate_law.uses_activities
            or float(p["A"]) < 0
            or reaction.kinetic_forward_orders != reaction.reactants
            or reaction.kinetic_reverse_orders != reaction.products
        ):
            raise ValueError(
                "reacting molecular phase supports elementary positive Arrhenius rates only"
            )
        if reaction.reversible and p.get("reverse_basis") != "ideal_gas_thermo":
            raise ValueError(
                "molecular reverse kinetics require original standard-G thermochemistry"
            )


def _rates(
    network: ReactionNetworkSpec,
    thermodynamics: MolecularIsochoricThermodynamics,
    state: CoupledStateView,
) -> dict[str, float]:
    molar = thermodynamics.molar_state(state)
    if molar.log_fugacity_coefficients is None:
        raise ValueError("reacting molecular phase requires positive pressure")
    # phi is relative to ideal at the same T,P; multiplying by Z converts it
    # to the reference at the same T,density used by concentration mass action.
    activity_concentrations = {
        key: state.amounts_mol[key] / state.volume_L * molar.compressibility_factor * exp(lnphi)
        for key, lnphi in zip(
            thermodynamics.inventory_keys, molar.log_fugacity_coefficients, strict=True
        )
    }
    return {
        reaction.reaction_id: evaluate_mechanism_rate(
            reaction,
            concentrations_mol_L=activity_concentrations,
            temperature_K=state.temperature_K,
            species_thermo=network.imported_gas_thermochemistry,
        )
        for reaction in network.reactions
    }


def molecular_reaction_rates(
    network: ReactionNetworkSpec,
    thermodynamics: MolecularIsochoricThermodynamics,
    state: CoupledStateView,
) -> dict[str, float]:
    """Net progress in mol/(L s), without changing or charging the inventory."""
    _validate_binding(network, thermodynamics)
    return _rates(network, thermodynamics, state)


def molecular_reaction_process(
    network: ReactionNetworkSpec,
    thermodynamics: MolecularIsochoricThermodynamics,
    *,
    process_id: str | None = None,
) -> CoupledProcess:
    _validate_binding(network, thermodynamics)

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        rates = _rates(network, thermodynamics, state)
        return ProcessContribution(
            amount_rates_mol_s=network.amount_derivatives_from_rates(
                rates, volume_L=state.volume_L
            ),
            reaction_progress_mol_s={
                f"{network.network_id}:{key}": value * state.volume_L
                for key, value in rates.items()
            },
        )

    return CoupledProcess(process_id or network.network_id, evaluate)
