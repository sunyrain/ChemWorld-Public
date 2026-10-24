"""Shared homogeneous molecular tank closure; no inventory or clock ownership."""

from collections.abc import Mapping
from math import isfinite

from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularCaloricState
from chemworld.physchem.molecular_processes import MolecularProcessDomain, molecular_process_state
from chemworld.physchem.residual_system import RootPolicy


def validate_molecular_amounts(model: MolecularCaloricModel, amounts: Mapping[str, float]) -> float:
    if set(amounts) != set(model.component_ids) or any(
        isinstance(x, bool) or not isfinite(x) or x <= 0 for x in amounts.values()
    ):
        raise ValueError("molecular tanks require positive declared binary inventories")
    return sum(amounts.values())


def molecular_tank_tp(
    domain: MolecularProcessDomain,
    pressure_Pa: float,
    amounts: Mapping[str, float],
    temperature: float,
) -> MolecularCaloricState:
    total = validate_molecular_amounts(domain.model, amounts)
    if not domain.temperature_range_K[0] <= temperature <= domain.temperature_range_K[1]:
        raise ValueError("tank temperature left its declared homogeneous domain")
    return domain.model.state_tp(
        temperature,
        pressure_Pa,
        tuple(amounts[k] / total for k in domain.model.component_ids),
        initial_density_mol_m3=domain.density_basin.initial_density_mol_m3,
        density_bounds_mol_m3=domain.density_basin.density_bounds_mol_m3,
    )


def molecular_tank_ph(
    domain: MolecularProcessDomain,
    pressure_Pa: float,
    amounts: Mapping[str, float],
    enthalpy: float,
    policy: RootPolicy,
) -> MolecularCaloricState:
    total = validate_molecular_amounts(domain.model, amounts)
    initial_temperature = sum(domain.temperature_range_K) / 2
    initial = molecular_tank_tp(domain, pressure_Pa, amounts, initial_temperature)
    result = molecular_process_state(
        domain,
        pressure_Pa=pressure_Pa,
        coordinate="H",
        value=enthalpy / total,
        mole_fractions=tuple(amounts[k] / total for k in domain.model.component_ids),
        initial_temperature_K=initial_temperature,
        initial_density_mol_m3=initial.molar_density_mol_m3,
        policy=policy,
    )
    if not result.success or result.state is None:
        raise ValueError("tank PH closure failed without phase switching: " + result.failure_reason)
    return result.state
