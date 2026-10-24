"""Declared ideal-gas caloric adapter for the common batch/heat integrator."""

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

from chemworld.physchem.coupled_system import CoupledSystem, HeatFlux
from chemworld.physchem.reaction_network import R_J_PER_MOL_K, ReactionNetworkSpec
from chemworld.physchem.reactor_solvers import reaction_network_process


@dataclass(frozen=True)
class IdealGasThermodynamics:
    """U=sum(n_i*(h_i(T)-RT)); the common solver alone advances heat/time."""

    species: dict
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 298.15

    def __post_init__(self):
        if not isfinite(self.support_heat_capacity_J_K) or self.support_heat_capacity_J_K < 0:
            raise ValueError("support heat capacity must be finite and nonnegative")
        if not isfinite(self.reference_temperature_K) or self.reference_temperature_K <= 0:
            raise ValueError("reference temperature must be finite and positive")

    @property
    def inventory_keys(self):
        return tuple(self.species)

    def heat_capacity_j_k(self, state):
        return self.support_heat_capacity_J_K + sum(
            state.amounts_mol[key]
            * (value.evaluate(state.temperature_K).cp_J_mol_K - R_J_PER_MOL_K)
            for key, value in self.species.items()
        )

    def species_energy_j_mol(self, key, temperature_K):
        return (
            self.species[key].evaluate(temperature_K).enthalpy_J_mol - R_J_PER_MOL_K * temperature_K
        )

    def internal_energy_j(self, state):
        return self.support_heat_capacity_J_K * (
            state.temperature_K - self.reference_temperature_K
        ) + sum(
            amount * self.species_energy_j_mol(key, state.temperature_K)
            for key, amount in state.amounts_mol.items()
        )

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        if any(item.legacy_reaction_energy_W is not None for item in contributions):
            raise ValueError("gas internal energy cannot also consume a separate reaction heat")
        return sum(
            rate * self.species_energy_j_mol(key, state.temperature_K)
            for key, rate in derivatives.items()
        )

    def validate_state(self, state):
        if set(state.amounts_mol) != set(self.species):
            raise ValueError("gas caloric inventory must match the mechanism")
        if self.heat_capacity_j_k(state) <= 0:
            raise ValueError("gas mixture must have positive finite heat capacity")

    def validate_temperature_interval(self, lower_K, upper_K):
        for species in self.species.values():
            if (
                species.min_temperature_K > lower_K
                or species.max_temperature_K < upper_K
            ):
                raise ValueError("thermal domain exceeds gas thermochemistry validity")
            for left, right in pairwise(getattr(species, "segments", ())):
                if left.max_temperature_K != right.min_temperature_K:
                    raise ValueError("gas thermochemistry domain contains a gap")

    def to_dict(self):
        return {
            "model": "ideal-gas-thermochemistry-v1",
            "species": {key: value.to_dict() for key, value in self.species.items()},
            "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
            "reference_temperature_K": self.reference_temperature_K,
        }

    @classmethod
    def from_dict(cls, payload):
        from chemworld.physchem.thermochemistry import species_thermo_from_dict

        values = dict(payload)
        if values.pop("model") not in {"nasa7-ideal-gas-v1", "ideal-gas-thermochemistry-v1"}:
            raise ValueError("unsupported gas caloric model")
        species = {key: species_thermo_from_dict(raw)
                   for key, raw in values.pop("species").items()}
        return cls(species, **values)


# Existing imports and historical declarations keep one implementation.
NASA7GasThermodynamics = IdealGasThermodynamics


def gas_batch_system(network: ReactionNetworkSpec, *, heat_flux=None, policy=None) -> CoupledSystem:
    """Actual constant-volume gas batch: one native inventory and energy advance.

    A caller supplies the existing HeatFlux interface for heating/losses. No
    Cantera import, alternate clock, copied inventories or extra reaction heat.
    """
    if any(species.phase != "gas" for species in network.species):
        raise ValueError("gas batch requires gas species")
    thermo = network.imported_gas_thermochemistry
    if set(thermo) != set(network.species_ids):
        raise ValueError("gas batch requires all declared thermochemistry")
    elements = {element for species in network.species for element in species.composition}
    return CoupledSystem(
        IdealGasThermodynamics(thermo),
        (reaction_network_process(network, species_thermo=thermo),),
        heat_flux=heat_flux or (lambda state: HeatFlux()),
        conservation_laws={
            element: {s.species_id: s.composition.get(element, 0) for s in network.species}
            for element in elements
        },
        **({"policy": policy} if policy is not None else {}),
    )
