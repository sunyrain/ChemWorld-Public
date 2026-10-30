"""Caller-owned pure water at fixed volume on the original thermal clock."""

from dataclasses import dataclass
from math import isfinite

from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.water_if97 import _positive
from chemworld.physchem.water_if97_properties import IF97WaterModel
from chemworld.physchem.water_if97_volume import (
    water_if97_internal_energy_derivatives,
    water_if97_trho,
)


@dataclass(frozen=True)
class IF97IsochoricThermodynamics:
    component: ComponentSpec
    volume_L: float
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        IF97WaterModel().validate_component(self.component)
        _positive(self.volume_L, "volume_L")
        _positive(self.reference_temperature_K, "reference_temperature_K")
        if (isinstance(self.support_heat_capacity_J_K, bool)
                or not isfinite(self.support_heat_capacity_J_K)
                or self.support_heat_capacity_J_K < 0):
            raise ValueError("IF97 support heat capacity must be finite and nonnegative")

    @property
    def inventory_keys(self):
        return (self.component.identifier,)

    @property
    def molar_mass_kg_mol(self):
        return self.component.molecular_weight_g_mol / 1000

    def validate_temperature_interval(self, low, high):
        if not 273.15 <= low < high <= 2273.15:
            raise ValueError("thermal bounds leave the IF97 fluid temperature domain")

    def water_state(self, state):
        if set(state.amounts_mol) != set(self.inventory_keys):
            raise ValueError("IF97 thermal stock must contain only the declared water component")
        if state.volume_L != self.volume_L:
            raise ValueError("IF97 isochoric thermal model rejects changing volume")
        amount = state.amounts_mol[self.component.identifier]
        _positive(amount, "water inventory")
        return water_if97_trho(state.temperature_K,
                              amount * self.molar_mass_kg_mol / (self.volume_L * 1e-3))

    def validate_state(self, state):
        self.water_state(state)

    def heat_capacity_j_k(self, state):
        current = self.water_state(state)
        capacity = water_if97_internal_energy_derivatives(current)[0]
        if not isfinite(capacity) or capacity <= 0:
            raise ValueError("IF97 equilibrium isochoric thermal response must be positive")
        return (state.amounts_mol[self.component.identifier] * self.molar_mass_kg_mol * capacity
                + self.support_heat_capacity_J_K)

    def internal_energy_j(self, state):
        return (state.amounts_mol[self.component.identifier] * self.molar_mass_kg_mol
                * self.water_state(state).internal_energy_J_kg + self.support_heat_capacity_J_K
                * (state.temperature_K - self.reference_temperature_K))

    def pressure_pa(self, state):
        return self.water_state(state).pressure_Pa

    def partial_inventory_energy_j_mol(self, state):
        current = self.water_state(state)
        density_derivative = water_if97_internal_energy_derivatives(current)[1]
        return self.molar_mass_kg_mol * (current.internal_energy_J_kg
                                       + current.mass_density_kg_m3 * density_derivative)

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        if set(derivatives) - set(self.inventory_keys):
            raise ValueError("IF97 energy rate refers to an undeclared component")
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("same-EOS IF97 internal energy rejects duplicate reaction heat")
        rate = derivatives.get(self.component.identifier, 0.0)
        if not isfinite(rate):
            raise ValueError("IF97 inventory rate must be finite")
        return self.partial_inventory_energy_j_mol(state) * rate if rate else 0.0

    def to_dict(self):
        return {"model": "if97-equilibrium-isochoric-v1", "component": self.component.to_dict(),
                "volume_L": self.volume_L,
                "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
                "reference_temperature_K": self.reference_temperature_K}

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        if values.pop("model", None) != "if97-equilibrium-isochoric-v1":
            raise ValueError("unknown IF97 thermal declaration")
        values["component"] = ComponentSpec.from_dict(values["component"])
        return cls(**values)
