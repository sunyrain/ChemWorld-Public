"""Retained condensed-phase enthalpy law, independent of runtime wiring."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution


@dataclass(frozen=True)
class LegacyCrystalEnthalpy:
    species: dict[str, float]
    support_heat_capacity_J_K: float
    reference_temperature_K: float = 298.15

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(self.species)

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return self.support_heat_capacity_J_K

    def internal_energy_j(self, state: CoupledStateView) -> None:
        return None

    def control_volume_enthalpy_j(self, state: CoupledStateView) -> float:
        return self.support_heat_capacity_J_K * (
            state.temperature_K - self.reference_temperature_K
        ) + sum(self.species[k] * n for k, n in state.amounts_mol.items())

    def species_energy_j_mol(self, key: str, temperature: float) -> float:
        return self.species[key]

    def chemical_energy_rate_w(
        self, state: CoupledStateView, derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError(
                "legacy crystal contact does not certify simultaneous reactive calorics"
            )
        return sum(self.species[k] * v for k, v in derivatives.items())
