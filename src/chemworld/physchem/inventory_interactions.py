"""Shared caloric correction from declared entity-owned interaction energies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Protocol

import numpy as np

from chemworld.physchem.coupled_system import CoupledThermodynamics

if TYPE_CHECKING:
    from chemworld.foundation.state_ownership import InventoryInteractionSpec, StateSlot
    from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution


class InventoryThermodynamics(CoupledThermodynamics, Protocol):
    """Caloric base used by the inventory interaction correction."""

    def internal_energy_j(self, state: CoupledStateView) -> float: ...

    def species_energy_j_mol(self, key: str, temperature_K: float) -> float: ...


@dataclass(frozen=True)
class InventoryInteraction:
    keys: tuple[str, ...]
    coefficients_J_mol: tuple[tuple[float, ...], ...]
    capacity_mol: float

    def __post_init__(self) -> None:
        if not isfinite(self.capacity_mol) or self.capacity_mol < 0:
            raise ValueError("interaction carrier capacity must be finite and nonnegative")

    @classmethod
    def from_slot(cls, slot: StateSlot, spec: InventoryInteractionSpec) -> InventoryInteraction:
        return cls(
            tuple(f"{slot.spec.slot_id}:{name}" for name in spec.fields),
            spec.coefficients_J_mol,
            sum(slot.values[key] for key in spec.capacity_fields),
        )

    def energy_and_gradient(self, amounts: Mapping[str, float]) -> tuple[float, dict[str, float]]:
        n = np.array([amounts[key] for key in self.keys])
        if self.capacity_mol == 0:
            if np.any(n != 0):
                raise ValueError("an empty carrier cannot hold interacting inventory")
            return 0.0, dict.fromkeys(self.keys, 0.0)
        gradient = np.asarray(self.coefficients_J_mol) @ n / self.capacity_mol
        return float(0.5 * n @ gradient), dict(zip(self.keys, map(float, gradient), strict=True))


@dataclass(frozen=True)
class InteractingThermodynamics:
    base: InventoryThermodynamics
    interactions: tuple[InventoryInteraction, ...]

    def __post_init__(self) -> None:
        if any(set(term.keys) - set(self.base.inventory_keys) for term in self.interactions):
            raise ValueError("interaction energy refers to an undeclared caloric inventory")

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return self.base.inventory_keys

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return self.base.heat_capacity_j_k(state)

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return self.base.internal_energy_j(state) + sum(
            term.energy_and_gradient(state.amounts_mol)[0] for term in self.interactions
        )

    def chemical_energy_rate_w(
        self, state: CoupledStateView, derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        return self.base.chemical_energy_rate_w(state, derivatives, contributions) + sum(
            mu * derivatives.get(key, 0)
            for term in self.interactions
            for key, mu in term.energy_and_gradient(state.amounts_mol)[1].items()
        )

    def species_energy_j_mol(self, key: str, temperature_K: float) -> float:
        # Used for a bare gas entering/leaving a carrier, never for a bound
        # species packet. Material mixing uses full state U above.
        if any(key in term.keys for term in self.interactions):
            raise ValueError("interacting species energy requires its complete inventory")
        return self.base.species_energy_j_mol(key, temperature_K)

    def validate_state(self, state: CoupledStateView) -> None:
        validate = getattr(self.base, "validate_state", None)
        if validate is not None:
            validate(state)
        for term in self.interactions:
            term.energy_and_gradient(state.amounts_mol)

    def validity_warnings(self, state: CoupledStateView) -> tuple[str, ...]:
        report = getattr(self.base, "validity_warnings", None)
        return () if report is None else report(state)


def with_inventory_interactions(
    base: InventoryThermodynamics, slots: Sequence[StateSlot],
) -> InventoryThermodynamics:
    terms = tuple(
        InventoryInteraction.from_slot(slot, spec)
        for slot in slots
        for spec in slot.spec.energy_interactions
    )
    return InteractingThermodynamics(base, terms) if terms else base
