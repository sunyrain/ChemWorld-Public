"""Mechanism-aware species role helpers for the transactional runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chemworld.foundation import WorldState, species_with_added_initial_amounts
from chemworld.foundation.state import SpeciesLedger
from chemworld.runtime.mechanisms import CompiledMechanism


@dataclass(frozen=True)
class MechanismSpeciesView:
    """Resolve semantic species roles from a compiled mechanism."""

    mechanism: CompiledMechanism

    def __post_init__(self) -> None:
        if self.mechanism is None:
            raise ValueError("Runtime species roles require a compiled mechanism")

    @property
    def target_species(self) -> tuple[str, ...]:
        return self._role_species("target", fallback=self.mechanism.score_spec.target_species)

    @property
    def impurity_species(self) -> tuple[str, ...]:
        return self._role_species("impurity", fallback=self.mechanism.score_spec.impurity_species)

    @property
    def byproduct_species(self) -> tuple[str, ...]:
        species = self.mechanism.observable_mapping.get("byproduct", ())
        return species or tuple(
            species_id
            for species_id in self.impurity_species
            if species_id not in self.degradation_species
        )

    @property
    def degradation_species(self) -> tuple[str, ...]:
        return self.mechanism.observable_mapping.get("degradation", ())

    @property
    def catalyst_species(self) -> tuple[str, ...]:
        return self.mechanism.observable_mapping.get("catalyst", ())

    @property
    def primary_target_species(self) -> str:
        if not self.target_species:
            raise ValueError(
                f"Mechanism {self.mechanism.mechanism_id!r} does not declare target species"
            )
        return self.target_species[0]

    @property
    def primary_impurity_species(self) -> str:
        if not self.impurity_species:
            raise ValueError(
                f"Mechanism {self.mechanism.mechanism_id!r} does not declare impurity species"
            )
        return self.impurity_species[0]

    def _role_species(
        self,
        role: str,
        *,
        fallback: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        return self.mechanism.observable_mapping.get(role, ()) or fallback

    def reactant_species(self, state: WorldState | None = None) -> str:
        if self.mechanism.score_spec.initial_limiting_species:
            species_id = self.mechanism.score_spec.initial_limiting_species
            if state is None or species_id in state.species_amounts:
                return species_id
        reactants = self.mechanism.observable_mapping.get("reactant", ())
        for species_id in reactants:
            if state is None or species_id in state.species_amounts:
                return species_id
        raise ValueError(
            f"Mechanism {self.mechanism.mechanism_id!r} does not declare a usable reactant species"
        )

    def active_catalyst_species(self, state: WorldState | None = None) -> str | None:
        for species_id in self.catalyst_species:
            is_active = "active" in species_id.lower()
            is_present = state is None or species_id in state.species_amounts
            if is_active and is_present:
                return species_id
        for species_id in self.catalyst_species:
            if state is None or species_id in state.species_amounts:
                return species_id
        return None

    def target_species_for_state(self, state: WorldState) -> tuple[str, ...]:
        species = tuple(
            species_id for species_id in self.target_species if species_id in state.species_amounts
        )
        return species or self.target_species

    def impurity_species_for_state(self, state: WorldState) -> tuple[str, ...]:
        species = tuple(
            species_id
            for species_id in self.impurity_species
            if species_id in state.species_amounts
        )
        return species or self.impurity_species

    def byproduct_species_for_state(self, state: WorldState) -> tuple[str, ...]:
        species = tuple(
            species_id
            for species_id in self.byproduct_species
            if species_id in state.species_amounts
        )
        return species or self.byproduct_species

    def degradation_species_for_state(self, state: WorldState) -> tuple[str, ...]:
        species = tuple(
            species_id
            for species_id in self.degradation_species
            if species_id in state.species_amounts
        )
        return species or self.degradation_species

    def amount(self, state: WorldState, species_ids: tuple[str, ...]) -> float:
        return sum(float(state.species_amounts.get(species_id, 0.0)) for species_id in species_ids)

    def target_amount(self, state: WorldState) -> float:
        return self.amount(state, self.target_species_for_state(state))

    def impurity_amount(self, state: WorldState) -> float:
        return self.amount(state, self.impurity_species_for_state(state))

    def byproduct_amount(self, state: WorldState) -> float:
        return self.amount(state, self.byproduct_species_for_state(state))

    def degradation_amount(self, state: WorldState) -> float:
        return self.amount(state, self.degradation_species_for_state(state))

    def reactant_amount(self, state: WorldState) -> float:
        coefficients = self.feed_equivalent_coefficients
        return min(
            float(state.species_amounts.get(key, 0.0)) / coefficient
            for key, coefficient in coefficients.items()
        )

    @property
    def feed_species(self) -> tuple[str, ...]:
        """Finite independently chargeable feed catalog, in mechanism order."""
        # Unreacted cofeeds may also be analytical impurities; that does not
        # make them ineligible as separately chargeable starting materials.
        excluded = {*self.catalyst_species, *self.target_species}
        return tuple(
            species.species_id
            for species in self.mechanism.network.species
            if self.mechanism.initial_amount_policy.get(species.species_id, 0.0) > 0.0
            and species.species_id not in excluded
        ) or (self.reactant_species(),)

    @property
    def feed_equivalent_coefficients(self) -> dict[str, float]:
        """Direct target stoichiometry, never the reference recipe's dose ratios.

        Multi-step single-feed networks retain their declared primary-feed basis.
        Independent cofeeds use the unique target-forming feed stoichiometry.
        """
        feeds = set(self.feed_species)
        candidates = []
        for reaction in self.mechanism.network.reactions:
            target = sum(
                max(reaction.stoichiometry.get(key, 0.0), 0.0) for key in self.target_species
            )
            consumed = (
                {
                    key: -value / target
                    for key, value in reaction.stoichiometry.items()
                    if value < 0 and key in feeds
                }
                if target > 0
                else {}
            )
            if consumed:
                candidates.append(consumed)
        if len(feeds) > 1:
            if not candidates or any(value != candidates[0] for value in candidates[1:]):
                raise ValueError("Independent cofeeds require unambiguous target stoichiometry")
            return candidates[0]
        return {self.reactant_species(): 1.0}

    def initial_reactant_amount(self, state: WorldState) -> float:
        if state.species is not None:
            return max(
                min(
                    float(state.species.initial_amounts_mol.get(key, 0.0)) / coefficient
                    for key, coefficient in self.feed_equivalent_coefficients.items()
                ),
                0.0,
            )
        return max(
            self.reactant_amount(state) + self.target_amount(state) + self.impurity_amount(state),
            0.0,
        )

    def record_added_reactant(
        self,
        species: SpeciesLedger | None,
        *,
        reactant_species: str,
        amount_mol: float,
    ) -> SpeciesLedger:
        return species_with_added_initial_amounts(
            species,
            {reactant_species: amount_mol},
        )

    def reagent_charge_amounts(
        self,
        state: WorldState,
        *,
        limiting_amount_mol: float,
    ) -> dict[str, float]:
        """Return non-catalytic mechanism-ratio additions for one reagent charge.

        ``initial_amount_policy`` describes a complete mechanism reference charge,
        which may include catalyst inventory for standalone model initialization.
        The public runtime exposes catalyst dosing through ``add_catalyst`` as a
        separate controllable operation, so silently co-charging catalyst here
        would couple reagent amount to a hidden catalyst dose and double-count
        recipes that subsequently call ``add_catalyst``.
        """

        reactant = self.reactant_species(state)
        policy = self.mechanism.initial_amount_policy
        reference = float(policy.get(reactant, 0.0))
        if reference <= 0.0:
            return {reactant: limiting_amount_mol}
        catalyst_species = set(self.catalyst_species)
        return {
            species_id: limiting_amount_mol * float(amount) / reference
            for species_id, amount in policy.items()
            if (
                amount > 0.0
                and species_id in state.species_amounts
                and species_id not in catalyst_species
            )
        } or {reactant: limiting_amount_mol}

    def truth_values(self, state: WorldState) -> dict[str, float]:
        initial = max(self.initial_reactant_amount(state), 0.0)
        denominator = max(initial, 1.0e-12)
        target = self.target_amount(state)
        impurity = self.impurity_amount(state)
        remaining = self.reactant_amount(state)
        consumed = max(initial - remaining, 0.0)
        return {
            "yield": float(np.clip(target / denominator, 0.0, 1.0)),
            "selectivity": float(np.clip(target / max(consumed, 1.0e-12), 0.0, 1.0)),
            "conversion": float(np.clip(consumed / denominator, 0.0, 1.0)),
            "byproduct_signal": float(np.clip(impurity / denominator, 0.0, 1.0)),
            "degradation_warning": float(
                np.clip(self.degradation_amount(state) / denominator, 0.0, 1.0)
            ),
        }


__all__ = [
    "MechanismSpeciesView",
]
