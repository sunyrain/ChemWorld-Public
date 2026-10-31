"""Transient caloric reuse for repeated species in a crystal population.

No physical approximation: only identical model objects at an exactly equal
temperature, requested quantities and validity policy share an evaluation.
The cache lives for one local population layout, never in a world snapshot.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution
from chemworld.physchem.property_packages import PropertyPackageThermodynamics
from chemworld.physchem.property_reports import PropertyEvaluation


@dataclass(frozen=True)
class CrystalCaloricLayout(PropertyPackageThermodynamics):
    _evaluations: dict[object, float | dict[str, PropertyEvaluation]] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def _at(
        self, key: str, temperature: float, quantities: Sequence[str], *, strict: bool,
    ) -> dict[str, PropertyEvaluation]:
        cache = self._evaluations
        if cache.get("temperature") != temperature:
            cache.clear()
            cache["temperature"] = temperature
        identity = (id(self.species[key]), tuple(quantities), strict)
        if identity not in cache:
            cache[identity] = super()._at(key, temperature, quantities, strict=strict)
        return cast(dict[str, PropertyEvaluation], cache[identity])

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        from chemworld.physchem.coupled_system import CoupledState
        strict = isinstance(state, CoupledState)
        values: dict[int, float] = {}

        def terms() -> Iterator[float]:
            for key, species in self.species.items():
                amount = state.amounts_mol[key]
                if amount:
                    identity = id(species)
                    if identity not in values:
                        values[identity] = self._at(key, state.temperature_K, ('cv',),
                                                    strict=strict)['cv'].value
                    yield amount * values[identity]

        return self.support_heat_capacity_J_K + sum(terms())

    def chemical_energy_rate_w(
        self, state: CoupledStateView, derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        if any(item.legacy_reaction_energy_W is not None for item in contributions):
            raise ValueError('native caloric U cannot also consume a separate reaction heat')
        values: dict[int, float] = {}

        def terms() -> Iterator[float]:
            for key, rate in derivatives.items():
                if rate:
                    identity = id(self.species[key])
                    if identity not in values:
                        values[identity] = self._at(
                            key, state.temperature_K, ('internal_energy',), strict=False
                        )['internal_energy'].value
                    yield values[identity] * rate

        # Preserve the original derivative iteration and summation order.
        return sum(terms())
