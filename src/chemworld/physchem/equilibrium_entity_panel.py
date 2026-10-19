"""Multi-entity substrate for the bounded EQ entity locus."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from chemworld.physchem.equilibrium_mechanism import (
    CoupledEquilibriumResult,
    solve_coupled_weak_acid_precipitation,
)


@dataclass(frozen=True)
class EquilibriumEntityProfile:
    """A private material-identity property vector under one common topology."""

    entity_id: str
    pka_shift: float
    log10_ksp: float
    cation_fraction: float
    activity_coefficient_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.pka_shift,
            self.log10_ksp,
            self.cation_fraction,
            self.activity_coefficient_ratio,
        )
        if not self.entity_id or not all(isfinite(value) for value in values):
            raise ValueError("EQ-E entity profile must be named and finite")
        if not 0.0 < self.cation_fraction < 1.0:
            raise ValueError("EQ-E cation fraction must be in (0, 1)")
        if self.activity_coefficient_ratio <= 0.0:
            raise ValueError("EQ-E activity ratio must be positive")


def solve_equilibrium_entity(
    *,
    profile: EquilibriumEntityProfile,
    acid_total_mol: float,
    volume_L: float,
    base_pka: float,
) -> CoupledEquilibriumResult:
    """Solve one entity while holding the registered reaction network fixed."""

    return solve_coupled_weak_acid_precipitation(
        acid_total_mol=acid_total_mol,
        volume_L=volume_L,
        pka=base_pka + profile.pka_shift,
        log10_ksp=profile.log10_ksp,
        mechanism_family="direct_free_ion_precipitation",
        cation_fraction=profile.cation_fraction,
        activity_coefficient_ratio=profile.activity_coefficient_ratio,
    )


__all__ = ["EquilibriumEntityProfile", "solve_equilibrium_entity"]
