"""Dufal Mie-fluid association integral feeding the shared site equilibrium."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import isfinite
from typing import Any

from chemworld.physchem.caloric_math import compensated_sum, scalar_value
from chemworld.physchem.dufal_coefficients import DUFAL_COEFFICIENTS
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL
from chemworld.physchem.saft_association import SAFTAssociation, association_helmholtz


@lru_cache(maxsize=128)
def _coefficients(repulsive: float) -> tuple[tuple[int, int, float], ...]:
    return tuple((i, j, sum(c*repulsive**k for k, c in enumerate(polynomial)))
                 for i, j, polynomial in DUFAL_COEFFICIENTS)


def dufal_integral(temperature_reduced: Any, density_reduced: Any, repulsive: float) -> Any:
    """Published triangular polynomial; negative fitted strengths are rejected.

    This is a correlation, not an exact pair-distribution function outside its
    fitted domain. No clipping of negative or nonfinite integral values.
    """
    if (not isfinite(repulsive) or repulsive <= 3
            or scalar_value(temperature_reduced) <= 0 or scalar_value(density_reduced) < 0):
        raise ValueError("Dufal integral requires positive Tstar, nonnegative rho and lambda>3")
    result = compensated_sum(a*density_reduced**i*temperature_reduced**j
                             for i, j, a in _coefficients(repulsive))
    if not isfinite(scalar_value(result)) or scalar_value(result) <= 0:
        raise ValueError("Dufal polynomial outside its positive finite integral domain")
    return result


@dataclass(frozen=True)
class DufalAssociation:
    """Explicit kernel parameters and site-pair bonding energies/volumes.

    `segment` uses rho_s and segment fractions (ThermoPack formulation);
    `molecular` records teqp0.23.2's molecular-density convention. Parameters
    belong to the association kernel and may differ from the nonpolar fit.
    Pair epsilon/lambda are explicit, so no unrecorded mixing is inferred.
    """

    component_ids: tuple[str, ...]
    segment_counts: tuple[float, ...]
    diameters_m: tuple[float, ...]
    pair_well_depths_K: tuple[tuple[float, ...], ...]
    pair_repulsive_exponents: tuple[tuple[float, ...], ...]
    sites: SAFTAssociation
    source: str
    density_basis: str = "segment"

    def __post_init__(self) -> None:
        for field in ('component_ids', 'segment_counts', 'diameters_m'):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        for field in ('pair_well_depths_K', 'pair_repulsive_exponents'):
            object.__setattr__(self, field, tuple(tuple(row) for row in getattr(self, field)))
        n = len(self.component_ids)
        if (not n or len(set(self.component_ids)) != n or not all(self.component_ids)
                or not self.source or self.density_basis not in ('segment', 'molecular')):
            raise ValueError("Dufal needs unique components, provenance and explicit density basis")
        if (len(self.segment_counts) != n or len(self.diameters_m) != n or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (*self.segment_counts, *self.diameters_m)
        )):
            raise ValueError("Dufal needs positive component segment counts and diameters")
        for matrix, minimum in ((self.pair_well_depths_K, 0), (self.pair_repulsive_exponents, 3)):
            if len(matrix) != n or any(len(row) != n for row in matrix):
                raise ValueError("Dufal requires full pair energy/exponent matrices")
            if any(isinstance(v, bool) or not isfinite(v) or v <= minimum or v != matrix[j][i]
                   for i, row in enumerate(matrix) for j, v in enumerate(row)):
                raise ValueError("Dufal requires positive symmetric energies and exponents>3")
        if any(s.component_id not in self.component_ids for s in self.sites.sites):
            raise ValueError("Dufal site belongs to an undeclared component")

    def contacts(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> list[list[Any]]:
        n = len(self.component_ids)
        xs = tuple(scalar_value(x) for x in fractions)
        if (len(xs) != n or any(not isfinite(x) or x < 0 for x in xs) or abs(sum(xs)-1) > 1e-10
                or not isfinite(scalar_value(density)) or scalar_value(density) < 0
                or not isfinite(scalar_value(temperature)) or scalar_value(temperature) <= 0):
            raise ValueError("Dufal state needs normalized composition and valid T/rho")
        weights, concentration = fractions, density*AVOGADRO_PER_MOL
        if self.density_basis == 'segment':
            mean = compensated_sum(x*m for x, m in zip(fractions, self.segment_counts, strict=True))
            weights = tuple(x*m/mean for x, m in zip(fractions, self.segment_counts, strict=True))
            concentration *= mean
        reduced_density = concentration*compensated_sum(
            weights[i]*weights[j]*((self.diameters_m[i]+self.diameters_m[j])/2)**3
            for i in range(n) for j in range(n))
        return [[dufal_integral(temperature/self.pair_well_depths_K[i][j], reduced_density,
                               self.pair_repulsive_exponents[i][j]) for j in range(n)]
                for i in range(n)]

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return association_helmholtz(self.sites, temperature, density*AVOGADRO_PER_MOL,
            self.component_ids, fractions, self.contacts(temperature, density, fractions))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DufalAssociation:
        data = dict(payload)
        data['sites'] = SAFTAssociation.from_dict(data['sites'])
        return cls(**data)
