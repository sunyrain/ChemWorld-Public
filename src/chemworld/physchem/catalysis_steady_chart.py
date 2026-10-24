"""Exact structural compatibility classes for complete surface configurations.

Only coordinates change: every trial reconstructs the original molar inventory.
The initial state fixes all left-nullspace totals, including disconnected pools.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from chemworld.physchem.catalysis_network import CatalyticMechanismSpec, make_catalytic_slot
from chemworld.physchem.stoichiometry import _rref


@dataclass(frozen=True)
class CatalyticSteadyChart:
    species: tuple[str, ...]
    # Per-species molecular mol per theta, i.e. physical site capacity / occupancy.
    capacities_mol: tuple[float, ...]
    initial_coverages: tuple[float, ...]
    coordinate_indices: tuple[int, ...]
    coverage_basis: tuple[tuple[float, ...], ...]
    conservation_rows: tuple[tuple[float, ...], ...]

    @property
    def rank(self) -> int:
        return len(self.coordinate_indices)

    def reconstruct(self, coordinates: Any) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        if values.shape != (self.rank,) or not np.all(np.isfinite(values)):
            raise ValueError("surface chart requires finite independent coordinates")
        return np.asarray(self.initial_coverages) + np.asarray(self.coverage_basis) @ values

    def reduce_rates(self, coverage_rates: Any) -> np.ndarray:
        return np.asarray(coverage_rates)[list(self.coordinate_indices)]

    def totals(self, coverages: Mapping[str, float]) -> np.ndarray:
        if set(coverages) != set(self.species):
            raise ValueError("compatibility class requires every original surface species")
        values = np.array([coverages[k] for k in self.species], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("compatibility class requires finite nonnegative coverages")
        return np.asarray(self.conservation_rows) @ (values * self.capacities_mol)

    def same_class(self, coverages: Mapping[str, float]) -> bool:
        initial = dict(zip(self.species, self.initial_coverages, strict=True))
        # Scale each invariant by its actual capacity-weighted coefficients.
        # A global absolute mol tolerance would merge tiny but distinct pools.
        scale = np.abs(np.asarray(self.conservation_rows)) @ self.capacities_mol
        return bool(np.all(np.abs(self.totals(coverages) - self.totals(initial)) <= 1e-10 * scale))

    def to_dict(self) -> dict[str, Any]:
        initial = dict(zip(self.species, self.initial_coverages, strict=True))
        return {
            "species": list(self.species),
            "molecular_amount_per_coverage_mol": list(self.capacities_mol),
            "reachable_dimension": self.rank,
            "conserved_dimension": len(self.species) - self.rank,
            "coordinate_species": [self.species[i] for i in self.coordinate_indices],
            "coverage_basis": [list(row) for row in self.coverage_basis],
            "conservation_rows": [list(row) for row in self.conservation_rows],
            "conserved_totals_mol": self.totals(initial).tolist(),
            "construction": "exact rational full surface stoichiometry; affine compatibility class",
            "spectrum_scope": "reachable subspace; genuine dynamical zero modes retained",
        }


def build_catalytic_steady_chart(
    spec: CatalyticMechanismSpec,
    capacities_mol: Mapping[str, float],
    initial_coverages: Mapping[str, float],
    *,
    coordinate_order: Sequence[str] | None = None,
) -> CatalyticSteadyChart:
    """Select independent physical coverage coordinates using exact elimination."""
    if not spec.site_species_complete:
        raise ValueError("conserved surface charts require complete explicit configurations")
    names = spec.adsorbates
    if set(initial_coverages) != set(names) or set(capacities_mol) != set(spec.site_types):
        raise ValueError("chart must use exactly the original surface species and capacities")
    order = tuple(names if coordinate_order is None else coordinate_order)
    if len(order) != len(names) or set(order) != set(names):
        raise ValueError("coordinate_order must be a permutation of all surface species")
    scales = spec.coverage_amount_scales_mol(capacities_mol)
    caps = tuple(float(scales[k]) for k in names)
    theta = tuple(float(initial_coverages[k]) for k in names)
    make_catalytic_slot(
        "chart-validation",
        "analysis",
        spec,
        capacities_mol=capacities_mol,
        initial_amounts_mol={k: t * c for k, t, c in zip(names, theta, caps, strict=True)},
    )
    stoich = [[Fraction(r.stoichiometry.get(k, 0)) for r in spec.reactions] for k in names]
    _, columns = _rref(stoich, width=len(spec.reactions))
    # Exact row selection and exact nullspace; no singular-value rank cutoff.
    transpose = [list(row) for row in zip(*stoich, strict=True)]
    reduced, pivots = _rref(transpose, width=len(names))
    conservation = []
    for free in (i for i in range(len(names)) if i not in pivots):
        row = [Fraction(int(i == free)) for i in range(len(names))]
        for index, pivot in enumerate(pivots):
            row[pivot] = -reduced[index][free]
        conservation.append(tuple(float(v) for v in row))
    permutation = [names.index(k) for k in order]
    _, ordered_pivots = _rref(
        [[row[i] for i in permutation] for row in transpose], width=len(names)
    )
    indices = tuple(permutation[i] for i in ordered_pivots)
    raw = np.array(stoich, dtype=float)[:, columns] / np.array(caps)[:, None]
    basis = raw @ np.linalg.inv(raw[list(indices), :]) if columns else raw
    return CatalyticSteadyChart(
        names,
        caps,
        theta,
        indices,
        tuple(tuple(float(v) for v in row) for row in basis),
        tuple(conservation),
    )
