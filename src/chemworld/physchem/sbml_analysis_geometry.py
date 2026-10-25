"""Reachable coordinates for read-only analyses of original smooth SBML ODEs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any, cast

import numpy as np

from chemworld.physchem.biochemical_analysis import _check_scope, _stoichiometry
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.physchem.sbml_variational import sbml_variational_rates
from chemworld.physchem.stoichiometry import _rref


@dataclass(frozen=True)
class SBMLAnalysisGeometry:
    model: SBMLSystemSpec
    initial: dict[str, float]
    keys: tuple[str, ...]
    species: tuple[str, ...]
    indices: list[int]
    metric: np.ndarray
    numerical_scales: np.ndarray
    basis: np.ndarray
    conservation: np.ndarray
    singular_values: np.ndarray

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    def values(self, scaled: np.ndarray) -> dict[str, float]:
        return dict(zip(self.keys, scaled * self.numerical_scales, strict=True))

    def directions(self) -> np.ndarray:
        result = np.zeros((len(self.keys), self.rank))
        result[self.indices] = self.metric[:, None] * self.basis
        return result

    def validate(self, values: Mapping[str, float], time_s: float) -> None:
        self.model.frame.validate_values(values, time_s)
        if any(values["N:" + s] <= 0 for s in self.species):
            raise ValueError("two-sided trajectory analysis requires strictly positive amounts")

    def reduced_jacobian(self, values: Mapping[str, float], time_s: float) -> np.ndarray:
        _, derivative = sbml_variational_rates(
            self.model.frame, values, self.keys, self.directions(), time_s
        )
        return self.basis.T @ (derivative[self.indices] / self.metric[:, None])

    def totals(self, values: Mapping[str, float]) -> np.ndarray:
        return self.conservation @ (
            np.asarray([values["N:" + s] for s in self.species]) / self.metric
        )

    def describe(self) -> dict[str, Any]:
        return {
            "initial_coordinates": dict(self.initial),
            "species_ids": list(self.species),
            "amount_scales_mol": dict(zip(self.species, map(float, self.metric), strict=True)),
            "basis_scaled_amounts": self.basis.tolist(),
            "reachable_dimension": self.rank,
            "conservation_rows_scaled_amounts": self.conservation.tolist(),
            "initial_conserved_totals": self.totals(self.initial).tolist(),
            "conserved_zero_modes_excluded": len(self.species) - self.rank,
            "stoichiometric_singular_values": self.singular_values.tolist(),
            "rank_method": "exact binary-rational stoichiometry; metric resolution checked",
            "metric": "Euclidean norm of delta_n / declared fixed mol scales",
            "controls": "held current parameters, volumes and boundary pools",
            "world_advanced": False,
            "live_resources_charged": False,
            "future_controls_predicted": False,
        }


def sbml_analysis_geometry(
    model: SBMLSystemSpec,
    current_values: Mapping[str, float],
    *,
    start_time_s: float,
    amount_scales_mol: Mapping[str, float],
    maximum_directions: int = 32,
    max_integer_bits: int = 8192,
) -> SBMLAnalysisGeometry:
    if not isinstance(model, SBMLSystemSpec) or model.numerical_form != "ode":
        raise ValueError("trajectory analysis requires the original explicit SBML ODE")
    _check_scope(model.frame)
    if isinstance(start_time_s, bool) or not isfinite(start_time_s) or start_time_s < 0:
        raise ValueError("analysis time must be finite nonnegative seconds")
    if any(type(v) is not int or v < 1 for v in (maximum_directions, max_integer_bits)):
        raise ValueError(
            "analysis dimension and exact arithmetic budgets must be positive integers"
        )
    frame = model.frame
    factors = {frame.model.declaration.get("conversionFactor")} | {
        s.get("conversionFactor") for s in frame._species.values()
    }
    if any(
        name is not None
        and (
            name not in frame._parameters
            or not frame._parameters[name].get("constant")
            or name in frame._assigned
        )
        for name in factors
    ):
        raise ValueError("reachable geometry requires constant stoichiometric conversionFactors")
    if set(current_values) != set(frame.initial_values) or any(
        isinstance(v, bool) or not isfinite(v) for v in current_values.values()
    ):
        raise ValueError("analysis requires the complete finite original coordinate state")
    current = {k: float(v) for k, v in current_values.items()}
    species = tuple(
        s
        for s, d in frame._species.items()
        if not d.get("boundaryCondition") and not d.get("constant")
    )
    if (
        not species
        or set(amount_scales_mol) != set(species)
        or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in amount_scales_mol.values())
    ):
        raise ValueError("explicit positive mol metric must cover exactly all dynamic species")
    keys = tuple(current)
    indices = [keys.index("N:" + s) for s in species]
    metric = np.asarray([amount_scales_mol[s] for s in species], dtype=float)
    all_species = tuple(frame._species)
    matrix = _stoichiometry(frame, current, start_time_s)[[all_species.index(s) for s in species]]
    _, pivots = _rref(
        [[Fraction(float(v)) for v in row] for row in matrix],
        width=matrix.shape[1],
        max_integer_bits=max_integer_bits,
    )
    rank = len(pivots)
    if rank == 0 or rank > maximum_directions:
        raise ValueError("reachable dimension is zero or exceeds the declared budget")
    weighted = matrix / metric[:, None]
    if not np.all(np.isfinite(weighted)):
        raise ValueError("declared metric cannot represent original stoichiometry")
    vectors, singular, _ = np.linalg.svd(weighted, full_matrices=True)
    if singular[rank - 1] <= 64 * max(weighted.shape) * np.finfo(float).eps * singular[0]:
        raise ValueError("exact stoichiometric rank is unresolved in the declared floating metric")
    declared = cast(Mapping[str, float], model.coordinate_scales)
    scales = np.asarray(
        [
            amount_scales_mol[k[2:]]
            if k in {"N:" + s for s in species}
            else declared.get(k, max(abs(current[k]), 1e-6))
            for k in keys
        ]
    )
    result = SBMLAnalysisGeometry(
        model,
        current,
        keys,
        species,
        indices,
        metric,
        scales,
        vectors[:, :rank],
        vectors[:, rank:].T,
        singular,
    )
    result.validate(current, start_time_s)
    return result
