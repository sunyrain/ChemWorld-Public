"""Sourced multi-reaction occupancies and shared-potential MSMR kinetics.

These are the PyBaMM 25.4.2 MSMR constitutive equations, evaluated from immutable
authored parameters, not a fitted OCV. The state owner remains total particle Li.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from math import exp, fsum, isclose, isfinite, log, log1p
from typing import Any, Self, cast

import numpy as np
from scipy.special import logsumexp

from chemworld.data.scientific_inputs import ScientificTable, _keys, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F
from chemworld.physchem.electrochemistry import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import (
    ResidualNumericalError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)

MSMR_MATERIAL_VERSION = "msmr-occupancy-kinetics-v1"
# Authored capacity weight, reference potential, width, exchange scale, alpha.
ReactionParameters = tuple[float, float, float, float, float]
ReactionRows = tuple[ReactionParameters, ...]
MSMR_UNITS = {
    "X": "dimensionless",
    "U0_V": "V",
    "w": "dimensionless",
    "jref_A_m2": "A/m^2",
    "alpha": "dimensionless",
}


def _temperature(temperature: float) -> float:
    if isinstance(temperature, bool) or not isfinite(temperature) or temperature <= 0:
        raise ValueError("MSMR temperature must be positive and finite")
    factor = F / (R * temperature)
    if not isfinite(factor) or factor <= 0:
        raise ValueError("MSMR inverse thermal voltage is not representable")
    return factor


def _fractions(rows: ReactionRows, potential: float, factor: float) -> tuple[float, ...]:
    return _fractions_and_vacancies(rows, potential, factor)[0]


def _fractions_and_vacancies(
    rows: ReactionRows, potential: float, factor: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isfinite(potential):
        raise ValueError("MSMR potential must be finite")
    arguments = tuple(factor * (potential - u0) / width for _, u0, width, _, _ in rows)
    # Evaluate both tails directly. 1-q would lose the small vacancy for a
    # perfectly valid, nearly full particle; exp(-logaddexp) also retains the
    # subnormal low-occupancy tail without exp(+large) overflow.
    return (
        tuple(exp(-float(np.logaddexp(0.0, a))) for a in arguments),
        tuple(exp(-float(np.logaddexp(0.0, -a))) for a in arguments),
    )


def _response(
    rows: ReactionRows, potential: float, factor: float,
) -> tuple[float, float, float]:
    fractions, vacancies = _fractions_and_vacancies(rows, potential, factor)
    x = fsum(row[0] * q for row, q in zip(rows, fractions, strict=True))
    first = -fsum(
        row[0] * factor / row[2] * q * v
        for row, q, v in zip(rows, fractions, vacancies, strict=True)
    )
    second = fsum(
        row[0] * (factor / row[2]) ** 2 * q * v * (v - q)
        for row, q, v in zip(rows, fractions, vacancies, strict=True)
    )
    if not all(isfinite(v) for v in (x, first, second)):
        raise ValueError("MSMR occupancy derivatives are not representable")
    return x, first, second


@lru_cache(maxsize=32768)
def _inverse(rows: ReactionRows, fraction: float, factor: float) -> float:
    if not isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("MSMR inverse needs a strictly interior total stoichiometry")
    total_weight = fsum(row[0] for row in rows)
    if fraction >= total_weight:
        raise ValueError("MSMR stoichiometry is outside the authored total occupancy")
    # The weights are not modified. The actual tiny difference from one in a
    # representable authored sum must not produce a fictitious root at a tail.
    vacancy = total_weight - fraction
    logits = log(vacancy) - log(fraction)
    candidates = tuple(u0 + width / factor * logits for _, u0, width, _, _ in rows)
    if len(rows) == 1:
        return candidates[0]
    lower, upper = min(candidates), max(candidates)
    if lower == upper:
        return lower
    # Each individual occupancy ratio equals fraction/total_weight at its
    # candidate U. The unchanged weighted sum therefore brackets fraction.
    guess = fsum(row[0] * u for row, u in zip(rows, candidates, strict=True)) / total_weight
    use_vacancy = fraction > 0.5 * total_weight
    target_log = log(vacancy if use_vacancy else fraction)

    def tail_response(potential: float) -> tuple[float, float]:
        args = tuple(factor * (potential - u0) / width for _, u0, width, _, _ in rows)
        terms = np.array(
            [
                log(row[0]) - float(np.logaddexp(0.0, -a if use_vacancy else a))
                for row, a in zip(rows, args, strict=True)
            ]
        )
        tail_log = float(logsumexp(terms))
        fractions, vacancies = _fractions_and_vacancies(rows, potential, factor)
        slope = fsum(
            exp(term - tail_log) * factor / row[2] * (q if use_vacancy else -v)
            for term, row, q, v in zip(terms, rows, fractions, vacancies, strict=True)
        )
        if not isfinite(slope) or slope == 0:
            raise ValueError("MSMR inverse derivative is not representable")
        return tail_log - target_log, slope

    result = solve_root(
        lambda y: [tail_response(y[0])[0]],
        [guess],
        jacobian=lambda y: [[tail_response(y[0])[1]]],
        variables=(ResidualVariable("msmr:potential", "V", 1.0, "algebraic", lower, upper),),
        residual_scales=(1.0,),
        policy=RootPolicy(tolerance=2e-14, maximum_iterations=60),
    )
    if not result.success:
        raise ResidualNumericalError("MSMR monotone inverse:" + result.failure_reason)
    return float(result.value[0])


@dataclass(frozen=True)
class MSMRMaterialSpec:
    parameters: ScientificTable
    reaction_row_ids: tuple[str, ...]
    material_id: str
    reference_electrolyte_concentration_mol_m3: float = 1000.0
    _rows: ReactionRows = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _text(self.material_id, "MSMR material_id")
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("MSMR requires a sourced parameter table")
        identities = tuple(self.reaction_row_ids)
        if not 1 <= len(identities) <= 2 or len(set(identities)) != len(identities):
            raise ValueError("MSMR first block requires one or two distinct reaction rows")
        rows = {row.row_id: row for row in self.parameters.rows}
        native = []
        for identity in identities:
            if identity not in rows:
                raise ValueError("unknown MSMR reaction row")
            values = rows[identity].values
            if values.keys() != MSMR_UNITS.keys() or any(
                v.quality != "valid" or v.value is None for v in values.values()
            ):
                raise ValueError("MSMR row needs exactly five valid constitutive parameters")
            row = cast(ReactionParameters, tuple(
                convert_value(cast(float, values[k].value), values[k].unit, unit)
                for k, unit in MSMR_UNITS.items()
            ))
            weight, _, width, exchange, alpha = row
            if (
                any(not isfinite(v) for v in row)
                or not 0 < weight <= 1
                or width <= 0
                or exchange <= 0
                or not 0 < alpha < 1
            ):
                raise ValueError("invalid MSMR capacity weight, width or kinetics")
            native.append(row)
        if not isclose(fsum(row[0] for row in native), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("MSMR capacity weights must sum to one; no normalization is applied")
        ce = self.reference_electrolyte_concentration_mol_m3
        if isinstance(ce, bool) or not isfinite(ce) or ce <= 0:
            raise ValueError("MSMR reference electrolyte concentration must be positive")
        object.__setattr__(self, "reaction_row_ids", identities)
        object.__setattr__(self, "_rows", tuple(native))

    @property
    def reaction_parameters(self) -> ReactionRows:
        return self._rows

    def occupancies(self, potential_V: float, temperature_K: float) -> tuple[float, ...]:
        q = _fractions(self._rows, potential_V, _temperature(temperature_K))
        return tuple(row[0] * v for row, v in zip(self._rows, q, strict=True))

    def stoichiometry(self, potential_V: float, temperature_K: float) -> float:
        return self.stoichiometry_derivatives(potential_V, temperature_K)[0]

    def stoichiometry_derivatives(
        self, potential_V: float, temperature_K: float,
    ) -> tuple[float, float, float]:
        return _response(self._rows, potential_V, _temperature(temperature_K))

    def potential(self, fraction: float, temperature_K: float) -> float:
        if isinstance(fraction, bool):
            raise ValueError("MSMR stoichiometry cannot be boolean")
        return _inverse(self._rows, float(fraction), _temperature(temperature_K))

    def exchange_current_densities(
        self, potential_V: float, temperature_K: float, electrolyte_concentration: float,
    ) -> tuple[float, ...]:
        factor = _temperature(temperature_K)
        ce = electrolyte_concentration
        if isinstance(ce, bool) or not isfinite(ce) or ce <= 0:
            raise ValueError("MSMR electrolyte concentration must be positive and finite")
        result = []
        for weight, u0, width, exchange, alpha in self._rows:
            argument = factor * (potential_V - u0) / width
            # log(x_j) remains finite even where the occupancy itself underflows.
            log_x = log(weight) - float(np.logaddexp(0.0, argument))
            log_j0 = (
                log(exchange)
                + width * log_x
                + factor * (1 - alpha) * (potential_V - u0)
                + (1 - alpha) * log(ce / self.reference_electrolyte_concentration_mol_m3)
            )
            value = exp(log_j0)
            if not isfinite(value) or value <= 0:
                raise ValueError("MSMR exchange density is outside representable positive range")
            result.append(value)
        return tuple(result)

    def reaction_currents(
        self, potential_V: float, overpotential_V: float, temperature_K: float,
        electrolyte_concentration: float,
    ) -> tuple[float, ...]:
        factor = _temperature(temperature_K)
        if not isfinite(overpotential_V):
            raise ValueError("MSMR overpotential must be finite")
        j0s = self.exchange_current_densities(potential_V, temperature_K, electrolyte_concentration)
        return tuple(
            j0
            * (
                exp((1 - row[4]) * factor * overpotential_V)
                - exp(-row[4] * factor * overpotential_V)
            )
            for row, j0 in zip(self._rows, j0s, strict=True)
        )

    def kinetic_response(
        self, potential_V: float, applied_current_density: float, temperature_K: float,
        electrolyte_concentration: float,
    ) -> tuple[float, tuple[float, ...], float, float]:
        factor = _temperature(temperature_K)
        target = applied_current_density
        if not isfinite(target):
            raise ValueError("MSMR applied current density must be finite")
        j0s = self.exchange_current_densities(potential_V, temperature_K, electrolyte_concentration)
        fractions = _fractions(self._rows, potential_V, factor)

        def response(eta: float) -> tuple[tuple[float, ...], float, float]:
            currents, slopes, potential_slopes = [], [], []
            for row, j0, q in zip(self._rows, j0s, fractions, strict=True):
                alpha = row[4]
                ep, em = exp((1 - alpha) * factor * eta), exp(-alpha * factor * eta)
                current = j0 * (ep - em)
                currents.append(current)
                slopes.append(j0 * factor * ((1 - alpha) * ep + alpha * em))
                potential_slopes.append(factor * (q - alpha) * current)
            return tuple(currents), fsum(slopes), fsum(potential_slopes)

        eta = 0.0
        if target:
            # Strictly increasing sum(j_j), with a finite analytic bracket.
            exponent = min(1 - row[4] if target > 0 else row[4] for row in self._rows)
            bound = log1p(abs(target) / fsum(j0s)) / (factor * exponent)
            lower, upper = (0.0, bound) if target > 0 else (-bound, 0.0)
            scale = max(fsum(j0s), abs(target))
            answer = solve_root(
                lambda y: [fsum(response(y[0])[0]) - target],
                [min(upper, max(lower, target / (factor * fsum(j0s))))],
                jacobian=lambda y: [[response(y[0])[1]]],
                variables=(ResidualVariable("msmr:eta", "V", 1.0, "algebraic", lower, upper),),
                residual_scales=(scale,),
                policy=RootPolicy(tolerance=2e-13),
            )
            if not answer.success:
                raise ResidualNumericalError("MSMR shared overpotential:" + answer.failure_reason)
            eta = float(answer.value[0])
        currents, slope, potential_slope = response(eta)
        if not isfinite(slope) or slope <= 0 or not isfinite(potential_slope):
            raise ValueError("MSMR kinetic derivative is not representable")
        inverse_slope, potential_derivative = 1 / slope, -potential_slope / slope
        if not all(isfinite(v) for v in (*currents, inverse_slope, potential_derivative)):
            raise ValueError("MSMR inverse kinetic derivative is not representable")
        return eta, currents, inverse_slope, potential_derivative

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": MSMR_MATERIAL_VERSION,
            "parameters": self.parameters.to_dict(),
            "reaction_row_ids": list(self.reaction_row_ids),
            "material_id": self.material_id,
            "reference_electrolyte_concentration_mol_m3": (
                self.reference_electrolyte_concentration_mol_m3
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _keys(
            value,
            (
                "model_version",
                "parameters",
                "reaction_row_ids",
                "material_id",
                "reference_electrolyte_concentration_mol_m3",
            ),
        )
        if value["model_version"] != MSMR_MATERIAL_VERSION:
            raise ValueError("unsupported MSMR material version")
        return cls(
            ScientificTable.from_dict(value["parameters"]),
            tuple(value["reaction_row_ids"]),
            value["material_id"],
            value["reference_electrolyte_concentration_mol_m3"],
        )
