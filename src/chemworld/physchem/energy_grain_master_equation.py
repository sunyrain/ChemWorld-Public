"""Detailed-balance grain collision kernels consumed by the original rate laws.

For a finite well, q_j = g_j / sum(g). A downward off-diagonal collision
probability is q_j exp(-abs(Ej-Ei)/alpha). Its upward partner follows detailed
balance at pi_i proportional to g_i exp(-Ei/RT); the diagonal is the remaining
self-collision probability. This explicitly declared finite exponential-down
convention does not claim equivalence to every RMG energy-transfer model.

No ODE solver, advancing state, or persistent population is defined here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from math import exp, fsum, isfinite, log
from typing import Any

from chemworld.physchem.energy_grain_specs import (
    EnergyGrainMasterEquationSpec,
    EnergyGrainWellSpec,
    EnergyTransferBath,
    R,
    finite,
)
from chemworld.physchem.reaction_network_specs import ReactionSpec


def _temperature(value: float) -> None:
    finite(value, "temperature_K", positive=True)


def _scaled_bath(bath: EnergyTransferBath, temperature_K: float) -> tuple[float, float]:
    ratio = temperature_K / bath.reference_temperature_K
    try:
        coefficient = bath.collision_coefficient_L_mol_s * ratio**bath.collision_exponent
        alpha = bath.alpha_J_mol * ratio**bath.alpha_exponent
    except OverflowError as exc:
        raise ValueError("energy-transfer law exceeds floating-point domain") from exc
    finite(coefficient, "collision coefficient at T", positive=True)
    finite(alpha, "alpha at T", positive=True)
    return coefficient, alpha


@lru_cache(maxsize=256)
def collision_transition_probabilities(
    well: EnergyGrainWellSpec, bath: EnergyTransferBath, temperature_K: float
) -> tuple[tuple[float, ...], ...]:
    """P[target][source], including self collisions, at the current temperature."""
    _temperature(temperature_K)
    if bath not in well.baths:
        raise ValueError("bath collision law must belong to this well")
    _, alpha = _scaled_bath(bath, temperature_K)
    grains = well.grains
    # Scale before summing so arbitrary common DOS normalization does not overflow.
    scale = max(g.statistical_weight for g in grains)
    weights = [g.statistical_weight / scale for g in grains]
    total = fsum(weights)
    matrix = [[0.0] * len(grains) for _ in grains]
    for source, origin in enumerate(grains):
        for target, destination in enumerate(grains):
            if source == target:
                continue
            gap = destination.energy_J_mol - origin.energy_J_mol
            matrix[target][source] = (
                weights[target]
                / total
                * exp(-abs(gap) / alpha - max(gap, 0.0) / (R * temperature_K))
            )
        matrix[source][source] = 1.0 - fsum(row[source] for row in matrix)
        if matrix[source][source] < 0:
            raise ArithmeticError("negative self-collision probability")
    return tuple(tuple(row) for row in matrix)


def boltzmann_grain_fractions(well: EnergyGrainWellSpec, temperature_K: float) -> dict[str, float]:
    _temperature(temperature_K)
    minimum = min(g.energy_J_mol for g in well.grains)
    logs = [
        log(g.statistical_weight) - (g.energy_J_mol - minimum) / (R * temperature_K)
        for g in well.grains
    ]
    offset = max(logs)
    weights = [exp(value - offset) for value in logs]
    total = fsum(weights)
    return {g.species_id: weight / total for g, weight in zip(well.grains, weights, strict=True)}


def well_collision_generator(
    well: EnergyGrainWellSpec, *, temperature_K: float, concentrations_mol_L: Mapping[str, float]
) -> tuple[tuple[float, ...], ...]:
    """Q[target][source] in 1/s; missing declared baths are zero inventory."""
    _temperature(temperature_K)
    matrix = [[0.0] * len(well.grains) for _ in well.grains]
    for bath in well.baths:
        concentration = float(concentrations_mol_L.get(bath.bath_species_id, 0.0))
        finite(concentration, "bath concentration")
        if concentration < 0:
            raise ValueError("bath concentration cannot be negative")
        coefficient, _ = _scaled_bath(bath, temperature_K)
        frequency = coefficient * concentration
        finite(frequency, "collision frequency")
        probability = collision_transition_probabilities(well, bath, temperature_K)
        for source in range(len(well.grains)):
            for target in range(len(well.grains)):
                if source != target:
                    matrix[target][source] += frequency * probability[target][source]
    for source in range(len(well.grains)):
        matrix[source][source] = -fsum(row[source] for row in matrix)
    if any(not isfinite(x) for row in matrix for x in row):
        raise ValueError("combined collision rate exceeds floating-point domain")
    return tuple(tuple(row) for row in matrix)


@lru_cache(maxsize=128)
def _well_from_json(value: str) -> EnergyGrainWellSpec:
    return EnergyGrainWellSpec.from_dict(json.loads(value))


def evaluate_energy_grain_collision_rate(
    reaction: ReactionSpec,
    *,
    concentrations_mol_L: Mapping[str, float],
    temperature_K: float,
    activity_coefficients: Mapping[str, float] | None = None,
) -> float:
    """Original rate-law dispatcher entry, returning mol/(L s)."""
    parameters = reaction.rate_law.parameters
    if activity_coefficients is not None or reaction.rate_law.uses_activities:
        raise ValueError("grain collisions use ideal-gas concentrations, not activities")
    if parameters.get("model") != "energy-grain-collision-v1":
        raise ValueError("unknown energy-grain collision declaration")
    _temperature(temperature_K)
    lower, upper = parameters["temperature_range_K"]
    if not lower <= temperature_K <= upper:
        raise ValueError("temperature outside declared energy-grain domain")
    well = _well_from_json(json.dumps(parameters["well"], sort_keys=True, separators=(",", ":")))
    source, target = parameters["source_grain_id"], parameters["target_grain_id"]
    names = tuple(g.species_id for g in well.grains)
    if (
        source not in names
        or target not in names
        or source == target
        or reaction.reversible
        or reaction.stoichiometry != {source: -1.0, target: 1.0}
        or reaction.kinetic_forward_orders != {source: 1.0}
    ):
        raise ValueError("grain collision rate requires its declared irreversible 1:1 pathway")
    concentration = float(concentrations_mol_L.get(source, 0.0))
    finite(concentration, "grain concentration")
    if concentration < 0:
        raise ValueError("grain concentration cannot be negative")
    rate = (
        well_collision_generator(
            well, temperature_K=temperature_K, concentrations_mol_L=concentrations_mol_L
        )[names.index(target)][names.index(source)]
        * concentration
    )
    finite(rate, "grain reaction rate")
    return rate


def energy_grain_generator(
    spec: EnergyGrainMasterEquationSpec,
    *,
    temperature_K: float,
    bath_concentrations_mol_L: Mapping[str, float],
) -> dict[str, Any]:
    """Read-only full molecular generator at fixed bath concentrations.

    Returned matrices are diagnostic values, never an advancing population owner.
    A flowing bath requires the original network's actual instantaneous rates.
    """
    if not spec.baseline.min_temperature_K <= temperature_K <= spec.baseline.max_temperature_K:
        raise ValueError("temperature outside declared energy-grain domain")
    if set(bath_concentrations_mol_L) - {b.species_id for b in spec.bath_gases}:
        raise ValueError("unknown bath species")
    names = tuple(g.species_id for g in spec.molecular_states)
    index = {name: i for i, name in enumerate(names)}
    states = {g.species_id: g for g in spec.molecular_states}
    matrix = [[0.0] * len(names) for _ in names]
    for well in spec.wells:
        local = well_collision_generator(
            well, temperature_K=temperature_K, concentrations_mol_L=bath_concentrations_mol_L
        )
        for i, target in enumerate(well.grains):
            for j, source in enumerate(well.grains):
                matrix[index[target.species_id]][index[source.species_id]] += local[i][j]
    grain_ids = {g.species_id for w in spec.wells for g in w.grains}
    for channel in spec.channels:
        i, j = index[channel.source_grain_id], index[channel.target_species_id]
        matrix[j][i] += channel.rate_s
        matrix[i][i] -= channel.rate_s
        if channel.target_species_id in grain_ids:
            rate = (
                channel.rate_s
                * states[channel.source_grain_id].statistical_weight
                / (states[channel.target_species_id].statistical_weight)
            )
            finite(rate, "reverse microcanonical rate", positive=True)
            matrix[i][j] += rate
            matrix[j][j] -= rate
    return {
        "species_ids": names,
        "generator_s_inverse": tuple(tuple(row) for row in matrix),
        "temperature_K": temperature_K,
        "bath_concentrations_mol_L": dict(bath_concentrations_mol_L),
    }


def grain_population_report(
    spec: EnergyGrainMasterEquationSpec, amounts_mol: Mapping[str, float]
) -> dict[str, Any]:
    """Query the original inventory without creating coarse well species/stocks."""
    known = {s.species_id for s in spec.molecular_states} | {b.species_id for b in spec.bath_gases}
    if set(amounts_mol) - known:
        raise ValueError("population query contains undeclared species")
    for value in amounts_mol.values():
        finite(value, "amount_mol")
        if value < 0:
            raise ValueError("negative grain population")
    wells = {
        w.well_id: fsum(amounts_mol.get(g.species_id, 0) for g in w.grains) for w in spec.wells
    }
    sinks = {s.species_id: amounts_mol.get(s.species_id, 0) for s in spec.sinks}
    return {
        "well_amounts_mol": wells,
        "sink_amounts_mol": sinks,
        "molecular_amount_mol": fsum((*wells.values(), *sinks.values())),
        "grained_internal_energy_J": fsum(
            amounts_mol.get(g.species_id, 0) * g.energy_J_mol for g in spec.molecular_states
        ),
    }
