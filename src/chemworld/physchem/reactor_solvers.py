"""Shared numerical helpers for zero-dimensional reactor models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from chemworld.physchem.reaction_network import ReactionNetworkSpec, ReactionSpec
from chemworld.physchem.reactor_shared import (
    HeatTransferSpec,
    PressureBoundarySpec,
    ReactorResult,
    ReactorState,
    ReactorValidityDomain,
    SemiBatchFeedSpec,
)
from chemworld.physchem.solver_backend import (
    DEFAULT_REACTOR_ODE_POLICY,
    ODESolveReport,
    solve_ode,
)
from chemworld.physchem.thermochemistry import (
    NASA7SpeciesThermo,
    reaction_thermochemistry,
)


def _integrate_result(
    *,
    model_id: str,
    reactor_id: str,
    network: ReactionNetworkSpec,
    initial_amounts_mol: Mapping[str, float],
    initial_volume_L: float,
    initial_temperature_K: float,
    y0: np.ndarray,
    duration_s: float,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    volume_getter: Callable[[float, np.ndarray], float],
    evaluation_times_s: Sequence[float] | None,
    temperature_index: int,
    energy_jacket_index: int,
    heat_reaction_index: int,
    heat_loss_index: int,
    material_in_slice: slice | None = None,
    material_out_slice: slice | None = None,
    pressure_boundary: PressureBoundarySpec | None = None,
    validity_domain: ReactorValidityDomain | None = None,
) -> ReactorResult:
    _validate_initial_state(
        network,
        initial_amounts_mol,
        volume_L=initial_volume_L,
        temperature_K=initial_temperature_K,
    )
    result = _solve(
        rhs,
        y0,
        duration_s=duration_s,
        evaluation_times_s=evaluation_times_s,
    )
    amounts = {
        species_id: tuple(_validated_amount(float(value), species_id) for value in result.y[idx])
        for idx, species_id in enumerate(network.species_ids)
    }
    final_amounts = {species_id: values[-1] for species_id, values in amounts.items()}
    final_y = result.y[:, -1]
    volumes = tuple(
        float(volume_getter(float(time_s), result.y[:, idx]))
        for idx, time_s in enumerate(result.t)
    )
    if any(volume <= 0 for volume in volumes):
        raise RuntimeError("Reactor integration crossed a nonpositive-volume boundary")
    final_volume = volumes[-1]
    material_in = _ledger_from_slice(network, final_y, material_in_slice)
    material_out = _ledger_from_slice(network, final_y, material_out_slice)
    initial_state = ReactorState(
        amounts_mol={
            species_id: float(initial_amounts_mol.get(species_id, 0.0))
            for species_id in network.species_ids
        },
        volume_L=initial_volume_L,
        temperature_K=initial_temperature_K,
        time_s=0.0,
        pressure_Pa=(pressure_boundary.pressure_Pa if pressure_boundary is not None else None),
    )
    final_state = ReactorState(
        amounts_mol=final_amounts,
        volume_L=max(final_volume, 1e-12),
        temperature_K=float(final_y[temperature_index]),
        time_s=duration_s,
        pressure_Pa=(pressure_boundary.pressure_Pa if pressure_boundary is not None else None),
        energy_jacket_J=float(final_y[energy_jacket_index]),
        heat_reaction_J=float(final_y[heat_reaction_index]),
        heat_loss_J=float(final_y[heat_loss_index]),
        material_in_mol=material_in,
        material_out_mol=material_out,
    )
    return ReactorResult(
        reactor_id=reactor_id,
        model_id=model_id,
        network_id=network.network_id,
        initial_state=initial_state,
        final_state=final_state,
        times_s=tuple(float(value) for value in result.t),
        amounts_mol=amounts,
        temperatures_K=tuple(float(value) for value in result.y[temperature_index]),
        material_balance_error_mol=_material_balance_error(
            network,
            initial_state,
            final_state,
        ),
        volumes_L=volumes,
        pressures_Pa=(
            tuple(pressure_boundary.pressure_Pa for _ in result.t)
            if pressure_boundary is not None
            else ()
        ),
        validity_domain=validity_domain,
        metadata={
            "solver_diagnostic": result.diagnostic.to_dict(),
            "pressure_boundary": (
                {
                    "mode": pressure_boundary.mode,
                    "pressure_Pa": pressure_boundary.pressure_Pa,
                }
                if pressure_boundary is not None
                else None
            ),
        },
    )


def _solve(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    *,
    duration_s: float,
    evaluation_times_s: Sequence[float] | None,
) -> ODESolveReport:
    result = solve_ode(
        rhs,
        y0,
        time_span_s=(0.0, duration_s),
        evaluation_times_s=evaluation_times_s,
        policy=DEFAULT_REACTOR_ODE_POLICY,
    )
    result.raise_for_failure("Reactor integration")
    return result


def _solve_interval(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    evaluation_times_s: Sequence[float],
) -> ODESolveReport:
    result = solve_ode(
        rhs,
        y0,
        time_span_s=(start_s, end_s),
        evaluation_times_s=evaluation_times_s,
        policy=DEFAULT_REACTOR_ODE_POLICY,
    )
    result.raise_for_failure("Reactor integration")
    return result


def _segment_evaluation_times(
    start_s: float,
    end_s: float,
    requested_times_s: Sequence[float] | None,
) -> tuple[float, ...]:
    if requested_times_s is None:
        return (start_s, end_s) if end_s > start_s else (start_s,)
    selected = [
        float(time_s) for time_s in requested_times_s if start_s - 1e-12 <= time_s <= end_s + 1e-12
    ]
    selected.extend([start_s, end_s])
    return tuple(sorted(set(selected)))


def _amount_vector(
    network: ReactionNetworkSpec,
    amounts_mol: Mapping[str, float],
) -> tuple[float, ...]:
    unknown = sorted(set(amounts_mol) - set(network.species_ids))
    if unknown:
        raise ValueError(f"initial state contains unknown species: {unknown}")
    values = tuple(float(amounts_mol.get(species_id, 0.0)) for species_id in network.species_ids)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("initial species amounts must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("initial species amounts cannot be negative")
    return values


def _amounts_from_vector(
    network: ReactionNetworkSpec,
    values: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    return {
        species_id: _rhs_amount(float(value), species_id)
        for species_id, value in zip(network.species_ids, values, strict=True)
    }


def _reaction_heat_w(
    network: ReactionNetworkSpec,
    amounts_mol: Mapping[str, float],
    *,
    volume_L: float,
    temperature_K: float,
    species_thermo: Mapping[str, NASA7SpeciesThermo] | None = None,
) -> float:
    rates = network.reaction_rates(
        amounts_mol,
        volume_L=volume_L,
        temperature_K=temperature_K,
        species_thermo=species_thermo,
    )
    return sum(
        _reaction_enthalpy_j_mol(
            reaction,
            temperature_K=temperature_K,
            species_thermo=species_thermo,
        )
        * rates[reaction.reaction_id]
        * volume_L
        for reaction in network.reactions
    )


def _reaction_enthalpy_j_mol(
    reaction: ReactionSpec,
    *,
    temperature_K: float,
    species_thermo: Mapping[str, NASA7SpeciesThermo] | None,
) -> float:
    if species_thermo is None:
        return reaction.delta_h_J_per_mol
    return reaction_thermochemistry(
        reaction_id=reaction.reaction_id,
        stoichiometry=reaction.stoichiometry,
        species_thermo=species_thermo,
        temperature_K=temperature_K,
    ).delta_h_J_mol


def _jacket_heat_w(
    thermal: HeatTransferSpec,
    *,
    temperature_K: float,
    jacket_temperature_K: float | None,
) -> float:
    jacket = 0.0
    if jacket_temperature_K is not None:
        jacket = thermal.jacket_ua_W_per_K * (jacket_temperature_K - temperature_K)
    return thermal.fixed_heat_W + thermal.limit_jacket_heat_w(jacket)


def _feed_heat_w(
    feeds: tuple[SemiBatchFeedSpec, ...],
    time_s: float,
    temperature_K: float,
    thermal: HeatTransferSpec,
) -> float:
    return sum(
        thermal.rho_cp_J_per_L_K
        * feed.volumetric_flow(time_s)
        * (feed.stream.temperature_K - temperature_K)
        for feed in feeds
    )


def _ledger_from_slice(
    network: ReactionNetworkSpec,
    y: np.ndarray,
    ledger_slice: slice | None,
) -> dict[str, float]:
    if ledger_slice is None:
        return {}
    values = y[ledger_slice]
    return {
        species_id: _validated_amount(float(value), species_id)
        for species_id, value in zip(network.species_ids, values, strict=True)
    }


def _element_totals(
    network: ReactionNetworkSpec,
    amounts_mol: Mapping[str, float],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for species in network.species:
        amount = float(amounts_mol.get(species.species_id, 0.0))
        for element, count in species.composition.items():
            totals[element] = totals.get(element, 0.0) + count * amount
    return totals


def _material_balance_error(
    network: ReactionNetworkSpec,
    initial: ReactorState,
    final: ReactorState,
) -> float:
    left = _element_totals(network, initial.amounts_mol)
    _add_element_totals(left, network, final.material_in_mol, sign=1.0)
    right = _element_totals(network, final.amounts_mol)
    _add_element_totals(right, network, final.material_out_mol, sign=1.0)
    elements = set(left) | set(right)
    return max(
        (abs(left.get(element, 0.0) - right.get(element, 0.0)) for element in elements),
        default=0.0,
    )


def _add_element_totals(
    totals: dict[str, float],
    network: ReactionNetworkSpec,
    amounts_mol: Mapping[str, float],
    *,
    sign: float,
) -> None:
    increment = _element_totals(network, amounts_mol)
    for element, value in increment.items():
        totals[element] = totals.get(element, 0.0) + sign * value


def _validate_initial_state(
    network: ReactionNetworkSpec,
    amounts_mol: Mapping[str, float],
    *,
    volume_L: float,
    temperature_K: float,
) -> None:
    if volume_L <= 0:
        raise ValueError("volume_L must be positive")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    unknown = sorted(set(amounts_mol) - set(network.species_ids))
    if unknown:
        raise ValueError(f"initial state contains unknown species: {unknown}")
    values = [float(value) for value in amounts_mol.values()]
    if any(not np.isfinite(value) for value in values):
        raise ValueError("initial species amounts must be finite")
    if any(value < 0 for value in values):
        raise ValueError("initial species amounts cannot be negative")


def _rhs_amount(value: float, species_id: str) -> float:
    if not np.isfinite(value):
        raise RuntimeError(f"nonfinite amount encountered for species {species_id}")
    # Adaptive ODE methods can overshoot a positive boundary by roundoff.  A
    # larger excursion is a failed physical integration, not a value to hide.
    if value < -1.0e-8:
        raise RuntimeError(f"negative amount encountered for species {species_id}: {value}")
    return max(value, 0.0)


def _validated_amount(value: float, species_id: str) -> float:
    if not np.isfinite(value):
        raise RuntimeError(f"nonfinite amount encountered for species {species_id}")
    if value < -1.0e-8:
        raise RuntimeError(f"negative amount encountered for species {species_id}: {value}")
    return max(value, 0.0)


__all__ = [
    "_amount_vector",
    "_amounts_from_vector",
    "_feed_heat_w",
    "_integrate_result",
    "_jacket_heat_w",
    "_material_balance_error",
    "_reaction_heat_w",
    "_segment_evaluation_times",
    "_solve",
    "_solve_interval",
]
