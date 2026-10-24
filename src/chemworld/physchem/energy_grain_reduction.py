"""Read-only single-well QSD rates at fixed temperature and actual bath density.

These are finite-generator slow modes, not a dynamic reduced reactor. A QSD
initial distribution has exact exponential survival only while its generator
stays fixed. General initial populations retain a prompt transient; their
long-time survival amplitude is the left mode projected onto that population.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum, isclose
from typing import Any

import numpy as np

from chemworld.physchem.energy_grain_master_equation import (
    boltzmann_grain_fractions,
    energy_grain_generator,
)
from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec, R, finite


def _baths(spec: EnergyGrainMasterEquationSpec, values: Mapping[str, float]) -> dict[str, float]:
    names = tuple(b.species_id for b in spec.bath_gases)
    if set(values) - set(names):
        raise ValueError("QSD conditions contain undeclared bath species")
    result = {}
    for name in names:
        value = values.get(name, 0.0)
        finite(value, "bath concentration")
        if value < 0:
            raise ValueError("bath concentration cannot be negative")
        result[name] = float(value)
    return result


def energy_grain_qsd(
    spec: EnergyGrainMasterEquationSpec,
    *,
    temperature_K: float,
    bath_concentrations_mol_L: Mapping[str, float],
    grain_amounts_mol: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Resolve the positive, simple slow mode of one absorbing finite well.

    Spectral separation is reported without an invented applicability cutoff.
    Numerical resolution checks only reject unresolvable eigensystems. No
    inventory, heat, clock, kinetic parameter or persistent state is changed.
    """
    finite(temperature_K, "temperature_K", positive=True)
    if len(spec.wells) != 1 or len(spec.wells[0].grains) < 2:
        raise ValueError("QSD analysis requires exactly one well with at least two grains")
    well = spec.wells[0]
    names = tuple(g.species_id for g in well.grains)
    products = tuple(s.species_id for s in spec.sinks)
    if not spec.channels or any(c.target_species_id not in products for c in spec.channels):
        raise ValueError("QSD analysis requires absorbing product channels only")
    baths = _baths(spec, bath_concentrations_mol_L)
    full = np.asarray(
        energy_grain_generator(spec, temperature_K=temperature_K, bath_concentrations_mol_L=baths)[
            "generator_s_inverse"
        ],
        dtype=float,
    )
    count = len(names)
    transient, exits = full[:count, :count], full[count:, :count]
    # Connectivity is checked on represented positive transitions, without
    # adding tiny rates or populations when Boltzmann factors underflow.
    reached = {0}
    while True:
        extended = reached | {
            i for j in reached for i in range(count) if i != j and transient[i, j] > 0
        }
        if extended == reached:
            break
        reached = extended
    equilibrium = np.asarray(list(boltzmann_grain_fractions(well, temperature_K).values()))
    if len(reached) != count or np.any(equilibrium <= 0):
        raise ValueError(
            "QSD requires a connected collision graph and represented Boltzmann support"
        )
    root = np.sqrt(equilibrium)
    symmetric = -transient * root[np.newaxis, :] / root[:, np.newaxis]
    scale = float(np.linalg.norm(symmetric, ord=np.inf))
    resolution = 128 * count * np.finfo(float).eps * scale
    if not np.all(np.isfinite(symmetric)) or not np.allclose(
        symmetric, symmetric.T, rtol=1e-11, atol=resolution
    ):
        raise ValueError("QSD collision generator is not numerically reversible")
    eigenvalues, modes = np.linalg.eigh((symmetric + symmetric.T) / 2)
    mode = modes[:, 0]
    if float(np.sum(mode)) < 0:
        mode = -mode
    if np.any(mode <= 0):
        raise ValueError("QSD positive slow mode is outside numerical resolution")
    normalization = float(root @ mode)
    fractions = root * mode / normalization
    left = mode * normalization / root
    product_rates = exits @ fractions
    rate = float(fsum(product_rates))
    second = float(eigenvalues[1])
    gap = second - rate
    if rate <= resolution or gap <= resolution or abs(float(eigenvalues[0]) - rate) > resolution:
        raise ValueError("QSD decay or simple spectral gap is outside numerical resolution")
    residual = float(np.max(np.abs(transient @ fractions + rate * fractions)))
    left_residual = float(np.max(np.abs(left @ transient + rate * left)))
    if residual > resolution or left_residual > resolution * max(1.0, float(np.max(left))):
        raise ValueError("QSD eigenmode residual exceeds numerical resolution")
    initial = None
    if grain_amounts_mol is not None:
        if set(grain_amounts_mol) - set(names):
            raise ValueError("initial QSD query accepts original well grains only")
        for value in grain_amounts_mol.values():
            finite(value, "grain amount")
            if value < 0:
                raise ValueError("grain amount cannot be negative")
        amounts = np.asarray([grain_amounts_mol.get(name, 0.0) for name in names])
        total = float(fsum(amounts))
        finite(total, "total grain amount")
        distribution = amounts / total if total > 0 else None
        with np.errstate(over="ignore", invalid="ignore"):
            amplitude = float(left @ amounts)
            initial_fluxes = exits @ amounts
        finite(amplitude, "projected slow-mode amplitude")
        for value in initial_fluxes:
            finite(float(value), "initial product flux")
        initial = {
            "well_amount_mol": total,
            "fractions": None
            if distribution is None
            else dict(zip(names, distribution.tolist(), strict=True)),
            "l1_distance_to_qsd": None
            if distribution is None
            else float(np.sum(np.abs(distribution - fractions))),
            "slow_mode_amplitude_mol": amplitude,
            "initial_product_fluxes_mol_s": dict(
                zip(products, initial_fluxes.tolist(), strict=True)
            ),
            "interpretation": (
                "l dot n is a long-time survival amplitude, not a new inventory; "
                "prompt loss and transient product branching remain outside this mode"
            ),
        }
    return {
        "model": "fixed-condition-energy-grain-qsd-v1",
        "network_id": spec.network_id,
        "well_id": well.well_id,
        "temperature_K": float(temperature_K),
        "bath_concentrations_mol_L": baths,
        "bath_partial_pressures_Pa": {k: v * 1000 * R * temperature_K for k, v in baths.items()},
        "bath_pressure_Pa": fsum(baths.values()) * 1000 * R * temperature_K,
        "pressure_basis": "sum of declared bath partial pressures; not total gas pressure",
        "qsd_fractions": dict(zip(names, fractions.tolist(), strict=True)),
        "left_mode_weights": dict(zip(names, left.tolist(), strict=True)),
        "normalization": {"sum_q": float(np.sum(fractions)), "left_dot_q": float(left @ fractions)},
        "effective_rate_s_inverse": rate,
        "product_rates_s_inverse": dict(zip(products, product_rates.tolist(), strict=True)),
        "qsd_product_branching": dict(zip(products, (product_rates / rate).tolist(), strict=True)),
        "second_decay_rate_s_inverse": second,
        "spectral_gap_s_inverse": gap,
        "spectral_gap_over_decay": gap / rate,
        "conditional_relaxation_time_s": 1 / gap,
        "lifetime_s": 1 / rate,
        "right_residual_s_inverse": residual,
        "left_residual_s_inverse": left_residual,
        "numerical_resolution_s_inverse": resolution,
        "current_population": initial,
        "validity": {
            "numerically_resolved": True,
            "fixed_temperature_and_bath_required": True,
            "exact_exponential_requires_qsd_initial_distribution": True,
            "dynamic_applicability_assessed": False,
            "spectrum": "single well, simple positive slow mode, absorbing same-formula products",
            "grid_scope": "finite declared grains; no continuum or sparse-DOS convergence claim",
            "thermochemistry": "original grain thermochemistry unchanged; no effective heat model",
            "initial_transient": "arbitrary initial distributions are not replaced by QSD",
        },
        "world_advanced": False,
    }


def energy_grain_qsd_table(
    spec: EnergyGrainMasterEquationSpec,
    *,
    temperatures_K: Sequence[float],
    bath_pressures_Pa: Sequence[float],
    bath_mole_fractions: Mapping[str, float],
) -> dict[str, Any]:
    """Tabulate authored fixed conditions; neither fit nor interpolate PLog.

    Every requested cell is retained, including its explicit numerical or
    unsupported-boundary failure. Fractions refer only to declared bath gases.
    """
    temperatures, pressures = tuple(temperatures_K), tuple(bath_pressures_Pa)
    if not temperatures or not pressures:
        raise ValueError("QSD table requires nonempty temperature and bath-pressure grids")
    for value in (*temperatures, *pressures):
        finite(value, "QSD grid coordinate", positive=True)
    fractions = _baths(spec, bath_mole_fractions)
    if not isclose(fsum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("declared bath mole fractions must sum to one")
    rows = []
    for temperature in temperatures:
        for pressure in pressures:
            row: dict[str, Any] = {"temperature_K": temperature, "bath_pressure_Pa": pressure}
            try:
                row.update(
                    status="resolved",
                    report=energy_grain_qsd(
                        spec,
                        temperature_K=temperature,
                        bath_concentrations_mol_L={
                            k: pressure * x / (1000 * R * temperature) for k, x in fractions.items()
                        },
                    ),
                )
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            rows.append(row)
    return {
        "model": "fixed-condition-energy-grain-qsd-table-v1",
        "network_id": spec.network_id,
        "bath_mole_fractions": fractions,
        "pressure_basis": "declared bath partial pressure, not total gas pressure",
        "requested_cells": len(rows),
        "resolved_cells": sum(row["status"] == "resolved" for row in rows),
        "rows": rows,
        "interpolation_or_dynamic_rate_law_created": False,
    }
