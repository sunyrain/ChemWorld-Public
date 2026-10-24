"""Whole-mixture diffusion in an explicit frame; no stocks, EOS or integrator.

N_i denotes diffusive molar flux relative to the selected average velocity.
First-order Stefan-Maxwell uses sum_j(x_j N_i - x_i N_j)/D_ij = -c d_i,
d_i = grad(x_i) + (x_i-y_i) grad(log P). No body force or Soret term is inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.gas_transport import GasTransportResponse, _frozen, _plain


@dataclass(frozen=True)
class GasTransportFluxResult:
    model: str
    frame: str
    molar_fluxes_mol_m2_s: Mapping[str, float]
    mass_fluxes_kg_m2_s: Mapping[str, float]
    driving_forces_per_m: Mapping[str, float]
    friction_residuals_per_m: Mapping[str, float] | None
    frame_residual: float
    frame_residual_unit: str
    fourier_heat_flux_W_m2: float
    diffusion_enthalpy_flux_W_m2: float
    total_heat_flux_W_m2: float

    def __post_init__(self) -> None:
        for name in (
            "molar_fluxes_mol_m2_s",
            "mass_fluxes_kg_m2_s",
            "driving_forces_per_m",
            "friction_residuals_per_m",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {name: _plain(value) for name, value in vars(self).items()}


def gas_transport_fluxes(
    response: GasTransportResponse,
    *,
    mole_fraction_gradients_per_m: Mapping[str, float],
    pressure_log_gradient_per_m: float = 0.0,
    temperature_gradient_K_m: float = 0.0,
    model: str,
    frame: str,
    thermal_diffusion: str,
) -> GasTransportFluxResult:
    """One Cartesian flux component, with mass or molar reference velocity.

    Mixture-average coefficients use the canonical mass-flux/mole-gradient
    approximation before correcting the whole vector and changing frames.
    Returned heat is Fourier plus species enthalpy transport exactly once,
    relative to the named frame. Bulk convective enthalpy remains with its
    existing material-flow consumer; it is not included in this diffusive flux.
    A consumer must not also add diffusion enthalpy as a separate heater term.
    """
    if model not in {"mixture-averaged", "stefan-maxwell"} or frame not in {"mass", "molar"}:
        raise ValueError("declare a supported whole-mixture flux model and reference frame")
    if thermal_diffusion != "off" or response.models.get("thermal_diffusion") != "off":
        raise ValueError("Soret transport is unsupported; declare thermal_diffusion='off'")
    ids = response.species_ids
    if set(mole_fraction_gradients_per_m) != set(ids):
        raise ValueError("composition gradient must contain exactly the original species")
    values = (
        *mole_fraction_gradients_per_m.values(),
        pressure_log_gradient_per_m,
        temperature_gradient_K_m,
    )
    if any(isinstance(v, bool) or not isfinite(v) for v in values):
        raise ValueError("transport gradients must be finite numbers")
    grad = np.asarray([mole_fraction_gradients_per_m[s] for s in ids])
    if abs(fsum(grad)) > 1e-12 + 1e-10 * float(np.max(np.abs(grad))):
        raise ValueError("mole-fraction gradients must sum to zero")
    x = np.asarray([response.mole_fractions[s] for s in ids])
    y = np.asarray([response.mass_fractions[s] for s in ids])
    masses = np.asarray([response.molecular_weights_kg_mol[s] for s in ids])
    d = grad + (x - y) * pressure_log_gradient_per_m
    count, density = len(ids), response.molar_density_mol_m3
    friction = np.zeros((count, count))
    for i, species in enumerate(ids):
        for j, target in enumerate(ids):
            if i != j:
                resistance = 1 / response.binary_diffusivities_m2_s[species][target]
                friction[i, i] += x[j] * resistance
                friction[i, j] -= x[i] * resistance
    weights = masses / float(x @ masses) if frame == "mass" else np.ones(count)
    flux: NDArray[np.float64]
    if count == 1:
        flux = np.zeros(1)
    elif model == "stefan-maxwell":
        scale = float(np.max(np.abs(friction)))
        # The augmented null-space constraint works at zero mole fractions;
        # no clipping or division by x_i and no artificial trace inventory.
        matrix = np.zeros((count + 1, count + 1))
        matrix[:count, :count] = friction / scale
        matrix[:count, count] = x
        matrix[count, :count] = weights
        rhs = np.concatenate((-d / scale, [0.0]))
        flux = np.linalg.solve(matrix, rhs)[:count] * density
    else:
        coefficients = response.diffusion_coefficients_m2_s["mass_mole"]
        if any(coefficients[s] is None and d[i] != 0 for i, s in enumerate(ids)):
            raise ValueError("pure-vertex mixture coefficient is directional; use stefan-maxwell")
        raw = -density * np.asarray([coefficients[s] or 0.0 for s in ids]) * d
        # One canonical approximation; the requested frame is a velocity change,
        # not a second independent approximation to the same physical flux.
        flux = raw - x * float(weights @ raw)
    mass_flux = masses * flux
    residual = friction @ (flux / density) + d if model == "stefan-maxwell" else None
    frame_residual = float(fsum(mass_flux) if frame == "mass" else fsum(flux))
    fourier = -response.conductivity_W_m_K * temperature_gradient_K_m
    enthalpy = fsum(
        response.partial_enthalpies_J_mol[s] * float(flux[i]) for i, s in enumerate(ids)
    )
    return GasTransportFluxResult(
        model,
        frame,
        dict(zip(ids, map(float, flux), strict=True)),
        dict(zip(ids, map(float, mass_flux), strict=True)),
        dict(zip(ids, map(float, d), strict=True)),
        None if residual is None else dict(zip(ids, map(float, residual), strict=True)),
        frame_residual,
        "kg/(m^2*s)" if frame == "mass" else "mol/(m^2*s)",
        fourier,
        enthalpy,
        fourier + enthalpy,
    )
