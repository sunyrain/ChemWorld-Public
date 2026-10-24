"""Ideal dilute, zero-current face fluxes; no independent ionic Fick evolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import fsum, isfinite, log
from typing import Any

from chemworld.physchem.property_reports import R_J_PER_MOL_K as R

FARADAY_C_MOL = 96485.33212


@dataclass(frozen=True)
class AqueousDiffusionCoefficient:
    species_id: str
    value_m2_s: float
    source_id: str
    temperature_bounds_K: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_bounds_K", tuple(self.temperature_bounds_K))
        if (
            not self.species_id
            or not self.source_id
            or self.value_m2_s <= 0
            or not isfinite(self.value_m2_s)
            or len(self.temperature_bounds_K) != 2
            or not 0 < self.temperature_bounds_K[0] < self.temperature_bounds_K[1]
            or not all(map(isfinite, self.temperature_bounds_K))
        ):
            raise ValueError("diffusivity requires positive m2/s, source and finite T domain")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def positive_log_mean(left: float, right: float) -> float:
    if not all(isfinite(v) and v > 0 for v in (left, right)):
        raise ValueError("electrodiffusion requires strictly positive face concentrations")
    mean = (left + right) / 2
    x = (right - left) / (left + right)
    if abs(x) < 1e-4:
        return mean * (1 - x * x / 3 - 4 * x**4 / 45 - 44 * x**6 / 945)
    return (right - left) / (log(right) - log(left))


def zero_current_aqueous_face(
    left: Mapping[str, float],
    right: Mapping[str, float],
    charges: Mapping[str, int],
    diffusivities: Mapping[str, float],
    *,
    distance_m: float,
    area_m2: float,
    temperature_K: float,
) -> dict[str, Any]:
    """Fluxes point left to right; electric potential is an algebraic difference.

    Positive ions on both sides are the declared domain, not an invented floor.
    Neutral zero values are allowed; their Fick flux uses the exact zero limit.
    """
    if set(left) != set(right) or set(left) != set(charges) or set(left) != set(diffusivities):
        raise ValueError("face declarations must name the same original solutes")
    if any(not isfinite(v) or v <= 0 for v in (distance_m, area_m2, temperature_K)):
        raise ValueError("face geometry and temperature must be positive")
    if any(not isfinite(v) or v < 0 for v in (*left.values(), *right.values())):
        raise ValueError("face concentrations must be finite and nonnegative")
    means = {k: positive_log_mean(left[k], right[k]) for k in left if charges[k]}
    denominator = fsum(charges[k] ** 2 * diffusivities[k] * means[k] for k in means)
    if denominator <= 0 or not isfinite(denominator):
        raise ValueError("zero mobile charge gives an undefined diffusion potential")
    delta = {k: right[k] - left[k] for k in left}
    psi = -fsum(charges[k] * diffusivities[k] * delta[k] for k in left) / denominator
    flux = {
        k: -diffusivities[k] * (delta[k] + charges[k] * means.get(k, 0) * psi) / distance_m
        for k in left
    }
    # Zero neutral concentration has an infinite ideal chemical-potential jump.
    # It is supported as a flux query, but no finite dissipation is invented.
    dissipation = (
        None
        if any(flux[k] and (left[k] == 0 or right[k] == 0) for k in left)
        else -R
        * temperature_K
        * area_m2
        * fsum(
            flux[k] * (log(right[k] / left[k]) + charges[k] * psi)
            for k in left
            if left[k] > 0 and right[k] > 0
        )
    )
    return {
        "dimensionless_potential_difference": psi,
        "potential_difference_V": R * temperature_K / FARADAY_C_MOL * psi,
        "fluxes_mol_m2_s": flux,
        "amount_rates_mol_s": {k: area_m2 * v for k, v in flux.items()},
        "current_molar_charge_rate_mol_s": area_m2 * fsum(charges[k] * flux[k] for k in flux),
        "dissipation_W": dissipation,
    }
