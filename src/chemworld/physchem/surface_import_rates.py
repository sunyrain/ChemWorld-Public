"""Declarative ideal-surface rates, evaluated by the original catalytic process.

Concentrations are mol/m3 (gas) and mol/m2 (surface). A rate of progress
is mol/m2/s; multiplying the original finite site capacity / density gives
the actual area. No upstream kinetic implementation is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, fsum, isfinite, pi, prod, sqrt
from types import MappingProxyType
from typing import Any

from chemworld.physchem.catalysis_descriptors import validate_source
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class ImportedSurfaceReaction:
    reaction_id: str
    reactants: Mapping[str, int]
    products: Mapping[str, int]
    rate_site: str
    source: DataSourceProvenance
    coefficient: float
    temperature_exponent: float
    activation_energy_J_mol: float
    orders: Mapping[str, float]
    reversible: bool = True
    sticking_species: str | None = None
    sticking_molar_mass_kg_mol: float | None = None
    motz_wise: bool = False
    coverage_dependencies: Mapping[str, tuple[float, float, float]] | None = None

    def __post_init__(self) -> None:
        validate_source(self.source)
        for name in ("reactants", "products", "orders"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self,
            "coverage_dependencies",
            MappingProxyType(
                {key: tuple(value) for key, value in (self.coverage_dependencies or {}).items()}
            ),
        )
        if not self.reaction_id or not self.rate_site or self.reactants == self.products:
            raise ValueError("surface reaction requires identity and net stoichiometry")
        for side in (self.reactants, self.products):
            if not side or any(not k or type(v) is not int or v <= 0 for k, v in side.items()):
                raise ValueError("surface elementary stoichiometry must be positive integers")
        numbers = (self.coefficient, self.temperature_exponent, self.activation_energy_J_mol)
        if any(isinstance(v, bool) or not isfinite(v) for v in numbers) or self.coefficient <= 0:
            raise ValueError("surface Arrhenius coefficient must be positive and finite")
        if set(self.orders) != set(self.reactants) or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0 for v in self.orders.values()
        ):
            raise ValueError("surface orders require strictly positive orders on every reactant")
        if self.reversible and dict(self.orders) != dict(self.reactants):
            raise ValueError("custom orders are supported only for irreversible reactions")
        if type(self.reversible) is not bool or type(self.motz_wise) is not bool:
            raise ValueError("reversibility and Motz-Wise options must be booleans")
        if self.sticking_species is None:
            if self.sticking_molar_mass_kg_mol is not None or self.motz_wise:
                raise ValueError("sticking options require a declared incident gas")
        elif (
            self.reactants.get(self.sticking_species) != 1
            or self.orders.get(self.sticking_species) != 1
            or self.sticking_molar_mass_kg_mol is None
            or not isfinite(self.sticking_molar_mass_kg_mol)
            or self.sticking_molar_mass_kg_mol <= 0
        ):
            raise ValueError("sticking needs one incident molecule and its positive molar mass")
        for values in (self.coverage_dependencies or {}).values():
            if len(values) != 3 or any(not isfinite(v) for v in values) or values[1] < 0:
                raise ValueError("coverage a/m/E needs finite coefficients and m >= 0")

    @property
    def stoichiometry(self) -> dict[str, int]:
        return {
            k: self.products.get(k, 0) - self.reactants.get(k, 0)
            for k in self.products.keys() | self.reactants.keys()
            if self.products.get(k, 0) != self.reactants.get(k, 0)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "imported-ideal-surface-rate-v1",
            "reaction_id": self.reaction_id,
            "reactants": dict(self.reactants),
            "products": dict(self.products),
            "rate_site": self.rate_site,
            "source": self.source.to_dict(),
            "coefficient": self.coefficient,
            "temperature_exponent": self.temperature_exponent,
            "activation_energy_J_mol": self.activation_energy_J_mol,
            "orders": dict(self.orders),
            "reversible": self.reversible,
            "sticking_species": self.sticking_species,
            "sticking_molar_mass_kg_mol": self.sticking_molar_mass_kg_mol,
            "motz_wise": self.motz_wise,
            "coverage_dependencies": {
                k: list(v) for k, v in (self.coverage_dependencies or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ImportedSurfaceReaction:
        values = dict(raw)
        if values.pop("model") != "imported-ideal-surface-rate-v1":
            raise ValueError("unknown surface rate model")
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(**values)


def imported_surface_rates(
    reaction: ImportedSurfaceReaction,
    *,
    temperature_K: float,
    activities: Mapping[str, float],
    gas_species: tuple[str, ...],
    reference_pressure_Pa: float,
    site_density_mol_m2: float,
    capacity_mol: float,
    delta_g_J_mol: float,
    site_occupancies: Mapping[str, int] | None = None,
) -> tuple[float, float]:
    """Return actual forward/reverse mol/s on one original finite carrier."""
    thermal = R * temperature_K
    coverage = reaction.coverage_dependencies or {}
    factor = exp(
        fsum(
            (2.302585092994046 * a - energy / thermal) * activities[k]
            for k, (a, _, energy) in coverage.items()
        )
    ) * prod(activities[k] ** m for k, (_, m, _) in coverage.items())
    coefficient = (
        reaction.coefficient
        * temperature_K**reaction.temperature_exponent
        * exp(-reaction.activation_energy_J_mol / thermal)
        * factor
    )
    gas_standard = reference_pressure_Pa / thermal
    if reaction.sticking_species is not None:
        if not 0 <= coefficient <= 1:
            raise ValueError("sticking probability outside [0,1]; no clipping")
        if reaction.motz_wise:
            coefficient /= 1 - coefficient / 2
        if reaction.sticking_molar_mass_kg_mol is None:
            raise ValueError("incident gas molar mass is required")
        turnover = (
            coefficient
            * sqrt(thermal / (2 * pi * reaction.sticking_molar_mass_kg_mol))
            * gas_standard
            / site_density_mol_m2
        )
    else:
        turnover = (
            coefficient
            / site_density_mol_m2
            * prod(
                (
                    gas_standard
                    if key in gas_species
                    else site_density_mol_m2 / (site_occupancies or {}).get(key, 1)
                )
                ** order
                for key, order in reaction.orders.items()
            )
        )
    forward = (
        capacity_mol
        * turnover
        * prod(activities[k] ** order for k, order in reaction.orders.items())
    )
    reverse = 0.0
    if reaction.reversible:
        reverse = (
            capacity_mol
            * turnover
            * exp(delta_g_J_mol / thermal)
            * prod(activities[k] ** order for k, order in reaction.products.items())
        )
    return forward, reverse
