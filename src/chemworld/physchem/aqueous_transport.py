"""Declared binary-electrolyte transport over the original aqueous G/H/V table.

The salt basis is an algebraic flux coordinate, never a material inventory.
Film concentrations use solvent volume explicitly; bulk volumes use G_P.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import exp, isclose, isfinite
from typing import Any, cast

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_phase_properties import (
    AqueousPhaseProperties,
    aqueous_phase_properties,
)
from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class AqueousTransportLaw:
    """Positive sourced law v=vref exp(E/R(1/Tr-1/T)+a(P-Pr)+b*c).

    c is mol of declared salt basis per m3 of solvent standard-state volume.
    The pressure coefficient has units Pa^-1 and concentration coefficient m3/mol.
    """

    value: float
    unit: str
    source_id: str
    temperature_bounds_K: tuple[float, float]
    pressure_bounds_Pa: tuple[float, float]
    maximum_concentration_mol_m3: float
    activation_energy_J_mol: float = 0.0
    per_Pa: float = 0.0
    per_mol_m3: float = 0.0
    reference_temperature_K: float = 298.15
    reference_pressure_Pa: float = 100000.0

    def __post_init__(self) -> None:
        for name in ("temperature_bounds_K", "pressure_bounds_Pa"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            lo, hi = getattr(self, name)
            if not 0 < lo < hi or not all(map(isfinite, (lo, hi))):
                raise ValueError("transport law requires finite positive T/P domain")
        numbers = (
            self.value,
            self.maximum_concentration_mol_m3,
            self.activation_energy_J_mol,
            self.per_Pa,
            self.per_mol_m3,
            self.reference_temperature_K,
            self.reference_pressure_Pa,
        )
        if (
            not self.source_id
            or not self.unit
            or self.value <= 0
            or self.maximum_concentration_mol_m3 <= 0
            or any(isinstance(v, bool) or not isfinite(v) for v in numbers)
        ):
            raise ValueError("transport law requires positive value, source, units and domain")
        self.evaluate(self.reference_temperature_K, self.reference_pressure_Pa, 0.0)

    def evaluate(
        self, temperature_K: float, pressure_Pa: float, concentration_mol_m3: float
    ) -> float:
        if any(
            not isfinite(v) or not lo <= v <= hi
            for v, (lo, hi) in (
                (temperature_K, self.temperature_bounds_K),
                (pressure_Pa, self.pressure_bounds_Pa),
                (concentration_mol_m3, (0, self.maximum_concentration_mol_m3)),
            )
        ):
            raise ValueError("transport state outside declared T/P/salinity domain")
        value = self.value * exp(
            self.activation_energy_J_mol
            / R
            * (1 / self.reference_temperature_K - 1 / temperature_K)
            + self.per_Pa * (pressure_Pa - self.reference_pressure_Pa)
            + self.per_mol_m3 * concentration_mol_m3
        )
        if not isfinite(value) or value <= 0:
            raise ValueError("transport correlation must remain finite and positive")
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BinaryAqueousProperties:
    database: AqueousDatabase
    salt_stoichiometry: tuple[tuple[str, int], ...]
    viscosity: AqueousTransportLaw

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "salt_stoichiometry", tuple(tuple(v) for v in self.salt_stoichiometry)
        )
        ions = {s.species_id: s for s in self.database.species if s.phase == "aqueous"}
        if (
            len(ions) != 2
            or len(self.database.species) != 3
            or self.database.reactions
            or len(self.salt_stoichiometry) != 2
            or set(dict(self.salt_stoichiometry)) != set(ions)
            or any(type(v) is not int or v <= 0 for _, v in self.salt_stoichiometry)
            or any(s.charge == 0 for s in ions.values())
            or sum(ions[k].charge * v for k, v in self.salt_stoichiometry) != 0
        ):
            raise ValueError("binary RO requires two ions and an exact electroneutral flux basis")
        if (
            any(s.standard_state.pressure_model != "constant-volume" for s in self.database.species)
            or self.water_volume_m3_mol <= 0
        ):
            raise ValueError(
                "RO needs explicit constant-volume standard states and positive water V"
            )
        if self.viscosity.unit != "Pa*s" or self.viscosity.source_id not in {
            s.source_id for s in self.database.sources
        }:
            raise ValueError("viscosity requires Pa*s and a declared database source")

    @property
    def water_id(self) -> str:
        return self.database.solvent_species.species_id

    @property
    def water_volume_m3_mol(self) -> float:
        return cast(float, self.database.solvent_species.standard_state.molar_volume_m3_mol)

    @property
    def water_mass_kg_mol(self) -> float:
        return molecular_weight(dict(self.database.solvent_species.composition)) / 1000

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.database.species)

    @property
    def temperature_bounds_K(self) -> tuple[float, float]:  # noqa: N802
        return self.viscosity.temperature_bounds_K

    def amounts(self, water_mol: float, salt_mol: float) -> dict[str, float]:
        return {self.water_id: water_mol, **{k: nu * salt_mol for k, nu in self.salt_stoichiometry}}

    def validate_amounts(self, amounts: Mapping[str, float], *, empty: bool = False) -> None:
        if set(amounts) != set(self.component_ids) or any(
            not isfinite(v) or v < 0 for v in amounts.values()
        ):
            raise ValueError("RO inventory requires all original species and nonnegative moles")
        if empty and not any(amounts.values()):
            return
        if amounts[self.water_id] <= 0:
            raise ValueError("nonempty RO liquid requires finite water")
        values = [amounts[k] / nu for k, nu in self.salt_stoichiometry]
        if not isclose(values[0], values[1], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("original ion inventory differs from the electroneutral salt basis")

    def concentration(self, amounts: Mapping[str, float]) -> float:
        key, nu = self.salt_stoichiometry[0]
        return amounts[key] / nu / (amounts[self.water_id] * self.water_volume_m3_mol)

    def liquid(
        self, amounts: Mapping[str, float], temperature_K: float, pressure_Pa: float
    ) -> AqueousPhaseProperties:
        return aqueous_phase_properties(
            self.database, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
        )

    def enthalpy(
        self, amounts: Mapping[str, float], temperature_K: float, pressure_Pa: float
    ) -> float:
        return (
            self.liquid(amounts, temperature_K, pressure_Pa).enthalpy_J
            if any(amounts.values())
            else 0.0
        )

    def volume(
        self, amounts: Mapping[str, float], temperature_K: float, pressure_Pa: float
    ) -> float:
        return (
            self.liquid(amounts, temperature_K, pressure_Pa).volume_m3
            if any(amounts.values())
            else 0.0
        )

    def enthalpy_rate(
        self,
        amounts: Mapping[str, float],
        rates: Mapping[str, float],
        temperature_K: float,
        pressure_Pa: float,
    ) -> float:
        """Directional dH/dn from the same excess G, including heat of mixing."""
        if not any(amounts.values()):
            if any(v < 0 for v in rates.values()):
                raise ValueError("an empty receiver cannot discharge material")
            return self.enthalpy(rates, temperature_K, pressure_Pa)
        states = self.database.standard_states(temperature_K, pressure_Pa)
        standard = sum(rates[k] * states[k].enthalpy_J_mol for k in self.component_ids)
        # All binary ionic excess-G families vanish faster than linearly at
        # pure water. Their first partial molar excess enthalpy limit is zero.
        if not any(amounts[k] for k, _ in self.salt_stoichiometry):
            return standard

        def excess(x: Mapping[str, Any]) -> Any:
            n = {k: amounts[k] + x["extent"] * rates[k] for k in self.component_ids}
            kg = n[self.water_id] * self.water_mass_kg_mol
            m = {k: n[k] / kg for k, _ in self.salt_stoichiometry}
            return kg * self.database.activity_model.excess_gibbs_RT_mol_kg(m, x["T"], pressure_Pa)

        return standard - R * temperature_K**2 * partial_derivative(
            excess, {"T": temperature_K, "extent": 0.0}, ("T", "extent")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "database": self.database.to_dict(),
            "salt_stoichiometry": [list(row) for row in self.salt_stoichiometry],
            "viscosity": self.viscosity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BinaryAqueousProperties:
        return cls(
            AqueousDatabase.from_dict(data["database"]),
            data["salt_stoichiometry"],
            AqueousTransportLaw(**data["viscosity"]),
        )
