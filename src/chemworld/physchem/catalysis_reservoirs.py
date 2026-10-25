"""Explicit dilute-solute chemostats and a declared SHE half-cell reference.

Solutes have real finite liquid inventories. Their partial volume is neglected in
this declared dilute approximation; the original solvent carrier owns liquid V/Cv.
Activities control rates once. They do not enter material U or inlet enthalpy.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose, log
from types import MappingProxyType
from typing import Any

from chemworld.physchem.catalysis_descriptors import finite_number, validate_source
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.heat_transfer_units import CondensedBackgroundSpec, ThermalDeviceSpec
from chemworld.physchem.reaction_network_specs import SpeciesSpec
from chemworld.physchem.thermochemistry import SpeciesThermoState

R = 8.31446261815324


@dataclass(frozen=True)
class AqueousStandardThermo:
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    heat_capacity_J_mol_K: float
    reference_temperature_K: float
    minimum_temperature_K: float
    maximum_temperature_K: float
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        validate_source(self.source)
        for key in (
            "enthalpy_J_mol",
            "entropy_J_mol_K",
            "heat_capacity_J_mol_K",
            "reference_temperature_K",
            "minimum_temperature_K",
            "maximum_temperature_K",
        ):
            finite_number(getattr(self, key), key)
        if (
            self.heat_capacity_J_mol_K < 0
            or not 0
            < self.minimum_temperature_K
            <= self.reference_temperature_K
            <= self.maximum_temperature_K
            or self.minimum_temperature_K == self.maximum_temperature_K
        ):
            raise ValueError("aqueous standard H/S/Cp needs a bounded temperature domain")

    def evaluate(self, temperature_K: float, *, species_id: str) -> SpeciesThermoState:
        if not self.minimum_temperature_K <= temperature_K <= self.maximum_temperature_K:
            raise ValueError("aqueous standard state outside declared temperature domain")
        cp = self.heat_capacity_J_mol_K
        h = self.enthalpy_J_mol + cp * (temperature_K - self.reference_temperature_K)
        s = self.entropy_J_mol_K + cp * log(temperature_K / self.reference_temperature_K)
        return SpeciesThermoState(
            species_id,
            temperature_K,
            cp / R,
            h / (R * temperature_K),
            s / R,
            cp,
            h,
            s,
            h - temperature_K * s,
            "declared-dilute-aqueous-standard",
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "source": self.source.to_dict()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AqueousStandardThermo":
        return cls(**{**values, "source": DataSourceProvenance.from_dict(values["source"])})


@dataclass(frozen=True)
class ChemostatSolute:
    activity: float
    activity_coefficient: float
    standard_concentration_mol_L: float
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        validate_source(self.source)
        for key in ("activity", "activity_coefficient", "standard_concentration_mol_L"):
            finite_number(getattr(self, key), key)
            if getattr(self, key) <= 0:
                raise ValueError("constant-activity reservoirs need positive a/gamma/c_standard")

    def amount_mol(self, liquid_volume_L: float) -> float:
        return (
            self.activity
            * self.standard_concentration_mol_L
            * liquid_volume_L
            / self.activity_coefficient
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "source": self.source.to_dict()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ChemostatSolute":
        return cls(**{**values, "source": DataSourceProvenance.from_dict(values["source"])})


@dataclass(frozen=True)
class CatalyticReservoirBoundary:
    solutes: Mapping[str, ChemostatSolute]
    liquid_volume_L: float
    gas_volume_L: float
    carrier: CondensedBackgroundSpec
    bath: ThermalDeviceSpec
    minimum_potential_V: float
    maximum_potential_V: float
    maximum_pressure_Pa: float
    maximum_abs_current_A: float
    source: DataSourceProvenance
    reference_note: str
    proton_species_id: str | None = None
    potential_reference: str = "SHE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "solutes", MappingProxyType(dict(self.solutes)))
        validate_source(self.source)
        for key in (
            "liquid_volume_L",
            "gas_volume_L",
            "minimum_potential_V",
            "maximum_potential_V",
            "maximum_pressure_Pa",
            "maximum_abs_current_A",
        ):
            finite_number(getattr(self, key), key)
        if (
            not self.solutes
            or any(not k for k in self.solutes)
            or min(
                self.liquid_volume_L,
                self.gas_volume_L,
                self.maximum_pressure_Pa,
                self.maximum_abs_current_A,
            )
            <= 0
            or self.minimum_potential_V >= self.maximum_potential_V
            or self.potential_reference != "SHE"
            or self.carrier.solvent_index != 0
            or self.carrier.source_vessel_id is not None
            or not self.reference_note.strip()
            or not self.bath.enabled
            or self.bath.mode != "isothermal"
            or (self.proton_species_id is not None and self.proton_species_id not in self.solutes)
        ):
            raise ValueError(
                "declare finite phases, constant SHE/pH boundaries and a common isotherm"
            )

    @property
    def activities(self) -> dict[str, float]:
        return {k: v.activity for k, v in self.solutes.items()}

    @property
    def amounts_mol(self) -> dict[str, float]:
        return {k: v.amount_mol(self.liquid_volume_L) for k, v in self.solutes.items()}

    def validate_species(self, species: Mapping[str, SpeciesSpec]) -> None:
        if set(species) != set(self.solutes) or any(s.phase != "liquid" for s in species.values()):
            raise ValueError("chemostat must bind exactly the declared real aqueous species")
        charge = sum(species[k].charge * n for k, n in self.amounts_mol.items())
        scale = sum(abs(species[k].charge) * n for k, n in self.amounts_mol.items())
        if not isclose(charge, 0, rel_tol=0, abs_tol=max(scale, 1e-20) * 1e-12):
            raise ValueError(
                "supported liquid chemostat requires an electroneutral authored inventory"
            )
        if self.proton_species_id is not None:
            proton = species[self.proton_species_id]
            if proton.composition != {"H": 1} or proton.charge != 1:
                raise ValueError("pH must refer to real aqueous H+ identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "constant-activity-catalytic-reservoirs-v1",
            "solutes": {k: v.to_dict() for k, v in self.solutes.items()},
            "liquid_volume_L": self.liquid_volume_L,
            "gas_volume_L": self.gas_volume_L,
            "carrier": asdict(self.carrier),
            "bath": self.bath.to_dict(),
            "minimum_potential_V": self.minimum_potential_V,
            "maximum_potential_V": self.maximum_potential_V,
            "maximum_pressure_Pa": self.maximum_pressure_Pa,
            "maximum_abs_current_A": self.maximum_abs_current_A,
            "source": self.source.to_dict(),
            "reference_note": self.reference_note,
            "proton_species_id": self.proton_species_id,
            "potential_reference": self.potential_reference,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CatalyticReservoirBoundary":
        values = dict(raw)
        if values.pop("schema", None) != "constant-activity-catalytic-reservoirs-v1":
            raise ValueError("unknown catalytic reservoir boundary")
        return cls(
            **{
                **values,
                "solutes": {k: ChemostatSolute.from_dict(v) for k, v in values["solutes"].items()},
                "carrier": CondensedBackgroundSpec(**values["carrier"]),
                "bath": ThermalDeviceSpec.from_dict(values["bath"]),
                "source": DataSourceProvenance.from_dict(values["source"]),
            }
        )
