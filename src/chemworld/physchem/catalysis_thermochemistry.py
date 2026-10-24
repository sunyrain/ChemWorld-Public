"""One gas/adsorbate table for Gibbs kinetics and original U evolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from math import fsum, isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.catalysis_descriptors import (
    LinearEnergyScaling,
    finite_number,
    validate_source,
)
from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.hindered_adsorbate import HinderedAdsorbateThermo, harmonic_mode_totals
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network_specs import SpeciesSpec
from chemworld.physchem.thermochemistry import SpeciesThermoState, species_thermo_from_cantera_yaml


def _freeze_declaration(value: Any) -> Any:
    """Detach nested authored data before a parsed model is cached."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_declaration(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_declaration(item) for item in value)
    return deepcopy(value)


def _plain_declaration(value: Any) -> Any:
    """Return fresh JSON containers; callers cannot mutate the stored model."""
    if isinstance(value, Mapping):
        return {key: _plain_declaration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_declaration(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class HarmonicAdsorbateThermo:
    electronic_energy: LinearEnergyScaling
    descriptors: Mapping[str, float]
    vibrational_energies_J_mol: tuple[float, ...]
    minimum_temperature_K: float
    maximum_temperature_K: float
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))
        object.__setattr__(
            self, "vibrational_energies_J_mol", tuple(self.vibrational_energies_J_mol)
        )
        validate_source(self.source)
        self.electronic_energy.evaluate(self.descriptors)
        for value in (
            *self.vibrational_energies_J_mol,
            self.minimum_temperature_K,
            self.maximum_temperature_K,
        ):
            finite_number(value, "harmonic energy/temperature")
            if value <= 0:
                raise ValueError(
                    "harmonic modes and temperature must be positive; no imaginary/zero modes"
                )
        if (
            not self.vibrational_energies_J_mol
            or self.minimum_temperature_K >= self.maximum_temperature_K
        ):
            raise ValueError(
                "harmonic model needs explicit positive modes and a temperature interval"
            )

    def evaluate(
        self, temperature_K: float, *, species_id: str = "adsorbate"
    ) -> SpeciesThermoState:
        finite_number(temperature_K, "temperature")
        if not self.minimum_temperature_K <= temperature_K <= self.maximum_temperature_K:
            raise ValueError("temperature outside declared harmonic domain")
        energy, entropy, capacity = harmonic_mode_totals(
            self.vibrational_energies_J_mol,
            temperature_K,
            self.electronic_energy.evaluate(self.descriptors),
        )
        return SpeciesThermoState(
            species_id,
            temperature_K,
            capacity / R,
            energy / (R * temperature_K),
            entropy / R,
            capacity,
            energy,
            entropy,
            energy - temperature_K * entropy,
            "harmonic-adsorbate-with-ZPE",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "electronic_energy": self.electronic_energy.to_dict(),
            "descriptors": dict(self.descriptors),
            "vibrational_energies_J_mol": list(self.vibrational_energies_J_mol),
            "minimum_temperature_K": self.minimum_temperature_K,
            "maximum_temperature_K": self.maximum_temperature_K,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> HarmonicAdsorbateThermo:
        values = dict(raw)
        values["electronic_energy"] = LinearEnergyScaling.from_dict(values["electronic_energy"])
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(**values)


@dataclass(frozen=True)
class CatalyticSpeciesThermo:
    species: SpeciesSpec
    reference_frame: str
    source: DataSourceProvenance
    gas_declaration: Mapping[str, Any] | None = None
    adsorbate: HarmonicAdsorbateThermo | HinderedAdsorbateThermo | None = None
    surface_declaration: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_source(self.source)
        if not self.reference_frame:
            raise ValueError(
                "catalytic energies require an explicit common caloric reference frame"
            )
        if self.species.phase == "gas":
            if (
                self.gas_declaration is None
                or self.adsorbate is not None
                or self.surface_declaration is not None
            ):
                raise ValueError(
                    "gas species need exactly the original gas thermochemistry declaration"
                )
            object.__setattr__(self, "gas_declaration", _freeze_declaration(self.gas_declaration))
            if set(self.gas_declaration) - {
                "thermo",
                "composition",
                "reference_pressure_Pa",
                "units",
            }:
                raise ValueError("unsupported gas thermochemistry declaration")
            if dict(self.gas_model.composition) != self.species.composition:
                raise ValueError("gas thermochemistry and native species composition disagree")
        elif self.species.phase == "surface":
            if (self.adsorbate is None) == (
                self.surface_declaration is None
            ) or self.gas_declaration is not None:
                raise ValueError(
                    "surface species need exactly one adsorbate or source thermochemistry model"
                )
            if isinstance(self.adsorbate, HinderedAdsorbateThermo) and (
                sum(self.species.composition.values()) != self.adsorbate.number_of_atoms
            ):
                raise ValueError("hindered molecular mode count disagrees with species composition")
            if self.surface_declaration is not None:
                object.__setattr__(
                    self, "surface_declaration", _freeze_declaration(self.surface_declaration)
                )
                if set(self.surface_declaration) - {
                    "thermo",
                    "composition",
                    "reference_pressure_Pa",
                    "units",
                }:
                    raise ValueError("unsupported surface thermochemistry declaration")
                if dict(self.surface_model.composition) != self.species.composition:
                    raise ValueError("surface caloric and native elemental composition disagree")
        else:
            raise ValueError("catalytic profile supports gas and surface species")

    @cached_property
    def gas_model(self) -> Any:
        if self.gas_declaration is None:
            raise ValueError("surface species do not have gas caloric semantics")
        return species_thermo_from_cantera_yaml(
            species_id=self.species.species_id, **_plain_declaration(self.gas_declaration)
        )

    @cached_property
    def surface_model(self) -> Any:
        if self.surface_declaration is None:
            raise ValueError("this surface species has no source polynomial thermochemistry")
        return species_thermo_from_cantera_yaml(
            species_id=self.species.species_id, **_plain_declaration(self.surface_declaration)
        )

    def evaluate(self, temperature_K: float) -> SpeciesThermoState:
        if self.adsorbate is not None:
            return self.adsorbate.evaluate(temperature_K, species_id=self.species.species_id)
        if self.surface_declaration is not None:
            return self.surface_model.evaluate(temperature_K)
        return self.gas_model.evaluate(temperature_K)

    def internal_energy(self, temperature_K: float) -> float:
        value = self.evaluate(temperature_K).enthalpy_J_mol
        return value - R * temperature_K if self.species.phase == "gas" else value

    def cv(self, temperature_K: float) -> float:
        value = self.evaluate(temperature_K).cp_J_mol_K
        return value - R if self.species.phase == "gas" else value

    def validate_temperature_interval(self, lower_K: float, upper_K: float) -> None:
        if self.adsorbate is not None:
            lower, upper = (
                self.adsorbate.minimum_temperature_K,
                self.adsorbate.maximum_temperature_K,
            )
        else:
            model = self.surface_model if self.surface_declaration is not None else self.gas_model
            lower, upper = model.min_temperature_K, model.max_temperature_K
            from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics

            IdealGasThermodynamics({self.species.species_id: model}).validate_temperature_interval(
                lower_K, upper_K
            )
        if not lower <= lower_K < upper_K <= upper:
            raise ValueError("thermal interval leaves declared catalytic thermochemistry domain")
        # Evaluations also reject unsupported caloric segments at the endpoints.
        if min(self.cv(lower_K), self.cv(upper_K)) < 0:
            raise ValueError("catalytic species require nonnegative Cv in the declared interval")

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": _plain_declaration(self.species.to_dict()),
            "reference_frame": self.reference_frame,
            "source": self.source.to_dict(),
            "gas_declaration": None
            if self.gas_declaration is None
            else _plain_declaration(self.gas_declaration),
            "adsorbate": None if self.adsorbate is None else self.adsorbate.to_dict(),
            **(
                {"surface_declaration": _plain_declaration(self.surface_declaration)}
                if self.surface_declaration is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CatalyticSpeciesThermo:
        values = dict(raw)
        native = dict(values["species"])
        composition = native.pop("composition", None)
        values["species"] = SpeciesSpec(**native)
        if composition is not None and composition != values["species"].composition:
            raise ValueError("encoded catalytic species composition conflicts with formula")
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        if values["adsorbate"] is not None:
            encoded = values["adsorbate"]
            values["adsorbate"] = (
                HinderedAdsorbateThermo.from_dict(encoded)
                if "model" in encoded
                else HarmonicAdsorbateThermo.from_dict(encoded)
            )
        return cls(**values)


@dataclass(frozen=True)
class CatalyticThermodynamics:
    species: Mapping[str, CatalyticSpeciesThermo]
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", MappingProxyType(dict(self.species)))
        if (
            not self.species
            or any(not key for key in self.species)
            or len({s.reference_frame for s in self.species.values()}) != 1
        ):
            raise ValueError(
                "catalytic thermal table needs species on one declared reference frame"
            )
        finite_number(self.support_heat_capacity_J_K, "support heat capacity")
        finite_number(self.reference_temperature_K, "support reference temperature")
        if self.support_heat_capacity_J_K < 0 or self.reference_temperature_K <= 0:
            raise ValueError("invalid catalytic support caloric data")

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(self.species)

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return self.support_heat_capacity_J_K + fsum(
            state.amounts_mol[k] * s.cv(state.temperature_K) for k, s in self.species.items()
        )

    def species_energy_j_mol(self, key: str, temperature_K: float) -> float:
        return self.species[key].internal_energy(temperature_K)

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return self.support_heat_capacity_J_K * (
            state.temperature_K - self.reference_temperature_K
        ) + fsum(
            state.amounts_mol[k] * s.internal_energy(state.temperature_K)
            for k, s in self.species.items()
        )

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("catalytic U must not also consume a separate reaction heat")
        return fsum(
            rate * self.species_energy_j_mol(key, state.temperature_K)
            for key, rate in derivatives.items()
            if rate
        )

    def validate_state(self, state: CoupledStateView) -> None:
        if set(state.amounts_mol) != set(self.species):
            raise ValueError("catalytic caloric inventory must match original physical inventory")
        if not isfinite(self.heat_capacity_j_k(state)) or self.heat_capacity_j_k(state) <= 0:
            raise ValueError("catalytic system needs positive finite heat capacity")

    def validate_temperature_interval(self, lower_K: float, upper_K: float) -> None:
        for value in self.species.values():
            value.validate_temperature_interval(lower_K, upper_K)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "declared-catalytic-thermochemistry-v1",
            "species": {k: v.to_dict() for k, v in self.species.items()},
            "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
            "reference_temperature_K": self.reference_temperature_K,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CatalyticThermodynamics:
        values = dict(raw)
        if values.pop("model") != "declared-catalytic-thermochemistry-v1":
            raise ValueError("unknown catalytic thermal model")
        values["species"] = {
            k: CatalyticSpeciesThermo.from_dict(v) for k, v in values["species"].items()
        }
        return cls(**values)
