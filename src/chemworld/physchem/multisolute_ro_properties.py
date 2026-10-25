"""Real aqueous inventories with independent, neutral membrane flux channels.

Channels are coordinates for flux calculation, never material entities. Every
component remains in the original aqueous database and n/H inventory.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from scipy.optimize import nnls

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_transport import AqueousTransportLaw
from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.seawater_ro import empirical_seawater_enthalpy


@dataclass(frozen=True)
class NeutralROChannel:
    channel_id: str
    stoichiometry: tuple[tuple[str, int], ...]
    permeability: AqueousTransportLaw | None
    mass_transfer: AqueousTransportLaw

    def __post_init__(self) -> None:
        object.__setattr__(self, "stoichiometry", tuple(tuple(x) for x in self.stoichiometry))
        if (
            not self.channel_id
            or not self.stoichiometry
            or len(dict(self.stoichiometry)) != len(self.stoichiometry)
            or any(type(v) is not int or v <= 0 for _, v in self.stoichiometry)
        ):
            raise ValueError("RO channels require named positive integer stoichiometry")
        if self.mass_transfer.unit != "m/s" or (
            self.permeability is not None and self.permeability.unit != "m/s"
        ):
            raise ValueError("RO channel permeability and film coefficient use m/s")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "stoichiometry": [list(x) for x in self.stoichiometry],
            "permeability": self.permeability.to_dict() if self.permeability else None,
            "mass_transfer": self.mass_transfer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NeutralROChannel":
        return cls(
            data["channel_id"],
            tuple(data["stoichiometry"]),
            AqueousTransportLaw(**data["permeability"]) if data["permeability"] else None,
            AqueousTransportLaw(**data["mass_transfer"]),
        )


@dataclass(frozen=True)
class ROFlatSheetChannel:
    """Declared spacer geometry; published flat-sheet Sherwood/Darcy laws."""

    height_m: float
    spacer_porosity: float
    diffusivities: tuple[tuple[str, AqueousTransportLaw], ...]
    finite_elements: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "diffusivities", tuple(tuple(x) for x in self.diffusivities))
        if (
            not isfinite(self.height_m)
            or self.height_m <= 0
            or not 0 < self.spacer_porosity < 1
            or len(dict(self.diffusivities)) != len(self.diffusivities)
            or not self.diffusivities
            or any(law.unit != "m^2/s" for _, law in self.diffusivities)
            or type(self.finite_elements) is not int
            or not 1 <= self.finite_elements <= 1000
        ):
            raise ValueError(
                "flat-sheet RO needs finite spacer geometry, diffusivities and axial cells"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_m": self.height_m,
            "spacer_porosity": self.spacer_porosity,
            "diffusivities": [[k, law.to_dict()] for k, law in self.diffusivities],
            "finite_elements": self.finite_elements,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ROFlatSheetChannel":
        return cls(
            data["height_m"],
            data["spacer_porosity"],
            tuple((k, AqueousTransportLaw(**v)) for k, v in data["diffusivities"]),
            data["finite_elements"],
        )


@dataclass(frozen=True)
class MultisoluteROProperties(MultiIonMembraneProperties):
    channels: tuple[NeutralROChannel, ...]
    property_approximation: str = "common-g"

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        sources = {s.source_id for s in self.database.sources}
        if self.property_approximation not in {"common-g", "mcas-ideal-volume"}:
            raise ValueError("RO properties require an explicit supported approximation")
        if (
            not self.channels
            or len({c.channel_id for c in self.channels}) != len(self.channels)
            or len(self.component_ids) != len(self.solute_ids) + 1
            or self.database.reactions
            or self.water_volume_m3_mol <= 0
            or any(
                s.standard_state.pressure_model != "constant-volume" for s in self.database.species
            )
            or self.viscosity.unit != "Pa*s"
            or self.viscosity.source_id not in sources
        ):
            raise ValueError("RO mixtures require explicit aqueous species, V and viscosity")
        covered: set[str] = set()
        for channel in self.channels:
            row = dict(channel.stoichiometry)
            if set(row) - set(self.solute_ids) or sum(self.charges[k] * v for k, v in row.items()):
                raise ValueError("every RO channel must conserve actual ionic charge exactly")
            covered.update(row)
            if any(
                law.source_id not in sources
                for law in (channel.permeability, channel.mass_transfer)
                if law is not None
            ):
                raise ValueError("RO transport laws must cite a declared source")
        if covered != set(self.solute_ids) or np.linalg.matrix_rank(self.basis) != len(
            self.channels
        ):
            raise ValueError("RO channel basis must independently span all declared solutes")

    @property
    def solute_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.database.species if s.phase == "aqueous")

    @property
    def basis(self) -> np.ndarray:
        return np.array(
            [[dict(c.stoichiometry).get(k, 0) for c in self.channels] for k in self.solute_ids],
            dtype=float,
        )

    def channel_amounts(self, amounts: Mapping[str, float]) -> dict[str, float]:
        target = np.array([amounts[k] for k in self.solute_ids])
        solution, _ = nnls(self.basis, target)
        if np.max(np.abs(self.basis @ solution - target)) > (
            1e-12 + 1e-10 * np.max(np.abs(target))
        ):
            raise ValueError("aqueous mixture lies outside declared neutral transport basis")
        return {c.channel_id: float(v) for c, v in zip(self.channels, solution, strict=True)}

    def validate_amounts(self, amounts: Mapping[str, float], *, empty: bool = False) -> None:
        if set(amounts) != set(self.component_ids) or any(
            not isfinite(v) or v < 0 for v in amounts.values()
        ):
            raise ValueError("RO mixture needs all actual nonnegative component inventories")
        if empty and not any(amounts.values()):
            return
        if amounts[self.water_id] <= 0:
            raise ValueError("nonempty RO liquid requires water")
        self.channel_amounts(amounts)

    def amounts(
        self, water_mol: float, channel_amounts: Mapping[str, float] | float
    ) -> dict[str, float]:
        if not isinstance(channel_amounts, Mapping):
            if channel_amounts != 0:
                raise ValueError("multiple solutes need a channel amount mapping")
            channel_amounts = dict.fromkeys((c.channel_id for c in self.channels), 0.0)
        if set(channel_amounts) != {c.channel_id for c in self.channels}:
            raise ValueError("every declared solute channel must be present")
        values = self.basis @ np.array([channel_amounts[c.channel_id] for c in self.channels])
        return {
            self.water_id: water_mol,
            **dict(zip(self.solute_ids, map(float, values), strict=True)),
        }

    def channel_concentrations(self, amounts: Mapping[str, float]) -> dict[str, float]:
        return {
            k: v / (amounts[self.water_id] * self.water_volume_m3_mol)
            for k, v in self.channel_amounts(amounts).items()
        }

    def concentration(self, amounts: Mapping[str, float]) -> float:
        return sum(self.channel_concentrations(amounts).values())

    @property
    def component_mass_kg_mol(self) -> dict[str, float]:
        return {
            s.species_id: molecular_weight(dict(s.composition)) / 1000
            for s in self.database.species
        }

    def volume(self, n: Mapping[str, float], t: float, p: float) -> float:
        if self.property_approximation == "common-g":
            return super().volume(n, t, p)
        return sum(n[k] * v for k, v in self.component_mass_kg_mol.items()) / 1000.0

    def enthalpy(self, n: Mapping[str, float], t: float, p: float) -> float:
        if self.property_approximation == "common-g":
            return super().enthalpy(n, t, p)
        mass = {k: n[k] * v for k, v in self.component_mass_kg_mol.items()}
        total = sum(mass.values())
        if not total:
            return 0.0
        s = 1 - mass[self.water_id] / total
        if not 283.15 <= t <= 393.15 or not 1e5 <= p <= 12e6 or not 0 <= s <= 0.12:
            raise ValueError("MCAS caloric approximation outside declared Nayar domain")
        return float(total * empirical_seawater_enthalpy(s, t, p))

    def liquid(self, n: Mapping[str, float], t: float, p: float) -> Any:
        if self.property_approximation == "common-g":
            return super().liquid(n, t, p)
        return EmpiricalROLiquid(self.volume(n, t, p), self.enthalpy(n, t, p))

    def enthalpy_rate(
        self, n: Mapping[str, float], rates: Mapping[str, float], t: float, p: float
    ) -> float:
        if not any(n.values()):
            if any(v < 0 for v in rates.values()):
                raise ValueError("empty RO receiver cannot discharge")
            return self.enthalpy(rates, t, p)
        if self.property_approximation == "mcas-ideal-volume":
            weights = self.component_mass_kg_mol

            def caloric(x: Mapping[str, Any]) -> Any:
                mass = {k: (n[k] + x["extent"] * rates[k]) * weights[k] for k in self.component_ids}
                total = sum(mass.values())
                s = 1 - mass[self.water_id] / total
                return total * empirical_seawater_enthalpy(s, t, p)

            return partial_derivative(caloric, {"extent": 0.0}, ("extent",))
        states = self.database.standard_states(t, p)
        standard = sum(rates[k] * states[k].enthalpy_J_mol for k in self.component_ids)
        if self.database.activity_model.family == "ideal" or not any(n[k] for k in self.solute_ids):
            return standard

        def excess(x: Mapping[str, Any]) -> Any:
            current = {k: n[k] + x["extent"] * rates[k] for k in self.component_ids}
            kg = current[self.water_id] * self.water_mass_kg_mol
            return kg * self.database.activity_model.excess_gibbs_RT_mol_kg(
                {k: current[k] / kg for k in self.solute_ids}, x["T"], p
            )

        return standard - R * t * t * partial_derivative(
            excess, {"T": t, "extent": 0.0}, ("T", "extent")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "database": self.database.to_dict(),
            "viscosity": self.viscosity.to_dict(),
            "channels": [c.to_dict() for c in self.channels],
            "property_approximation": self.property_approximation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultisoluteROProperties":
        return cls(
            AqueousDatabase.from_dict(data["database"]),
            AqueousTransportLaw(**data["viscosity"]),
            tuple(NeutralROChannel.from_dict(c) for c in data["channels"]),
            data.get("property_approximation", "common-g"),
        )


@dataclass(frozen=True)
class EmpiricalROLiquid:
    """Independent extensive H/V; deliberately has no G or S fields."""

    volume_m3: float
    enthalpy_J: float
