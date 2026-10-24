"""Read-only multi-ion aqueous queries; original species and H own all inventory."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_transport import AqueousTransportLaw
from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class MultiIonMembraneProperties:
    database: AqueousDatabase
    viscosity: AqueousTransportLaw

    def __post_init__(self) -> None:
        if (
            len(self.ion_ids) < 2
            or len(self.component_ids) != len(self.ion_ids) + 1
            or self.database.reactions
            or any(self.charges[k] == 0 for k in self.ion_ids)
            or not min(self.charges.values()) < 0 < max(self.charges.values())
            or any(
                s.standard_state.pressure_model != "constant-volume" for s in self.database.species
            )
            or self.water_volume_m3_mol <= 0
            or self.viscosity.unit != "Pa*s"
            or self.viscosity.source_id not in {s.source_id for s in self.database.sources}
        ):
            raise ValueError(
                "membrane properties require explicit water/charged ions, V and viscosity"
            )

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
    def ion_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.database.species if s.phase == "aqueous")

    @property
    def charges(self) -> dict[str, int]:
        return {s.species_id: s.charge for s in self.database.species if s.phase == "aqueous"}

    @property
    def temperature_bounds_K(self) -> tuple[float, float]:  # noqa: N802
        return self.viscosity.temperature_bounds_K

    def validate_amounts(self, amounts: Mapping[str, float], *, empty: bool = False) -> None:
        if set(amounts) != set(self.component_ids) or any(
            not isfinite(v) or v < 0 for v in amounts.values()
        ):
            raise ValueError(
                "multi-ion membrane needs all original nonnegative species inventories"
            )
        if empty and not any(amounts.values()):
            return
        if amounts[self.water_id] <= 0:
            raise ValueError("nonempty multi-ion membrane liquid requires actual solvent")
        charge = sum(self.charges[k] * amounts[k] for k in self.ion_ids)
        scale = sum(abs(self.charges[k]) * amounts[k] for k in self.ion_ids)
        if abs(charge) > 1e-10 * max(scale, 1e-5):
            raise ValueError("multi-ion membrane inventory is not electroneutral")

    def concentrations(self, amounts: Mapping[str, float]) -> dict[str, float]:
        return {
            k: amounts[k] / (amounts[self.water_id] * self.water_volume_m3_mol)
            for k in self.ion_ids
        }

    def amounts(self, water_mol: float, concentrations: Mapping[str, float]) -> dict[str, float]:
        if set(concentrations) != set(self.ion_ids):
            raise ValueError("all declared ion concentrations are required")
        return {
            self.water_id: water_mol,
            **{k: water_mol * self.water_volume_m3_mol * c for k, c in concentrations.items()},
        }

    def liquid(self, n: Mapping[str, float], t: float, p: float) -> Any:
        return aqueous_phase_properties(self.database, n, temperature_K=t, pressure_Pa=p)

    def enthalpy(self, n: Mapping[str, float], t: float, p: float) -> float:
        return self.liquid(n, t, p).enthalpy_J if any(n.values()) else 0.0

    def volume(self, n: Mapping[str, float], t: float, p: float) -> float:
        return self.liquid(n, t, p).volume_m3 if any(n.values()) else 0.0

    def enthalpy_rate(
        self, n: Mapping[str, float], rates: Mapping[str, float], t: float, p: float
    ) -> float:
        if not any(n.values()):
            if any(v < 0 for v in rates.values()):
                raise ValueError("empty membrane receiver cannot discharge")
            return self.enthalpy(rates, t, p)
        states = self.database.standard_states(t, p)
        standard = sum(rates[k] * states[k].enthalpy_J_mol for k in self.component_ids)
        if self.database.activity_model.family == "ideal":
            return standard

        def excess(x: Mapping[str, Any]) -> Any:
            current = {k: n[k] + x["extent"] * rates[k] for k in self.component_ids}
            kg = current[self.water_id] * self.water_mass_kg_mol
            return kg * self.database.activity_model.excess_gibbs_RT_mol_kg(
                {k: current[k] / kg for k in self.ion_ids}, x["T"], p
            )

        return standard - R * t * t * partial_derivative(
            excess, {"T": t, "extent": 0.0}, ("T", "extent")
        )

    def to_dict(self) -> dict[str, Any]:
        return {"database": self.database.to_dict(), "viscosity": self.viscosity.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultiIonMembraneProperties":
        return cls(
            AqueousDatabase.from_dict(payload["database"]),
            AqueousTransportLaw(**payload["viscosity"]),
        )
