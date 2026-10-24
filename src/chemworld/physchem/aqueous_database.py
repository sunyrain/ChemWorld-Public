"""Source-aware species/reaction databases and actual aqueous chemical potentials.

The database owns declarations only. Temperature thermochemistry is delegated
to the existing species-thermo contract, activities to electrolyte_activity, and
caller-owned finite solvent amounts determine the molality scale. This module
does not solve a phase split, repair charge or own evolving material.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import exp, fsum, isfinite, log
from types import MappingProxyType
from typing import Any, cast

from chemworld.physchem.electrolyte_activity import (
    ElectrolyteActivityModel,
    ElectrolyteActivityResult,
)
from chemworld.physchem.elements import element_matrix, molecular_weight
from chemworld.physchem.property_reports import (
    R_J_PER_MOL_K,
    PropertyEvaluation,
    StatePropertyReport,
)
from chemworld.physchem.stoichiometry import stoichiometric_matrix

_UNITS = {
    "temperature": "K",
    "pressure": "Pa",
    "amount": "mol",
    "enthalpy": "J/mol",
    "entropy": "J/(mol*K)",
    "heat_capacity": "J/(mol*K)",
    "molar_volume": "m^3/mol",
    "molality": "mol/kg",
}


@dataclass(frozen=True)
class AqueousDataSource:
    source_id: str
    citation: str
    source_url: str
    license_id: str
    license_url: str
    redistribution: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v.strip() for v in asdict(self).values()
        ) or self.redistribution not in ("permitted", "restricted", "unknown", "synthetic"):
            raise ValueError("database source needs citation, location, license and redistribution")


@dataclass(frozen=True)
class AqueousStandardStateResult:
    temperature_K: float
    pressure_Pa: float
    reference_pressure_Pa: float
    heat_capacity_J_mol_K: float
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    gibbs_J_mol: float
    pressure_model: str
    reference_basis_id: str
    molar_volume_m3_mol: float | None = None


@dataclass(frozen=True)
class AqueousStandardState:
    thermo: Any
    reference_basis_id: str
    pressure_model: str
    pressure_bounds_Pa: tuple[float, float]
    source_id: str
    molar_volume_m3_mol: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pressure_bounds_Pa", tuple(self.pressure_bounds_Pa))
        low, high = self.pressure_bounds_Pa
        if (
            not self.reference_basis_id
            or not self.source_id
            or not 0 < low <= high
            or not isfinite(low)
            or not isfinite(high)
            or self.pressure_model
            not in ("reference-only", "constant-volume", "gas-fugacity-reference", "aqueous-gibbs")
            or not hasattr(self.thermo, "evaluate")
            or not hasattr(self.thermo, "to_dict")
            or not low <= self.thermo.reference_pressure_Pa <= high
        ):
            raise ValueError("aqueous standard state needs thermo, basis, pressure law and source")
        if self.pressure_model == "aqueous-gibbs" and (
            not hasattr(self.thermo, "evaluate_tp")
            or self.thermo.reference_basis.basis_id != self.reference_basis_id
        ):
            raise ValueError("aqueous Gibbs state requires its explicit matching reference basis")
        if self.pressure_model == "constant-volume":
            if (
                self.molar_volume_m3_mol is None
                or isinstance(self.molar_volume_m3_mol, bool)
                or not isfinite(self.molar_volume_m3_mol)
            ):
                raise ValueError(
                    "constant-volume standard state needs explicit finite molar volume"
                )
        elif self.molar_volume_m3_mol is not None:
            raise ValueError("molar volume is only used by the declared constant-volume law")

    def evaluate(self, temperature_K: float, pressure_Pa: float) -> AqueousStandardStateResult:
        if (
            not isfinite(pressure_Pa)
            or not self.pressure_bounds_Pa[0] <= pressure_Pa <= self.pressure_bounds_Pa[1]
        ):
            raise ValueError("standard-state pressure outside declared domain")
        pr = self.thermo.reference_pressure_Pa
        if self.pressure_model == "reference-only" and pressure_Pa != pr:
            raise ValueError("standard state has no declared pressure dependence")
        state = (self.thermo.evaluate_tp(temperature_K, pressure_Pa)
                 if self.pressure_model == "aqueous-gibbs" else self.thermo.evaluate(temperature_K))
        shift = (
            self.molar_volume_m3_mol * (pressure_Pa - pr)
            if self.molar_volume_m3_mol is not None
            else 0
        )
        return AqueousStandardStateResult(
            temperature_K,
            pressure_Pa,
            pr,
            state.cp_J_mol_K,
            state.enthalpy_J_mol + shift,
            state.entropy_J_mol_K,
            state.gibbs_J_mol + shift,
            self.pressure_model,
            self.reference_basis_id,
            (state.molar_volume_m3_mol if self.pressure_model == "aqueous-gibbs"
             else self.molar_volume_m3_mol),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thermo": self.thermo.to_dict(),
            "reference_basis_id": self.reference_basis_id,
            "pressure_model": self.pressure_model,
            "pressure_bounds_Pa": list(self.pressure_bounds_Pa),
            "source_id": self.source_id,
            "molar_volume_m3_mol": self.molar_volume_m3_mol,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AqueousStandardState:
        from chemworld.physchem.thermochemistry import species_thermo_from_dict

        data = dict(payload)
        if data["pressure_model"] == "aqueous-gibbs":
            from chemworld.physchem.aqueous_hkf import aqueous_gibbs_thermo_from_dict

            data["thermo"] = aqueous_gibbs_thermo_from_dict(data["thermo"])
        else:
            data["thermo"] = species_thermo_from_dict(data["thermo"])
        return cls(**data)


@dataclass(frozen=True)
class AqueousSpecies:
    species_id: str
    composition: Mapping[str, float]
    charge: int
    phase: str
    standard_state: AqueousStandardState
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "composition", MappingProxyType(dict(self.composition)))
        element_matrix([dict(self.composition)])
        if (
            not self.species_id
            or not self.source_id
            or type(self.charge) is not int
            or self.phase not in ("aqueous", "solvent", "gas", "mineral")
            or self.standard_state.thermo.species_id != self.species_id
        ):
            raise ValueError(
                "database species needs explicit identity, composition, charge and phase"
            )
        if self.phase != "aqueous" and self.charge != 0:
            raise ValueError("this database declares charged species only in its aqueous phase")
        if (self.phase == "gas") != (
            self.standard_state.pressure_model == "gas-fugacity-reference"
        ):
            raise ValueError(
                "gas standard states require the explicit fugacity/reference-pressure basis"
            )
        thermo_composition = getattr(self.standard_state.thermo, "composition", {})
        if thermo_composition and dict(thermo_composition) != dict(self.composition):
            raise ValueError("species and thermochemistry compositions disagree")
        if getattr(self.standard_state.thermo, "charge", self.charge) != self.charge:
            raise ValueError("species and thermochemistry charges disagree")
        if self.standard_state.pressure_model == "aqueous-gibbs" and self.phase not in (
            "aqueous", "solvent"
        ):
            raise ValueError("aqueous Gibbs standard states require a liquid species phase")
        reference = getattr(self.standard_state.thermo, "reference", None)
        phase = {"aqueous": "liquid", "solvent": "liquid", "mineral": "solid", "gas": "gas"}
        if reference is not None and reference.phase != phase[self.phase]:
            raise ValueError("species and caloric-reference phases disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "composition": dict(self.composition),
            "charge": self.charge,
            "phase": self.phase,
            "source_id": self.source_id,
            "standard_state": self.standard_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AqueousSpecies:
        data = dict(payload)
        data["standard_state"] = AqueousStandardState.from_dict(data["standard_state"])
        return cls(**data)


@dataclass(frozen=True)
class AqueousReaction:
    reaction_id: str
    stoichiometry: Mapping[str, float]
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stoichiometry", MappingProxyType(dict(self.stoichiometry)))
        if (
            not self.reaction_id
            or not self.source_id
            or len(self.stoichiometry) < 2
            or any(
                not k or isinstance(v, bool) or not isfinite(v) or not v
                for k, v in self.stoichiometry.items()
            )
            or not any(v < 0 for v in self.stoichiometry.values())
            or not any(v > 0 for v in self.stoichiometry.values())
        ):
            raise ValueError(
                "reaction needs signed finite reactant/product stoichiometry and source"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "stoichiometry": dict(self.stoichiometry),
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class AqueousChemicalPotentialReport:
    database_id: str
    temperature_K: float
    pressure_Pa: float
    solvent_mass_kg: float
    amounts_mol: dict[str, float]
    standard_states: dict[str, AqueousStandardStateResult]
    log_activities: dict[str, float | None]
    chemical_potentials_J_mol: dict[str, float | None]
    electrolyte: ElectrolyteActivityResult

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AqueousDatabase:
    database_id: str
    sources: tuple[AqueousDataSource, ...]
    species: tuple[AqueousSpecies, ...]
    reactions: tuple[AqueousReaction, ...]
    activity_model: ElectrolyteActivityModel

    def __post_init__(self) -> None:
        for name in ("sources", "species", "reactions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.database_id or not self.species or not self.sources:
            raise ValueError("database requires identity, species and source declarations")
        for values, key in (
            (self.sources, "source_id"),
            (self.species, "species_id"),
            (self.reactions, "reaction_id"),
        ):
            if len({getattr(v, key) for v in values}) != len(values):
                raise ValueError("duplicate database declaration identity")
        source_ids = {s.source_id for s in self.sources}
        used_sources = {
            *self.activity_model.source_ids(),
            *(s.source_id for s in self.species),
            *(s.standard_state.source_id for s in self.species),
            *(r.source_id for r in self.reactions),
        }
        for species in self.species:
            if species.standard_state.pressure_model == "aqueous-gibbs":
                used_sources.update((species.standard_state.thermo.source_id,
                                     species.standard_state.thermo.reference_basis.source_id))
        if used_sources - source_ids:
            raise ValueError(
                "undeclared data sources: " + ", ".join(sorted(used_sources - source_ids))
            )
        if len({s.standard_state.reference_basis_id for s in self.species}) != 1:
            raise ValueError("database species must explicitly share one chemical-potential basis")
        gibbs_bases = [s.standard_state.thermo.reference_basis.to_dict() for s in self.species
                       if s.standard_state.pressure_model == "aqueous-gibbs"]
        if gibbs_bases and any(basis != gibbs_bases[0] for basis in gibbs_bases[1:]):
            raise ValueError("aqueous Gibbs reference basis declarations disagree despite identity")
        solvents = [s for s in self.species if s.phase == "solvent"]
        if len(solvents) != 1 or dict(solvents[0].composition) != {"H": 2, "O": 1}:
            raise ValueError("aqueous database requires exactly one explicit H2O solvent")
        if {s.species_id: s.charge for s in self.species if s.phase == "aqueous"} != dict(
            self.activity_model.charges
        ):
            raise ValueError("activity declaration must cover every aqueous species and charge")
        declared = {s.species_id: s for s in self.species}
        for reaction in self.reactions:
            if set(reaction.stoichiometry) - set(declared):
                raise ValueError("reaction references undeclared species")
            ids = tuple(reaction.stoichiometry)
            matrix = stoichiometric_matrix(
                [dict(declared[k].composition) for k in ids],
                [reaction.stoichiometry[k] < 0 for k in ids],
                charges=[declared[k].charge for k in ids],
            )
            coefficients = [abs(Fraction(str(reaction.stoichiometry[k]))) for k in ids]
            if any(sum(v * c for v, c in zip(row, coefficients, strict=True)) for row in matrix):
                raise ValueError("unbalanced database reaction: " + reaction.reaction_id)

    @property
    def solvent_species(self) -> AqueousSpecies:
        return next(s for s in self.species if s.phase == "solvent")

    def standard_states(
        self, temperature_K: float, pressure_Pa: float
    ) -> dict[str, AqueousStandardStateResult]:
        return {
            s.species_id: s.standard_state.evaluate(temperature_K, pressure_Pa)
            for s in self.species
        }

    def reaction_standard_properties(
        self, temperature_K: float, pressure_Pa: float
    ) -> dict[str, dict[str, float]]:
        states = self.standard_states(temperature_K, pressure_Pa)
        result = {}
        for reaction in self.reactions:
            values = {
                field: fsum(
                    nu * getattr(states[key], field) for key, nu in reaction.stoichiometry.items()
                )
                for field in (
                    "gibbs_J_mol",
                    "enthalpy_J_mol",
                    "entropy_J_mol_K",
                    "heat_capacity_J_mol_K",
                )
            }
            result[reaction.reaction_id] = {"delta_" + k: v for k, v in values.items()}
            result[reaction.reaction_id]["log_equilibrium_constant"] = -values["gibbs_J_mol"] / (
                R_J_PER_MOL_K * temperature_K
            )
        return result

    def chemical_potentials(
        self,
        amounts_mol: Mapping[str, float],
        *,
        temperature_K: float,
        pressure_Pa: float,
        gas_fugacities_Pa: Mapping[str, float] | None = None,
    ) -> AqueousChemicalPotentialReport:
        declared = {s.species_id: s for s in self.species}
        if set(amounts_mol) - set(declared) or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in amounts_mol.values()
        ):
            raise ValueError("amounts require declared species and finite nonnegative moles")
        amounts = {key: float(amounts_mol.get(key, 0)) for key in declared}
        solvent = self.solvent_species
        mass = molecular_weight(dict(solvent.composition)) / 1000
        water_kg = amounts[solvent.species_id] * mass
        if water_kg <= 0:
            raise ValueError(
                "aqueous chemical potentials require actual positive solvent inventory"
            )
        molalities = {
            s.species_id: amounts[s.species_id] / water_kg
            for s in self.species
            if s.phase == "aqueous"
        }
        activities = self.activity_model.evaluate(
            molalities,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            solvent_molar_mass_kg_mol=mass,
        )
        f = {} if gas_fugacities_Pa is None else dict(gas_fugacities_Pa)
        if any(key not in declared or declared[key].phase != "gas" for key in f) or any(
            not isfinite(v) or v < 0 for v in f.values()
        ):
            raise ValueError(
                "gas fugacities must explicitly name declared gases and nonnegative Pa"
            )
        states = self.standard_states(temperature_K, pressure_Pa)
        log_activities = dict(activities.log_activities)
        log_activities[solvent.species_id] = activities.log_solvent_activity
        for s in self.species:
            key = s.species_id
            if s.phase == "mineral":
                log_activities[key] = 0
            elif s.phase == "gas":
                if amounts[key] and (key not in f or f[key] <= 0):
                    raise ValueError(
                        "present gas requires an explicit positive fugacity in Pa: " + key
                    )
                log_activities[key] = (
                    log(f[key] / states[key].reference_pressure_Pa) if f.get(key, 0) else None
                )
        potentials = {
            key: states[key].gibbs_J_mol + R_J_PER_MOL_K * temperature_K * value
            if value is not None
            else None
            for key, value in log_activities.items()
        }
        return AqueousChemicalPotentialReport(
            self.database_id,
            temperature_K,
            pressure_Pa,
            water_kg,
            amounts,
            states,
            log_activities,
            potentials,
            activities,
        )

    def reaction_affinities(
        self, report: AqueousChemicalPotentialReport
    ) -> dict[str, dict[str, Any]]:
        if report.database_id != self.database_id:
            raise ValueError("chemical potential report belongs to a different database")
        standard = self.reaction_standard_properties(report.temperature_K, report.pressure_Pa)
        results = {}
        for r in self.reactions:
            absent = [k for k in r.stoichiometry if report.chemical_potentials_J_mol[k] is None]
            dg = (
                fsum(
                    nu * cast(float, report.chemical_potentials_J_mol[k])
                    for k, nu in r.stoichiometry.items()
                )
                if not absent
                else None
            )
            log_q = (
                fsum(
                    nu * cast(float, report.log_activities[k]) for k, nu in r.stoichiometry.items()
                )
                if not absent
                else None
            )
            results[r.reaction_id] = {
                **standard[r.reaction_id],
                "log_reaction_quotient": log_q,
                "reaction_gibbs_J_mol": dg,
                "affinity_J_mol": -dg if dg is not None else None,
                "absent_species": absent,
            }
        return results

    def property_report(
        self, amounts_mol: Mapping[str, float], *, temperature_K: float, pressure_Pa: float
    ) -> StatePropertyReport:
        """Existing liquid StatePropertyReport consumer; no bulk caloric closure is inferred."""
        liquid = {s.species_id for s in self.species if s.phase in ("solvent", "aqueous")}
        if set(amounts_mol) - liquid:
            raise ValueError("single-liquid property report accepts only water and aqueous species")
        report = self.chemical_potentials(
            amounts_mol, temperature_K=temperature_K, pressure_Pa=pressure_Pa
        )
        inputs = {"temperature_K": temperature_K, "pressure_Pa": pressure_Pa}
        chemical = {
            k: PropertyEvaluation(
                "chemical_potential", self.database_id, "mu0_plus_RT_lna", value, "J/mol", inputs
            )
            for k, value in report.chemical_potentials_J_mol.items()
            if k in liquid and value is not None
        }
        gamma = {
            k: PropertyEvaluation(
                "activity_coefficient",
                self.activity_model.model_id,
                self.activity_model.family,
                exp(v),
                "1",
                inputs,
            )
            for k, v in report.electrolyte.log_activity_coefficients.items()
        }
        return StatePropertyReport(
            self.database_id,
            "liquid",
            temperature_K,
            pressure_Pa,
            {k: report.amounts_mol[k] for k in liquid},
            {},
            component_properties={"chemical_potential": chemical, "activity_coefficient": gamma},
            model_details={
                "standard_state_units": dict(_UNITS),
                "activity_basis": "molality / (1 mol/kg)",
                "solvent_activity": report.electrolyte.solvent_activity,
                "ionic_strength_mol_kg": report.electrolyte.ionic_strength_mol_kg,
                "net_charge_mol_kg": report.electrolyte.net_charge_mol_kg,
                "absent_chemical_potentials": sorted(liquid - set(chemical)),
                "source_ids": [s.source_id for s in self.sources],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-aqueous-database-v1",
            "database_id": self.database_id,
            "units": dict(_UNITS),
            "sources": [asdict(s) for s in self.sources],
            "species": [s.to_dict() for s in self.species],
            "reactions": [r.to_dict() for r in self.reactions],
            "activity_model": self.activity_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AqueousDatabase:
        data = dict(payload)
        if (
            data.pop("model", None) != "chemworld-aqueous-database-v1"
            or data.pop("units", None) != _UNITS
        ):
            raise ValueError("unknown aqueous database schema or noncanonical units")
        data["sources"] = tuple(AqueousDataSource(**v) for v in data["sources"])
        data["species"] = tuple(AqueousSpecies.from_dict(v) for v in data["species"])
        data["reactions"] = tuple(AqueousReaction(**v) for v in data["reactions"])
        data["activity_model"] = ElectrolyteActivityModel.from_dict(data["activity_model"])
        return cls(**data)
