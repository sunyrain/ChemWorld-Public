"""Explicit synthetic declarations; none of these coefficients is material data."""

from dataclasses import replace

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousReaction,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.electrolyte_activity import (
    ElectrolyteActivityModel,
    ElectrolyteInteraction,
    ElectrolyteParameter,
    electrolyte_activity_catalog,
)
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo

SOURCE = "synthetic-aqueous39"
T_BOUNDS, P_BOUNDS = (275.0, 350.0), (100000.0, 1000000.0)
VALUES = {
    "A": 0.509,
    "B": 1.5,
    "b": 0.3,
    "Aphi": 0.392,
    "epsilon": 0.03,
    "beta0": 0.07,
    "beta1": 0.2,
    "beta2": 0.015,
    "C0": 0.0004,
    "C1": 0.0002,
    "alpha1": 2.0,
    "alpha2": 12.0,
    "omega": 2.5,
    "theta": 0.03,
    "psi": 0.005,
    "lambda": 0.02,
    "zeta": 0.007,
    "mu": 0.001,
}


def parameter(value, unit):
    return ElectrolyteParameter(value, unit, SOURCE, T_BOUNDS, P_BOUNDS)


def activity(family="Pitzer", charges=(("Na+", 1), ("Cl-", -1))):
    catalog = electrolyte_activity_catalog()[family]
    parameters = {
        k: parameter(1.2 if family == "Pitzer" and k == "b" else VALUES[k], unit)
        for k, unit in catalog["parameters"].items()
    }
    model = ElectrolyteActivityModel(
        SOURCE + "-" + family, family, charges, SOURCE, 5.0, T_BOUNDS, P_BOUNDS, parameters
    )
    interactions = tuple(
        ElectrolyteInteraction(
            kind,
            ids,
            {
                key: parameter(VALUES[key], unit)
                for key, unit in catalog["interactions"][kind].items()
            },
        )
        for kind, ids in model.required_interactions()
    )
    return replace(model, interactions=interactions)


def mixed_activity():
    return activity(charges=(("Na+", 1), ("Ca++", 2), ("Cl-", -1), ("SO4--", -2), ("CO2*", 0)))


def database():
    rows = (
        ("H2O", "H2O", 0, "solvent"),
        ("H+", "H", 1, "aqueous"),
        ("OH-", "OH", -1, "aqueous"),
        ("Na+", "Na", 1, "aqueous"),
        ("Cl-", "Cl", -1, "aqueous"),
        ("Ca++", "Ca", 2, "aqueous"),
        ("CO2*", "CO2", 0, "aqueous"),
        ("HCO3-", "HCO3", -1, "aqueous"),
        ("CO3--", "CO3", -2, "aqueous"),
        ("Calcite", "CaCO3", 0, "mineral"),
        ("Aragonite", "CaCO3", 0, "mineral"),
        ("CO2g", "CO2", 0, "gas"),
    )
    species = []
    for i, (key, formula, charge, phase) in enumerate(rows):
        caloric_phase = {
            "solvent": "liquid",
            "aqueous": "liquid",
            "mineral": "solid",
            "gas": "gas",
        }[phase]
        cp = PropertyCorrelation(
            key + "-cp",
            "cp_polynomial",
            {"a": 30.0 + i * 2},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": T_BOUNDS},
            "Artificial constant Cp, no substance-data claim",
            {"phase": caloric_phase},
        )
        ref = CaloricReference(
            caloric_phase,
            298.15,
            100000.0,
            "enthalpy",
            i * 300.0,
            20.0 + i,
            "Artificial shared elemental reference gauge",
        )
        thermo = CorrelationSpeciesThermo(key, cp, ref, parse_formula(formula))
        standard = AqueousStandardState(
            thermo,
            "synthetic-shared-gauge",
            "gas-fugacity-reference" if phase == "gas" else "constant-volume",
            P_BOUNDS,
            SOURCE,
            None if phase == "gas" else (i + 1) * 1e-5,
        )
        species.append(AqueousSpecies(key, parse_formula(formula), charge, phase, standard, SOURCE))
    reactions = (
        AqueousReaction("water", {"H2O": -1, "H+": 1, "OH-": 1}, SOURCE),
        AqueousReaction("acid1", {"H2O": -1, "CO2*": -1, "H+": 1, "HCO3-": 1}, SOURCE),
        AqueousReaction("acid2", {"HCO3-": -1, "H+": 1, "CO3--": 1}, SOURCE),
        AqueousReaction("calcite", {"Calcite": -1, "Ca++": 1, "CO3--": 1}, SOURCE),
        AqueousReaction("polymorph", {"Calcite": -1, "Aragonite": 1}, SOURCE),
        AqueousReaction("gas", {"CO2g": -1, "CO2*": 1}, SOURCE),
    )
    source = AqueousDataSource(
        SOURCE,
        "Artificial deterministic fixture, not material data",
        "urn:chemworld:synthetic:aqueous39",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
    )
    charges = tuple((s.species_id, s.charge) for s in species if s.phase == "aqueous")
    return AqueousDatabase(
        SOURCE, (source,), tuple(species), reactions, activity("Davies", charges)
    )


def inventory(db):
    water_mass = molecular_weight(dict(db.solvent_species.composition)) / 1000
    return {s.species_id: 1 / water_mass if s.phase == "solvent" else 0.001 for s in db.species}
