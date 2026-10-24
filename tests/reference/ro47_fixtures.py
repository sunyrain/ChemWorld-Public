"""Synthetic CC0 binary-electrolyte RO declarations, not material data."""

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_transport import AqueousTransportLaw, BinaryAqueousProperties
from chemworld.physchem.distributed_ro import DistributedROSpec
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel, ElectrolyteParameter
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo
from chemworld.physchem.transport import PipeSpec, PumpSpec

SOURCE = "synthetic-ro47"
T_BOUNDS, P_BOUNDS = (280.0, 340.0), (100000.0, 2000000.0)
# Match the original elemental-mass declaration H=1.008, O=15.999 g/mol.
WATER_VOLUME, WATER_MASS = 18e-6, 0.018015
ROWS = (
    ("H2O", "H2O", 0, 75.0, 0.0, WATER_VOLUME),
    ("Na+", "Na", 1, 40.0, 1000.0, 2e-6),
    ("Cl-", "Cl", -1, 35.0, -500.0, 3e-6),
)


def law(value, unit, **kwargs):
    return AqueousTransportLaw(value, unit, SOURCE, T_BOUNDS, P_BOUNDS, 500.0, **kwargs)


def spec(family="Davies", *, temperature=310.0, length=1.0):
    source = AqueousDataSource(
        SOURCE,
        "Synthetic authored equations; no substance data claim",
        "urn:chemworld:synthetic:ro47",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
    )
    species = []
    for key, formula, charge, cp, h, volume in ROWS:
        correlation = PropertyCorrelation(
            key + "-cp",
            "cp_polynomial",
            {"a": cp},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": T_BOUNDS},
            "Synthetic constant Cp",
            {"phase": "liquid"},
        )
        thermo = CorrelationSpeciesThermo(
            key,
            correlation,
            CaloricReference(
                "liquid", 298.15, 100000.0, "enthalpy", h, 10.0, "Synthetic shared gauge"
            ),
            parse_formula(formula),
        )
        species.append(
            AqueousSpecies(
                key,
                parse_formula(formula),
                charge,
                "solvent" if not charge else "aqueous",
                AqueousStandardState(
                    thermo, "synthetic-ro47-common", "constant-volume", P_BOUNDS, SOURCE, volume
                ),
                SOURCE,
            )
        )
    parameters = (
        {}
        if family == "ideal"
        else {
            "A": ElectrolyteParameter(
                0.509, "(kg/mol)^0.5", SOURCE, T_BOUNDS, P_BOUNDS, per_K=2e-4, per_Pa=1e-9
            ),
            "b": ElectrolyteParameter(0.3, "(kg/mol)^0.5", SOURCE, T_BOUNDS, P_BOUNDS),
        }
    )
    activity = ElectrolyteActivityModel(
        SOURCE + family,
        family,
        (("Na+", 1), ("Cl-", -1)),
        SOURCE,
        1.0,
        T_BOUNDS,
        P_BOUNDS,
        parameters,
    )
    db = AqueousDatabase(SOURCE + family, (source,), tuple(species), (), activity)
    properties = BinaryAqueousProperties(
        db,
        (("Na+", 1), ("Cl-", 1)),
        law(0.001, "Pa*s", activation_energy_J_mol=-10000, per_mol_m3=1e-3),
    )
    return DistributedROSpec(
        properties,
        law(2e-12, "m/(Pa*s)", activation_energy_J_mol=12000),
        law(1e-7, "m/s", activation_energy_J_mol=8000),
        law(1e-5, "m/s", activation_energy_J_mol=3000),
        PipeSpec(length, 0.0015),
        PumpSpec(1.5e6, 1.0, 0.75),
        0.01,
        temperature,
        1e5,
        5e5,
        0.2,
        0.5,
        maximum_interface_concentration_mol_m3=200,
    )


def feed(model, concentration=30.0, water_mol=20.0):
    return model.properties.amounts(water_mol, concentration * water_mol * WATER_VOLUME)
