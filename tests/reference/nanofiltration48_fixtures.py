"""Authored ideal dilute-ion NF tables; synthetic declarations, not real membranes."""

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_transport import AqueousTransportLaw
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.nanofiltration import NanofiltrationSpec, PoreIon
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo
from chemworld.physchem.transport import PumpSpec

SOURCE = "synthetic-nf48"
T_BOUNDS, P_BOUNDS = (280.0, 340.0), (1e5, 2e6)
ROWS = {
    "H2O": ("H2O", 0, 75.0, 0.0, 18e-6),
    "Na+": ("Na", 1, 40.0, 1000.0, 2e-6),
    "Ca2+": ("Ca", 2, 45.0, 500.0, 3e-6),
    "Cl-": ("Cl", -1, 35.0, -500.0, 3e-6),
    "SO4--": ("SO4", -2, 50.0, -1000.0, 4e-6),
}
PORE = {
    "Na+": (0.2e-9, 0.25e-9, 1.3e-9),
    "Ca2+": (0.25e-9, 0.3e-9, 0.8e-9),
    "Cl-": (0.2e-9, 0.25e-9, 1.3e-9),
    "SO4--": (0.3e-9, 0.35e-9, 1e-9),
}


def spec(ions=3, charge=-20.0, pressure=5e5, segments=16):
    ids = {2: ("Na+", "Cl-"), 3: ("Na+", "Ca2+", "Cl-"), 4: ("Na+", "Ca2+", "Cl-", "SO4--")}[ions]
    species = []
    for key in ("H2O", *ids):
        formula, z, cp, href, volume = ROWS[key]
        curve = PropertyCorrelation(
            key + "-Cp",
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
            curve,
            CaloricReference(
                "liquid", 298.15, 1e5, "enthalpy", href, 10.0, "Synthetic common gauge"
            ),
            parse_formula(formula),
        )
        species.append(
            AqueousSpecies(
                key,
                parse_formula(formula),
                z,
                "aqueous" if z else "solvent",
                AqueousStandardState(thermo, SOURCE, "constant-volume", P_BOUNDS, SOURCE, volume),
                SOURCE,
            )
        )
    source = AqueousDataSource(
        SOURCE,
        "Authored DSPM-DE constant-Cp dilute fixtures",
        "urn:chemworld:synthetic:nf48",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
    )
    activity = ElectrolyteActivityModel(
        SOURCE, "ideal", tuple((k, ROWS[k][1]) for k in ids), SOURCE, 1.0, T_BOUNDS, P_BOUNDS, {}
    )
    db = AqueousDatabase(SOURCE, (source,), tuple(species), (), activity)
    properties = MultiIonMembraneProperties(
        db, AqueousTransportLaw(0.001, "Pa*s", SOURCE, T_BOUNDS, P_BOUNDS, 500)
    )
    return NanofiltrationSpec(
        properties,
        tuple(PoreIon(k, *PORE[k], SOURCE) for k in ids),
        PumpSpec(1.5e6, 1.0, 0.75),
        1e-5,
        1e-9,
        1e-6,
        charge,
        78.5,
        60,
        310.0,
        1e5,
        pressure,
        0.04,
        0.04,
        SOURCE,
        pore_segments=segments,
    )


def concentrations(model):
    return {
        2: {"Na+": 10.0, "Cl-": 10.0},
        3: {"Na+": 10.0, "Ca2+": 5.0, "Cl-": 20.0},
        4: {"Na+": 10.0, "Ca2+": 5.0, "Cl-": 10.0, "SO4--": 5.0},
    }[len(model.ions)]


def amounts(model, water=1.0, dilution=1.0):
    return model.properties.amounts(
        water, {k: c * dilution for k, c in concentrations(model).items()}
    )
