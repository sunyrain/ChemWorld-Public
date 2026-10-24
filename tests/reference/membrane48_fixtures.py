"""Explicit synthetic thermal-membrane declarations; no measured-material claim."""

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_transport import AqueousTransportLaw, BinaryAqueousProperties
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel, ElectrolyteParameter
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.membrane_distillation import MembraneDistillationSpec
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo

SOURCE, BASIS = "synthetic-md48", "synthetic-md48-common"
T_BOUNDS, P_BOUNDS = (280.0, 340.0), (100000.0, 2000000.0)
ROWS = (
    ("H2O", "H2O", 0, 75.0, 0.0, 18e-6),
    ("Na+", "Na", 1, 40.0, 1000.0, 2e-6),
    ("Cl-", "Cl", -1, 35.0, -500.0, 3e-6),
)


def species(key, formula, charge, cp, href, volume, *, phase="liquid", entropy=10.0):
    correlation = PropertyCorrelation(
        key + "-cp",
        "cp_polynomial",
        {"a": cp},
        {"temperature": "K"},
        "J/(mol*K)",
        "heat_capacity",
        {"temperature": T_BOUNDS},
        "Synthetic constant Cp",
        {"phase": phase},
    )
    thermo = CorrelationSpeciesThermo(
        key,
        correlation,
        CaloricReference(
            phase, 298.15, 100000.0, "enthalpy", href, entropy, "Synthetic shared gauge"
        ),
        parse_formula(formula),
    )
    return AqueousSpecies(
        key,
        parse_formula(formula),
        charge,
        "gas" if phase == "gas" else "solvent" if charge == 0 else "aqueous",
        AqueousStandardState(
            thermo,
            BASIS,
            "gas-fugacity-reference" if phase == "gas" else "constant-volume",
            P_BOUNDS,
            SOURCE,
            volume,
        ),
        SOURCE,
    )


def spec(family="ideal"):
    source = AqueousDataSource(
        SOURCE,
        "Authored reduced Knudsen and constant-Cp equations",
        "urn:chemworld:synthetic:md48",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
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
    db = AqueousDatabase(
        SOURCE + family, (source,), tuple(species(*row) for row in ROWS), (), activity
    )
    properties = BinaryAqueousProperties(
        db,
        (("Na+", 1), ("Cl-", 1)),
        AqueousTransportLaw(0.001, "Pa*s", SOURCE, T_BOUNDS, P_BOUNDS, 500),
    )
    return MembraneDistillationSpec(
        properties,
        species("H2O(v)", "H2O", 0, 34.0, 44000.0, None, phase="gas", entropy=130.0),
        SOURCE,
        1e-4,
        1e-8,
        1e-4,
        0.5,
        2.0,
        0.002,
        2.7e-10,
        1e6,
        (0.04, 0.04),
    )


def amounts(model, water=1.0, concentration=50.0):
    return model.properties.amounts(water, concentration * water * 18e-6)


def devices(enabled=True):
    return (
        ThermalDeviceSpec(
            0.05, 0.25, 325.0, enabled=enabled, minimum_setpoint_K=280, maximum_setpoint_K=340
        ),
        ThermalDeviceSpec(
            0.05,
            0.0,
            295.0,
            enabled=enabled,
            maximum_cooling_power_W=0.2,
            minimum_setpoint_K=280,
            maximum_setpoint_K=340,
        ),
    )
