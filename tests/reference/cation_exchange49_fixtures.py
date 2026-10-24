"""Synthetic finite ionic liquid and Na/Ca resin qualification inputs."""

from dataclasses import replace

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_transport import AqueousTransportLaw
from chemworld.physchem.cation_exchange import (
    CationExchangeSpec,
    cation_exchange_inventory,
)
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo
from chemworld.physchem.transport import PipeSpec, PumpSpec

SOURCE = "synthetic-cation-49D"
DATA = {
    "water": ("H2O", 75.0, 1000.0, 0),
    "Na": ("Na", 30.0, 500.0, 1),
    "Ca": ("Ca", 35.0, 700.0, 2),
    "Cl": ("Cl", 40.0, 300.0, -1),
}
WATER_VOLUME = 18e-6
WATER_MASS = 0.018015


def spec_case(temperature=298.15, *, kinetics=True):
    species = []
    for name, (formula, cp, h, charge) in DATA.items():
        composition = parse_formula(formula)
        corr = PropertyCorrelation(
            name + ":cp",
            "cp_polynomial",
            {"a": cp},
            {"temperature": "K"},
            "J/(mol*K)",
            "heat_capacity",
            {"temperature": (280.0, 340.0)},
            "Synthetic qualification Cp",
            {"phase": "liquid"},
        )
        thermo = CorrelationSpeciesThermo(
            name,
            corr,
            CaloricReference(
                "liquid", 298.15, 1e5, "enthalpy", h, 0.0, "Synthetic common caloric datum"
            ),
            composition,
        )
        standard = AqueousStandardState(
            thermo,
            "synthetic49D-common",
            "constant-volume",
            (1e4, 2e6),
            SOURCE,
            WATER_VOLUME if name == "water" else 0.0,
        )
        species.append(
            AqueousSpecies(
                name,
                composition,
                charge,
                "solvent" if name == "water" else "aqueous",
                standard,
                SOURCE,
            )
        )
    db = AqueousDatabase(
        SOURCE,
        (
            AqueousDataSource(
                SOURCE,
                "Synthetic finite resin qualification",
                "urn:chemworld:synthetic:49D",
                "CC0-1.0",
                "https://creativecommons.org/publicdomain/zero/1.0/",
                "synthetic",
            ),
        ),
        tuple(species),
        (),
        ElectrolyteActivityModel(
            "synthetic49D-ideal",
            "ideal",
            (("Na", 1), ("Ca", 2), ("Cl", -1)),
            SOURCE,
            1.0,
            (280.0, 340.0),
            (1e4, 2e6),
        ),
    )
    properties = MultiIonMembraneProperties(
        db, AqueousTransportLaw(0.001, "Pa*s", SOURCE, (280.0, 340.0), (1e4, 2e6), 250.0)
    )
    return CationExchangeSpec(
        properties,
        ThermalDeviceSpec(
            mode="isothermal",
            setpoint_K=temperature,
            minimum_setpoint_K=280.0,
            maximum_setpoint_K=340.0,
            maximum_power_W=10.0,
            maximum_cooling_power_W=10.0,
        ),
        PumpSpec(1e5, 10.0, 0.7),
        PipeSpec(0.1, 0.002),
        0.001,
        1.0,
        10**0.8,
        -3000.0,
        100.0 if kinetics else 0.0,
        SOURCE,
        pickup_volume_L=0.005,
    )


def initial_amounts(spec, *, contact="ordinary"):
    def row(v, na=0.0, ca=0.0):
        return {
            "water": v / 1000 / WATER_VOLUME,
            "Na": na * v,
            "Ca": ca * v,
            "Cl": (na + 2 * ca) * v,
        }

    middle = {"ordinary": (0.005, 0.005), "salt": (0.08, 0.0), "rinse": (0.04, 0.002)}[contact]
    return (
        row(0.02, 0.005, 0.01),
        row(0.005, *middle),
        row(0.001),
        row(0.02, 0.08),
        row(0.02),
        row(0.001),
    )


def inventory_case(spec, *, fraction=0.0, contact="ordinary"):
    return cation_exchange_inventory(
        spec, initial_amounts(spec, contact=contact), initial_calcium_equivalent_fraction=fraction
    )


def fixed_cases():
    cases = []
    for temperature in (298.15, 318.15):
        for mode in ("batch-Na", "batch-Ca", "process", "regenerate", "rinse", "no-exchange"):
            spec = spec_case(temperature, kinetics=mode != "no-exchange")
            options = {
                "fraction": 1.0
                if mode in ("batch-Ca", "regenerate")
                else 0.5
                if mode == "rinse"
                else 0.0,
                "contact": "salt"
                if mode in ("batch-Ca", "regenerate")
                else "rinse"
                if mode == "rinse"
                else "ordinary",
            }
            cases.append(
                (
                    f"{mode}-{temperature}",
                    spec,
                    options,
                    0.0 if mode.startswith("batch") else 10 / 6e7,
                    100.0 if mode.startswith("batch") else 30.0,
                    mode if mode in ("regenerate", "rinse") else "process",
                    None,
                )
            )
    for mode in (
        "process-pickup",
        "regenerate-pickup",
        "rinse-pickup",
        "product-full",
        "waste-full",
        "pressure",
        "shaft",
    ):
        spec = spec_case()
        if mode in ("product-full", "waste-full"):
            capacities = list(spec.capacities_L)
            capacities[2 if mode == "product-full" else 5] = 0.002
            spec = replace(spec, capacities_L=tuple(capacities))
        elif mode == "pressure":
            spec = replace(spec, pump=PumpSpec(1.0, 10.0, 0.7))
        elif mode == "shaft":
            spec = replace(spec, pump=PumpSpec(1e5, 1e-7, 0.7))
        path = (
            "regenerate"
            if mode in ("regenerate-pickup", "waste-full")
            else "rinse"
            if mode == "rinse-pickup"
            else "process"
        )
        options = {"fraction": 0.5, "contact": "rinse"} if path != "process" else {}
        cases.append((mode, spec, options, 10 / 6e7, 200.0, path, None))
    for shared in (False, True):
        cases.append(
            (
                "world-shared-cycle" if shared else "world-local-cycle",
                spec_case(),
                {},
                10 / 6e7,
                30.0,
                "process",
                shared,
            )
        )
    return cases
