"""Synthetic finite neutral liquid and carbon-lot qualification inputs."""

from dataclasses import replace

from chemworld.physchem.aqueous_adsorption import (
    AqueousAdsorptionSpec,
    aqueous_adsorption_inventory,
)
from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_transport import AqueousTransportLaw
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.sorbent_kinetics import SorbentKineticsSpec
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo
from chemworld.physchem.transport import PipeSpec, PumpSpec

SOURCE = "synthetic-aqueous-49C"
DATA = {"water": ("H2O", 75.0, 1000.0), "A": ("CH2O", 80.0, 500.0), "B": ("C2H6O", 90.0, 700.0)}
AFFINITIES = {"A": 200.0, "B": 100.0}
RATES = {"A": 0.08, "B": 0.03}
ENERGIES = {"A": -2000.0, "B": -4000.0}
WATER_VOLUME = 18e-6


def spec_case(temperature=298.15, *, kinetics=True):
    species = []
    for name, (formula, cp, h) in DATA.items():
        composition = parse_formula(formula)
        correlation = PropertyCorrelation(
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
            correlation,
            CaloricReference(
                "liquid",
                298.15,
                1e5,
                "enthalpy",
                h,
                0.0,
                "Synthetic common-reference caloric datum",
            ),
            composition,
        )
        standard = AqueousStandardState(
            thermo,
            "synthetic49C-common",
            "constant-volume",
            (1e4, 2e6),
            SOURCE,
            WATER_VOLUME if name == "water" else 0.0,
        )
        species.append(
            AqueousSpecies(
                name, composition, 0, "solvent" if name == "water" else "aqueous", standard, SOURCE
            )
        )
    database = AqueousDatabase(
        SOURCE,
        (
            AqueousDataSource(
                SOURCE,
                "Synthetic finite adsorption qualification",
                "urn:chemworld:synthetic:49C",
                "CC0-1.0",
                "https://creativecommons.org/publicdomain/zero/1.0/",
                "synthetic",
            ),
        ),
        tuple(species),
        (),
        ElectrolyteActivityModel(
            "synthetic49C-ideal",
            "ideal",
            (("A", 0), ("B", 0)),
            SOURCE,
            1.0,
            (280.0, 340.0),
            (1e4, 2e6),
        ),
    )
    return AqueousAdsorptionSpec(
        database,
        SorbentKineticsSpec(
            "synthetic49C-carbon",
            1.0,
            AFFINITIES,
            RATES if kinetics else {"A": 0.0, "B": 0.0},
            ENERGIES,
            temperature_bounds_K=(280.0, 340.0),
        ),
        AqueousTransportLaw(0.001, "Pa*s", SOURCE, (280.0, 340.0), (1e4, 2e6), 250.0),
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
        pickup_volume_L=0.005,
    )


def initial_amounts(spec, *, single=False, preloaded=False):
    def row(volume, a=0.0, b=0.0):
        return {"water": volume / 1000 / WATER_VOLUME, "A": a * volume, "B": b * volume}

    return (
        row(0.02, 0.05, 0.0 if single else 0.025),
        row(0.005, 0.0 if preloaded else 0.02, 0.0 if single or preloaded else 0.01),
        row(0.001),
        row(0.02),
        row(0.001),
    )


def inventory_case(spec, *, single=False, preloaded=False):
    return aqueous_adsorption_inventory(
        spec,
        initial_amounts(spec, single=single, preloaded=preloaded),
        initial_loading_mol_per_kg={"A": 0.6, "B": 0.0 if single else 0.2} if preloaded else None,
    )


def fixed_cases():
    cases = []
    for temperature in (298.15, 318.15):
        for mode in (
            "batch-single",
            "batch-competitive",
            "flow-competitive",
            "rinse-preloaded",
            "no-adsorption",
        ):
            spec = spec_case(temperature, kinetics=mode != "no-adsorption")
            options = {"single": mode == "batch-single", "preloaded": mode == "rinse-preloaded"}
            cases.append(
                (
                    f"{mode}-{temperature}",
                    spec,
                    options,
                    0.0 if mode.startswith("batch") else 10 / 6e7,
                    1000.0 if mode == "batch-single" else 30.0,
                    "rinse" if mode == "rinse-preloaded" else "process",
                    None,
                )
            )
    spec = spec_case()
    for mode in (
        "process-pickup",
        "rinse-pickup",
        "product-full",
        "waste-full",
        "pressure",
        "shaft",
    ):
        model = spec
        if mode in {"product-full", "waste-full"}:
            capacities = list(spec.capacities_L)
            capacities[2 if mode == "product-full" else 4] = 0.002
            model = replace(spec, capacities_L=tuple(capacities))
        elif mode == "pressure":
            model = replace(spec, pump=PumpSpec(1.0, 10.0, 0.7))
        elif mode == "shaft":
            model = replace(spec, pump=PumpSpec(1e5, 1e-7, 0.7))
        path = "rinse" if mode in {"rinse-pickup", "waste-full"} else "process"
        cases.append((mode, model, {"preloaded": path == "rinse"}, 10 / 6e7, 200.0, path, None))
    for shared in (False, True):
        cases.append(
            (
                "world-shared-cycle" if shared else "world-local-cycle",
                spec,
                {},
                10 / 6e7,
                30.0,
                "process",
                shared,
            )
        )
    return cases
