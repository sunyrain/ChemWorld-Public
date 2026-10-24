"""Synthetic CC0 through-cycle and multistable adsorption models."""

from math import log

from chemworld.physchem.catalysis_analysis import CatalyticAnalysisConditions
from chemworld.physchem.catalysis_descriptors import (
    DescriptorAxis,
    FreeEnergyBEP,
    LinearEnergyScaling,
)
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec, CatalyticReactionSpec
from chemworld.physchem.catalysis_thermochemistry import (
    CatalyticSpeciesThermo,
    HarmonicAdsorbateThermo,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reaction_network_specs import SpeciesSpec

R, T0, QUANTUM = 8.31446261815324, 350.0, 1000.0
SOURCE = DataSourceProvenance(
    "synthetic-catalysis54",
    0,
    "Authored mathematical He-labelled states, not helium catalyst data",
    version="1",
    license="CC0-1.0",
)
AXES = (DescriptorAxis("binding", -6000, 6000), DescriptorAxis("strain", -2000, 2000))


def gas(key, shift):
    return CatalyticSpeciesThermo(
        SpeciesSpec(key, "He", phase="gas"),
        "synthetic54-shared",
        SOURCE,
        gas_declaration={
            "thermo": {
                "model": "NASA7",
                "temperature-ranges": [250.0, 600.0],
                "data": [[2.5, 0, 0, 0, 0, -2.5 * T0 + shift / R, -2.5 * log(T0)]],
            },
            "composition": {"He": 1},
            "reference_pressure_Pa": 100000.0,
        },
    )


def mechanism(*, multistable=False):
    from math import exp

    # Authored target standard G at T0 makes the analytic benchmark symmetric.
    target_g = 9000.0 if multistable else -2500.0
    oscillator_g = 0.5 * QUANTUM + R * T0 * log(1 - exp(-QUANTUM / (R * T0)))
    harmonic = HarmonicAdsorbateThermo(
        LinearEnergyScaling(AXES, {"binding": 1.0, "strain": 0.5}, target_g - oscillator_g, SOURCE),
        {"binding": 0.0, "strain": 0.0},
        (QUANTUM,),
        250,
        600,
        SOURCE,
    )
    adsorbate = CatalyticSpeciesThermo(
        SpeciesSpec("bound", "He", phase="surface"),
        "synthetic54-shared",
        SOURCE,
        adsorbate=harmonic,
    )
    first = CatalyticReactionSpec(
        "adsorption",
        {"A": 1, "vacancy:metal": 1},
        {"bound": 1},
        "metal",
        1000.0,
        FreeEnergyBEP(30000 if multistable else 12000, 0.5, -60000, 60000, SOURCE),
        SOURCE,
    )
    reactions = (
        (first,)
        if multistable
        else (
            first,
            CatalyticReactionSpec(
                "conversion",
                {"bound": 1},
                {"B": 1, "vacancy:metal": 1},
                "metal",
                1000.0,
                FreeEnergyBEP(12000, 0.5, -60000, 60000, SOURCE),
                SOURCE,
            ),
        )
    )
    return CatalyticMechanismSpec(
        "synthetic54-frumkin" if multistable else "synthetic54-cycle",
        (gas("A", 0), adsorbate) if multistable else (gas("A", 0), gas("B", -5000), adsorbate),
        ("metal",),
        {"bound": "metal"},
        reactions,
        ((-18000.0 if multistable else 0.0,),),
        300.0,
        450.0,
        SOURCE,
    )


def conditions(*, temperature=350.0, multistable=False, capacity=1e-6):
    return CatalyticAnalysisConditions(
        temperature,
        {"A": 100000.0} if multistable else {"A": 100000.0, "B": 1000.0},
        {"metal": capacity},
    )
