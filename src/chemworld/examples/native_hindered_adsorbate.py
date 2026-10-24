"""Finite gas / hindered molecular adsorbate example using the original world.

The three small profiles and gas/BEP data are authored demonstration inputs, not
measured CO materials. The fourth thermochemistry-only profile reproduces the
public ASE ethane example's declared input; it is not material qualification.
"""

from __future__ import annotations

from math import log
from typing import Any

from chemworld.physchem.catalysis_descriptors import (
    DescriptorAxis,
    FreeEnergyBEP,
    LinearEnergyScaling,
)
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec, CatalyticReactionSpec
from chemworld.physchem.catalysis_thermochemistry import CatalyticSpeciesThermo
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.hindered_adsorbate import HinderedAdsorbateThermo
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network_specs import SpeciesSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world

SOURCE = DataSourceProvenance(
    "authored-hindered53B",
    0,
    "Declared mathematical examples; not measured catalytic materials",
    version="1",
    license="CC0-1.0",
)
ETHANE_SOURCE = DataSourceProvenance(
    "ase-public-ethane-example",
    0,
    "ASE thermochemistry documentation ethane/Pt(111) input example; Sprowl et al. 2016",
    version="ASE-3.26.0-doc-example",
    license="LGPL-2.1-or-later; factual parameter attribution",
)
PROFILE_NAMES = ("soft", "intermediate", "stiff", "ethane-example")


def profile_parameters(name: str) -> dict[str, Any]:
    """Fresh SI input declaration, shared as data with independent references."""
    if name not in PROFILE_NAMES:
        raise ValueError("unknown declared hindered profile")
    settings = {
        "soft": (100.0, 50.0, 1.5e-5, 1, 1),
        "intermediate": (5000.0, 2000.0, 2.5e-5, 3, 1),
        "stiff": (100000.0, 120000.0, 2.5e-5, 6, 2),
        "ethane-example": (0.0, 0.0, 1.0, 1, 1),
    }
    wt, wr, density, minima, symmetry = settings[name]
    values = {
        "number_of_atoms": 2,
        "translation_barrier_J_mol": wt,
        "rotation_barrier_J_mol": wr,
        "site_density_mol_m2": density,
        "molar_mass_kg_mol": 0.02801,
        "moment_of_inertia_kg_m2": 1.0e-46,
        "rotational_minima": minima,
        "symmetry_number": symmetry,
        "vibrational_energies_J_mol": (1000.0, 2500.0, 6000.0),
        "electronic_energy_J_mol": -5000.0,
    }
    if name == "ethane-example":
        # Explicitly selected 21 harmonic coordinates from the documented 24
        # frequencies; the three replaced coordinates are 77.262869, 60.278004,
        # 25.825447 cm^-1. Native evaluation does not silently sort any modes.
        frequencies = (
            3049.060670,
            3040.796863,
            3001.661338,
            2997.961647,
            2866.153162,
            2750.855460,
            1436.792655,
            1431.413595,
            1415.952186,
            1395.726300,
            1358.412432,
            1335.922737,
            1167.009954,
            1142.126116,
            1013.918680,
            803.400098,
            783.026031,
            310.448278,
            136.112935,
            112.939853,
            103.926392,
        )
        ev_per_mol = 1.602176634e-19 * 6.02214076e23
        values.update(
            number_of_atoms=8,
            translation_barrier_J_mol=0.049313 * ev_per_mol,
            rotation_barrier_J_mol=0.017675 * ev_per_mol,
            site_density_mol_m2=1.5e19 / 6.02214076e23,
            molar_mass_kg_mol=30.07 * 1.66053906660e-27 * 6.02214076e23,
            moment_of_inertia_kg_m2=73.149 * 1.66053906660e-47,
            rotational_minima=6,
            symmetry_number=1,
            vibrational_energies_J_mol=tuple(v / 8065.54429 * ev_per_mol for v in frequencies),
            electronic_energy_J_mol=0.0,
        )
    return values


def example_adsorbate(
    name: str = "intermediate", *, binding_J_mol: float = 0
) -> HinderedAdsorbateThermo:
    values = profile_parameters(name)
    offset = values.pop("electronic_energy_J_mol")
    source = ETHANE_SOURCE if name == "ethane-example" else SOURCE
    return HinderedAdsorbateThermo(
        electronic_energy=LinearEnergyScaling(
            (DescriptorAxis("binding", -5000, 5000),),
            {"binding": 0.5},
            offset,
            source,
        ),
        descriptors={"binding": binding_J_mol},
        harmonic_mode_ids=tuple(f"vib-{k}" for k in range(3 * values["number_of_atoms"] - 3)),
        hindered_mode_ids=("translation-x", "translation-y", "rotation-z"),
        minimum_temperature_K=200,
        maximum_temperature_K=1200,
        source=source,
        **values,
    )


def example_mechanism(*, binding_J_mol: float = 0) -> CatalyticMechanismSpec:
    model = example_adsorbate(binding_J_mol=binding_J_mol)
    declaration = {
        "thermo": {
            "model": "NASA7",
            "temperature-ranges": [200.0, 1200.0],
            "data": [[3.5, 0, 0, 0, 0, -3.5 * 298.15, 40 / R - 3.5 * log(298.15)]],
        },
        "composition": {"C": 1, "O": 1},
        "reference_pressure_Pa": 100000.0,
    }
    species = (
        CatalyticSpeciesThermo(
            SpeciesSpec("CO", "CO", phase="gas"),
            "authored53B-common",
            SOURCE,
            gas_declaration=declaration,
        ),
        CatalyticSpeciesThermo(
            SpeciesSpec("CO_site", "CO", phase="surface"),
            "authored53B-common",
            SOURCE,
            adsorbate=model,
        ),
    )
    reaction = CatalyticReactionSpec(
        "adsorb",
        {"CO": 1, "vacancy:A": 1},
        {"CO_site": 1},
        "A",
        10000,
        FreeEnergyBEP(30000, 0.5, -50000, 50000, SOURCE),
        SOURCE,
    )
    return CatalyticMechanismSpec(
        "hindered53B-example",
        species,
        ("A",),
        {"CO_site": "A"},
        (reaction,),
        ((0.0,),),
        220,
        1100,
        SOURCE,
        site_densities_mol_m2={"A": model.site_density_mol_m2},
    )


def example_world(
    *, temperature_K: float = 298.15, isothermal: bool = True
) -> WorldCompositionSpec:
    return descriptor_catalysis_world(
        example_mechanism(),
        {"CO": 0.002},
        capacities_mol={"A": 1e-4},
        initial_adsorbates_mol={"CO_site": 2e-5},
        volume_L=1,
        temperature_K=temperature_K,
        support_heat_capacity_J_K=0.05,
        stop_temperature_K=1050,
        maximum_pressure_Pa=1e6,
        isothermal=isothermal,
    )


def run_example() -> dict[str, Any]:
    world = ChemWorldRuntime(composition=example_world(isothermal=False))
    try:
        world.reset(seed=53)
        reply = world.execute({"operation": "wait", "duration_s": 8})
        if reply.info.get("transaction_status") != "committed":
            raise RuntimeError(str(reply.info))
        state = world._state
        return {
            "temperature_K": state.temperature_K,
            "gas_mol": dict(state.species_amounts),
            "surface_mol": state.state_slots.slots[0].values["CO_site"],
            "heat_J": state.ledger.energy_jacket_J,
            "elapsed_s": state.ledger.time_s,
        }
    finally:
        world.close()
