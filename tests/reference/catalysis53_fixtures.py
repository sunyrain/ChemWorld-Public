"""Authored synthetic non-CO catalytic inputs; all constants are explicit."""

from math import log

from chemworld.physchem.catalysis_descriptors import (
    DescriptorAxis,
    FreeEnergyBEP,
    LinearEnergyScaling,
)
from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    CatalyticReactionSpec,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.physchem.catalysis_thermochemistry import (
    CatalyticSpeciesThermo,
    HarmonicAdsorbateThermo,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reaction_network_specs import SpeciesSpec

SOURCE = DataSourceProvenance(
    "authored-catalysis53",
    0,
    "Synthetic algebraic catalyst fixtures, not measured materials",
    version="1",
    license="CC0-1.0",
)
AXES = (DescriptorAxis("binding", -10000, 10000), DescriptorAxis("strain", -5000, 5000))


def harmonic(*, offset=-5000, descriptors=None, modes=(1000.0, 2000.0, 3000.0)):
    return HarmonicAdsorbateThermo(
        LinearEnergyScaling(AXES, {"binding": 0.5, "strain": 0.25}, offset, SOURCE),
        {"binding": 0.0, "strain": 0.0} if descriptors is None else descriptors,
        modes,
        250,
        600,
        SOURCE,
    )


def mechanism(*, descriptors=None, dissociative=False, interacting=True):
    gas, bound = ("H2", "H") if dissociative else ("He", "He")
    cp_r = 3.5 if dissociative else 2.5
    declaration = {
        "thermo": {
            "model": "NASA7",
            "temperature-ranges": [250.0, 600.0],
            "data": [[cp_r, 0, 0, 0, 0, (1 - cp_r) * 350, -cp_r * log(350)]],
        },
        "composition": {"H": 2} if dissociative else {"He": 1},
        "reference_pressure_Pa": 100000.0,
    }
    species = (
        CatalyticSpeciesThermo(
            SpeciesSpec(gas, gas, phase="gas"),
            "synthetic53-common-energy",
            SOURCE,
            gas_declaration=declaration,
        ),
        CatalyticSpeciesThermo(
            SpeciesSpec("bound_A", bound, phase="surface"),
            "synthetic53-common-energy",
            SOURCE,
            adsorbate=harmonic(descriptors=descriptors),
        ),
        CatalyticSpeciesThermo(
            SpeciesSpec("bound_B", bound, phase="surface"),
            "synthetic53-common-energy",
            SOURCE,
            adsorbate=harmonic(offset=-4000, descriptors=descriptors),
        ),
    )
    n = 2 if dissociative else 1
    bep = FreeEnergyBEP(20000, 0.5, -35000, 35000, SOURCE)
    reactions = (
        CatalyticReactionSpec(
            "adsorb", {gas: 1, "vacancy:A": n}, {"bound_A": n}, "A", 1000, bep, SOURCE
        ),
        CatalyticReactionSpec(
            "exchange",
            {"bound_A": 1, "vacancy:B": 1},
            {"bound_B": 1, "vacancy:A": 1},
            "A",
            500,
            bep,
            SOURCE,
        ),
    )
    return CatalyticMechanismSpec(
        "synthetic53-dissociation" if dissociative else "synthetic53-adsorption",
        species,
        ("A", "B"),
        {"bound_A": "A", "bound_B": "B"},
        reactions,
        ((2000, 800), (800, 1000)) if interacting else ((0, 0), (0, 0)),
        300,
        450,
        SOURCE,
    )


def state_and_slot(spec=None, *, temperature=350, amounts=(2e-5, 1e-5), mobile=False):
    spec = mechanism() if spec is None else spec
    slot = make_catalytic_slot(
        "carrier",
        "reactor",
        spec,
        capacities_mol={"A": 1e-4, "B": 2e-4},
        initial_amounts_mol={"bound_A": amounts[0], "bound_B": amounts[1]},
        fixed=not mobile,
        phase_id="gas" if mobile else None,
    )
    state = pack_catalytic_state(
        spec, (slot,), {spec.gas_species[0]: 0.002}, temperature_K=temperature, volume_L=1
    )
    return spec, slot, state
