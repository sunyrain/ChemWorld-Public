"""Explicit synthetic energy tables generate two actual native catalytic worlds.

These CC0 teaching energies are not measured catalysts. Both table responses and
their declared sigmas are authored; no model computes its own fitting targets.
"""

from __future__ import annotations

from math import log
from typing import Any

from chemworld.data.calibration_uncertainty import KNOWN_INDEPENDENT_SIGMA
from chemworld.data.catalysis_calibration import (
    CatalyticEnergyCalibration,
    CatalyticEnergyFitParameter,
)
from chemworld.data.catalysis_inputs import assemble_catalytic_mechanism
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.catalysis_descriptors import (
    DescriptorAxis,
    FreeEnergyBEP,
    LinearEnergyScaling,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reaction_network_specs import SpeciesSpec
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world

FRAME = "authored53-close-common-reference"
SOURCE = DataSourceProvenance(
    "authored53-close",
    0,
    "CC0 synthetic energy tables; not measured catalysts",
    version="1",
    license="CC0-1.0",
)
AXES = (DescriptorAxis("binding", -10000, 10000), DescriptorAxis("strain", -5000, 5000))
POINTS = (
    (-6000, -3000),
    (-6000, 3000),
    (6000, -3000),
    (6000, 3000),
    (0, -3000),
    (0, 3000),
    (-6000, 0),
    (6000, 0),
    (-2000, 1000),
    (3000, -1500),
)
PERTURBATIONS = (-10, 20, -30, 10, 0, 15, -5, 12, -8, 9)


def energy_table(
    name: str = "bound_A",
    *,
    bep: bool = False,
    mixed_units: bool = False,
    confounded: bool = False,
) -> ScientificTable:
    """Fixed source observations; the final two independent groups are held out."""
    conditions = ("delta_g",) if bep else ("binding", "strain")
    unit = "kJ/mol" if mixed_units else "J/mol"
    limit = 100 if mixed_units else 100000
    columns = (
        *(QuantityColumn(k, unit, "condition", (-limit, limit), FRAME) for k in conditions),
        QuantityColumn("energy", unit, "observation", (-limit, limit), FRAME),
    )
    rows = []
    for i, point in enumerate(POINTS):
        binding, strain = map(float, point)
        if confounded:
            strain = binding / 2
        values = {"delta_g": binding * 2} if bep else {"binding": binding, "strain": strain}
        energy = (
            20000 + 0.5 * values["delta_g"]
            if bep
            else (-5000 if name == "bound_A" else -4000) + 0.5 * binding + 0.25 * strain
        )
        cells = {
            key: InputValue(float(v), "J/mol", 0.0, "fixed authored input")
            for key, v in values.items()
        }
        cells["energy"] = InputValue(
            float(energy + PERTURBATIONS[i]),
            "J/mol",
            float(40 + 10 * (i % 3)),
            "Authored known independent Gaussian sigma; fixed perturbation realization",
        )
        rows.append(
            ScientificRow(
                str(i),
                SOURCE.source_id,
                f"group-{i}",
                f"sample-{i}",
                "train" if i < 8 else "validation",
                cells,
            )
        )
    return ScientificTable(
        "energy-" + ("bep" if bep else name),
        "measurements",
        columns,
        (InputSource(SOURCE, "synthetic"),),
        tuple(rows),
    )


def energy_calibration(
    name: str = "bound_A",
    *,
    bep: bool = False,
    mixed_units: bool = False,
    confounded: bool = False,
    table: ScientificTable | None = None,
    max_forward_groups: int = 6000,
) -> CatalyticEnergyCalibration:
    table = table or energy_table(name, bep=bep, mixed_units=mixed_units, confounded=confounded)
    model: LinearEnergyScaling | FreeEnergyBEP
    parameters: tuple[CatalyticEnergyFitParameter, ...]
    if bep:
        model = FreeEnergyBEP(21000, 0.4, -35000, 35000, SOURCE)
        parameters = (
            CatalyticEnergyFitParameter("intercept_J_mol", (18000, 30000), 21000, "J/mol"),
            CatalyticEnergyFitParameter("alpha", (0.3, 0.7), 0.4, "dimensionless"),
        )
    else:
        model = LinearEnergyScaling(AXES, {"binding": 0.4, "strain": 0.1}, -6000, SOURCE)
        parameters = (
            CatalyticEnergyFitParameter("intercept_J_mol", (-10000, 0), -6000, "J/mol"),
            CatalyticEnergyFitParameter("coefficient:binding", (-1, 2), 0.4, "dimensionless"),
            CatalyticEnergyFitParameter("coefficient:strain", (-1, 2), 0.1, "dimensionless"),
        )
    return CatalyticEnergyCalibration(
        table,
        model,
        parameters,
        fitted_source=DataSourceProvenance(
            "fit:" + table.dataset_id,
            0,
            "Bounded fit of " + table.dataset_id,
            version="1",
            license="CC0-1.0",
        ),
        reference_frame=FRAME,
        energy_kind="activation_free_energy" if bep else "electronic_energy",
        noise_model=KNOWN_INDEPENDENT_SIGMA,
        max_forward_groups=max_forward_groups,
    )


def mechanism_declaration(topology: str = "exchange") -> dict[str, Any]:
    """An explicit native topology with energy holes, not placeholder fitted values."""
    if topology not in {"exchange", "dissociation"}:
        raise ValueError("choose exchange or dissociation teaching topology")
    gas = "He" if topology == "exchange" else "N2"
    formulas = (gas, gas) if topology == "exchange" else ("N2", "N")
    cp_r = 2.5 if gas == "He" else 3.5
    species = [
        {
            "species": SpeciesSpec(gas, gas, phase="gas").to_dict(),
            "reference_frame": FRAME,
            "source": SOURCE.to_dict(),
            "adsorbate": None,
            "gas_declaration": {
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [250.0, 600.0],
                    "data": [[cp_r, 0, 0, 0, 0, (1 - cp_r) * 350, -cp_r * log(350)]],
                },
                "composition": {"He": 1} if gas == "He" else {"N": 2},
                "reference_pressure_Pa": 100000.0,
            },
        }
    ]
    for index, formula in enumerate(formulas):
        adsorbate: dict[str, Any] = {
            "vibrational_energies_J_mol": [1000.0, 2000.0, 3000.0],
            "minimum_temperature_K": 250.0,
            "maximum_temperature_K": 600.0,
            "source": SOURCE.to_dict(),
        }
        if topology == "dissociation" and index == 0:
            adsorbate.update(
                model="hindered-adsorbate-fixed-site-v1",
                harmonic_mode_ids=["v1", "v2", "v3"],
                hindered_mode_ids=["tx", "ty", "rz"],
                number_of_atoms=2,
                translation_barrier_J_mol=12000.0,
                rotation_barrier_J_mol=8000.0,
                site_density_mol_m2=2e-5,
                molar_mass_kg_mol=0.028,
                moment_of_inertia_kg_m2=1.4e-46,
                rotational_minima=1,
                symmetry_number=2,
            )
        species.append(
            {
                "species": SpeciesSpec("bound_" + "AB"[index], formula, phase="surface").to_dict(),
                "reference_frame": FRAME,
                "source": SOURCE.to_dict(),
                "gas_declaration": None,
                "adsorbate": adsorbate,
            }
        )
    reactions = [
        {
            "reaction_id": "adsorb",
            "reactants": {gas: 1, "vacancy:A": 1},
            "products": {"bound_A": 1},
            "rate_site": "A",
            "attempt_frequency_s_inv": 1000.0,
            "source": SOURCE.to_dict(),
        },
        {
            "reaction_id": "exchange" if topology == "exchange" else "dissociate",
            "reactants": {"bound_A": 1, "vacancy:B": 1 if topology == "exchange" else 2},
            "products": {"bound_B": 1 if topology == "exchange" else 2, "vacancy:A": 1},
            "rate_site": "A",
            "attempt_frequency_s_inv": 500.0,
            "source": SOURCE.to_dict(),
        },
    ]
    return {
        "model": "descriptor-catalysis-v1",
        "mechanism_id": "table-" + topology,
        "species": species,
        "site_types": ["A", "B"],
        "adsorbate_sites": {"bound_A": "A", "bound_B": "B"},
        "reactions": reactions,
        "interaction_matrix_J_mol": [[2000.0, 800.0], [800.0, 1000.0]],
        "minimum_temperature_K": 300.0,
        "maximum_temperature_K": 450.0,
        "source": SOURCE.to_dict(),
        **({"site_densities_mol_m2": {"A": 2e-5}} if topology == "dissociation" else {}),
    }


def fitted_catalytic_mechanism(topology: str = "exchange") -> tuple[Any, dict[str, Any]]:
    fits = {name: energy_calibration(name).fit(max_nfev=60) for name in ("bound_A", "bound_B")}
    fits["bep"] = energy_calibration(bep=True).fit(max_nfev=60)
    declaration = mechanism_declaration(topology)
    return assemble_catalytic_mechanism(
        declaration,
        fits,
        species_fits={name: name for name in ("bound_A", "bound_B")},
        reaction_fits={r["reaction_id"]: "bep" for r in declaration["reactions"]},
        descriptors={"binding": 1000.0, "strain": -500.0},
    )


def catalytic_energy_table_world(topology: str = "exchange", *, isothermal: bool = True) -> Any:
    mechanism, _ = fitted_catalytic_mechanism(topology)
    return descriptor_catalysis_world(
        mechanism,
        {mechanism.gas_species[0]: 0.002},
        capacities_mol={"A": 1e-4, "B": 2e-4},
        initial_adsorbates_mol={"bound_A": 2e-5, "bound_B": 1e-5},
        volume_L=1.0,
        temperature_K=350.0,
        support_heat_capacity_J_K=5.0,
        stop_temperature_K=440.0,
        maximum_pressure_Pa=1e6,
        isothermal=isothermal,
    )
