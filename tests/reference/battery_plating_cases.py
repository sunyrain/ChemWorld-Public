"""Authored inputs for the prospective 13-unit/64-route plating block."""

from copy import deepcopy

import numpy as np

F, R = 96485.33212, 8.31446261815324
GRIDS = {"native": (40, 80), "reference": (120, 200)}
RTOL, ATOL, ROOT_TOL = 1e-9, 1e-11, 1e-9
PROFILE_RADII = np.linspace(0.05, 0.95, 9)
BASE = {
    "collector_area_m2": 0.02,
    "negative_electrode_volume_m3": 2e-6,
    "positive_thickness_m": 1e-4,
    "separator_thickness_m": 2.5e-5,
    "temperature_K": 298.15,
    "ce_mol_m3": 1000.0,
    "k_m_s": 1e-8,
    "alpha": 0.5,
    "mode": "reversible",
    "initial_plated_mol": 0.0,
    "initial_dead_mol": 0.0,
    "surface_margin": 0.01,
    "minimum_voltage_V": 2.5,
    "maximum_voltage_V": 4.5,
    "maximum_current_A": 2.0,
    "series_resistance_ohm": 0.02,
    "samples_per_segment": 41,
    "waveform": ((300.0, -0.6, "A"),),
    "expected": "complete",
    "electrodes": [
        {
            "name": "negative",
            "radius_m": 1e-5,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 1e-6,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.7,
            "polynomial": (0.105, -0.1),
        },
        {
            "name": "positive",
            "radius_m": 1e-5,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 1e-6,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.3,
            "polynomial": (4.2, -0.6),
        },
    ],
}
NAMES = (
    "zero-rate-old-SPM",
    "zero-metal-charge",
    "seeded-stripping",
    "internal-rest",
    "CC-rest-reverse",
    "irreversible-charge",
    "irreversible-rest-reverse",
    "kinetic-geometry-temperature",
    "interior-CV",
    "limited-CV",
    "surface-stop",
    "voltage-stop",
    "refusals",
)
REFUSALS = (
    "negative-k",
    "alpha-domain",
    "nonpositive-ce",
    "geometric-volume",
    "negative-plated",
    "negative-dead",
    "partial-mode",
    "SEI",
    "thermal",
    "EIS",
    "SOH",
    "chemical-energy",
)
RESPONSES = (
    "voltage_V",
    "current_A",
    "charge_C",
    "work_J",
    "stripping_current_A",
    "plated_mol",
    "dead_mol",
    "negative_potential_V",
    "plated_concentration",
    *(
        f"{e}:{v}"
        for e in ("negative", "positive")
        for v in (
            "reaction_current_A",
            "surface_fraction",
            "average_fraction",
            "amount_mol",
            "ocp_V",
            "overpotential_V",
        )
    ),
)
RESPONSE_ATOL = np.array(
    [
        0.002,
        0.005,
        0.05,
        0.2,
        0.005,
        2e-7,
        2e-7,
        0.002,
        0.2,
        *([0.005, 0.003, 0.003, 2e-7, 0.002, 0.002] * 2),
    ]
)
RESPONSE_RTOL = np.array(
    [
        1e-5,
        0.005,
        1e-5,
        1e-5,
        0.005,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        *([0.005, 0.0, 0.0, 1e-5, 1e-5, 1e-5] * 2),
    ]
)


def case_data(index):
    if index not in range(1, 14):
        raise ValueError("plating block fixes exactly PL01-PL13")
    data = deepcopy(BASE)
    data.update(name=NAMES[index - 1], index=index)
    if index == 1:
        data.update(k_m_s=0.0, waveform=((150.0, 0.4, "A"), (50.0, 0.0, "A"), (150.0, -0.4, "A")))
    elif index in (3, 4):
        data.update(initial_plated_mol=0.0005, waveform=((300.0, 0.4 if index == 3 else 0.0, "A"),))
    elif index == 5:
        data["waveform"] = (
            (150.0, -0.6, "A"),
            (150.0, 0.0, "A"),
            (150.0, 0.4, "A"),
            (100.0, 0.0, "A"),
        )
    elif index in (6, 7):
        data["mode"] = "irreversible"
        if index == 7:
            data["waveform"] = ((150.0, -0.6, "A"), (150.0, 0.0, "A"), (150.0, 0.4, "A"))
    elif index == 8:
        data.update(
            negative_electrode_volume_m3=4e-6,
            k_m_s=5e-9,
            alpha=0.65,
            temperature_K=318.15,
            initial_plated_mol=0.0005,
            waveform=((300.0, -0.4, "A"),),
        )
        data["electrodes"][0]["diffusivity_m2_s"] = 5e-14
    elif index == 9:
        data["waveform"] = ((150.0, 4.01, "V"), (150.0, 3.98, "V"))
    elif index == 10:
        data.update(maximum_current_A=0.3, waveform=((100.0, 4.4, "V"), (100.0, 3.5, "V")))
    elif index == 11:
        data["electrodes"][0]["initial_fraction"] = 0.03
        for e in data["electrodes"]:
            e["diffusivity_m2_s"] = 1e-12
        data.update(waveform=((300.0, 0.8, "A"),), expected="battery:negative:depleted")
    elif index == 12:
        data.update(
            minimum_voltage_V=3.85,
            waveform=((600.0, 0.8, "A"),),
            expected="battery:minimum-voltage",
        )
    elif index == 13:
        data["expected"] = "refusals"
    return data


def source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(
        DataSourceProvenance(
            "plating-spm-synthetic",
            0,
            "Explicit synthetic single OCP/geometry and OKane-form plating kinetics; actual "
            "PyBaMM25.4.2 equations, not an OKane2022 commercial-cell parameter reproduction",
            "fixed-plating-block-1",
            "authored development qualification",
        ),
        "synthetic",
    )


def native_spec(data, cells, *, old_spm=False):
    from chemworld.data.scientific_inputs import (
        InputValue,
        QuantityColumn,
        ScientificRow,
        ScientificTable,
    )
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
    from chemworld.physchem.battery_plating import LithiumPlatingSpec

    declaration = source()
    particles, curves = [], []
    for e in data["electrodes"]:
        particles.append(
            ParticleSpec(
                **{
                    k: e[k]
                    for k in (
                        "radius_m",
                        "diffusivity_m2_s",
                        "maximum_concentration_mol_m3",
                        "active_volume_m3",
                        "exchange_current_scale_A_m2",
                    )
                },
                cells=cells,
                mesh_exponent=1.0,
            )
        )
        curves.append(
            ElectrodeOCV(
                e["name"],
                declaration,
                e["polynomial"],
                supported_models=("SPM",),
                stoichiometry_domain=(data["surface_margin"], 1 - data["surface_margin"]),
            )
        )
    units = {
        "k_m_s": "m/s",
        "alpha": "dimensionless",
        "ce_mol_m3": "mol/m^3",
        "negative_electrode_volume_m3": "m^3",
    }
    table = ScientificTable(
        "plating-synthetic",
        "parameters",
        tuple(
            QuantityColumn(k, u, "parameter", (-1.0, 1e12), "Authored qualification value")
            for k, u in units.items()
        ),
        (declaration,),
        (
            ScientificRow(
                "plating",
                declaration.provenance.source_id,
                "plating",
                "plating",
                "parameters",
                {
                    k: InputValue(data[k], unit, None, "Authored value; no measured uncertainty")
                    for k, unit in units.items()
                },
            ),
        ),
    )
    plating = (
        None
        if old_spm
        else LithiumPlatingSpec(
            table, "plating", data["mode"], data["initial_plated_mol"], data["initial_dead_mol"]
        )
    )
    return BatterySPMSpec(
        negative=particles[0],
        positive=particles[1],
        negative_ocv=curves[0],
        positive_ocv=curves[1],
        plating=plating,
        initial_negative_fraction=data["electrodes"][0]["initial_fraction"],
        initial_positive_fraction=data["electrodes"][1]["initial_fraction"],
        **{
            k: data[k]
            for k in (
                "temperature_K",
                "surface_margin",
                "minimum_voltage_V",
                "maximum_voltage_V",
                "maximum_current_A",
                "series_resistance_ohm",
            )
        },
        provenance="Declared synthetic plating SPM qualification",
    )
