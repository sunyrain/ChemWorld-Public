"""Fixed, explicitly synthetic inputs; no live model parameter fitting."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

F, R = 96485.33212, 8.31446261815324
RTOL, ATOL = 1e-9, 1e-11
GRIDS = {"native": (32, 64), "reference": (80, 128)}
PROFILE_RADII = np.linspace(0.05, 0.95, 10)
REFUSALS = (
    "non-normalized-X",
    "nonpositive-w",
    "boundary-alpha",
    "nonuniform-mesh",
    "EIS",
    "SOH",
    "energy",
    "thermal",
)
BASE = {
    "temperature_K": 298.15,
    "electrolyte_concentration_mol_m3": 1000.0,
    "collector_area_m2": 0.01,
    "electrode_thickness_m": 1e-4,
    "surface_margin": 0.01,
    "minimum_voltage_V": 2.5,
    "maximum_voltage_V": 4.5,
    "maximum_current_A": 2.0,
    "resistance_ohm": 0.02,
    "samples_per_segment": 41,
    "waveform": ((240.0, 0.4, "A"),),
    "expected": "complete",
    "electrodes": [
        {
            "name": "negative",
            "radius_m": 5e-6,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 6e-7,
            "initial_fraction": 0.7,
            "profile_amplitude_V": 0.0,
            "rows": [[0.65, 0.12, 0.8, 6.0, 0.45], [0.35, 0.21, 1.4, 4.0, 0.55]],
        },
        {
            "name": "positive",
            "radius_m": 5e-6,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 6e-7,
            "initial_fraction": 0.3,
            "profile_amplitude_V": 0.0,
            "rows": [[0.4, 3.85, 1.1, 5.0, 0.55], [0.6, 4.05, 0.7, 7.0, 0.45]],
        },
    ],
}
NAMES = (
    "static-occupancy-and-kinetics",
    "one-reaction-D-limit",
    "one-reaction-2D-limit",
    "two-reaction-discharge",
    "reverse-charge",
    "CC-rest-reverse",
    "nonuniform-potential-rest",
    "diffusion-and-kinetic-contrast",
    "interior-CV",
    "both-CV-current-limits",
    "voltage-cutoff",
    "domain-refusals",
)


def case_data(index):
    if index not in range(1, 13):
        raise ValueError("fixed MSMR block contains MS01-MS12")
    data = deepcopy(BASE)
    data["name"] = NAMES[index - 1]
    n, p = data["electrodes"]
    if index == 1:
        data["expected"] = "static"
    elif index in (2, 3):
        data["fick_multiplier"] = float(index - 1)
        for electrode, u0 in zip((n, p), (0.15, 4.0), strict=True):
            electrode["rows"] = [[1.0, u0, float(index - 1), 6.0, 0.5]]
        data["waveform"] = ((240.0, 0.02, "A"),)
    elif index == 5:
        n["initial_fraction"], p["initial_fraction"] = 0.45, 0.55
        data["waveform"] = ((240.0, -0.4, "A"),)
    elif index == 6:
        data["waveform"] = (
            (100.0, 0.4, "A"),
            (80.0, 0.0, "A"),
            (100.0, -0.3, "A"),
            (60.0, 0.0, "A"),
        )
    elif index == 7:
        n["profile_amplitude_V"], p["profile_amplitude_V"] = 0.006, -0.004
        data["waveform"] = ((200.0, 0.0, "A"),)
    elif index == 8:
        n["diffusivity_m2_s"], p["diffusivity_m2_s"] = 3e-14, 3e-13
        n["rows"][0][3], n["rows"][1][3] = 2.0, 9.0
        p["rows"][0][3], p["rows"][1][3] = 8.0, 2.0
    elif index == 9:
        data["waveform"] = ((180.0, 3.86, "V"),)
        data["control_expectation"] = "interior"
    elif index == 10:
        data["waveform"] = ((120.0, 3.5, "V"), (120.0, 4.3, "V"))
        data["maximum_current_A"] = 0.3
        data["control_expectation"] = "both-limits"
    elif index == 11:
        data["waveform"] = ((1800.0, 0.3, "A"),)
        data["minimum_voltage_V"] = 3.86
        data["expected"] = "battery:minimum-voltage"
    elif index == 12:
        data["expected"] = "refusals"
    return data


def occupancies(rows, potential, temperature):
    """Independent scalar logistic formula, also used for source initial inputs."""
    return np.array(
        [
            x * expit(-F / (R * temperature) * (np.asarray(potential) - u0) / w)
            for x, u0, w, _jref, _alpha in rows
        ]
    )


def potential(rows, fraction, temperature):
    return brentq(
        lambda u: occupancies(rows, u, temperature).sum() - fraction,
        min(row[1] for row in rows) - 5.0,
        max(row[1] for row in rows) + 5.0,
        xtol=1e-14,
        rtol=1e-14,
    )


def initial_profile(electrode, cells, temperature):
    radial = (np.arange(cells) + 0.5) / cells
    u0 = potential(electrode["rows"], electrode["initial_fraction"], temperature)
    u = u0 + electrode["profile_amplitude_V"] * np.cos(np.pi * radial)
    x = occupancies(electrode["rows"], u, temperature).sum(axis=0)
    return radial, u, x, np.diff(np.linspace(0.0, 1.0, cells + 1) ** 3)


def source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(
        DataSourceProvenance(
            "msmr-spm-synthetic",
            0,
            "Explicit synthetic MSMR X/U0/w/jref/alpha and particle inputs; equations "
            "compared with PyBaMM25.4.2, not published graphite/NMC material parameters",
            "fixed-msmr-block-1",
            "authored qualification inputs",
        ),
        "synthetic",
    )


def native_material(electrode):
    from chemworld.data.scientific_inputs import (
        InputValue,
        QuantityColumn,
        ScientificRow,
        ScientificTable,
    )
    from chemworld.physchem.battery_msmr_materials import MSMRMaterialSpec

    units = {
        "X": "dimensionless",
        "U0_V": "V",
        "w": "dimensionless",
        "jref_A_m2": "A/m^2",
        "alpha": "dimensionless",
    }
    ids = tuple(f"reaction-{j}" for j in range(len(electrode["rows"])))
    table = ScientificTable(
        "msmr-" + electrode["name"],
        "parameters",
        tuple(
            QuantityColumn(k, u, "parameter", (-10.0, 1e6), "Fixed synthetic input")
            for k, u in units.items()
        ),
        (source(),),
        tuple(
            ScientificRow(
                identity,
                source().provenance.source_id,
                "synthetic",
                electrode["name"]+":"+identity,
                "parameters",
                {
                    k: InputValue(v, u, None, "Fixed synthetic qualification input")
                    for (k, u), v in zip(units.items(), row, strict=True)
                },
            )
            for identity, row in zip(ids, electrode["rows"], strict=True)
        ),
    )
    return MSMRMaterialSpec(table, ids, electrode["name"])


def native_spec(data, cells):
    from chemworld.physchem.battery_models import ParticleSpec
    from chemworld.physchem.battery_msmr import BatteryMSMRSPMSpec

    kwargs = {
        k: data[k]
        for k in (
            "temperature_K",
            "electrolyte_concentration_mol_m3",
            "surface_margin",
            "minimum_voltage_V",
            "maximum_voltage_V",
            "maximum_current_A",
            "resistance_ohm",
        )
    }
    for e in data["electrodes"]:
        kwargs[e["name"]] = ParticleSpec(
            **{
                k: e[k]
                for k in (
                    "radius_m",
                    "diffusivity_m2_s",
                    "maximum_concentration_mol_m3",
                    "active_volume_m3",
                )
            },
            exchange_current_scale_A_m2=1.0,
            cells=cells,
            mesh_exponent=1.0,
        )
        kwargs[e["name"] + "_material"] = native_material(e)
        _, _, x, weights = initial_profile(e, cells, data["temperature_K"])
        kwargs["initial_" + e["name"] + "_fraction"] = float(weights @ x)
    return BatteryMSMRSPMSpec(**kwargs, provenance="Explicit synthetic MSMR development block")


def native_initial_state(spec, data, cells):
    state, amounts = spec.initial_state(), {}
    for e in data["electrodes"]:
        _, _, x, weights = initial_profile(e, cells, data["temperature_K"])
        amounts.update(
            zip(
                getattr(spec, e["name"]).keys(e["name"]),
                x * weights * e["maximum_concentration_mol_m3"] * e["active_volume_m3"],
                strict=True,
            )
        )
    return replace(state, amounts_mol=amounts)
