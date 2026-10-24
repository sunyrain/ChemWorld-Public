"""Prospectively fixed synthetic x-full inputs, with units explicit in every card."""

from copy import deepcopy
from itertools import pairwise

import numpy as np

F = 96485.33212
RTOL, ATOL = 1e-9, 1e-11
GRIDS = {"native": ((6, 40), (12, 80)), "reference": ((12, 120), (24, 200))}
PROFILE_POINTS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
BASE = {
    "width_m": 0.1,
    "height_m": 0.1,
    "ambient_temperature_K": 298.15,
    "initial_temperature_K": 298.15,
    "minimum_temperature_K": 270.0,
    "maximum_temperature_K": 370.0,
    "edge_heat_transfer_W_m2_K": 2.0,
    "layers": [
        {
            "name": "negative",
            "length_m": 1e-4,
            "heat_capacity_J_m3_K": 2e6,
            "conductivity_W_m_K": 0.002,
        },
        {
            "name": "separator",
            "length_m": 2.5e-5,
            "heat_capacity_J_m3_K": 1.5e6,
            "conductivity_W_m_K": 0.0005,
        },
        {
            "name": "positive",
            "length_m": 1e-4,
            "heat_capacity_J_m3_K": 2.5e6,
            "conductivity_W_m_K": 0.003,
        },
    ],
    "collectors": [
        {
            "name": "negative",
            "length_m": 1e-5,
            "heat_capacity_J_m3_K": 3e6,
            "face_heat_transfer_W_m2_K": 12.0,
            "tab_heat_transfer_W_m2_K": 5.0,
            "tab_width_m": 0.01,
        },
        {
            "name": "positive",
            "length_m": 1.5e-5,
            "heat_capacity_J_m3_K": 2.8e6,
            "face_heat_transfer_W_m2_K": 12.0,
            "tab_heat_transfer_W_m2_K": 5.0,
            "tab_width_m": 0.01,
        },
    ],
    "electrodes": [
        {
            "name": "negative",
            "radius_m": 5e-6,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 6e-7,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.6,
            "ocv_V": 0.2,
        },
        {
            "name": "positive",
            "radius_m": 5e-6,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 6e-7,
            "exchange_current_scale_A_m2": 2.0,
            "initial_fraction": 0.35,
            "ocv_V": 4.0,
        },
    ],
    "surface_margin": 0.01,
    "minimum_voltage_V": 2.5,
    "maximum_voltage_V": 4.5,
    "maximum_current_A": 2.0,
    "electrolyte_concentration_mol_m3": 1000.0,
    "samples_per_segment": 41,
    "waveform": ((90.0, 0.8, "A"),),
    "expected": "complete",
}
NAMES = (
    "adiabatic-zero-current-gradient",
    "adiabatic-CC-energy",
    "cooled-CC",
    "conductivity-contrast",
    "asymmetric-cooling",
    "CC-rest-reverse",
    "kinetic-heat-contrast",
    "interior-CV",
    "finite-CV-both-limits",
    "local-hotspot-cutoff",
    "lumped-thermal-limit",
    "domain-refusals",
)
REFUSALS = (
    "nonconstant-ocv",
    "invalid-layer-cells",
    "nonpositive-heat-capacity",
    "negative-cooling",
    "initial-hotspot-outside-domain",
    "external-thermal-device",
    "EIS",
    "SOH",
)


def case_data(index):
    if index not in range(1, 13):
        raise ValueError("fixed spatial-thermal block contains XT01-XT12")
    data = deepcopy(BASE)
    data["name"] = NAMES[index - 1]
    if index in (1, 2, 10):
        data["edge_heat_transfer_W_m2_K"] = 0.0
        for collector in data["collectors"]:
            collector["face_heat_transfer_W_m2_K"] = 0.0
            collector["tab_heat_transfer_W_m2_K"] = 0.0
    if index == 1:
        data["initial_profile"] = "linear-304-to-294"
        data["waveform"] = ((60.0, 0.0, "A"),)
    elif index == 2:
        data["waveform"] = ((120.0, 0.6, "A"),)
    elif index == 4:
        for layer, value in zip(data["layers"], (0.0004, 0.0002, 0.008), strict=True):
            layer["conductivity_W_m_K"] = value
    elif index == 5:
        data["waveform"] = ((120.0, 0.8, "A"),)
        data["edge_heat_transfer_W_m2_K"] = 4.0
        for collector, face, tab in zip(
            data["collectors"], (60.0, 1.0), (100.0, 10.0), strict=True
        ):
            collector["face_heat_transfer_W_m2_K"] = face
            collector["tab_heat_transfer_W_m2_K"] = tab
    elif index == 6:
        data["waveform"] = tuple((40.0, value, "A") for value in (0.8, 0.0, -0.6, 0.0))
    elif index == 7:
        data["waveform"] = ((90.0, 0.7, "A"),)
        for electrode, value in zip(data["electrodes"], (1.0, 6.0), strict=True):
            electrode["exchange_current_scale_A_m2"] = value
    elif index == 8:
        data["waveform"], data["control_expectation"] = ((120.0, 3.74, "V"),), "interior"
    elif index == 9:
        data["waveform"] = ((40.0, 3.5, "V"), (40.0, 4.1, "V"))
        data["maximum_current_A"], data["control_expectation"] = 0.3, "both-limits"
    elif index == 10:
        data["maximum_temperature_K"] = 299.2
        data["waveform"], data["expected"] = ((120.0, 0.8, "A"),), "maximum-temperature"
    elif index == 11:
        data["initial_temperature_K"] = 310.0
        data["waveform"] = ((100.0, 0.0, "A"),)
        for layer in data["layers"]:
            layer["conductivity_W_m_K"] = 100.0
    elif index == 12:
        data["expected"] = "refusals"
    return data


def thermal_geometry(data, cells):
    """Raw geometry only; no native heat-stack matrix or material dependencies."""
    area = data["width_m"] * data["height_m"]
    perimeter = 2 * (data["width_m"] + data["height_m"])
    lengths, capacities, resistance, losses, locations = [], [], [], [], []
    offset = 0.0
    for layer in data["layers"]:
        dx = layer["length_m"] / cells
        lengths.extend([dx] * cells)
        capacities.extend([area * dx * layer["heat_capacity_J_m3_K"]] * cells)
        resistance.extend([dx / (2 * area * layer["conductivity_W_m_K"])] * cells)
        losses.extend([perimeter * dx * data["edge_heat_transfer_W_m2_K"]] * cells)
        locations.extend(offset + (np.arange(cells) + 0.5) * dx)
        offset += layer["length_m"]
    cc_capacity, cc_loss, cc_length = [], [], []
    for collector in data["collectors"]:
        length = collector["length_m"]
        tab_area = collector["tab_width_m"] * length
        cc_length.append(length)
        cc_capacity.append(area * length * collector["heat_capacity_J_m3_K"])
        cc_loss.append(
            area * collector["face_heat_transfer_W_m2_K"]
            + tab_area * collector["tab_heat_transfer_W_m2_K"]
            + (perimeter * length - tab_area) * data["edge_heat_transfer_W_m2_K"]
        )
    conductances = [
        1 / resistance[0],
        *(1 / (a + b) for a, b in pairwise(resistance)),
        1 / resistance[-1],
    ]
    return {
        "capacities_J_K": np.array([cc_capacity[0], *capacities, cc_capacity[1]]),
        "volumes_m3": area * np.array([cc_length[0], *lengths, cc_length[1]]),
        "loss_conductances_W_K": np.array([cc_loss[0], *losses, cc_loss[1]]),
        "face_conductances_W_K": np.array(conductances),
        "sandwich_locations_m": np.array(locations),
    }


def initial_temperatures(data, cells):
    if data.get("initial_profile") == "linear-304-to-294":
        geo = thermal_geometry(data, cells)
        length = sum(layer["length_m"] for layer in data["layers"])
        return np.r_[304.0, 304.0 - 10 * geo["sandwich_locations_m"] / length, 294.0]
    return np.full(3 * cells + 2, data["initial_temperature_K"])


def matrix_temperature(data, cells, times):
    """Independent finite heat network; valid only for the zero-current limits."""
    from scipy.linalg import expm

    geo = thermal_geometry(data, cells)
    c, g = geo["capacities_J_K"], geo["face_conductances_W_K"]
    laplacian = np.diag(geo["loss_conductances_W_K"])
    for i, value in enumerate(g):
        laplacian[i, i] += value
        laplacian[i + 1, i + 1] += value
        laplacian[i, i + 1] -= value
        laplacian[i + 1, i] -= value
    matrix = -laplacian / c[:, None]
    ambient = data["ambient_temperature_K"]
    initial = initial_temperatures(data, cells) - ambient
    return np.array([ambient + expm(matrix * t) @ initial for t in times])


def source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(
        DataSourceProvenance(
            "spatial-thermal-spm-synthetic",
            0,
            "Explicit synthetic constant-net-OCV particles and finite thermal layers/collectors; "
            "compared with PyBaMM25.4.2 actual x-full SPM, not Chen2020 material validation",
            "fixed-xthermal-block-1",
            "authored qualification inputs",
        ),
        "synthetic",
    )


def native_spec(data, grid):
    from chemworld.physchem.battery_heat_stack import (
        BatteryHeatStackSpec,
        HeatStackCollector,
        HeatStackLayer,
    )
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_models import ParticleSpec
    from chemworld.physchem.battery_spatial_thermal import BatterySpatialThermalSPMSpec

    cells, radial = grid
    layers = [
        HeatStackLayer(
            **{k: v for k, v in layer.items() if k != "name"}, cells=cells, source=source()
        )
        for layer in data["layers"]
    ]
    collectors = [
        HeatStackCollector(**{k: v for k, v in cc.items() if k != "name"}, source=source())
        for cc in data["collectors"]
    ]
    stack = BatteryHeatStackSpec(
        *layers,
        *collectors,
        **{
            k: data[k]
            for k in (
                "width_m",
                "height_m",
                "edge_heat_transfer_W_m2_K",
                "ambient_temperature_K",
                "minimum_temperature_K",
                "maximum_temperature_K",
            )
        },
        source=source(),
    )
    particles, curves = [], []
    for electrode in data["electrodes"]:
        particles.append(
            ParticleSpec(
                **{
                    k: electrode[k]
                    for k in (
                        "radius_m",
                        "diffusivity_m2_s",
                        "maximum_concentration_mol_m3",
                        "active_volume_m3",
                        "exchange_current_scale_A_m2",
                    )
                },
                cells=radial,
                mesh_exponent=1.0,
            )
        )
        curves.append(
            ElectrodeOCV(
                electrode["name"] + "-constant-synthetic",
                source(),
                (electrode["ocv_V"],),
                supported_models=("SPM",),
                stoichiometry_domain=(data["surface_margin"], 1 - data["surface_margin"]),
            )
        )
    return BatterySpatialThermalSPMSpec(
        *particles,
        stack,
        *curves,
        initial_negative_fraction=data["electrodes"][0]["initial_fraction"],
        initial_positive_fraction=data["electrodes"][1]["initial_fraction"],
        temperature_K=data["initial_temperature_K"],
        initial_temperatures_K=tuple(initial_temperatures(data, cells)),
        **{
            k: data[k]
            for k in (
                "surface_margin",
                "minimum_voltage_V",
                "maximum_voltage_V",
                "maximum_current_A",
            )
        },
        provenance="Explicit synthetic x-full development qualification; fixed block 1",
    )
