"""Declared synthetic inputs for a fixed three-material SPM qualification block."""

from copy import deepcopy

import numpy as np

F = 96485.33212
GRIDS = {"native": (40, 80), "reference": (120, 200)}
RTOL, ATOL = 1e-9, 1e-11
PROFILE_RADII = np.linspace(.05, .95, 9)
PHASES = ("host_a", "host_b", "cathode")
BASE = {
    "collector_area_m2": .01, "electrode_thickness_m": 1e-4,
    "temperature_K": 298.15, "electrolyte_concentration_mol_m3": 1000.,
    "surface_margin": .01, "minimum_voltage_V": 2.5, "maximum_voltage_V": 4.5,
    "maximum_current_A": 2., "series_resistance_ohm": .02,
    "samples_per_segment": 41, "waveform": ((300., .6, "A"),),
    "expected": "complete",
    "phases": [
        {"phase_id": "host_a", "electrode": "negative", "radius_m": 5e-6,
         "diffusivity_m2_s": 1e-13, "maximum_concentration_mol_m3": 30000.,
         "active_volume_m3": 4e-7, "exchange_current_scale_A_m2": 4.,
         "initial_fraction": .6, "polynomial": (.3, -.25)},
        {"phase_id": "host_b", "electrode": "negative", "radius_m": 5e-6,
         "diffusivity_m2_s": 1e-13, "maximum_concentration_mol_m3": 30000.,
         "active_volume_m3": 2e-7, "exchange_current_scale_A_m2": 4.,
         "initial_fraction": .6, "polynomial": (.38, -.35)},
        {"phase_id": "cathode", "electrode": "positive", "radius_m": 5e-6,
         "diffusivity_m2_s": 1e-13, "maximum_concentration_mol_m3": 30000.,
         "active_volume_m3": 6e-7, "exchange_current_scale_A_m2": 4.,
         "initial_fraction": .35, "polynomial": (4.2, -.8)},
    ],
}
NAMES = ("identical-material-limit", "distinct-material-discharge", "reverse-charge",
         "zero-terminal-internal-exchange", "CC-rest-reverse", "diffusion-contrast",
         "kinetics-contrast", "interior-CV", "limited-CV", "individual-phase-cutoff",
         "voltage-cutoff", "domain-refusals")
REFUSALS = ("duplicate-phase-id", "wrong-topology", "nonmonotone-ocv", "zero-diffusivity",
            "EIS", "SOH", "energy", "thermal")
RESPONSES = ("voltage_V", "current_A", "charge_C", "work_J", *(
    f"{phase}:{name}" for phase in PHASES for name in (
        "reaction_current_A", "surface_fraction", "average_fraction", "inventory_mol",
        "kinetic_overpotential_V")))
RESPONSE_ATOL = np.array([.002, .005, .02, .1, *([.005, .003, .003, 2e-7, .001] * 3)])
RESPONSE_RTOL = np.array([1e-5, .005, 1e-5, 1e-5, *([.005, 0., 0., 1e-6, 1e-5] * 3)])


def case_data(index):
    if index not in range(1, 13):
        raise ValueError("composite block fixes exactly CP01-CP12")
    data = deepcopy(BASE)
    data["name"] = NAMES[index - 1]
    a, b, p = data["phases"]
    if index == 1:
        b["polynomial"] = a["polynomial"]
        data["same_material_limit"] = True
    elif index == 3:
        a["initial_fraction"] = b["initial_fraction"] = .45
        p["initial_fraction"] = .55
        data["waveform"] = ((300., -.6, "A"),)
    elif index == 4:
        data["waveform"] = ((300., 0., "A"),)
        data["internal_exchange"] = True
    elif index == 5:
        data["waveform"] = ((100., .6, "A"), (100., 0., "A"),
                            (100., -.4, "A"), (60., 0., "A"))
    elif index == 6:
        a["diffusivity_m2_s"], b["diffusivity_m2_s"] = 5e-13, 2e-14
    elif index == 7:
        a["exchange_current_scale_A_m2"], b["exchange_current_scale_A_m2"] = .8, 6.
        p["exchange_current_scale_A_m2"] = 2.
    elif index == 8:
        data["waveform"], data["control_expectation"] = ((300., 3.74, "V"),), "interior"
    elif index == 9:
        data["waveform"], data["control_expectation"] = ((300., 3.4, "V"),), "positive_limit"
        data["maximum_current_A"] = .3
    elif index == 10:
        a["polynomial"] = b["polynomial"] = (.2,)
        a["initial_fraction"] = .03
        for phase in data["phases"]:
            phase["diffusivity_m2_s"] = 1e-12
        data["waveform"] = ((300., .8, "A"),)
        data["expected"] = "battery:negative:phase:host_a:depleted"
    elif index == 11:
        data["minimum_voltage_V"] = 3.62
        data["waveform"] = ((600., .8, "A"),)
        data["expected"] = "battery:minimum-voltage"
    elif index == 12:
        data["expected"] = "refusals"
    return data


def source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(DataSourceProvenance("composite-spm-synthetic", 0,
        "Explicit synthetic single-valued linear OCV and particle inputs; equations compared "
        "with PyBaMM25.4.2 multiphase SPM, not Chen2020 composite material data",
        "fixed-composite-block-1", "authored qualification inputs"), "synthetic")


def native_spec(data, cells):
    from chemworld.physchem.battery_composite import (
        ActiveMaterialPhaseSpec,
        BatteryCompositeSPMSpec,
    )
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_models import ParticleSpec

    phases = []
    for value in data["phases"]:
        particle = ParticleSpec(**{k: value[k] for k in ("radius_m", "diffusivity_m2_s",
            "maximum_concentration_mol_m3", "active_volume_m3", "exchange_current_scale_A_m2")},
            cells=cells, mesh_exponent=1.)
        curve = ElectrodeOCV(value["phase_id"], source(), value["polynomial"],
            supported_models=("SPM",),
            stoichiometry_domain=(data["surface_margin"], 1 - data["surface_margin"]))
        phases.append(ActiveMaterialPhaseSpec(value["phase_id"], particle, curve,
                                              value["initial_fraction"], source()))
    return BatteryCompositeSPMSpec(tuple(phases[:2]), phases[2],
        **{k: data[k] for k in ("temperature_K", "surface_margin", "minimum_voltage_V",
            "maximum_voltage_V", "maximum_current_A", "series_resistance_ohm")},
        provenance="Explicit synthetic three-material SPM development qualification")
