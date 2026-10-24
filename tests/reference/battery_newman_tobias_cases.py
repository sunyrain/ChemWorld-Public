"""Fixed synthetic Newman-Tobias inputs, independent of production calculations."""

from copy import deepcopy

import numpy as np

F, SOURCE_F, R = 96485.33212, 96485.33212331001, 8.31446261815324
RTOL, ATOL = 1e-9, 1e-11
SOURCE_ROOT_TOL = 1e-6
SOURCE_ALG_ATOL = 1e-7
GRIDS = {"native": ((12, 6, 12), (24, 12, 24)), "reference": ((24, 12, 24), (48, 24, 48))}
PROFILE_POINTS = np.linspace(0.05, 0.95, 11)
REFUSALS = (
    "invalid-active-volume",
    "nonpositive-conductivity",
    "invalid-inventory",
    "radial-state",
    "EIS",
    "SOH",
    "energy",
    "thermal",
)
BASE = {
    "temperature_K": 298.15,
    "electrolyte_concentration_mol_m3": 1000.0,
    "reference_electrolyte_concentration_mol_m3": 1000.0,
    "area_m2": 0.01,
    "lengths_m": (1e-4, 2.5e-5, 1e-4),
    "porosities": (0.4, 0.5, 0.4),
    "bruggeman": (1.5, 1.5, 1.5),
    "solid_bruggeman": (1.5, 1.5),
    "conductivity_S_m": 1.0,
    "solid_conductivity_S_m": (100.0, 100.0),
    "surface_margin": 0.01,
    "minimum_voltage_V": 2.5,
    "maximum_voltage_V": 4.5,
    "maximum_current_A": 2.0,
    "resistance_ohm": 0.0,
    "samples_per_segment": 31,
    "waveform": ((240.0, 0.6, "A"),),
    "expected": "complete",
    "electrodes": [
        {
            "name": "negative",
            "radius_m": 5e-6,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 5e-7,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.7,
            "polynomial": (0.2, -0.1),
        },
        {
            "name": "positive",
            "radius_m": 5e-6,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 5e-7,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.3,
            "polynomial": (4.2, -0.8),
        },
    ],
}
NAMES = (
    "rest-exact-limit",
    "forward-CC",
    "reverse-CC",
    "CC-rest-reverse",
    "low-electrolyte-conductivity",
    "low-solid-conductivity",
    "asymmetric-geometry-and-kinetics",
    "high-conductivity-BV-limit",
    "interior-CV",
    "both-CV-current-limits",
    "voltage-cutoff",
    "domain-refusals",
)
RESPONSES = (
    "voltage_V",
    "current_A",
    "charge_C",
    "work_J",
    "negative_fraction",
    "positive_fraction",
    "negative_amount_mol",
    "positive_amount_mol",
)
RESPONSE_ATOL = np.array([0.002, 0.005, 0.02, 0.1, 2e-5, 2e-5, 2e-8, 2e-8])
RESPONSE_RTOL = np.array([1e-5, 0.01, 1e-5, 1e-5, 0.0, 0.0, 1e-6, 1e-6])
FIELD_TOLERANCES = {
    "potential_V": (0.002, 1e-5),
    "face_current_A": (0.005, 0.01),
    "reaction_density_A": (0.01, 0.02),
}


def case_data(index):
    if index not in range(1, 13):
        raise ValueError("fixed NT block contains NT01-NT12")
    data = deepcopy(BASE)
    data["name"] = NAMES[index - 1]
    if index == 1:
        data["waveform"] = ((240.0, 0.0, "A"),)
    elif index == 3:
        data["electrodes"][0]["initial_fraction"] = 0.45
        data["electrodes"][1]["initial_fraction"] = 0.55
        data["waveform"] = ((240.0, -0.6, "A"),)
    elif index == 4:
        data["waveform"] = (
            (100.0, 0.6, "A"),
            (80.0, 0.0, "A"),
            (100.0, -0.4, "A"),
            (60.0, 0.0, "A"),
        )
    elif index == 5:
        data["conductivity_S_m"] = 0.15
        data["nonuniform_current"] = True
    elif index == 6:
        data["solid_conductivity_S_m"] = (0.2, 0.1)
        data["nonuniform_current"] = True
    elif index == 7:
        data.update(
            lengths_m=(1.4e-4, 4e-5, 8e-5),
            porosities=(0.35, 0.6, 0.45),
            bruggeman=(1.3, 1.8, 1.4),
            temperature_K=310.0,
            electrolyte_concentration_mol_m3=800.0,
        )
        data["electrodes"][0].update(active_volume_m3=7e-7, exchange_current_scale_A_m2=2.0)
        data["electrodes"][1].update(active_volume_m3=4e-7, exchange_current_scale_A_m2=7.0)
    elif index == 8:
        data.update(
            conductivity_S_m=100.0, solid_conductivity_S_m=(1e4, 1e4), analytic_BV_limit=True
        )
    elif index == 9:
        data.update(waveform=((240.0, 3.72, "V"),), control_expectation="interior")
    elif index == 10:
        data.update(
            waveform=((120.0, 3.4, "V"), (120.0, 4.3, "V")),
            maximum_current_A=0.3,
            control_expectation="both_limits",
        )
    elif index == 11:
        data.update(
            minimum_voltage_V=3.6, waveform=((900.0, 0.6, "A"),), expected="minimum-voltage"
        )
    elif index == 12:
        data["expected"] = "refusals"
    return data


def source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(
        DataSourceProvenance(
            "newman-tobias-synthetic",
            0,
            "Explicit synthetic uniform-concentration porous-electrode parameters; "
            "actual PyBaMM25.4.2 NewmanTobias equations, not fitted Chen2020 chemistry",
            "fixed-NT-block-1",
            "authored qualification inputs",
        ),
        "synthetic",
    )


def native_spec(data, cells):
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_newman_tobias import (
        AveragedElectrodeSpec,
        BatteryNewmanTobiasSpec,
    )
    from chemworld.physchem.battery_porous_electrical import PorousElectrodeGeometrySpec

    electrodes, curves = [], []
    for e in data["electrodes"]:
        electrodes.append(
            AveragedElectrodeSpec(
                **{
                    k: e[k]
                    for k in (
                        "radius_m",
                        "maximum_concentration_mol_m3",
                        "active_volume_m3",
                        "exchange_current_scale_A_m2",
                    )
                }
            )
        )
        curves.append(
            ElectrodeOCV(
                e["name"],
                source(),
                e["polynomial"],
                supported_models=("NewmanTobias",),
                stoichiometry_domain=(data["surface_margin"], 1 - data["surface_margin"]),
            )
        )
    geometry = PorousElectrodeGeometrySpec(
        cells=tuple(cells),
        **{
            k: data[k]
            for k in (
                "area_m2",
                "lengths_m",
                "porosities",
                "bruggeman",
                "conductivity_S_m",
                "solid_conductivity_S_m",
            )
        },
    )
    return BatteryNewmanTobiasSpec(
        electrodes[0],
        electrodes[1],
        geometry,
        negative_ocv=curves[0],
        positive_ocv=curves[1],
        initial_negative_fraction=data["electrodes"][0]["initial_fraction"],
        initial_positive_fraction=data["electrodes"][1]["initial_fraction"],
        **{
            k: data[k]
            for k in (
                "temperature_K",
                "electrolyte_concentration_mol_m3",
                "reference_electrolyte_concentration_mol_m3",
                "surface_margin",
                "minimum_voltage_V",
                "maximum_voltage_V",
                "maximum_current_A",
                "resistance_ohm",
            )
        },
        provenance="Fixed synthetic NT development qualification",
    )
