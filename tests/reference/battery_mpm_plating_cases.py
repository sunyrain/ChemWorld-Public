"""Declared inputs for the complete 12-unit/62-route MPM plating block."""

from copy import deepcopy
from dataclasses import fields, replace

import numpy as np
from battery_plating_cases import (
    ATOL,
    REFUSALS,
    RESPONSES,
    ROOT_TOL,
    RTOL,
    F,
    R,
)
from battery_plating_cases import case_data as spm_data
from battery_plating_cases import native_spec as spm_spec

GRIDS = {"native": ((8, 24), (16, 40)), "reference": ((20, 48), (32, 80))}
RADIAL_PROBES = np.array((0.1, 0.3, 0.5, 0.7, 0.9))
SIZE_PROBES = np.array((0.1, 0.3, 0.5, 0.7, 0.9))
RESPONSE_ATOL = np.array(
    [
        0.003,
        0.008,
        0.08,
        0.3,
        0.008,
        3e-7,
        3e-7,
        0.003,
        0.3,
        *([0.008, 0.01, 0.01, 3e-7, 0.003, 0.003] * 2),
    ]
)
RESPONSE_RTOL = np.array(
    [
        1e-5,
        0.01,
        1e-5,
        1e-5,
        0.01,
        1e-5,
        1e-5,
        1e-5,
        0.01,
        *([0.01, 0.0, 0.0, 1e-5, 1e-5, 1e-5] * 2),
    ]
)


def case_data(index):
    if type(index) is not int or not 1 <= index <= 12:
        raise ValueError("fixed MPM plating block contains MP01 through MP12")
    data = deepcopy(spm_data(2))
    data.update(
        index=index,
        id=f"MP{index:02d}",
        samples_per_segment=31,
        waveform=((240.0, -0.6, "A"),),
        expected="complete",
        single_bin=False,
    )
    data["pdfs"] = {
        "negative": {"median_m": 10e-6, "sigma": 0.35, "low_m": 4e-6, "high_m": 20e-6},
        "positive": {"median_m": 8e-6, "sigma": 0.25, "low_m": 3e-6, "high_m": 16e-6},
    }
    if index == 1:
        data.update(
            single_bin=True, waveform=((120.0, -0.4, "A"), (60.0, 0.0, "A"), (120.0, 0.4, "A"))
        )
    elif index == 2:
        data.update(k_m_s=0.0, waveform=((120.0, 0.4, "A"), (60.0, 0.0, "A"), (120.0, -0.4, "A")))
    elif index == 4:
        data.update(initial_plated_mol=0.002, waveform=((240.0, 0.0, "A"),))
    elif index == 5:
        data["waveform"] = (
            (120.0, -0.6, "A"),
            (120.0, 0.0, "A"),
            (120.0, 0.4, "A"),
            (60.0, 0.0, "A"),
        )
    elif index == 6:
        data.update(
            mode="irreversible", waveform=((120.0, -0.6, "A"), (60.0, 0.0, "A"), (120.0, 0.4, "A"))
        )
    elif index == 7:
        data["pdfs"] = {
            "negative": {"median_m": 12e-6, "sigma": 0.45, "low_m": 3e-6, "high_m": 24e-6},
            "positive": {"median_m": 6e-6, "sigma": 0.35, "low_m": 2e-6, "high_m": 16e-6},
        }
        data.update(
            negative_electrode_volume_m3=4e-6,
            temperature_K=318.15,
            k_m_s=5e-9,
            alpha=0.65,
            waveform=((240.0, -0.4, "A"),),
        )
        data["electrodes"][0]["diffusivity_m2_s"] = 5e-14
    elif index == 8:
        data["waveform"] = ((120.0, 4.01, "V"), (120.0, 3.98, "V"))
    elif index == 9:
        data.update(maximum_current_A=0.3, waveform=((80.0, 4.4, "V"), (80.0, 3.5, "V")))
    elif index == 10:
        data["electrodes"][0]["initial_fraction"] = 0.03
        for e in data["electrodes"]:
            e["diffusivity_m2_s"] = 1e-12
        data.update(waveform=((300.0, 0.8, "A"),), expected="battery:negative:depleted")
    elif index == 11:
        data.update(
            minimum_voltage_V=3.85,
            waveform=((600.0, 0.8, "A"),),
            expected="battery:minimum-voltage",
        )
    elif index == 12:
        data["expected"] = "refusals"
    return data


def native_spec(data, size_nodes, radial_cells, *, old_mpm=False, single_spm=False):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.battery_mpm import BatteryMPMSpec
    from chemworld.physchem.battery_particles import lognormal_size_distribution
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    spec = spm_spec(data, radial_cells, old_spm=old_mpm)
    if single_spm:
        return spec
    values = {f.name: getattr(spec, f.name) for f in fields(spec) if f.init}
    for name in ("negative", "positive"):
        curve = getattr(spec, name + "_ocv")
        values[name + "_ocv"] = replace(curve, supported_models=("SPM", "MPM"))
        if data["single_bin"]:
            values[name + "_distribution"] = None
        else:
            pdf = data["pdfs"][name]
            source = InputSource(
                DataSourceProvenance(
                    "declared-mpm-plating-area-pdf",
                    0,
                    "Synthetic truncated area lognormal; no measured-material claim",
                    "1",
                    "CC0-1.0",
                ),
                "synthetic",
            )
            values[name + "_distribution"] = lognormal_size_distribution(
                median_radius_m=pdf["median_m"],
                log_standard_deviation=pdf["sigma"],
                lower_radius_m=pdf["low_m"],
                upper_radius_m=pdf["high_m"],
                nodes=size_nodes,
                source=source,
                basis="area",
            )
    return BatteryMPMSpec(**values)


def comparison_radii(data, name):
    if data["single_bin"]:
        return np.full(5, next(e["radius_m"] for e in data["electrodes"] if e["name"] == name))
    pdf = data["pdfs"][name]
    return pdf["low_m"] + SIZE_PROBES * (pdf["high_m"] - pdf["low_m"])


__all__ = ["ATOL", "REFUSALS", "RESPONSES", "ROOT_TOL", "RTOL", "F", "R"]
