"""Prospective Axen/main-current SPM inputs: 13 units, 70 routes."""

from copy import deepcopy
from dataclasses import replace

from battery_plating_cases import ATOL, GRIDS, PROFILE_RADII, RESPONSES, ROOT_TOL, RTOL, F, R
from battery_plating_cases import case_data as prior_case
from battery_plating_cases import native_spec as single_spec

__all__ = ["ATOL", "GRIDS", "PROFILE_RADII", "RESPONSES", "ROOT_TOL", "RTOL", "F", "R"]
NAMES = (
    "zero-k-old-Axen",
    "identical-branches-old-plating",
    "negative-Axen-charge",
    "positive-Axen-charge",
    "dual-Axen-reversal",
    "internal-rest-main-current",
    "h-endpoints-and-reversal",
    "kinetics-geometry-temperature",
    "interior-CV",
    "signed-limited-CV",
    "surface-stop",
    "voltage-stop",
    "refusals",
)
REFUSALS = (
    "initial-h-domain",
    "negative-decay",
    "missing-decay",
    "wrong-decay-unit",
    "MPM",
    "sigmoid",
    "partial",
    "irreversible",
    "SEI",
    "thermal",
    "EIS",
    "SOH",
    "chemical-energy",
    "negative-plated",
)


def case_data(index):
    if index not in range(1, 14):
        raise ValueError("Axen plating fixes exactly A01-A13")
    data = deepcopy(prior_case({5: 5, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}.get(index, 2)))
    data.update(
        index=index,
        name=NAMES[index - 1],
        mode="reversible",
        negative_mode="Axen",
        positive_mode="single",
        negative_gap_V=0.02,
        positive_gap_V=0.04,
        negative_initial_h=0.2,
        positive_initial_h=-0.3,
        lithiation_decay=20.0,
        delithiation_decay=40.0,
    )
    if index == 1:
        data.update(
            k_m_s=0.0,
            positive_mode="Axen",
            waveform=((150.0, 0.4, "A"), (50.0, 0.0, "A"), (150.0, -0.4, "A")),
        )
    elif index == 2:
        data.update(
            negative_gap_V=0.0,
            positive_gap_V=0.0,
            positive_mode="Axen",
            waveform=((150.0, -0.6, "A"), (100.0, 0.0, "A"), (150.0, 0.4, "A")),
        )
    elif index == 4:
        data.update(negative_mode="single", positive_mode="Axen")
    elif index == 5:
        data["positive_mode"] = "Axen"
    elif index == 6:
        data.update(initial_plated_mol=0.002, waveform=((300.0, 0.0, "A"),))
    elif index == 7:
        data.update(
            positive_mode="Axen",
            negative_initial_h=1.0,
            positive_initial_h=-1.0,
            waveform=((120.0, 0.4, "A"), (120.0, -0.4, "A"), (80.0, 0.0, "A")),
        )
    elif index == 13:
        data["expected"] = "refusals"
    return data


def active_electrodes(data, *, baseline=False):
    if baseline and data["index"] == 2:
        return ()
    return tuple(e["name"] for e in data["electrodes"] if data[e["name"] + "_mode"] == "Axen")


def native_spec(data, cells, *, baseline=False):
    from battery_thevenin_cases import table

    from chemworld.physchem.battery_ocp import ElectrodeOCPSpec

    original = single_spec(data, cells, old_spm=baseline and data["index"] == 1)
    if baseline and data["index"] == 2:
        return original
    values = {}
    for e in data["electrodes"]:
        name = e["name"]
        if data[name + "_mode"] != "Axen":
            continue
        curve = getattr(original, name + "_ocv")
        gap = data[name + "_gap_V"]
        branches = {}
        for key, offset in (("lithiation", -gap / 2), ("delithiation", gap / 2)):
            coeffs = (e["polynomial"][0] + offset, *e["polynomial"][1:])
            branches[key] = replace(curve, polynomial_V=coeffs)
        row = {
            "initial_h": data[name + "_initial_h"],
            "lithiation_decay": data["lithiation_decay"],
            "delithiation_decay": data["delithiation_decay"],
        }
        values[name + "_ocp"] = ElectrodeOCPSpec(
            mode="axen",
            **branches,
            parameters=table(name + "-axen-plating", dict.fromkeys(row, "dimensionless"), [row]),
            parameter_row_id="0",
        )
    return replace(original, **values, provenance="Declared synthetic Axen plus reversible plating")
