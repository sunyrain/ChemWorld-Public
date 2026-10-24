"""Authored Split OCV-R inputs and thin native declaration mapping."""

from copy import deepcopy

import numpy as np

F = 96485.33212
PRECISIONS = {"coarse": {"rtol": 1e-8, "atol": 1e-10}, "fine": {"rtol": 1e-10, "atol": 1e-12}}
REFUSALS = (
    "negative-capacity",
    "positive-capacity",
    "negative-R",
    "initial-fraction",
    "inverted-bounds",
    "zero-R-CV",
    "thermal",
    "chemical-energy",
)
RESPONSES = (
    "voltage_V",
    "current_A",
    "charge_C",
    "negative_fraction",
    "positive_fraction",
    "negative_ocv_V",
    "positive_ocv_V",
    "ohmic_drop_V",
)
ATOL = np.array([2e-6, 1e-7, 1e-4, 1e-8, 1e-8, 2e-6, 2e-6, 2e-6])
NAMES = (
    "rest",
    "forward-CC",
    "reverse-CC",
    "round-trip",
    "nonlinear-asymmetric",
    "zero-R-CC",
    "minimum-voltage",
    "maximum-voltage",
    "negative-minimum-fraction",
    "positive-maximum-fraction",
    "negative-maximum-fraction",
    "positive-minimum-fraction",
    "interior-CV",
    "limited-releasing-CV",
    "refusals",
    "EIS",
    "SOH",
)


def base_data():
    return {
        "negative_capacity_C": 3600.0,
        "positive_capacity_C": 4500.0,
        "negative_polynomial_V": (0.2, -0.1),
        "positive_polynomial_V": (4.4, -0.6),
        "initial_negative_fraction": 0.7,
        "initial_positive_fraction": 0.3,
        "resistance_ohm": 0.1,
        "maximum_current_A": 1.0,
        "temperature_K": 298.15,
        "housing_volume_L": 0.01,
        "negative_minimum_fraction": 0.0001,
        "negative_maximum_fraction": 0.999,
        "positive_minimum_fraction": 0.0001,
        "positive_maximum_fraction": 0.999,
        "minimum_voltage_V": 2.5,
        "maximum_voltage_V": 4.5,
        "samples_per_segment": 31,
        "expected": "complete",
        "provenance": "Self-authored polynomial Split OCV-R qualification; synthetic curves",
    }


def nonlinear_data():
    data = base_data()
    data.update(
        negative_capacity_C=1800.0,
        positive_capacity_C=7200.0,
        resistance_ohm=0.3,
        negative_polynomial_V=(0.25, -0.12, -0.02),
        positive_polynomial_V=(4.5, -0.5, -0.08),
    )
    return data


def case_data(index):
    from battery_split_ocvr_oracle import cv_at, equilibrium_polynomial, stop_time

    data = nonlinear_data() if index == 5 else base_data()
    data.update(id=f"S{index:02d}", name=NAMES[index - 1], waveform=((600.0, 0.5, "A"),))
    if index == 1:
        data["waveform"] = ((1200.0, 0.0, "A"),)
    elif index == 3:
        data["waveform"] = ((600.0, -0.5, "A"),)
    elif index == 4:
        data["waveform"] = ((600.0, 0.5, "A"), (300.0, 0.0, "A"), (600.0, -0.5, "A"))
    elif index == 5:
        data["waveform"] = ((600.0, 0.4, "A"),)
    elif index == 6:
        data["resistance_ohm"] = 0.0
    elif index == 7:
        data.update(minimum_voltage_V=3.9, expected="minimum-voltage")
    elif index == 8:
        data.update(
            maximum_voltage_V=4.2, expected="maximum-voltage", waveform=((600.0, -0.5, "A"),)
        )
    elif index == 9:
        data.update(initial_negative_fraction=0.02, expected="negative-minimum-fraction")
    elif index == 10:
        data.update(initial_positive_fraction=0.98, expected="positive-maximum-fraction")
    elif index == 11:
        data.update(
            initial_negative_fraction=0.98,
            expected="negative-maximum-fraction",
            waveform=((600.0, -0.5, "A"),),
        )
    elif index == 12:
        data.update(
            initial_positive_fraction=0.02,
            expected="positive-minimum-fraction",
            waveform=((600.0, -0.5, "A"),),
        )
    elif index in (13, 14):
        duration, offset = (600.0, 0.04) if index == 13 else (800.0, 0.08)
        if index == 14:
            data["maximum_current_A"] = 0.5
        ep = equilibrium_polynomial(data)
        first = float(ep(0.0) - offset)
        q1 = cv_at(data, 0.0, first, duration)[0]
        data["waveform"] = ((duration, first, "V"), (duration, float(ep(q1) + offset), "V"))
    elif index >= 15:
        data["waveform"] = ()
    if 7 <= index <= 12:
        time = stop_time(data, data["waveform"][0][1])
        data["analytic_stop_time_s"] = time
        data["waveform"] = ((1.1 * time, data["waveform"][0][1], "A"),)
        data["observation_horizons_s"] = (time,)
    else:
        data["observation_horizons_s"] = tuple(v[0] for v in data["waveform"])
    return data


def analysis_data(kind, name):
    data = nonlinear_data() if name == "nonlinear" else base_data()
    data["analysis_charge_C"] = (
        120.0
        if kind == "EIS" and name == "nonlinear"
        else (200.0 if kind == "SOH" and name == "shifted" else 0.0)
    )
    if name == "zero-R":
        data["resistance_ohm"] = 0.0
    if kind == "SOH":
        data["minimum_voltage_V"], data["maximum_voltage_V"] = (
            (4.05, 4.22) if name == "nonlinear" else (3.9, 4.15)
        )
    return data


def native_spec(data):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_split_ocvr import BatterySplitOCVRSpec
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    source = InputSource(
        DataSourceProvenance(
            "split-ocvr-qualification", 0, "Self-authored exact polynomial", "1", "CC0-1.0"
        ),
        "synthetic",
    )
    curves = {}
    for name in ("negative", "positive"):
        curves[name + "_ocv"] = ElectrodeOCV(
            "split-" + name,
            source,
            tuple(data[name + "_polynomial_V"]),
            stoichiometry_domain=(0.0, 1.0),
            supported_models=("SplitOCVR",),
        )
    keys = (
        "negative_capacity_C",
        "positive_capacity_C",
        "initial_negative_fraction",
        "initial_positive_fraction",
        "resistance_ohm",
        "maximum_current_A",
        "temperature_K",
        "housing_volume_L",
        "negative_minimum_fraction",
        "negative_maximum_fraction",
        "positive_minimum_fraction",
        "positive_maximum_fraction",
        "minimum_voltage_V",
        "maximum_voltage_V",
        "provenance",
    )
    return BatterySplitOCVRSpec(**{key: deepcopy(data[key]) for key in keys}, **curves)
