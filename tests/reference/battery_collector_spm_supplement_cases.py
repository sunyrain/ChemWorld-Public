"""Prospective collector-SPM behavior/physical-port supplementary block.

These are new authored questions. The original 12-unit/48-route cases and all
their outcomes remain unchanged. One paired history route executes two models.
"""

from copy import deepcopy

from battery_collector_spm_cases import BASE, CASADI_VERSION, GRIDS, LAYOUTS, PYBAMM_VERSION

THRESHOLDS = {
    "time_s": 1e-8,
    "sample_time_s": 1e-7,
    "Li_mol": 2e-10,
    "Faraday_C": 1e-5,
    "current_A": 1e-8,
    "CV_voltage_V": 1e-6,
    "CV_work_J": 1e-5,
    "green_voltage_V": 1e-8,
    "green_power_W": 1e-8,
    "static_voltage_V": 1e-7,
    "static_current_A": 1e-7,
    "response_voltage_V": 2e-3,
    "response_current_A": 1e-3,
    "response_charge_C": 5e-2,
    "response_work_J": 2e-1,
    "response_Li_mol": 5e-7,
    "response_interval_current_A": 2e-3,
    "high_sigma_voltage_V": 3e-5,
    "high_sigma_current_A": 1e-8,
    "high_sigma_charge_C": 1e-4,
    "high_sigma_work_J": 1e-3,
    "high_sigma_Li_mol": 2e-9,
    "event_time_s": 1.0,
    "event_voltage_V": 1e-6,
    "event_surface_fraction": 1e-6,
}

NAMES = tuple(f"S{i:02d}" for i in range(1, 11))
REFUSAL_NAMES = (
    "thermal",
    "electrolyte",
    "SEI",
    "plating",
    "invalid-tab",
    "negative-geometry",
    "EIS",
    "SOH",
)
HISTORY_ROUTES = 36
BASELINE_ROUTES = 4
REFUSAL_ROUTES = 8
EXPECTED_ROUTES = HISTORY_ROUTES + BASELINE_ROUTES + REFUSAL_ROUTES
EXPECTED_HISTORIES = 2 * HISTORY_ROUTES + BASELINE_ROUTES


def case_data(index):
    if isinstance(index, int):
        index = f"S{index:02d}"
    if index not in NAMES:
        raise KeyError(index)
    data = deepcopy(BASE)
    data.update(
        id=index,
        sigma_cc_S_m=1e4,
        samples_per_segment=31,
        source_versions={"pybamm": PYBAMM_VERSION, "casadi": CASADI_VERSION},
        layouts=tuple(LAYOUTS),
        grids=tuple(GRIDS),
        expected="complete",
        controlled_port="physical-tab",
        waveform=((120.0, 0.03, "A"),),
    )
    descriptions = {
        "S01": "feasible finite-conductivity CC",
        "S02": "nonuniform SOC internal redistribution at zero port current",
        "S03": "continuous CC rest reverse",
        "S04": "interior physical-tab CV",
        "S05": "positive-current-limited physical-tab CV",
        "S06": "negative-current-limited physical-tab CV",
        "S07": "finite high-conductivity limit against actual uniform SPM",
        "S08": "finite-time physical minimum-voltage cutoff",
        "S09": "finite-time negative surface depletion cutoff",
        "S10": "actual public-consumer and construction refusals",
    }
    data["name"] = descriptions[index]
    if index == "S02":
        data["waveform"] = ((120.0, 0.0, "A"),)
    elif index == "S03":
        data["waveform"] = ((120.0, 0.03, "A"), (60.0, 0.0, "A"), (120.0, -0.03, "A"))
    elif index in ("S04", "S05", "S06"):
        data["waveform"] = ((120.0, {"S04": 3.78, "S05": 3.40, "S06": 4.10}[index], "V"),)
        data["expected_mode"] = {
            "S04": "interior",
            "S05": "positive-limit",
            "S06": "negative-limit",
        }[index]
        if index != "S04":
            data["maximum_current_A"] = 0.001
    elif index == "S07":
        data["sigma_cc_S_m"] = 1e8
    elif index in ("S08", "S09"):
        data["sigma_cc_S_m"] = 1e8
        for name in ("negative", "positive"):
            data[name]["active_volume_m3"] /= 100
            data[name]["diffusivity_m2_s"] *= 100
            data[name]["exchange_current_scale_A_m2"] *= 10
        data["expected"] = "stopped"
        if index == "S08":
            data["waveform"] = ((60.0, 0.03, "A"),)
            data["minimum_voltage_V"] = 3.68
            data["expected_event"] = "minimum-voltage"
        else:
            data["negative"]["initial_fraction"] = 0.03
            data["waveform"] = ((25.0, 0.01, "A"),)
            data["expected_event"] = "negative-minimum-surface"
    elif index == "S10":
        data["expected"] = "refusals"
        data["refusal_cases"] = REFUSAL_NAMES
    data["initial_z_fractions"] = {
        name: (data[name]["initial_fraction"],) * 4 for name in ("negative", "positive")
    }
    if index == "S02":
        data["initial_z_fractions"] = {
            "negative": (0.52, 0.56, 0.60, 0.64),
            "positive": (0.44, 0.42, 0.40, 0.38),
        }
    return data


def route_plan(index):
    data = case_data(index)
    if data["id"] == "S10":
        return [("refusal_" + name, "refusal", None, "coarse") for name in REFUSAL_NAMES]
    routes = [
        (f"history_{layout}_{grid}", "history", layout, grid)
        for layout in LAYOUTS
        for grid in GRIDS
    ]
    if data["id"] == "S07":
        routes += [
            (f"uniform_{kind}_{grid}", kind + "-uniform", None, grid)
            for kind in ("native", "source")
            for grid in GRIDS
        ]
    return routes


assert sum(len(route_plan(name)) for name in NAMES) == EXPECTED_ROUTES == 48
assert EXPECTED_HISTORIES == 76
