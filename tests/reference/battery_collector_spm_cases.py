"""Frozen authored inputs for the 45-COLLECTOR-SPM reference line."""

from copy import deepcopy

import numpy as np

F = 96485.33212
PYBAMM_VERSION = "25.4.2"
CASADI_VERSION = "3.6.7"
RTOL = 1e-6
ATOL = 1e-8
ROOT_TOL = 1e-8
SOURCE_SHORT_DURATION_S = 1e-4
SOURCE_SHORT_CURRENT_A = 0.5
PROFILE_POINTS = (0.1, 0.3, 0.5, 0.7, 0.9)
GRIDS = {"coarse": {"z": 8, "r": 12}, "fine": {"z": 16, "r": 24}}
LAYOUTS = {
    "opposite": {"negative_tab": "left", "positive_tab": "right"},
    "same_end": {"negative_tab": "right", "positive_tab": "right"},
}

BASE = {
    "width_m": 0.1,
    "height_m": 0.1,
    "collector_thickness_m": 25e-6,
    "sigma_cc_S_m": 100.0,
    "electrolyte_concentration_mol_m3": 1000.0,
    "temperature_K": 298.15,
    "minimum_voltage_V": 2.5,
    "maximum_voltage_V": 4.5,
    "maximum_current_A": 1.0,
    "series_resistance_ohm": 0.0,
    "lengths_m": (100e-6, 25e-6, 100e-6),
    "porosities": (0.4, 0.5, 0.4),
    "bruggeman": (1.5, 1.5, 1.5),
    "ocp_barrier_V": 1e-6,
    "surface_margin": 0.01,
    "negative": {
        "radius_m": 5e-6,
        "diffusivity_m2_s": 1e-13,
        "maximum_concentration_mol_m3": 30000.0,
        "active_volume_m3": 3e-7,
        "exchange_current_scale_A_m2": 4.0,
        "initial_fraction": 0.60,
        "ocv": (0.25, -0.10),
    },
    "positive": {
        "radius_m": 5e-6,
        "diffusivity_m2_s": 1e-13,
        "maximum_concentration_mol_m3": 30000.0,
        "active_volume_m3": 4e-7,
        "exchange_current_scale_A_m2": 4.0,
        "initial_fraction": 0.40,
        "ocv": (4.10, -0.40),
    },
}

# The waveforms are (duration_s, target, mode), with mode A or V.
_HISTORIES = (
    ("CC01", "uniform finite conductivity", ((120.0, 0.30, "A"),)),
    ("CC02", "uniform high conductivity", ((120.0, 0.30, "A"),), {"sigma_cc_S_m": 1e4}),
    ("CC03", "uniform low conductivity", ((120.0, 0.30, "A"),), {"sigma_cc_S_m": 25.0}),
    ("CC04", "same-end tabs", ((120.0, 0.30, "A"),), {"layouts": ("same_end",)}),
    ("CC05", "nonuniform SOC left-to-right rest", ((60.0, 0.0, "A"),)),
    ("CC06", "nonuniform SOC right-to-left rest", ((60.0, 0.0, "A"),)),
    ("CC07", "CC rest reverse", ((60.0, 0.30, "A"), (30.0, 0.0, "A"), (60.0, -0.30, "A"))),
    ("CC08", "finite CV interior", ((120.0, 3.78, "V"),)),
    ("CC09", "finite CV current-limited", ((120.0, 3.40, "V"),), {"maximum_current_A": 0.15}),
    ("CC10", "voltage and surface cutoff", ((300.0, 0.60, "A"),), {"surface_margin": 0.03}),
)
_REFUSALS = (
    (
        "CC11",
        "legacy uniform SPM high-conductivity limit",
        "refusals",
        ("legacy-uniform-SPM-high-conductivity-limit",),
    ),
    (
        "CC12",
        "unsupported collector combinations",
        "refusals",
        (
            "thermal",
            "lithium-plating",
            "hysteresis",
            "particle-size-distribution",
            "composite-phase",
            "EIS",
            "SOH",
            "invalid-tab-or-geometry",
        ),
    ),
)


def _unit(index, name, waveform, overrides=None, *, expected="complete"):
    data = deepcopy(BASE)
    data.update(deepcopy(overrides or {}))
    data["id"] = index
    data["name"] = name
    data["waveform"] = tuple(tuple(x) for x in waveform)
    data["expected"] = expected
    data["samples_per_segment"] = 31
    data["layouts"] = tuple(data.pop("layouts", LAYOUTS))
    data["grids"] = tuple(GRIDS)
    data["source_versions"] = {"pybamm": PYBAMM_VERSION, "casadi": CASADI_VERSION}
    if index == "CC05":
        data["initial_z_fractions"] = {
            "negative": (0.52, 0.56, 0.60, 0.64),
            "positive": (0.44, 0.42, 0.40, 0.38),
        }
    elif index == "CC06":
        data["initial_z_fractions"] = {
            "negative": (0.64, 0.60, 0.56, 0.52),
            "positive": (0.38, 0.40, 0.42, 0.44),
        }
    else:
        data["initial_z_fractions"] = {
            "negative": (data["negative"]["initial_fraction"],) * 4,
            "positive": (data["positive"]["initial_fraction"],) * 4,
        }
    data["refusal_cases"] = (
        (
            "thermal",
            "lithium-plating",
            "hysteresis",
            "particle-size-distribution",
            "composite-phase",
            "EIS",
            "SOH",
            "invalid-tab-or-geometry",
        )
        if expected == "refusals"
        else ()
    )
    return data


def case_data(index):
    for item in _HISTORIES:
        if item[0] == index:
            _, name, waveform, *rest = item
            return _unit(index, name, waveform, rest[0] if rest else None)
    for item in _REFUSALS:
        if item[0] == index:
            data = _unit(index, item[1], ((0.0, 0.0, "A"),), expected="refusals")
            data["refusal_cases"] = tuple(item[3])
            return data
    raise KeyError(index)


NAMES = tuple(x[0] for x in (*_HISTORIES, *_REFUSALS))
HISTORY_NAMES = tuple(x[0] for x in _HISTORIES)
REFUSAL_NAMES = tuple(x[0] for x in _REFUSALS)
EXPECTED_ROUTES = len(NAMES) * len(LAYOUTS) * len(GRIDS)
# Scientific matrix is fixed before any full run: 10 history units and two refusal units.
assert EXPECTED_ROUTES == 48


def all_cases():
    return tuple(case_data(name) for name in NAMES)


def _z_fractions(values, cells):
    """Interpolate the authored four-cell SOC profile onto the fixed z grid."""
    values = np.asarray(values, dtype=float)
    if len(values) == cells:
        return tuple(values.tolist())
    authored = np.linspace(0.0, 1.0, len(values))
    target = (np.arange(cells, dtype=float) + 0.5) / cells
    return tuple(np.interp(target, authored, values).tolist())


def native_spec(data, grid, layout):
    """Construct the native collector SPM for an authored case and grid."""
    from chemworld.physchem.battery_collector_spm import BatteryCollectorSPMSpec
    from chemworld.physchem.battery_models import ParticleSpec

    cells_z, cells_r = int(grid["z"]), int(grid["r"])
    particles = {}
    for name in ("negative", "positive"):
        p = data[name]
        particles[name] = ParticleSpec(
            radius_m=p["radius_m"],
            diffusivity_m2_s=p["diffusivity_m2_s"],
            maximum_concentration_mol_m3=p["maximum_concentration_mol_m3"],
            active_volume_m3=p["active_volume_m3"],
            cells=cells_r,
            mesh_exponent=1.0,
            exchange_current_scale_A_m2=p["exchange_current_scale_A_m2"],
        )
    return BatteryCollectorSPMSpec(
        negative=particles["negative"],
        positive=particles["positive"],
        initial_negative_fractions=_z_fractions(data["initial_z_fractions"]["negative"], cells_z),
        initial_positive_fractions=_z_fractions(data["initial_z_fractions"]["positive"], cells_z),
        height_m=data["height_m"],
        width_m=data["width_m"],
        negative_collector_thickness_m=data["collector_thickness_m"],
        positive_collector_thickness_m=data["collector_thickness_m"],
        negative_collector_conductivity_S_m=data["sigma_cc_S_m"],
        positive_collector_conductivity_S_m=data["sigma_cc_S_m"],
        collector_cells=cells_z,
        negative_tab=layout["negative_tab"],
        negative_ocv_intercept_V=data["negative"]["ocv"][0],
        negative_ocv_slope_V=data["negative"]["ocv"][1],
        positive_ocv_intercept_V=data["positive"]["ocv"][0],
        positive_ocv_slope_V=data["positive"]["ocv"][1],
        temperature_K=data["temperature_K"],
        surface_margin=data["surface_margin"],
        minimum_voltage_V=data["minimum_voltage_V"],
        maximum_voltage_V=data["maximum_voltage_V"],
        maximum_current_A=data["maximum_current_A"],
        series_resistance_ohm=data["series_resistance_ohm"],
        provenance="collector-spm45-authored-native-v1",
    )


def route_plan(index):
    """Return the fixed four routes for one unit."""
    data = case_data(index)
    if data["expected"] == "refusals":
        return [
            (f"refusal_{layout}_{grid}", "refusal", layout, grid)
            for layout in LAYOUTS
            for grid in GRIDS
        ]
    return [
        (f"history_{layout}_{grid}", "history", layout, grid)
        for layout in LAYOUTS
        for grid in GRIDS
    ]
