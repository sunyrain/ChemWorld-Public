"""Prospective Chen2020 dynamic inputs; preparation is not block execution.

Both implementations keep their own radial meshes. Equal cell counts do not
mean identical finite-volume operators or initial boundary extrapolations.
The coordinator's experiment note must bind the final thresholds before GO.
"""

from copy import deepcopy
from math import isfinite

PYBAMM_VERSION = "25.4.2"
CASADI_VERSION = "3.6.7"
RTOL, ATOL, ROOT_TOL = 1e-9, 1e-11, 1e-10
GRIDS = {"native": (32, 64), "source": (32, 64)}
GRID_LEVELS = ("coarse", "fine")
CASE_IDS = tuple(f"P{i:02d}" for i in range(1, 7))
RESPONSE_ORDER = (
    "voltage_V", "current_A", "charge_C", "work_J",
    "negative_Li_mol", "positive_Li_mol",
    "negative_surface_fraction", "positive_surface_fraction",
    "negative_average_fraction", "positive_average_fraction",
    "negative_ocp_V", "positive_ocp_V",
    "negative_overpotential_V", "positive_overpotential_V",
)
PROFILE_R_OVER_R = tuple(i / 20 for i in range(1, 20))
THRESHOLD_CANDIDATES = {
    "cross_implementation": {
        "voltage_V": 5e-3, "current_A": 5e-2, "charge_C": 5.0, "work_J": 25.0,
        "electrode_Li_mol": 5e-5, "surface_fraction": 5e-3,
        "average_fraction": 3e-4, "radial_profile_fraction": 5e-3, "event_time_s": 2.0,
    },
    "within_implementation_refinement_factor": 0.5,
    "particle_Li_closure_mol": 1e-9,
    "Faraday_closure_C": 1e-4,
    "CV_voltage_V": 1e-6,
    "hardware_current_A": 1e-7,
    "source_original_voltage_event_V": 2e-5,
    "native_voltage_event_V": 1e-6,
}


def _stage(duration, target, unit="A", samples=31, expected="complete"):
    return {"duration_s": float(duration), "target": float(target), "unit": unit,
            "samples": samples, "expected": expected}


_CASES = {
    "P01": ("released-initial-rest", (_stage(600, 0),)),
    "P02": ("one-ampere-discharge", (_stage(1800, 1),)),
    "P03": ("five-ampere-voltage-cutoff",
            (_stage(5000, 5, samples=101, expected="minimum-voltage"),)),
    "P04": ("pre-discharge-then-charge-cutoff",
            (_stage(1800, 1), _stage(3000, -1, samples=101, expected="maximum-voltage"))),
    "P05": ("CC-rest-reverse-rest",
            (_stage(600, 1), _stage(300, 0), _stage(600, -.5), _stage(300, 0))),
    "P06": ("pre-discharge-rest-interior-CV",
            (_stage(1200, 1), _stage(300, 0), _stage(600, 4.05, "V"))),
}


def case_data(identity):
    if type(identity) is int:
        identity = f"P{identity:02d}"
    if identity not in _CASES:
        raise ValueError("unknown source-parameter dynamic unit")
    name, stages = _CASES[identity]
    return {
        "id": identity, "name": name, "purpose": "prospective qualification",
        "source_name": "Chen2020", "source_version": "PyBaMM-25.4.2",
        "temperature_K": 298.15, "ce_mol_m3": 1000.0,
        "source_nominal_capacity_Ah": 5.0, "source_default_current_A": 5.0,
        "maximum_current_A": 10.0, "surface_margin": .01,
        "common_event_scale": 1e4, "control_gain_A_V": 1.0,
        "stages": deepcopy(stages),
        "meshes": {
            "native": {"radial_cells": GRIDS["native"], "mesh_exponent": 2.0,
                       "geometry": "original ParticleSpec spherical finite volumes"},
            "source": {"radial_cells": GRIDS["source"], "submesh": "Uniform1DSubMesh",
                       "geometry": "original PyBaMM dimensional particle radii"},
        },
        "comparison_policy": {
            "preserve_all_original_times_states_and_event_tails": True,
            "common_available_interval_only_for_paired_response": True,
            "compare_each_actual_stop_time_event_and_endpoint_separately": True,
            "equal_cell_counts_are_not_equal_meshes": True,
            "fixed_common_profile_r_over_R": PROFILE_R_OVER_R,
            "no_parameter_or_OCP_barrier_compensation": True,
        },
    }


def validate_case(data):
    if (data["source_name"] != "Chen2020" or data["source_version"] != "PyBaMM-25.4.2"
            or data["temperature_K"] != 298.15 or data["ce_mol_m3"] != 1000.0
            or data["source_nominal_capacity_Ah"] != 5.0
            or data["source_default_current_A"] != 5.0):
        raise ValueError("source dynamic block requires the released fixed-T Chen2020 identity")
    for name in ("maximum_current_A", "surface_margin", "common_event_scale", "control_gain_A_V"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("invalid declared source-adapter control policy")
    if (data["maximum_current_A"] <= 0 or not 1e-8 < data["surface_margin"] < .5
            or data["common_event_scale"] <= 0 or data["control_gain_A_V"] <= 0):
        raise ValueError("invalid declared source-adapter control policy")
    if not data["stages"]:
        raise ValueError("source dynamic unit needs an explicit stage")
    for stage in data["stages"]:
        if (set(stage) != {"duration_s", "target", "unit", "samples", "expected"}
                or stage["unit"] not in {"A", "V"}
                or isinstance(stage["duration_s"], bool)
                or not isinstance(stage["duration_s"], (int, float))
                or not isfinite(stage["duration_s"])
                or stage["duration_s"] <= 0 or isinstance(stage["target"], bool)
                or not isinstance(stage["target"], (int, float))
                or not isfinite(stage["target"]) or type(stage["samples"]) is not int
                or stage["samples"] < 2):
            raise ValueError("invalid source dynamic stage")
        if stage["unit"] == "A" and abs(stage["target"]) > data["maximum_current_A"]:
            raise ValueError("CC request exceeds the separately declared hardware current")
    return data


def stage_times(stage):
    return tuple(stage["duration_s"] * i / (stage["samples"] - 1)
                 for i in range(stage["samples"]))


def routes():
    return tuple({"unit": unit, "implementation": implementation,
                  "grid_level": level, "radial_cells": GRIDS[implementation][index]}
                 for unit in CASE_IDS for implementation in ("native", "source")
                 for index, level in enumerate(GRID_LEVELS))


def native_spec(data, radial_cells):
    from chemworld.physchem.battery_parameter_sets import battery_parameter_set

    validate_case(data)
    return battery_parameter_set(data["source_name"], version=data["source_version"]).build_spm(
        maximum_current_A=data["maximum_current_A"], radial_cells=radial_cells,
        surface_margin=data["surface_margin"], temperature_K=data["temperature_K"],
        electrolyte_concentration_mol_m3=data["ce_mol_m3"],
    )
