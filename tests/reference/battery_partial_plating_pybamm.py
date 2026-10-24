"""Actual release partial kinetics; constant-gamma boundary, no source RHS patch."""

import numpy as np
from battery_partial_plating_cases import ATOL, PROFILE_RADII, RESPONSES, ROOT_TOL, RTOL
from battery_plating_pybamm import build_reference as original_source_builder


def build_reference(data, cells, *, old_reversible=False):
    import pybamm

    simulation = original_source_builder(
        {**data, "mode": "reversible" if old_reversible else "partially reversible"}, cells
    )
    # The original Simulation is not built yet. Use its declared parameter interface.
    gamma = 0.0 if old_reversible else data["gamma_s_inv"]
    simulation.parameter_values.update(
        {"Dead lithium decay rate [s-1]": lambda thickness: pybamm.Scalar(gamma)},
        check_already_exists=False,
    )
    model = simulation.model
    dead_variable = next(v for v in model.rhs if "dead lithium concentration" in v.name)
    model.variables["qualification:deactivation_rate_mol_s"] = (
        model.rhs[dead_variable] * data["negative_electrode_volume_m3"]
    )
    return simulation


def reference_time(data, cells, *, old_reversible=False):
    import pybamm

    simulation, segments = build_reference(data, cells, old_reversible=old_reversible), []
    names = (
        "Voltage [V]",
        "Current [A]",
        "Discharge capacity [A.h]",
        "Discharge energy [W.h]",
        *("qualification:" + name for name in RESPONSES[4:]),
    )
    for stage, (duration, target, unit) in enumerate(data["waveform"]):
        try:
            solution = simulation.step(
                duration,
                save=False,
                t_eval=np.linspace(0.0, duration, data["samples_per_segment"]),
                inputs={
                    "plating voltage target" if unit == "V" else "Current function [A]": target
                },
            )
            times = np.asarray(solution.t)
            columns = []
            for name in names:
                values = np.asarray(solution[name](times))
                if values.size != times.size:
                    raise ValueError("source scalar output shape: " + name)
                columns.append(values.reshape(-1))
            rows = np.column_stack(columns)
            rows[:, 2:4] *= 3600
            profiles, nodes, raw_profiles = [], [], []
            for e in data["electrodes"]:
                variable = solution["qualification:" + e["name"] + ":profile"]
                radial_nodes = np.asarray(variable.mesh.nodes)
                values = np.asarray(variable.entries) / e["maximum_concentration_mol_m3"]
                if values.shape != (cells, len(times)):
                    raise ValueError("source original radial node shape")
                profiles.append(
                    np.asarray(
                        [
                            np.interp(PROFILE_RADII * e["radius_m"], radial_nodes, values[:, i])
                            for i in range(len(times))
                        ]
                    ).tolist()
                )
                nodes.append((radial_nodes / e["radius_m"]).tolist())
                raw_profiles.append(values.T.tolist())
            segments.append(
                {
                    "times_s": times.tolist(),
                    "rows": rows.tolist(),
                    "deactivation_rate_mol_s": np.asarray(
                        solution["qualification:deactivation_rate_mol_s"](times)
                    )
                    .reshape(-1)
                    .tolist(),
                    "sei_thickness_m": np.asarray(
                        solution["X-averaged negative total SEI thickness [m]"](times)
                    )
                    .reshape(-1)
                    .tolist(),
                    "start_time_s": float(times[0]),
                    "end_time_s": float(times[-1]),
                    "termination": solution.termination,
                    "profiles_theta": profiles,
                    "raw_r_over_R": nodes,
                    "raw_profiles_theta": raw_profiles,
                }
            )
            if solution.termination != "final time":
                break
        except Exception as exc:
            exc.partial_history = {
                "segments": segments,
                "stage": stage,
                "target": target,
                "duration_s": duration,
                "last_completed_time_s": segments[-1]["end_time_s"] if segments else 0.0,
            }
            raise
    return {
        "implementation": "actual PyBaMM25.4.2 SPM partial constant decay"
        if not old_reversible
        else "actual PyBaMM25.4.2 original reversible SPM plating",
        "radial_cells": cells,
        "old_reversible": old_reversible,
        "segments": segments,
        "response_order": RESPONSES,
        "profile_r_over_R": PROFILE_RADII.tolist(),
        "F_C_mol": float(pybamm.constants.F.value),
        "R_J_mol_K": float(pybamm.constants.R.value),
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ROOT_TOL},
        "source_CV_boundary": (
            "authored algebraic limited source via public callable; "
            "not native limited-CV capability"
        ),
    }
