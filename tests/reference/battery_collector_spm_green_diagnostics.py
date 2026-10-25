"""Independent electrical checks of every saved collector-SPM sample.

Descriptive only: these do not supply new trajectory qualification thresholds.
The production collector, field reader and solver are deliberately not imported.
"""

from __future__ import annotations

import numpy as np
from battery_collector_spm_cases import GRIDS, LAYOUTS
from battery_collector_spm_diagnostics import recover_trace
from battery_collector_spm_green import (
    StaticClosureError,
    integrate_collectors,
    solve_static_collector,
    static_inputs_from_case,
)


def _array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: expected finite shape {shape}, got {result.shape}")
    return result


def _maximum(values):
    if not values:
        return {"available": False, "reason": "no successful samples"}
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite derived electrical diagnostic")
    return {"available": True, "samples": len(values), "max_abs": float(np.max(abs(array)))}


def diagnose_segment(segment, kind, data, grid, layout, segment_index):
    """Compare exact archived observations, retaining each failed sample separately."""
    times = np.asarray(segment["times_s"], dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("a finite nonempty saved time axis is required")
    nt, nz, nr = len(times), grid["z"], grid["r"]
    values = _array(segment["values"], (nt, 4), "values")
    profiles = segment["raw_profiles"]
    qn = _array(profiles["negative"]["reaction_current_A"], (nt, nz), "negative q")
    qp = _array(profiles["positive"]["reaction_current_A"], (nt, nz), "positive q")
    if kind == "native":
        potentials = {
            name: _array(profiles[name]["collector_potential_V"], (nt, nz), name + " potential")
            for name in ("negative", "positive")
        }
        edges = np.linspace(0, data["height_m"], nz + 1)
        theta = {
            name: _array(profiles[name]["theta"], (nt, nz, nr), name + " theta")
            for name in ("negative", "positive")
        }
        local = _array(profiles["negative"]["local_voltage_V"], (nt, nz), "local V")
    else:
        fields = segment["source_fields"]
        potentials = {
            name: _array(
                fields[name.capitalize() + " current collector potential [V]"],
                (nt, nz),
                name + " potential",
            )
            for name in ("negative", "positive")
        }
        edges = _array(profiles["negative"]["z_edges_m"], (nz + 1,), "source z edges")
        local = _array(fields["Local voltage [V]"], (nt, nz), "source local V")
        source_bv = {}
        for name in ("negative", "positive"):
            source_bv[name] = {
                field: _array(profiles[name][field], (nt, nz), name + " " + field)
                for field in ("kinetic_overpotential_V", "open_circuit_potential_V")
            }
    boundary = _array(segment["source_boundary_voltage_V"], (nt,), "boundary V")
    kn = kp = data["sigma_cc_S_m"] * data["collector_thickness_m"] * data["width_m"]
    result = {
        "saved_samples": nt,
        "linear_samples": 0,
        "static_samples": 0,
        "source_identity_samples": 0,
        "failures": [],
        "metrics": {},
        "static_definition": (
            "independent nonlinear BV closure at saved outer-shell state, not a time prediction"
            if kind == "native"
            else "raw source OCP-plus-overpotential voltage identity only; "
            "source surfaces not replaced by native shell formula"
        ),
        "voltage_definition": (
            "recorded physical terminal plus declared series drop"
            if kind == "native"
            else "physical tab inferred from last FV centre and port-current half-cell drop; "
            "the anchoring centre is not independently predicted"
        ),
    }
    metrics = {}

    def collect(name, value):
        metrics.setdefault(name, []).append(float(np.max(abs(np.asarray(value)))))

    for k, time in enumerate(times):
        try:
            internal_voltage = (
                values[k, 0] + values[k, 1] * data["series_resistance_ohm"]
                if kind == "native"
                else potentials["positive"][k, -1] - values[k, 1] * np.diff(edges)[-1] / (2 * kp)
            )
            reference = integrate_collectors(
                qn[k],
                z_edges_m=edges,
                negative_conductance_S_m=kn,
                positive_conductance_S_m=kp,
                negative_tab=layout["negative_tab"],
                physical_tab_voltage_V=internal_voltage,
            )
            collect("negative_plus_positive_q_A", qn[k] + qp[k])
            collect("sum_negative_q_minus_port_I_A", np.sum(qn[k]) - values[k, 1])
            for name in ("negative", "positive"):
                field = reference[name]
                collect(
                    name + "_FV_potential_residual_V", potentials[name][k] - field["fv_centre_V"]
                )
                collect(
                    name + "_continuous_centre_minus_FV_V",
                    field["continuous_centre_V"] - field["fv_centre_V"],
                )
                collect(
                    name + "_continuous_mean_minus_FV_V",
                    field["continuous_mean_V"] - field["fv_centre_V"],
                )
            collect(
                "local_V_minus_recorded_collector_difference_V",
                local[k] - potentials["positive"][k] + potentials["negative"][k],
            )
            collect(
                "source_boundary_reconstruction_residual_V",
                reference["source_boundary_voltage_V"] - boundary[k],
            )
            collect("inferred_physical_minus_raw_boundary_V", internal_voltage - boundary[k])
            if kind == "source":
                collect(
                    "reaction_sum_minus_port_half_cell_voltage_V",
                    (np.sum(qn[k]) - values[k, 1]) * np.diff(edges)[-1] / (2 * kp),
                )
            collect(
                "FV_electrical_power_identity_residual_W",
                np.dot(qn[k], potentials["positive"][k] - potentials["negative"][k])
                - np.sum(qn[k]) * internal_voltage
                - reference["fv_joule_W"],
            )
            collect(
                "continuous_minus_FV_joule_W",
                reference["continuous_joule_W"] - reference["fv_joule_W"],
            )
            result["linear_samples"] += 1
        except Exception as error:
            result["failures"].append(
                {
                    "sample": k,
                    "time_s": float(time),
                    "stage": "linear",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        try:
            if kind == "source":
                equilibrium = sum(
                    sign
                    * (
                        source_bv[name]["open_circuit_potential_V"][k]
                        + source_bv[name]["kinetic_overpotential_V"][k]
                    )
                    for name, sign in (("negative", -1), ("positive", 1))
                )
                collect("source_raw_OCP_plus_eta_identity_residual_V", local[k] - equilibrium)
                result["source_identity_samples"] += 1
                continue
            kwargs = static_inputs_from_case(
                data,
                grid,
                layout,
                negative_outer_fraction=theta["negative"][k, :, -1],
                positive_outer_fraction=theta["positive"][k, :, -1],
            )
            _duration, target, mode = data["waveform"][segment_index]
            answer = solve_static_collector(**kwargs, control_mode=mode, target=target)
            collect("static_q_minus_recorded_q_A", answer["through_current_A"] - qn[k])
            collect("static_terminal_minus_recorded_V", answer["voltage_V"] - values[k, 0])
            collect("static_port_I_minus_recorded_A", answer["current_A"] - values[k, 1])
            collect("static_BV_solver_residual_V", answer["local_residual_V"])
            for name in ("negative", "positive"):
                surface = _array(profiles[name]["surface_fraction"], (nt, nz), name + " surface")
                collect(
                    name + "_static_surface_minus_recorded_fraction",
                    answer[name]["surface_fraction"] - surface[k],
                )
            result["static_samples"] += 1
        except Exception as error:
            failure = {
                "sample": k,
                "time_s": float(time),
                "stage": "static",
                "type": type(error).__name__,
                "message": str(error),
            }
            if isinstance(error, StaticClosureError):
                failure["solver_diagnostic"] = error.diagnostic
            result["failures"].append(failure)
    result["metrics"] = {name: _maximum(values) for name, values in metrics.items()}
    return result


def diagnose_electrical(report, progress=None):
    """Analyze all recoverable histories without changing the raw report."""
    result = {
        "analysis": "descriptive independent electrical reconstruction; no new qualification score",
        "selection": "all recorded samples of all recoverable history segments",
        "units": [],
        "failures": [],
    }
    for unit in report["units"]:
        data = unit["inputs"]
        item = {"id": unit["id"], "routes": {}}
        result["units"].append(item)
        for name, raw in unit["routes"].items():
            if not name.startswith("history_"):
                continue
            layout_name, grid_name = name.removeprefix("history_").rsplit("_", 1)
            entry = item["routes"][name] = {}
            for kind in ("native", "source"):
                history, origin = recover_trace(raw, kind)
                output = entry[kind] = {"origin": origin, "segments": []}
                if not history or not history.get("segments"):
                    output["unavailable"] = (
                        "no extracted saved segment; see original failure/initialization"
                    )
                    continue
                for i, segment in enumerate(history["segments"]):
                    try:
                        output["segments"].append(
                            diagnose_segment(
                                segment,
                                kind,
                                data,
                                history.get("grid", GRIDS[grid_name]),
                                history.get("layout", LAYOUTS[layout_name]),
                                i,
                            )
                        )
                    except Exception as error:
                        failure = {
                            "unit": unit["id"],
                            "route": name,
                            "kind": kind,
                            "segment": i,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        output["segments"].append({"unavailable": failure})
                        result["failures"].append(failure)
            if progress:
                progress(unit["id"], name)
    return result
