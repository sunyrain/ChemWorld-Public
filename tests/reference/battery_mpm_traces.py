"""Time traces of the actual native MPM and independent released models."""

import numpy as np
from battery_mpm_cases import WAVEFORM, independent_area_pdfs, ocp_functions
from battery_spme_pybamm import build_spme_reference

from chemworld.physchem.battery_models import advance_battery, battery_readout
from chemworld.physchem.battery_mpm import mpm_fields
from chemworld.physchem.reactor_shared import ScalarControlProgram

RESPONSE_NAMES = (
    "voltage_V",
    "area_surface_n",
    "area_surface_p",
    "volume_theta_n",
    "volume_theta_p",
    "charge_C",
    "current_A",
)
RESPONSE_BOUNDS = np.array([1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 1e-4, 1e-3])


def protocol(index):
    if index == 6:
        return ((120.0, 3.75, "V"),)
    if index == 7:
        return ((1000.0, 0.8, "A"),)
    return tuple((duration, current, "A") for duration, current in WAVEFORM)


def native_trace(spec, index, progress):
    state, segments = spec.initial_state(), []
    for number, (duration, target, unit) in enumerate(protocol(index), 1):
        progress(f"native segment {number}/{len(protocol(index))}")
        program = ScalarControlProgram(((0.0, target),), unit)
        times = np.linspace(0, duration, 101 if index == 7 else 21)
        result = advance_battery(spec, state, program, duration, evaluation_times_s=times)
        rows, maximum_bv, current_error, allocations, local = [], 0.0, 0.0, [], {}
        for name in ("negative", "positive"):
            bins = [b for b in spec.particle_bins if b[0] == name]
            local[name] = {
                "radii_m": [p.radius_m for _, _, p, _, _ in bins],
                "surface": [],
                "j_A_m2": [],
                "current_A": [],
            }
        for snapshot in result.trajectory:
            fields = mpm_fields(spec, snapshot, program=program)
            reading = battery_readout(spec, snapshot, fields["current_A"])
            rows.append(
                [
                    reading.voltage_V,
                    reading.negative_surface_fraction,
                    reading.positive_surface_fraction,
                    reading.negative_average_fraction,
                    reading.positive_average_fraction,
                    snapshot.coordinates["discharged_charge_C"],
                    fields["current_A"],
                ]
            )
            maximum_bv = max(maximum_bv, float(np.max(np.abs(fields["bv_residual_V"]))))
            first, deviation = 0, 0.0
            for name, count, sign in zip(
                ("negative", "positive"), spec.counts, (1, -1), strict=True
            ):
                selection = slice(first, first + count)
                q = fields["reaction_current_A"][selection]
                local[name]["surface"].append(fields["surface_fraction"][selection].tolist())
                local[name]["current_A"].append(q.tolist())
                local[name]["j_A_m2"].append((q / fields["area_m2"][selection]).tolist())
                capacities = np.array(
                    [p.capacity_mol for n, _, p, _, _ in spec.particle_bins if n == name]
                )
                deviation = max(
                    deviation,
                    float(
                        np.max(
                            np.abs(q - sign * fields["current_A"] * capacities / capacities.sum())
                        )
                    ),
                )
                current_error = max(current_error, abs(float(sum(q)) - sign * fields["current_A"]))
                first += count
            allocations.append(deviation)
        conservation = dict(result.conservation_residuals)
        conserved = all(
            abs(value) <= (1e-5 if key.endswith("_C") else 1e-10)
            for key, value in conservation.items()
        )
        segments.append(
            {
                "times_s": [s.time_s for s in result.trajectory],
                "rows": rows,
                "local": local,
                "target": target,
                "unit": unit,
                "outcome": result.outcome,
                "event": result.stopping_event,
                "end_time_s": result.final_state.time_s,
                "actual_duration_s": result.actual_duration_s,
                "requested_duration_s": duration,
                "conservation_residuals": conservation,
                "maximum_bv_residual_V": maximum_bv,
                "maximum_current_sum_error_A": current_error,
                "allocation_deviation_from_volume_A": allocations,
                "invariants_passed": bool(
                    conserved and maximum_bv <= 1e-8 and current_error <= 1e-9
                ),
                "final_min_surface": float(
                    np.min(
                        mpm_fields(spec, result.final_state, program=program)["surface_fraction"]
                    )
                ),
                "final_max_surface": float(
                    np.max(
                        mpm_fields(spec, result.final_state, program=program)["surface_fraction"]
                    )
                ),
            }
        )
        state = result.final_state
        if result.outcome != "success":
            break
    return {"spec": spec.to_dict(), "segments": segments}


def build_reference(spec, index, size_cells, radial_cells):
    return build_spme_reference(
        spec,
        family="SPM" if index == 1 else "MPM",
        cells=5,
        radial_cells=radial_cells,
        size_cells=size_cells,
        radial_mesh="exponential-right",
        ocp_functions=ocp_functions(index, compensate_barrier=index == 1),
        size_pdf_functions=None if index == 1 else independent_area_pdfs(spec),
        voltage_control=index == 6,
    )


def reference_trace(spec, index, size_cells, radial_cells, progress):
    simulation = build_reference(spec, index, size_cells, radial_cells)
    segments = []
    for number, (duration, target, unit) in enumerate(protocol(index), 1):
        progress(f"reference segment {number}/{len(protocol(index))}")
        begin = 0.0 if simulation.solution is None else float(simulation.solution.t[-1])
        clock = np.linspace(0, duration, 101 if index == 7 else 21)
        inputs = {"native voltage target" if unit == "V" else "Current function [A]": target}
        solution = simulation.step(duration, t_eval=clock, inputs=inputs)
        end = float(solution.t[-1])
        times = begin + clock
        times = times[times < end - 1e-9]
        times = np.r_[times, end]
        if number > 1:
            times[0] += 1e-8
        surface_names = [
            f"X-averaged {name} particle surface concentration"
            if index == 1
            else f"native:{name}:area-surface"
            for name in ("negative", "positive")
        ]
        names = [
            "Voltage [V]",
            *surface_names,
            "Average negative particle concentration",
            "Average positive particle concentration",
            "Discharge capacity [A.h]",
            "Current [A]",
        ]
        rows = np.column_stack([np.asarray(solution[name](times)).reshape(-1) for name in names])
        rows[:, 5] *= 3600
        local = {}
        for name in ("negative", "positive"):
            if index == 1:
                radii = np.array([getattr(spec, name).radius_m])
                surface_name = f"X-averaged {name} particle surface concentration"
                current_name = f"X-averaged {name} electrode interfacial current density [A.m-2]"
                surface = np.asarray(solution[surface_name](times)).reshape(-1, 1)
                j = np.asarray(solution[current_name](times)).reshape(-1, 1)
            else:
                radii = simulation.mesh[name + " particle size"].nodes
                surface_name = f"X-averaged {name} particle surface concentration distribution"
                current_name = (
                    f"X-averaged {name} electrode interfacial current density distribution [A.m-2]"
                )
                surface = (
                    np.asarray(solution[surface_name](times, R=radii)).reshape(len(radii), -1).T
                )
                j = np.asarray(solution[current_name](times, R=radii)).reshape(len(radii), -1).T
            local[name] = {
                "radii_m": radii.tolist(),
                "surface": surface.tolist(),
                "j_A_m2": j.tolist(),
            }
        segments.append(
            {
                "times_s": times.tolist(),
                "rows": rows.tolist(),
                "local": local,
                "target": target,
                "unit": unit,
                "termination": solution.termination,
                "end_time_s": end,
                "actual_duration_s": end - begin,
                "requested_duration_s": duration,
                "final_min_surface": min(min(local[n]["surface"][-1]) for n in local),
                "final_max_surface": max(max(local[n]["surface"][-1]) for n in local),
            }
        )
        if solution.termination != "final time":
            break
    return {
        "grid": {
            "size_cells": size_cells,
            "radial_cells": radial_cells,
            "radial_mesh": "exponential-right",
            "stretch": 2.3,
        },
        "segments": segments,
    }
