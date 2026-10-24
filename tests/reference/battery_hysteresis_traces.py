"""Native measurements, independent analytical history checks and refusal probes."""

from dataclasses import replace

import numpy as np
from battery_hysteresis_cases import (
    FREQUENCIES,
    analytic_h,
    native_case,
    response_names,
    sweep_currents,
)

from chemworld.physchem.battery_models import battery_readout, battery_system
from chemworld.physchem.battery_mpm import mpm_fields
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F
from chemworld.physchem.reactor_shared import ScalarControlProgram


def native_row(spec, data, state, current):
    read = battery_readout(spec, state, current)
    row = [read.voltage_V, current, read.negative_surface_fraction, read.positive_surface_fraction,
           read.negative_average_fraction, read.positive_average_fraction,
           state.coordinates["discharged_charge_C"]]
    row += [state.coordinates[f"battery:{name}:hysteresis"] for name in ("negative", "positive")
            if data[name+"_mode"] == "Axen"]
    local, bv, current_error = {}, 0., 0.
    if data["family"] == "MPM":
        fields = mpm_fields(spec, state, current_A=current)
        bv = float(np.max(abs(fields["bv_residual_V"])))
        for name, sign in (("negative", 1), ("positive", -1)):
            indices = [i for i, b in enumerate(spec.particle_bins) if b[0] == name]
            selected = [spec.particle_bins[i] for i in indices]
            areas = fields["area_m2"][indices]
            ocps = [spec.operational_ocp(name, fields["surface_fraction"][i], current, state)
                    for i in indices]
            row.append(float(np.average(ocps, weights=areas)))
            currents = fields["reaction_current_A"][indices]
            current_error = max(current_error, abs(sum(currents)-sign*current))
            local[name] = {"radii_m": [b[2].radius_m for b in selected],
                "surface": fields["surface_fraction"][indices].tolist(),
                "j_A_m2": (currents/areas).tolist(), "current_A": currents.tolist()}
    else:
        row += [spec.operational_ocp(name, theta, current, state) for name, theta in (
            ("negative", read.negative_surface_fraction),
            ("positive", read.positive_surface_fraction))]
    return row, local, bv, current_error


def native_time(data, radial_cells, size_cells, tight):
    from chemworld.physchem.battery_models import battery_control_current

    spec = native_case(data, radial_cells, size_cells)
    state, segments = spec.initial_state(), []
    for duration, target, unit in data["waveform"]:
        program = ScalarControlProgram(((0., target),), unit)
        system = battery_system(spec, program)
        rtol, atol = (1e-10, 1e-12) if tight else (1e-8, 1e-10)
        system = replace(system, policy=replace(system.policy, rtol=rtol, atol=atol))
        result = system.advance(state, duration, evaluation_times_s=np.linspace(
            0., duration, data["samples_per_segment"]))
        rows, local, max_bv, current_error = [], {}, 0., 0.
        for saved in result.trajectory:
            current = battery_control_current(spec, saved, program)
            row, single, bv, cerr = native_row(spec, data, saved, current)
            rows.append(row)
            max_bv, current_error = max(max_bv, bv), max(current_error, cerr)
            for name, values in single.items():
                if name not in local:
                    local[name] = {"radii_m": values["radii_m"], "surface": [], "j_A_m2": [],
                                   "current_A": []}
                for key in ("surface", "j_A_m2", "current_A"):
                    local[name][key].append(values[key])
        segments.append({"times_s": [s.time_s for s in result.trajectory], "rows": rows,
            "local": local, "termination": result.stopping_event, "outcome": result.outcome,
            "end_time_s": result.final_state.time_s,
            "conservation_residuals": dict(result.conservation_residuals),
            "maximum_bv_residual_V": max_bv, "maximum_current_sum_error_A": current_error,
            "solver": dict(result.solver_diagnostic)})
        state = result.final_state
        if result.outcome != "success":
            break
    return {"implementation": "native", "response_order": response_names(data),
            "radial_cells": radial_cells, "size_cells": size_cells, "tight": tight,
            "segments": segments}


def history_metrics(trace, data):
    names, parts = response_names(data), trace["segments"]
    all_rows = np.array([row for part in parts for row in part["rows"]])
    capacity = F*data["maximum_concentration_mol_m3"]*data["active_volume_m3"]
    n = capacity*(all_rows[:, 4]-data["initial_negative_fraction"])+all_rows[:, 6]
    p = capacity*(all_rows[:, 5]-data["initial_positive_fraction"])-all_rows[:, 6]
    errors, rest, jumps, maximum_h = [], [], [], []
    start, previous = 0., None
    for part, (_, current, unit) in zip(parts, data["waveform"], strict=False):
        times, rows = np.array(part["times_s"]), np.array(part["rows"])
        for electrode in ("negative", "positive"):
            if data[electrode+"_mode"] != "Axen":
                continue
            column = names.index(electrode+"_h")
            values = rows[:, column]
            maximum_h.extend(abs(values))
            if previous is not None:
                jumps.append(abs(values[0]-previous[column]))
            if unit == "A":
                expected = np.array([analytic_h(data, electrode, values[0], current, t-start, F)
                                     for t in times])
                errors.extend(abs(values-expected))
                if current == 0:
                    rest.extend(abs(values-values[0]))
        start, previous = part["end_time_s"], rows[-1]
    lithium = ((all_rows[:, 4]-data["initial_negative_fraction"])
               +(all_rows[:, 5]-data["initial_positive_fraction"]))*capacity/F
    return {"maximum_faradaic_residual_C": float(max(np.max(abs(n)), np.max(abs(p)))),
            "maximum_total_lithium_error_mol": float(np.max(abs(lithium))),
            "maximum_analytic_h_error": float(max(errors, default=0.)),
            "maximum_rest_h_change": float(max(rest, default=0.)),
            "maximum_h_switch_jump": float(max(jumps, default=0.)),
            "maximum_h_magnitude": float(max(maximum_h, default=0.))}


def native_static(data, radial_cells=64):
    spec = native_case(data, radial_cells)
    state = spec.initial_state()
    return {"response_order": response_names(data), "points": [
        {"current_A": current, "row": native_row(spec, data, state, current)[0]}
        for current in sweep_currents()]}


def native_frequency(data, radial_cells):
    from chemworld.physchem.battery_impedance import battery_impedance

    spec = native_case(data, radial_cells)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    result = battery_impedance(spec, FREQUENCIES, state=state)
    points = [{"frequency_Hz": p.frequency_Hz, "success": p.success,
               "response_real": [z.real] if z is not None else None,
               "response_imag": [z.imag] if z is not None else None,
               "relative_linear_residual": p.relative_linear_residual,
               "failure_reason": p.failure_reason}
              for p, z in zip(result.response.points, result.impedance_ohm, strict=True)]
    return {"success": result.success, "points": points,
            "input_state_unchanged": before == (
                dict(state.amounts_mol), dict(state.coordinates), state.time_s)}


def refusal(data, *, nonsmooth=False):
    from chemworld.physchem.battery_degradation import SEISpec
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import BatteryThermalSpec
    from chemworld.physchem.battery_soh import soh_from_battery

    spec = native_case(data, 32)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    try:
        if nonsmooth:
            battery_impedance(spec, FREQUENCIES, state=state)
        elif data["name"] == "initial-h-outside":
            native_case({**data, "negative_initial_h": 1.01}, 32)
        elif data["name"] == "negative-decay":
            native_case({**data, "lithiation_decay": -1.}, 32)
        elif data["name"] == "zero-collector-area":
            native_case({**data, "negative_mode": "current sigmoid",
                         "effective_collector_area_m2": 0.}, 32)
        elif data["name"] == "thermal-combination":
            replace(spec, thermal=BatteryThermalSpec())
        elif data["name"] == "SEI-combination":
            replace(spec, sei=SEISpec())
        elif data["name"] == "equilibrium-SOH":
            soh_from_battery(spec, state)
        else:
            raise RuntimeError("unknown fixed rejection probe")
        result = {"rejected": False}
    except Exception as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {**result, "input_state_unchanged": before == (
        dict(state.amounts_mol), dict(state.coordinates), state.time_s)}
