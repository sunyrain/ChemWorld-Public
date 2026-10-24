"""Native trajectories and independent conservation measurements."""

from dataclasses import replace

import numpy as np
from battery_half_cell_cases import ATOL, RESPONSES, RTOL, F, initial_metal, native_spec
from battery_half_cell_pybamm import PROFILE_RADII


def native_time(data, cells):
    from chemworld.physchem.battery_half_cell import (
        half_cell_control_current,
        half_cell_readout,
        half_cell_system,
    )
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, cells)
    state, segments = spec.initial_state(), []
    for duration, target, unit in data["waveform"]:
        program = ScalarControlProgram(((0., target),), unit)
        system = half_cell_system(spec, program)
        system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
        result = system.advance(state, duration,
            evaluation_times_s=np.linspace(0., duration, data["samples_per_segment"]))
        states = list(result.trajectory)
        if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
            states.append(result.final_state)
        rows = []
        for saved in states:
            current = half_cell_control_current(spec, saved, program)
            read = half_cell_readout(spec, saved, current)
            rows.append([read.voltage_V, current, read.working_surface_fraction,
                read.working_average_fraction, read.metal_amount_mol,
                saved.coordinates["discharged_charge_C"], read.metal_overpotential_V,
                read.working_overpotential_V, read.metal_ohmic_V])
        particle = spec.positive
        profile = np.array([result.final_state.amounts_mol[k]
            for k in particle.keys("positive")]) / (
                particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3)
        profile = np.interp(PROFILE_RADII, particle.mesh.centers_m / particle.radius_m, profile)
        segments.append({"times_s": [s.time_s for s in states], "rows": rows,
            "end_time_s": result.final_state.time_s, "termination": result.stopping_event,
            "outcome": result.outcome, "profile_theta": profile.tolist(),
            "conservation_residuals": dict(result.conservation_residuals),
            "solver": dict(result.solver_diagnostic)})
        state = result.final_state
        if result.outcome != "success":
            break
    return {"implementation": "native fixed-geometry finite-metal SPM",
        "response_order": RESPONSES, "radial_cells": cells, "segments": segments,
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "profile_r_over_R": PROFILE_RADII.tolist()}


def refusal(data, name):
    from chemworld.physchem.battery_half_cell import LithiumMetalCounterElectrodeSpec
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import battery_chemical_energy_j
    from chemworld.physchem.battery_soh import soh_from_battery

    spec = native_spec(data, 8)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    try:
        if name == "zero-area":
            native_spec({**data, "area_m2": 0.}, 8)
        elif name == "negative-conductivity":
            native_spec({**data, "conductivity_S_m": -.1}, 8)
        elif name == "temperature-mismatch":
            replace(spec, temperature_K=300.)
        elif name == "concentration-mismatch":
            replace(spec, electrolyte_concentration_mol_m3=1200.)
        elif name == "missing-source-row":
            LithiumMetalCounterElectrodeSpec(spec.metal.parameters, "missing")
        elif name == "EIS":
            battery_impedance(spec, (1.,), state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        elif name == "energy":
            battery_chemical_energy_j(spec, state)
        else:
            raise RuntimeError("unknown fixed refusal")
        answer = {"rejected": False}
    except Exception as exc:
        answer = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {**answer, "input_state_unchanged": before == (
        dict(state.amounts_mol), dict(state.coordinates), state.time_s)}


def history_metrics(trace, data):
    from battery_half_cell_cases import working_capacity

    capacity = working_capacity(data)
    all_rows = np.asarray([row for s in trace["segments"] for row in s["rows"]])
    metal, average, charge = all_rows[:, 4], all_rows[:, 3], all_rows[:, 5]
    metal_error = metal - initial_metal(data) + charge / F
    work_error = capacity * (average - data["initial_positive_fraction"]) - charge / F
    cc_errors, rest_m, rest_q, jump_m, jump_a, jump_q = [], [], [], [], [], []
    previous = None
    for segment, (_, target, unit) in zip(trace["segments"], data["waveform"], strict=False):
        rows, times = np.asarray(segment["rows"]), np.asarray(segment["times_s"])
        if unit == "A":
            cc_errors.extend(abs(rows[:, 5] - rows[0, 5] - target * (times - times[0])))
            if target == 0:
                rest_m.extend(abs(rows[:, 4] - rows[0, 4]))
                rest_q.extend(abs(rows[:, 5] - rows[0, 5]))
        if previous is not None:
            jump_m.append(abs(rows[0, 4] - previous[4]))
            jump_a.append(abs(rows[0, 3] - previous[3]))
            jump_q.append(abs(rows[0, 5] - previous[5]))
        previous = rows[-1]
    return {"lithium_balance_mol": float(np.max(abs(metal_error + work_error))),
        "faradaic_error_C": float(F * max(np.max(abs(metal_error)), np.max(abs(work_error)))),
        "metal_mapping_error_mol": float(np.max(abs(metal_error))),
        "CC_charge_error_C": float(max(cc_errors, default=0.)),
        "rest_metal_change_mol": float(max(rest_m, default=0.)),
        "rest_charge_change_C": float(max(rest_q, default=0.)),
        "stage_metal_jump_mol": float(max(jump_m, default=0.)),
        "stage_average_jump": float(max(jump_a, default=0.)),
        "stage_charge_jump_C": float(max(jump_q, default=0.))}
