"""Thin original-native tracing and independently measured inventory balances."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_plating_cases import ATOL, PROFILE_RADII, RESPONSES, RTOL, F, R, native_spec
from scipy.integrate import simpson


def native_time(data, cells, *, old_spm=False):
    from chemworld.physchem.battery_models import (
        battery_control_current,
        battery_readout,
        battery_system,
    )
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, cells, old_spm=old_spm)
    state, segments = spec.initial_state(), []
    for stage, (duration, target, unit) in enumerate(data["waveform"]):
        try:
            program = ScalarControlProgram(((0.0, target),), unit)
            system = battery_system(spec, program)
            system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
            result = system.advance(
                state,
                duration,
                evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
            )
            states = list(result.trajectory)
            if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
                states.append(result.final_state)
            rows, raw_profiles = [], [[], []]
            for saved in states:
                current = battery_control_current(spec, saved, program)
                reading = battery_readout(spec, saved, current)
                side = getattr(reading, "stripping_current_A", 0.0)
                plated, dead = (
                    saved.amounts_mol.get("plating:Li", 0.0),
                    saved.amounts_mol.get("dead:Li", 0.0),
                )
                un = spec.ocv("negative", reading.negative_surface_fraction)
                phi = un + reading.negative_overpotential_V
                row = [
                    reading.voltage_V,
                    current,
                    saved.coordinates["discharged_charge_C"],
                    saved.coordinates["delivered_work_J"],
                    side,
                    plated,
                    dead,
                    phi,
                    plated / data["negative_electrode_volume_m3"],
                ]
                for i, e in enumerate(data["electrodes"]):
                    name, particle = e["name"], getattr(spec, e["name"])
                    surface = getattr(reading, name + "_surface_fraction")
                    average = getattr(reading, name + "_average_fraction")
                    amounts = np.array([saved.amounts_mol[k] for k in particle.keys(name)])
                    row.extend(
                        [
                            current - side if i == 0 else -current,
                            surface,
                            average,
                            float(sum(amounts)),
                            spec.ocv(name, surface),
                            getattr(reading, name + "_overpotential_V"),
                        ]
                    )
                    raw_profiles[i].append(
                        (
                            amounts
                            / (
                                particle.mesh.cell_volumes_m3
                                * particle.maximum_concentration_mol_m3
                            )
                        ).tolist()
                    )
                rows.append(row)
            nodes = [
                (getattr(spec, e["name"]).mesh.centers_m / e["radius_m"]).tolist()
                for e in data["electrodes"]
            ]
            profiles = [
                [
                    np.interp(PROFILE_RADII, nodes[i], profile).tolist()
                    for profile in raw_profiles[i]
                ]
                for i in range(2)
            ]
            segments.append(
                {
                    "times_s": [s.time_s for s in states],
                    "rows": rows,
                    "start_time_s": states[0].time_s,
                    "end_time_s": result.final_state.time_s,
                    "termination": result.stopping_event,
                    "outcome": result.outcome,
                    "profiles_theta": profiles,
                    "raw_r_over_R": nodes,
                    "raw_profiles_theta": raw_profiles,
                    "raw_amounts_mol": [dict(s.amounts_mol) for s in states],
                    "solver": dict(result.solver_diagnostic),
                    "conservation_residuals": dict(result.conservation_residuals),
                    "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
                }
            )
            state = result.final_state
            if result.outcome != "success":
                break
        except Exception as exc:
            exc.partial_history = {
                "segments": segments,
                "stage": stage,
                "target": target,
                "duration_s": duration,
                "current_time_s": state.time_s,
            }
            raise
    return {
        "implementation": "native original SPM without plating"
        if old_spm
        else "native original coupled SPM with shared-potential plating",
        "radial_cells": cells,
        "old_spm": old_spm,
        "segments": segments,
        "response_order": RESPONSES,
        "profile_r_over_R": PROFILE_RADII.tolist(),
        "F_C_mol": F,
        "R_J_mol_K": R,
        "tolerances": {"rtol": RTOL, "atol": ATOL},
    }


def refusal(data, name):
    from chemworld.physchem.battery_degradation import SEISpec
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import BatteryThermalSpec, battery_chemical_energy_j
    from chemworld.physchem.battery_soh import soh_from_battery

    spec = native_spec(data, 8)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s, spec.to_dict())
    altered = deepcopy(data)
    try:
        changes = {
            "negative-k": ("k_m_s", -1e-8),
            "alpha-domain": ("alpha", 1.2),
            "nonpositive-ce": ("ce_mol_m3", 0.0),
            "geometric-volume": ("negative_electrode_volume_m3", 5e-7),
            "negative-plated": ("initial_plated_mol", -1e-6),
            "negative-dead": ("initial_dead_mol", -1e-6),
            "partial-mode": ("mode", "partially reversible"),
        }
        if name in changes:
            key, value = changes[name]
            altered[key] = value
            native_spec(altered, 8)
        elif name == "SEI":
            replace(spec, sei=SEISpec())
        elif name == "thermal":
            replace(spec, thermal=BatteryThermalSpec())
        elif name == "EIS":
            battery_impedance(spec, (1.0,), state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        elif name == "chemical-energy":
            battery_chemical_energy_j(spec, state)
        else:
            raise RuntimeError("unknown fixed refusal")
        result = {"rejected": False}
    except Exception as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {
        **result,
        "input_state_unchanged": before
        == (dict(state.amounts_mol), dict(state.coordinates), state.time_s, spec.to_dict()),
    }


def history_metrics(trace, data):
    rows = np.asarray([row for s in trace["segments"] for row in s["rows"]])
    initial = [
        e["active_volume_m3"] * e["maximum_concentration_mol_m3"] * e["initial_fraction"]
        for e in data["electrodes"]
    ]
    metal0 = data["initial_plated_mol"] + data["initial_dead_mol"]
    f, r = trace["F_C_mol"], trace["R_J_mol_K"]
    charge = rows[:, 2]
    neg, pos, metal = rows[:, 12], rows[:, 18], rows[:, 5] + rows[:, 6]
    exponent = f * rows[:, 7] / (r * data["temperature_K"])
    area = 3 * data["electrodes"][0]["active_volume_m3"] / data["electrodes"][0]["radius_m"]
    independent_q = (
        -area * f * data["k_m_s"] * data["ce_mol_m3"] * np.exp(-data["alpha"] * exponent)
    )
    if data["mode"] == "reversible":
        independent_q += (
            area * f * data["k_m_s"] * rows[:, 8] * np.exp((1 - data["alpha"]) * exponent)
        )
    work_ratios, jumps, cc_error = [], [], []
    for index, s in enumerate(trace["segments"]):
        row, times = np.asarray(s["rows"]), np.asarray(s["times_s"])
        integral = simpson(row[:, 0] * row[:, 1], x=times)
        work_ratios.append(abs(row[-1, 3] - row[0, 3] - integral) / (0.02 + 0.02 * abs(integral)))
        if data["waveform"][index][2] == "A":
            target = data["waveform"][index][1]
            cc_error.append(float(np.max(abs(row[:, 2] - row[0, 2] - target * (times - times[0])))))
        if index:
            previous = np.asarray(trace["segments"][index - 1]["rows"][-1])
            jumps.append(float(np.max(abs(row[0, [5, 6, 12, 18]] - previous[[5, 6, 12, 18]]))))
    return {
        "total_Li_error_mol": float(np.max(abs(neg + pos + metal - sum(initial) - metal0))),
        "negative_Faraday_error_mol": float(
            np.max(abs(neg + metal - initial[0] - metal0 + charge / f))
        ),
        "positive_Faraday_error_mol": float(np.max(abs(pos - initial[1] - charge / f))),
        "algebraic_current_error_A": float(
            max(
                np.max(abs(rows[:, 9] + rows[:, 4] - rows[:, 1])),
                np.max(abs(rows[:, 15] + rows[:, 1])),
            )
        ),
        "independent_kinetic_limit_ratio": float(
            np.max(abs(rows[:, 4] - independent_q) / (1e-7 + 1e-7 * abs(independent_q)))
        ),
        "concentration_amount_error_mol": float(
            np.max(abs(rows[:, 5] - rows[:, 8] * data["negative_electrode_volume_m3"]))
        ),
        "minimum_metal_mol": float(np.min(rows[:, 5:7])),
        "work_quadrature_limit_ratio": max(work_ratios),
        "CC_charge_error_C": max(cc_error, default=0.0),
        "stage_inventory_jump_mol": max(jumps, default=0.0),
    }
