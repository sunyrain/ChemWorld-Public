"""Original native state tracing and read-only size/radial comparison views."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_mpm_plating_cases import (
    ATOL,
    RADIAL_PROBES,
    RESPONSES,
    RTOL,
    F,
    R,
    comparison_radii,
    native_spec,
)
from scipy.integrate import simpson


def local_views(local, data, name):
    """Interpolated comparison view only; original arrays remain in local."""
    sizes, radial = np.asarray(local["radii_m"]), np.asarray(local["r_over_R"])
    target = comparison_radii(data, name)
    raw = np.asarray(local["profiles_theta"])
    radial_view = np.array([[np.interp(RADIAL_PROBES, radial, p) for p in stage] for stage in raw])
    profiles = np.array(
        [
            [np.interp(target, sizes, t[:, j]) for j in range(len(RADIAL_PROBES))]
            for t in radial_view
        ]
    ).transpose(0, 2, 1)
    return {
        "comparison_profiles_theta": profiles.tolist(),
        "comparison_surface_fraction": [
            np.interp(target, sizes, row).tolist() for row in local["surface_fraction"]
        ],
        "comparison_j_A_m2": [np.interp(target, sizes, row).tolist() for row in local["j_A_m2"]],
    }


def native_time(data, size_nodes, radial_cells, *, old_mpm=False, single_spm=False):
    from chemworld.physchem.battery_models import (
        battery_control_current,
        battery_readout,
        battery_system,
    )
    from chemworld.physchem.battery_mpm import mpm_fields
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, size_nodes, radial_cells, old_mpm=old_mpm, single_spm=single_spm)
    state, segments = spec.initial_state(), []
    bins = (
        tuple(
            (name, 0, getattr(spec, name), getattr(spec, name).keys(name), name)
            for name in ("negative", "positive")
        )
        if single_spm
        else spec.particle_bins
    )
    mapping = {}
    for name in ("negative", "positive"):
        parts = [p for n, _, p, _, _ in bins if n == name]
        volumes = np.array([p.active_volume_m3 for p in parts])
        radii = np.array([p.radius_m for p in parts])
        areas = 3 * volumes / radii
        mapping[name] = {
            "radii_m": radii.tolist(),
            "volume_weights": (volumes / volumes.sum()).tolist(),
            "area_weights": (areas / areas.sum()).tolist(),
            "bin_area_m2": areas.tolist(),
            "active_area_m2": float(areas.sum()),
            "area_mean_radius_m": float(np.dot(areas / areas.sum(), radii)),
            "input_basis": "single bin" if data["single_bin"] else "area lognormal Gauss-Legendre",
        }
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
            local = {
                n: {
                    "radii_m": mapping[n]["radii_m"],
                    "r_over_R": (
                        next(p for name, _, p, _, _ in bins if name == n).mesh.centers_m
                        / mapping[n]["radii_m"][0]
                    ).tolist(),
                    **{
                        key: []
                        for key in (
                            "surface_fraction",
                            "average_fraction",
                            "current_A",
                            "j_A_m2",
                            "ocp_V",
                            "overpotential_V",
                            "profiles_theta",
                        )
                    },
                }
                for n in ("negative", "positive")
            }
            rows = []
            for saved in states:
                current = battery_control_current(spec, saved, program)
                reading = battery_readout(spec, saved, current)
                if single_spm:
                    side = reading.stripping_current_A
                    sf = np.array(
                        [reading.negative_surface_fraction, reading.positive_surface_fraction]
                    )
                    av = np.array(
                        [reading.negative_average_fraction, reading.positive_average_fraction]
                    )
                    eta = np.array(
                        [reading.negative_overpotential_V, reading.positive_overpotential_V]
                    )
                    q = np.array([current - side, -current])
                else:
                    f = mpm_fields(spec, saved, current_A=current)
                    side = f.get("stripping_current_A", 0.0)
                    sf, av, eta, q = (
                        np.asarray(f[k])
                        for k in (
                            "surface_fraction",
                            "average_fraction",
                            "kinetic_overpotential_V",
                            "reaction_current_A",
                        )
                    )
                pl = saved.amounts_mol.get("plating:Li", 0.0)
                dead = saved.amounts_mol.get("dead:Li", 0.0)
                ocp = np.array(
                    [spec.ocv(n, float(theta)) for (n, *_), theta in zip(bins, sf, strict=True)]
                )
                phi = float(ocp[0] + eta[0])
                row = [
                    reading.voltage_V,
                    current,
                    saved.coordinates["discharged_charge_C"],
                    saved.coordinates["delivered_work_J"],
                    side,
                    pl,
                    dead,
                    phi,
                    pl / data["negative_electrode_volume_m3"],
                ]
                for name in ("negative", "positive"):
                    idx = np.array([i for i, b in enumerate(bins) if b[0] == name])
                    m = mapping[name]
                    aw, vw = np.array(m["area_weights"]), np.array(m["volume_weights"])
                    amounts = sum(saved.amounts_mol[k] for i in idx for k in bins[i][3])
                    row.extend(
                        [
                            float(sum(q[idx])),
                            float(aw @ sf[idx]),
                            float(vw @ av[idx]),
                            amounts,
                            float(aw @ ocp[idx]),
                            float(aw @ eta[idx]),
                        ]
                    )
                    for key, values in (
                        ("surface_fraction", sf[idx]),
                        ("average_fraction", av[idx]),
                        ("current_A", q[idx]),
                        ("j_A_m2", q[idx] / np.array(m["bin_area_m2"])),
                        ("ocp_V", ocp[idx]),
                        ("overpotential_V", eta[idx]),
                    ):
                        local[name][key].append(values.tolist())
                    local[name]["profiles_theta"].append(
                        [
                            (
                                np.array([saved.amounts_mol[k] for k in bins[i][3]])
                                / (
                                    bins[i][2].mesh.cell_volumes_m3
                                    * bins[i][2].maximum_concentration_mol_m3
                                )
                            ).tolist()
                            for i in idx
                        ]
                    )
                rows.append(row)
            for name in local:
                local[name].update(local_views(local[name], data, name))
            segments.append(
                {
                    "times_s": [s.time_s for s in states],
                    "rows": rows,
                    "local": local,
                    "start_time_s": states[0].time_s,
                    "end_time_s": result.final_state.time_s,
                    "termination": result.stopping_event,
                    "outcome": result.outcome,
                    "raw_amounts_mol": [dict(s.amounts_mol) for s in states],
                    "solver": dict(result.solver_diagnostic),
                    "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
                    "conservation_residuals": dict(result.conservation_residuals),
                }
            )
            state = result.final_state
            if result.outcome != "success":
                break
        except Exception as exc:
            exc.partial_history = {
                "segments": segments,
                "mapping": mapping,
                "stage": stage,
                "target": target,
                "duration_s": duration,
                "current_time_s": state.time_s,
            }
            raise
    return {
        "implementation": "native SPM limit" if single_spm else "native original MPM",
        "old_mpm": old_mpm,
        "single_spm": single_spm,
        "size_nodes": size_nodes,
        "radial_cells": radial_cells,
        "mapping": mapping,
        "segments": segments,
        "response_order": RESPONSES,
        "F_C_mol": F,
        "R_J_mol_K": R,
        "tolerances": {"rtol": RTOL, "atol": ATOL},
    }


def refusal(data, name):
    from chemworld.physchem.battery_degradation import SEISpec
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import BatteryThermalSpec, battery_chemical_energy_j
    from chemworld.physchem.battery_soh import soh_from_battery

    spec = native_spec(data, 4, 8)
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
            native_spec(altered, 4, 8)
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
    area = trace["mapping"]["negative"]["active_area_m2"]
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


def local_metrics(trace, data):
    bv_ratio, allocation, closure, area_error, weight_error = 0.0, 0.0, 0.0, 0.0, 0.0
    f, r = trace["F_C_mol"], trace["R_J_mol_K"]
    for e in data["electrodes"]:
        name = e["name"]
        m = trace["mapping"][name]
        area = np.array(m["bin_area_m2"])
        aw = np.array(m["area_weights"])
        expected_area = (
            3 * e["active_volume_m3"] * np.sum(np.array(m["volume_weights"]) / m["radii_m"])
        )
        area_error = max(area_error, abs(m["active_area_m2"] / expected_area - 1))
        weight_error = max(weight_error, abs(aw.sum() - 1), abs(sum(m["volume_weights"]) - 1))
        for segment in trace["segments"]:
            local = segment["local"][name]
            rows = np.array(segment["rows"])
            theta = np.array(local["surface_fraction"])
            eta = np.array(local["overpotential_V"])
            q = np.array(local["current_A"])
            expected = (
                2
                * e["exchange_current_scale_A_m2"]
                * np.sqrt(theta * (1 - theta))
                * np.sinh(f * eta / (2 * r * data["temperature_K"]))
                * area
            )
            bv_ratio = max(
                bv_ratio, float(np.max(abs(q - expected) / (1e-7 + 1e-7 * abs(expected))))
            )
            total = rows[:, 1] - rows[:, 4] if name == "negative" else -rows[:, 1]
            closure = max(closure, float(np.max(abs(q.sum(axis=1) - total))))
            active = abs(rows[:, 1]) > 1e-10
            if active.any():
                allocation = max(
                    allocation, float(np.max(abs(q[active] - total[active, None] * aw)))
                )
    return {
        "local_BV_limit_ratio": bv_ratio,
        "allocation_departure_from_area_A": allocation,
        "bin_current_closure_A": closure,
        "area_identity_relative_error": area_error,
        "weight_normalization_error": weight_error,
    }
