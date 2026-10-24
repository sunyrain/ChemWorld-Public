"""Native-state extraction and separately assembled linear/thermodynamic limits."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_msmr_cases import (
    ATOL,
    PROFILE_RADII,
    RTOL,
    F,
    initial_profile,
    native_initial_state,
    native_material,
    native_spec,
)
from battery_msmr_pybamm import build_reference, independent_bv, static_reference_values
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


def native_time(data, cells):
    from chemworld.physchem.battery_msmr import msmr_fields, msmr_system
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, cells)
    state, segments = native_initial_state(spec, data, cells), []
    for duration, target, unit in data["waveform"]:
        program = ScalarControlProgram(((0.0, target),), unit)
        system = msmr_system(spec, program)
        system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
        result = system.advance(
            state,
            duration,
            evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
        )
        states = list(result.trajectory)
        if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
            states.append(result.final_state)
        values, raw = (
            {},
            {
                e["name"]: {
                    k: [] for k in ("theta", "potential_V", "sampled_theta", "sampled_potential_V")
                }
                for e in data["electrodes"]
            },
        )
        for saved in states:
            fields = msmr_fields(spec, saved, program=program)
            row = {
                "voltage_V": fields["voltage_V"],
                "current_A": fields["current_A"],
                "charge_C": saved.coordinates["discharged_charge_C"],
                "work_J": saved.coordinates["delivered_work_J"],
            }
            for e in data["electrodes"]:
                name, p = e["name"], getattr(spec, e["name"])
                theta = np.asarray(fields[name + "_particle_fractions"])
                u = np.asarray(fields[name + "_particle_potentials_V"])
                radial = p.mesh.centers_m / p.radius_m
                row.update(
                    {
                        name + ":surface": fields[name + "_surface_fraction"],
                        name + ":source_surface": sum(fields[name + "_source_surface_occupancies"]),
                        name + ":surface_potential": fields[name + "_potential_V"],
                        name + ":eta": fields[name + "_overpotential_V"],
                        name + ":average": fields[name + "_average_fraction"],
                        name + ":amount_mol": sum(saved.amounts_mol[k] for k in p.keys(name)),
                    }
                )
                for j in range(len(e["rows"])):
                    row[name + f":kinetic_x{j}"] = fields[name + "_surface_occupancies"][j]
                    row[name + f":source_x{j}"] = fields[name + "_source_surface_occupancies"][j]
                    row[name + f":current{j}_A"] = (
                        fields[name + "_reaction_current_densities_A_m2"][j] * p.area_m2
                    )
                raw[name]["r_over_R"] = radial.tolist()
                raw[name]["shell_weights"] = (
                    p.mesh.cell_volumes_m3 / e["active_volume_m3"]
                ).tolist()
                raw[name]["theta"].append(theta.tolist())
                raw[name]["potential_V"].append(u.tolist())
                raw[name]["sampled_theta"].append(np.interp(PROFILE_RADII, radial, theta).tolist())
                raw[name]["sampled_potential_V"].append(
                    np.interp(PROFILE_RADII, radial, u).tolist()
                )
            for key, value in row.items():
                values.setdefault(key, []).append(float(value))
        segments.append(
            {
                "times_s": [s.time_s for s in states],
                "values": values,
                "raw_profiles": raw,
                "end_time_s": result.final_state.time_s,
                "termination": result.stopping_event,
                "outcome": result.outcome,
                "conservation_residuals": dict(result.conservation_residuals),
                "solver": dict(result.solver_diagnostic),
            }
        )
        state = result.final_state
        if result.outcome != "success":
            break
    return {
        "implementation": "Native unique particle Li / original coupled residual",
        "radial_cells": cells,
        "segments": segments,
        "profile_r_over_R": PROFILE_RADII.tolist(),
        "tolerances": {"rtol": RTOL, "atol": ATOL},
    }


def linear_fick_limit(data, cells):
    """Independent spherical-shell diffusion matrix, not the native/source MSMR RHS."""
    if "fick_multiplier" not in data:
        raise ValueError("linear limit is fixed only for MS02/MS03")
    duration, current, unit = data["waveform"][0]
    if unit != "A" or len(data["waveform"]) != 1:
        raise ValueError("linear limit requires the fixed constant-current history")
    times = np.linspace(0.0, duration, data["samples_per_segment"])
    profiles = {}
    for e, sign in zip(data["electrodes"], (1.0, -1.0), strict=True):
        radial, _, x, weights = initial_profile(e, cells, data["temperature_K"])
        edges = np.linspace(0.0, e["radius_m"], cells + 1)
        volumes = np.diff(edges**3) / 3
        conductance = (
            data["fick_multiplier"]
            * e["diffusivity_m2_s"]
            * edges[1:-1] ** 2
            / (e["radius_m"] / cells)
        )
        matrix = np.zeros((cells + 1, cells + 1))
        for j, g in enumerate(conductance):
            matrix[j, j] -= g / volumes[j]
            matrix[j, j + 1] += g / volumes[j]
            matrix[j + 1, j] += g / volumes[j + 1]
            matrix[j + 1, j + 1] -= g / volumes[j + 1]
        capacity = e["maximum_concentration_mol_m3"] * e["active_volume_m3"]
        matrix[-2, -1] = -sign * current / (F * capacity * weights[-1])
        theta = expm_multiply(
            csr_matrix(matrix), np.r_[x, 1.0], start=0.0, stop=duration, num=len(times)
        )[:, :-1]
        amounts = theta @ weights * capacity
        profiles[e["name"]] = {
            "r_over_R": radial.tolist(),
            "theta": theta.tolist(),
            "sampled_theta": [np.interp(PROFILE_RADII, radial, row).tolist() for row in theta],
            "amount_mol": amounts.tolist(),
            "shell_weights": weights.tolist(),
            "faraday_error_C": float(
                np.max(abs(F * (amounts - amounts[0]) + sign * current * times))
            ),
            "diffusivity_m2_s": data["fick_multiplier"] * e["diffusivity_m2_s"],
        }
    total = sum(np.array(value["amount_mol"]) for value in profiles.values())
    return {
        "implementation": "Independent linear spherical FV matrix exponential",
        "radial_cells": cells,
        "times_s": times.tolist(),
        "profiles": profiles,
        "lithium_balance_mol": float(np.max(abs(total - total[0]))),
        "meaning": "Continuum/small-gradient wD limit, not identical finite-grid MSMR mobility",
    }


def static_block(data):
    sim, points = build_reference(data, 8), []
    for e in data["electrodes"]:
        material = native_material(e)
        for temperature in (280.0, 298.15, 320.0):
            for u in np.linspace(
                min(r[1] for r in e["rows"]) - 0.12, max(r[1] for r in e["rows"]) + 0.12, 9
            ):
                for eta in (-0.05, 0.0, 0.05):
                    for ratio in (0.5, 1.0, 1.5):
                        independent = independent_bv(e["rows"], u, eta, temperature, ratio * 1000.0)
                        actual = static_reference_values(
                            data, e, temperature, u, eta, ratio * 1000.0, simulation=sim
                        )
                        native = {
                            "x": material.occupancies(u, temperature),
                            "dx_total": material.stoichiometry_derivatives(u, temperature)[1],
                            "j0": material.exchange_current_densities(
                                u, temperature, ratio * 1000.0
                            ),
                            "j": material.reaction_currents(u, eta, temperature, ratio * 1000.0),
                        }
                        fraction = sum(independent["x"])
                        inverse = material.potential(fraction, temperature)
                        comparisons = {}
                        for implementation, output in (("native", native), ("source", actual)):
                            for key in ("x", "j0", "j"):
                                a, b = np.asarray(output[key]), np.asarray(independent[key])
                                comparisons[implementation + ":" + key] = float(
                                    np.max(abs(a - b) / (2e-10 + 2e-9 * abs(b)))
                                )
                            derivative = output.get("dx_total", sum(output.get("dx", [])))
                            target = sum(independent["dx"])
                            comparisons[implementation + ":dx"] = abs(derivative - target) / (
                                2e-10 + 2e-9 * abs(target)
                            )
                        comparisons["native:inverse"] = abs(inverse - u) / 2e-10
                        points.append(
                            {
                                "electrode": e["name"],
                                "temperature_K": temperature,
                                "potential_V": u,
                                "overpotential_V": eta,
                                "ce_ratio": ratio,
                                "independent": independent,
                                "source": actual,
                                "native": native,
                                "inverse_V": inverse,
                                "tolerance_ratios": comparisons,
                                "passed": all(v <= 1 for v in comparisons.values()),
                            }
                        )
    return {
        "points": points,
        "point_denominator": 486,
        "comparison_denominator": 486 * 9,
        "passed_points": sum(p["passed"] for p in points),
        "maximum_tolerance_ratio": max(max(p["tolerance_ratios"].values()) for p in points),
    }


def refusal(data, name):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import battery_chemical_energy_j
    from chemworld.physchem.battery_msmr import msmr_system
    from chemworld.physchem.battery_soh import soh_from_battery
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, 8)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s, state.temperature_K)
    try:
        if name in {"non-normalized-X", "nonpositive-w", "boundary-alpha"}:
            e = deepcopy(data["electrodes"][0])
            key, value = {
                "non-normalized-X": (0, 0.5),
                "nonpositive-w": (2, 0.0),
                "boundary-alpha": (4, 1.0),
            }[name]
            e["rows"][0][key] = value
            native_material(e)
        elif name == "nonuniform-mesh":
            replace(spec, negative=replace(spec.negative, mesh_exponent=2.0))
        elif name == "EIS":
            battery_impedance(spec, (1.0,), state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        elif name == "energy":
            battery_chemical_energy_j(spec, state)
        elif name == "thermal":
            msmr_system(
                spec, ScalarControlProgram(((0.0, 0.1),), "A"), thermal_device=ThermalDeviceSpec()
            )
        else:
            raise RuntimeError("unknown fixed refusal")
        result = {"rejected": False}
    except Exception as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    after = (dict(state.amounts_mol), dict(state.coordinates), state.time_s, state.temperature_K)
    return {**result, "input_state_unchanged": before == after}
