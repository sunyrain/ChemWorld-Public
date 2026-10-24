"""Released PyBaMM plating equations with explicit external-boundary adapters."""

import numpy as np
from battery_axen_plating_cases import ATOL, PROFILE_RADII, RESPONSES, ROOT_TOL, RTOL


def build_reference(data, cells, *, baseline=False):
    old_spm = baseline and data["index"] == 1
    old_single = baseline and data["index"] == 2
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("fixed plating source is PyBaMM25.4.2")

    def finite_source(variables):
        current = variables["Current [A]"]
        requested = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("plating voltage target")
        )
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], requested)
        )

    cv = data["waveform"][0][2] == "V"
    model = pybamm.lithium_ion.SPM(
        {
            "surface form": "algebraic",
            "thermal": "isothermal",
            "open-circuit potential": ("single", "single")
            if old_single
            else (data["negative_mode"], data["positive_mode"]),
            "particle": "Fickian diffusion",
            "SEI": "none",
            "lithium plating": "none" if old_spm else data["mode"],
            "contact resistance": "true",
            "calculate discharge energy": "true",
            **({"operating mode": finite_source} if cv else {}),
        }
    )
    p = pybamm.ParameterValues("OKane2022")
    length_n = data["negative_electrode_volume_m3"] / data["collector_area_m2"]
    updates = {
        "Electrode height [m]": data["collector_area_m2"],
        "Electrode width [m]": 1.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Negative electrode thickness [m]": length_n,
        "Positive electrode thickness [m]": data["positive_thickness_m"],
        "Separator thickness [m]": data["separator_thickness_m"],
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Reference temperature [K]": data["temperature_K"],
        "Initial concentration in electrolyte [mol.m-3]": data["ce_mol_m3"],
        "Contact resistance [Ohm]": data["series_resistance_ohm"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Lithium plating kinetic rate constant [m.s-1]": data["k_m_s"],
        "Lithium plating transfer coefficient": data["alpha"],
        "Initial plated lithium concentration [mol.m-3]": data["initial_plated_mol"]
        / data["negative_electrode_volume_m3"],
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Current function [A]": "[input]",
    }
    for e in data["electrodes"]:
        domain, title = e["name"], e["name"].capitalize()
        length = length_n if domain == "negative" else data["positive_thickness_m"]

        def compensated_ocv(theta, coefficients=e["polynomial"]):
            return sum(c * theta**i for i, c in enumerate(coefficients)) - 1e-6 * (
                1 / theta + 1 / (theta - 1)
            )

        def exchange(c_e, c_s, c_max, temperature, scale=e["exchange_current_scale_A_m2"]):
            theta = c_s / c_max
            return scale * pybamm.sqrt(theta * (1 - theta))

        updates.update(
            {
                f"{title} particle radius [m]": e["radius_m"],
                f"{title} particle diffusivity [m2.s-1]": e["diffusivity_m2_s"],
                f"Maximum concentration in {domain} electrode [mol.m-3]": e[
                    "maximum_concentration_mol_m3"
                ],
                f"Initial concentration in {domain} electrode [mol.m-3]": e[
                    "maximum_concentration_mol_m3"
                ]
                * e["initial_fraction"],
                f"{title} electrode active material volume fraction": e["active_volume_m3"]
                / (data["collector_area_m2"] * length),
                f"{title} electrode exchange-current density [A.m-2]": exchange,
                f"{title} electrode OCP [V]": compensated_ocv,
                f"{title} electrode OCP entropic change [V.K-1]": 0.0,
            }
        )
        if not old_single and data[domain + "_mode"] == "Axen":
            for direction, offset in (
                ("lithiation", -data[domain + "_gap_V"] / 2),
                ("delithiation", data[domain + "_gap_V"] / 2),
            ):

                def branch(theta, coeffs=e["polynomial"], offset=offset):
                    return (
                        sum(c * theta**i for i, c in enumerate(coeffs))
                        + offset
                        - 1e-6 * (1 / theta + 1 / (theta - 1))
                    )

                updates[f"{title} electrode {direction} OCP [V]"] = branch
            updates.update(
                {
                    f"Initial hysteresis state in {domain} electrode": data[domain + "_initial_h"],
                    f"{title} particle lithiation hysteresis decay rate": data["lithiation_decay"],
                    f"{title} particle delithiation hysteresis decay rate": data[
                        "delithiation_decay"
                    ],
                }
            )
            h = model.variables[f"{title} electrode hysteresis state"]
            model.variables[f"qualification:{domain}:h"] = pybamm.x_average(h)
            model.variables[f"qualification:{domain}:h-rate"] = pybamm.x_average(model.rhs[h])
            model.variables[f"qualification:{domain}:raw-h"] = h
            model.variables[f"qualification:{domain}:raw-h-rate"] = model.rhs[h]
        prefix = f"qualification:{domain}:"
        surface = model.variables[f"X-averaged {domain} particle surface concentration"]
        average = model.variables[f"Average {domain} particle concentration"]
        model.variables.update(
            {
                prefix + "reaction_current_A": model.variables[
                    f"X-averaged {domain} electrode interfacial current density [A.m-2]"
                ]
                * 3
                * e["active_volume_m3"]
                / e["radius_m"],
                prefix + "surface_fraction": surface,
                prefix + "average_fraction": average,
                prefix + "amount_mol": average
                * e["maximum_concentration_mol_m3"]
                * e["active_volume_m3"],
                prefix + "ocp_V": model.variables[
                    f"X-averaged {domain} electrode open-circuit potential [V]"
                ],
                prefix + "overpotential_V": model.variables[
                    f"X-averaged {domain} electrode reaction overpotential [V]"
                ],
                prefix + "profile": model.variables[
                    f"X-averaged {domain} particle concentration [mol.m-3]"
                ],
            }
        )
        model.events.extend(
            [
                pybamm.Event(
                    f"battery:{domain}:depleted", 1e4 * (surface - data["surface_margin"])
                ),
                pybamm.Event(
                    f"battery:{domain}:saturated", 1e4 * (1 - data["surface_margin"] - surface)
                ),
            ]
        )
    area = 3 * data["electrodes"][0]["active_volume_m3"] / data["electrodes"][0]["radius_m"]
    plated = model.variables["X-averaged negative lithium plating concentration [mol.m-3]"]
    dead = model.variables["X-averaged negative dead lithium concentration [mol.m-3]"]
    model.variables.update(
        {
            "qualification:stripping_current_A": area
            * model.variables[
                "X-averaged negative electrode lithium plating interfacial current density [A.m-2]"
            ],
            "qualification:plated_mol": plated * data["negative_electrode_volume_m3"],
            "qualification:dead_mol": dead * data["negative_electrode_volume_m3"],
            "qualification:negative_potential_V": model.variables[
                "X-averaged negative electrode surface potential difference [V]"
            ],
            "qualification:plated_concentration": plated,
        }
    )
    model.events = [
        event
        for event in model.events
        if event.name not in {"Minimum voltage [V]", "Maximum voltage [V]"}
    ]
    voltage = model.variables["Voltage [V]"]
    model.events.extend(
        [
            pybamm.Event("battery:minimum-voltage", 1e4 * (voltage - data["minimum_voltage_V"])),
            pybamm.Event("battery:maximum-voltage", 1e4 * (data["maximum_voltage_V"] - voltage)),
        ]
    )
    p.update(updates, check_already_exists=False)
    points = {var: cells if str(var).startswith("r_") else 5 for var in model.default_var_pts}
    return pybamm.Simulation(
        model,
        parameter_values=p,
        var_pts=points,
        solver=pybamm.CasadiSolver(mode="safe", rtol=RTOL, atol=ATOL, root_tol=ROOT_TOL),
    )


def reference_time(data, cells, *, baseline=False):
    import pybamm
    from battery_axen_plating_cases import active_electrodes

    simulation, segments = build_reference(data, cells, baseline=baseline), []
    names = (
        "Voltage [V]",
        "Current [A]",
        "Discharge capacity [A.h]",
        "Discharge energy [W.h]",
        *("qualification:" + name for name in RESPONSES[4:]),
    )
    active = active_electrodes(data, baseline=baseline)
    state_scales = None
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
                    [
                        np.interp(
                            PROFILE_RADII * e["radius_m"], radial_nodes, values[:, i]
                        ).tolist()
                        for i in range(len(times))
                    ]
                )
                nodes.append((radial_nodes / e["radius_m"]).tolist())
                raw_profiles.append(values.T.tolist())
            hysteresis = {}
            for name in active:
                h = {}
                for quantity in ("h", "h-rate", "raw-h", "raw-h-rate"):
                    variable = solution[f"qualification:{name}:{quantity}"]
                    values = np.asarray(variable.entries)
                    expected = (5, len(times)) if quantity.startswith("raw") else (len(times),)
                    if values.shape != expected:
                        raise ValueError(
                            f"source {name}:{quantity} expected {expected}, got {values.shape}"
                        )
                    h[quantity] = values.tolist()
                hysteresis[name] = h
            if state_scales is None:
                state_scales = {
                    v.name: float(v.scale.evaluate())
                    for v in simulation.built_model.rhs
                    if any(
                        k in v.name
                        for k in ("hysteresis", "lithium plating concentration", "dead lithium")
                    )
                }
            segments.append(
                {
                    "times_s": times.tolist(),
                    "rows": rows.tolist(),
                    "start_time_s": float(times[0]),
                    "end_time_s": float(times[-1]),
                    "termination": solution.termination,
                    "hysteresis": hysteresis,
                    "profiles_theta": profiles,
                    "raw_r_over_R": nodes,
                    "raw_profiles_theta": raw_profiles,
                    "sei_thickness_m": np.asarray(
                        solution["X-averaged negative total SEI thickness [m]"](times)
                    )
                    .reshape(-1)
                    .tolist(),
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
        "implementation": "actual PyBaMM25.4.2 SPM Axen/plating",
        "radial_cells": cells,
        "baseline": baseline,
        "has_plating": not (baseline and data["index"] == 1),
        "h_electrodes": active,
        "source_state_scales": state_scales,
        "segments": segments,
        "response_order": RESPONSES,
        "profile_r_over_R": PROFILE_RADII.tolist(),
        "F_C_mol": float(pybamm.constants.F.value),
        "R_J_mol_K": float(pybamm.constants.R.value),
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ROOT_TOL},
        "source_CV_boundary": "authored ideal algebraic limited source via public callable",
        "geometric_negative_volume_m3": data["negative_electrode_volume_m3"],
    }
