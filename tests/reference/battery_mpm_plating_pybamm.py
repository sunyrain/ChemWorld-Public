"""Released PyBaMM plating equations with explicit external-boundary adapters."""

import numpy as np
from battery_mpm_plating_cases import ATOL, ROOT_TOL, RTOL


def build_reference(data, size_cells, radial_cells, *, old_mpm=False):
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
    model_class = pybamm.lithium_ion.SPM if data["single_bin"] else pybamm.lithium_ion.MPM
    model = model_class(
        {
            "surface form": "algebraic",
            "thermal": "isothermal",
            "open-circuit potential": "single",
            "particle": "Fickian diffusion",
            "SEI": "none",
            "lithium plating": "none" if old_mpm else data["mode"],
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
        prefix = f"qualification:{domain}:"
        surface = model.variables[
            f"X-averaged {domain} particle surface concentration"
            + ("" if data["single_bin"] else " distribution")
        ]
        if not data["single_bin"]:
            pdf = data["pdfs"][domain]

            def density(radius, median=pdf["median_m"], sigma=pdf["sigma"]):
                return pybamm.exp(-(pybamm.log(radius / median) ** 2) / (2 * sigma**2)) / radius

            updates.update(
                {
                    f"{title} minimum particle radius [m]": pdf["low_m"],
                    f"{title} maximum particle radius [m]": pdf["high_m"],
                    f"{title} area-weighted particle-size distribution [m-1]": density,
                }
            )
        average = model.variables[f"Average {domain} particle concentration"]
        area = model.variables[f"X-averaged {domain} electrode surface area to volume ratio [m-1]"]
        area = area * data["collector_area_m2"] * length
        model.variables.update(
            {
                prefix + "surface_fraction": pybamm.size_average(surface),
                prefix + "average_fraction": average,
                prefix + "amount_mol": average
                * e["maximum_concentration_mol_m3"]
                * e["active_volume_m3"],
                prefix + "area_m2": area,
                prefix + "reaction_current_A": area
                * model.variables[
                    f"X-averaged {domain} electrode interfacial current density [A.m-2]"
                ],
                prefix + "ocp_V": model.variables[
                    f"X-averaged {domain} electrode open-circuit potential [V]"
                ],
                prefix + "overpotential_V": model.variables[
                    f"X-averaged {domain} electrode reaction overpotential [V]"
                ],
            }
        )
        model.events.extend(
            [
                pybamm.Event(
                    f"battery:{domain}:depleted",
                    1e4 * (pybamm.min(surface) - data["surface_margin"]),
                ),
                pybamm.Event(
                    f"battery:{domain}:saturated",
                    1e4 * (1 - data["surface_margin"] - pybamm.max(surface)),
                ),
            ]
        )
    plated = model.variables["X-averaged negative lithium plating concentration [mol.m-3]"]
    dead = model.variables["X-averaged negative dead lithium concentration [mol.m-3]"]
    model.variables.update(
        {
            "qualification:stripping_current_A": model.variables["qualification:negative:area_m2"]
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
    points = {
        var: radial_cells
        if str(var).startswith("r_")
        else size_cells
        if str(var).startswith("R_")
        else 5
        for var in model.default_var_pts
    }
    return pybamm.Simulation(
        model,
        parameter_values=p,
        var_pts=points,
        solver=pybamm.CasadiSolver(mode="safe", rtol=RTOL, atol=ATOL, root_tol=ROOT_TOL),
    )


def reference_time(data, size_cells, radial_cells, *, old_mpm=False):
    import pybamm
    from battery_mpm_plating_cases import RESPONSES
    from battery_mpm_plating_traces import local_views

    simulation = build_reference(data, size_cells, radial_cells, old_mpm=old_mpm)
    segments, mapping = [], {}
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
            rows = np.column_stack(
                [np.asarray(solution[name](times)).reshape(-1) for name in names]
            )
            rows[:, 2:4] *= 3600
            local = {}
            for e in data["electrodes"]:
                name = e["name"]
                nb = 1 if data["single_bin"] else size_cells
                sizes = (
                    np.array([e["radius_m"]])
                    if data["single_bin"]
                    else np.asarray(simulation.mesh[name + " particle size"].nodes)
                )
                radial = np.asarray(simulation.mesh[name + " particle"].nodes) / e["radius_m"]

                def read_variable(key, solution=solution, times=times, sizes=sizes, nb=nb):
                    if data["single_bin"]:
                        return np.asarray(solution[key](times)).reshape(-1, 1)
                    return np.asarray(solution[key](times, R=sizes)).reshape(nb, -1).T

                suffix = "" if data["single_bin"] else " distribution"
                area = float(np.asarray(solution[f"qualification:{name}:area_m2"](times[0])).item())
                if data["single_bin"]:
                    aw = vw = np.ones(1)
                else:
                    widths = np.diff(simulation.mesh[name + " particle size"].edges)
                    aw = (
                        np.asarray(
                            solution[
                                f"X-averaged {name} area-weighted particle-size distribution [m-1]"
                            ](times[0], R=sizes)
                        ).reshape(-1)
                        * widths
                    )
                    vw = (
                        np.asarray(
                            solution[
                                f"X-averaged {name} volume-weighted particle-size "
                                "distribution [m-1]"
                            ](times[0], R=sizes)
                        ).reshape(-1)
                        * widths
                    )
                mapping[name] = {
                    "radii_m": sizes.tolist(),
                    "volume_weights": vw.tolist(),
                    "area_weights": aw.tolist(),
                    "bin_area_m2": (area * aw).tolist(),
                    "active_area_m2": area,
                    "area_mean_radius_m": float(aw @ sizes),
                    "input_basis": "single bin"
                    if data["single_bin"]
                    else "source normalized area PDF FV",
                }
                profile_name = f"X-averaged {name} particle concentration{suffix} [mol.m-3]"
                raw = np.asarray(solution[profile_name].entries) / e["maximum_concentration_mol_m3"]
                if data["single_bin"]:
                    if raw.shape != (radial_cells, len(times)):
                        raise ValueError("source single profile shape")
                    raw = raw[:, None, :]
                elif raw.shape != (radial_cells, nb, len(times)):
                    raise ValueError(f"source MPM raw profile shape {raw.shape}")
                surface = read_variable(f"X-averaged {name} particle surface concentration{suffix}")
                average = read_variable(f"Average {name} particle concentration{suffix}")
                currentdensity = read_variable(
                    f"X-averaged {name} electrode interfacial current density{suffix} [A.m-2]"
                )
                ocp = read_variable(
                    f"X-averaged {name} electrode open-circuit potential{suffix} [V]"
                )
                eta = read_variable(
                    f"X-averaged {name} electrode reaction overpotential{suffix} [V]"
                )
                local[name] = {
                    "radii_m": sizes.tolist(),
                    "r_over_R": radial.tolist(),
                    "surface_fraction": surface.tolist(),
                    "average_fraction": average.tolist(),
                    "current_A": (currentdensity * area * aw).tolist(),
                    "j_A_m2": currentdensity.tolist(),
                    "ocp_V": ocp.tolist(),
                    "overpotential_V": eta.tolist(),
                    "profiles_theta": raw.transpose(2, 1, 0).tolist(),
                }
                local[name].update(local_views(local[name], data, name))
            segments.append(
                {
                    "times_s": times.tolist(),
                    "rows": rows.tolist(),
                    "local": local,
                    "start_time_s": float(times[0]),
                    "end_time_s": float(times[-1]),
                    "termination": solution.termination,
                }
            )
            if solution.termination != "final time":
                break
        except Exception as exc:
            exc.partial_history = {
                "segments": segments,
                "mapping": mapping,
                "stage": stage,
                "target": target,
                "duration_s": duration,
                "last_completed_time_s": segments[-1]["end_time_s"] if segments else 0.0,
            }
            raise
    return {
        "implementation": "actual PyBaMM25.4.2 SPM limit"
        if data["single_bin"]
        else "actual PyBaMM25.4.2 MPM",
        "old_mpm": old_mpm,
        "size_nodes": size_cells,
        "radial_cells": radial_cells,
        "mapping": mapping,
        "segments": segments,
        "response_order": RESPONSES,
        "F_C_mol": float(pybamm.constants.F.value),
        "R_J_mol_K": float(pybamm.constants.R.value),
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ROOT_TOL},
        "source_CV_boundary": "authored algebraic limited source via public callable",
    }
