"""Actual released x-full SPM with explicit synthetic thermal/electrode parameters."""

import numpy as np
from battery_spatial_thermal_cases import ATOL, PROFILE_POINTS, RTOL, F, thermal_geometry


def build_reference(data, thermal_cells, radial_cells):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("x-full reference requires PyBaMM25.4.2")

    def finite_source(variables):
        current = variables["Current [A]"]
        request = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("spatial thermal voltage target")
        )
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], request)
        )

    options = {
        "thermal": "x-full",
        "dimensionality": 0,
        "cell geometry": "pouch",
        "surface temperature": "ambient",
        "particle": "Fickian diffusion",
        "heat of mixing": "true",
        "SEI": "none",
        "lithium plating": "none",
        "calculate discharge energy": "true",
        "contact resistance": "false",
        **({"operating mode": finite_source} if data["waveform"][0][2] == "V" else {}),
    }
    model = pybamm.lithium_ion.SPM(options)
    parameters = pybamm.ParameterValues("Chen2020")
    area = data["width_m"] * data["height_m"]
    updates = {
        "Electrode width [m]": data["width_m"],
        "Electrode height [m]": data["height_m"],
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Initial concentration in electrolyte [mol.m-3]": data["electrolyte_concentration_mol_m3"],
        "Initial temperature [K]": data["initial_temperature_K"],
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": data["ambient_temperature_K"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Edge heat transfer coefficient [W.m-2.K-1]": data["edge_heat_transfer_W_m2_K"],
        "Current function [A]": "[input]",
    }
    for layer in data["layers"]:
        name = (
            "Separator"
            if layer["name"] == "separator"
            else layer["name"].capitalize() + " electrode"
        )
        updates.update(
            {
                name + " thickness [m]": layer["length_m"],
                name + " density [kg.m-3]": 1000.0,
                name + " specific heat capacity [J.kg-1.K-1]": layer["heat_capacity_J_m3_K"] / 1000,
                name + " thermal conductivity [W.m-1.K-1]": layer["conductivity_W_m_K"],
            }
        )
    for collector in data["collectors"]:
        name = collector["name"].capitalize()
        updates.update(
            {
                name + " current collector thickness [m]": collector["length_m"],
                name + " current collector density [kg.m-3]": 1000.0,
                name + " current collector specific heat capacity [J.kg-1.K-1]": collector[
                    "heat_capacity_J_m3_K"
                ]
                / 1000,
                name + " current collector conductivity [S.m-1]": np.inf,
                name
                + " current collector surface heat transfer coefficient [W.m-2.K-1]": collector[
                    "face_heat_transfer_W_m2_K"
                ],
                name + " tab width [m]": collector["tab_width_m"],
                name + " tab heat transfer coefficient [W.m-2.K-1]": collector[
                    "tab_heat_transfer_W_m2_K"
                ],
            }
        )
    for electrode in data["electrodes"]:
        name, domain = electrode["name"].capitalize(), electrode["name"]
        layer = next(item for item in data["layers"] if item["name"] == domain)

        def compensated_ocv(theta, value=electrode["ocv_V"]):
            return value - 1e-6 * (1 / theta + 1 / (theta - 1))

        def exchange(c_e, c_s, c_max, temperature, scale=electrode["exchange_current_scale_A_m2"]):
            theta = c_s / c_max
            return scale * pybamm.sqrt(theta * (1 - theta))

        updates.update(
            {
                name + " particle radius [m]": electrode["radius_m"],
                name + " particle diffusivity [m2.s-1]": electrode["diffusivity_m2_s"],
                f"Maximum concentration in {domain} electrode [mol.m-3]": electrode[
                    "maximum_concentration_mol_m3"
                ],
                f"Initial concentration in {domain} electrode [mol.m-3]": electrode[
                    "maximum_concentration_mol_m3"
                ]
                * electrode["initial_fraction"],
                name + " electrode active material volume fraction": electrode["active_volume_m3"]
                / (area * layer["length_m"]),
                name + " electrode exchange-current density [A.m-2]": exchange,
                name + " electrode OCP [V]": compensated_ocv,
                name + " electrode OCP entropic change [V.K-1]": 0.0,
            }
        )
        surface = model.variables[f"X-averaged {domain} particle surface concentration"]
        mean = model.variables[f"Average {domain} particle concentration"]
        model.variables[f"qualification:{domain}:surface"] = surface
        model.variables[f"qualification:{domain}:average"] = mean
        model.variables[f"qualification:{domain}:amount"] = (
            mean * electrode["maximum_concentration_mol_m3"] * electrode["active_volume_m3"]
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
    parameters.update(updates, check_already_exists=False)

    temperature = model.variables["Cell temperature [K]"]
    negative_cc = model.variables["Negative current collector temperature [K]"]
    positive_cc = model.variables["Positive current collector temperature [K]"]
    if data.get("initial_profile") == "linear-304-to-294":
        length = sum(layer["length_m"] for layer in data["layers"])
        profile = pybamm.concatenation(
            *(
                304.0 - 10 * getattr(pybamm.standard_spatial_vars, "x_" + short) / length
                for short in ("n", "s", "p")
            )
        )
        model.initial_conditions[temperature] = profile
        model.initial_conditions[negative_cc] = pybamm.Scalar(304.0)
        model.initial_conditions[positive_cc] = pybamm.Scalar(294.0)

    sensible, capacity, weighted, loss, volume_weighted, volume = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    perimeter = 2 * (data["width_m"] + data["height_m"])
    for layer in data["layers"]:
        domain = "separator" if layer["name"] == "separator" else layer["name"] + " electrode"
        mean = model.variables[f"X-averaged {domain} temperature [K]"]
        c = area * layer["length_m"] * layer["heat_capacity_J_m3_K"]
        v = area * layer["length_m"]
        sensible += c * (mean - 298.15)
        capacity += c
        weighted += c * mean
        volume_weighted += v * mean
        volume += v
        loss += (
            perimeter
            * layer["length_m"]
            * data["edge_heat_transfer_W_m2_K"]
            * (mean - data["ambient_temperature_K"])
        )
    for collector, value in zip(data["collectors"], (negative_cc, positive_cc), strict=True):
        c = area * collector["length_m"] * collector["heat_capacity_J_m3_K"]
        v = area * collector["length_m"]
        sensible += c * (value - 298.15)
        capacity += c
        weighted += c * value
        volume_weighted += v * value
        volume += v
        tab = collector["tab_width_m"] * collector["length_m"]
        conductance = (
            area * collector["face_heat_transfer_W_m2_K"]
            + tab * collector["tab_heat_transfer_W_m2_K"]
            + (perimeter * collector["length_m"] - tab) * data["edge_heat_transfer_W_m2_K"]
        )
        loss += conductance * (value - data["ambient_temperature_K"])
    loss_receipt = pybamm.Variable("qualification:loss_J")
    model.rhs[loss_receipt], model.initial_conditions[loss_receipt] = loss, pybamm.Scalar(0.0)
    model.variables.update(
        {
            "qualification:sensible_J": sensible,
            "qualification:C_mean_K": weighted / capacity,
            "qualification:volume_mean_K": volume_weighted / volume,
            "qualification:loss_W": loss,
            "qualification:loss_J": loss_receipt,
            "qualification:chemical_J": -F
            * sum(
                e["ocv_V"] * model.variables[f"qualification:{e['name']}:amount"]
                for e in data["electrodes"]
            ),
        }
    )
    # Separate scalar roots avoid broadcasting a scalar sandwich Max/Min over
    # the CC domain. The earliest of these roots is the actual local boundary.
    for name, local_max, local_min in (
        ("sandwich", pybamm.max(temperature), pybamm.min(temperature)),
        ("negative-collector", negative_cc, negative_cc),
        ("positive-collector", positive_cc, positive_cc),
    ):
        model.events.extend(
            [
                pybamm.Event(
                    f"battery:spatial:{name}:maximum-temperature",
                    data["maximum_temperature_K"] - local_max,
                ),
                pybamm.Event(
                    f"battery:spatial:{name}:minimum-temperature",
                    local_min - data["minimum_temperature_K"],
                ),
            ]
        )
    voltage = model.variables["Voltage [V]"]
    model.events = [
        event
        for event in model.events
        if event.name not in {"Minimum voltage [V]", "Maximum voltage [V]"}
    ]
    model.events.extend(
        [
            pybamm.Event("battery:minimum-voltage", 1e4 * (voltage - data["minimum_voltage_V"])),
            pybamm.Event("battery:maximum-voltage", 1e4 * (data["maximum_voltage_V"] - voltage)),
        ]
    )
    points = {
        var: radial_cells if str(var).startswith("r_") else thermal_cells
        for var in model.default_var_pts
    }
    return pybamm.Simulation(
        model,
        parameter_values=parameters,
        var_pts=points,
        solver=pybamm.CasadiSolver(mode="safe", rtol=RTOL, atol=ATOL, root_tol=ATOL),
    )


OUTPUT_NAMES = {
    "voltage_V": "Voltage [V]",
    "current_A": "Current [A]",
    "charge_C": "Discharge capacity [A.h]",
    "work_J": "Discharge energy [W.h]",
    "negative_surface": "qualification:negative:surface",
    "negative_average": "qualification:negative:average",
    "negative_amount_mol": "qualification:negative:amount",
    "positive_surface": "qualification:positive:surface",
    "positive_average": "qualification:positive:average",
    "positive_amount_mol": "qualification:positive:amount",
    "negative_collector_K": "Negative current collector temperature [K]",
    "negative_mean_K": "X-averaged negative electrode temperature [K]",
    "separator_mean_K": "X-averaged separator temperature [K]",
    "positive_mean_K": "X-averaged positive electrode temperature [K]",
    "positive_collector_K": "Positive current collector temperature [K]",
    "C_mean_K": "qualification:C_mean_K",
    "volume_mean_K": "qualification:volume_mean_K",
    "sensible_J": "qualification:sensible_J",
    "chemical_J": "qualification:chemical_J",
    "loss_W": "qualification:loss_W",
    "loss_J": "qualification:loss_J",
    "generated_W": "Total heating [W]",
    "irreversible_W": "Irreversible electrochemical heating [W]",
    "reversible_W": "Reversible heating [W]",
    "mixing_W": "Heat of mixing [W]",
    "ohmic_W": "Ohmic heating [W]",
    "negative_CC_ohmic_W_m3": "Negative current collector Ohmic heating [W.m-3]",
    "positive_CC_ohmic_W_m3": "Positive current collector Ohmic heating [W.m-3]",
}


def reference_time(data, grid):
    cells, radial = grid
    geometry = thermal_geometry(data, cells)
    simulation, segments = build_reference(data, cells, radial), []
    for duration, target, unit in data["waveform"]:
        solution = simulation.step(
            duration,
            save=False,
            t_eval=np.linspace(0, duration, data["samples_per_segment"]),
            inputs={
                "spatial thermal voltage target" if unit == "V" else "Current function [A]": target
            },
        )
        times = np.asarray(solution.t)
        values = {}
        for key, name in OUTPUT_NAMES.items():
            value = np.asarray(solution[name](times))
            if value.size != times.size:
                raise ValueError("scalar source output has non-scalar shape: " + name)
            values[key] = value.reshape(-1) * (3600 if key in {"charge_C", "work_J"} else 1)
        processed_temperature = solution["Cell temperature [K]"]
        # Bare processed-variable calls append extrapolated endpoint values.
        # Read every original FV node explicitly; no source value is replaced.
        raw = np.asarray(processed_temperature(times, x=processed_temperature.mesh.nodes)).T
        if raw.shape != (times.size, 3 * cells):
            raise ValueError("actual x-full temperature mesh differs from declared source grid")
        full = np.column_stack(
            (values["negative_collector_K"], raw, values["positive_collector_K"])
        )
        # Keep source expressions and independently recompute both averages from
        # every actual FV/CC node; a source volume average is not a C average.
        for key, weight in (
            ("C_mean_K", geometry["capacities_J_K"]),
            ("volume_mean_K", geometry["volumes_m3"]),
        ):
            values[key + "_raw_nodes"] = full @ weight / weight.sum()
        values["maximum_K"], values["minimum_K"] = full.max(axis=1), full.min(axis=1)
        profiles = np.stack(
            [
                np.array(
                    [
                        np.interp(
                            PROFILE_POINTS,
                            (np.arange(cells) + 0.5) / cells,
                            row[i * cells : (i + 1) * cells],
                        )
                        for row in raw
                    ]
                )
                for i in range(3)
            ],
            axis=1,
        )
        segments.append(
            {
                "times_s": times.tolist(),
                "values": {name: value.tolist() for name, value in values.items()},
                "temperatures_K": full.tolist(),
                "sampled_profiles_K": profiles.tolist(),
                "end_time_s": float(times[-1]),
                "termination": solution.termination,
            }
        )
        if solution.termination != "final time":
            break
    return {
        "implementation": "PyBaMM25.4.2 actual x-full SPM with two finite CC capacities",
        "thermal_cells_per_layer": cells,
        "radial_cells": radial,
        "segments": segments,
        "geometry": {name: value.tolist() for name, value in geometry.items()},
        "profile_local_x_over_L": PROFILE_POINTS.tolist(),
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ATOL},
        "CC_electrical_conductivity": "explicit positive infinity, ideal limit",
    }
