"""Released actual MSMR SPM; independent input and observable mapping."""

import numpy as np
from battery_msmr_cases import ATOL, PROFILE_RADII, RTOL, occupancies, potential


def build_reference(data, cells):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("MSMR reference requires PyBaMM25.4.2")

    def finite_source(variables):
        current = variables["Current [A]"]
        requested = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("MSMR voltage target")
        )
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], requested)
        )

    options = {
        "open-circuit potential": "MSMR",
        "particle": "MSMR",
        "intercalation kinetics": "MSMR",
        "surface form": "algebraic",
        "number of MSMR reactions": tuple(str(len(e["rows"])) for e in data["electrodes"]),
        "thermal": "isothermal",
        "SEI": "none",
        "lithium plating": "none",
        "contact resistance": "true",
        "calculate discharge energy": "true",
    }
    if data["waveform"][0][2] == "V":
        options["operating mode"] = finite_source
    model = pybamm.lithium_ion.SPM(options)
    parameters = pybamm.ParameterValues("MSMR_Example")
    updates = {
        "Electrode height [m]": data["collector_area_m2"],
        "Electrode width [m]": 1.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Negative electrode thickness [m]": data["electrode_thickness_m"],
        "Positive electrode thickness [m]": data["electrode_thickness_m"],
        "Contact resistance [Ohm]": data["resistance_ohm"],
        "Initial concentration in electrolyte [mol.m-3]": data["electrolyte_concentration_mol_m3"],
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Reference temperature [K]": data["temperature_K"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Current function [A]": "[input]",
    }
    for e in data["electrodes"]:
        domain, title, short = e["name"], e["name"].capitalize(), e["name"][0]
        initial_u = potential(e["rows"], e["initial_fraction"], data["temperature_K"])
        updates.update(
            {
                f"{title} particle radius [m]": e["radius_m"],
                f"{title} particle diffusivity [m2.s-1]": e["diffusivity_m2_s"],
                f"Maximum concentration in {domain} electrode [mol.m-3]": e[
                    "maximum_concentration_mol_m3"
                ],
                f"Initial voltage in {domain} electrode [V]": initial_u,
                f"{title} electrode active material volume fraction": e["active_volume_m3"]
                / (data["collector_area_m2"] * data["electrode_thickness_m"]),
                f"{title} electrode OCP entropic change [V.K-1]": 0.0,
            }
        )
        for j, row in enumerate(e["rows"]):
            for key, value in zip(("X", "U0", "w", "j0_ref", "a"), row, strict=True):
                updates[f"{key}_{short}_{j}"] = value
        u = model.variables[f"X-averaged {domain} particle potential [V]"]
        us = model.variables[f"X-averaged {domain} particle surface potential [V]"]
        param = getattr(model.param, short).prim
        base = f"qualification:{domain}:"
        model.variables[base + "potential"] = u
        model.variables[base + "surface_potential"] = us
        model.variables[base + "profile"] = model.variables[
            f"X-averaged {domain} particle stoichiometry"
        ]
        model.variables[base + "source_surface"] = model.variables[
            f"X-averaged {domain} particle surface stoichiometry"
        ]
        model.variables[base + "surface"] = param.x(us, pybamm.Scalar(data["temperature_K"]))
        model.variables[base + "eta"] = model.variables[
            f"X-averaged {domain} electrode reaction overpotential [V]"
        ]
        for j in range(len(e["rows"])):
            model.variables[base + f"kinetic_x{j}"] = param.x_j(
                us, pybamm.Scalar(data["temperature_K"]), j
            )
        if e["profile_amplitude_V"]:
            # Only initial data change. Actual released MSMR equations and boundaries remain.
            variable = next(v for v in model.initial_conditions if v.name == u.name)
            radius = pybamm.SpatialVariable(
                "r_" + short,
                domain=variable.domain,
                auxiliary_domains={"secondary": "current collector"},
                coord_sys="spherical polar",
            )
            model.initial_conditions[variable] = initial_u + e["profile_amplitude_V"] * pybamm.cos(
                np.pi * radius / e["radius_m"]
            )
        surface = model.variables[base + "surface"]
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
    voltage = model.variables["Voltage [V]"]
    model.events = [
        e for e in model.events if e.name not in {"Minimum voltage [V]", "Maximum voltage [V]"}
    ]
    model.events.extend(
        [
            pybamm.Event("battery:minimum-voltage", 1e4 * (voltage - data["minimum_voltage_V"])),
            pybamm.Event("battery:maximum-voltage", 1e4 * (data["maximum_voltage_V"] - voltage)),
        ]
    )
    points = {var: cells if str(var).startswith("r_") else 5 for var in model.default_var_pts}
    return pybamm.Simulation(
        model,
        parameter_values=parameters,
        var_pts=points,
        solver=pybamm.CasadiSolver(mode="safe", rtol=RTOL, atol=ATOL, root_tol=ATOL),
    )


def extract_segment(solution, data):
    times = np.asarray(solution.t)
    values, raw = {}, {}

    def scalar(key, name, factor=1.0):
        array = np.asarray(solution[name](times))
        if array.size != times.size:
            raise ValueError("reference scalar shape mismatch: " + name)
        values[key] = (array.reshape(-1) * factor).tolist()

    for key, name, factor in (
        ("voltage_V", "Voltage [V]", 1.0),
        ("current_A", "Current [A]", 1.0),
        ("charge_C", "Discharge capacity [A.h]", 3600.0),
        ("work_J", "Discharge energy [W.h]", 3600.0),
    ):
        scalar(key, name, factor)
    for e in data["electrodes"]:
        domain, short = e["name"], e["name"][0]
        base, area = f"qualification:{domain}:", 3 * e["active_volume_m3"] / e["radius_m"]
        for key in ("surface", "source_surface", "surface_potential", "eta"):
            scalar(domain + ":" + key, base + key)
        for j in range(len(e["rows"])):
            scalar(domain + f":kinetic_x{j}", base + f"kinetic_x{j}")
            scalar(domain + f":source_x{j}", f"X-averaged surface x_{short}_{j}")
            scalar(domain + f":current{j}_A", f"X-averaged j_{short}_{j} [A.m-2]", area)
        variable = solution[base + "profile"]
        radii = np.asarray(variable.mesh.nodes)
        theta = np.asarray(variable(times, r=radii)).reshape(len(radii), len(times)).T
        u = (
            np.asarray(solution[base + "potential"](times, r=radii))
            .reshape(len(radii), len(times))
            .T
        )
        weights = np.diff(np.asarray(variable.mesh.edges) ** 3) / e["radius_m"] ** 3
        average = theta @ weights
        values[domain + ":average"] = average.tolist()
        values[domain + ":amount_mol"] = (
            average * e["maximum_concentration_mol_m3"] * e["active_volume_m3"]
        ).tolist()
        raw[domain] = {
            "r_over_R": (radii / e["radius_m"]).tolist(),
            "theta": theta.tolist(),
            "potential_V": u.tolist(),
            "shell_weights": weights.tolist(),
            "sampled_theta": [
                np.interp(PROFILE_RADII, radii / e["radius_m"], x).tolist() for x in theta
            ],
            "sampled_potential_V": [
                np.interp(PROFILE_RADII, radii / e["radius_m"], x).tolist() for x in u
            ],
        }
    return {
        "times_s": times.tolist(),
        "values": values,
        "raw_profiles": raw,
        "end_time_s": float(times[-1]),
        "termination": solution.termination,
    }


def reference_time(data, cells):
    simulation, segments = build_reference(data, cells), []
    for duration, target, unit in data["waveform"]:
        solution = simulation.step(
            duration,
            save=False,
            t_eval=np.linspace(0.0, duration, data["samples_per_segment"]),
            inputs={"MSMR voltage target" if unit == "V" else "Current function [A]": target},
        )
        segments.append(extract_segment(solution, data))
        if solution.termination != "final time":
            break
    return {
        "implementation": "Actual PyBaMM25.4.2 SPM / three MSMR options / algebraic surface",
        "radial_cells": cells,
        "segments": segments,
        "profile_r_over_R": PROFILE_RADII.tolist(),
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ATOL},
        "initial_inventory_definition": "Actual raw node occupancy times spherical shell volume",
        "surface_semantics": "Kinetic x_j(surf(U)) and original surf(x_j(U)) retained separately",
    }


def static_reference_values(data, electrode, temperature, u, eta, ce, *, simulation=None):
    """Evaluate actual released source symbolic functions, not a copied numerical model."""
    import pybamm

    sim = simulation if simulation is not None else build_reference(data, 8)
    parameter = getattr(sim.model.param, electrode["name"][0]).prim
    U, T, C = map(pybamm.Scalar, (u, temperature, ce))
    result = {"x": [], "dx": [], "j0": [], "j": []}
    for j, _row in enumerate(electrode["rows"]):
        x, dx, j0 = parameter.x_j(U, T, j), parameter.dxdU_j(U, T, j), parameter.j0_j(C, U, T, j)
        alpha = parameter.alpha_bv_j(T, j)
        factor = sim.model.param.F / (sim.model.param.R * T)
        current = j0 * (pybamm.exp((1 - alpha) * factor * eta) - pybamm.exp(-alpha * factor * eta))
        for key, expression in zip(("x", "dx", "j0", "j"), (x, dx, j0, current), strict=True):
            result[key].append(float(sim.parameter_values.evaluate(expression)))
    return result


def independent_bv(rows, u, eta, temperature, ce):
    from battery_msmr_cases import F, R

    factor = F / (R * temperature)
    x = occupancies(rows, u, temperature)
    dx, j0, j = [], [], []
    for row, occupancy in zip(rows, x, strict=True):
        weight, u0, width, exchange, alpha = row
        dx.append(-factor / width * occupancy * (1 - occupancy / weight))
        rate = (
            exchange
            * occupancy**width
            * np.exp(factor * (1 - alpha) * (u - u0))
            * (ce / 1000.0) ** (1 - alpha)
        )
        j0.append(rate)
        j.append(
            rate * np.expm1((1 - alpha) * factor * eta) - rate * np.expm1(-alpha * factor * eta)
        )
    return {"x": x.tolist(), "dx": dx, "j0": j0, "j": j}
