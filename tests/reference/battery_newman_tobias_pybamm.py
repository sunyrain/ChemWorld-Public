"""Actual released NewmanTobias model; no production electrical solver imports."""

import numpy as np
from battery_newman_tobias_cases import (
    ATOL,
    PROFILE_POINTS,
    RTOL,
    SOURCE_ALG_ATOL,
    SOURCE_ROOT_TOL,
)


def build_reference(data, cells):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("NT reference requires PyBaMM25.4.2")

    def finite_source(variables):
        current = variables["Current [A]"]
        request = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("NT voltage target")
        )
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], request)
        )

    model = pybamm.lithium_ion.NewmanTobias(
        {
            "particle": "uniform profile",
            "surface form": "false",
            "thermal": "isothermal",
            "dimensionality": 0,
            "SEI": "none",
            "lithium plating": "none",
            "calculate discharge energy": "true",
            "contact resistance": "false",
            **({"operating mode": finite_source} if data["waveform"][0][2] == "V" else {}),
        }
    )
    values = pybamm.ParameterValues("Chen2020")
    area = data["area_m2"]
    updates = {
        "Electrode width [m]": 0.1,
        "Electrode height [m]": area / 0.1,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Initial concentration in electrolyte [mol.m-3]": data["electrolyte_concentration_mol_m3"],
        "Initial temperature [K]": data["temperature_K"],
        "Reference temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Electrolyte conductivity [S.m-1]": data["conductivity_S_m"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Current function [A]": "[input]",
    }
    for i, name in enumerate(("Negative electrode", "Separator", "Positive electrode")):
        updates.update(
            {
                name + " thickness [m]": data["lengths_m"][i],
                name + " porosity": data["porosities"][i],
                name + " Bruggeman coefficient (electrolyte)": data["bruggeman"][i],
            }
        )
    for i, e in enumerate(data["electrodes"]):
        domain, name = e["name"], e["name"].capitalize()
        region = 0 if i == 0 else 2

        def ocv(theta, coefficients=e["polynomial"]):
            return sum(a * theta**i for i, a in enumerate(coefficients)) - 1e-6 * (
                1 / theta + 1 / (theta - 1)
            )

        def exchange(ce, cs, cmax, temperature, scale=e["exchange_current_scale_A_m2"]):
            theta = cs / cmax
            return scale * pybamm.sqrt(
                ce / data["reference_electrolyte_concentration_mol_m3"] * theta * (1 - theta)
            )

        updates.update(
            {
                name + " particle radius [m]": e["radius_m"],
                f"Maximum concentration in {domain} electrode [mol.m-3]": e[
                    "maximum_concentration_mol_m3"
                ],
                f"Initial concentration in {domain} electrode [mol.m-3]": e[
                    "maximum_concentration_mol_m3"
                ]
                * e["initial_fraction"],
                name + " electrode active material volume fraction": e["active_volume_m3"]
                / (area * data["lengths_m"][region]),
                name + " electrode conductivity [S.m-1]": data["solid_conductivity_S_m"][i]
                / (1 - data["porosities"][region]) ** data["solid_bruggeman"][i],
                name + " electrode Bruggeman coefficient (electrode)": data["solid_bruggeman"][i],
                name + " electrode exchange-current density [A.m-2]": exchange,
                name + " electrode OCP [V]": ocv,
                name + " electrode OCP entropic change [V.K-1]": 0.0,
            }
        )
        mean = model.variables[f"Average {domain} particle concentration [mol.m-3]"]
        theta = mean / e["maximum_concentration_mol_m3"]
        model.variables[f"qualification:{domain}:fraction"] = theta
        model.variables[f"qualification:{domain}:amount"] = mean * e["active_volume_m3"]
        model.events.extend(
            [
                pybamm.Event(f"battery:{domain}:depleted", 1e4 * (theta - data["surface_margin"])),
                pybamm.Event(
                    f"battery:{domain}:saturated", 1e4 * (1 - data["surface_margin"] - theta)
                ),
            ]
        )
    values.update(updates, check_already_exists=False)
    var_points = dict.fromkeys(model.default_var_pts, 5)
    var_points.update(
        {
            getattr(pybamm.standard_spatial_vars, "x_" + name): count
            for name, count in zip(("n", "s", "p"), cells, strict=True)
        }
    )
    simulation = pybamm.Simulation(
        model,
        parameter_values=values,
        var_pts=var_points,
        solver=pybamm.IDAKLUSolver(rtol=RTOL, atol=ATOL, root_tol=SOURCE_ROOT_TOL),
    )
    simulation.build()
    built = simulation.built_model
    simulation.solver.atol = np.r_[
        np.full(built.len_rhs, ATOL), np.full(built.len_alg, SOURCE_ALG_ATOL)
    ]
    return simulation


def _time_rows(solution, name, times):
    # entries are actual source nodes/faces; default 1D __call__ adds ghost points.
    array = np.asarray(solution[name].entries, dtype=float)
    return array.reshape(-1, len(times)).T


def extract_segment(solution, data, cells):
    times = solution.t
    scalar_names = [
        "Voltage [V]",
        "Current [A]",
        "Discharge capacity [A.h]",
        "Discharge energy [W.h]",
        "qualification:negative:fraction",
        "qualification:positive:fraction",
        "qualification:negative:amount",
        "qualification:positive:amount",
    ]
    values = np.column_stack([_time_rows(solution, name, times)[:, 0] for name in scalar_names])
    values[:, 2:4] *= 3600
    raw, sampled = {}, {}
    starts = np.cumsum((0.0, *data["lengths_m"]))
    phi_e = _time_rows(solution, "Electrolyte potential [V]", times)
    liquid = _time_rows(solution, "Electrolyte current density [A.m-2]", times) * data["area_m2"]
    all_nodes = np.asarray(solution["Electrolyte potential [V]"].mesh.nodes)
    all_edges = np.asarray(solution["Electrolyte current density [A.m-2]"].mesh.edges)
    node_offset = 0
    for region, (name, count) in enumerate(
        zip(("negative", "separator", "positive"), cells, strict=True)
    ):
        nodes = all_nodes[node_offset : node_offset + count]
        edges = all_edges[node_offset : node_offset + count + 1]
        xi = (nodes - starts[region]) / data["lengths_m"][region]
        xf = (edges - starts[region]) / data["lengths_m"][region]
        raw[name] = {
            "nodes_m": nodes.tolist(),
            "edges_m": edges.tolist(),
            "electrolyte_potential_V": phi_e[:, node_offset : node_offset + count].tolist(),
            "electrolyte_face_current_A": liquid[:, node_offset : node_offset + count + 1].tolist(),
        }
        sampled[name] = {
            "electrolyte_potential_V": [
                np.interp(PROFILE_POINTS, xi, row).tolist()
                for row in phi_e[:, node_offset : node_offset + count]
            ],
            "electrolyte_face_current_A": [
                np.interp(PROFILE_POINTS, xf, row).tolist()
                for row in liquid[:, node_offset : node_offset + count + 1]
            ],
        }
        if name != "separator":
            e = data["electrodes"][0 if region == 0 else 1]
            phi_s = _time_rows(solution, name.capitalize() + " electrode potential [V]", times)
            solid = (
                _time_rows(
                    solution, name.capitalize() + " electrode current density [A.m-2]", times
                )
                * data["area_m2"]
            )
            j = _time_rows(
                solution,
                name.capitalize() + " electrode interfacial current density [A.m-2]",
                times,
            )
            reaction = j * 3 * e["active_volume_m3"] / e["radius_m"]
            raw[name].update(
                solid_potential_V=phi_s.tolist(),
                solid_face_current_A=solid.tolist(),
                reaction_current_A=(reaction / count).tolist(),
                interfacial_current_density_A_m2=j.tolist(),
            )
            sampled[name].update(
                solid_potential_V=[np.interp(PROFILE_POINTS, xi, row).tolist() for row in phi_s],
                solid_face_current_A=[np.interp(PROFILE_POINTS, xf, row).tolist() for row in solid],
                reaction_density_A=[
                    np.interp(PROFILE_POINTS, xi, row).tolist() for row in reaction
                ],
            )
        node_offset += count
    return {
        "times_s": times.tolist(),
        "values": values.tolist(),
        "raw_profiles": raw,
        "sampled_profiles": sampled,
        "end_time_s": float(times[-1]),
        "termination": solution.termination,
    }


def reference_time(data, cells):
    progress = {"completed_segments": [], "stage": "build", "current_time_s": 0.0}
    try:
        return _reference_time(data, cells, progress)
    except Exception as exc:
        exc.partial_history = progress
        raise


def _reference_time(data, cells, progress):
    simulation, segments = build_reference(data, cells), progress["completed_segments"]
    for index, (duration, target, unit) in enumerate(data["waveform"]):
        start = sum(part[0] for part in data["waveform"][:index])
        progress.update(
            stage="integrate",
            current_time_s=start,
            active_segment={
                "index": index,
                "start_time_s": start,
                "duration_s": duration,
                "target": target,
                "unit": unit,
                "requested_end_time_s": start + duration,
            },
        )
        samples = np.linspace(0.0, duration, data["samples_per_segment"])
        solution = simulation.step(
            duration,
            save=False,
            t_eval=samples,
            t_interp=samples,
            inputs={"NT voltage target" if unit == "V" else "Current function [A]": target},
        )
        progress.update(stage="extract", current_time_s=float(solution.t[-1]))
        progress["active_segment"].update(
            returned_times_s=solution.t.tolist(),
            termination=solution.termination,
            returned_state_coordinates=np.asarray(solution.y).tolist(),
        )
        segments.append(extract_segment(solution, data, cells))
        if solution.termination != "final time":
            break
    return {
        "implementation": "Actual PyBaMM25.4.2 NewmanTobias uniform profile / surface false",
        "thickness_cells": list(cells),
        "segments": segments,
        "profile_fractional_coordinates": PROFILE_POINTS.tolist(),
        "solver": "PyBaMM25.4.2 IDAKLUSolver; CasADi3.6.7 expressions",
        "tolerances": {
            "rtol": RTOL,
            "differential_atol": ATOL,
            "algebraic_atol": SOURCE_ALG_ATOL,
            "root_tol": SOURCE_ROOT_TOL,
        },
        "potential_gauge": "negative current collector zero; original source potentials",
    }
