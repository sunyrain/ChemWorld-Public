"""Actual PyBaMM 25.4.2 1+1D potential-pair SPM reference builder.

Finite CV is an explicitly authored algebraic external boundary condition. The source
Voltage and all original termination events remain unchanged; no Green voltage is inserted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from battery_collector_spm_cases import (
    ATOL,
    CASADI_VERSION,
    PYBAMM_VERSION,
    ROOT_TOL,
    RTOL,
    _z_fractions,
)


def _model_options(data=None):
    import pybamm

    def finite_source(variables):
        current = variables["Current [A]"]
        requested = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("Collector voltage target [V]")
        )
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], requested)
        )

    modes = {stage[2] for stage in data["waveform"]} if data else {"A"}
    if modes not in ({"A"}, {"V"}):
        raise ValueError("collector source requires one declared control mode per history")
    return {
        "current collector": "potential pair",
        "dimensionality": 1,
        "surface form": "algebraic",
        "thermal": "isothermal",
        "SEI": "none",
        "lithium plating": "none",
        "particle": "Fickian diffusion",
        "particle phases": "1",
        "particle size": "single",
        "open-circuit potential": "single",
        "calculate discharge energy": "true",
        "operating mode": finite_source if modes == {"V"} else "current",
    }


def _parameter_updates(data: dict[str, Any], layout: dict[str, str], grid: dict[str, int]):
    import pybamm

    area = data["width_m"] * data["height_m"]
    negative_tab = 0.0 if layout["negative_tab"] == "left" else data["height_m"]
    positive_tab = 0.0 if layout["positive_tab"] == "left" else data["height_m"]
    updates: dict[str, Any] = {
        "Electrode width [m]": data["width_m"],
        "Electrode height [m]": data["height_m"],
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Negative current collector thickness [m]": data["collector_thickness_m"],
        "Positive current collector thickness [m]": data["collector_thickness_m"],
        "Negative current collector conductivity [S.m-1]": data["sigma_cc_S_m"],
        "Positive current collector conductivity [S.m-1]": data["sigma_cc_S_m"],
        "Negative tab centre z-coordinate [m]": negative_tab,
        "Positive tab centre z-coordinate [m]": positive_tab,
        "Initial concentration in electrolyte [mol.m-3]": data["electrolyte_concentration_mol_m3"],
        "Reference temperature [K]": data["temperature_K"],
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Current function [A]": "[input]",
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Negative electrode thickness [m]": data["lengths_m"][0],
        "Separator thickness [m]": data["lengths_m"][1],
        "Positive electrode thickness [m]": data["lengths_m"][2],
        "Negative electrode porosity": data["porosities"][0],
        "Separator porosity": data["porosities"][1],
        "Positive electrode porosity": data["porosities"][2],
    }
    for name, electrode, region in (
        ("Negative", data["negative"], 0),
        ("Positive", data["positive"], 2),
    ):
        domain = name.lower()

        # Keep the source variable's reference/scale scalar; only its actual IC below
        # carries a z field. The scalar initial-Li metadata uses the true FV mean.
        fractions = _z_fractions(data["initial_z_fractions"][domain], grid["z"])

        def ocv(theta, coefficients=electrode["ocv"], barrier=data["ocp_barrier_V"]):
            return (
                coefficients[0] + coefficients[1] * theta - barrier * (1 / theta + 1 / (theta - 1))
            )

        def exchange(
            c_e,
            c_s,
            c_max,
            temperature,
            scale=electrode["exchange_current_scale_A_m2"],
        ):
            theta = c_s / c_max
            return scale * pybamm.sqrt(
                c_e / data["electrolyte_concentration_mol_m3"] * theta * (1 - theta)
            )

        updates.update(
            {
                f"{name} particle radius [m]": electrode["radius_m"],
                f"{name} electrode diffusivity [m2.s-1]": electrode["diffusivity_m2_s"],
                f"Maximum concentration in {domain} electrode [mol.m-3]": electrode[
                    "maximum_concentration_mol_m3"
                ],
                f"Initial concentration in {domain} electrode [mol.m-3]": (
                    electrode["maximum_concentration_mol_m3"] * float(np.mean(fractions))
                ),
                f"{name} electrode active material volume fraction": electrode["active_volume_m3"]
                / (area * data["lengths_m"][region]),
                f"{name} electrode exchange-current density [A.m-2]": exchange,
                f"{name} electrode OCP [V]": ocv,
                f"{name} electrode OCP entropic change [V.K-1]": 0.0,
            }
        )
    return updates


def build_reference(data: dict[str, Any], grid: dict[str, int], layout: dict[str, str]):
    """Build and discretise the released source model; no integration is performed."""
    import pybamm

    if pybamm.__version__ != PYBAMM_VERSION:
        raise RuntimeError(f"collector source requires PyBaMM {PYBAMM_VERSION}")
    if layout not in (
        {"negative_tab": "left", "positive_tab": "right"},
        {"negative_tab": "right", "positive_tab": "right"},
    ):
        raise ValueError("only positive-right and negative-left/right tab layouts are supported")
    options = _model_options(data)
    model = pybamm.lithium_ion.SPM(options=options)
    # Replace only the authored initial condition, not a source RHS/BC/observable.
    # PyBaMM's particle variable reference uses scalar c_init and is not itself a
    # spatially discretised expression; using the profile as that reference is invalid.
    z_centres = (np.arange(grid["z"]) + 0.5) * data["height_m"] / grid["z"]
    for name in ("negative", "positive"):
        fractions = _z_fractions(data["initial_z_fractions"][name], grid["z"])
        profile = pybamm.Interpolant(
            np.r_[0.0, z_centres, data["height_m"]],
            data[name]["maximum_concentration_mol_m3"]
            * np.r_[fractions[0], fractions, fractions[-1]],
            pybamm.standard_spatial_vars.z,
            interpolator="linear",
        )
        variable = model.variables[f"X-averaged {name} particle concentration [mol.m-3]"]
        model.initial_conditions[variable] = pybamm.PrimaryBroadcast(profile, f"{name} particle")
    values = pybamm.ParameterValues("Marquis2019")
    values.update(_parameter_updates(data, layout, grid), check_already_exists=False)
    geometry = pybamm.battery_geometry(options=options)
    values.process_geometry(geometry)
    submesh_types = dict.fromkeys(geometry, pybamm.Uniform1DSubMesh)
    var_pts = {
        "x_n": 1,
        "x_s": 1,
        "x_p": 1,
        "r_n": int(grid["r"]),
        "r_p": int(grid["r"]),
        "z": int(grid["z"]),
    }
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    spatial_methods = {key: pybamm.FiniteVolume() for key in geometry}
    discretisation = pybamm.Discretisation(mesh, spatial_methods)
    values.process_model(model)
    discretisation.process_model(model)
    return {
        "model": model,
        "parameter_values": values,
        "geometry": geometry,
        "mesh": mesh,
        "grid": dict(grid),
        "layout": dict(layout),
        "control": "finite-CV-source-boundary" if data["waveform"][0][2] == "V" else "CC",
        "solver": {"name": "IDAKLU", "rtol": RTOL, "atol": ATOL, "root_tol": ROOT_TOL},
    }


def source_contract_snapshot(built: dict[str, Any]) -> dict[str, Any]:
    """Return stable build/discretise evidence, retaining source boundary semantics."""
    model = built["model"]
    mesh = built["mesh"]
    cc_mesh = mesh["current collector"]
    variable_names = (
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Current collector current density [A.m-2]",
        "Negative electrode current density [A.m-2]",
        "Positive electrode current density [A.m-2]",
        "Local voltage [V]",
        "Terminal voltage [V]",
        "Voltage [V]",
    )
    variables = {}
    for name in variable_names:
        symbol = model.variables[name]
        variables[name] = {
            "shape": list(symbol.shape),
            "domain": list(symbol.domain),
            "secondary_domain": list(symbol.secondary_domain),
        }
    return {
        "pybamm_version": PYBAMM_VERSION,
        "casadi_version": CASADI_VERSION,
        "layout": built["layout"],
        "grid": built["grid"],
        "tabs": dict(cc_mesh.tabs),
        "z_nodes_m": cc_mesh.nodes.tolist(),
        "z_edges_m": cc_mesh.edges.tolist(),
        "rhs_count": len(model.rhs),
        "algebraic_count": len(model.algebraic),
        "initial_condition_count": len(model.initial_conditions),
        "states": [
            {
                "name": variable.name,
                "slices": [[part.start, part.stop] for part in model.y_slices[variable]],
                "scale": float(built["parameter_values"].evaluate(variable.scale)),
                "reference": float(built["parameter_values"].evaluate(variable.reference)),
                "domains": {key: list(domain) for key, domain in variable.domains.items()},
                "kind": "differential" if variable in model.rhs else "algebraic",
            }
            for variable in (*model.rhs, *model.algebraic)
        ],
        "control": built["control"],
        "discharge_energy": "actual signed PyBaMM energy observer, dW/dt=I*source Voltage",
        "events": [{"name": event.name, "type": event.event_type.name} for event in model.events],
        "variables": variables,
        "source_boundary_voltage": {
            "definition": (
                "Voltage [V] = positive-tab boundary value of positive collector "
                "potential minus contact drop"
            ),
            "boundary_extrapolation": (
                "finite-volume linear boundary reconstruction retained by PyBaMM source"
            ),
            "independent_physical_tab_voltage": (
                "not projected into source; reserved for fixed scientific reference Green mapping"
            ),
        },
    }
