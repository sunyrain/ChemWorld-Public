"""Released positive-working SPM with explicit finite-metal validation mapping."""

import numpy as np
from battery_half_cell_cases import (
    ATOL,
    EXPONENTIAL,
    POLYNOMIAL,
    RESPONSES,
    RTOL,
    F,
    initial_metal,
)

PROFILE_RADII = np.linspace(.05, .95, 9)


def build_reference(data, radial_cells):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("half-cell reference requires PyBaMM25.4.2")
    cv = data["waveform"][0][2] == "V"

    def finite_source(variables):
        current = variables["Current [A]"]
        # Fixed equation scaling Imax/(1 V), not a second dynamic controller.
        requested = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("half-cell voltage target"))
        return current - pybamm.maximum(-data["maximum_current_A"], pybamm.minimum(
            data["maximum_current_A"], requested))

    model = pybamm.lithium_ion.SPM({"working electrode": "positive",
        "surface form": "false", "particle": "Fickian diffusion", "thermal": "isothermal",
        "SEI": "none", "lithium plating": "none", "contact resistance": "false",
        **({"operating mode": finite_source} if cv else {})})
    parameters = pybamm.ParameterValues("Xu2019")

    def compensated_ocv(theta):
        amplitude, scale, shift = EXPONENTIAL
        return (sum(c * theta**i for i, c in enumerate(POLYNOMIAL))
                + amplitude * pybamm.exp(scale * (theta - shift))
                - 1e-6 * (1 / theta + 1 / (theta - 1)))

    def working_exchange(c_e, c_s, c_max, temperature):
        theta = c_s / c_max
        return data["exchange_current_scale_A_m2"] * pybamm.sqrt(theta * (1 - theta))

    def metal_exchange(c_e, c_li, temperature):
        # Evaluated Xu-derived scalar input at the declared fixed ce,T.
        return pybamm.Scalar(data["exchange_current_density_A_m2"])

    parameters.update({
        "Electrode height [m]": data["area_m2"], "Electrode width [m]": 1.,
        "Number of electrodes connected in parallel to make a cell": 1.,
        "Negative electrode thickness [m]": data["thickness_m"],
        "Negative electrode conductivity [S.m-1]": data["conductivity_S_m"],
        "Lithium metal partial molar volume [m3.mol-1]": data["partial_molar_volume_m3_mol"],
        "Exchange-current density for lithium metal electrode [A.m-2]": metal_exchange,
        "Positive electrode thickness [m]": data["active_volume_m3"] / (.518 * data["area_m2"]),
        "Positive electrode active material volume fraction": .518,
        "Positive particle radius [m]": data["radius_m"],
        "Positive particle diffusivity [m2.s-1]": data["diffusivity_m2_s"],
        "Maximum concentration in positive electrode [mol.m-3]":
            data["maximum_concentration_mol_m3"],
        "Initial concentration in positive electrode [mol.m-3]":
            data["maximum_concentration_mol_m3"] * data["initial_positive_fraction"],
        "Positive electrode OCP [V]": compensated_ocv,
        "Positive electrode OCP entropic change [V.K-1]": 0.,
        "Positive electrode exchange-current density [A.m-2]": working_exchange,
        "Initial concentration in electrolyte [mol.m-3]":
            data["electrolyte_concentration_mol_m3"],
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Reference temperature [K]": data["temperature_K"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Current function [A]": "[input]",
    })
    current = model.variables["Current [A]"]
    voltage = model.variables["Voltage [V]"]
    surface = model.variables["X-averaged positive particle surface concentration"]
    metal = initial_metal(data) - 3600 * model.variables["Discharge capacity [A.h]"] / F
    model.variables["qualification:metal amount [mol]"] = metal
    model.variables["qualification:metal ohmic [V]"] = current * data["thickness_m"] / (
        data["conductivity_S_m"] * data["area_m2"])
    # Exact physical zero faces, scaled to control the source solver's root displacement.
    model.events = [event for event in model.events if event.name not in
                    {"Minimum voltage [V]", "Maximum voltage [V]"}]
    model.events.extend([
        pybamm.Event("half-cell:metal:depleted", 1e4 * metal / initial_metal(data)),
        pybamm.Event("half-cell:working:depleted", 1e4 * (surface - data["surface_margin"])),
        pybamm.Event("half-cell:working:saturated", 1e4 * (1 - data["surface_margin"] - surface)),
        pybamm.Event("half-cell:minimum-voltage", 1e4 * (voltage - data["minimum_voltage_V"])),
        pybamm.Event("half-cell:maximum-voltage", 1e4 * (data["maximum_voltage_V"] - voltage)),
    ])
    points = {var: radial_cells if str(var) == "r_p" else 5
              for var in model.default_var_pts}
    return pybamm.Simulation(model, parameter_values=parameters, var_pts=points,
        solver=pybamm.CasadiSolver(mode="safe", rtol=RTOL, atol=ATOL, root_tol=ATOL))


def output_names():
    return ("Voltage [V]", "Current [A]", "X-averaged positive particle surface concentration",
        "Average positive particle concentration", "qualification:metal amount [mol]",
        "Discharge capacity [A.h]", "Lithium metal interface reaction overpotential [V]",
        "X-averaged positive electrode reaction overpotential [V]", "qualification:metal ohmic [V]")


def reference_time(data, cells):
    simulation, segments = build_reference(data, cells), []
    for duration, target, unit in data["waveform"]:
        solution = simulation.step(duration, save=False,
            t_eval=np.linspace(0., duration, data["samples_per_segment"]),
            inputs={"half-cell voltage target" if unit == "V" else "Current function [A]": target})
        times = np.asarray(solution.t)
        columns = []
        for key in output_names():
            values = np.asarray(solution[key](times))
            if values.size != times.size:
                raise ValueError("reference scalar shape mismatch: " + key)
            columns.append(values.reshape(-1))
        rows = np.column_stack(columns)
        rows[:, 5] *= 3600.
        profile = np.asarray(solution["X-averaged positive particle concentration [mol.m-3]"](
            float(times[-1]), r=PROFILE_RADII * data["radius_m"])).reshape(-1)
        if profile.size != PROFILE_RADII.size:
            raise ValueError("reference profile shape mismatch")
        segments.append({"times_s": times.tolist(), "rows": rows.tolist(),
            "end_time_s": float(times[-1]), "termination": solution.termination,
            "profile_theta": (profile / data["maximum_concentration_mol_m3"]).tolist()})
        if solution.termination != "final time":
            break
    return {"implementation": "PyBaMM25.4.2 positive-working SPM + finite-metal mapping",
        "response_order": RESPONSES, "radial_cells": cells, "segments": segments,
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ATOL},
        "profile_r_over_R": PROFILE_RADII.tolist()}
