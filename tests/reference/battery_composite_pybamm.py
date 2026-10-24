"""Actual released multiphase SPM, with independently declared material mappings."""

from copy import deepcopy

import numpy as np
from battery_composite_cases import ATOL, PROFILE_RADII, RESPONSES, RTOL


def source_phases(data, single=False):
    values = deepcopy(data["phases"])
    if single:
        if not data.get("same_material_limit"):
            raise ValueError("single-material comparison is declared only for CP01")
        values[0]["active_volume_m3"] += values[1]["active_volume_m3"]
        return [(values[0], "", ""), (values[2], "", "")]
    return [(values[0], "Primary: ", "primary "),
            (values[1], "Secondary: ", "secondary "), (values[2], "", "")]


def build_reference(data, radial_cells, *, single=False):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("composite reference requires PyBaMM25.4.2")

    def finite_source(variables):
        current = variables["Current [A]"]
        requested = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("composite voltage target"))
        return current - pybamm.maximum(-data["maximum_current_A"], pybamm.minimum(
            data["maximum_current_A"], requested))

    cv = data["waveform"][0][2] == "V"
    options = {"particle phases": ("1", "1") if single else ("2", "1"),
        "surface form": "algebraic", "open-circuit potential": "single",
        "particle": "Fickian diffusion", "thermal": "isothermal",
        "SEI": "none", "lithium plating": "none", "contact resistance": "true",
        "calculate discharge energy": "true", **({"operating mode": finite_source} if cv else {})}
    model = pybamm.lithium_ion.SPM(options)
    parameters = pybamm.ParameterValues("Chen2020" if single else "Chen2020_composite")
    updates = {
        "Electrode height [m]": data["collector_area_m2"], "Electrode width [m]": 1.,
        "Number of electrodes connected in parallel to make a cell": 1.,
        "Negative electrode thickness [m]": data["electrode_thickness_m"],
        "Positive electrode thickness [m]": data["electrode_thickness_m"],
        "Contact resistance [Ohm]": data["series_resistance_ohm"],
        "Initial concentration in electrolyte [mol.m-3]":
            data["electrolyte_concentration_mol_m3"],
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Reference temperature [K]": data["temperature_K"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Current function [A]": "[input]",
    }
    phases = source_phases(data, single)
    for value, prefix, label in phases:
        domain = value["electrode"]
        title = domain.capitalize()

        def compensated_ocv(theta, coefficients=value["polynomial"]):
            return (sum(c * theta**i for i, c in enumerate(coefficients))
                    - 1e-6 * (1 / theta + 1 / (theta - 1)))

        def exchange(c_e, c_s, c_max, temperature, scale=value["exchange_current_scale_A_m2"]):
            theta = c_s / c_max
            return scale * pybamm.sqrt(theta * (1 - theta))

        updates.update({
            prefix + f"{title} particle radius [m]": value["radius_m"],
            prefix + f"{title} particle diffusivity [m2.s-1]": value["diffusivity_m2_s"],
            prefix + f"Maximum concentration in {domain} electrode [mol.m-3]":
                value["maximum_concentration_mol_m3"],
            prefix + f"Initial concentration in {domain} electrode [mol.m-3]":
                value["maximum_concentration_mol_m3"] * value["initial_fraction"],
            prefix + f"{title} electrode active material volume fraction":
                value["active_volume_m3"] / (
                    data["collector_area_m2"] * data["electrode_thickness_m"]),
            prefix + f"{title} electrode exchange-current density [A.m-2]": exchange,
            prefix + f"{title} electrode OCP [V]": compensated_ocv,
            prefix + f"{title} electrode OCP entropic change [V.K-1]": 0.,
        })
        base = f"qualification:{value['phase_id']}:"
        surface = model.variables[f"X-averaged {domain} {label}particle surface concentration"]
        average = model.variables[f"Average {domain} {label}particle concentration"]
        current_name = f"X-averaged {domain} electrode {label}interfacial current density [A.m-2]"
        model.variables[base + "q"] = (
            model.variables[current_name]
            * 3 * value["active_volume_m3"] / value["radius_m"])
        model.variables[base + "surface"] = surface
        model.variables[base + "average"] = average
        model.variables[base + "amount"] = (
            average * value["maximum_concentration_mol_m3"] * value["active_volume_m3"])
        model.variables[base + "eta"] = model.variables[
            f"X-averaged {domain} electrode {label}reaction overpotential [V]"]
        model.variables[base + "profile"] = model.variables[
            f"X-averaged {domain} {label}particle concentration [mol.m-3]"]
        model.events.extend([
            pybamm.Event(f"battery:{domain}:phase:{value['phase_id']}:depleted",
                         1e4 * (surface - data["surface_margin"])),
            pybamm.Event(f"battery:{domain}:phase:{value['phase_id']}:saturated",
                         1e4 * (1 - data["surface_margin"] - surface)),
        ])
    # The secondary material has no single-valued source OCP; it is explicitly synthetic.
    parameters.update(updates, check_already_exists=False)
    voltage = model.variables["Voltage [V]"]
    model.events = [event for event in model.events if event.name not in
                    {"Minimum voltage [V]", "Maximum voltage [V]"}]
    model.events.extend([
        pybamm.Event("battery:minimum-voltage", 1e4 * (voltage - data["minimum_voltage_V"])),
        pybamm.Event("battery:maximum-voltage", 1e4 * (data["maximum_voltage_V"] - voltage)),
    ])
    points = {var: radial_cells if str(var).startswith("r_") else 5
              for var in model.default_var_pts}
    return pybamm.Simulation(model, parameter_values=parameters, var_pts=points,
        solver=pybamm.CasadiSolver(mode="safe", rtol=RTOL, atol=ATOL, root_tol=ATOL))


def output_names(data, single=False):
    return ("Voltage [V]", "Current [A]", "Discharge capacity [A.h]", "Discharge energy [W.h]",
        *(f"qualification:{value['phase_id']}:{key}" for value, _, _ in source_phases(data, single)
          for key in ("q", "surface", "average", "amount", "eta")))


def reference_time(data, cells, *, single=False):
    simulation, segments = build_reference(data, cells, single=single), []
    for duration, target, unit in data["waveform"]:
        solution = simulation.step(duration, save=False,
            t_eval=np.linspace(0., duration, data["samples_per_segment"]),
            inputs={"composite voltage target" if unit == "V" else "Current function [A]": target})
        times = np.asarray(solution.t)
        columns = []
        for name in output_names(data, single):
            values = np.asarray(solution[name](times))
            if values.size != times.size:
                raise ValueError("reference scalar output shape mismatch: " + name)
            columns.append(values.reshape(-1))
        rows = np.column_stack(columns)
        rows[:, 2:4] *= 3600
        profiles = []
        for value, _, _ in source_phases(data, single):
            profile = np.asarray(solution[f"qualification:{value['phase_id']}:profile"](
                float(times[-1]), r=PROFILE_RADII * value["radius_m"])).reshape(-1)
            if profile.size != PROFILE_RADII.size:
                raise ValueError("reference phase profile shape mismatch")
            profiles.append((profile / value["maximum_concentration_mol_m3"]).tolist())
        segments.append({"times_s": times.tolist(), "rows": rows.tolist(),
            "end_time_s": float(times[-1]), "termination": solution.termination,
            "profiles_theta": profiles})
        if solution.termination != "final time":
            break
    return {"implementation": "PyBaMM25.4.2 actual single SPM" if single else
            "PyBaMM25.4.2 actual multiphase SPM, surface form algebraic",
        "radial_cells": cells, "segments": segments, "single_material": single,
        "response_order": (*RESPONSES[:9], *RESPONSES[14:]) if single else RESPONSES,
        "profile_r_over_R": PROFILE_RADII.tolist(),
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ATOL}}
