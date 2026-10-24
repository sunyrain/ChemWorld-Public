"""Native and independently discretized released/corrected diffusion responses."""

from dataclasses import replace

import numpy as np
from battery_frequency_pybamm import frequency_from_simulation
from thevenin_diffusion_cases import (
    CORRECTION_LABEL,
    CORRECTION_URL,
    FREQUENCIES,
    case_data,
    frequency_names,
    native_case,
    time_names,
)

from chemworld.physchem.battery_thevenin import (
    AMBIENT_LOSS,
    CELL_T,
    CHARGE,
    JIG_T,
    MODEL_HEAT,
    SOC,
    WORK,
    rc_key,
    thevenin_fields,
    thevenin_system,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram

INTEGRALS = ("charge_C", "work_J", "model_heat_J", "ambient_loss_J")


def reference_model(index, cells, variant, *, frequency=False):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("the fixed independent diffusion reference requires PyBaMM 25.4.2")
    if variant not in {"released", "corrected"}:
        raise ValueError("reference variant must explicitly identify released or corrected")
    data = case_data(index)
    count = len(data["rc_resistances_ohm"])
    is_cv = not frequency and data["segments"][0]["unit"] == "V"
    limit = data["maximum_current_A"]

    def finite_source(variables):
        current, voltage = variables["Current [A]"], variables["Voltage [V]"]
        requested = current+limit/data["maximum_voltage_V"]*(
            voltage-pybamm.InputParameter("Voltage target [V]"))
        return current-pybamm.maximum(-limit, pybamm.minimum(limit, requested))

    model = pybamm.equivalent_circuit.Thevenin(options={
        "number of rc elements": count, "diffusion element": "true",
        "operating mode": finite_source if is_cv else "current"}, build=False)
    if is_cv:
        model.submodels["external circuit"] = pybamm.external_circuit.FunctionControl(
            model.param, finite_source, model.options, control="algebraic")
    if variant == "corrected":
        class CorrectedDiffusion(pybamm.equivalent_circuit_elements.DiffusionElement):
            """Only PR5869's diffusion-overpotential sign correction."""

            def get_coupled_variables(self, variables):
                variables = super().get_coupled_variables(variables)
                variables["Diffusion overpotential [V]"] = (
                    self.param.ocv(variables["Surface SoC"])
                    - self.param.ocv(variables["SoC"]))
                return variables

        class CorrectedThermal(pybamm.equivalent_circuit_elements.ThermalSubModel):
            """Only PR5869's previously omitted diffusion heat contribution."""

            def get_coupled_variables(self, variables):
                variables = super().get_coupled_variables(variables)
                added = -variables["Current [A]"]*variables["Diffusion overpotential [V]"]
                variables["Irreversible heat generation [W]"] += added
                variables["Total heat generation [W]"] += added
                return variables

        model.submodels["Diffusion"] = CorrectedDiffusion(model.param, model.options)
        model.submodels["Thermal"] = CorrectedThermal(model.param, model.options)
    model.build_model()
    nodes = data["ocv_nodes"]
    parameters = {
        "Cell capacity [A.h]": data["capacity_C"]/3600.,
        "Nominal cell capacity [A.h]": data["capacity_C"]/3600.,
        "Initial SoC": data["initial_soc"],
        "Initial temperature [K]": data["initial_temperature_K"],
        "Ambient temperature [K]": data["ambient_K"],
        "Cell thermal mass [J/K]": data["cell_heat_capacity_J_K"],
        "Jig thermal mass [J/K]": data["jig_heat_capacity_J_K"],
        "Cell-jig heat transfer coefficient [W/K]": data["cell_jig_conductance_W_K"],
        "Jig-air heat transfer coefficient [W/K]": data["jig_ambient_conductance_W_K"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "RCR lookup limit [A]": limit, "Current function [A]": 0. if is_cv else "[input]",
        "Diffusion time constant [s]": data["tau_s"],
        "Open-circuit voltage [V]": lambda z: pybamm.Interpolant(
            np.array([p[0] for p in nodes]), np.array([p[1] for p in nodes]), z,
            interpolator="linear"),
        "Entropic change [V/K]": data["entropy_V_K"], "R0 [Ohm]": data["r0_ohm"],
    }
    for k, (resistance, capacitance) in enumerate(zip(
        data["rc_resistances_ohm"], data["rc_capacitances_F"], strict=True
    )):
        parameters.update({f"R{k+1} [Ohm]": resistance, f"C{k+1} [F]": capacitance,
                           f"Element-{k+1} initial overpotential [V]": 0.})
        model.variables[f"Positive RC-{k} voltage drop [V]"] = -model.variables[
            f"Element-{k+1} overpotential [V]"]
    current = model.variables["Current [A]"]
    model.variables["Diagnostic diffusion heat [W]"] = (
        -current*model.variables["Diffusion overpotential [V]"])
    if not frequency:
        rates = (current, current*model.variables["Voltage [V]"],
                 model.variables["Total heat generation [W]"],
                 -model.variables["Heat transfer from jig to ambient [W]"])
        for name, rate in zip(INTEGRALS, rates, strict=True):
            variable = pybamm.Variable("Qualification integral: "+name)
            model.rhs[variable] = rate
            model.initial_conditions[variable] = pybamm.Scalar(0)
            model.variables["Qualification integral: "+name] = variable
    x = next(iter(model.default_var_pts))
    return pybamm.Simulation(model, parameter_values=pybamm.ParameterValues(parameters),
        var_pts={x: cells}, submesh_types={"ECMD particle": pybamm.Uniform1DSubMesh},
        spatial_methods={"ECMD particle": pybamm.FiniteVolume(options={
            "extrapolation": {"order": {"value": "linear", "gradient": "quadratic"},
                              "use bcs": False}})},
        solver=pybamm.CasadiSolver(mode="safe", rtol=1e-9, atol=1e-11, root_tol=1e-11))


def reference_time(index, cells, variant):
    data = case_data(index)
    simulation = reference_model(index, cells, variant)
    keys = ["Voltage [V]", "Current [A]", "SoC", "Surface SoC", "Cell temperature [K]",
            "Jig temperature [K]", *("Qualification integral: "+key for key in INTEGRALS),
            "Diffusion overpotential [V]", "Diagnostic diffusion heat [W]",
            *(f"Positive RC-{k} voltage drop [V]" for k in range(len(data["rc_resistances_ohm"])))]
    segments = []
    for part in data["segments"]:
        name = "Current function [A]" if part["unit"] == "A" else "Voltage target [V]"
        solution = simulation.step(part["duration_s"], save=False,
            t_eval=np.linspace(0., part["duration_s"], data["samples_per_segment"]),
            inputs={name: part["value"]})
        times = np.asarray(solution.t)
        rows = np.column_stack([np.asarray(solution[key](times)).reshape(-1) for key in keys])
        profiles = np.asarray(solution["Distributed SoC"].entries).T
        segments.append({"times_s": times.tolist(), "rows": rows.tolist(),
                         "profiles": profiles.tolist(), "termination": solution.termination,
                         "end_time_s": float(times[-1])})
        if solution.termination != "final time":
            break
    return time_summary(index, cells, segments, variant)


def native_time(index, cells):
    data, spec = case_data(index), native_case(index, cells)
    state, segments = spec.initial_state(), []
    for part in data["segments"]:
        program = ScalarControlProgram(((0., part["value"]),), part["unit"])
        system = thevenin_system(spec, program)
        system = replace(system, policy=replace(system.policy, rtol=1e-9, atol=1e-12))
        # CoupledSystem takes segment-relative evaluation times, then stamps its
        # trajectory with the persistent state's absolute time.
        times = np.linspace(0., part["duration_s"], data["samples_per_segment"])
        result = system.advance(state, part["duration_s"], evaluation_times_s=times)
        rows, profiles = [], []
        for saved in result.trajectory:
            f, c = thevenin_fields(spec, saved, program), saved.coordinates
            rows.append([f["voltage_V"], f["current_A"], c[SOC], f["surface_soc"],
                c[CELL_T], c[JIG_T], c[CHARGE], c[WORK], c[MODEL_HEAT], c[AMBIENT_LOSS],
                f["diffusion_overpotential_V"], f["diffusion_heat_W"],
                *(c[rc_key(k)] for k in range(len(spec.rc_elements)))])
            profiles.append(list(f["distributed_soc"]))
        segments.append({"times_s": [saved.time_s for saved in result.trajectory],
            "rows": rows, "profiles": profiles, "termination": result.stopping_event,
            "outcome": result.outcome, "end_time_s": result.final_state.time_s,
            "solver": dict(result.solver_diagnostic)})
        state = result.final_state
        if result.outcome != "success":
            break
    return time_summary(index, cells, segments, "native")


def time_summary(index, cells, segments, variant):
    data = case_data(index)
    rows = np.array([row for part in segments for row in part["rows"]])
    profiles = np.array([row for part in segments for row in part["profiles"]])
    charge = data["capacity_C"]*(rows[:, 2]-data["initial_soc"])+rows[:, 6]
    heat = (data["cell_heat_capacity_J_K"]*(rows[:, 4]-data["initial_temperature_K"])
            + data["jig_heat_capacity_J_K"]*(rows[:, 5]-data["initial_temperature_K"])
            - rows[:, 8]+rows[:, 9])
    return {"variant": variant, "cells": cells, "response_order": time_names(index),
        "segments": segments, "maximum_charge_balance_C": float(np.max(abs(charge))),
        "maximum_profile_mean_error": float(np.max(abs(profiles.mean(axis=1)-rows[:, 2]))),
        "maximum_node_heat_balance_J": float(np.max(abs(heat))),
        "final_net_charge_C": float(rows[-1, 6]), "final_work_J": float(rows[-1, 7]),
        "final_model_heat_J": float(rows[-1, 8]),
        "final_profile_from_initial": float(np.max(abs(profiles[-1]-data["initial_soc"]))),
        "minimum_diffusion_heat_W": float(np.min(rows[:, 11])),
        "source": CORRECTION_LABEL if variant == "corrected" else variant,
        "correction_url": CORRECTION_URL if variant == "corrected" else None}


def reference_frequency(index, cells, variant):
    simulation = reference_model(index, cells, variant, frequency=True)
    outputs = ["Voltage [V]", "SoC", "Surface SoC", "Cell temperature [K]",
               "Jig temperature [K]", *(f"Positive RC-{k} voltage drop [V]"
                for k in range(len(case_data(index)["rc_resistances_ohm"])))]
    trace = frequency_from_simulation(simulation, FREQUENCIES, family="Thevenin diffusion",
        grid={"cells": cells, "geometry": "normalized Cartesian 0..1",
              "surface": "linear cell-center extrapolation, without BC substitution"},
        output_names=outputs)
    trace.update(response_order=frequency_names(index), variant=variant,
                 correction_url=CORRECTION_URL if variant == "corrected" else None)
    return trace


def native_frequency(index, cells):
    from chemworld.physchem.thevenin_frequency import thevenin_impedance

    spec = native_case(index, cells)
    state = spec.initial_state()
    before = (dict(state.coordinates), state.time_s)
    result = thevenin_impedance(spec, FREQUENCIES, state=state)
    positions = {variable.name: k for k, variable in enumerate(result.response.variables)}
    points = []
    for point, impedance in zip(result.response.points, result.impedance_ohm, strict=True):
        values = None
        if point.success:
            response = point.response[:, 0]
            mean = response[positions[SOC]]
            deviations = [response[positions[f"thevenin:diffusion_deviation_{k}"]]
                          for k in range(cells-1)]
            surface = mean-1.5*sum(deviations)-.5*deviations[-1]
            values = np.array([impedance, mean, surface, response[positions[CELL_T]],
                response[positions[JIG_T]], *(response[positions[rc_key(k)]]
                                              for k in range(len(spec.rc_elements)))])
        points.append({"frequency_Hz": point.frequency_Hz, "success": point.success,
            "response_real": None if values is None else values.real.tolist(),
            "response_imag": None if values is None else values.imag.tolist(),
            "relative_linear_residual": point.relative_linear_residual,
            "failure_reason": point.failure_reason})
    return {"success": result.success, "points": points, "variant": "native",
        "response_order": frequency_names(index), "diagnostics": dict(result.response.diagnostics),
        "input_state_unchanged": before == (dict(state.coordinates), state.time_s)}


def analytic_frequency(index):
    data = case_data(index)
    points = []
    for frequency in FREQUENCIES:
        s = 2j*np.pi*frequency
        q = np.sqrt(s*data["tau_s"])
        mean = -1/(s*data["capacity_C"])
        surface = mean*q/np.tanh(q)
        rc = [r/(1+s*r*c) for r, c in zip(data["rc_resistances_ohm"],
                                         data["rc_capacitances_F"], strict=True)]
        cell, jig, link, ambient = (data[k] for k in (
            "cell_heat_capacity_J_K", "jig_heat_capacity_J_K",
            "cell_jig_conductance_W_K", "jig_ambient_conductance_W_K"))
        denominator = (s+link/cell)*(s+(link+ambient)/jig)-link**2/(cell*jig)
        thermal = -data["initial_temperature_K"]*data["entropy_V_K"]/cell
        values = np.array([data["r0_ohm"]+sum(rc)-surface, mean, surface,
            thermal*(s+(link+ambient)/jig)/denominator,
            thermal*link/jig/denominator, *rc])
        points.append({"frequency_Hz": float(frequency), "success": True,
                       "response_real": values.real.tolist(),
                       "response_imag": values.imag.tolist()})
    return {"success": True, "points": points, "response_order": frequency_names(index),
        "method": "continuous planar diffusion coth transfer, independent RC and two-node heat"}
