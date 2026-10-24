"""Actual native residual, released-model derivatives and independent closed responses."""

from itertools import pairwise

import numpy as np
from battery_frequency_pybamm import frequency_from_simulation
from battery_thevenin_traces import build_reference
from thevenin_frequency_cases import FREQUENCIES, case_data, native_case, response_order

from chemworld.physchem.battery_thevenin import CELL_T, JIG_T, SOC, rc_key
from chemworld.physchem.thevenin_frequency import thevenin_impedance


def native_trace(index):
    spec = native_case(index)
    state = spec.initial_state()
    before = (dict(state.coordinates), state.time_s)
    result = thevenin_impedance(spec, FREQUENCIES, state=state)
    identities = {v.name: k for k, v in enumerate(result.response.variables)}
    columns = [identities[SOC], identities[CELL_T], identities[JIG_T],
               *(identities[rc_key(k)] for k in range(len(spec.rc_elements)))]
    points = []
    for point, z in zip(result.response.points, result.impedance_ohm, strict=True):
        values = None if not point.success else np.r_[z, point.response[columns, 0]]
        points.append({"frequency_Hz": point.frequency_Hz, "success": point.success,
            "response_real": None if values is None else values.real.tolist(),
            "response_imag": None if values is None else values.imag.tolist(),
            "relative_linear_residual": point.relative_linear_residual,
            "failure_reason": point.failure_reason})
    return {"success": result.success, "response_order": response_order(index),
            "points": points, "diagnostics": dict(result.response.diagnostics),
            "input_state_unchanged": before == (dict(state.coordinates), state.time_s)}


def reference_model(index):
    import pybamm

    data = case_data(index)
    # Reuse only the existing released-model construction and explicit constant
    # dictionary. No native solver, parameter-law evaluation or derivative runs.
    source_index = {0: 1, 1: 3, 2: 5, 4: 7}[data["rc_count"]]
    model, parameters = build_reference(source_index)
    for state in list(model.rhs):
        if state.name.startswith("Qualification integral: "):
            del model.rhs[state]
            del model.initial_conditions[state]
            del model.variables[state.name]
    nodes = data["ocv_nodes"]
    values = {"Current function [A]": "[input]", "Initial SoC": data["soc"],
        "RCR lookup limit [A]": data["maximum_current_A"],
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["ambient_K"],
        "Cell capacity [A.h]": data["capacity_C"]/3600.,
        "Nominal cell capacity [A.h]": data["capacity_C"]/3600.,
        "Cell thermal mass [J/K]": data["cell_heat_capacity_J_K"],
        "Jig thermal mass [J/K]": data["jig_heat_capacity_J_K"],
        "Cell-jig heat transfer coefficient [W/K]": data["cell_jig_conductance_W_K"],
        "Jig-air heat transfer coefficient [W/K]": data["jig_ambient_conductance_W_K"],
        "Open-circuit voltage [V]": lambda z: pybamm.Interpolant(
            np.array([p[0] for p in nodes]), np.array([p[1] for p in nodes]), z,
            interpolator="linear"),
        "R0 [Ohm]": data["r0_ohm"], "Entropic change [V/K]": data["entropy_V_K"],
        **{f"R{k+1} [Ohm]": r for k, r in enumerate(data["resistances_ohm"])},
        **{f"C{k+1} [F]": c for k, c in enumerate(data["capacitances_F"])}}
    if data["feedback"] == "affine":
        values.update({
            "R0 [Ohm]": lambda T, i, z: .08+.02*z+.0001*(T+273.15-298.15)+.002*i,
            "R1 [Ohm]": lambda T, i, z: .15+.05*z+.0002*(T+273.15-298.15)+.001*i,
            "C1 [F]": lambda T, i, z: 4.+z+.002*(T+273.15-298.15)+.01*i})
    elif data["feedback"] == "current_cusp":
        values.update({"R0 [Ohm]": lambda T, i, z: .1+.02*abs(i),
                       "R1 [Ohm]": lambda T, i, z: .2+.01*abs(i)})
    parameters.update(values)
    outputs = ["Voltage [V]", "SoC", "Cell temperature [K]", "Jig temperature [K]"]
    for k in range(data["rc_count"]):
        name = f"Positive RC-{k} voltage drop [V]"
        model.variables[name] = -model.variables[f"Element-{k+1} overpotential [V]"]
        outputs.append(name)
    return pybamm.Simulation(model, parameter_values=parameters,
        solver=pybamm.CasadiSolver(mode="safe", rtol=1e-10, atol=1e-12, root_tol=1e-12)), outputs


def reference_trace(index):
    simulation, outputs = reference_model(index)
    result = frequency_from_simulation(simulation, FREQUENCIES, family="Thevenin",
        grid={"spatial_discretization": "none; released finite-dimensional RC/two-node model"},
        output_names=outputs)
    result["response_order"] = response_order(index)
    return result


def closed_trace(index):
    data = case_data(index)
    r0, resistances, capacitances = (data[k] for k in (
        "r0_ohm", "resistances_ohm", "capacitances_F"))
    if data["feedback"] == "affine":
        r0, resistances, capacitances = .094, [.185], [4.7]
    nodes = data["ocv_nodes"]
    slope = next((right[1]-left[1])/(right[0]-left[0])
                 for left, right in pairwise(nodes)
                 if left[0] < data["soc"] < right[0])
    cell, jig, link, ambient, temperature, entropy = (data[k] for k in (
        "cell_heat_capacity_J_K", "jig_heat_capacity_J_K", "cell_jig_conductance_W_K",
        "jig_ambient_conductance_W_K", "temperature_K", "entropy_V_K"))
    points = []
    for frequency in FREQUENCIES:
        s = 2j*np.pi*frequency
        rc = [r/(1+s*r*c) for r, c in zip(resistances, capacitances, strict=True)]
        soc = -1/(s*data["capacity_C"])
        denominator = (s+link/cell)*(s+(link+ambient)/jig)-link**2/(cell*jig)
        thermal_forcing = -temperature*entropy/cell
        values = np.array([r0+sum(rc)-slope*soc, soc,
            thermal_forcing*(s+(link+ambient)/jig)/denominator,
            thermal_forcing*link/jig/denominator, *rc])
        points.append({"frequency_Hz": float(frequency), "success": True,
                       "response_real": values.real.tolist(),
                       "response_imag": values.imag.tolist()})
    return {"success": True, "response_order": response_order(index), "points": points,
            "method": "independent RC/SOC rational response and explicit two-node thermal inverse"}
