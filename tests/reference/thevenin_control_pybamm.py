"""Actual released named external-circuit submodels; no native control equations."""

import numpy as np
from thevenin_control_cases import dynamic

INPUT_NAMES = {"power": "Power function [W]", "resistance": "Resistance function [Ohm]",
    "differential_power": "Power function [W]",
    "differential_resistance": "Resistance function [Ohm]"}
INTEGRALS = ("charge_C", "work_J", "throughput_charge_C", "throughput_work_J",
             "model_heat_J", "ambient_loss_J")


def build_reference(data, tight):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("named controller reference requires PyBaMM25.4.2")
    mode = "CCCV" if data["mode"] == "cccv" else data["mode"].replace("_", " ")
    n, cells = len(data["rc_resistances_ohm"]), data["diffusion_cells"]
    model = pybamm.equivalent_circuit.Thevenin(options={"operating mode": mode,
        "number of rc elements": n, "diffusion element": "true" if cells else "false"}, build=False)
    if cells:
        class CorrectedDiffusion(pybamm.equivalent_circuit_elements.DiffusionElement):
            """Only the explicitly attributed PR5869 sign correction."""

            def get_coupled_variables(self, variables):
                variables = super().get_coupled_variables(variables)
                variables["Diffusion overpotential [V]"] = (
                    self.param.ocv(variables["Surface SoC"])-self.param.ocv(variables["SoC"]))
                return variables

        class CorrectedThermal(pybamm.equivalent_circuit_elements.ThermalSubModel):
            """Only PR5869's diffusion heat term; controller untouched."""

            def get_coupled_variables(self, variables):
                variables = super().get_coupled_variables(variables)
                added = -variables["Current [A]"]*variables["Diffusion overpotential [V]"]
                variables["Irreversible heat generation [W]"] += added
                variables["Total heat generation [W]"] += added
                return variables

        model.submodels["Diffusion"] = CorrectedDiffusion(model.param, model.options)
        model.submodels["Thermal"] = CorrectedThermal(model.param, model.options)
    model.build_model()
    # Explicit IC changes only the declared state/guess, never the upstream RHS.
    model.initial_conditions[model.variables["Current variable [A]"]] = pybamm.Scalar(
        data["initial_current_A"])
    p = {"Cell capacity [A.h]": data["capacity_C"]/3600.,
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
        "RCR lookup limit [A]": data["maximum_current_A"], "Current function [A]": 0.,
        "Open-circuit voltage [V]": lambda z: data["ocv_intercept_V"]+z,
        "Entropic change [V/K]": data["entropy_V_K"], "R0 [Ohm]": data["r0_ohm"]}
    if data["mode"] == "cccv":
        p.update({"CCCV current function [A]": data["current_limit_A"],
                  "Voltage function [V]": data["segments"][0]["target"]})
    else:
        p[INPUT_NAMES[data["mode"]]] = "[input]"
    if data["r0_law"] == "linear":
        p["R0 [Ohm]"] = lambda temperature, current, soc: .1+.01*current
    elif data["r0_law"] == "absolute":
        p["R0 [Ohm]"] = lambda temperature, current, soc: .1+.4*abs(current)
    if cells:
        p["Diffusion time constant [s]"] = data["diffusion_tau_s"]
    for k, (r, c) in enumerate(zip(
        data["rc_resistances_ohm"], data["rc_capacitances_F"], strict=True
    )):
        p.update({f"R{k+1} [Ohm]": r, f"C{k+1} [F]": c,
                  f"Element-{k+1} initial overpotential [V]": 0.})
        model.variables[f"Positive RC-{k} voltage drop [V]"] = -model.variables[
            f"Element-{k+1} overpotential [V]"]
    current, voltage = model.variables["Current [A]"], model.variables["Voltage [V]"]
    rates = (current, current*voltage, abs(current), abs(current*voltage),
             model.variables["Total heat generation [W]"],
             -model.variables["Heat transfer from jig to ambient [W]"])
    for name, rate in zip(INTEGRALS, rates, strict=True):
        variable = pybamm.Variable("Control qualification integral: "+name)
        model.rhs[variable] = rate
        model.initial_conditions[variable] = pybamm.Scalar(0.)
        model.variables["Control qualification integral: "+name] = variable
    model.variables["Diagnostic diffusion heat [W]"] = (
        -current*model.variables["Diffusion overpotential [V]"])
    options = {}
    if cells:
        x = next(iter(model.default_var_pts))
        options = {"var_pts": {x: cells},
            "submesh_types": {"ECMD particle": pybamm.Uniform1DSubMesh},
            "spatial_methods": {"ECMD particle": pybamm.FiniteVolume(options={"extrapolation": {
                "order": {"value": "linear", "gradient": "quadratic"}, "use bcs": False}})}}
    rtol, atol = (1e-10, 1e-12) if tight else (1e-8, 1e-10)
    simulation = pybamm.Simulation(model, parameter_values=pybamm.ParameterValues(p),
        solver=pybamm.CasadiSolver(mode="safe", rtol=rtol, atol=atol, root_tol=atol), **options)
    return simulation


def reference_inputs(data, part):
    return {} if data["mode"] == "cccv" else {INPUT_NAMES[data["mode"]]: part["target"]}


def output_names(data):
    names = ["Voltage [V]", "Current [A]", "SoC", "Cell temperature [K]", "Jig temperature [K]",
             *("Control qualification integral: "+name for name in INTEGRALS),
             *(f"Positive RC-{k} voltage drop [V]" for k in range(len(data["rc_resistances_ohm"])))]
    if dynamic(data):
        names += ["Current variable [A]"]
    if data["diffusion_cells"]:
        names += ["Surface SoC", "Diffusion overpotential [V]", "Diagnostic diffusion heat [W]"]
    return names


def reference_probe(data):
    simulation = build_reference(data, True)
    simulation.build()
    model, solver = simulation.built_model, simulation.solver
    inputs = reference_inputs(data, data["segments"][0])
    solver.set_up(model, inputs=inputs)
    state = np.asarray(solver.calculate_consistent_state(model, inputs=inputs)).ravel()
    current, voltage = (float(np.asarray(model.variables[key].evaluate(
                            t=0., y=state, inputs=inputs)).item())
                        for key in ("Current [A]", "Voltage [V]"))
    return {"upstream_returned_state": True, "current_A": current, "voltage_V": voltage,
            "finite": bool(np.isfinite(state).all()), "state": state.tolist(),
            "note": "upstream numerical outcome is not a root-uniqueness certificate"}
