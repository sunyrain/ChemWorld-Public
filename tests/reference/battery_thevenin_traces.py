"""Released ECM reference and real native traces, with explicit sign/unit mapping."""

from dataclasses import replace

import numpy as np
from battery_thevenin_cases import CASES, TIMES, native_case

from chemworld.physchem.battery_thevenin import (
    AMBIENT_LOSS,
    CELL_T,
    CHARGE,
    JIG_T,
    MODEL_HEAT,
    SOC,
    THROUGHPUT_E,
    THROUGHPUT_Q,
    WORK,
    rc_key,
    thevenin_fields,
    thevenin_system,
)

BASE_RESPONSES = ("voltage_V", "current_A", "soc", "cell_temperature_K", "jig_temperature_K",
                  "charge_C", "work_J", "throughput_charge_C", "throughput_work_J",
                  "model_heat_J", "ambient_loss_J")
BASE_BOUNDS = (1e-6, 1e-6, 1e-7, 1e-5, 1e-5, 1e-5, 5e-5, 1e-5, 5e-5, 1e-5, 1e-5)
COORDINATES = (SOC, CELL_T, JIG_T, CHARGE, WORK, THROUGHPUT_Q, THROUGHPUT_E,
               MODEL_HEAT, AMBIENT_LOSS)


def response_names(index):
    return (*BASE_RESPONSES, *(f"rc{k}_voltage_drop_V" for k in range(CASES[index-1]["rc_count"])))


def response_bounds(index):
    return (*BASE_BOUNDS, *(1e-6 for _ in range(CASES[index-1]["rc_count"])))


def native_trace(index, tight):
    spec, program = native_case(index)
    system = thevenin_system(spec, program)
    if tight:
        system = replace(system, policy=replace(system.policy, rtol=1e-10, atol=1e-13))
    result = system.advance(spec.initial_state(), 10., evaluation_times_s=TIMES)
    rows, charge_residuals, heat_residuals = [], [], []
    p = spec.native_parameters
    for state in result.trajectory:
        fields = thevenin_fields(spec, state, program)
        x = state.coordinates
        rows.append([fields["voltage_V"], fields["current_A"],
                     *(x[k] for k in COORDINATES),
                     *(x[rc_key(k)] for k in range(len(spec.rc_elements)))])
        charge_residuals.append(p["capacity_C"]*(x[SOC]-p["initial_soc"])+x[CHARGE])
        heat_residuals.append(p["cell_heat_capacity_J_K"]*(x[CELL_T]-310.)
                              + p["jig_heat_capacity_J_K"]*(x[JIG_T]-310.)
                              - x[MODEL_HEAT]+x[AMBIENT_LOSS])
    return {"times_s": [s.time_s for s in result.trajectory], "rows": rows,
            "outcome": result.outcome, "end_time_s": result.final_state.time_s,
            "event": result.stopping_event, "solver": dict(result.solver_diagnostic),
            "policy": {"rtol": system.policy.rtol, "atol": system.policy.atol},
            "maximum_charge_balance_C": max(map(abs, charge_residuals)),
            "maximum_node_heat_balance_J": max(map(abs, heat_residuals)),
            "chemical_total_energy_residual_J": result.energy_balance_residual_J}


def build_reference(index):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("fixed Thevenin reference requires PyBaMM 25.4.2")
    case = CASES[index-1]
    n, target, limit = case["rc_count"], case["target"], case["limit_A"]

    def finite_source(variables):
        current, voltage = variables["Current [A]"], variables["Voltage [V]"]
        requested = current+limit/10.*(voltage-target)
        return current-pybamm.maximum(-limit, pybamm.minimum(limit, requested))

    model = pybamm.equivalent_circuit.Thevenin(options={
        "number of rc elements": n,
        "operating mode": "current" if case["unit"] == "A" else finite_source}, build=False)
    if case["unit"] == "V":
        # This release selects differential control for a callable ECM option.
        # Use its public algebraic FunctionControl submodel explicitly, not an
        # unintended current controller ODE or a modified thermal/RC equation.
        model.submodels["external circuit"] = pybamm.external_circuit.FunctionControl(
            model.param, finite_source, model.options, control="algebraic")
    model.build_model()
    # Explicit independent authored expressions: do not call the native law or
    # use the release's default material parameter set.
    parameters = {
        "Cell capacity [A.h]": 100./3600., "Nominal cell capacity [A.h]": 100./3600.,
        "Initial SoC": .7, "Initial temperature [K]": 310.,
        "Ambient temperature [K]": 298.15, "Cell thermal mass [J/K]": 100.,
        "Jig thermal mass [J/K]": 500., "Cell-jig heat transfer coefficient [W/K]": .5,
        "Jig-air heat transfer coefficient [W/K]": .2, "Lower voltage cut-off [V]": .1,
        "Upper voltage cut-off [V]": 10., "RCR lookup limit [A]": 10.,
        "Current function [A]": target if case["unit"] == "A" else 0.,
        "Open-circuit voltage [V]": lambda z: 3+z,
        "Entropic change [V/K]": lambda ocv, T: 1e-4,
        "R0 [Ohm]": lambda T, current, z: .1,
    }
    for k in range(n):
        parameters.update({f"R{k+1} [Ohm]": lambda T, current, z, k=k: .2*(k+1),
                           f"C{k+1} [F]": lambda T, current, z, k=k: 5./(k+1),
                           f"Element-{k+1} initial overpotential [V]": 0.})
    if case["feedback"]:
        parameters.update({
            "R0 [Ohm]": lambda T, current, z: .08+.02*z+.0001*(T+273.15-298.15)+.002*current,
            "R1 [Ohm]": lambda T, current, z: .15+.05*z+.0002*(T+273.15-298.15)+.001*current,
            "C1 [F]": lambda T, current, z: 4.+z+.002*(T+273.15-298.15)+.01*current,
        })
    current, voltage = model.variables["Current [A]"], model.variables["Voltage [V]"]
    # Passive record ODEs do not alter the released physical equations. This
    # release's calculate-discharge-energy option does not add these ECM states.
    rates = (current, current*voltage, abs(current), abs(current*voltage),
             model.variables["Total heat generation [W]"],
             -model.variables["Heat transfer from jig to ambient [W]"])
    for name, rate in zip(BASE_RESPONSES[5:], rates, strict=True):
        variable = pybamm.Variable("Qualification integral: "+name)
        model.rhs[variable] = rate
        model.initial_conditions[variable] = pybamm.Scalar(0)
        model.variables["Qualification integral: "+name] = variable
    return model, pybamm.ParameterValues(parameters)


def reference_trace(index, tight):
    import pybamm

    model, parameters = build_reference(index)
    tolerance = (1e-10, 1e-12) if tight else (1e-8, 1e-10)
    solver = pybamm.CasadiSolver(mode="safe", rtol=tolerance[0], atol=tolerance[1],
                                root_tol=tolerance[1])
    solution = pybamm.Simulation(model, parameter_values=parameters, solver=solver).solve(TIMES)
    keys = ("Voltage [V]", "Current [A]", "SoC", "Cell temperature [K]", "Jig temperature [K]",
            *("Qualification integral: "+name for name in BASE_RESPONSES[5:]))
    columns = [np.asarray(solution[key](TIMES), dtype=float).reshape(-1) for key in keys]
    columns.extend(-np.asarray(solution[f"Element-{k+1} overpotential [V]"](TIMES),
                               dtype=float).reshape(-1)
                   for k in range(CASES[index-1]["rc_count"]))
    return {"times_s": TIMES.tolist(), "rows": np.column_stack(columns).tolist(),
            "outcome": "success" if solution.termination == "final time" else "stopped",
            "termination": solution.termination, "end_time_s": float(solution.t[-1]),
            "solver": {"name": "CasadiSolver", "rtol": tolerance[0], "atol": tolerance[1],
                       "root_tol": tolerance[1], "mode": "safe"},
            "reference_state_scope": "released physical states plus six passive integral ODEs"}
