"""Persistent native and released-controller trajectories on fixed observation grids."""

from dataclasses import replace

import numpy as np
from thevenin_control_cases import dynamic, native_program, native_spec, response_names
from thevenin_control_pybamm import build_reference, output_names, reference_inputs

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
from chemworld.physchem.battery_thevenin_control import CONTROLLER_CURRENT


def summary(data, segments, implementation, tight):
    rows = np.array([row for part in segments for row in part["rows"]])
    charge = data["capacity_C"]*(rows[:, 2]-data["initial_soc"])+rows[:, 5]
    heat = (data["cell_heat_capacity_J_K"]*(rows[:, 3]-data["initial_temperature_K"])
            + data["jig_heat_capacity_J_K"]*(rows[:, 4]-data["initial_temperature_K"])
            - rows[:, 9]+rows[:, 10])
    profiles = np.array([p for part in segments for p in part["profiles"]])
    return {"implementation": implementation, "tight": tight, "segments": segments,
            "response_order": response_names(data),
            "maximum_charge_balance_C": float(np.max(abs(charge))),
            "maximum_node_heat_balance_J": float(np.max(abs(heat))),
            "maximum_profile_mean_error": (float(np.max(abs(profiles.mean(axis=1)-rows[:, 2])))
                                           if data["diffusion_cells"] else 0.),
            "maximum_current_magnitude_A": float(np.max(abs(rows[:, 1])))}


def native_trace(data, tight):
    spec = native_spec(data)
    state, segments = spec.initial_state(), []
    for part in data["segments"]:
        program = native_program(data, part)
        system = thevenin_system(spec, program)
        rtol, atol = (1e-10, 1e-13) if tight else (1e-8, 1e-11)
        system = replace(system, policy=replace(system.policy, rtol=rtol, atol=atol))
        result = system.advance(state, part["duration_s"], evaluation_times_s=np.linspace(
            0., part["duration_s"], data["samples_per_segment"]))
        rows, profiles = [], []
        for saved in result.trajectory:
            f, c = thevenin_fields(spec, saved, program), saved.coordinates
            row = [f["voltage_V"], f["current_A"], c[SOC], c[CELL_T], c[JIG_T],
                   c[CHARGE], c[WORK], c[THROUGHPUT_Q], c[THROUGHPUT_E],
                   c[MODEL_HEAT], c[AMBIENT_LOSS],
                   *(c[rc_key(k)] for k in range(len(spec.rc_elements)))]
            if dynamic(data):
                row.append(c[CONTROLLER_CURRENT])
            if data["diffusion_cells"]:
                row.extend(f[key] for key in (
                    "surface_soc", "diffusion_overpotential_V", "diffusion_heat_W"))
                profiles.append(list(f["distributed_soc"]))
            rows.append(row)
        segments.append({"times_s": [s.time_s for s in result.trajectory], "rows": rows,
            "profiles": profiles, "termination": result.stopping_event,
            "outcome": result.outcome, "end_time_s": result.final_state.time_s,
            "solver": dict(result.solver_diagnostic)})
        state = result.final_state
        if result.outcome != "success":
            break
    return summary(data, segments, "native", tight)


def reference_trace(data, tight):
    simulation, segments = build_reference(data, tight), []
    for part in data["segments"]:
        solution = simulation.step(part["duration_s"], save=False,
            t_eval=np.linspace(0., part["duration_s"], data["samples_per_segment"]),
            inputs=reference_inputs(data, part))
        times = np.asarray(solution.t)
        rows = np.column_stack([np.asarray(solution[key](times)).reshape(-1)
                                for key in output_names(data)])
        profiles = (np.asarray(solution["Distributed SoC"].entries).T.tolist()
                    if data["diffusion_cells"] else [])
        segments.append({"times_s": times.tolist(), "rows": rows.tolist(), "profiles": profiles,
                         "termination": solution.termination, "end_time_s": float(times[-1])})
        if solution.termination != "final time":
            break
    return summary(data, segments, "PyBaMM25.4.2 named external circuit", tight)


def native_probe(data):
    spec = native_spec(data)
    state = spec.initial_state()
    before = (dict(state.coordinates), state.time_s)
    try:
        fields = thevenin_fields(spec, state, native_program(data, data["segments"][0]))
        result = {"rejected": False, "fields": fields}
    except Exception as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    result["input_state_unchanged"] = before == (dict(state.coordinates), state.time_s)
    return result


def independent_rejection_fact(data):
    """Elementary initial-state algebra, independent of both implementations."""
    u = data["ocv_intercept_V"]+data["initial_soc"]
    expected = data["expected"]
    if expected in {"fold_rejection", "no_root_rejection"}:
        r, p = data["r0_ohm"], data["segments"][0]["target"]
        discriminant = u*u-4*r*p
        derivative = u-2*r*data["initial_current_A"]
        passed = (abs(discriminant) <= 1e-12 and abs(derivative) <= 1e-12
                  if expected == "fold_rejection" else discriminant < 0)
        return {"discriminant": discriminant, "initial_power_derivative": derivative,
                "passed": bool(passed)}
    if expected == "nonmonotone_power_rejection":
        # For positive I, R=.1+.4I, P=U*I-.1I²-.4I³.
        lo, hi = data["branch_A"]
        derivatives = [u-.2*i-1.2*i*i for i in (lo, hi)]
        roots = np.roots([.4, .1, -u, data["segments"][0]["target"]])
        inside = [float(z.real) for z in roots if abs(z.imag) < 1e-10 and lo < z.real < hi]
        return {"endpoint_power_derivatives": derivatives, "real_roots_inside_branch_A": inside,
                "IR_derivative_lower_bound": .1,
                "passed": bool(derivatives[0]*derivatives[1] < 0 and len(inside) == 2)}
    if expected == "singular_initial_current_rejection":
        lo, hi = data["branch_A"]
        initial = data["initial_current_A"]
        return {"initial_current_A": initial, "branch_A": [lo, hi],
                "passed": bool(initial == 0 and not lo <= initial <= hi)}
    raise ValueError("not a fixed rejection probe")
