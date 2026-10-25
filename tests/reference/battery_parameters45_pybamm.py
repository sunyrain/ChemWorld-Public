"""Actual pinned Chen2020 SPM with an explicitly adapted finite external source.

All 94 released material entries remain untouched. The only additions are a
CC/finite-CV external circuit, shared physical-domain events and readout aliases.
Source energy uses its declared signed discharge-energy state, not throughput.
"""

from copy import deepcopy

import numpy as np
from battery_parameters45_cases import (
    ATOL,
    CASADI_VERSION,
    PYBAMM_VERSION,
    RESPONSE_ORDER,
    ROOT_TOL,
    RTOL,
    stage_times,
    validate_case,
)

_PREFIX = "parameters45:"
_MODE = _PREFIX + "voltage-mode"
_CURRENT = _PREFIX + "current-request-A"
_VOLTAGE = _PREFIX + "voltage-request-V"


def _dependencies():
    import casadi
    import pybamm

    if pybamm.__version__ != PYBAMM_VERSION or casadi.__version__ != CASADI_VERSION:
        raise RuntimeError("source parameter reference requires PyBaMM25.4.2 and CasADi3.6.7")
    return pybamm


def stage_inputs(stage):
    # BaseSolver.step sorts the input vector before compiling CasADi functions.
    # The separate boundary root must use exactly that same order.
    values = {_MODE: float(stage["unit"] == "V"),
              _CURRENT: stage["target"] if stage["unit"] == "A" else 0.0,
              _VOLTAGE: stage["target"] if stage["unit"] == "V" else 0.0}
    return dict(sorted(values.items()))


def finite_control_residual(current, voltage, *, voltage_mode, current_target,
                            voltage_target, maximum_current_A, gain_A_V):
    """Independent scalar statement of the adapted algebraic external source."""
    desired = current + gain_A_V * (voltage - voltage_target)
    limited = max(-maximum_current_A, min(maximum_current_A, desired))
    return current - ((1 - voltage_mode) * current_target + voltage_mode * limited)


def build_reference(data, radial_cells):
    pybamm = _dependencies()
    validate_case(data)
    if type(radial_cells) is not int or radial_cells < 2:
        raise ValueError("source radial grid requires an integer >= 2")

    def finite_source(variables):
        current, voltage = variables["Current [A]"], variables["Voltage [V]"]
        mode = pybamm.InputParameter(_MODE)
        requested = current + data["control_gain_A_V"] * (
            voltage - pybamm.InputParameter(_VOLTAGE))
        clipped = pybamm.maximum(-data["maximum_current_A"],
                                pybamm.minimum(data["maximum_current_A"], requested))
        return current - ((1 - mode) * pybamm.InputParameter(_CURRENT) + mode * clipped)

    model = pybamm.lithium_ion.SPM({
        "thermal": "isothermal", "surface form": "false", "particle": "Fickian diffusion",
        "open-circuit potential": "single", "SEI": "none", "lithium plating": "none",
        "current collector": "uniform", "dimensionality": 0,
        "calculate discharge energy": "true", "operating mode": finite_source,
    })
    parameters = pybamm.ParameterValues("Chen2020")
    released = pybamm.parameter_sets["Chen2020"]
    original_events = tuple((e.name, e.event_type.name) for e in model.events)
    scalar_outputs = {
        "voltage_V": model.variables["Voltage [V]"],
        "current_A": model.variables["Current [A]"],
        "charge_C": 3600 * model.variables["Discharge capacity [A.h]"],
        "work_J": 3600 * model.variables["Discharge energy [W.h]"],
    }
    for electrode in ("negative", "positive"):
        surface = model.variables[f"X-averaged {electrode} particle surface concentration"]
        average = model.variables[f"Average {electrode} particle concentration"]
        scalar_outputs.update({
            electrode + "_Li_mol": model.variables[f"Total lithium in {electrode} electrode [mol]"],
            electrode + "_surface_fraction": surface,
            electrode + "_average_fraction": average,
            electrode + "_ocp_V": model.variables[
                f"X-averaged {electrode} electrode open-circuit potential [V]"],
            electrode + "_overpotential_V": model.variables[
                f"X-averaged {electrode} electrode reaction overpotential [V]"],
        })
        model.events.extend((
            pybamm.Event(_PREFIX + electrode + ":minimum-surface",
                         data["common_event_scale"] * (surface - data["surface_margin"])),
            pybamm.Event(_PREFIX + electrode + ":maximum-surface",
                         data["common_event_scale"] * (1 - data["surface_margin"] - surface)),
        ))
    for name, expression in scalar_outputs.items():
        model.variables[_PREFIX + name] = expression
    points = {var: radial_cells if str(var).startswith("r_") else 5
              for var in model.default_var_pts}
    simulation = pybamm.Simulation(
        model, parameter_values=parameters, var_pts=points,
        solver=pybamm.CasadiSolver(mode="safe", rtol=RTOL, atol=ATOL, root_tol=ROOT_TOL),
    )
    simulation.build()
    built = simulation.built_model
    return {
        "simulation": simulation, "model": built, "parameter_values": parameters,
        "case": deepcopy(data), "radial_cells": radial_cells,
        "original_events": original_events,
        "source_parameter_manifest": {
            k: {"kind": "function", "function_id": v.__name__} if callable(v)
            else {"kind": "number" if isinstance(v, (int, float)) else "metadata", "value": v}
            for k, v in released.items()
        },
        "parameter_updates": {},
    }


def _read(expression, time_s, state, inputs):
    return np.asarray(expression.evaluate(t=time_s, y=state, inputs=inputs), dtype=float)


def _scalar(expression, time_s, state, inputs):
    result = _read(expression, time_s, state, inputs)
    if result.size != 1:
        raise ValueError("source scalar readout has unexpected shape")
    return float(result.ravel()[0])


def state_record(built, time_s, state, inputs):
    model = built["model"]
    raw = np.asarray(state, dtype=float).reshape(-1, 1)
    row = [_scalar(model.variables[_PREFIX + name], time_s, raw, inputs)
           for name in RESPONSE_ORDER]
    profiles = {}
    for electrode in ("negative", "positive"):
        concentration = _read(model.variables[
            f"X-averaged {electrode} particle concentration [mol.m-3]"], time_s, raw, inputs)
        maximum = built["parameter_values"][
            f"Maximum concentration in {electrode} electrode [mol.m-3]"]
        profiles[electrode] = {
            "concentration_mol_m3": concentration.ravel().tolist(),
            "fraction": (concentration.ravel() / maximum).tolist(),
        }
    return {"time_s": float(time_s), "row": row, "profiles": profiles,
            "source_y": raw.ravel().tolist(),
            "source_rhs": _read(model.concatenated_rhs, time_s, raw, inputs).ravel().tolist(),
            "source_algebraic": _read(model.concatenated_algebraic, time_s, raw, inputs)
            .ravel().tolist(),
            "source_events": {e.name: _read(e.expression, time_s, raw, inputs).ravel().tolist()
                              for e in model.events},
            "source_particle_Li_mol": _scalar(
                model.variables["Total lithium in particles [mol]"], time_s, raw, inputs),
            "source_total_Li_including_electrolyte_mol": _scalar(
                model.variables["Total lithium [mol]"], time_s, raw, inputs)}


def initial_state_record(built, stage, *, time_s=0.0, previous_state=None):
    """Only source algebraic consistency at the exact boundary; no time advance."""
    simulation, model = built["simulation"], built["model"]
    inputs = stage_inputs(stage)
    simulation.solver.set_up(model, inputs=inputs)
    original = (np.asarray(model.concatenated_initial_conditions.evaluate(inputs=inputs),
                           dtype=float)
                if previous_state is None else np.asarray(previous_state, dtype=float))
    model.y0 = original.reshape(-1, 1)
    state = np.asarray(simulation.solver.calculate_consistent_state(
        model, time=time_s, inputs=inputs), dtype=float).reshape(-1, 1)
    record = state_record(built, time_s, state, inputs)
    record["differential_state_unchanged_by_boundary_root"] = bool(
        np.array_equal(state[:model.len_rhs], original.reshape(-1, 1)[:model.len_rhs]))
    record["boundary_operation"] = "source algebraic root only; no integration"
    return record


def source_contract(built):
    pybamm = _dependencies()
    simulation, model, values = (built[k] for k in ("simulation", "model", "parameter_values"))
    mesh = {}
    for electrode in ("negative", "positive"):
        grid = simulation.mesh[electrode + " particle"]
        mesh[electrode] = {"class": type(grid).__name__, "nodes_m": grid.nodes.tolist(),
                           "edges_m": grid.edges.tolist(), "cells": grid.npts,
                           "radius_m": values[electrode.capitalize() + " particle radius [m]"]}
    return {
        "implementation": "actual PyBaMM25.4.2 Chen2020 SPM", "pybamm_version": PYBAMM_VERSION,
        "casadi_version": CASADI_VERSION, "radial_cells": built["radial_cells"], "mesh": mesh,
        "parameter_updates": built["parameter_updates"],
        "source_parameter_manifest": built["source_parameter_manifest"],
        "rhs_size": model.len_rhs, "algebraic_size": model.len_alg,
        "state_layout": [{
            "name": var.name, "slices": [[s.start, s.stop] for s in model.y_slices[var]],
            "scale": float(values.evaluate(var.scale)),
            "reference": float(values.evaluate(var.reference)),
            "domains": {k: list(v) for k, v in var.domains.items()},
            "kind": "differential" if var in model.rhs else "algebraic",
        } for var in (*model.rhs, *model.algebraic)],
        "original_events": [{"name": name, "type": kind}
                            for name, kind in built["original_events"]],
        "all_events": [{"name": e.name, "type": e.event_type.name} for e in model.events],
        "F_C_mol": float(pybamm.constants.F.value), "R_J_mol_K": float(pybamm.constants.R.value),
        "response_order": RESPONSE_ORDER,
        "external_source_adapter": {
            "equation": "I-[(1-mode)*Icc+mode*clip(I+gain*(V-Vtarget),-Imax,Imax)]=0",
            "gain_A_V": built["case"]["control_gain_A_V"],
            "maximum_current_A": built["case"]["maximum_current_A"],
            "scope": "Added hardware control; original electrochemistry and material data "
            "untouched",
        },
        "work_adapter": {
            "source_state": "Discharge energy [W.h]", "conversion_to_J": 3600,
            "rhs": "I*Voltage/3600", "signed": True,
            "scope": "Declared source calculate-discharge-energy option; no added observer state",
        },
        "event_adapter": {
            "original_events_preserved": True,
            "common_surface_event_scale": built["case"]["common_event_scale"],
            "source_Casadi_termination_expression_shift": 1e-5,
        },
    }


def reference_time(data, radial_cells, *, progress_callback=None):
    """Execute one explicitly requested route; the coordinator owns the GO gate."""
    built = build_reference(data, radial_cells)
    simulation, segments = built["simulation"], []
    contract = source_contract(built)
    previous_state, previous_time = None, 0.0
    for index, stage in enumerate(data["stages"]):
        boundary = None
        try:
            boundary = initial_state_record(built, stage, time_s=previous_time,
                                            previous_state=previous_state)
            if progress_callback is not None:
                progress_callback({"contract": contract, "case": deepcopy(data),
                                   "segments": segments, "active_stage": index,
                                   "active_request": deepcopy(stage), "boundary_t0": boundary,
                                   "phase": "before-source-integration"})
            inputs = stage_inputs(stage)
            solution = simulation.step(stage["duration_s"], save=False,
                                       t_eval=np.asarray(stage_times(stage)), inputs=inputs)
            times, states = np.asarray(solution.t), np.asarray(solution.y, dtype=float)
            if progress_callback is not None:
                progress_callback({"contract": contract, "case": deepcopy(data),
                                   "segments": segments, "active_stage": index,
                                   "boundary_t0": boundary, "phase": "source-solution-returned",
                                   "pending_source_solution": {
                                       "times_s": times.tolist(), "y": states.tolist(),
                                       "all_ts": [np.asarray(t).tolist() for t in solution.all_ts],
                                       "all_ys": [np.asarray(y, dtype=float).tolist()
                                                  for y in solution.all_ys],
                                       "termination": solution.termination,
                                       "t_event": None if solution.t_event is None
                                       else np.asarray(solution.t_event).tolist(),
                                       "y_event": None if solution.y_event is None
                                       else np.asarray(solution.y_event, dtype=float).tolist(),
                                   }})
            if times.size == 0 or states.shape != (built["model"].len_rhs
                                                  + built["model"].len_alg, times.size):
                raise ValueError("source original solution state shape is incomplete")
            records = [state_record(built, t, states[:, i:i+1], inputs)
                       for i, t in enumerate(times)]
            segment = {
                "index": index, "request": deepcopy(stage), "inputs": inputs,
                "requested_start_time_s": previous_time,
                "requested_end_time_s": previous_time + stage["duration_s"],
                "requested_relative_times_s": stage_times(stage),
                "boundary_t0": boundary,
                "original_reported_times_s": times.tolist(),
                "original_all_ts": [np.asarray(t).tolist() for t in solution.all_ts],
                "original_all_ys": [np.asarray(y, dtype=float).tolist() for y in solution.all_ys],
                "original_t_event": None if solution.t_event is None
                else np.asarray(solution.t_event).tolist(),
                "original_y_event": None if solution.y_event is None
                else np.asarray(solution.y_event, dtype=float).tolist(),
                "records": records, "termination": solution.termination,
                "actual_start_time_s": float(times[0]), "actual_end_time_s": float(times[-1]),
                "sampling_scope": "All stored source solution states; internal integrator steps "
                "are not exposed. Exact stage t0 is separately algebraically reclosed.",
                "solver": {"name": "CasadiSolver-safe", "rtol": RTOL, "atol": ATOL,
                           "root_tol": ROOT_TOL},
            }
            segments.append(segment)
            if progress_callback is not None:
                progress_callback({"contract": contract, "case": deepcopy(data),
                                   "segments": segments, "active_stage": index,
                                   "phase": "source-stage-captured"})
            previous_time, previous_state = float(times[-1]), states[:, -1:]
            if solution.termination != "final time":
                break
        except Exception as exc:
            exc.partial_history = {"contract": contract, "segments": segments,
                                   "failed_stage": index, "failed_request": deepcopy(stage),
                                   "failed_boundary_t0": boundary,
                                   "last_completed_time_s": previous_time}
            raise
    return {"contract": contract, "case": deepcopy(data), "segments": segments,
            "actual_stages": len(segments), "requested_stages": len(data["stages"])}
