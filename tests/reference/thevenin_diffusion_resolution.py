"""Independent continuous step response and fixed D08 refinement observations."""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq
from thevenin_diffusion_cases import case_data, native_case, time_atol
from thevenin_diffusion_traces import INTEGRALS, reference_model, time_summary

from chemworld.physchem.battery_thevenin import (
    AMBIENT_LOSS,
    CELL_T,
    CHARGE,
    JIG_T,
    MODEL_HEAT,
    SOC,
    WORK,
    thevenin_fields,
    thevenin_system,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram

RESOLUTION_GRIDS = (128, 256, 512, 1024)
COMMON_TIMES = np.linspace(0., .65, 21)
SERIES_TERMS = 512


def continuous_surface(time_s):
    data = case_data(8)
    if time_s < 0:
        raise ValueError("continuous diffusion time must be nonnegative")
    if time_s == 0:
        return data["initial_soc"]
    current, charge, tau = .5, data["capacity_C"], data["tau_s"]
    n = np.arange(1, SERIES_TERMS+1)
    series = np.sum(np.exp(-n*n*np.pi**2*time_s/tau)/(n*n))
    return data["initial_soc"]-current*time_s/charge-current*tau/charge*(
        1/3-2/np.pi**2*series)


def continuous_solution():
    data, root_queries = case_data(8), []

    def objective(time_s):
        root_queries.append(time_s)
        return continuous_surface(time_s)

    event = brentq(objective, 0., 20., xtol=1e-12, rtol=1e-12)
    surface = np.array([continuous_surface(t) for t in COMMON_TIMES])
    mean = data["initial_soc"]-.5*COMMON_TIMES/data["capacity_C"]
    voltage = 3+surface-.5*data["r0_ohm"]
    positive_times = [t for t in [*COMMON_TIMES, *root_queries] if t > 0]
    bounds = [2*abs(.5*data["tau_s"]/data["capacity_C"])/np.pi**2
              *np.exp(-np.pi**2*t/data["tau_s"]*(SERIES_TERMS+1)**2)/SERIES_TERMS
              for t in positive_times]
    return {"times_s": COMMON_TIMES.tolist(), "response_order": ["soc", "surface_soc", "voltage_V"],
        "rows": np.column_stack([mean, surface, voltage]).tolist(), "event_time_s": event,
        "series_terms": SERIES_TERMS, "root_queries_s": root_queries,
        "maximum_series_tail_bound": float(max(bounds)),
        "source": "continuous Cartesian Neumann step solution; not fitted to discrete responses"}


def native_resolution(cells):
    if cells not in RESOLUTION_GRIDS:
        raise ValueError("resolution block fixes four grids")
    base = native_case(8, 128)
    spec = replace(base, diffusion=replace(
        base.diffusion, faces=tuple(np.linspace(0., 1., cells+1))))
    program = ScalarControlProgram(((0., .5),), "A")
    system = thevenin_system(spec, program)
    system = replace(system, policy=replace(system.policy, rtol=1e-9, atol=1e-12))
    result = system.advance(spec.initial_state(), 20., evaluation_times_s=[*COMMON_TIMES, 20.])
    rows, profiles = [], []
    for state in result.trajectory:
        f, c = thevenin_fields(spec, state, program), state.coordinates
        rows.append([f["voltage_V"], f["current_A"], c[SOC], f["surface_soc"], c[CELL_T], c[JIG_T],
            c[CHARGE], c[WORK], c[MODEL_HEAT], c[AMBIENT_LOSS],
            f["diffusion_overpotential_V"], f["diffusion_heat_W"]])
        profiles.append(list(f["distributed_soc"]))
    segments = [{"times_s": [state.time_s for state in result.trajectory], "rows": rows,
        "profiles": profiles, "termination": result.stopping_event, "outcome": result.outcome,
        "end_time_s": result.final_state.time_s, "solver": dict(result.solver_diagnostic)}]
    return time_summary(8, cells, segments, "native")


def reference_resolution(cells):
    if cells not in RESOLUTION_GRIDS:
        raise ValueError("resolution block fixes four grids")
    simulation = reference_model(8, cells, "corrected")
    # CasadiSolver refines an event's output interval to 100 subpoints, then
    # interpolates. A .65-to-20 s output gap is not an accurate event bracket.
    # Internal solver nodes refine that bracket without changing observations.
    internal_times = np.r_[COMMON_TIMES, np.arange(66, 2001)*.01]
    solution = simulation.step(20., save=False, t_eval=internal_times,
                               inputs={"Current function [A]": .5})
    keys = ["Voltage [V]", "Current [A]", "SoC", "Surface SoC", "Cell temperature [K]",
        "Jig temperature [K]", *("Qualification integral: "+key for key in INTEGRALS),
        "Diffusion overpotential [V]", "Diagnostic diffusion heat [W]"]
    times = np.asarray(solution.t)
    rows = np.column_stack([np.asarray(solution[key](times)).reshape(-1) for key in keys])
    profiles = np.asarray(solution["Distributed SoC"].entries).T
    selected = observation_indices(times)
    segments = [{"times_s": times[selected].tolist(), "rows": rows[selected].tolist(),
        "profiles": profiles[selected].tolist(),
        "internal_solver_times_s": times.tolist(), "internal_solver_rows": rows.tolist(),
        "internal_solver_profiles": profiles.tolist(), "event_bracketing_step_s": .01,
        "termination": solution.termination, "end_time_s": float(times[-1])}]
    return time_summary(8, cells, segments, "corrected")


def observation_indices(times):
    indices = []
    for requested in COMMON_TIMES:
        matches = np.flatnonzero(np.isclose(times, requested, rtol=0., atol=1e-12))
        if len(matches) != 1:
            raise ValueError("fixed common observation absent or duplicated in solver output")
        indices.append(int(matches[0]))
    indices.append(len(times)-1)
    return indices


def sampled_rows(trace):
    part = trace["segments"][0]
    times = np.asarray(part["times_s"])
    if len(times) != 22 or not np.allclose(times[:21], COMMON_TIMES, rtol=0., atol=1e-12):
        raise ValueError("the fixed common observations must be present; no interpolation allowed")
    return np.asarray(part["rows"])[:21], np.asarray(part["profiles"])[:21]


def assess_resolution_grid(record, analytic):
    from run_thevenin_diffusion_block import valid_time

    checks, errors = {}, {}
    for name in ("native", "corrected"):
        trace = record[name]
        complete = valid_time(trace, 8, record["cells"])
        checks[name+"_complete"] = complete
        if not complete:
            continue
        try:
            rows, _ = sampled_rows(trace)
        except (KeyError, ValueError, TypeError):
            checks[name+"_common_samples"] = False
            continue
        checks[name+"_common_samples"] = True
        final = trace["segments"][0]["rows"][-1]
        checks[name+"_surface_precedes_mean"] = abs(final[3]) <= 2e-5 and final[2] > 1e-3
        checks[name+"_balances"] = (trace["maximum_charge_balance_C"] <= 1e-5
            and trace["maximum_profile_mean_error"] <= 1e-7
            and trace["maximum_node_heat_balance_J"] <= 2e-4)
        absolute = np.max(abs(rows[:, [2, 3, 0]]-np.asarray(analytic["rows"])), axis=0)
        event_error = abs(trace["segments"][0]["end_time_s"]-analytic["event_time_s"])
        errors[name] = {"maximum_absolute_error": absolute.tolist(),
            "event_time_error_s": event_error,
            "continuous_target_met": bool(np.all(absolute <= 5e-4) and event_error <= 1e-3)}
    checks["same_grid_agreement"] = False
    if all(checks.get(name+"_common_samples", False) for name in ("native", "corrected")):
        left, lp = sampled_rows(record["native"])
        right, rp = sampled_rows(record["corrected"])
        event_error = abs(record["native"]["segments"][0]["end_time_s"]
                          -record["corrected"]["segments"][0]["end_time_s"])
        absolute = abs(left-right)
        record["same_grid_comparison"] = {
            "maximum_tolerance_ratio": np.max(
                absolute/(time_atol(8)+1e-5*abs(right)), axis=0).tolist(),
            "maximum_profile_error": float(np.max(abs(lp-rp))), "event_time_error_s": event_error,
            "comparisons": int(absolute.size)}
        checks["same_grid_agreement"] = bool(np.all(absolute <= time_atol(8)+1e-5*abs(right))
            and np.max(abs(lp-rp)) <= 2e-5 and event_error <= 1e-3)
    record.update(checks=checks, analytic_comparisons=errors,
        internally_valid=bool(all(checks.values())),
        passed=bool(all(checks.values()) and len(errors) == 2
                    and all(error["continuous_target_met"] for error in errors.values())))


def assess_resolution_block(data):
    from itertools import pairwise

    units = data["units"]
    checks = {"all_four_grids_retained": len(units) == 4,
        "all_eight_paths_internally_valid": len(units) == 4 and all(
            unit.get("internally_valid", False) for unit in units),
        "analytic_series_certified": data["analytic"]["maximum_series_tail_bound"] <= 1e-12,
        "fixed_finest_continuous_target": len(units) == 4 and units[-1].get("passed", False)}
    for name in ("native", "corrected"):
        errors = [unit.get("analytic_comparisons", {}).get(name) for unit in units]
        checks[name+"_error_decreases"] = bool(len(errors) == 4 and all(errors) and all(
            np.all(np.array(fine["maximum_absolute_error"])
                   <= .65*np.array(coarse["maximum_absolute_error"])+1e-5)
            and fine["event_time_error_s"] <= .65*coarse["event_time_error_s"]+1e-3
            for coarse, fine in pairwise(errors)))
    data.update(checks=checks, passed=sum(unit.get("passed", False) for unit in units),
                resolution_qualified=bool(all(checks.values())))
