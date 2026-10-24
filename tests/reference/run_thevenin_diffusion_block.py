"""Execute the fixed twelve-unit diffusion block without dropping failed routes or units."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from functools import partial
from itertools import pairwise
from pathlib import Path

import numpy as np
from run_thevenin_frequency_block import capture, convert
from thevenin_diffusion_cases import (
    CORRECTION_URL,
    FREQUENCIES,
    GRIDS,
    case_data,
    frequency_atol,
    frequency_names,
    mesh_atol,
    time_atol,
    time_names,
)
from thevenin_diffusion_traces import (
    analytic_frequency,
    native_frequency,
    native_time,
    reference_frequency,
    reference_time,
)


def valid_time(trace, index, cells):
    try:
        data = case_data(index)
        if trace["response_order"] != time_names(index) or trace["cells"] != cells:
            return False
        if len(trace["segments"]) != len(data["segments"]):
            return False
        for part in trace["segments"]:
            times, rows, profiles = (np.array(part[k]) for k in ("times_s", "rows", "profiles"))
            if (len(times) < 2 or np.any(np.diff(times) <= 0)
                    or rows.shape != (len(times), len(time_names(index)))
                    or profiles.shape != (len(times), cells)
                    or not all(np.isfinite(a).all() for a in (times, rows, profiles))):
                return False
            if not data["expected_surface_event"] and len(times) != data["samples_per_segment"]:
                return False
        final = trace["segments"][-1]
        if data["expected_surface_event"]:
            return ("minimum-surface-soc" in str(final["termination"])
                    or final["termination"] == "event: Minimum surface SoC")
        return (final.get("outcome") == "success" if trace["variant"] == "native"
                else final["termination"] == "final time")
    except (KeyError, TypeError, ValueError):
        return False


def aligned_rows(left, right, index):
    all_left, all_right, profile_errors = [], [], []
    start = 0.
    for lpart, rpart, planned in zip(left["segments"], right["segments"],
                                   case_data(index)["segments"], strict=True):
        common_end = min(lpart["end_time_s"], rpart["end_time_s"])
        planned_times = np.linspace(start, start+planned["duration_s"], 21)
        times = planned_times[planned_times <= common_end+1e-10]
        if common_end-times[-1] > 1e-10:
            times = np.r_[times, common_end]
        rows = []
        for part in (lpart, rpart):
            source = np.array(part["rows"])
            rows.append(np.column_stack([np.interp(times, part["times_s"], source[:, k])
                                         for k in range(source.shape[1])]))
        all_left.extend(rows[0])
        all_right.extend(rows[1])
        if len(lpart["profiles"][0]) == len(rpart["profiles"][0]):
            profiles = []
            for part in (lpart, rpart):
                source = np.array(part["profiles"])
                profiles.append(np.column_stack([np.interp(times, part["times_s"], source[:, k])
                                                 for k in range(source.shape[1])]))
            profile_errors.append(float(np.max(abs(profiles[0]-profiles[1]))))
        start += planned["duration_s"]
    return np.array(all_left), np.array(all_right), max(profile_errors, default=0.)


def time_comparison(left, right, index):
    a, b, profile_error = aligned_rows(left, right, index)
    errors, bounds = abs(a-b), time_atol(index)+1e-5*abs(b)
    end_error = abs(left["segments"][-1]["end_time_s"]-right["segments"][-1]["end_time_s"])
    return {"maximum_absolute_error": np.max(errors, axis=0).tolist(),
            "maximum_tolerance_ratio": np.max(errors/bounds, axis=0).tolist(),
            "maximum_profile_absolute_error": profile_error, "end_time_error_s": end_error,
            "comparisons": int(errors.size),
            "passed": bool(np.all(errors <= bounds)
                           and profile_error <= 2e-5 and end_error <= 1e-3)}


def time_assess(record, index):
    checks, comparisons, version_differences = {}, {}, {}
    for cells in GRIDS:
        paths = record["grids"][str(cells)]
        for name in ("native", "corrected"):
            trace = paths[name]
            complete = valid_time(trace, index, cells)
            checks[f"{cells}_{name}_complete"] = complete
            checks[f"{cells}_{name}_balances"] = (
                trace.get("maximum_charge_balance_C", np.inf) <= 1e-5
                and trace.get("maximum_profile_mean_error", np.inf) <= 1e-7
                and trace.get("maximum_node_heat_balance_J", np.inf) <= 2e-4)
            if case_data(index)["expected_surface_event"] and complete:
                final = trace["segments"][-1]["rows"][-1]
                checks[f"{cells}_{name}_surface_precedes_mean"] = (
                    abs(final[3]) <= 2e-5 and final[2] > 1e-3)
            if index == 10 and complete:
                checks[f"{cells}_{name}_closed_passive_cycle"] = (
                    abs(trace["final_net_charge_C"]) <= 1e-5
                    and trace["final_profile_from_initial"] <= 2e-5
                    and trace["final_work_J"] <= 1e-6
                    and trace["final_model_heat_J"] > 0
                    and abs(trace["final_work_J"]+trace["final_model_heat_J"]) <= 2e-3)
        checks[f"{cells}_same_grid_agreement"] = False
        if all(valid_time(paths[name], index, cells) for name in ("native", "corrected")):
            comparison = time_comparison(paths["native"], paths["corrected"], index)
            comparisons[str(cells)] = comparison
            checks[f"{cells}_same_grid_agreement"] = comparison["passed"]
        version_differences[str(cells)] = {
            "released_completed": valid_time(paths["released"], index, cells),
            "released_final_work_J": paths["released"].get("final_work_J"),
            "released_final_model_heat_J": paths["released"].get("final_model_heat_J"),
            "purpose": "original release discrepancy retained, not a native pass target"}
        if all(valid_time(paths[name], index, cells) for name in ("released", "corrected")):
            version_differences[str(cells)]["comparison"] = time_comparison(
                paths["released"], paths["corrected"], index)
    mesh = {}
    for name in ("native", "corrected"):
        traces = [record["grids"][str(n)][name] for n in GRIDS]
        checks[f"{name}_mesh_convergence"] = False
        if all(valid_time(trace, index, cells) for trace, cells in zip(traces, GRIDS, strict=True)):
            differences = []
            for left, right in pairwise(traces):
                a, b, _ = aligned_rows(left, right, index)
                differences.append(np.max(abs(a-b), axis=0))
            coarse, fine = differences
            event_errors = [abs(a["segments"][-1]["end_time_s"]-b["segments"][-1]["end_time_s"])
                            for a, b in pairwise(traces)]
            passed = bool(np.all(fine <= .65*coarse+1e-5) and np.all(fine <= mesh_atol(index)))
            if case_data(index)["expected_surface_event"]:
                passed = passed and event_errors[1] <= .65*event_errors[0]+1e-3
            mesh[name] = {"coarse_difference": coarse.tolist(), "fine_difference": fine.tolist(),
                          "fine_absolute_bound": mesh_atol(index).tolist(),
                          "event_time_differences_s": event_errors, "passed": passed}
            checks[f"{name}_mesh_convergence"] = passed
    record.update(checks=checks, comparisons=comparisons, mesh_convergence=mesh,
                  version_differences=version_differences, passed=bool(all(checks.values())))


def frequency_values(trace):
    return (np.array([p["response_real"] for p in trace["points"]])
            + 1j*np.array([p["response_imag"] for p in trace["points"]]))


def valid_frequency(trace, index):
    try:
        values = frequency_values(trace)
        return (trace["success"] and trace["response_order"] == frequency_names(index)
                and [p["frequency_Hz"] for p in trace["points"]] == FREQUENCIES.tolist()
                and values.shape == (13, len(frequency_names(index))) and np.isfinite(values).all()
                and all(p["success"] for p in trace["points"]))
    except (KeyError, TypeError, ValueError):
        return False


def frequency_assess(record, index):
    analytic, checks, comparisons = record["analytic"], {}, {}
    checks["analytic_complete"] = valid_frequency(analytic, index)
    analytic_errors = {name: [] for name in ("native", "corrected")}
    for cells in GRIDS:
        paths = record["grids"][str(cells)]
        for name in ("native", "corrected"):
            trace = paths[name]
            complete = valid_frequency(trace, index)
            checks[f"{cells}_{name}_complete"] = complete
            stationary = (trace.get("diagnostics", {}).get(
                "maximum_scaled_stationary_residual", np.inf) if name == "native" else max(
                    trace.get("maximum_stationary_rhs", np.inf),
                    trace.get("maximum_stationary_algebraic", np.inf)))
            checks[f"{cells}_{name}_stationary_and_linear"] = bool(
                stationary <= 1e-9 and len(trace.get("points", [])) == 13
                and all(p.get("relative_linear_residual", np.inf) is not None
                        and p.get("relative_linear_residual", np.inf) <= 1e-10
                        for p in trace.get("points", [])))
            if name == "native":
                checks[f"{cells}_state_unchanged"] = trace.get("input_state_unchanged") is True
            if complete and checks["analytic_complete"]:
                errors = abs(frequency_values(trace)-frequency_values(analytic))
                analytic_errors[name].append(np.max(errors, axis=0))
                comparisons[f"{cells}_{name}_analytic"] = {
                    "maximum_absolute_error": np.max(errors, axis=0).tolist(),
                    "maximum_tolerance_ratio": np.max(
                        errors/(1e-6+.01*abs(frequency_values(analytic))), axis=0).tolist()}
                if cells == 128:
                    checks[f"{name}_continuous_limit"] = bool(np.all(
                        errors <= 1e-6+.01*abs(frequency_values(analytic))))
        checks[f"{cells}_same_grid_agreement"] = False
        if all(valid_frequency(paths[name], index) for name in ("native", "corrected")):
            a, b = frequency_values(paths["native"]), frequency_values(paths["corrected"])
            errors, bounds = abs(a-b), frequency_atol(index)+1e-5*abs(b)
            comparisons[f"{cells}_native_corrected"] = {
                "maximum_absolute_error": np.max(errors, axis=0).tolist(),
                "maximum_tolerance_ratio": np.max(errors/bounds, axis=0).tolist(),
                "comparisons": int(errors.size)}
            checks[f"{cells}_same_grid_agreement"] = bool(np.all(errors <= bounds))
    for name, errors in analytic_errors.items():
        checks[f"{name}_mesh_convergence"] = bool(len(errors) == 3 and all(
            np.all(fine <= .65*coarse+1e-7) for coarse, fine in pairwise(errors)))
        checks.setdefault(f"{name}_continuous_limit", False)
    record.update(checks=checks, comparisons=comparisons, passed=bool(all(checks.values())))


def run_unit(record, index, progress, checkpoint):
    record.update(inputs=case_data(index), grids={})
    for cells in GRIDS:
        paths = record["grids"][str(cells)] = {}
        for name in ("native", "released", "corrected"):
            progress(f"D{index:02d} grid={cells} {name}")
            started = time.monotonic()
            if index < 11:
                calculate = partial(native_time, index, cells) if name == "native" else (
                    partial(reference_time, index, cells, name))
            else:
                calculate = partial(native_frequency, index, cells) if name == "native" else (
                    partial(reference_frequency, index, cells, name))
            paths[name] = capture(calculate)
            paths[name]["elapsed_s"] = time.monotonic()-started
            checkpoint()
    if index >= 11:
        record["analytic"] = capture(lambda: analytic_frequency(index))
        checkpoint()
        frequency_assess(record, index)
    else:
        time_assess(record, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import casadi
    import pybamm
    import scipy

    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("fixed references require PyBaMM25.4.2 and CasADi3.6.7")
    data = {"mode": "development", "denominator": 12, "units": [], "passed": 0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_THEVENIN_DIFFUSION_EXPERIMENT.md",
        "reference_versions": {"pybamm": pybamm.__version__, "casadi": casadi.__version__,
                               "numpy": np.__version__, "scipy": scipy.__version__},
        "correction_url": CORRECTION_URL, "grids": list(GRIDS),
        "reference_contract": ("released retained; corrected two expressions; "
                               "independent continuous limit")}
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:
        def checkpoint():
            data["elapsed_s"] = time.monotonic()-start
            output.seek(0)
            output.write(json.dumps(data, default=convert, allow_nan=False, indent=2)+"\n")
            output.truncate()
            output.flush()

        def progress(name):
            stage["name"] = name
            print(name, flush=True)

        def heartbeat():
            while not stop.wait(20):
                elapsed, done = time.monotonic()-start, len(data["units"])
                rate = done/max(elapsed, 1.)
                eta = f"{(12-done)/rate:.1f}s" if rate else "unknown"
                print(f"Thevenin diffusion {done}/12; {stage['name']}; elapsed={elapsed:.1f}s; "
                      f"throughput={rate:.5f}/s; ETA={eta}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            checkpoint()
            for index in range(1, 13):
                record = {"id": f"D{index:02d}", "passed": False}
                data["active_unit"] = record
                started = time.monotonic()
                try:
                    run_unit(record, index, progress, checkpoint)
                except Exception as exc:
                    record.update(exception=str(exc), traceback=traceback.format_exc())
                record["elapsed_s"] = time.monotonic()-started
                data["units"].append(record)
                data["passed"] += int(record["passed"])
                data.pop("active_unit")
                checkpoint()
                print(f"D{index:02d} complete; passed={data['passed']}/{index}", flush=True)
        finally:
            stop.set()
            thread.join()
    return int(data["passed"] != 12)


if __name__ == "__main__":
    raise SystemExit(main())
