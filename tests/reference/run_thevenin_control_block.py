"""Run every fixed named-control unit, retaining all histories and refusal probes."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from run_thevenin_frequency_block import capture, convert
from thevenin_control_cases import UNITS, case_data, response_atol, response_names
from thevenin_control_pybamm import reference_probe
from thevenin_control_traces import (
    independent_rejection_fact,
    native_probe,
    native_trace,
    reference_trace,
)

ROUTES = ("native_standard", "native_tight", "reference_standard", "reference_tight")


def complete(trace, data):
    try:
        if trace["response_order"] != response_names(data):
            return False
        if len(trace["segments"]) != len(data["segments"]):
            return False
        start = 0.
        for part, planned in zip(trace["segments"], data["segments"], strict=True):
            times, rows = np.array(part["times_s"]), np.array(part["rows"])
            count = data["samples_per_segment"]
            expected_times = np.linspace(start, start+planned["duration_s"], count)
            if (times.shape != (count,) or rows.shape != (count, len(response_names(data)))
                    or not np.isfinite(rows).all()
                    or not np.allclose(times, expected_times, rtol=0, atol=1e-9)):
                return False
            if trace["implementation"] == "native":
                if part["outcome"] != "success" or part["termination"] is not None:
                    return False
            elif part["termination"] != "final time":
                return False
            profiles = np.array(part["profiles"])
            if data["diffusion_cells"] and (
                profiles.shape != (count, data["diffusion_cells"])
                or not np.isfinite(profiles).all()
            ):
                return False
            start += planned["duration_s"]
        return True
    except (KeyError, TypeError, ValueError):
        return False


def rows_of(trace):
    return np.array([row for part in trace["segments"] for row in part["rows"]])


def compare(left, right, data):
    a, b = rows_of(left), rows_of(right)
    errors, bounds = abs(a-b), response_atol(data)+1e-5*abs(b)
    profile_error = 0.
    if data["diffusion_cells"]:
        profiles = [np.array([row for part in trace["segments"] for row in part["profiles"]])
                    for trace in (left, right)]
        profile_error = float(np.max(abs(profiles[0]-profiles[1])))
    return {"maximum_absolute_error": np.max(errors, axis=0).tolist(),
            "maximum_tolerance_ratio": np.max(errors/bounds, axis=0).tolist(),
            "maximum_profile_absolute_error": profile_error, "comparisons": int(errors.size),
            "passed": bool(np.all(errors <= bounds) and profile_error <= 2e-5)}


def behavior(trace, data, index):
    rows, checks, metrics = rows_of(trace), {}, {}
    if data["mode"] in {"power", "resistance"}:
        errors = []
        for part, planned in zip(trace["segments"], data["segments"], strict=True):
            values = np.array(part["rows"])
            error = (values[:, 0]*values[:, 1]-planned["target"] if data["mode"] == "power"
                     else values[:, 0]-planned["target"]*values[:, 1])
            errors.extend(error)
        metrics["maximum_algebraic_target_residual"] = float(np.max(abs(np.array(errors))))
        checks["algebraic_target"] = metrics["maximum_algebraic_target_residual"] <= 1e-5
    if index == 3:
        u, r, p = data["ocv_intercept_V"]+data["initial_soc"], data["r0_ohm"], 3.
        sign = 1 if data["name"] == "high-root" else -1
        root = (u+sign*np.sqrt(u*u-4*r*p))/(2*r)
        metrics["independent_initial_current_root_A"] = root
        checks["declared_quadratic_branch"] = (
            abs(rows[0, 1]-root) <= 1e-5
            and np.all(rows[:, 1] >= data["branch_A"][0])
            and np.all(rows[:, 1] <= data["branch_A"][1]))
    if index == 9:
        checks["persistent_initial_current"] = abs(rows[0, 1]-data["initial_current_A"]) <= 1e-9
        checks["not_instant_algebraic_target"] = abs(rows[0, 0]*rows[0, 1]-.72) > .1
    if index == 10:
        q = response_names(data).index("controller_current_A")
        before, after = trace["segments"][0]["rows"][-1], trace["segments"][1]["rows"][0]
        metrics["switch_q_jump_A"] = abs(before[q]-after[q])
        checks["continuous_signed_reversal"] = (
            rows[0, 1] > 0 and rows[-1, 1] < 0 and metrics["switch_q_jump_A"] <= 1e-9)
    if index == 11:
        checks["released_unstable_direction_preserved"] = rows[-1, 1] > rows[0, 1]
    if index in {13, 16}:
        limit = data["current_limit_A"]
        checks["actual_cc_then_dynamic_cv"] = (
            abs(rows[0, 1]-limit) <= 1e-9 and np.max(rows[:, 1]-limit) > 1e-4)
    if index == 14:
        q = response_names(data).index("controller_current_A")
        metrics["initial_windup_A"] = abs(rows[0, q]-rows[0, 1])
        metrics["final_windup_A"] = abs(rows[-1, q]-rows[-1, 1])
        checks["distinct_state_and_antiwindup_recovery"] = (
            metrics["initial_windup_A"] >= .5
            and metrics["final_windup_A"] < metrics["initial_windup_A"])
    return {"checks": checks, "metrics": metrics, "passed": bool(all(checks.values()))}


def assess_history(record, data, index):
    checks, comparisons, behaviors = {}, {}, {}
    for route, trace in record["routes"].items():
        checks[f"{route}_complete"] = complete(trace, data)
        checks[f"{route}_conservation_and_capability"] = (
            trace.get("maximum_charge_balance_C", np.inf) <= 1e-5
            and trace.get("maximum_node_heat_balance_J", np.inf) <= 2e-4
            and trace.get("maximum_profile_mean_error", np.inf) <= 1e-7
            and trace.get("maximum_current_magnitude_A", np.inf)
            <= data["maximum_current_A"]+1e-9)
        checks[f"{route}_behavior"] = False
        if checks[f"{route}_complete"]:
            behaviors[route] = behavior(trace, data, index)
            checks[f"{route}_behavior"] = behaviors[route]["passed"]
    for left, right in (("native_standard", "reference_standard"),
                        ("native_tight", "reference_tight"),
                        ("native_standard", "native_tight"),
                        ("reference_standard", "reference_tight")):
        label = left+"_vs_"+right
        checks[label] = False
        if checks[f"{left}_complete"] and checks[f"{right}_complete"]:
            comparisons[label] = compare(record["routes"][left], record["routes"][right], data)
            checks[label] = comparisons[label]["passed"]
    record.update(checks=checks, comparisons=comparisons, behavior=behaviors,
                  passed=bool(all(checks.values())))


def assess_refusal(record, data):
    proof = independent_rejection_fact(data)
    native = record["routes"]["native_probe"]
    checks = {"independent_mathematical_fact": proof["passed"],
        "explicit_native_domain_refusal": (
            native.get("rejected") is True
            and native.get("exception_type") in {"ResidualDomainError", "ValueError"}
            and any(word in native.get("exception", "").lower()
                    for word in ("branch", "fold", "singular", "zero current"))),
        "input_state_unchanged": native.get("input_state_unchanged") is True}
    record.update(checks=checks, independent_proof=proof, passed=bool(all(checks.values())),
        upstream_interpretation="raw numerical behavior retained; not a uniqueness certificate")


def run_unit(record, index, progress, checkpoint):
    record["variants"] = []
    for j in range(len(UNITS[index-1])):
        data = case_data(index, j)
        item = {"inputs": data, "routes": {}, "passed": False}
        record["variants"].append(item)
        if data["expected"] == "trajectory":
            routes = [(route, partial(native_trace if route.startswith("native") else
                                      reference_trace, data, route.endswith("tight")))
                      for route in ROUTES]
        else:
            routes = [("native_probe", partial(native_probe, data)),
                      ("reference_probe", partial(reference_probe, data))]
        for name, calculate in routes:
            progress(f"C{index:02d} {data['name']} {name}")
            start = time.monotonic()
            item["routes"][name] = capture(calculate)
            item["routes"][name]["elapsed_s"] = time.monotonic()-start
            checkpoint()
        if data["expected"] == "trajectory":
            assess_history(item, data, index)
        else:
            assess_refusal(item, data)
    record["passed"] = all(item["passed"] for item in record["variants"])


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
        raise RuntimeError("fixed reference requires PyBaMM25.4.2 and CasADi3.6.7")
    data = {"mode": "development", "denominator": 16, "history_denominator": 56,
        "refusal_probe_denominator": 8, "units": [], "passed": 0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "execution_source": "development working tree; source_commit identifies base, "
                            "not frozen execution tree",
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_THEVENIN_CONTROL_EXPERIMENT.md",
        "reference_versions": {"pybamm": pybamm.__version__, "casadi": casadi.__version__,
                               "numpy": np.__version__, "scipy": scipy.__version__},
        "reference_contract": "actual released named external-circuit classes; "
                              "explicit PR5869 diffusion corrections only in C15/C16",
        "diffusion_correction_url": "https://github.com/pybamm-team/PyBaMM/pull/5869"}
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
                eta = f"{(16-done)/rate:.1f}s" if rate else "unknown"
                print(f"Thevenin control {done}/16; {stage['name']}; elapsed={elapsed:.1f}s; "
                      f"throughput={rate:.5f}/s; ETA={eta}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            checkpoint()
            for index in range(1, 17):
                record = {"id": f"C{index:02d}", "passed": False}
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
                print(f"C{index:02d} complete; passed={data['passed']}/{index}", flush=True)
        finally:
            stop.set()
            thread.join()
    return int(data["passed"] != 16)


if __name__ == "__main__":
    raise SystemExit(main())
