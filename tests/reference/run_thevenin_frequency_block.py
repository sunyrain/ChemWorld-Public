"""Run the complete fixed 12-unit empirical-circuit frequency qualification."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from run_battery_thevenin_block import guarded
from thevenin_frequency_cases import (
    FREQUENCIES,
    case_data,
    native_case,
    response_order,
    response_tolerances,
)
from thevenin_frequency_traces import closed_trace, native_trace, reference_trace


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def capture(calculate):
    trace = guarded(calculate)
    try:
        # A nonfinite reference must not prevent the other units from running
        # or be silently clipped to a plausible quantitative value.
        return json.loads(json.dumps(trace, default=convert, allow_nan=False))
    except (TypeError, ValueError) as exc:
        return {"exception": "invalid numerical trace: "+str(exc), "raw_repr": repr(trace)}


def valid(trace, index):
    try:
        points = trace["points"]
        return (trace["success"] and trace["response_order"] == response_order(index)
            and [p["frequency_Hz"] for p in points] == FREQUENCIES.tolist()
            and all(p["success"] and len(p["response_real"]) == len(p["response_imag"])
                == len(response_order(index)) and np.isfinite(p["response_real"]).all()
                and np.isfinite(p["response_imag"]).all() for p in points))
    except (KeyError, TypeError, ValueError):
        return False


def values(trace):
    return np.array([p["response_real"] for p in trace["points"]]) + 1j*np.array(
        [p["response_imag"] for p in trace["points"]])


def assess(record, index):
    names = ("native", "reference", "closed")
    checks = {"all_three_complete": all(valid(record[n], index) for n in names)}
    native, reference = record["native"], record["reference"]
    checks["native_stationary_and_linear"] = (
        native.get("input_state_unchanged") is True
        and native.get("diagnostics", {}).get("physical_time_advanced_s") == 0.
        and native.get("diagnostics", {}).get("maximum_scaled_stationary_residual", np.inf) <= 1e-9
        and len(native.get("points", [])) == 13
        and all(p.get("relative_linear_residual", np.inf) is not None
                and p.get("relative_linear_residual", np.inf) <= 1e-10
                for p in native.get("points", [])))
    checks["reference_stationary_and_linear"] = (
        reference.get("maximum_stationary_rhs", np.inf) <= 1e-10
        and reference.get("maximum_stationary_algebraic", np.inf) <= 1e-10
        and len(reference.get("points", [])) == 13
        and all(p.get("relative_linear_residual", np.inf) is not None
                and p.get("relative_linear_residual", np.inf) <= 1e-10
                for p in reference.get("points", [])))
    atol, rtol = response_tolerances(index)
    record.update(response_order=response_order(index), atol=atol.tolist(), rtol=rtol.tolist())
    comparisons = {}
    for left, right in (("native", "reference"), ("native", "closed"), ("reference", "closed")):
        name = left+"_"+right
        checks[name] = False
        if checks["all_three_complete"]:
            a, b = values(record[left]), values(record[right])
            errors, bounds = abs(a-b), atol+rtol*abs(b)
            comparisons[name] = {"maximum_absolute_error": np.max(errors, axis=0).tolist(),
                "maximum_tolerance_ratio": np.max(errors/bounds, axis=0).tolist()}
            checks[name] = bool((errors <= bounds).all())
    record.update(comparisons=comparisons, checks=checks, passed=bool(all(checks.values())))


def run_unit(record, index, progress, checkpoint):
    record.update(inputs=case_data(index), native_spec=native_case(index).to_dict())
    for name, calculate in (("native", native_trace), ("reference", reference_trace),
                            ("closed", closed_trace)):
        progress(f"F{index:02d} {name}")
        record[name] = capture(lambda calculate=calculate: calculate(index))
        checkpoint()
    assess(record, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pybamm
    import scipy

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("fixed reference requires PyBaMM 25.4.2")
    data = {"mode": "development", "denominator": 12, "units": [], "passed": 0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_THEVENIN_FREQUENCY_EXPERIMENT.md",
        "pybamm_version": pybamm.__version__, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "frequencies_Hz": FREQUENCIES.tolist()}
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:
        def checkpoint():
            data["elapsed_s"] = time.monotonic()-start
            output.seek(0)
            output.write(json.dumps(data, default=convert, indent=2, allow_nan=False)+"\n")
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
                print(f"Thevenin frequency {done}/12; {stage['name']}; elapsed={elapsed:.1f}s; "
                      f"throughput={rate:.4f}/s; ETA={eta}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            checkpoint()
            for index in range(1, 13):
                record = {"id": f"F{index:02d}", "passed": False}
                data["active_unit"] = record
                unit_start = time.monotonic()
                try:
                    run_unit(record, index, progress, checkpoint)
                except Exception as exc:
                    record.update(exception=str(exc), traceback=traceback.format_exc())
                record["elapsed_s"] = time.monotonic()-unit_start
                data["units"].append(record)
                data["passed"] += int(record["passed"])
                data.pop("active_unit")
                checkpoint()
                print(f"F{index:02d} complete; passed={data['passed']}/{index}", flush=True)
        finally:
            stop.set()
            thread.join()
    return int(data["passed"] != 12)


if __name__ == "__main__":
    raise SystemExit(main())
