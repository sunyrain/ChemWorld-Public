"""Fixed 14-unit released ECM comparison; all failures and all units retained."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from battery_thevenin_cases import CASES, TIMES, native_case
from battery_thevenin_traces import (
    native_trace,
    reference_trace,
    response_bounds,
    response_names,
)


def valid(trace, index):
    try:
        rows = np.asarray(trace["rows"], dtype=float)
        return (trace["outcome"] == "success" and trace["end_time_s"] == 10.
                and np.array_equal(trace["times_s"], TIMES)
                and rows.shape == (21, len(response_names(index))) and np.isfinite(rows).all()
                and ((rows[:, 0] >= .1) & (rows[:, 0] <= 10.)).all()
                and (abs(rows[:, 1]) <= CASES[index-1]["limit_A"]+1e-9).all()
                and ((rows[:, 2] >= 0) & (rows[:, 2] <= 1)).all()
                and ((rows[:, 3:5] >= 250) & (rows[:, 3:5] <= 400)).all())
    except (KeyError, TypeError, ValueError):
        return False


def assess(record, index):
    traces = [*record["native"], *record["reference"]]
    checks = {"all_four_runs_complete": len(record["native"]) == len(record["reference"]) == 2
              and all(valid(t, index) for t in traces)}
    record["response_names"] = list(response_names(index))
    record["response_bounds"] = list(response_bounds(index))
    checks["native_balances"] = len(record["native"]) == 2 and all(
        t.get("maximum_charge_balance_C", float("inf")) <= 1e-8
        and t.get("maximum_node_heat_balance_J", float("inf")) <= 1e-6
        and t.get("chemical_total_energy_residual_J", "missing") is None
        for t in record["native"])
    if checks["all_four_runs_complete"]:
        n0, n1, r0, r1 = (np.asarray(t["rows"]) for t in traces)
        bounds = np.asarray(response_bounds(index))
        differences = {"native_reference": np.max(abs(n1-r1), axis=0),
                       "native_precision": np.max(abs(n0-n1), axis=0),
                       "reference_precision": np.max(abs(r0-r1), axis=0)}
        for name, fraction in (("native_reference", 1.), ("native_precision", .5),
                               ("reference_precision", 1/3)):
            checks[name] = bool((differences[name] <= bounds*fraction).all())
        record["maximum_absolute_differences"] = {k: v.tolist() for k, v in differences.items()}
    else:
        checks.update(native_reference=False, native_precision=False, reference_precision=False)
    record["checks"], record["passed"] = checks, bool(all(checks.values()))


def guarded(function):
    try:
        return function()
    except Exception as exc:
        return {"exception": str(exc), "traceback": traceback.format_exc()}


def run_unit(record, index, progress, checkpoint):
    spec, program = native_case(index)
    record.update(inputs=spec.to_dict(), program={"unit": program.unit,
        "setpoints": program.setpoints}, case=dict(CASES[index-1]), native=[], reference=[])
    for name, calculator in (("native", native_trace), ("reference", reference_trace)):
        for tight in (False, True):
            progress(f"T{index:02d} {name} {'tight' if tight else 'default'}")
            record[name].append(guarded(lambda calculator=calculator, tight=tight:
                                        calculator(index, tight)))
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
    data = {"mode": "development", "denominator": len(CASES), "units": [], "passed": 0,
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_THEVENIN_EXPERIMENT.md",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "pybamm_version": pybamm.__version__, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "times_s": TIMES.tolist()}
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:
        def checkpoint():
            def convert(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)

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
                eta = f"{(len(CASES)-done)/rate:.1f}s" if rate else "unknown"
                print(f"Thevenin {done}/{len(CASES)}; {stage['name']}; elapsed={elapsed:.1f}s; "
                      f"throughput={rate:.4f}/s; ETA={eta}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            checkpoint()
            for index in range(1, len(CASES)+1):
                record = {"id": f"T{index:02d}", "passed": False}
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
                print(f"T{index:02d} complete; passed={data['passed']}/{index}", flush=True)
        finally:
            stop.set()
            thread.join()
    return int(data["passed"] != len(CASES))


if __name__ == "__main__":
    raise SystemExit(main())
