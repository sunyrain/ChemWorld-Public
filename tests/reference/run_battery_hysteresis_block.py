"""Complete fixed sixteen-unit OCP block; all 126 internal route attempts are retained."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from battery_hysteresis_assess import (
    aligned_rows,
    assess_frequency,
    assess_static,
    assess_time,
    complete,
    valid_refusal,
)
from battery_hysteresis_cases import MPM_GRIDS, SPM_GRIDS, UNITS, case_data
from battery_hysteresis_pybamm import (
    reference_frequency,
    reference_nonsmooth,
    reference_static,
    reference_time,
)
from battery_hysteresis_traces import native_frequency, native_static, native_time, refusal
from run_thevenin_frequency_block import capture, convert


def grid(data, implementation, fine):
    if data["size_distribution"] == "narrow":
        size, radial = MPM_GRIDS[implementation][int(fine)]
        return radial, size
    return SPM_GRIDS[implementation][int(fine)], None


def run_variant(item, data, progress, checkpoint):
    kind, routes = data["expected"], []
    if kind in {"trajectory", "voltage_event"}:
        for implementation in ("native", "reference"):
            function = native_time if implementation == "native" else reference_time
            for fine in (False, True):
                for tight in (False, True):
                    label = (implementation+("_fine" if fine else "_coarse")
                             +("_tight" if tight else "_standard"))
                    routes.append((label, partial(function, data, *grid(data, implementation, fine),
                                                  tight)))
    elif kind == "static_sweep":
        routes = [("native", partial(native_static, data)),
                  ("reference", partial(reference_static, data))]
    elif kind == "frequency":
        for implementation in ("native", "reference"):
            function = native_frequency if implementation == "native" else reference_frequency
            for fine in (False, True):
                for mode in ("single", "sigmoid"):
                    actual_mode = "single" if mode == "single" else "current sigmoid"
                    parameters = {**data, "negative_mode": actual_mode,
                                  "positive_mode": actual_mode}
                    label = implementation+("_fine_" if fine else "_coarse_")+mode
                    routes.append((label, partial(function, parameters,
                                                  grid(data, implementation, fine)[0])))
    elif kind == "nonsmooth_probe":
        routes = [("native_refusal", partial(refusal, data, nonsmooth=True)),
                  ("reference_sided_derivatives", partial(reference_nonsmooth, data))]
    elif kind == "domain_refusal":
        routes = [("native_refusal", partial(refusal, data))]
    else:
        raise ValueError("unknown fixed kind")
    item["routes"] = {}
    for label, function in routes:
        progress(data["name"]+" "+label)
        start = time.monotonic()
        item["routes"][label] = capture(function)
        item["routes"][label]["elapsed_s"] = time.monotonic()-start
        checkpoint()
    if kind in {"trajectory", "voltage_event"}:
        assess_time(item, data)
    elif kind == "static_sweep":
        assess_static(item, data)
    elif kind == "frequency":
        assess_frequency(item)
    elif kind == "nonsmooth_probe":
        checks = {"native_nonsmooth_refusal": valid_refusal(item["routes"]["native_refusal"]),
                  "source_sides_differ": False}
        reference = item["routes"]["reference_sided_derivatives"]
        if "points" in reference:
            checks["source_sides_differ"] = all(abs(p["left_derivative_s_inv_A"]-
                p["right_derivative_s_inv_A"]) > 1e-6 for p in reference["points"].values())
        item.update(checks=checks, passed=bool(all(checks.values())))
    else:
        item.update(passed=valid_refusal(item["routes"]["native_refusal"]))


def single_limit(record, prior):
    """H12 cross-checks the already executed H07 native fine-grid history."""
    item = record["variants"][0]
    a, b = item["routes"]["native_fine_tight"], prior["variants"][0]["routes"]["native_fine_tight"]
    data = item["inputs"]
    passed = False
    if complete(a, data) and complete(b, data):
        left, right = aligned_rows(a, b, data)
        error = float(np.max(abs(left-right)))
        record["single_radius_maximum_error"] = error
        passed = error <= 1e-7
    record["single_radius_limit_passed"] = passed
    record["passed"] &= passed


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
        raise RuntimeError("fixed independent reference requires PyBaMM25.4.2/CasADi3.6.7")
    data = {"mode": "development", "denominator": 16, "route_denominator": 126,
        "time_history_denominator": 104, "static_scan_denominator": 6,
        "frequency_route_denominator": 8, "probe_denominator": 8, "passed": 0, "units": [],
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_HYSTERESIS_EXPERIMENT.md",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "execution_source": "development working tree; source_commit identifies base, "
                            "not frozen execution tree",
        "reference_versions": {"pybamm": pybamm.__version__, "casadi": casadi.__version__,
                               "numpy": np.__version__, "scipy": scipy.__version__},
        "reference_mapping": "actual named released OCP classes; explicit input compensation "
                             "of U() endpoint barrier gives the same declared net affine branches",
        "qualification_scope": "synthetic isothermal single-material SPM and finite "
                               "narrow-PSD MPM; does not replace previous MPM mesh failures"}
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
                route_count = sum(len(v["routes"]) for u in data["units"]
                                  for v in u.get("variants", []))
                route_count += sum(len(v.get("routes", {})) for v in data.get(
                    "active_unit", {}).get("variants", []))
                eta = f"{(16-done)/rate:.1f}s" if rate else "unknown"
                print(f"OCP {done}/16 units; {route_count}/126 routes; {stage['name']}; "
                      f"elapsed={elapsed:.1f}s; rate={rate:.5f}/s; ETA={eta}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            checkpoint()
            for index in range(1, 17):
                record = {"id": f"H{index:02d}", "variants": [], "passed": False}
                data["active_unit"] = record
                started = time.monotonic()
                for j in range(len(UNITS[index-1])):
                    parameters = case_data(index, j)
                    item = {"inputs": parameters, "passed": False}
                    record["variants"].append(item)
                    try:
                        run_variant(item, parameters, progress, checkpoint)
                    except Exception as exc:
                        item.update(exception=str(exc), traceback=traceback.format_exc())
                record["passed"] = all(v["passed"] for v in record["variants"])
                if index == 12:
                    try:
                        single_limit(record, data["units"][6])
                    except Exception as exc:
                        record.update(passed=False, exception=str(exc),
                                      traceback=traceback.format_exc())
                record["elapsed_s"] = time.monotonic()-started
                data["units"].append(record)
                data["passed"] += int(record["passed"])
                data.pop("active_unit")
                checkpoint()
                print(f"H{index:02d} complete; passed={data['passed']}/{index}", flush=True)
        finally:
            stop.set()
            thread.join()
    return int(data["passed"] != 16)


if __name__ == "__main__":
    raise SystemExit(main())
