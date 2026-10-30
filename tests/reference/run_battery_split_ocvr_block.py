"""Execute the fixed 17-unit/66-route development block without selective retries."""

import argparse
import json
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from battery_split_ocvr_assess import assess
from battery_split_ocvr_cases import PRECISIONS, REFUSALS, case_data
from battery_split_ocvr_pybamm import reference_time
from battery_split_ocvr_traces import analysis, native_time, refusal


def capture(function):
    start = time.monotonic()
    try:
        result = function()
    except Exception as exc:
        result = {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "partial_history": getattr(exc, "partial_history", None),
        }
    return {**result, "elapsed_s": time.monotonic() - start}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def unit_tasks(index, data):
    if index == 15:
        return [(name, partial(refusal, data, name)) for name in REFUSALS]
    if index in (16, 17):
        kind = "EIS" if index == 16 else "SOH"
        names = ("base", "zero-R", "nonlinear") if index == 16 else ("base", "shifted", "nonlinear")
        return [(name, partial(analysis, kind, name)) for name in names]
    implementations = (
        (("native", native_time), ("reference", reference_time))
        if index <= 12
        else (("native", native_time),)
    )
    return [
        (name + "_" + precision, partial(function, data, precision))
        for name, function in implementations
        for precision in PRECISIONS
    ]


def main():
    import casadi
    import pybamm
    import scipy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("fixed external reference requires PyBaMM25.4.2/CasADi3.6.7")
    report = {
        "mode": "development",
        "status": "running",
        "claim": "1431a2fb",
        "denominator": 17,
        "route_denominator": 66,
        "time_history_denominator": 52,
        "refusal_denominator": 8,
        "analysis_denominator": 6,
        "eis_frequency_points": 51,
        "soh_soc_points": 15,
        "passed": 0,
        "units": [],
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_SPLIT_OCVR_EXPERIMENT.md",
        "reference_versions": {
            "pybamm": pybamm.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scope": "Actual SplitOCVR only prescribed-current theta_n/theta_p/Q. "
        "Native CV/W and EIS/SOH use independent polynomial/closed-form oracles; "
        "no original-source CV/W invented.",
    }
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:

        def checkpoint():
            report["elapsed_s"] = time.monotonic() - start
            output.seek(0)
            output.write(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
            output.truncate()
            output.flush()

        def heartbeat():
            while not stop.wait(20):
                routes = sum(len(u["routes"]) for u in report["units"]) + len(
                    report.get("active_unit", {}).get("routes", {})
                )
                elapsed = time.monotonic() - start
                speed = routes / max(elapsed, 1.0)
                eta = (66 - routes) / speed if speed else None
                print(
                    f"SPLIT_OCVR {len(report['units'])}/17 units; {routes}/66 routes; "
                    f"{stage['name']}; elapsed={elapsed:.1f}s; routes/s={speed:.3f}; ETA={eta}",
                    flush=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for index in range(1, 18):
                item = {
                    "id": f"S{index:02d}",
                    "inputs": case_data(index),
                    "routes": {},
                    "passed": False,
                }
                report["active_unit"] = item
                for name, function in unit_tasks(index, item["inputs"]):
                    item["active_route"] = name
                    stage["name"] = item["id"] + " " + name
                    print(stage["name"], flush=True)
                    checkpoint()
                    item["routes"][name] = capture(function)
                    item.pop("active_route")
                    checkpoint()
                try:
                    assess(item)
                except Exception as exc:
                    item.update(
                        passed=False,
                        assessment_exception=str(exc),
                        assessment_traceback=traceback.format_exc(),
                    )
                report["units"].append(item)
                report.pop("active_unit")
                report["passed"] = sum(u["passed"] for u in report["units"])
                print(f"{item['id']} {'PASS' if item['passed'] else 'FAIL'}", flush=True)
                checkpoint()
            report.update(
                status="completed",
                routes_executed=sum(len(u["routes"]) for u in report["units"]),
                time_histories_executed=sum(len(u["routes"]) for u in report["units"][:14]),
                refusals_executed=len(report["units"][14]["routes"]),
                analyses_executed=sum(len(u["routes"]) for u in report["units"][15:]),
                route_exceptions=sum(
                    "exception" in r for u in report["units"] for r in u["routes"].values()
                ),
                assessment_exceptions=sum("assessment_exception" in u for u in report["units"]),
            )
            report["all_passed"] = all(
                report[k] == v
                for k, v in {
                    "passed": 17,
                    "routes_executed": 66,
                    "time_histories_executed": 52,
                    "refusals_executed": 8,
                    "analyses_executed": 6,
                    "route_exceptions": 0,
                    "assessment_exceptions": 0,
                }.items()
            )
            checkpoint()
        finally:
            stop.set()
            thread.join()
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "passed", "denominator", "routes_executed", "elapsed_s")
            }
        ),
        flush=True,
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
