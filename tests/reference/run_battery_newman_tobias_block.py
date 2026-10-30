"""Run all twelve fixed NT units and 52 routes, checkpointing every failure."""

import argparse
import json
import threading
import time
import traceback
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
from battery_newman_tobias_assess import assess
from battery_newman_tobias_cases import GRIDS, REFUSALS, case_data
from battery_newman_tobias_pybamm import reference_time
from battery_newman_tobias_traces import native_time, refusal


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
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    return value


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
        raise RuntimeError("fixed NT reference requires PyBaMM25.4.2/CasADi3.6.7")
    report = {
        "mode": "development",
        "status": "running",
        "claim": "311453b0",
        "denominator": 12,
        "route_denominator": 52,
        "time_history_denominator": 44,
        "refusal_denominator": 8,
        "passed": 0,
        "units": [],
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_NT_EXPERIMENT.md",
        "reference_versions": {
            "pybamm": pybamm.__version__,
            "casadi": casadi.__version__,
            "pybammsolvers": version("pybammsolvers"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "mapping": "Actual released NewmanTobias uniform-profile model; net OCV barrier "
        "compensated; effective solid conductivity mapped; original mesh node/face entries",
        "scope": "Fixed-T/ce, two average Li states, full porous electrical fields; no radial "
        "diffusion, dynamic salt, energy closure, EIS or SOH",
    }
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:

        def checkpoint():
            report["elapsed_s"] = time.monotonic() - start
            output.seek(0)
            output.write(json.dumps(json_safe(report), allow_nan=False, indent=2) + "\n")
            output.truncate()
            output.flush()

        def heartbeat():
            while not stop.wait(20):
                routes = sum(len(u["routes"]) for u in report["units"])
                routes += len(report.get("active_unit", {}).get("routes", {}))
                elapsed = time.monotonic() - start
                speed = routes / max(1.0, elapsed)
                eta = (52 - routes) / speed if speed else None
                print(
                    f"NT {len(report['units'])}/12 units; {routes}/52 routes; {stage['name']}; "
                    f"elapsed={elapsed:.1f}s; routes/s={speed:.3f}; ETA={eta}",
                    flush=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for index in range(1, 13):
                item = {
                    "id": f"NT{index:02d}",
                    "inputs": case_data(index),
                    "routes": {},
                    "passed": False,
                }
                report["active_unit"] = item
                tasks = []
                if index == 12:
                    tasks = [(name, partial(refusal, item["inputs"], name)) for name in REFUSALS]
                else:
                    for implementation, function in (
                        ("native", native_time),
                        ("reference", reference_time),
                    ):
                        tasks.extend(
                            (implementation + "_" + label, partial(function, item["inputs"], cells))
                            for label, cells in zip(
                                ("coarse", "fine"), GRIDS[implementation], strict=True
                            )
                        )
                for name, function in tasks:
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
            report["status"] = "completed"
            report["routes_executed"] = sum(len(u["routes"]) for u in report["units"])
            report["time_histories_executed"] = sum(
                len(u["routes"]) for u in report["units"] if u["inputs"]["expected"] != "refusals"
            )
            report["refusals_executed"] = sum(
                len(u["routes"]) for u in report["units"] if u["inputs"]["expected"] == "refusals"
            )
            report["route_exceptions"] = sum(
                "exception" in route for u in report["units"] for route in u["routes"].values()
            )
            report["assessment_exceptions"] = sum(
                "assessment_exception" in u for u in report["units"]
            )
            report["all_passed"] = (
                report["passed"] == 12
                and report["routes_executed"] == 52
                and report["time_histories_executed"] == 44
                and report["refusals_executed"] == 8
                and report["route_exceptions"] == 0
                and report["assessment_exceptions"] == 0
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
