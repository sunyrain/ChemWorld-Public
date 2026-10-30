"""Complete fixed x-full thermal block, retaining every route and failure."""

import argparse
import json
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from battery_spatial_thermal_assess import assess
from battery_spatial_thermal_cases import GRIDS, REFUSALS, case_data
from battery_spatial_thermal_pybamm import reference_time
from battery_spatial_thermal_traces import native_time, refusal


def capture(function):
    start = time.monotonic()
    try:
        result = function()
    except Exception as exc:
        result = {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
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


def run_unit(item, progress, checkpoint):
    data, tasks = item["inputs"], []
    if data["expected"] == "refusals":
        tasks = [(name, partial(refusal, data, name)) for name in REFUSALS]
    else:
        for implementation, function in (("native", native_time), ("reference", reference_time)):
            for label, cells in zip(("coarse", "fine"), GRIDS[implementation], strict=True):
                tasks.append((implementation + "_" + label, partial(function, data, cells)))
    item["routes"] = {}
    for name, function in tasks:
        item["active_route"] = name
        progress(item["id"] + " " + name)
        checkpoint()
        item["routes"][name] = capture(function)
        item.pop("active_route")
        checkpoint()
    try:
        assess(item)
    except Exception as exc:
        item.update(passed=False, assessment_exception=str(exc), traceback=traceback.format_exc())


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
        raise RuntimeError("fixed reference requires PyBaMM25.4.2/CasADi3.6.7")
    report = {
        "mode": "development",
        "status": "running",
        "claim_base_commit": "db29c3e2",
        "execution_source": "Development working tree; claim base is not a frozen source tree",
        "denominator": 12,
        "route_denominator": 52,
        "time_history_denominator": 44,
        "refusal_denominator": 8,
        "passed": 0,
        "units": [],
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_XTHERMAL_EXPERIMENT.md",
        "reference_versions": {
            "pybamm": pybamm.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "reference_mapping": "Released actual x-full SPM; finite CC capacities; explicit synthetic "
        "thermal stack; constant net OCV with source barrier compensation; "
        "source CC conductivity infinity verified by evaluated zero heat",
        "qualification_scope": "Finite n/s/p thermal field, fixed ambient and constant OCV; "
        "not a commercial cell or general electrode thermodynamics",
    }
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:

        def checkpoint():
            report["elapsed_s"] = time.monotonic() - start
            output.seek(0)
            output.write(json.dumps(json_safe(report), allow_nan=False, indent=2) + "\n")
            output.truncate()
            output.flush()

        def progress(name):
            stage["name"] = name
            print(name, flush=True)

        def heartbeat():
            while not stop.wait(20):
                routes = sum(len(u["routes"]) for u in report["units"])
                routes += len(report.get("active_unit", {}).get("routes", {}))
                elapsed = time.monotonic() - start
                rate = routes / max(1.0, elapsed)
                eta = f"{(52 - routes) / rate:.1f}s" if rate else "unknown"
                print(
                    f"XTHERMAL {len(report['units'])}/12 units; {routes}/52 routes; "
                    f"{stage['name']}; elapsed={elapsed:.1f}s; routes/s={rate:.4f}; ETA={eta}",
                    flush=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            checkpoint()
            for index in range(1, 13):
                item = {"id": f"XT{index:02d}", "inputs": case_data(index), "passed": False}
                report["active_unit"] = item
                run_unit(item, progress, checkpoint)
                report["units"].append(item)
                report["passed"] += int(item["passed"])
                report.pop("active_unit")
                checkpoint()
                print(f"XT{index:02d} complete; passed={report['passed']}/{index}", flush=True)
            report["executed_routes"] = sum(len(u["routes"]) for u in report["units"])
            report["failed_units"] = [u["id"] for u in report["units"] if not u["passed"]]
            report["route_exceptions"] = [
                {"unit": u["id"], "route": name, "exception": route["exception"]}
                for u in report["units"]
                for name, route in u["routes"].items()
                if "exception" in route and not route.get("rejected")
            ]
            report["execution_complete"] = (
                len(report["units"]) == 12 and report["executed_routes"] == 52
            )
            report["status"] = "completed"
            checkpoint()
        finally:
            if report["status"] == "running":
                report["status"] = "interrupted"
                report["execution_complete"] = False
                report["completed_routes"] = sum(len(u["routes"]) for u in report["units"])
                report["completed_routes"] += len(report.get("active_unit", {}).get("routes", {}))
                checkpoint()
            stop.set()
            thread.join()
    return int(report["passed"] != 12)


if __name__ == "__main__":
    raise SystemExit(main())
