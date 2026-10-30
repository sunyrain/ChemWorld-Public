"""Run the original fixed 13-unit/64-route plating block; never select successes."""

import argparse
import json
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from battery_plating_assess import assess
from battery_plating_cases import GRIDS, REFUSALS, case_data
from battery_plating_pybamm import reference_time
from battery_plating_traces import native_time, refusal


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


def tasks(index, data):
    if index == 13:
        return [(name, partial(refusal, data, name)) for name in REFUSALS]
    routes = [
        (name + "_" + level, partial(function, data, cells))
        for name, function in (("native", native_time), ("reference", reference_time))
        for level, cells in zip(("coarse", "fine"), GRIDS[name], strict=True)
    ]
    if index == 1:
        routes.extend(
            ("old_" + name + "_" + level, partial(function, data, cells, old_spm=True))
            for name, function in (("native", native_time), ("reference", reference_time))
            for level, cells in zip(("coarse", "fine"), GRIDS[name], strict=True)
        )
    return routes


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
        raise RuntimeError("fixed actual source requires PyBaMM25.4.2/CasADi3.6.7")
    report = {
        "claim": "dd36fe64",
        "mode": "development",
        "status": "running",
        "unit_denominator": 13,
        "route_denominator": 64,
        "history_denominator": 52,
        "refusal_denominator": 12,
        "units": [],
        "passed": 0,
        "grids": GRIDS,
        "reference_versions": {
            "pybamm": pybamm.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "protocol": "workstreams/world_foundation/NATIVE_CONV46_PLATING_EXPERIMENT.md",
        "source_scope": (
            "Actual SPM plating/stripping and native shared residual; "
            "authored algebraic limited external source and explicit OCP/event-boundary "
            "adapters. Synthetic inputs; no lifespan prediction."
        ),
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
                count = sum(len(u["routes"]) for u in report["units"]) + len(
                    report.get("active_unit", {}).get("routes", {})
                )
                elapsed = time.monotonic() - start
                speed = count / max(elapsed, 1.0)
                eta = (64 - count) / speed if speed else None
                print(
                    f"PLATING {len(report['units'])}/13 units; {count}/64 routes; {stage['name']}; "
                    f"elapsed={elapsed:.1f}s; routes/s={speed:.3f}; ETA={eta}",
                    flush=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for index in range(1, 14):
                item = {
                    "id": f"PL{index:02d}",
                    "inputs": case_data(index),
                    "routes": {},
                    "passed": False,
                }
                report["active_unit"] = item
                for name, function in tasks(index, item["inputs"]):
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
                units_executed=len(report["units"]),
                routes_executed=sum(len(u["routes"]) for u in report["units"]),
                histories_executed=sum(len(u["routes"]) for u in report["units"][:12]),
                refusals_executed=len(report["units"][12]["routes"]),
                route_exceptions=sum(
                    "exception" in r and "rejected" not in r
                    for u in report["units"]
                    for r in u["routes"].values()
                ),
                assessment_exceptions=sum("assessment_exception" in u for u in report["units"]),
            )
            report["evaluated_checks"] = sum(len(u.get("checks", {})) for u in report["units"])
            report["failed_checks"] = sum(len(u.get("failures", [])) for u in report["units"])
            report["all_passed"] = all(
                report[k] == v
                for k, v in {
                    "passed": 13,
                    "units_executed": 13,
                    "routes_executed": 64,
                    "histories_executed": 52,
                    "refusals_executed": 12,
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
                for k in (
                    "status",
                    "passed",
                    "unit_denominator",
                    "routes_executed",
                    "evaluated_checks",
                    "failed_checks",
                    "elapsed_s",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
