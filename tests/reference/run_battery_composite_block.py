"""Complete fixed three-material block, retaining every route and failure."""

import argparse
import json
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from battery_composite_assess import assess
from battery_composite_cases import GRIDS, REFUSALS, case_data
from battery_composite_pybamm import reference_time
from battery_composite_traces import native_time, refusal


def capture(function):
    start = time.monotonic()
    try:
        result = function()
    except Exception as exc:
        result = {"exception_type": type(exc).__name__, "exception": str(exc),
                  "traceback": traceback.format_exc()}
    return {**result, "elapsed_s": time.monotonic()-start}


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
                tasks.append((implementation+"_"+label, partial(function, data, cells)))
        if data.get("same_material_limit"):
            tasks.extend(("single_"+label, partial(reference_time, data, cells, single=True))
                for label, cells in zip(("coarse", "fine"), GRIDS["reference"], strict=True))
    item["routes"] = {}
    for name, function in tasks:
        progress(item["id"]+" "+name)
        item["routes"][name] = capture(function)
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
    report = {"mode": "development", "claim_base_commit": "c01836b3",
        "execution_source": "Development working tree; claim base is not a frozen source tree",
        "denominator": 12, "route_denominator": 54, "time_history_denominator": 46,
        "multiphase_history_denominator": 44, "single_material_history_denominator": 2,
        "refusal_denominator": 8, "passed": 0, "units": [],
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_COMPOSITE_EXPERIMENT.md",
        "reference_versions": {"pybamm": pybamm.__version__, "casadi": casadi.__version__,
                               "numpy": np.__version__, "scipy": scipy.__version__},
        "reference_mapping": "Released actual three-material SPM; explicit synthetic curves; "
                             "U() barrier compensation; original surface observations retained",
        "qualification_scope": "Isothermal two-negative/one-positive single-valued materials; "
                               "not a complete Chen2020 composite material reproduction"}
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:
        def checkpoint():
            report["elapsed_s"] = time.monotonic()-start
            output.seek(0)
            output.write(json.dumps(json_safe(report), allow_nan=False, indent=2)+"\n")
            output.truncate()
            output.flush()

        def progress(name):
            stage["name"] = name
            print(name, flush=True)

        def heartbeat():
            while not stop.wait(20):
                routes = sum(len(u["routes"]) for u in report["units"])
                routes += len(report.get("active_unit", {}).get("routes", {}))
                elapsed = time.monotonic()-start
                rate = routes/max(1., elapsed)
                eta = f"{(54-routes)/rate:.1f}s" if rate else "unknown"
                print(f"COMPOSITE {len(report['units'])}/12 units; {routes}/54 routes; "
                      f"{stage['name']}; elapsed={elapsed:.1f}s; routes/s={rate:.4f}; ETA={eta}",
                      flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            checkpoint()
            for index in range(1, 13):
                item = {"id": f"CP{index:02d}", "inputs": case_data(index), "passed": False}
                report["active_unit"] = item
                run_unit(item, progress, checkpoint)
                report["units"].append(item)
                report["passed"] += int(item["passed"])
                report.pop("active_unit")
                checkpoint()
                print(f"CP{index:02d} complete; passed={report['passed']}/{index}", flush=True)
        finally:
            stop.set()
            thread.join()
    return int(report["passed"] != 12)


if __name__ == "__main__":
    raise SystemExit(main())
