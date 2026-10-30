"""Execute the complete frozen 14-unit/72-route spatial composite DFN block."""

import argparse
import json
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from battery_dfn_composite_assess import assess
from battery_dfn_composite_cases import GRIDS, case_data, route_plan
from battery_dfn_composite_pybamm import source_time
from battery_dfn_composite_traces import native_time, refusal
from run_battery_partial_plating_block import archive_verified, capture, compact_write, json_safe


def tasks(index, data):
    for name, kind, level, limit in route_plan(index):
        if kind == "refusal":
            yield name, partial(refusal, data, name, GRIDS[level])
        else:
            function = native_time if kind == "native" else source_time
            yield name, partial(function, data, GRIDS[level], limit=limit)


def main():
    import casadi
    import pybamm
    import pybammsolvers
    import scipy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.parent / (args.output.stem + "-routes")
    if any(p.exists() for p in (args.output, args.archive, args.summary, folder)):
        raise FileExistsError("New attempt requires new immutable output/checkpoint paths")
    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("Fixed actual source requires PyBaMM25.4.2/CasADi3.6.7")
    folder.mkdir(parents=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "claim": "a6d88a5825c943a8df641a7ea060c5a3affe1ea0",
        "attempt": 1,
        "status": "running",
        "mode": "development",
        "unit_denominator": 14,
        "route_denominator": 72,
        "history_denominator": 60,
        "refusal_denominator": 12,
        "units": [],
        "passed": 0,
        "grids": GRIDS,
        "reference_versions": {
            "pybamm": pybamm.__version__,
            "casadi": casadi.__version__,
            "pybammsolvers": pybammsolvers.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_COMPOSITE_DFN_EXPERIMENT.md",
        "source_scope": (
            "Actual modular DFN 2/1 single OCP; explicit synthetic materials; "
            "authored bounded CV and phase margin observation events; "
            "original source endpoint values retained."
        ),
    }
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting", "routes": 0}

    def checkpoint():
        brief = {k: v for k, v in report.items() if k not in ("units", "active_unit")}
        brief.update(
            elapsed_s=time.monotonic() - start,
            completed_routes=stage["routes"],
            stage=stage["name"],
            active_unit=report.get("active_unit"),
            units=[{k: v for k, v in u.items() if k != "routes"} for u in report["units"]],
        )
        path = folder / "manifest.json"
        temporary = folder / "manifest.tmp"
        temporary.write_text(json.dumps(json_safe(brief), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def heartbeat():
        while not stop.wait(20):
            elapsed = time.monotonic() - start
            speed = stage["routes"] / max(elapsed, 1.0)
            eta = (
                f"{(72 - stage['routes']) / speed:.1f}s"
                if speed and stage["routes"] < 72
                else "unknown"
            )
            print(
                f"COMPOSITE-DFN {len(report['units'])}/14 units {stage['routes']}/72 routes; "
                f"{stage['name']}; elapsed={elapsed:.1f}s; routes/s={speed:.3f}; ETA={eta}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for index in range(1, 15):
            item = {
                "id": f"CD{index:02d}",
                "inputs": case_data(index),
                "routes": {},
                "passed": False,
            }
            report["active_unit"] = {"id": item["id"], "inputs": item["inputs"]}
            for name, function in tasks(index, item["inputs"]):
                stage["name"] = item["id"] + " " + name
                print(stage["name"], flush=True)
                checkpoint()
                item["routes"][name] = capture(function)
                compact_write(folder / (item["id"] + "-" + name + ".json"), item["routes"][name])
                stage["routes"] += 1
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
            print(item["id"] + (" PASS" if item["passed"] else " FAIL"), flush=True)
            checkpoint()
        report.update(
            status="completed",
            units_executed=len(report["units"]),
            routes_executed=sum(len(u["routes"]) for u in report["units"]),
            histories_executed=sum(len(u["routes"]) for u in report["units"][:13]),
            refusals_executed=len(report["units"][13]["routes"]),
            route_exceptions=sum(
                "exception" in r for u in report["units"] for r in u["routes"].values()
            ),
            assessment_exceptions=sum("assessment_exception" in u for u in report["units"]),
        )
        report["evaluated_checks"] = sum(len(u.get("checks", {})) for u in report["units"])
        report["failed_checks"] = sum(len(u.get("failures", [])) for u in report["units"])
        report["all_passed"] = all(
            report[k] == v
            for k, v in {
                "passed": 14,
                "units_executed": 14,
                "routes_executed": 72,
                "histories_executed": 60,
                "refusals_executed": 12,
                "route_exceptions": 0,
                "assessment_exceptions": 0,
            }.items()
        )
        report["elapsed_s"] = time.monotonic() - start
        stage["name"] = "write complete compact raw"
        print(stage["name"], flush=True)
        compact_write(args.output, report)
        stage["name"] = "lossless gzip and byte identity"
        print(stage["name"], flush=True)
        archive = archive_verified(args.output, args.archive)
        summary = {k: v for k, v in report.items() if k != "units"}
        summary["units"] = [{k: v for k, v in u.items() if k != "routes"} for u in report["units"]]
        summary.update(archive=archive, archive_and_block_elapsed_s=time.monotonic() - start)
        compact_write(args.summary, summary)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "status",
                        "passed",
                        "routes_executed",
                        "evaluated_checks",
                        "failed_checks",
                        "elapsed_s",
                    )
                }
            ),
            flush=True,
        )
        checkpoint()
    finally:
        stop.set()
        thread.join()
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
