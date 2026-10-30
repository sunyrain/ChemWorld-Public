"""Run the original fixed 12-unit/62-route MPM plating block; never select successes."""

import argparse
import gzip
import hashlib
import json
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from battery_mpm_plating_assess import assess
from battery_mpm_plating_cases import GRIDS, REFUSALS, case_data
from battery_mpm_plating_pybamm import reference_time
from battery_mpm_plating_traces import native_time, refusal


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
    if index == 12:
        return [(name, partial(refusal, data, name)) for name in REFUSALS]
    routes = [
        (name + "_" + level, partial(function, data, *grid))
        for name, function in (("native", native_time), ("reference", reference_time))
        for level, grid in zip(("coarse", "fine"), GRIDS[name], strict=True)
    ]
    if index == 1:
        routes.extend(
            ("single_spm_" + level, partial(native_time, data, *grid, single_spm=True))
            for level, grid in zip(("coarse", "fine"), GRIDS["native"], strict=True)
        )
    if index == 2:
        routes.extend(
            ("old_" + name + "_" + level, partial(function, data, *grid, old_mpm=True))
            for name, function in (("native", native_time), ("reference", reference_time))
            for level, grid in zip(("coarse", "fine"), GRIDS[name], strict=True)
        )
    return routes


def archive_result(raw_path, archive_path):
    print(f"MPM archive start: {raw_path.stat().st_size} raw bytes", flush=True)
    raw_digest = hashlib.sha256()
    started = last = time.monotonic()
    completed = 0
    with (
        raw_path.open("rb") as source,
        archive_path.open("xb") as destination,
        gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as zipped,
    ):
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            raw_digest.update(chunk)
            zipped.write(chunk)
            completed += len(chunk)
            if time.monotonic() - last >= 20:
                elapsed = time.monotonic() - started
                print(
                    f"MPM archive {completed}/{raw_path.stat().st_size} bytes; "
                    f"elapsed={elapsed:.1f}s; bytes/s={completed / elapsed:.1f}",
                    flush=True,
                )
                last = time.monotonic()
    archive_digest = hashlib.sha256()
    with gzip.open(archive_path, "rb") as archived:
        for chunk in iter(lambda: archived.read(1024 * 1024), b""):
            archive_digest.update(chunk)
    if archive_digest.digest() != raw_digest.digest():
        raise RuntimeError("lossless archive identity check failed")
    print(
        f"MPM archive complete and verified: {archive_path.stat().st_size} gzip bytes", flush=True
    )
    return {
        "raw_path": str(raw_path),
        "archive_path": str(archive_path),
        "raw_bytes": raw_path.stat().st_size,
        "archive_bytes": archive_path.stat().st_size,
        "uncompressed_sha256": raw_digest.hexdigest(),
        "archive_identical": True,
    }


def main():
    import casadi
    import pybamm
    import scipy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.archive.exists():
        raise FileExistsError("raw or archive already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("fixed actual source requires PyBaMM25.4.2/CasADi3.6.7")
    report = {
        "claim": "e62ab806",
        "mode": "development",
        "status": "running",
        "unit_denominator": 12,
        "route_denominator": 62,
        "history_denominator": 50,
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
        "protocol": "workstreams/world_foundation/NATIVE_CONV46_MPM_PLATING_EXPERIMENT.md",
        "source_scope": (
            "Actual MPM plating/stripping and native shared residual; "
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
                eta = (62 - count) / speed if speed else None
                print(
                    f"PLATING {len(report['units'])}/12 units; {count}/62 routes; {stage['name']}; "
                    f"elapsed={elapsed:.1f}s; routes/s={speed:.3f}; ETA={eta}",
                    flush=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for index in range(1, 13):
                item = {
                    "id": f"MP{index:02d}",
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
                histories_executed=sum(len(u["routes"]) for u in report["units"][:11]),
                refusals_executed=len(report["units"][11]["routes"]),
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
                    "passed": 12,
                    "units_executed": 12,
                    "routes_executed": 62,
                    "histories_executed": 50,
                    "refusals_executed": 12,
                    "route_exceptions": 0,
                    "assessment_exceptions": 0,
                }.items()
            )
            checkpoint()
        finally:
            stop.set()
            thread.join()
    print(json.dumps(archive_result(args.output, args.archive)), flush=True)
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
