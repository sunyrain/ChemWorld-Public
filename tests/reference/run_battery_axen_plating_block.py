"""Run every fixed Axen/plating route with independent compact checkpoints."""

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
from battery_axen_plating_assess import assess
from battery_axen_plating_cases import GRIDS, REFUSALS, case_data
from battery_axen_plating_pybamm import reference_time
from battery_axen_plating_traces import native_time, refusal


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
    if index in (1, 2):
        routes += [
            ("old_" + name + "_" + level, partial(function, data, cells, baseline=True))
            for name, function in (("native", native_time), ("reference", reference_time))
            for level, cells in zip(("coarse", "fine"), GRIDS[name], strict=True)
        ]
    return routes


def compact_write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(json_safe(value), stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def archive_verified(raw, archive):
    digest = hashlib.sha256()
    with (
        raw.open("rb") as src,
        archive.open("xb") as destination,
        gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as output,
    ):
        while chunk := src.read(1 << 20):
            digest.update(chunk)
            output.write(chunk)
    recovered = hashlib.sha256()
    size = 0
    with gzip.open(archive, "rb") as source:
        while chunk := source.read(1 << 20):
            recovered.update(chunk)
            size += len(chunk)
    if digest.digest() != recovered.digest() or size != raw.stat().st_size:
        raise ValueError("Lossless scientific archive verification failed")
    return {
        "raw_bytes": size,
        "gzip_bytes": archive.stat().st_size,
        "sha256": digest.hexdigest(),
        "decompressed_byte_identity": True,
    }


def main():
    import casadi
    import pybamm
    import scipy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.parent / (args.output.stem + "-routes")
    if any(p.exists() for p in (args.output, args.archive, args.summary, folder)):
        raise FileExistsError("Every attempt requires new result/archive/summary/checkpoint paths")
    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("fixed actual source requires PyBaMM25.4.2/CasADi3.6.7")
    folder.mkdir(parents=True)
    report = {
        "claim": "9c8c11ad",
        "mode": "development",
        "status": "running",
        "unit_denominator": 13,
        "route_denominator": 70,
        "history_denominator": 56,
        "refusal_denominator": 14,
        "units": [],
        "passed": 0,
        "grids": GRIDS,
        "reference_versions": {
            "pybamm": pybamm.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "protocol": "workstreams/world_foundation/NATIVE_CONV46_AXEN_PLATING_EXPERIMENT.md",
        "source_scope": "Actual SPM Axen main-current OCP plus reversible plating, NoSEI L=0; "
        "source initial dead=0; authored finite-CV boundary. MPM unsupported.",
    }
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "starting", "routes": 0}

    def checkpoint():
        brief = {k: v for k, v in report.items() if k not in ("units", "active_unit")}
        brief["elapsed_s"] = time.monotonic() - start
        brief["units"] = [{k: v for k, v in u.items() if k != "routes"} for u in report["units"]]
        brief["completed_routes"] = stage["routes"]
        brief["stage"] = stage["name"]
        brief["active_unit"] = report.get("active_unit")
        path = folder / "manifest.json"
        path.write_text(json.dumps(json_safe(brief), indent=2, allow_nan=False) + "\n")

    def heartbeat():
        while not stop.wait(20):
            elapsed = time.monotonic() - start
            speed = stage["routes"] / max(elapsed, 1.0)
            eta = (70 - stage["routes"]) / speed if speed else None
            print(
                f"AXEN {len(report['units'])}/13 units; {stage['routes']}/70 routes; "
                f"{stage['name']}; elapsed={elapsed:.1f}s; routes/s={speed:.3f}; ETA={eta}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for index in range(1, 14):
            item = {
                "id": f"A{index:02d}",
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
                "routes_executed": 70,
                "histories_executed": 56,
                "refusals_executed": 14,
                "route_exceptions": 0,
                "assessment_exceptions": 0,
            }.items()
        )
        report["elapsed_s"] = time.monotonic() - start
        stage["name"] = "write complete compact raw result"
        print(stage["name"], flush=True)
        compact_write(args.output, report)
        stage["name"] = "lossless gzip and decompression verification"
        print(stage["name"], flush=True)
        archive = archive_verified(args.output, args.archive)
        summary = {k: v for k, v in report.items() if k != "units"}
        summary["units"] = [{k: v for k, v in u.items() if k != "routes"} for u in report["units"]]
        summary["archive"] = archive
        summary["archive_and_block_elapsed_s"] = time.monotonic() - start
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
