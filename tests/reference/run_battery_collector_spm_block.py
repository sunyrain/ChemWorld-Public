"""Execute the immutable 12-unit/48-route collector-SPM block.

The command is intentionally separate from preparation contracts.  It must only be called
after the coordinator records an explicit GO; all route outputs and exceptions are retained.
"""

from __future__ import annotations

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
from battery_collector_spm_assess import assess
from battery_collector_spm_cases import GRIDS, case_data, route_plan
from battery_collector_spm_traces import paired_time, refusal


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return float(value)
        return {"__nonfinite__": str(value)}
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def compact_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def archive_verified(source, archive):
    source, archive = Path(source), Path(archive)
    with (
        source.open("rb") as src,
        archive.open("wb") as destination,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, compresslevel=9, mtime=0
        ) as dst,
    ):
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    with gzip.open(archive, "rb") as restored:
        restored_bytes = restored.read()
    source_bytes = source.read_bytes()
    if restored_bytes != source_bytes:
        raise RuntimeError("lossless collector archive mismatch")
    return {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_bytes": len(source_bytes),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "byte_identity": True,
    }


def capture(function):
    try:
        return function()
    except Exception as exc:  # retain partial route output; do not hide a source failure
        return {
            "exception": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "partial_history": getattr(exc, "partial_history", None),
        }


def tasks(index, data):
    for name, kind, layout_name, grid_name in route_plan(index):
        layout = {
            "negative_tab": "left" if layout_name == "opposite" else "right",
            "positive_tab": "right",
        }
        grid = GRIDS[grid_name]
        if kind == "refusal":
            function = partial(refusal, data, grid, layout)
        elif kind == "history":
            function = partial(paired_time, data, grid, layout)
        else:
            raise ValueError(f"unknown collector route kind: {kind}")
        yield name, function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    folder = args.output.parent / (args.output.stem + "-routes")
    if any(p.exists() for p in (args.output, args.archive, args.summary, folder)):
        raise FileExistsError("collector attempts require new immutable paths")
    folder.mkdir(parents=True)
    report = {
        "claim": "eb2fbcf7",
        "attempt": args.attempt,
        "status": "running",
        "mode": "development",
        "unit_denominator": 12,
        "route_denominator": 48,
        "history_denominator": 40,
        "refusal_denominator": 8,
        "units": [],
        "passed": 0,
        "grids": GRIDS,
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_COLLECTOR_SPM_EXPERIMENT.md",
        "source_boundary_convention": (
            "source finite-volume boundary Voltage retained; independent Green physical-tab "
            "voltage is not projected by this runner"
        ),
    }
    started = time.monotonic()
    stop = threading.Event()
    stage = {"name": "starting", "routes": 0}

    def checkpoint():
        brief = {k: v for k, v in report.items() if k not in {"units", "active_unit"}}
        brief.update(
            elapsed_s=time.monotonic() - started,
            completed_routes=stage["routes"],
            stage=stage["name"],
            active_unit=report.get("active_unit"),
            units=[{k: v for k, v in u.items() if k != "routes"} for u in report["units"]],
        )
        compact_write(folder / "manifest.json", brief)

    def heartbeat():
        while not stop.wait(20):
            elapsed = time.monotonic() - started
            speed = stage["routes"] / max(elapsed, 1.0)
            eta = f"{(48 - stage['routes']) / speed:.1f}s" if speed else "unknown"
            print(
                f"COLLECTOR-SPM {len(report['units'])}/12 units {stage['routes']}/48 routes; "
                f"{stage['name']}; elapsed={elapsed:.1f}s; routes/s={speed:.3f}; ETA={eta}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for index in (*range(1, 11), 11, 12):
            data = case_data(f"CC{index:02d}")
            item = {"id": data["id"], "inputs": data, "routes": {}, "passed": False}
            report["active_unit"] = {"id": item["id"], "inputs": data}
            for name, function in tasks(data["id"], data):
                stage["name"] = data["id"] + " " + name
                print(stage["name"], flush=True)
                checkpoint()
                item["routes"][name] = capture(function)
                compact_write(folder / (data["id"] + "-" + name + ".json"), item["routes"][name])
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
            report["passed"] = sum(bool(u.get("passed")) for u in report["units"])
            print(data["id"] + (" PASS" if item["passed"] else " FAIL"), flush=True)
            checkpoint()
        report.update(
            status="completed",
            units_executed=len(report["units"]),
            routes_executed=sum(len(u["routes"]) for u in report["units"]),
            histories_executed=sum(len(u["routes"]) for u in report["units"][:10]) * 1,
            refusals_executed=sum(len(u["routes"]) for u in report["units"][10:]),
            route_exceptions=sum(
                "exception" in route for u in report["units"] for route in u["routes"].values()
            ),
            assessment_exceptions=sum("assessment_exception" in u for u in report["units"]),
        )
        report["evaluated_checks"] = sum(len(u.get("checks", {})) for u in report["units"])
        report["failed_checks"] = sum(len(u.get("failures", [])) for u in report["units"])
        report["all_passed"] = (
            report["units_executed"] == 12
            and report["routes_executed"] == 48
            and report["histories_executed"] == 40
            and report["refusals_executed"] == 8
            and report["route_exceptions"] == 0
            and report["assessment_exceptions"] == 0
            and report["passed"] == 12
        )
        report["elapsed_s"] = time.monotonic() - started
        compact_write(args.output, report)
        archive = archive_verified(args.output, args.archive)
        summary = {k: v for k, v in report.items() if k != "units"}
        summary["units"] = [
            {k: v for k, v in unit.items() if k != "routes"} for unit in report["units"]
        ]
        summary["archive"] = archive
        compact_write(args.summary, summary)
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
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
    finally:
        stop.set()
        thread.join()
    return 0 if report.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
