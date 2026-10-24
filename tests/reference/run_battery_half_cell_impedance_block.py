"""Execute exactly the prospective 10-unit / 50-route half_cell rest-EIS block."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from battery_half_cell_impedance_assess import assess_unit, check, compare
from battery_half_cell_impedance_cases import FREQUENCIES, GRIDS, case_data, route_plan
from battery_half_cell_impedance_oracle import analytic_spectrum
from battery_half_cell_impedance_traces import native_spectrum, refusal, source_spectrum


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {k: safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return {
                "__nonfinite_float__": "NaN"
                if np.isnan(item)
                else "+Infinity"
                if item > 0
                else "-Infinity"
            }
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(safe(value), stream, allow_nan=False, separators=(",", ":"))
    temporary.replace(path)


def summary(result):
    units = result["units"]
    checks = [dict(c, unit=u["id"]) for u in units for c in u["assessment"]["checks"]]
    routes = [(u["id"], name, r) for u in units for name, r in u["routes"].items()]
    points = [
        p
        for u in units
        if u["index"] <= 8
        for r in u["routes"].values()
        for p in r.get("points", [])
    ]
    errors = [c["maximum_allowance_ratio"] for c in checks if "maximum_allowance_ratio" in c]
    return {
        "block": result["block"],
        "mode": "development",
        "attempt": 1,
        "status": result["status"],
        "expected_units": 10,
        "completed_units": len(units),
        "passed_units": sum(u["assessment"]["passed"] for u in units),
        "expected_routes": 50,
        "completed_routes": len(routes),
        "expected_spectral_routes": 40,
        "expected_FV_spectral_routes": 32,
        "expected_continuous_routes": 8,
        "expected_nonstationary_routes": 2,
        "expected_domain_routes": 8,
        "expected_frequency_points": 680,
        "completed_frequency_points": len(points),
        "successful_frequency_points": sum(p.get("success") is True for p in points),
        "route_exceptions": [
            {"unit": u, "route": name, "exception": r["exception"]}
            for u, name, r in routes
            if r.get("execution_exception")
        ],
        "assessment_exceptions": [
            {"unit": u["id"], "exception": u["assessment"].get("exception")}
            for u in units
            if u["assessment"].get("exception")
        ],
        "checks": len(checks),
        "passed_checks": sum(c["passed"] for c in checks),
        "failures": [c for c in checks if not c["passed"]],
        "maximum_response_allowance_ratio": max(errors, default=None),
        "elapsed_s": result.get("elapsed_s"),
        "versions": result["versions"],
        "unit_summary": [
            {
                "id": u["id"],
                "name": u["data"]["name"],
                "routes": len(u["routes"]),
                "passed": u["assessment"]["passed"],
                "elapsed_s": u["elapsed_s"],
            }
            for u in units
        ],
        "archive": result.get("archive"),
        "raw_output": result["raw_output"],
        "note": "Original complete finite block; all failures retained; no selected retries.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.archive, args.summary, args.log):
        if path.exists():
            raise FileExistsError(f"preserve previous attempt: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {"stage": "starting", "routes": 0, "unit": 0}
    stop = threading.Event()
    log = args.log.open("x", buffering=1)

    def progress(message):
        line = f"[{time.monotonic() - started:.1f}s] {message}"
        print(line, flush=True)
        log.write(line + "\n")

    def heartbeat():
        while not stop.wait(20):
            elapsed = time.monotonic() - started
            eta = (50 - state["routes"]) * elapsed / state["routes"] if state["routes"] else None
            progress(
                f"alive: {state['stage']}; units {state['unit']}/10; "
                f"routes {state['routes']}/50; ETA {eta}"
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result = {
        "block": "45-HALFCELL-EIS",
        "mode": "development",
        "status": "running",
        "raw_output": str(args.output),
        "frequencies_Hz": FREQUENCIES.tolist(),
        "grids": GRIDS,
        "versions": {
            "python": platform.python_version(),
            **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pybamm", "casadi")},
        },
        "expected_units": 10,
        "expected_routes": 50,
        "units": [],
    }
    try:
        for index in range(1, 11):
            data, routes, unit_start = case_data(index), {}, time.monotonic()
            for name, kind, grid in route_plan(index):
                state["stage"] = f"H{index:02d}/{name}"
                progress("start " + state["stage"])
                began = time.monotonic()
                try:
                    if kind == "analytic":
                        record = analytic_spectrum(data)
                    elif kind == "refusal":
                        record = refusal(data, name)
                    else:
                        function = native_spectrum if kind == "native" else source_spectrum
                        record = function(
                            data,
                            GRIDS[grid],
                            gradient=index == 9,
                        )
                except Exception:
                    record = {
                        "execution_exception": True,
                        "exception": traceback.format_exc(),
                        "implementation": kind,
                        "success": False,
                    }
                record["elapsed_s"] = time.monotonic() - began
                routes[name] = record
                state["routes"] += 1
                progress(
                    f"done {state['stage']}; routes {state['routes']}/50; "
                    f"{record['elapsed_s']:.3f}s"
                )
            try:
                assessment = assess_unit(index, data, routes)
                if index in (6, 8):
                    base = result["units"][0]["routes"]
                    shift = (
                        (
                            data["thickness_m"]
                            / data["area_m2"]
                            * (
                                1 / data["conductivity_S_m"]
                                - 1 / result["units"][0]["data"]["conductivity_S_m"]
                            )
                        )
                        if index == 6
                        else 0.0
                    )
                    for route in (f"{k}_{g}" for k in ("native", "reference") for g in GRIDS):
                        try:
                            row = compare(
                                routes[route], base[route], exact=True, resistance_shift=shift
                            )
                        except Exception as exc:
                            row = check(
                                "fixed-geometry-reservoir-or-series-shift",
                                False,
                                exception=str(exc),
                            )
                        row.update(
                            route=route,
                            cross_unit="H01",
                            comparison="metal-series-shift" if index == 6 else "finite-reservoir",
                        )
                        assessment["checks"].append(row)
                    assessment["passed"] = all(c["passed"] for c in assessment["checks"])
            except Exception:
                assessment = {
                    "passed": False,
                    "exception": traceback.format_exc(),
                    "checks": [check("assessment-exception", False)],
                }
            result["units"].append(
                {
                    "index": index,
                    "id": data["id"],
                    "data": data,
                    "routes": routes,
                    "assessment": assessment,
                    "elapsed_s": time.monotonic() - unit_start,
                }
            )
            state["unit"] = index
            progress(f"unit {data['id']} qualified={assessment['passed']}; checkpoint")
            result["elapsed_s"] = time.monotonic() - started
            write_json(args.output, result)
        result["status"] = "complete"
        result["elapsed_s"] = time.monotonic() - started
        write_json(args.output, result)
        state["stage"] = "lossless gzip archive and verification"
        progress(state["stage"])
        raw = args.output.read_bytes()
        with gzip.GzipFile(filename=str(args.archive), mode="wb", mtime=0) as stream:
            stream.write(raw)
        with gzip.open(args.archive, "rb") as stream:
            restored = stream.read()
        if restored != raw:
            raise RuntimeError("gzip round-trip mismatch")
        result["archive"] = {
            "path": str(args.archive),
            "raw_bytes": len(raw),
            "gzip_bytes": args.archive.stat().st_size,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "gzip_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            "lossless_verified": True,
        }
        output_summary = summary(result)
        output_summary["elapsed_with_archive_s"] = time.monotonic() - started
        write_json(args.summary, output_summary)
        progress(
            f"complete: {output_summary['passed_units']}/10 units, {state['routes']}/50 routes, "
            f"{output_summary['passed_checks']}/{output_summary['checks']} checks; archive verified"
        )
    finally:
        stop.set()
        thread.join(timeout=1)
        log.close()
    return 0 if all(u["assessment"]["passed"] for u in result["units"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
