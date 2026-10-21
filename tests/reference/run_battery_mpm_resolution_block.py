"""Three predeclared resolution checks; never replaces the original nine units."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from battery_frequency_pybamm import frequency_from_simulation
from battery_mpm_cases import FREQUENCIES_HZ, case
from battery_mpm_traces import (
    RESPONSE_NAMES,
    build_reference,
    native_trace,
    reference_trace,
)
from run_battery_mpm_block import complete, difference, guarded, valid

from chemworld.physchem.battery_impedance import battery_impedance

ORIGINAL_CASES = {1: 6, 2: 7, 3: 9}
NATIVE_GRIDS = {
    1: ((16, 512), (32, 512), (64, 512), (64, 128), (64, 256)),
    2: ((8, 64), (16, 64), (32, 64), (32, 16), (32, 32)),
    3: ((16, 64), (32, 64), (64, 64), (64, 16), (64, 32)),
}
REFERENCE_GRIDS = {
    1: ((160, 512), (320, 1024)),
    2: ((160, 160), (320, 256)),
    3: ((40, 128), (80, 256)),
}
FINE = {1: (64, 512), 2: (32, 64), 3: (64, 64)}


def native_spec(index, size, radial):
    spec = case(ORIGINAL_CASES[index], size_nodes=size, radial_cells=radial)
    if index == 1:
        spec = replace(
            spec,
            negative=replace(spec.negative, mesh_exponent=1.0),
            positive=replace(spec.positive, mesh_exponent=1.0),
        )
    return spec


def assess_cv(record, *, fine_grid, directions):
    """One CV criterion shared by separately declared resolution sequences."""
    native = {(r["size_nodes"], r["radial_cells"]): r for r in record["native"]}
    reference = record["reference"]
    missing = {"exception": "missing declared grid"}
    fine, ref = native.get(fine_grid, missing), reference[-1] if reference else missing
    expected = {grid for direction in directions for grid in direction}
    checks = {
        "all_grids_ran": (
            native.keys() == expected and len(native) == len(record["native"])
            and len(reference) == 2 and all(valid(r) for r in (*native.values(), *reference))
        ),
        "native_invariants": all(
            valid(r) and all(s["invariants_passed"] for s in r["segments"]) for r in native.values()
        ),
    }
    checks["complete_all_segments"] = all(complete(r, 6, True) for r in native.values())
    checks["reference_complete"] = all(complete(r, 6, False) for r in reference)
    comparison = record["fine_comparison"] = difference(fine, ref)
    checks["response_accuracy"] = comparison["passed"]
    checks["local_accuracy"] = comparison.get("local_passed", False)
    radii = (
        {n: fine["segments"][0]["local"][n]["radii_m"] for n in ("negative", "positive")}
        if valid(fine) else None
    )
    delta = record["reference_convergence"] = difference(
        reference[0] if reference else missing, ref, radii=radii, divisor=3)
    checks["reference_convergence"] = delta["passed"] and delta.get("local_passed", False)
    results = []
    for coarse, medium, finest in directions:
        change = difference(native.get(medium, missing), native.get(finest, missing), divisor=2)
        errors = [difference(native.get(g, missing), ref) for g in (coarse, finest)]
        trend = all("maximum_response_errors" in e for e in errors) and bool(
            np.all(
                np.array(errors[1]["maximum_response_errors"])
                <= np.array(errors[0]["maximum_response_errors"]) + 1e-7
            )
        )
        results.append({"grids": [coarse, medium, finest], "last_difference": change,
                        "errors_against_reference": errors, "trend": trend})
    record["native_convergence"] = results
    checks["native_convergence"] = all(
        r["last_difference"]["passed"] and r["trend"] for r in results)
    record.update(checks=checks, passed=all(checks.values()))


def assess_time(index, record):
    if index == 1:
        return assess_cv(
            record, fine_grid=FINE[1],
            directions=(((16, 512), (32, 512), (64, 512)),
                        ((64, 128), (64, 256), (64, 512))),
        )
    native = {(r["size_nodes"], r["radial_cells"]): r for r in record["native"]}
    reference = record["reference"]
    fine, ref = native[FINE[index]], reference[-1]
    checks = {
        "all_grids_ran": all(valid(r) for r in (*native.values(), *reference)),
        "native_invariants": all(
            valid(r) and all(s["invariants_passed"] for s in r["segments"]) for r in native.values()
        ),
    }
    if checks["all_grids_ran"]:
        traces = (*native.values(), *reference)
        checks["all_stop"] = (
            all(len(r["segments"]) == 1 for r in traces)
            and all(r["segments"][0]["outcome"] == "stopped" for r in native.values())
            and all(r["segments"][0]["termination"] != "final time" for r in reference)
        )
        checks["same_first_physical_stop"] = all(
            "negative:depleted" in (r["segments"][0].get("event") or "") for r in native.values()
        ) and all("negative:depleted" in r["segments"][0]["termination"] for r in reference)
        a, b = fine["segments"][-1], ref["segments"][-1]
        tolerance = 0.05 + 0.02 * abs(b["end_time_s"])
        checks["time"] = abs(a["end_time_s"] - b["end_time_s"]) <= tolerance
        checks["voltage"] = abs(a["rows"][-1][0] - b["rows"][-1][0]) <= 1e-3
        checks["surface_stop"] = all(
            min(abs(s["final_min_surface"] - 0.01), abs(s["final_max_surface"] - 0.99)) <= 1e-6
            for r in traces
            for s in r["segments"]
        )
        checks["native_time_convergence"] = all(
            abs(native[x]["segments"][-1]["end_time_s"] - native[y]["segments"][-1]["end_time_s"])
            <= tolerance / 2
            for x, y in (((16, 64), (32, 64)), ((32, 32), (32, 64)))
        )
        checks["reference_time_convergence"] = (
            abs(reference[0]["segments"][-1]["end_time_s"] - b["end_time_s"]) <= tolerance / 2
        )
        record["cutoff_comparison"] = {
            "time_tolerance_s": tolerance,
            "time_error_s": abs(a["end_time_s"] - b["end_time_s"]),
        }
    record.update(checks=checks, passed=all(checks.values()))


def time_unit(index, record, progress, checkpoint):
    original = ORIGINAL_CASES[index]
    record.update(native=[], reference=[], response_order=list(RESPONSE_NAMES))
    for size, radial in NATIVE_GRIDS[index]:
        progress(f"C{index:02} native size={size}, radial={radial}")
        answer = guarded(
            lambda size=size, radial=radial: native_trace(
                native_spec(index, size, radial), original, progress
            ),
            size_nodes=size,
            radial_cells=radial,
        )
        record["native"].append(answer)
        checkpoint()
    for size, radial in REFERENCE_GRIDS[index]:
        progress(f"C{index:02} reference size={size}, radial={radial}")
        answer = guarded(
            lambda size=size, radial=radial: reference_trace(
                case(original), original, size, radial, progress
            ),
            size_nodes=size,
            radial_cells=radial,
        )
        record["reference"].append(answer)
        checkpoint()
    assess_time(index, record)


def frequency_unit(record, progress, checkpoint):
    record.update(native=[], reference=[], frequencies_Hz=list(FREQUENCIES_HZ))
    for size, radial in NATIVE_GRIDS[3]:
        progress(f"C03 native spectrum size={size}, radial={radial}")

        def calculate(size=size, radial=radial):
            raw = battery_impedance(native_spec(3, size, radial), FREQUENCIES_HZ)
            return {
                "success": raw.success,
                "raw": raw.to_dict(),
                "real": [z.real for z in raw.impedance_ohm],
                "imag": [z.imag for z in raw.impedance_ohm],
            }

        record["native"].append(guarded(calculate, size_nodes=size, radial_cells=radial))
        checkpoint()
    for size, radial in REFERENCE_GRIDS[3]:
        progress(f"C03 reference spectrum size={size}, radial={radial}")

        def calculate(size=size, radial=radial):
            raw = frequency_from_simulation(
                build_reference(case(9), 9, size, radial),
                FREQUENCIES_HZ,
                family="MPM",
                grid={
                    "size_cells": size,
                    "radial_cells": radial,
                    "radial_mesh": "exponential-right",
                    "stretch": 2.3,
                },
                output_names=("Voltage [V]",),
            )
            return {
                "success": raw["success"],
                "raw": raw,
                "real": [p["response_real"][0] for p in raw["points"]] if raw["success"] else [],
                "imag": [p["response_imag"][0] for p in raw["points"]] if raw["success"] else [],
            }

        record["reference"].append(guarded(calculate, size_nodes=size, radial_cells=radial))
        checkpoint()
    native = {(r["size_nodes"], r["radial_cells"]): r for r in record["native"]}
    checks = {
        "all_spectra": all(
            valid(r) and r.get("success", False) for r in (*native.values(), *record["reference"])
        )
    }
    if checks["all_spectra"]:

        def z(r):
            return np.array(r["real"]) + 1j * np.array(r["imag"])

        ref, fine = z(record["reference"][-1]), z(native[FINE[3]])
        bounds, error = 0.001 + 0.01 * np.abs(ref), np.abs(fine - ref)
        checks["accuracy"] = bool(np.all(error <= bounds))
        checks["reference_convergence"] = bool(
            np.all(np.abs(z(record["reference"][0]) - ref) <= bounds / 3)
        )
        details = []
        for coarse, medium in (((16, 64), (32, 64)), ((64, 16), (64, 32))):
            change = np.abs(z(native[medium]) - fine)
            trend = bool(np.all(error <= np.abs(z(native[coarse]) - ref) + 1e-7))
            details.append(
                {
                    "coarse": coarse,
                    "medium": medium,
                    "last_difference": change.tolist(),
                    "trend": trend,
                }
            )
        checks["native_convergence"] = all(
            np.all(np.array(d["last_difference"]) <= bounds / 2) and d["trend"] for d in details
        )
        record.update(
            absolute_errors_ohm=error.tolist(),
            bounds_ohm=bounds.tolist(),
            native_convergence=details,
        )
    record.update(checks=checks, passed=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pybamm
    import scipy

    data = {
        "mode": "development",
        "denominator": 3,
        "units": [],
        "passed": 0,
        "pybamm_version": pybamm.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_MPM_RESOLUTION_EXPERIMENT.md",
        "does_not_replace_original_nine": True,
    }
    start, stop, lock = time.monotonic(), threading.Event(), threading.Lock()
    stage = {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:

        def checkpoint():
            data["elapsed_s"] = time.monotonic() - start

            def convert(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)

            payload = json.dumps(data, default=convert, indent=2, allow_nan=False) + "\n"
            output.seek(0)
            output.write(payload)
            output.truncate()
            output.flush()

        def progress(name):
            with lock:
                stage["name"] = name
            print(name, flush=True)

        def heartbeat():
            while not stop.wait(20):
                done, elapsed = len(data["units"]), time.monotonic() - start
                eta = f"{elapsed / done * (3 - done):.1f}s rough" if done else "unknown"
                with lock:
                    name = stage["name"]
                print(
                    f"MPM resolution active {done}/3; {name}; elapsed={elapsed:.1f}s; "
                    f"throughput={done / max(elapsed, 1):.4f}/s; ETA={eta}",
                    flush=True,
                )

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        checkpoint()
        try:
            for index in range(1, 4):
                unit_start = time.monotonic()
                record = {
                    "id": f"C{index:02}",
                    "original_case": f"G{ORIGINAL_CASES[index]:02}",
                    "passed": False,
                    "inputs": case(ORIGINAL_CASES[index]).to_dict(),
                    "native_mesh_exponent": 1.0 if index == 1 else 2.0,
                }
                data["active_unit"] = record
                checkpoint()
                try:
                    if index == 3:
                        frequency_unit(record, progress, checkpoint)
                    else:
                        time_unit(index, record, progress, checkpoint)
                except Exception as exc:
                    record.update(
                        passed=False, exception=str(exc), traceback=traceback.format_exc()
                    )
                record["elapsed_s"] = time.monotonic() - unit_start
                data["units"].append(record)
                data.pop("active_unit", None)
                data["passed"] = sum(r["passed"] for r in data["units"])
                checkpoint()
                print(
                    f"{record['id']} passed={record['passed']}; completed={index}/3; "
                    f"elapsed={data['elapsed_s']:.1f}s",
                    flush=True,
                )
        finally:
            stop.set()
            worker.join()
    return int(data["passed"] != 3)


if __name__ == "__main__":
    raise SystemExit(main())
