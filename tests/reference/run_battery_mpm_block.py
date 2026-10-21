"""All nine predeclared MPM units, with complete grid and failure retention."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from battery_frequency_pybamm import frequency_from_simulation
from battery_mpm_cases import (
    FREQUENCIES_HZ,
    IMPEDANCE_REFERENCE_GRIDS,
    NATIVE_GRIDS,
    REFERENCE_GRIDS,
    case,
)
from battery_mpm_traces import (
    RESPONSE_BOUNDS,
    RESPONSE_NAMES,
    build_reference,
    native_trace,
    protocol,
    reference_trace,
)
from scipy.interpolate import interp1d
from scipy.special import ndtr

from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.physchem.battery_particles import lognormal_size_distribution


def guarded(function, **metadata):
    try:
        return {**metadata, **function()}
    except Exception as exc:
        return {**metadata, "exception": str(exc), "traceback": traceback.format_exc()}


def valid(trace):
    return "exception" not in trace


def complete(trace, index, native):
    return (
        valid(trace)
        and len(trace["segments"]) == len(protocol(index))
        and all(
            s["outcome"] == "success" if native else s["termination"] == "final time"
            for s in trace["segments"]
        )
    )


def interpolate_local(segment, name, field, radii):
    data = segment["local"][name]
    values = np.asarray(data[field])
    source = np.asarray(data["radii_m"])
    if len(source) == 1:
        return np.repeat(values, len(radii), axis=1)
    return interp1d(source, values, axis=1, kind="linear", fill_value="extrapolate")(radii)


def difference(first, second, *, radii=None, divisor=1.0):
    if not valid(first) or not valid(second):
        return {"passed": False, "reason": "a grid failed"}
    if len(first["segments"]) != len(second["segments"]):
        return {"passed": False, "reason": "different segment counts"}
    aggregate, surface, j_scaled = np.zeros(len(RESPONSE_NAMES)), 0.0, 0.0
    for a, b in zip(first["segments"], second["segments"], strict=True):
        x, y = np.asarray(a["rows"]), np.asarray(b["rows"])
        if (
            x.shape != y.shape
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or not np.allclose(a["times_s"], b["times_s"], atol=1e-7, rtol=0)
        ):
            return {"passed": False, "reason": "different/nonfinite observation grids"}
        aggregate = np.maximum(aggregate, np.max(np.abs(x - y), axis=0))
        for name in ("negative", "positive"):
            target = a["local"][name]["radii_m"] if radii is None else radii[name]
            sa = interpolate_local(a, name, "surface", target)
            sb = interpolate_local(b, name, "surface", target)
            ja = interpolate_local(a, name, "j_A_m2", target)
            jb = interpolate_local(b, name, "j_A_m2", target)
            surface = max(surface, float(np.max(np.abs(sa - sb))))
            j_scaled = max(j_scaled, float(np.max(np.abs(ja - jb) / (0.02 + 0.02 * np.abs(jb)))))
    return {
        "maximum_response_errors": aggregate.tolist(),
        "response_bounds": (RESPONSE_BOUNDS / divisor).tolist(),
        "maximum_local_surface_error": surface,
        "maximum_local_j_error_fraction": j_scaled,
        "passed": bool(np.all(aggregate <= RESPONSE_BOUNDS / divisor)),
        "local_passed": bool(surface <= 3e-3 / divisor and j_scaled <= 1 / divisor),
    }


def time_unit(index, record, progress, checkpoint):
    native_grids = ((1, 16), (1, 32), (1, 64)) if index == 1 else NATIVE_GRIDS
    reference_grids = ((1, 128), (1, 256)) if index == 1 else REFERENCE_GRIDS
    native, reference = {}, {}
    record.update(native=[], reference=[], response_order=list(RESPONSE_NAMES))
    for size, radial in native_grids:
        progress(f"G{index:02} native grid size={size}, radial={radial}")
        answer = guarded(
            lambda size=size, radial=radial: native_trace(
                case(index, size_nodes=size, radial_cells=radial), index, progress
            ),
            size_nodes=size,
            radial_cells=radial,
        )
        native[(size, radial)] = answer
        record["native"].append(answer)
        checkpoint()
    for size, radial in reference_grids:
        progress(f"G{index:02} released grid size={size}, radial={radial}")
        answer = guarded(
            lambda size=size, radial=radial: reference_trace(
                case(index), index, size, radial, progress
            ),
            size_nodes=size,
            radial_cells=radial,
        )
        reference[(size, radial)] = answer
        record["reference"].append(answer)
        checkpoint()
    fine = native[(1, 64) if index == 1 else (32, 64)]
    ref = reference[reference_grids[-1]]
    checks = {
        "all_grids_ran": all(valid(r) for r in (*native.values(), *reference.values())),
        "native_invariants": all(
            valid(r) and all(s["invariants_passed"] for s in r["segments"]) for r in native.values()
        ),
    }
    if index == 7:

        def stopped(r, native):
            return (
                valid(r)
                and len(r["segments"]) == 1
                and (
                    r["segments"][0]["outcome"] == "stopped"
                    if native
                    else r["segments"][0]["termination"] != "final time"
                )
            )

        checks["all_stop"] = all(stopped(r, True) for r in native.values()) and all(
            stopped(r, False) for r in reference.values()
        )
        if checks["all_grids_ran"]:

            def event(s, native):
                value = s["event"] if native else s["termination"]
                for name in (
                    "negative:depleted",
                    "negative:saturated",
                    "positive:depleted",
                    "positive:saturated",
                    "minimum-voltage",
                    "maximum-voltage",
                ):
                    if value and name in value:
                        return name
                return value

            a, b = fine["segments"][-1], ref["segments"][-1]
            tolerance = 0.05 + 0.02 * abs(b["end_time_s"])
            checks["same_first_physical_stop"] = (
                event(a, True) == event(b, False)
                and all(event(r["segments"][-1], True) == event(b, False) for r in native.values())
                and all(
                    event(r["segments"][-1], False) == event(b, False) for r in reference.values()
                )
            )
            checks["time"] = abs(a["end_time_s"] - b["end_time_s"]) <= tolerance
            checks["voltage"] = abs(a["rows"][-1][0] - b["rows"][-1][0]) <= 1e-3
            checks["surface_stop"] = all(
                min(abs(s["final_min_surface"] - 0.01), abs(s["final_max_surface"] - 0.99)) <= 1e-6
                for r in (*native.values(), *reference.values())
                for s in r["segments"]
            )
            pairs = (((16, 64), (32, 64)), ((32, 32), (32, 64)))
            checks["native_time_convergence"] = all(
                abs(
                    native[x]["segments"][-1]["end_time_s"]
                    - native[y]["segments"][-1]["end_time_s"]
                )
                <= tolerance / 2
                for x, y in pairs
            )
            checks["reference_time_convergence"] = (
                abs(record["reference"][0]["segments"][-1]["end_time_s"] - b["end_time_s"])
                <= tolerance / 2
            )
            record["cutoff_comparison"] = {
                "native_event": event(a, True),
                "reference_event": event(b, False),
                "time_tolerance_s": tolerance,
                "time_error_s": abs(a["end_time_s"] - b["end_time_s"]),
            }
    else:
        checks["complete_all_segments"] = all(complete(r, index, True) for r in native.values())
        checks["reference_complete"] = all(complete(r, index, False) for r in reference.values())
        record["fine_comparison"] = comparison = difference(fine, ref)
        checks["response_accuracy"] = comparison["passed"]
        checks["local_accuracy"] = comparison.get("local_passed", False)
        radii = (
            {n: fine["segments"][0]["local"][n]["radii_m"] for n in ("negative", "positive")}
            if valid(fine)
            else None
        )
        record["reference_convergence"] = delta = difference(
            record["reference"][0], ref, radii=radii, divisor=3
        )
        checks["reference_convergence"] = delta["passed"] and delta.get("local_passed", False)
        directions = (
            (((1, 16), (1, 32), (1, 64)),)
            if index == 1
            else (((8, 64), (16, 64), (32, 64)), ((32, 16), (32, 32), (32, 64)))
        )
        record["native_convergence"] = []
        for coarse, medium, finest in directions:
            change = difference(native[medium], native[finest], divisor=2)
            errors = [difference(native[g], ref) for g in (coarse, finest)]
            trend = all("maximum_response_errors" in e for e in errors) and bool(
                np.all(
                    np.array(errors[1]["maximum_response_errors"])
                    <= np.array(errors[0]["maximum_response_errors"]) + 1e-7
                )
            )
            record["native_convergence"].append(
                {
                    "grids": [coarse, medium, finest],
                    "last_difference": change,
                    "errors_against_reference": errors,
                    "trend": trend,
                }
            )
        checks["native_convergence"] = all(
            r["last_difference"]["passed"] and r["trend"] for r in record["native_convergence"]
        )
        if index in (3, 4):
            checks["not_volume_prescribed"] = valid(fine) and any(
                max(s["allocation_deviation_from_volume_A"], default=0) > 1e-8
                for s in fine["segments"]
                if s["target"] != 0
            )
        if index == 3:
            checks["rest_internal_current"] = valid(fine) and any(
                max(s["allocation_deviation_from_volume_A"], default=0) > 1e-8
                for s in fine["segments"]
                if s["target"] == 0
            )
    record.update(checks=checks, passed=all(checks.values()))


def moment_unit(record):
    source = case(8).negative_distribution.source
    records, finest_ok, trends = [], [], []
    for basis, power in (("number", 3), ("area", 1), ("volume", 0)):
        errors = {}

        def integral(exponent):
            sigma = 0.4
            return np.exp(0.5 * exponent**2 * sigma**2) * (
                ndtr(np.log(2.5) / sigma - exponent * sigma)
                - ndtr(np.log(0.25) / sigma - exponent * sigma)
            )

        for nodes in (8, 16, 32):
            d = lognormal_size_distribution(
                median_radius_m=1e-5,
                log_standard_deviation=0.4,
                lower_radius_m=2.5e-6,
                upper_radius_m=25e-6,
                nodes=nodes,
                source=source,
                basis=basis,
            )
            for k in (-3, -1, 0, 1, 2, 3):
                reference = float(integral(power + k) / integral(power))
                actual = float(np.dot(d.volume_weights, (np.array(d.radii_m) / 1e-5) ** k))
                error = abs(actual - reference) / abs(reference)
                errors[(nodes, k)] = error
                records.append(
                    {
                        "basis": basis,
                        "nodes": nodes,
                        "moment_order": k,
                        "reference": reference,
                        "native": actual,
                        "relative_error": error,
                    }
                )
                if nodes == 32:
                    finest_ok.append(error <= 1e-7)
        trends.extend(errors[(32, k)] <= errors[(8, k)] + 1e-10 for k in (-3, -1, 0, 1, 2, 3))
    record.update(
        comparisons=records,
        comparison_denominator=54,
        finest_denominator=18,
        finest_passed=sum(finest_ok),
        trend_passed=sum(trends),
        passed=all(finest_ok) and all(trends),
    )


def frequency_unit(record, progress, checkpoint):
    record.update(native=[], reference=[], frequencies_Hz=list(FREQUENCIES_HZ))
    native, reference = {}, []
    for size, radial in NATIVE_GRIDS:
        progress(f"G09 native spectrum size={size}, radial={radial}")

        def calculate(size=size, radial=radial):
            answer = battery_impedance(
                case(9, size_nodes=size, radial_cells=radial), FREQUENCIES_HZ
            )
            return {
                "success": answer.success,
                "raw": answer.to_dict(),
                "real": [z.real for z in answer.impedance_ohm],
                "imag": [z.imag for z in answer.impedance_ohm],
            }

        value = guarded(calculate, size_nodes=size, radial_cells=radial)
        native[(size, radial)] = value
        record["native"].append(value)
        checkpoint()
    for size, radial in IMPEDANCE_REFERENCE_GRIDS:
        progress(f"G09 released spectrum size={size}, radial={radial}")

        def calculate(size=size, radial=radial):
            simulation = build_reference(case(9), 9, size, radial)
            raw = frequency_from_simulation(
                simulation,
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

        value = guarded(calculate, size_nodes=size, radial_cells=radial)
        reference.append(value)
        record["reference"].append(value)
        checkpoint()
    checks = {
        "all_spectra": all(
            valid(v) and v.get("success", False) for v in (*native.values(), *reference)
        )
    }
    if checks["all_spectra"]:

        def z(value):
            return np.array(value["real"]) + 1j * np.array(value["imag"])

        ref, fine = z(reference[-1]), z(native[(32, 64)])
        bounds = 0.001 + 0.01 * np.abs(ref)
        error = np.abs(fine - ref)
        checks["accuracy"] = bool(np.all(error <= bounds))
        checks["reference_convergence"] = bool(np.all(np.abs(z(reference[0]) - ref) <= bounds / 3))
        details = []
        for coarse, medium in (((8, 64), (16, 64)), ((32, 16), (32, 32))):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pybamm
    import scipy

    data = {
        "mode": "development",
        "denominator": 9,
        "units": [],
        "passed": 0,
        "pybamm_version": pybamm.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_MPM_EXPERIMENT.md",
    }
    start, stop = time.monotonic(), threading.Event()
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
            stage["name"] = name
            print(name, flush=True)

        def heartbeat():
            while not stop.wait(20):
                done, elapsed = len(data["units"]), time.monotonic() - start
                estimate = f"{elapsed / done * (9 - done):.1f}s rough" if done else "unknown"
                print(
                    f"MPM active {done}/9; {stage['name']}; elapsed={elapsed:.1f}s; "
                    f"throughput={done / max(elapsed, 1):.4f}/s; ETA={estimate}",
                    flush=True,
                )

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        checkpoint()
        try:
            for index in range(1, 10):
                unit_start = time.monotonic()
                record = {"id": f"G{index:02}", "passed": False, "inputs": case(index).to_dict()}
                data["active_unit"] = record
                checkpoint()
                try:
                    if index == 8:
                        moment_unit(record)
                    elif index == 9:
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
                    f"{record['id']} passed={record['passed']}; completed={index}/9; "
                    f"elapsed={data['elapsed_s']:.1f}s",
                    flush=True,
                )
        finally:
            stop.set()
            worker.join()
    return int(data["passed"] != 9)


if __name__ == "__main__":
    raise SystemExit(main())
