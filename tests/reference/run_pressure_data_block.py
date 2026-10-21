"""Fixed 13-source pressure qualification; all records and failures retained."""

import argparse
import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import chemicals
from pressure_reference import check_raw, interval, read_source, source_curves, valid_pressure

from chemworld.data.pressure import PressureDatabase
from chemworld.data.pressure_curves import source_pressures
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.saturation import pure_saturation_temperature_report
from chemworld.physchem.vapor_pressure import vapor_pressure_temperature_derivative

COUNTS = dict(  # noqa: C408 -- named fixed source denominators
    ALCOCK_PRESSURE=43,
    ALCOCK_SUBLIMATION=51,
    ANTOINE_POLING=325,
    TRC_ANTOINE_POLING=97,
    LANDOLT_SUBLIMATION=1084,
    LANDOLT_PRESSURE=6346,
    PERRY_PRESSURE=340,
    VDI_PPDS_PRESSURE=275,
    WAGNER_POLING=104,
    WAGNER_MCGARRY=245,
    WEBBOOK=13779,
    CHEMSEP_XML=431,
    VDI=58,
)


def near(actual, expected, rtol=3e-10, atol=1e-12):
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (actual, expected)
    return abs(actual - expected)


def rejected(function):
    try:
        function()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("invalid/missing pressure data were accepted")


def check_inverse(curve, T, pressure, bounds):
    answer = pure_saturation_temperature_report(
        curve,
        pressure_Pa=pressure,
        temperature_bounds_K=bounds,
        tolerance_K=1e-9,
        tolerance_log_pressure=1e-12,
    )
    assert abs(answer.saturation_temperature_K - T) <= 2e-6, (answer.saturation_temperature_K, T)
    assert abs(answer.log_pressure_residual) <= 2e-8


def compare_table(native, ref, metrics):
    assert native.temperatures_K == ref.temperatures
    assert native.pressures_Pa == ref.values
    metrics["table_points"] += len(ref.temperatures)
    outcomes = []
    for i in range(len(ref.temperatures) - 1):
        lo, hi = ref.temperatures[i : i + 2]
        p0, p1 = ref.values[i : i + 2]
        options = {
            "temperature_bounds_K": (lo, hi),
            "interpolation": "log_pressure_inverse_temperature",
        }
        if not (0 < lo < hi and all(v is not None and valid_pressure(v) for v in (p0, p1))):
            outcomes.append(
                {
                    "interval": i,
                    "status": "noncomputable",
                    "reason": rejected(lambda options=options: native.build(**options)),
                }
            )
            continue
        curve = native.build(**options)
        slope = math.log(p1 / p0) / (1 / hi - 1 / lo)
        for T in (lo, 2 / (1 / lo + 1 / hi), hi):
            pressure = p0 * math.exp(slope * (1 / T - 1 / lo))
            near(evaluate_correlation(curve, temperature_K=T).value, pressure)
            l1, l2, l3 = -slope / T**2, 2 * slope / T**3, -6 * slope / T**4
            for order, expected in enumerate(
                (pressure * l1, pressure * (l1 * l1 + l2), pressure * (l1**3 + 3 * l1 * l2 + l3)), 1
            ):
                near(
                    vapor_pressure_temperature_derivative(
                        curve, temperature_K=T, order=order
                    ).value,
                    expected,
                )
                metrics["table_derivatives"] += 1
            metrics["pressure_points"] += 1
        outcomes.append({"interval": i, "status": "computed"})
    return outcomes


def compare_curve(native, ref, metrics):
    assert (native.curve_key, native.property_id) == (ref.key, ref.property_id)
    assert native.source_bounds_K == ref.bounds
    metrics["curve_specs"] += 1
    if ref.method == "tabulated":
        return {"curve": native.curve_id, "intervals": compare_table(native, ref, metrics)}
    lo, hi = interval(ref.bounds)
    if ref.method.startswith("unresolved"):
        reason = rejected(lambda: native.build(temperature_bounds_K=(lo, hi)))
        metrics["unresolved_curves"] += 1
        return {"curve": native.curve_id, "status": "unresolved", "reason": reason}
    if None in ref.coefficients.values() or not 0 < lo < hi:
        reason = rejected(lambda: native.build(temperature_bounds_K=(lo, hi)))
        metrics["noncomputable_curves"] += 1
        return {"curve": native.curve_id, "status": "noncomputable", "reason": reason}
    curve = native.build(temperature_bounds_K=(lo, hi))
    values, points, outcomes = [], [], []
    for fraction in (0.25, 0.5, 0.75):
        T = lo + fraction * (hi - lo)
        try:
            expected = ref.evaluate(T)
            valid = valid_pressure(expected)
            reason = "reference pressure is nonpositive or nonfinite"
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            valid, reason = False, str(exc)
        if not valid:
            rejected(
                lambda T=T: evaluate_correlation(curve, temperature_K=T, validity_policy="raise")
            )
            metrics["noncomputable_points"] += 1
            outcomes.append({"temperature_K": T, "status": "noncomputable", "reason": reason})
            continue
        near(evaluate_correlation(curve, temperature_K=T, validity_policy="raise").value, expected)
        values.append(expected)
        points.append(T)
        metrics["pressure_points"] += 1
        for order in (1, 2):
            derivative = ref.evaluate(T, order)
            actual = vapor_pressure_temperature_derivative(
                curve, temperature_K=T, order=order
            ).value
            near(actual, derivative)
            metrics["analytic_derivatives"] += 1
        outcomes.append({"temperature_K": T, "status": "computed"})
    if len(values) == 3 and (
        values[0] < values[1] < values[2] or values[0] > values[1] > values[2]
    ):
        check_inverse(curve, points[1], values[1], (points[0], points[2]))
        metrics["inverse_solves"] += 1
    else:
        metrics["inverse_not_bracketed"] += 1
    metrics["constructed_curves"] += 1
    return {"curve": native.curve_id, "status": "constructed", "points": outcomes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert chemicals.__version__ == "1.5.2"
    started = time.perf_counter()
    db = PressureDatabase(args.source_directory)
    summary = {
        "mode": "development",
        "reference_version": chemicals.__version__,
        "expected_records": sum(COUNTS.values()),
        "sources": [],
        "failures": [],
    }
    completed = 0
    with args.output.open("x", encoding="utf-8") as output:
        for method, count in COUNTS.items():
            print(f"source={method} start completed={completed}/{sum(COUNTS.values())}", flush=True)
            records = db.records(method)
            raw = read_source(args.source_directory / db.source(method).relative_path)
            result = {"source": method, "expected": count, "records": [], "metrics": Counter()}
            if len(records) != count or len(raw) != count:
                summary["failures"].append(
                    {
                        "source": method,
                        "error": "source count mismatch",
                        "native": len(records),
                        "reference": len(raw),
                    }
                )
            for index, (record, original) in enumerate(zip(records, raw, strict=True), 1):
                outcome = {"record_id": record.record_id}
                try:
                    check_raw(record, original)
                    refs, natives = source_curves(method, original), source_pressures(record)
                    assert len(refs) == len(natives), (len(refs), len(natives))
                    outcome["curves"] = [
                        compare_curve(n, r, result["metrics"])
                        for n, r in zip(natives, refs, strict=True)
                    ]
                    outcome["status"] = (
                        "unresolved" if any(c.get("status") == "unresolved"
                                            for c in outcome["curves"]) else "passed")
                except Exception:
                    outcome.update(status="failed", error=traceback.format_exc())
                    summary["failures"].append(outcome.copy())
                result["records"].append(outcome)
                completed += 1
                if index % 250 == 0 or index == count:
                    elapsed = time.perf_counter() - started
                    rate = completed / max(elapsed, 1e-9)
                    print(
                        f"source={method} {index}/{count} total={completed}/{sum(COUNTS.values())} "
                        f"rate={rate:.1f}/s ETA={(sum(COUNTS.values()) - completed) / rate:.1f}s "
                        f"failures={len(summary['failures'])}",
                        flush=True,
                    )
            result["passed"] = sum(r["status"] == "passed" for r in result["records"])
            summary["sources"].append(result)
        summary.update(
            completed=completed,
            elapsed_seconds=time.perf_counter() - started,
            passed=sum(s["passed"] for s in summary["sources"]),
            unresolved=sum(s["metrics"]["unresolved_curves"] for s in summary["sources"]),
        )
        json.dump(summary, output, indent=2, ensure_ascii=False)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"sources", "failures"}}),
        flush=True,
    )
    return bool(summary["failures"] or summary["unresolved"])


if __name__ == "__main__":
    raise SystemExit(main())
