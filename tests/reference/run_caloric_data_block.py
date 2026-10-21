"""Fixed complete-source 22C2 development block; every failure is retained."""

import argparse
import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import chemicals
import numpy as np
from caloric_source_reference import interval, numeric, read_source, source_curves

from chemworld.data.caloric import CaloricDatabase
from chemworld.data.caloric_curves import source_heat_capacities
from chemworld.data.constants import SourceCorrelation
from chemworld.physchem.property_equations import evaluate_correlation, integrate_correlation

COUNTS = dict(  # noqa: C408 -- compact named scientific source counts
    CRC_THERMODYNAMIC=2470,
    CRC_SOLID_CP=42,
    JANAF_GAS_CP=889,
    JANAF_LIQUID_CP=221,
    JANAF_SOLID_CP=342,
    PERRY_LIQUID_100=332,
    PERRY_LIQUID_114=11,
    PERRY_151_CLEAN=403,
    PERRY_151_JSON=319,
    PERRY_151_RAW=477,
    POLING_CP=367,
    PSI4_ADJUSTED_THETA=276,
    PSI4_UNADJUSTED_THETA=276,
    TRC_GAS=1961,
    WEBBOOK_SHOMATE=880,
    ZABRANSKY=1292,
    API_TDB_GAS=571,
    ATCT_GAS=595,
    ATCT_LIQUID=34,
    JANAF_THERMODYNAMIC=950,
    YAWS_THERMODYNAMIC=5030,
    JOBACK=52224,
    WEBBOOK=13779,
    CHEMSEP_XML=431,
    VDI=58,
)
SCALARS = [
    "Hfs", "Hfl", "Hfg", "Gfs", "Gfl", "Gfg", "Hfg_0K", "Hfl_0K",
    "S0s", "S0l", "S0g", "Cps", "Cpl", "Cpg", "uncertainty",
]
GAUSS = {n: np.polynomial.legendre.leggauss(n) for n in (32, 64)}


def near(actual, expected, rtol=2e-10, atol=2e-8):
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (actual, expected)
    return abs(actual - expected)


def rejected(function):
    try:
        function()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("invalid/missing source data were accepted")


def integrals(curve, function, low, high, metrics):
    for entropy in (False, True):
        reference = []
        for order in (32, 64):
            nodes, weights = GAUSS[order]
            points = (low + high) / 2 + (high - low) / 2 * nodes
            reference.append(
                (high - low)
                / 2
                * math.fsum(
                    float(w) * function(float(T)) / (float(T) if entropy else 1)
                    for w, T in zip(weights, points, strict=True)
                )
            )
        near(reference[0], reference[1], 2e-8, 2e-6)
        result = integrate_correlation(
            curve, initial_temperature_K=low, final_temperature_K=high, over_temperature=entropy
        )
        # Generic integrals retain the curve's declared amount basis.
        value = result.value * (0.001 if curve.output_unit == "J/(kmol*K)" else 1)
        error = near(value, reference[1], 2e-8, 2e-6)
        metrics["integrals"] += 1
        metrics["max_integral_error"] = max(metrics["max_integral_error"], error)


def compare_curve(native, ref, metrics):
    assert (native.curve_key, native.phase) == (ref.key, ref.phase)
    if ref.condition is not None:
        assert native.condition == ref.condition
    if ref.subphase is not None:
        assert native.subphase == ref.subphase
    metrics["curves"] += 1
    if ref.temperatures:
        return compare_table(native, ref, metrics)
    bounds = interval(ref.bounds)
    def build():
        return native.build(parameters=ref.parameters, temperature_bounds_K=bounds)
    if ref.missing or bounds[0] >= bounds[1]:
        metrics["missing_or_invalid_curves"] += 1
        return {"source_unavailable": rejected(build)}
    curve = build()
    outcomes = []
    for fraction in (0.25, 0.5, 0.75):
        T = bounds[0] + fraction * (bounds[1] - bounds[0])
        try:
            expected = ref.function(T)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            metrics["reference_domain_rejections"] += 1
            outcomes.append(
                {
                    "temperature": T,
                    "reference_domain": str(exc),
                    "native_rejection": rejected(
                        lambda T=T: evaluate_correlation(curve, temperature_K=T)
                    ),
                }
            )
            continue
        if expected <= 0 or not math.isfinite(expected):
            metrics["source_nonpositive_points"] += 1
            outcomes.append(
                {
                    "temperature": T,
                    "source_value": expected,
                    "native_rejection": rejected(
                        lambda T=T: evaluate_correlation(curve, temperature_K=T)
                    ),
                }
            )
            continue
        actual = evaluate_correlation(curve, temperature_K=T).to("J/(mol*K)").value
        error = near(actual, expected)
        metrics["value_points"] += 1
        metrics["max_value_error"] = max(metrics["max_value_error"], error)
    low, high = (bounds[0] + f * (bounds[1] - bounds[0]) for f in (0.45, 0.55))
    values = [ref.function(low + (high - low) * i / 32) for i in range(33)]
    if min(values) <= 0:
        metrics["source_nonpositive_intervals"] += 1
        outcomes.append(
            {
                "nonpositive_integral": rejected(
                    lambda: integrate_correlation(
                        curve, initial_temperature_K=low, final_temperature_K=high
                    )
                )
            }
        )
    else:
        integrals(curve, ref.function, low, high, metrics)
    return outcomes


def compare_table(native, ref, metrics):
    assert native.temperatures_K == tuple(map(numeric, ref.temperatures))
    assert native.capacities_J_mol_K == tuple(map(numeric, ref.values))
    qualities = tuple(
        "valid" if numeric(v) is not None else "missing" if v is None or v == "" else "invalid"
        for v in ref.values
    )
    assert native.capacity_qualities == qualities
    metrics["table_points"] += len(ref.values)
    metrics["table_missing"] += qualities.count("missing")
    metrics["table_invalid"] += qualities.count("invalid")
    outcomes = []
    for i in range(len(ref.temperatures) - 1):
        low, high = map(numeric, ref.temperatures[i : i + 2])
        first, second = map(numeric, ref.values[i : i + 2])
        if low is None or high is None or low <= 0 or high <= low:
            metrics["unavailable_table_intervals"] += 1
            continue
        def build(low=low, high=high):
            return native.build(temperature_bounds_K=(low, high), interpolation="linear")
        if first is None or second is None:
            outcomes.append({"index": i, "missing_interval": rejected(build)})
            metrics["missing_table_intervals"] += 1
            continue
        curve = build()

        def function(T, first=first, second=second, low=low, high=high):
            return first + (second - first) * (T - low) / (high - low)

        for T in (low, (low + high) / 2, high):
            expected = function(T)
            if expected <= 0:
                metrics["source_nonpositive_points"] += 1
                rejected(lambda curve=curve, T=T: evaluate_correlation(curve, temperature_K=T))
                continue
            error = near(evaluate_correlation(curve, temperature_K=T).value, expected)
            metrics["value_points"] += 1
            metrics["max_value_error"] = max(metrics["max_value_error"], error)
        if min(first, second) > 0:
            integrals(curve, function, low, high, metrics)
        else:
            metrics["source_nonpositive_intervals"] += 1
            rejected(
                lambda curve=curve, low=low, high=high: integrate_correlation(
                    curve, initial_temperature_K=low, final_temperature_K=high
                )
            )
    return outcomes


def compare_record(source, raw, record, metrics):
    if isinstance(raw, dict):
        assert record.headers == tuple(raw)
        assert record.raw_fields == tuple(raw.values())
        fields = raw
    elif isinstance(raw, tuple):
        key, payload = raw
        assert record.source_key == key
        if source == "VDI":
            assert dict(zip(record.field_names, record.raw_fields, strict=True)) == {
                k: tuple(v) if isinstance(v, list) else v for k, v in payload.items()
            }
        else:
            assert record.raw("payload") == payload
        fields = {}
    else:
        assert record.headers == tuple(child.tag for child in raw)
        for child, value in zip(raw, record.raw_fields, strict=True):
            if isinstance(value, SourceCorrelation):
                from xml.etree import ElementTree as ET

                assert value.xml == ET.tostring(child, encoding="unicode")
            else:
                assert value == child.get("value")
            if child.tag in {"HeatOfFormation", "GibbsEnergyOfFormation", "AbsEntropy"}:
                expected = numeric(child.get("value"))
                value = record.value(child.tag)
                metrics["scalar_fields"] += 1
                if expected is None:
                    assert value.quality in {"missing", "invalid"}
                    metrics["scalar_unavailable"] += 1
                else:
                    unit = "J/(mol*K)" if child.tag == "AbsEntropy" else "J/mol"
                    near(value.quantity(unit).value, 0.001 * expected, 2e-12, 1e-10)
        fields = {}
    for name in SCALARS:
        if name not in fields:
            continue
        expected = numeric(fields[name])
        value = record.value(name)
        quality = (
            "valid" if expected is not None else "missing" if fields[name] == "" else "invalid"
        )
        assert value.quality == quality
        metrics["scalar_fields"] += 1
        if expected is not None:
            unit = "kJ/(mol*K)" if name.startswith(("S0", "Cp")) else "kJ/mol"
            near(value.quantity(unit).value, 0.001 * expected, 2e-12, 1e-10)
        else:
            metrics["scalar_unavailable"] += 1
    expected = source_curves(source, raw)
    actual = source_heat_capacities(record)
    assert len(actual) == len(expected), (len(actual), len(expected))
    outcomes, failures = [], []
    for native, reference in zip(actual, expected, strict=True):
        try:
            result = compare_curve(native, reference, metrics)
            if result:
                outcomes.append({"curve": reference.key, "outcomes": result})
        except Exception:
            failures.append({"curve": reference.key, "error": traceback.format_exc()})
    return outcomes, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert chemicals.__version__ == "1.5.2"
    started = time.perf_counter()
    database = CaloricDatabase(args.source_directory)
    results, sources, done = [], {}, 0
    for source, expected_count in COUNTS.items():
        raw_records = read_source(args.source_directory / database.source(source).relative_path)
        records = database.records(source)
        assert len(raw_records) == len(records) == expected_count, (
            source,
            len(raw_records),
            len(records),
        )
        metrics = Counter()
        print(f"source={source} records={len(records)} completed={done}/84230", flush=True)
        for raw, record in zip(raw_records, records, strict=True):
            result = {"record_id": record.record_id}
            try:
                outcomes, failures = compare_record(source, raw, record, metrics)
                result.update(outcomes=outcomes, failures=failures, passed=not failures)
            except Exception:
                result.update(passed=False, failures=[{"error": traceback.format_exc()}])
            results.append(result)
            metrics["records"] += 1
            metrics["passed" if result["passed"] else "failed"] += 1
            done += 1
            if done % 250 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"completed={done}/84230 rate={done / elapsed:.1f}/s "
                    f"eta={(84230 - done) * elapsed / done:.1f}s",
                    flush=True,
                )
        sources[source] = dict(metrics)
    assert done == 84230
    summary = {
        "mode": "development",
        "source_version": chemicals.__version__,
        "records": done,
        "passed": sum(r["passed"] for r in results),
        "failed": sum(not r["passed"] for r in results),
        "elapsed_s": time.perf_counter() - started,
        "sources": sources,
        "results": results,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"sources", "results"}}),
        flush=True,
    )
    raise SystemExit(bool(summary["failed"]))


if __name__ == "__main__":
    main()
