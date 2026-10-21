"""Fixed complete 15-source development block with every record and failure."""
# ruff: noqa: C408 -- explicit source denominators

import argparse
import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import chemicals
from phase_change_reference import numeric, read_source, source_curves
from pressure_reference import check_raw
from run_phase_change_methods_block import compare_derivative, near

from chemworld.data.phase_change import PhaseChangeDatabase
from chemworld.data.phase_change_curves import source_latent_heats
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)

COUNTS = dict(
    ALIBAKHSHI=1890,
    BELL_TWU=2571,
    CRC_FUSION=1112,
    CRC_VAPORIZATION=926,
    GHAZARATI_SUBLIMATION=1241,
    GHAZARATI_VAPORIZATION=2723,
    OPENNOTEBOOK_MELTING=11549,
    PERRY_VAPORIZATION=344,
    VDI_PPDS_VAPORIZATION=272,
    YAWS_BOILING=13461,
    WEBBOOK=13779,
    JOBACK=52224,
    WIKIDATA=10122,
    CHEMSEP_XML=431,
    VDI=58,
)


def rejected(function):
    try:
        function()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("invalid or unavailable latent heat was accepted")


def compare_curve(native, ref, metrics):
    assert (native.curve_key, native.property_id) == (ref.key, ref.property_id)
    metrics["curve_specs"] += 1
    if ref.method == "table":
        assert native.temperatures_K == ref.temperatures
        assert native.enthalpies_J_mol == ref.values
        metrics["table_points"] += len(ref.values)
        intervals = []
        for i in range(len(ref.values) - 1):
            lo, hi = ref.temperatures[i : i + 2]
            h0, h1 = ref.values[i : i + 2]
            options = {"temperature_bounds_K": (lo, hi), "interpolation": "linear"}
            if not 0 < lo < hi or any(numeric(h) is None or h < 0 for h in (h0, h1)):
                metrics["noncomputable_table_intervals"] += 1
                intervals.append(
                    {
                        "index": i,
                        "status": "noncomputable",
                        "reason": rejected(lambda options=options: native.build(**options)),
                    }
                )
                continue
            curve = native.build(**options)
            slope = (h1 - h0) / (hi - lo)
            for T in (lo, (lo + hi) / 2, hi):
                near(evaluate_correlation(curve, temperature_K=T).value, h0 + slope * (T - lo))
                near(correlation_temperature_derivative(curve, temperature_K=T).value, slope)
                metrics["table_value_derivative_pairs"] += 1
            metrics["computed_table_intervals"] += 1
            intervals.append({"index": i, "status": "computed"})
        return {"curve": native.curve_id, "intervals": intervals}
    options = {"parameters": ref.supplied}
    if ref.method == "point":
        metrics["point_specs"] += 1
        if any(numeric(v) is None or v <= 0 for v in ref.parameters.values()):
            metrics["noncomputable_points"] += 1
            return {
                "curve": native.curve_id,
                "status": "noncomputable",
                "reason": rejected(lambda: native.build(**options)),
            }
        curve = native.build(**options)
        T = ref.parameters["T_ref"]
        error = near(
            evaluate_correlation(curve, temperature_K=T).value,
            ref.evaluate(T),
            rtol=1e-12,
            atol=1e-12,
        )
        assert curve.validity_ranges["temperature"] == (T, T)
        metrics["computed_points"] += 1
        return {
            "curve": native.curve_id,
            "status": "computed",
            "error": error,
            "supplemental_conditions": ref.supplied,
        }
    options["temperature_bounds_K"] = ref.bounds
    bounds_valid = (
        ref.bounds is not None
        and all(numeric(v) is not None for v in ref.bounds)
        and 0 < ref.bounds[0] < ref.bounds[1]
    )
    if None in ref.parameters.values() or not bounds_valid:
        metrics["noncomputable_curves"] += 1
        return {
            "curve": native.curve_id,
            "status": "noncomputable",
            "reason": rejected(lambda: native.build(**options)),
        }
    curve = native.build(**options)
    points = []
    lo, hi = ref.bounds
    for fraction in (0.2, 0.5, 0.8):
        T = lo + (hi - lo) * fraction
        try:
            expected = ref.evaluate(T)
            valid = (
                math.isfinite(expected) and expected >= 0 and ref.parameters.get("Tc", math.inf) > T
            )
        except (ValueError, OverflowError, ZeroDivisionError):
            valid = False
        if not valid:
            metrics["noncomputable_curve_points"] += 1
            points.append(
                {
                    "T": T,
                    "status": "noncomputable",
                    "reason": rejected(lambda T=T: evaluate_correlation(curve, temperature_K=T)),
                }
            )
            continue
        error = near(evaluate_correlation(curve, temperature_K=T).value, expected)
        derivative_error = compare_derivative(curve, ref.evaluate, T)
        metrics["analytic_value_derivative_pairs"] += 1
        points.append(
            {
                "T": T,
                "status": "computed",
                "value_error": error,
                "derivative_error": derivative_error,
            }
        )
    metrics["constructed_analytic_curves"] += 1
    return {"curve": native.curve_id, "points": points, "supplemental_parameters": ref.supplied}


def compare_record(source, record, raw, db, metrics):
    if source == "BELL_TWU":
        assert record.headers == tuple(raw) and record.raw_fields == tuple(raw.values())
        assert record.source_key == raw["inchikey"] and record.cas_number is None
        for field in ("Tc", "Pc", "c0", "c1", "c2"):
            near(record.value(field).quantity().value, numeric(raw[field]), rtol=1e-12)
        metrics["twu_parameter_records"] += 1
    else:
        check_raw(record, raw)
    assert record in db.records_for(record.source_key, method=source)
    if source in {"YAWS_BOILING", "OPENNOTEBOOK_MELTING"}:
        field = "Tb" if source == "YAWS_BOILING" else "Tm"
        expected = numeric(raw[field])
        if expected is not None:
            near(record.value(field).quantity("K").value, expected, rtol=1e-12)
        else:
            assert record.value(field).quality != "valid"
        metrics["transition_temperature_records"] += 1
    native, references = source_latent_heats(record), source_curves(source, raw)
    assert len(native) == len(references), (source, record.record_id, len(native), len(references))
    return [
        compare_curve(curve, ref, metrics) for curve, ref in zip(native, references, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain earlier attempts; choose a new output path")
    assert chemicals.__version__ == "1.5.2"
    db, rows, summaries, metrics = PhaseChangeDatabase(args.source_directory), [], [], Counter()
    start = time.perf_counter()
    for source, count in COUNTS.items():
        before = len(rows)
        try:
            raw = read_source(args.source_directory / db.source(source).relative_path)
            records = db.records(source)
            assert len(raw) == len(records) == count, (source, len(raw), len(records), count)
        except Exception:
            summaries.append(
                {
                    "source": source,
                    "expected": count,
                    "executed": 0,
                    "failure": traceback.format_exc(),
                }
            )
            continue
        for i, (record, data) in enumerate(zip(records, raw, strict=True), 1):
            row = {"record": record.record_id, "source": source}
            try:
                row.update(
                    status="passed", curves=compare_record(source, record, data, db, metrics)
                )
            except Exception:
                row.update(status="failed", failure=traceback.format_exc())
            rows.append(row)
            if i % 2000 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"{source} {i}/{count}; total={len(rows)}/{sum(COUNTS.values())}; "
                    f"{len(rows) / elapsed:.1f} records/s; ETA="
                    f"{(sum(COUNTS.values()) - len(rows)) * elapsed / len(rows):.1f}s",
                    flush=True,
                )
        passed = sum(row["status"] == "passed" for row in rows[before:])
        summaries.append(
            {
                "source": source,
                "expected": count,
                "executed": count,
                "passed": passed,
                "failed": count - passed,
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
    passed = sum(row["status"] == "passed" for row in rows)
    result = {
        "mode": "development",
        "reference": chemicals.__version__,
        "expected": sum(COUNTS.values()),
        "executed": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "unexecuted": sum(COUNTS.values()) - len(rows),
        "seconds": time.perf_counter() - start,
        "metrics": dict(metrics),
        "sources": summaries,
        "records": rows,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"records", "sources"}}), flush=True
    )
    return int(passed != sum(COUNTS.values()) or len(rows) != sum(COUNTS.values()))


if __name__ == "__main__":
    raise SystemExit(main())
