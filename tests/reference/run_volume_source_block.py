"""Fixed complete D2C external-source block; only public reference calls/data."""

import argparse
import csv
import json
import math
import time
import traceback
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

import chemicals
from chemicals import dippr, volume

from chemworld.data.constants import SourceCorrelation
from chemworld.data.volume import VolumeDatabase
from chemworld.data.volume_curves import source_volumes
from chemworld.physchem import PropertyCorrelation, evaluate_correlation

COUNTS = {
    "COSTALD_PARAMETERS": 192,
    "CRC_MOLTEN": 177,
    "CRC_LIQUID_VOLUME": 116,
    "CRC_SOLID_VOLUME": 1872,
    "CRC_VIRIAL": 105,
    "MCHAWEH_SN0": 73,
    "PERRY_VOLUME": 344,
    "VDI_PPDS_VOLUME": 272,
    "CHEMSEP_XML": 431,
    "VDI": 58,
}
SUPPLEMENT = {"Tc": 600.0, "Pc": 4e6, "Vc": 0.0002, "omega": 0.2}


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def reference(source, raw, key, temperature):
    def n(k):
        return float(raw[k])

    if source == "COSTALD_PARAMETERS":
        return (
            volume.COSTALD(temperature, 600.0, n("Vchar"), n("omega_SRK"))
            if key == "COSTALD"
            else volume.Rackett(temperature, 600.0, 4e6, n("Z_RA"))
        )
    if source == "MCHAWEH_SN0":
        return volume.SNM0(temperature, 600.0, 0.0002, 0.2, n("delta_SRK"))
    if source == "CRC_MOLTEN":
        return volume.CRC_inorganic(temperature, n("rho"), n("k"), n("Tm"))
    if source in {"CRC_LIQUID_VOLUME", "CRC_SOLID_VOLUME"}:
        return n("Vm")
    if source == "PERRY_VOLUME":
        return 1 / dippr.EQ105(temperature, *[n(f"C{i}") for i in range(1, 5)])
    if source == "VDI_PPDS_VOLUME":
        return volume.volume_VDI_PPDS(temperature, n("Tc"), n("rhoc"), *[n(k) for k in "ABCD"])
    if source == "CRC_VIRIAL":
        x = 298.15 / temperature - 1
        return 1e-6 * sum(n(f"a{i}") * x ** (i - 1) for i in range(1, 6))
    if source == "CHEMSEP_XML":
        values = {c.tag: c.attrib.get("value") for c in raw[key]}
        eq = values["eqno"]
        params = {k: float(values[k]) for k in ("ABCD" if eq == "105" else "ABCDE") if k in values}
        if eq == "2":
            return 1 / (1000 * (params["A"] + params["B"] * temperature))
        if eq == "106":
            params["Tc"] = float(raw["CriticalTemperature"].attrib["value"])
        value = getattr(dippr, "EQ" + eq)(temperature, **params)
        return value / 1000 if key == "SecondVirialCoefficient" else 1 / (1000 * value)
    raise AssertionError(source)


def compare(curve, t, expected):
    restored = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))
    actual = evaluate_correlation(restored, temperature_K=t, validity_policy="raise").value
    atol = 1e-8 if curve.output_unit == "kg/m^3" else 1e-12
    if abs(actual - expected) > atol + 3e-10 * abs(expected):
        raise AssertionError({"temperature": t, "actual": actual, "reference": expected})
    return {
        "temperature_K": t,
        "actual": actual,
        "reference": expected,
        "absolute_error": abs(actual - expected),
    }


def source_rows(db, source):
    path = db.source_directory / db.source(source).relative_path
    if source == "CHEMSEP_XML":
        return [{child.tag: child for child in node} for node in ET.parse(path).getroot()]
    if source == "VDI":
        return list(json.loads(path.read_text(encoding="utf-8")).values())
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def raw_equal(record, raw):
    for key, value in raw.items():
        actual = record.raw(key)
        if record.source_id == "CHEMSEP_XML":
            if isinstance(actual, SourceCorrelation):
                assert actual.xml == ET.tostring(value, encoding="unicode")
                assert actual.source_unit == value.attrib.get("units", "")
            else:
                assert actual == value.attrib["value"]
        elif isinstance(value, list):
            assert tuple(actual) == tuple(value)
        else:
            assert actual == value


def record_unit(record, raw):
    raw_equal(record, raw)
    result = []
    for candidate in source_volumes(record):
        row = {"curve_id": candidate.curve_id, "comparisons": [], "source_rejections": []}
        result.append(row)
        if candidate.equation_id == "tabulated":
            ts, ys = raw["T"], raw[candidate.curve_key]
            for i, (a, b) in enumerate(pairwise(ts)):
                y0, y1 = ys[i : i + 2]
                if any(numeric(v) is None or float(v) <= 0 for v in (y0, y1)):
                    try:
                        candidate.build(interpolation="linear", temperature_bounds_K=(a, b))
                    except ValueError as exc:
                        row["source_rejections"].append({"interval": [a, b], "reason": str(exc)})
                    else:
                        raise AssertionError("missing table interval accepted")
                    continue
                curve = candidate.build(interpolation="linear", temperature_bounds_K=(a, b))
                for t in (a, (a + b) / 2, b):
                    expected = y0 + (y1 - y0) * (t - a) / (b - a)
                    row["comparisons"].append(compare(curve, t, expected))
            continue
        if any(v is None for v in candidate.coefficients.values()):
            try:
                candidate.build(
                    parameters={k: SUPPLEMENT[k] for k in candidate.required_inputs},
                    temperature_bounds_K=(250, 500),
                )
            except ValueError as exc:
                row["source_rejections"].append({"reason": str(exc), "kind": "missing_coefficient"})
            else:
                raise AssertionError("missing coefficients accepted")
            continue
        if candidate.equation_id == "unlocated_point":
            curve = candidate.build(
                application_temperature_K=300, application_note="Fixed development point assumption"
            )
            row["comparisons"].append(
                compare(curve, 300, reference(record.source_id, raw, candidate.curve_key, 300))
            )
            try:
                evaluate_correlation(curve, temperature_K=301, validity_policy="ignore")
            except ValueError:
                row["point_extrapolation_rejected"] = True
            else:
                raise AssertionError("point extrapolation accepted")
            continue
        lo, hi = candidate.source_bounds_K
        if record.source_id in {"COSTALD_PARAMETERS", "MCHAWEH_SN0"}:
            lo, hi = 240.0, 480.0
        elif record.source_id == "VDI_PPDS_VOLUME":
            lo, hi = 0.4 * float(raw["Tc"]), 0.8 * float(raw["Tc"])
        elif record.source_id == "CRC_VIRIAL":
            lo, hi = 250.0, 500.0
        elif record.source_id == "CRC_MOLTEN" and hi is None:
            hi = lo + 10
        if lo == 0 and hi is not None:
            lo = hi / 2
        if lo is None or hi is None or lo >= hi:
            try:
                candidate.build()
            except ValueError as exc:
                row["source_rejections"].append({"reason": str(exc), "kind": "source_interval"})
            else:
                raise AssertionError("incomplete source interval accepted")
            continue
        curve = candidate.build(
            temperature_bounds_K=(lo, hi),
            parameters={k: SUPPLEMENT[k] for k in candidate.required_inputs},
        )
        for f in (0.25, 0.5, 0.75):
            t = lo + (hi - lo) * f
            try:
                expected = reference(record.source_id, raw, candidate.curve_key, t)
            except ZeroDivisionError:
                # Source zero density has no reciprocal volume. Retain this
                # domain defect separately, as fixed before the block started.
                try:
                    evaluate_correlation(curve, temperature_K=t)
                except ValueError as exc:
                    row["source_rejections"].append(
                        {
                            "temperature": t,
                            "reference": "zero density",
                            "reason": str(exc),
                            "kind": "nonpositive",
                        }
                    )
                else:
                    raise AssertionError("zero source density accepted")
                continue
            if expected <= 0 and candidate.property_id != "second_virial_coefficient":
                try:
                    evaluate_correlation(curve, temperature_K=t)
                except ValueError as exc:
                    row["source_rejections"].append(
                        {
                            "temperature": t,
                            "reference": expected,
                            "reason": str(exc),
                            "kind": "nonpositive",
                        }
                    )
                else:
                    raise AssertionError("nonpositive source volume accepted")
            else:
                row["comparisons"].append(compare(curve, t, expected))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    db = VolumeDatabase(Path(chemicals.__file__).parent)
    started = time.perf_counter()
    units = []
    for source, count in COUNTS.items():
        originals, records = source_rows(db, source), db.records(source)
        assert len(originals) == len(records) == count, (source, len(records), count)
        for r, raw in zip(records, originals, strict=True):
            unit = {"source": source, "record_id": r.record_id, "passed": False}
            try:
                unit["curves"] = record_unit(r, raw)
                unit["passed"] = True
            except Exception:
                unit["failure"] = traceback.format_exc()
            units.append(unit)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": source,
                    "completed": len(units),
                    "total": sum(COUNTS.values()),
                    "units_per_s": len(units) / elapsed,
                    "eta_s": (sum(COUNTS.values()) - len(units)) * elapsed / len(units),
                }
            ),
            flush=True,
        )
    failures = [r for r in units if not r["passed"]]
    result = {
        "mode": "development",
        "reference_version": chemicals.__version__,
        "source_counts": COUNTS,
        "total": len(units),
        "passed": len(units) - len(failures),
        "failed": len(failures),
        "unexecuted": 0,
        "elapsed_s": time.perf_counter() - started,
        "comparisons": sum(len(c["comparisons"]) for r in units for c in r.get("curves", [])),
        "source_rejections": sum(
            len(c["source_rejections"]) for r in units for c in r.get("curves", [])
        ),
        "complete_and_passed": not failures,
        "units": units,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "units"}))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
