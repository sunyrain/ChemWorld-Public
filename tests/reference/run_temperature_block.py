"""Fixed F5 development blocks; scientific reference imports stay in tests."""

import argparse
import itertools
import json
import math
import time
import traceback
from pathlib import Path

import chemicals
from chemicals import temperature as reference
from scipy.interpolate import CubicSpline

from chemworld.physchem.temperature_scale_data import TEMPERATURE_SCALE_TABLES
from chemworld.physchem.temperature_scales import (
    TEMPERATURE_SCALES,
    its90_68_difference,
    temperature_convert,
    temperature_scale_bounds,
    temperature_scale_conversion,
)

COUNTS = {
    "tables": 500,
    "intervals": 1488,
    "pairs": 25,
    "difference": 24,
    "difference-compatibility": 24,
    "consumers": 26,
    "inventory": 41,
}


def comparison(a, b, tolerance):
    return {
        "native": float(a),
        "reference": float(b),
        "error": abs(float(a) - float(b)),
        "tolerance": tolerance,
        "passed": math.isfinite(a) and abs(a - b) <= tolerance,
    }


def reference_comparison(value, current, desired, native):
    try:
        result = comparison(native, reference.T_converter(value, current, desired), 2e-8)
        return {"status": "passed" if result["passed"] else "failed", **result}
    except Exception:
        return {"status": "failed", "failure": traceback.format_exc()}


def reject(call):
    try:
        call()
    except ValueError:
        return True
    return False


def table_unit(scale, index):
    nodes, delta = TEMPERATURE_SCALE_TABLES[scale]
    key = scale.removeprefix("ITS-")
    expected_nodes = getattr(reference, "Ts_" + key)
    expected_delta = getattr(reference, "diffs_" + key)
    assert len(nodes) == len(expected_nodes) and len(delta) == len(expected_delta)
    assert nodes[index] == expected_nodes[index] and delta[index] == expected_delta[index]
    assert index == 0 or nodes[index] > nodes[index - 1]
    t = nodes[index]
    value = temperature_convert(t, scale, "ITS-90")
    checks = [
        comparison(value, t + delta[index], 2e-10),
        comparison(temperature_convert(value, "ITS-90", scale), t, 2e-10),
    ]
    return {
        "checks": checks,
        "reference_checks": [
            reference_comparison(t, scale, "ITS-90", value),
            reference_comparison(value, "ITS-90", scale, t),
        ],
    }


def interval_unit(scale, index, fraction):
    nodes, delta = TEMPERATURE_SCALE_TABLES[scale]
    low, high = nodes[index : index + 2]
    t = low + fraction * (high - low)
    value, derivative = temperature_scale_conversion(t, scale, "ITS-90")
    inverse, reverse_derivative = temperature_scale_conversion(value, "ITS-90", scale)
    h = min(1e-3, (high - low) / 100)
    fd = (
        temperature_convert(t + h, scale, "ITS-90") - temperature_convert(t - h, scale, "ITS-90")
    ) / (2 * h)
    # Independent polynomial representation, test derivative at its exact minimum.
    cs = CubicSpline(nodes, delta)
    a, b, c, _ = cs.c[:, index]
    candidates = [0.0, high - low]
    if a and 0 < -b / (3 * a) < high - low:
        candidates.append(-b / (3 * a))
    minimum = min(1 + c + 2 * b * x + 3 * a * x * x for x in candidates)
    checks = [
        comparison(inverse, t, 2e-10),
        comparison(derivative, fd, 2e-7),
        comparison(derivative * reverse_derivative, 1, 2e-12),
        comparison(value, t + float(cs(t)), 2e-10),
        {"passed": bool(minimum > 0)},
    ]
    return {
        "minimum_interval_derivative": float(minimum),
        "checks": checks,
        "reference_checks": [
            reference_comparison(t, scale, "ITS-90", value),
            reference_comparison(value, "ITS-90", scale, t),
        ],
    }


def pair_unit(old, new):
    lo = max(temperature_scale_bounds(s, basis="ITS-90")[0] for s in (old, new))
    hi = min(temperature_scale_bounds(s, basis="ITS-90")[1] for s in (old, new))
    if lo > hi:
        low, high = temperature_scale_bounds(old)
        assert reject(lambda: temperature_convert((low + high) / 2, old, new))
        return {"empty_intersection": True, "checks": [{"passed": True}]}
    points = (
        [0, 5, 273.15, 1000, 4300]
        if math.isinf(hi)
        else [lo + f * (hi - lo) for f in (0, 0.25, 0.5, 0.75, 1)]
    )
    checks, refs = [], []
    for common in points:
        original = temperature_convert(common, "ITS-90", old)
        expected = temperature_convert(common, "ITS-90", new)
        actual = temperature_convert(original, old, new)
        checks += [
            comparison(actual, expected, 2e-10),
            comparison(temperature_convert(actual, new, old), original, 2e-10),
        ]
        refs.append(reference_comparison(original, old, new, actual))
    for edge in (lo - 1e-5, hi + 1e-5):
        if not math.isfinite(edge):
            continue

        def outside(common=edge):
            return temperature_convert(temperature_convert(common, "ITS-90", old), old, new)

        checks.append({"passed": reject(outside)})
    return {
        "intersection_ITS90_K": [lo, hi if math.isfinite(hi) else None],
        "checks": checks,
        "reference_checks": refs,
    }


def difference_unit(t):
    # Direct powers, separately transcribed from Harvey (2021), Table 2.
    if t < 13.8:
        expected = 0.0
    elif t < 73.15:
        coefficients = [
            -0.005903,
            0.008174,
            -0.061924,
            -0.193388,
            1.490793,
            1.252347,
            -9.835868,
            1.411912,
            25.277595,
            -19.183815,
            -18.437089,
            27.000895,
            -8.716324,
        ]
        expected = math.fsum(a * ((t - 40) / 40) ** i for i, a in enumerate(coefficients))
    elif t < 903.75:
        coefficients = [
            0,
            -0.148759,
            -0.267408,
            1.08076,
            1.269056,
            -4.089591,
            -1.871251,
            7.438081,
            -3.536296,
        ]
        expected = math.fsum(a * ((t - 273.15) / 630) ** i for i, a in enumerate(coefficients))
    elif t < 1337.33:
        coefficients = [
            78.687209,
            -0.47135991,
            0.0010954715,
            -1.2357884e-6,
            6.7736583e-10,
            -1.4458081e-13,
        ]
        expected = math.fsum(a * (t - 273.15) ** i for i, a in enumerate(coefficients))
    else:
        expected = -1.398e-7 * t**2
    actual = its90_68_difference(t)
    ref = comparison(actual, reference.ITS90_68_difference(t), 2e-10)
    return {
        "checks": [comparison(actual, expected, 2e-10)],
        "reference_checks": [dict(status="passed" if ref["passed"] else "failed", **ref)],
    }


def units(block):
    if block in {"consumers", "inventory"}:
        from temperature_consumers import consumer_units, inventory_units

        yield from consumer_units() if block == "consumers" else inventory_units()
    if block == "tables":
        for scale, (nodes, _) in TEMPERATURE_SCALE_TABLES.items():
            for i, t in enumerate(nodes):
                yield (
                    f"{scale}/{i}",
                    {"scale": scale, "temperature_K": t},
                    (lambda s=scale, i=i: table_unit(s, i)),
                )
    elif block == "intervals":
        for scale, (nodes, _) in TEMPERATURE_SCALE_TABLES.items():
            for i in range(len(nodes) - 1):
                for fraction in (0.25, 0.5, 0.75):
                    yield (
                        f"{scale}/{i}/{fraction}",
                        {"scale": scale, "interval": i, "fraction": fraction},
                        lambda s=scale, i=i, f=fraction: interval_unit(s, i, f),
                    )
    elif block == "pairs":
        for old, new in itertools.product(TEMPERATURE_SCALES, repeat=2):
            yield (
                f"{old}/{new}",
                {"current": old, "desired": new},
                (lambda o=old, n=new: pair_unit(o, n)),
            )
    elif block in {"difference", "difference-compatibility"}:
        points = [0, 13.8, 20, 40, 73.15, 77, 83.8, 273.15, 500, 903.75, 1000, 1337.33, 2000, 4300]
        points += [x + d for x in (13.8, 73.15, 83.8, 903.75, 1337.33) for d in (-1e-7, 1e-7)]
        for t in points:
            if block == "difference":
                yield str(t), {"temperature_K": t}, lambda t=t: difference_unit(t)
            else:
                yield (
                    str(t),
                    {"temperature_K": t, "convention": "chemicals_1_5_2"},
                    (
                        lambda t=t: {
                            "checks": [
                                comparison(
                                    its90_68_difference(t, convention="chemicals_1_5_2"),
                                    reference.ITS90_68_difference(t),
                                    2e-10,
                                )
                            ]
                        }
                    ),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=COUNTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    work = list(units(args.block))
    assert len(work) == COUNTS[args.block]
    start, rows = time.perf_counter(), []
    for identity, inputs, call in work:
        row = {"unit": identity, "inputs": inputs}
        try:
            row["metrics"] = call()
            row["status"] = (
                "passed" if all(c["passed"] for c in row["metrics"]["checks"]) else "failed"
            )
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 200 == 0 or len(rows) == len(work):
            elapsed = time.perf_counter() - start
            print(
                f"{args.block} {len(rows)}/{len(work)}; {len(rows) / elapsed:.1f}/s; "
                f"ETA {(len(work) - len(rows)) * elapsed / len(rows):.1f}s",
                flush=True,
            )
    refs = [r for row in rows for r in row.get("metrics", {}).get("reference_checks", [])]
    summary = {
        "mode": "development",
        "block": args.block,
        "total": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "reference_comparisons": len(refs),
        "reference_failed": sum(r["status"] == "failed" for r in refs),
        "seconds": time.perf_counter() - start,
        "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for row in [r for r in rows if r["status"] == "failed"][:3]:
        print(row)
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
