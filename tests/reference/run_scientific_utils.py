"""Fixed complete public utility comparison and independent partial-molar calculus."""

import argparse
import inspect
import json
import time
from copy import deepcopy
from itertools import product
from math import isfinite
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals import utils
from scientific_utils_catalog import ADAPTERS, EXCLUDED, cases

from chemworld.physchem import composition_math as c


def compare(actual, expected, path="result"):
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            return [
                {
                    "path": path,
                    "actual": actual,
                    "expected": expected,
                    "passed": False,
                    "failure": "shape mismatch",
                }
            ]
        return [
            item
            for i, (a, e) in enumerate(zip(actual, expected, strict=True))
            for item in compare(a, e, f"{path}/{i}")
        ]
    if isinstance(expected, str):
        return [
            {"path": path, "actual": actual, "expected": expected, "passed": actual == expected}
        ]
    a, e = float(actual), float(expected)
    finite = isfinite(a) and isfinite(e)
    error = abs(a - e) if finite else None
    return [
        {
            "path": path,
            "actual": a if isfinite(a) else repr(a),
            "expected": e if isfinite(e) else repr(e),
            "absolute_error": error,
            "passed": bool(finite and error <= 2e-10 * abs(e) + 1e-12),
        }
    ]


def partial_case(fractions, total_moles):
    mp.mp.dps = 60
    xs = tuple(mp.mpf(str(v)) for v in fractions)
    ns = tuple(x * mp.mpf(str(total_moles)) for x in xs)

    def molar(*x):
        return 12 + x[0] ** 2 + 3 * x[0] * x[1] + 4 * x[1] ** 2 + 5 * x[2] ** 3

    def with_moles(*n):
        return molar(*(v / sum(n) for v in n))

    def extensive(*n):
        return sum(n) * with_moles(*n)

    def deriv(fun, point, *indices):
        orders = tuple(indices.count(i) for i in range(len(point)))
        return mp.diff(fun, point, orders)

    g = tuple(float(deriv(molar, xs, i)) for i in range(3))
    h = tuple(tuple(float(deriv(molar, xs, i, j)) for j in range(3)) for i in range(3))
    dn = tuple(float(deriv(with_moles, ns, i)) for i in range(3))
    dnn = tuple(tuple(float(deriv(with_moles, ns, i, j)) for j in range(3)) for i in range(3))
    partial = c.partial_molar_properties(float(molar(*xs)), g, fractions)
    output = []
    output += compare(
        c.composition_gradient_to_moles(g, fractions, total_moles=total_moles), dn, "molar-gradient"
    )
    output += compare(partial, tuple(deriv(extensive, ns, i) for i in range(3)), "partial-molar")
    output += compare(
        c.extensive_hessian_from_moles(dnn, dn, total_moles=total_moles),
        tuple(tuple(deriv(extensive, ns, i, j) for j in range(3)) for i in range(3)),
        "extensive-hessian",
    )

    def constrained(a, b):
        return molar(a, b, 1 - a - b)

    output += compare(
        c.constrained_composition_gradient(g),
        tuple(deriv(constrained, xs[:2], i) for i in range(2)),
        "constrained-gradient",
    )
    output += compare(
        c.constrained_composition_hessian(h),
        tuple(tuple(deriv(constrained, xs[:2], i, j) for j in range(2)) for i in range(2)),
        "constrained-hessian",
    )
    jacobian = c.partial_molar_composition_jacobian(h, fractions)
    for i in range(3):

        def partial_extension(*x, i=i):
            return molar(*x) + deriv(molar, x, i) - sum(x[k] * deriv(molar, x, k) for k in range(3))

        output += compare(
            jacobian[i],
            tuple(deriv(partial_extension, xs, j) for j in range(3)),
            f"partial-composition/{i}",
        )
    output += compare(
        sum(x * p for x, p in zip(fractions, partial, strict=True)), molar(*xs), "Euler"
    )
    output += compare(
        tuple(sum(fractions[i] * jacobian[i][j] for i in range(3)) for j in range(3)),
        (0.0, 0.0, 0.0),
        "Gibbs-Duhem",
    )
    return {
        "fractions": fractions,
        "total_moles": total_moles,
        "checks": output,
        "passed": all(row["passed"] for row in output),
    }


def run(destination):
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    public = {name for name in utils.__all__ if inspect.isfunction(getattr(utils, name))}
    if public != set(ADAPTERS) | set(EXCLUDED):
        raise ValueError("public inventory changed; revise design before running")
    fixtures = cases()
    if len(fixtures) != 205 or {row["name"] for row in fixtures} != set(ADAPTERS):
        raise ValueError("fixed scientific fixture coverage changed")
    records = []
    for index, fixture in enumerate(fixtures):
        row = dict(fixture)
        try:
            args, kwargs = deepcopy((fixture["args"], fixture["kwargs"]))
            # Public stream functions document lists; an intermediate reference
            # merge creates a list and cannot concatenate our original tuple.
            args = json.loads(json.dumps(args))
            expected = getattr(utils, fixture["name"])(*args, **kwargs)
            args, kwargs = deepcopy((fixture["args"], fixture["kwargs"]))
            actual = ADAPTERS[fixture["name"]](*args, **kwargs)
            row["checks"] = compare(actual, expected)
            row["passed"] = all(item["passed"] for item in row["checks"])
        except Exception as exc:
            row.update(passed=False, failure=f"{type(exc).__name__}: {exc}", checks=[])
        records.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(fixtures):
            elapsed = time.perf_counter() - start
            rate = (index + 1) / elapsed
            print(
                f"Public utilities {index + 1}/{len(fixtures)}, {rate:.1f} calls/s, "
                f"ETA {(len(fixtures) - index - 1) / rate:.2f}s",
                flush=True,
            )
    calculus = []
    for fractions, total in product(((0.7, 0.2, 0.1), (0.1, 0.3, 0.6)), (0.1, 1.0, 10.0)):
        try:
            row = partial_case(fractions, total)
        except Exception as exc:
            row = {
                "fractions": fractions,
                "total_moles": total,
                "passed": False,
                "failure": f"{type(exc).__name__}: {exc}",
                "checks": [],
            }
        calculus.append(row)
        print(f"Independent partial-molar states {len(calculus)}/6", flush=True)
    all_rows = records + calculus
    checks = [item for row in all_rows for item in row["checks"]]
    result = {
        "schema": "chemworld-scientific-utils-1",
        "mode": "development",
        "reference_versions": {"chemicals": chemicals.__version__, "mpmath": mp.__version__},
        "catalog": sorted(ADAPTERS),
        "excluded": EXCLUDED,
        "expected": {"public_functions": 63, "public_calls": 205, "partial_molar_states": 6},
        "completed": {
            "public_calls": len(records),
            "partial_molar_states": len(calculus),
            "comparisons": len(checks),
        },
        "passed_count": {
            "public_calls": sum(row["passed"] for row in records),
            "partial_molar_states": sum(row["passed"] for row in calculus),
            "comparisons": sum(item["passed"] for item in checks),
        },
        "public_calls": records,
        "partial_molar_states": calculus,
        "failures": [row for row in all_rows if not row["passed"]],
        "seconds": time.perf_counter() - start,
        "boundary": "Public mathematical identities and synthetic composition calculus; "
        "not material accuracy or general phase-equilibrium qualification.",
        "passed": all(row["passed"] for row in all_rows),
    }
    (destination / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({key: result[key] for key in ("completed", "passed_count", "passed", "seconds")})
    )
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
