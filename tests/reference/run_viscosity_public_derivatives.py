"""The six fixed direct-public and independent high-order viscosity responses."""

import argparse
import json
import time
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals import viscosity
from run_viscosity_block import METHOD_CASES

from chemworld.physchem.viscosity_kernels import viscosity_derivatives


def run(output):
    if output.exists():
        raise ValueError("new attempt path required")
    mp.mp.dps = 80
    started = time.perf_counter()
    results = []
    for method, temperatures, params in METHOD_CASES:
        if method not in {"PPDS9", "mu_Yaws"}:
            continue
        for temperature in temperatures:
            row = {
                "method": method,
                "temperature_K": temperature,
                "parameters": params,
                "checks": [],
            }
            try:
                actual = viscosity_derivatives(method, temperature, params, order=3)
                if method == "PPDS9":
                    d1, value = viscosity.dPPDS9_dT(temperature, **params)
                else:
                    value = viscosity.mu_Yaws(temperature, **params)
                    d1 = viscosity.dmu_Yaws_dT(temperature, **params)
                p = {name: mp.mpf(str(value)) for name, value in params.items()}

                def formula(t, p=p, method=method):
                    if method == "mu_Yaws":
                        return mp.exp(
                            mp.log(10) * (p["A"] + p["B"] / t + p["C"] * t + p["D"] * t * t)
                        )
                    ratio = (p["C"] - t) / (t - p["D"])
                    root = mp.sign(ratio) * abs(ratio) ** (mp.mpf(1) / 3)
                    return p["E"] * mp.exp(p["A"] * root + p["B"] * ratio * root)

                expected = tuple(mp.diff(formula, mp.mpf(temperature), i) for i in range(4))
                for basis, values in (("chemicals", (value, d1)), ("mpmath", expected)):
                    for order, expected_value in enumerate(values):
                        e, a = float(expected_value), actual[order]
                        error = abs(a - e)
                        row["checks"].append(
                            {
                                "reference": basis,
                                "order": order,
                                "actual": a,
                                "expected": e,
                                "absolute_error": error,
                                "passed": error <= 2e-9 * abs(e) + 1e-12,
                            }
                        )
                row["passed"] = all(c["passed"] for c in row["checks"])
            except Exception as exc:
                row.update(passed=False, failure=f"{type(exc).__name__}: {exc}")
            results.append(row)
            print(
                f"Completed {len(results)}/6: {method}/{temperature}, passed={row['passed']}",
                flush=True,
            )
    checks = [c for row in results for c in row["checks"]]
    report = {
        "schema": "chemworld-public-viscosity-derivatives-1",
        "mode": "development",
        "reference_versions": {"chemicals": chemicals.__version__, "mpmath": mp.__version__},
        "expected": {"states": 6, "comparisons": 36},
        "completed": len(results),
        "comparisons": len(checks),
        "passed_count": sum(row["passed"] for row in results),
        "passed_comparisons": sum(c["passed"] for c in checks),
        "results": results,
        "failures": [row for row in results if not row["passed"]],
        "seconds": time.perf_counter() - started,
    }
    report["passed"] = report["passed_count"] == 6 and report["passed_comparisons"] == 36
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps({k: report[k] for k in ("completed", "comparisons", "passed", "seconds")}),
        flush=True,
    )
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
