"""Seven fixed resistances: 80-digit analytical Lambert W against native roots."""

import argparse
import json
import time
from pathlib import Path

import mpmath as mp

from chemworld.physchem.gas_flow import isothermal_critical_pressure_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rows = []
    with mp.workdps(80):
        for index, text in enumerate(("1e-8", "1e-4", "1", "10", "100", "1000", "10000"), 1):
            resistance = mp.mpf(text)
            row = {"unit": index, "resistance": text, "passed": False}
            try:
                reference = mp.exp((mp.lambertw(-mp.exp(-1 - resistance), -1)
                    + 1 + resistance) / 2)
                native = isothermal_critical_pressure_ratio(
                    darcy_friction_factor=.02, diameter_m=.1, length_m=5 * float(resistance))
                exact_native = mp.mpf(native)
                relative = abs(exact_native / reference - 1)
                residual = (exact_native**-2 - 1 + 2 * mp.log(exact_native)
                    - resistance) / resistance
                row.update(native_pressure_Pa=native * 1e6,
                    reference_pressure_Pa=mp.nstr(reference * 1e6, 80),
                    relative_error=float(relative), normalized_equation_residual=float(residual),
                    pressure_passed=relative <= mp.mpf("2e-10"),
                    equation_passed=abs(residual) <= mp.mpf("2e-9"))
                row["passed"] = row["pressure_passed"] and row["equation_passed"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"High precision critical {index}/7: {row['passed']}", flush=True)
    result = {"mode": "development", "reference": f"mpmath {mp.__version__} 80 digits",
        "expected_inputs": 7, "expected_comparisons": 14,
        "passed_inputs": sum(r["passed"] for r in rows),
        "passed_comparisons": sum(int(r.get("pressure_passed", False))
            + int(r.get("equation_passed", False)) for r in rows),
        "elapsed_s": time.perf_counter() - start,
        "failures": [r for r in rows if not r["passed"]], "records": rows}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "failures"}}))
    return 0 if result["passed_inputs"] == 7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
