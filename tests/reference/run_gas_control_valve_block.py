"""Fixed 20 IEC gas sizing inputs / 60 comparisons."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

import fluids
from fluids.control_valve import size_control_valve_g

from chemworld.physchem.gas_control_valves import GasControlValveSpec, size_gas_control_valve
from chemworld.physchem.gas_flow import IsothermalGasState


def inputs():
    for reducers, ratio, flow in product((False, True), (0.1, 0.5, 0.9), (0.01, 0.1, 1)):
        yield {
            "T": 300,
            "MW": 28.97,
            "mu": 1e-5,
            "gamma": 1.4,
            "Z": 0.98,
            "P1": 680000,
            "P2": ratio * 680000,
            "Q": flow,
            "D1": 0.08 if reducers else 0.05,
            "D2": 0.1 if reducers else 0.05,
            "d": 0.05,
            "FL": 0.85,
            "Fd": 0.42,
            "xT": 0.6,
        }
    yield {
        "T": 433,
        "MW": 44.01,
        "mu": 1.4665e-4,
        "gamma": 1.3,
        "Z": 0.988,
        "P1": 680000,
        "P2": 310000,
        "Q": 38 / 36,
        "D1": 0.08,
        "D2": 0.1,
        "d": 0.05,
        "FL": 0.85,
        "Fd": 0.42,
        "xT": 0.6,
    }
    yield {
        "T": 320,
        "MW": 39.95,
        "mu": 5.625e-5,
        "gamma": 1.67,
        "Z": 1,
        "P1": 280000,
        "P2": 130000,
        "Q": 0.46 / 3600,
        "D1": 0.015,
        "D2": 0.015,
        "d": 0.015,
        "FL": 0.98,
        "Fd": 0.07,
        "xT": 0.8,
    }


def run(case, values):
    spec = GasControlValveSpec(
        1,
        case["d"],
        case["D1"],
        case["D2"],
        case["FL"],
        case["Fd"],
        case["xT"],
        "Fixed IEC gas engineering comparison",
        expansion_basis="bare_valve_reference",
    )
    gas = IsothermalGasState(case["T"], 8.31446261815324 / (case["MW"] / 1000), case["Z"])
    result = size_gas_control_valve(
        spec,
        gas,
        reference_flow_m3_s=case["Q"],
        viscosity_Pa_s=case["mu"],
        heat_capacity_ratio=case["gamma"],
        upstream_pressure_Pa=case["P1"],
        downstream_pressure_Pa=case["P2"],
    )
    reference = size_control_valve_g(**case, full_output=True)
    reducers = case["d"] != case["D1"] or case["d"] != case["D2"]
    for name, native, expected, tolerance in (
        ("Kv", result.kv_m3_h, reference["Kv"], 0.01 if reducers else 2e-11),
        ("Y", result.expansion_factor, reference["Y"], 2e-11),
        (
            "FR",
            result.reynolds_factor,
            reference["FR"] if reference["FR"] is not None else 1,
            2e-11,
        ),
    ):
        values.append(
            {
                "quantity": name,
                "native": native,
                "reference": expected,
                "relative_tolerance": tolerance,
                "relative_error": abs(native - expected) / abs(expected),
                "passed": isclose(native, expected, rel_tol=tolerance, abs_tol=1e-12),
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if fluids.__version__ != "1.3.1":
        raise ValueError("requires fluids 1.3.1")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    records = []
    for index, case in enumerate(inputs(), 1):
        row = {"unit": index, "input": case, "measurements": []}
        try:
            run(case, row["measurements"])
            row["passed"] = all(m["passed"] for m in row["measurements"])
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(row)
        print(f"Gas valve {index}/20: {'passed' if row['passed'] else 'FAILED'}", flush=True)
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 20,
        "expected_comparisons": 60,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [r for r in records if not r["passed"]],
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))
    return 0 if summary["passed_inputs"] == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
