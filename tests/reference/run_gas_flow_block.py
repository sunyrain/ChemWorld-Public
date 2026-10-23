"""Fixed 28 inputs / 172 comparisons for constant-T, constant-Z gas pipes."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

import fluids
from fluids.compressible import P_isothermal_critical_flow, isothermal_gas

from chemworld.physchem.gas_flow import (
    IsothermalGasState,
    isothermal_critical_pressure_ratio,
    isothermal_gas_pipe,
    solve_isothermal_gas_pipe,
)


def inputs():
    for temperature, diameter, length, fraction in product(
        (280, 330), (.02, .2), (10, 1000), (.05, .5, .95),
    ):
        yield {"T": temperature, "D": diameter, "L": length, "fraction": fraction,
            "P1": 1e6, "R": 287.05, "Z": .98, "fd": .02, "family": "pipe"}
    for resistance in (1e-8, 1e-4, 1, 1e4):
        yield {"T": 300, "D": .1, "L": resistance * 5, "resistance": resistance,
            "P1": 1e6, "R": 287.05, "Z": 1, "fd": .02, "family": "critical"}


def compare(name, native, reference, tolerance):
    return {"quantity": name, "native": native, "reference": reference,
        "relative_error": abs(native - reference) / max(abs(reference), 1e-300),
        "relative_tolerance": tolerance,
        "passed": isclose(native, reference, rel_tol=tolerance, abs_tol=1e-12)}


def run(case, comparisons):
    geometry = {"diameter_m": case["D"], "length_m": case["L"],
        "darcy_friction_factor": case["fd"]}
    critical = P_isothermal_critical_flow(P=case["P1"], fd=case["fd"],
        L=case["L"], D=case["D"])
    native_critical = case["P1"] * isothermal_critical_pressure_ratio(**geometry)
    comparisons.append(compare("critical_pressure_Pa", native_critical, critical, 2e-9))
    if case["family"] == "critical":
        return comparisons
    p2 = critical + case["fraction"] * (case["P1"] - critical)
    rho = case["P1"] / (case["Z"] * case["R"] * case["T"])
    mass = isothermal_gas(rho=rho, fd=case["fd"], P1=case["P1"], P2=p2,
        L=case["L"], D=case["D"])
    state = IsothermalGasState(case["T"], case["R"], case["Z"])
    native = isothermal_gas_pipe(state, upstream_pressure_Pa=case["P1"],
        downstream_pressure_Pa=p2, **geometry)
    comparisons.append(compare("mass_flow_kg_s", native.mass_flow_kg_s, mass, 2e-9))
    for unknown in ("upstream_pressure_Pa", "downstream_pressure_Pa", "diameter_m", "length_m"):
        data = {**geometry, "upstream_pressure_Pa": case["P1"],
            "downstream_pressure_Pa": p2, "mass_flow_kg_s": mass}
        expected = data.pop(unknown)
        inverse = solve_isothermal_gas_pipe(state, **data)
        comparisons.append(compare(unknown, getattr(inverse, unknown), expected, 5e-8))
    comparisons.append(compare("inlet_density_kg_m3", native.inlet_density_kg_m3, rho, 1e-12))
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if fluids.__version__ != "1.3.1":
        raise ValueError("reference version must be fluids 1.3.1")
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
        print(f"Isothermal gas {index}/28: {'passed' if row['passed'] else 'FAILED'}", flush=True)
    summary = {"reference_version": fluids.__version__, "mode": "development",
        "expected_inputs": 28, "expected_comparisons": 172, "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [r for r in records if not r["passed"]], "records": records}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))
    return 0 if summary["passed_inputs"] == 28 else 1


if __name__ == "__main__":
    raise SystemExit(main())
