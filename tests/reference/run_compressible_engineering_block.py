"""Prospective fixed 60 inputs / 420 comparisons; all failures remain visible."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

import fluids
import fluids.compressible as reference

from chemworld.physchem.gas_flow_thermo import (
    critical_gas_pressure,
    critical_gas_temperature,
    gas_efficiency_conversion,
    gas_polytropic_exponent,
    gas_stagnation_pressure,
    gas_stagnation_temperature,
    is_critical_gas_flow,
    isentropic_gas_outlet_temperature,
    isentropic_gas_work_molar,
    isothermal_gas_work_molar,
    solve_isentropic_gas_work,
    stagnation_energy_J_kg,
    stagnation_temperature_from_velocity,
)
from chemworld.physchem.gas_pipeline import (
    GAS_PIPELINE_METHODS,
    EmpiricalGasPipelineSpec,
    solve_empirical_gas_pipeline,
)


def inputs():
    for ratio, gamma, efficiency in product((1.01, 2, 10), (1.1, 1.4, 1.67), (0.65, 0.95)):
        yield {
            "family": "compression",
            "ratio": ratio,
            "k": gamma,
            "eta": efficiency,
            "P1": 1e5,
            "P2": 1e5 * ratio,
            "T": 300,
            "Z": 0.97,
        }
    for gamma, velocity in product((1.1, 1.4, 1.67), (50, 350)):
        yield {"family": "stagnation", "k": gamma, "V": velocity, "T": 280, "P": 1e5, "Cp": 1005}
    for method, variant in product(GAS_PIPELINE_METHODS, range(4)):
        args = {
            "D": 0.340,
            "P1": 9e6,
            "P2": 2e6,
            "L": 160e3,
            "SG": 0.693,
            "Tavg": 277.15,
            "Ts": 288.7,
            "Ps": 101325,
            "Zavg": 1.0,
        }
        if variant == 1:
            args.update(D=0.05, P1=3e5, P2=2e5, L=100, SG=0.6, Tavg=300)
        if method == "Spitzglass_low":
            args.update(P1=6720.3199, P2=0, D=0.154051, L=54.864, SG=0.6, Tavg=288.7)
        if variant == 2:
            args.update(Ts=310, Ps=120000)
        if variant == 3:
            args.update(E=0.85, Zavg=0.91)
        if method == "Oliphant":
            args.update(E=1, Zavg=1)
        if method in {"IGT", "Muller"}:
            args["mu"] = 1e-5
        yield {"family": "pipeline", "method": method, "variant": variant, "args": args}


def compare(name, native, expected, tolerance=2e-12):
    return {
        "quantity": name,
        "native": native,
        "reference": expected,
        "relative_error": abs(native - expected) / max(abs(expected), 1e-300),
        "relative_tolerance": tolerance,
        "absolute_tolerance": 1e-9,
        "passed": isclose(native, expected, rel_tol=tolerance, abs_tol=1e-9),
    }


def run_compression(case, values):
    args = {
        "inlet_pressure_Pa": case["P1"],
        "outlet_pressure_Pa": case["P2"],
        "inlet_temperature_K": case["T"],
        "heat_capacity_ratio": case["k"],
        "efficiency": case["eta"],
        "compressibility_factor": case["Z"],
    }
    ref = {
        "P1": case["P1"],
        "P2": case["P2"],
        "T1": case["T"],
        "k": case["k"],
        "eta": case["eta"],
        "Z": case["Z"],
    }
    work = reference.isentropic_work_compression(**ref)
    values.append(
        compare(
            "isothermal_work_J_mol",
            isothermal_gas_work_molar(
                inlet_pressure_Pa=case["P1"],
                outlet_pressure_Pa=case["P2"],
                temperature_K=case["T"],
                compressibility_factor=case["Z"],
            ),
            reference.isothermal_work_compression(
                P1=case["P1"], P2=case["P2"], T=case["T"], Z=case["Z"]
            ),
        )
    )
    values.append(compare("isentropic_work_J_mol", isentropic_gas_work_molar(**args), work))
    values.append(
        compare(
            "outlet_temperature_K",
            isentropic_gas_outlet_temperature(
                **{k: v for k, v in args.items() if k != "compressibility_factor"}
            ),
            reference.isentropic_T_rise_compression(**{k: v for k, v in ref.items() if k != "Z"}),
        )
    )
    efficiency_args = {
        "inlet_pressure_Pa": case["P1"],
        "outlet_pressure_Pa": case["P2"],
        "heat_capacity_ratio": case["k"],
    }
    eta_s = reference.isentropic_efficiency(
        P1=case["P1"], P2=case["P2"], k=case["k"], eta_p=case["eta"]
    )
    values.append(
        compare(
            "eta_s",
            gas_efficiency_conversion(**efficiency_args, polytropic_efficiency=case["eta"]),
            eta_s,
        )
    )
    values.append(
        compare(
            "eta_p",
            gas_efficiency_conversion(**efficiency_args, isentropic_efficiency=eta_s),
            reference.isentropic_efficiency(P1=case["P1"], P2=case["P2"], k=case["k"], eta_s=eta_s),
        )
    )
    exponent = reference.polytropic_exponent(k=case["k"], eta_p=case["eta"])
    values.append(
        compare(
            "polytropic_exponent",
            gas_polytropic_exponent(heat_capacity_ratio=case["k"], efficiency=case["eta"]),
            exponent,
        )
    )
    values.append(
        compare(
            "polytropic_inverse_efficiency",
            gas_polytropic_exponent(heat_capacity_ratio=case["k"], exponent=exponent),
            reference.polytropic_exponent(k=case["k"], n=exponent),
        )
    )
    for unknown in ("work_J_mol", "inlet_pressure_Pa", "outlet_pressure_Pa", "efficiency"):
        data = {**args, "work_J_mol": work}
        expected = data.pop(unknown)
        values.append(compare(unknown, solve_isentropic_gas_work(**data), expected, 5e-8))


def run_stagnation(case, values):
    t0 = reference.T_stagnation_ideal(T=case["T"], V=case["V"], Cp=case["Cp"])
    p0 = reference.P_stagnation(P=case["P"], T=case["T"], Tst=t0, k=case["k"])
    values.append(
        compare(
            "stagnation_energy_J_kg",
            stagnation_energy_J_kg(case["V"]),
            reference.stagnation_energy(case["V"]),
        )
    )
    values.append(
        compare(
            "velocity_to_T0",
            stagnation_temperature_from_velocity(
                static_temperature_K=case["T"],
                velocity_m_s=case["V"],
                heat_capacity_J_kg_K=case["Cp"],
            ),
            t0,
        )
    )
    values.append(
        compare(
            "temperature_to_P0",
            gas_stagnation_pressure(
                static_temperature_K=case["T"],
                static_pressure_Pa=case["P"],
                stagnation_temperature_K=t0,
                heat_capacity_ratio=case["k"],
            ),
            p0,
        )
    )
    values.append(
        compare(
            "pressure_to_T0",
            gas_stagnation_temperature(
                static_temperature_K=case["T"],
                static_pressure_Pa=case["P"],
                stagnation_pressure_Pa=p0,
                heat_capacity_ratio=case["k"],
            ),
            reference.T_stagnation(T=case["T"], P=case["P"], Pst=p0, k=case["k"]),
        )
    )
    values.append(
        compare(
            "Pcrit",
            critical_gas_pressure(stagnation_pressure_Pa=p0, heat_capacity_ratio=case["k"]),
            reference.P_critical_flow(P=p0, k=case["k"]),
        )
    )
    values.append(
        compare(
            "Tcrit",
            critical_gas_temperature(stagnation_temperature_K=t0, heat_capacity_ratio=case["k"]),
            reference.T_critical_flow(T=t0, k=case["k"]),
        )
    )
    values.append(
        compare(
            "critical_flag",
            is_critical_gas_flow(
                stagnation_pressure_Pa=p0,
                downstream_pressure_Pa=case["P"],
                heat_capacity_ratio=case["k"],
            ),
            reference.is_critical_flow(P1=p0, P2=case["P"], k=case["k"]),
        )
    )


def run_pipeline(case, values):
    args, method = case["args"], case["method"]
    spec = EmpiricalGasPipelineSpec(
        method,
        args["SG"],
        args["Tavg"],
        "Fixed formula comparison; Oliphant high-pressure example is outside recommendation",
        reference_temperature_K=args["Ts"],
        reference_pressure_Pa=args["Ps"],
        compressibility_factor=args["Zavg"],
        efficiency=args.get("E"),
        viscosity_Pa_s=args.get("mu"),
        pressure_basis="gauge" if method == "Spitzglass_low" else "absolute",
    )
    geometry = {
        "upstream_pressure_Pa": args["P1"],
        "downstream_pressure_Pa": args["P2"],
        "diameter_m": args["D"],
        "length_m": args["L"],
    }
    expected_flow = getattr(reference, method)(**args)
    result = solve_empirical_gas_pipeline(spec, **geometry)
    values.append(compare("reference_flow_m3_s", result.reference_flow_m3_s, expected_flow))
    for unknown in geometry:
        data = {**geometry, "reference_flow_m3_s": expected_flow}
        expected = data.pop(unknown)
        values.append(
            compare(
                unknown,
                getattr(solve_empirical_gas_pipeline(spec, **data), unknown),
                expected,
                5e-8,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if fluids.__version__ != "1.3.1":
        raise ValueError("requires fluids reference 1.3.1")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    records = []
    for index, case in enumerate(inputs(), 1):
        row = {"unit": index, "input": case, "measurements": []}
        try:
            {
                "compression": run_compression,
                "stagnation": run_stagnation,
                "pipeline": run_pipeline,
            }[case["family"]](case, row["measurements"])
            row["passed"] = all(m["passed"] for m in row["measurements"])
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(row)
        print(f"Compressible {index}/60: {'passed' if row['passed'] else 'FAILED'}", flush=True)
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 60,
        "expected_comparisons": 420,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [r for r in records if not r["passed"]],
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"failures", "records"}}))
    return 0 if summary["passed_inputs"] == 60 and summary["completed_comparisons"] == 420 else 1


if __name__ == "__main__":
    raise SystemExit(main())
