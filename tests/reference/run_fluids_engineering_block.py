"""Fixed 46-input / 148-comparison IEC-orifice development block.

Independent reference packages are imported here only, never in production.
Write each attempt to a new external directory; retain every failed input.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import product
from math import isclose, pi
from pathlib import Path

import fluids
from fluids.control_valve import Reynolds_factor, Reynolds_valve, size_control_valve_l
from fluids.flow_meter import (
    C_Reader_Harris_Gallagher,
    differential_pressure_meter_solver,
    dP_orifice,
)

from chemworld.physchem.flow_meters import (
    LiquidOrificeSpec,
    liquid_orifice_bore,
    liquid_orifice_mass_flow,
    liquid_orifice_pressure,
)
from chemworld.physchem.fluids_devices import (
    LiquidControlValveSpec,
    liquid_valve_flow,
    liquid_valve_pressure_drop,
    size_liquid_control_valve,
    valve_reynolds_factor,
)
from chemworld.physchem.transport import FluidState


def fixed_inputs():
    for taps, diameter, beta, reynolds in product(
        ("corner", "flange", "D-D/2"), (.05, .1), (.4, .7), (20000, 80000),
    ):
        yield {"family": "orifice", "taps": taps, "D": diameter, "beta": beta,
            "reynolds": reynolds, "rho": 998.2, "mu": .001}
    for trim, reynolds in product((True, False), (1, 9.999, 10, 100, 9999, 10000)):
        yield {"family": "FR", "full_trim": trim, "reynolds": reynolds,
            "FL": .9, "Kv": .02, "d": .015}
    for index, (diameter, pipe, fl, fd) in enumerate((
        (.15, .15, .9, .46), (.1, .1, .6, .98),
        (.15, .2, .9, .46), (.15, .2, .6, .98),
    )):
        yield {"family": "sizing", "index": index, "d": diameter, "D": pipe,
            "FL": fl, "Fd": fd, "rho": 965.4, "mu": 3.1472e-4,
            "P1": 680e3, "P2": 220e3, "Psat": 70100, "Pc": 22120e3, "Q": .1}
    for coefficient, flow in product((.002, .02, .2), (1e-8, 1e-7)):
        yield {"family": "installed_valve", "Kv": coefficient, "Q": flow,
            "d": .001, "FL": .9, "Fd": .5, "rho": 1000, "mu": .001,
            "P2": 101325, "Psat": 2000, "Pc": 22e6}


def measurement(name, native, reference, tolerance):
    return {"quantity": name, "native": native, "reference": reference,
        "absolute_error": abs(native - reference),
        "relative_error": abs(native - reference) / max(abs(reference), 1e-300),
        "relative_tolerance": tolerance,
        "passed": isclose(native, reference, rel_tol=tolerance, abs_tol=1e-12)}


def run_case(case):
    family = case["family"]
    if family == "orifice":
        diameter, bore = case["D"], case["D"] * case["beta"]
        mass = case["reynolds"] * pi * diameter * case["mu"] / 4
        spec = LiquidOrificeSpec(diameter, bore, taps=case["taps"],
            installation_note="Reference standard-compliant installation")
        reference_taps = "D" if case["taps"] == "D-D/2" else case["taps"]
        coefficient = C_Reader_Harris_Gallagher(D=diameter, Do=bore,
            rho=case["rho"], mu=case["mu"], m=mass, taps=reference_taps)
        p2 = differential_pressure_meter_solver(D=diameter, D2=bore, P1=1e6,
            m=mass, rho=case["rho"], mu=case["mu"], epsilon_specified=1,
            meter_type="ISO 5167 orifice", taps=reference_taps)
        drop = 1e6 - p2
        loss = dP_orifice(diameter, bore, 1e6, p2, coefficient)
        native = liquid_orifice_pressure(spec, mass_flow_kg_s=mass,
            density_kg_m3=case["rho"], viscosity_Pa_s=case["mu"])
        inverse = liquid_orifice_mass_flow(spec, differential_pressure_Pa=drop,
            density_kg_m3=case["rho"], viscosity_Pa_s=case["mu"])
        resized = liquid_orifice_bore(replace(spec, bore_diameter_m=.5 * diameter),
            mass_flow_kg_s=mass, differential_pressure_Pa=drop,
            density_kg_m3=case["rho"], viscosity_Pa_s=case["mu"])
        return [measurement("C", native.discharge_coefficient, coefficient, 2e-10),
            measurement("tap_differential_Pa", native.differential_pressure_Pa, drop, 2e-9),
            measurement("permanent_loss_Pa", native.permanent_pressure_loss_Pa, loss, 2e-9),
            measurement("mass_inverse_kg_s", inverse.mass_flow_kg_s, mass, 2e-8),
            measurement("bore_inverse_m", resized, bore, 2e-8)]
    if family == "FR":
        reference = Reynolds_factor(case["FL"], case["Kv"], case["d"] * 1000,
            case["reynolds"], case["full_trim"])
        native = valve_reynolds_factor(recovery_factor=case["FL"], kv_m3_h=case["Kv"],
            diameter_m=case["d"], reynolds=case["reynolds"], full_trim=case["full_trim"])
        return [measurement("FR", native, reference, 2e-10)]
    spec = LiquidControlValveSpec(case.get("Kv", 1), case["d"],
        case.get("D", case["d"]), case.get("D", case["d"]), case["FL"], case["Fd"],
        source_note="Fixed IEC development reference geometry")
    fluid = FluidState(case["rho"], case["mu"])
    if family == "sizing":
        reference = size_control_valve_l(rho=case["rho"], Psat=case["Psat"], Pc=case["Pc"],
            mu=case["mu"], P1=case["P1"], P2=case["P2"], Q=case["Q"],
            D1=case["D"], D2=case["D"], d=case["d"], FL=case["FL"], Fd=case["Fd"])
        native = size_liquid_control_valve(spec, fluid, flow_m3_s=case["Q"],
            upstream_pressure_Pa=case["P1"], downstream_pressure_Pa=case["P2"],
            saturation_pressure_Pa=case["Psat"], critical_pressure_Pa=case["Pc"])
        return [measurement("nominal_Kv_m3_h", native.kv_m3_h, reference,
            .01 if spec.has_reducers else 2e-10)]
    reynolds = Reynolds_valve(case["mu"] / case["rho"], case["Q"] * 3600,
        case["d"] * 1000, case["FL"], case["Fd"], case["Kv"])
    fr = 1 if reynolds > 10000 else Reynolds_factor(case["FL"], case["Kv"],
        case["d"] * 1000, reynolds,
        case["Kv"] / (case["d"] * 1000)**2 <= .016 * .865)
    # IEC reference density is fixed independently, not imported from native code.
    reference_drop = 1000 * case["rho"] / 999.1032907570233 * (
        case["Q"] * 3600 / (.1 * case["Kv"] * fr))**2
    point = liquid_valve_pressure_drop(spec, fluid, flow_m3_s=case["Q"],
        downstream_pressure_Pa=case["P2"], saturation_pressure_Pa=case["Psat"],
        critical_pressure_Pa=case["Pc"])
    inverse = liquid_valve_flow(spec, fluid, upstream_pressure_Pa=case["P2"] + reference_drop,
        downstream_pressure_Pa=case["P2"], saturation_pressure_Pa=case["Psat"],
        critical_pressure_Pa=case["Pc"])
    return [measurement("installed_pressure_Pa", point.pressure_drop_Pa, reference_drop, 2e-10),
        measurement("installed_flow_inverse_m3_s", inverse.flow_m3_s, case["Q"], 2e-8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if fluids.__version__ != "1.3.1":
        raise RuntimeError("fixed reference requires fluids 1.3.1")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    records = []
    for index, case in enumerate(fixed_inputs(), 1):
        record = {"unit": index, "input": case}
        try:
            record["measurements"] = run_case(case)
            record["passed"] = all(m["passed"] for m in record["measurements"])
        except Exception as exc:
            record.update(passed=False, measurements=[], error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        print(f"Fluids engineering {index}/46 {case['family']}: "
            f"{'passed' if record['passed'] else 'FAILED'}", flush=True)
    result = {"mode": "development", "reference_version": fluids.__version__,
        "expected_inputs": 46, "expected_comparisons": 148,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "failures": [r for r in records if not r["passed"]],
        "elapsed_s": time.perf_counter() - start, "records": records}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "failures"}}))
    return 0 if result["passed_inputs"] == 46 and result["passed_comparisons"] == 148 else 1


if __name__ == "__main__":
    raise SystemExit(main())
