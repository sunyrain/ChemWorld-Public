"""Fixed coupled thermal consumers versus independent CoolProp/LSODA dynamics."""

import argparse
import json
import time
from itertools import product
from pathlib import Path

from scipy.integrate import solve_ivp

from chemworld.data.helmholtz import load_helmholtz_fluid
from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    MaterialPort,
    ProcessContribution,
)
from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics
from chemworld.physchem.solver_backend import ODESolverPolicy

NAMES = ("Water", "CarbonDioxide", "R134a", "Nitrogen", "Methanol", "Ammonia")
TIMES = (0, 1, 2, 3, 4, 5)
TOLERANCES = {"amount_mol": 1e-8, "temperature_K": 2e-5, "internal_energy_J": 2e-3,
              "outlet_enthalpy_J": 2e-3, "heat_J": 1e-8}


def main():
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tight-policy", action="store_true")
    args = parser.parse_args()
    if CoolProp.__version__ != "8.0.0":
        raise ValueError("fixed reference version changed")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in args.sources.glob("*.json"):
        data = json.loads(path.read_text())
        sources[data["INFO"]["NAME"]] = data
    rows = []
    started = time.monotonic()
    for name, form, process in product(NAMES, ("ode", "residual"), ("closed", "outlet")):
        row = {"fluid": name, "numerical_form": form, "process": process,
               "expected_comparisons": 30, "points": []}
        try:
            source = sources[name]
            fluid = load_helmholtz_fluid(source, source=f"CoolProp8/{name}", eos_index=0)
            temperature = (source["STATES"]["critical"]["T"]+fluid.maximum_temperature_K)/2
            density = .25*fluid.reducing_molar_density_mol_m3
            volume = 2000/density
            model = HelmholtzIsochoricThermodynamics(fluid, "fluid", volume)
            rate = .02 if process == "outlet" else 0
            reference = CP.AbstractState("HEOS", name)
            reference.specify_phase(CP.iphase_gas)

            def rhs(_t, y, volume=volume, rate=rate, reference=reference):
                T, n, _out = y
                rho = n/(volume*1e-3)
                reference.update(CP.DmolarT_INPUTS, rho, T)
                u, h, cv = reference.umolar(), reference.hmolar(), reference.cvmolar()
                urho = reference.first_partial_deriv(CP.iUmolar, CP.iDmolar, CP.iT)
                return [(60-rate*h+rate*(u+rho*urho))/(n*cv), -rate, rate*h]

            independent = solve_ivp(rhs, (0, 5), [temperature, 2, 0], t_eval=TIMES,
                                    method="LSODA", rtol=1e-11, atol=1e-11)
            if not independent.success:
                raise RuntimeError("independent LSODA: "+independent.message)
            processes = ()
            ports = ()
            if rate:
                ports = (MaterialPort("outlet", "fluid", "vessel", "sink", "out"),)

                def outlet(state, rate=rate, model=model):
                    return ProcessContribution(port_rates_mol_s={"outlet": rate},
                        port_enthalpies_J_mol={"outlet": model.molar_state(state).enthalpy_J_mol})

                processes = (CoupledProcess("same-fluid-outlet", outlet),)
            system = CoupledSystem(model, processes, ports=ports,
                heat_flux=lambda _: HeatFlux(60), numerical_form=form,
                policy=ODESolverPolicy("conv24-thermal-tight" if args.tight_policy
                    else "conv24-thermal", method="BDF",
                    rtol=1e-11 if args.tight_policy else 1e-9,
                    atol=1e-13 if args.tight_policy else 1e-11),
                energy_scale_J=1000)
            start = CoupledState({"fluid": 2}, temperature, volume)
            report = system.advance(start, 5, evaluation_times_s=TIMES)
            for state, expected in zip(report.trajectory, independent.y.T, strict=True):
                T, n, outlet_energy = expected
                reference.update(CP.DmolarT_INPUTS, n/(volume*1e-3), T)
                values = {"amount_mol": (state.amounts_mol["fluid"], n),
                    "temperature_K": (state.temperature_K, T),
                    "internal_energy_J": (model.internal_energy_j(state), n*reference.umolar()),
                    "outlet_enthalpy_J": (state.exchange_energy_J.get("outlet", 0), outlet_energy),
                    "heat_J": (state.boundary_heat_J, 60*state.time_s)}
                metrics = {key: {"native": a, "reference": float(b), "error": abs(a-b),
                    "passed": bool(abs(a-b) <= TOLERANCES[key])} for key, (a, b) in values.items()}
                row["points"].append({"time_s": state.time_s, "metrics": metrics})
            row["energy_balance_residual_J"] = report.energy_balance_residual_J
            row["checks"] = {
                "energy_balance": abs(report.energy_balance_residual_J) <= 2e-3,
                "duration": report.actual_duration_s == 5,
                "fixed_volume": all(s.volume_L == volume for s in report.trajectory),
                "outlet_amount": abs(
                    report.final_state.exchanged_mol.get("outlet", 0)-5*rate) < 1e-8}
            row["passed"] = all(row["checks"].values()) and all(
                m["passed"] for point in row["points"] for m in point["metrics"].values())
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        elapsed = time.monotonic()-started
        print(f"Helmholtz thermal {len(rows)}/24 {name}/{form}/{process}: "
              f"{sum(r['passed'] for r in rows)} passed; {elapsed:.1f}s; "
              f"ETA {elapsed/len(rows)*(24-len(rows)):.1f}s", flush=True)
    result = {"reference": "CoolProp8.0.0 + independent SciPy LSODA", "units": len(rows),
        "policy_block": "tight_supplement" if args.tight_policy else "original",
        "units_passed": sum(r["passed"] for r in rows), "expected_comparisons": 720,
        "executed_comparisons": sum(len(p["metrics"]) for r in rows for p in r["points"]),
        "comparisons_passed": sum(m["passed"] for r in rows for p in r["points"]
                                   for m in p["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    (args.output/"summary.json").write_text(json.dumps(result, ensure_ascii=True, indent=2),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
