"""Fixed 24-unit equilibrium thermal dynamics, independent U-coordinate reference."""

import argparse
import json
import time
from itertools import product
from pathlib import Path

from scipy.integrate import solve_ivp

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model
from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    MaterialPort,
    ProcessContribution,
)
from chemworld.physchem.helmholtz_equilibrium_thermal import (
    HelmholtzEquilibriumIsochoricThermodynamics,
)
from chemworld.physchem.helmholtz_flash import HelmholtzTwoPhaseState
from chemworld.physchem.solver_backend import ODESolverPolicy

NAMES = ("Water", "CarbonDioxide", "R134a", "Nitrogen", "Methanol", "Ammonia")
TIMES = (0, 1, 2, 3, 4, 5)
TOL = {"amount": 1e-8, "T": 2e-5, "U": 2e-3, "P": 2e-2, "Q": 2e-7,
       "outlet_H": 2e-3, "heat": 1e-8}


def main():
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if CoolProp.__version__ != "8.0.0":
        raise ValueError("reference version changed")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in args.sources.glob("*.json"):
        data = json.loads(path.read_text())
        sources[data["INFO"]["NAME"]] = data
    rows = []
    started = time.monotonic()
    for name, form, process in product(NAMES, ("ode", "residual"), ("closed", "vapor_outlet")):
        row = {"fluid": name, "form": form, "process": process,
               "expected_comparisons": 42, "metrics": {}}
        try:
            model = load_helmholtz_phase_model(sources[name], source=f"CoolProp8/{name}",
                                              eos_index=0)
            temperature = model.fluid.minimum_temperature_K+.6*(
                model.critical_temperature_K-model.fluid.minimum_temperature_K)
            reference = CP.AbstractState("HEOS", name)
            reference.update(CP.QT_INPUTS, .35, temperature)
            volume = 2000/reference.rhomolar()
            initial_U = 2*reference.umolar()
            rate = .02 if process == "vapor_outlet" else 0

            def rhs(_t, y, rate=rate, volume=volume, reference=reference):
                U, n, _out = y
                reference.update(CP.DmolarUmolar_INPUTS, n/(volume*1e-3), U/n)
                hgas = reference.saturated_vapor_keyed_output(CP.iHmolar) if rate else 0
                return [60-rate*hgas, -rate, rate*hgas]

            independent = solve_ivp(rhs, (0, 5), [initial_U, 2, 0], t_eval=TIMES,
                                    method="LSODA", rtol=1e-11, atol=1e-10)
            if not independent.success:
                raise ValueError("independent dynamics:"+independent.message)
            thermo = HelmholtzEquilibriumIsochoricThermodynamics(model, "fluid", volume)
            ports, processes = (), ()
            if rate:
                ports = (MaterialPort("vapor", "fluid", "vessel", "sink", "out"),)

                def outlet(state, thermo=thermo, rate=rate):
                    current = thermo.molar_state(state)
                    if not isinstance(current, HelmholtzTwoPhaseState):
                        raise ValueError("declared vapor-withdrawal block left two-phase domain")
                    return ProcessContribution(port_rates_mol_s={"vapor": rate},
                        port_enthalpies_J_mol={"vapor": current.gas.enthalpy_J_mol})

                processes = (CoupledProcess("vapor-withdrawal", outlet),)
            system = CoupledSystem(thermo, processes, ports=ports,
                heat_flux=lambda _: HeatFlux(60), numerical_form=form, energy_scale_J=1000,
                policy=ODESolverPolicy("equilibrium-two-phase", method="BDF",
                                       rtol=1e-11, atol=1e-13))
            report = system.advance(CoupledState({"fluid": 2}, temperature, volume), 5,
                                      evaluation_times_s=TIMES)
            for i, (state, expected) in enumerate(zip(report.trajectory,
                                                      independent.y.T, strict=True)):
                U, n, energy = expected
                reference.update(CP.DmolarUmolar_INPUTS, n/(volume*1e-3), U/n)
                current = thermo.molar_state(state)
                if not isinstance(current, HelmholtzTwoPhaseState):
                    raise ValueError("declared reference block left two-phase domain")
                values = {"amount": (state.amounts_mol["fluid"], n),
                    "T": (state.temperature_K, reference.T()),
                    "U": (thermo.internal_energy_j(state), U),
                    "P": (current.pressure_Pa, reference.p()),
                    "Q": (current.vapor_fraction, reference.Q()),
                    "outlet_H": (state.exchange_energy_J.get("vapor", 0), energy),
                    "heat": (state.boundary_heat_J, 60*state.time_s)}
                row["metrics"].update({f"t{i}:{k}": {"native": float(a), "reference": float(b),
                    "error": abs(float(a-b)), "passed": bool(abs(a-b) <= TOL[k]
                        +(3e-7*abs(b) if k == "P" else 0))} for k, (a, b) in values.items()})
            row["energy_balance_residual_J"] = report.energy_balance_residual_J
            row["passed"] = all(m["passed"] for m in row["metrics"].values())
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        (args.output/"partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        elapsed = time.monotonic()-started
        print(f"Equilibrium thermal {len(rows)}/24 {name}/{form}/{process}: "
              f"{sum(r['passed'] for r in rows)} passed; {elapsed:.1f}s; "
              f"ETA {elapsed/len(rows)*(24-len(rows)):.1f}s", flush=True)
    result = {"reference": "CoolProp8.0.0 DU flash + independent U-coordinate LSODA",
        "units": len(rows), "units_passed": sum(r["passed"] for r in rows),
        "expected_comparisons": 1008, "executed_comparisons": sum(len(r["metrics"]) for r in rows),
        "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
