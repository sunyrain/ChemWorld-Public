"""Fixed24 finite heat/cool phase-boundary crossings against independent DU flash."""

import argparse
import json
import time
from itertools import product
from pathlib import Path

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model
from chemworld.physchem.coupled_system import CoupledState, CoupledSystem, HeatFlux
from chemworld.physchem.helmholtz_equilibrium_thermal import (
    HelmholtzEquilibriumIsochoricThermodynamics,
)
from chemworld.physchem.helmholtz_flash import HelmholtzTwoPhaseState
from chemworld.physchem.solver_backend import ODESolverPolicy

NAMES = ("Water", "CarbonDioxide", "R134a", "Nitrogen", "Methanol", "Ammonia")
TOL = {"n": 1e-8, "T": 2e-5, "U": 2e-3, "P": 2e-2, "Q": 2e-7, "heat": 1e-8}
TIMES = (0, 1, 2, 3, 4, 5)


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
    sources = {d["INFO"]["NAME"]: d for p in args.sources.glob("*.json")
               if (d := json.loads(p.read_text()))}
    rows = []
    started = time.monotonic()
    for name, form, direction in product(NAMES, ("ode", "residual"), ("heat", "cool")):
        row = {"fluid": name, "form": form, "direction": direction,
               "expected_comparisons": 36, "metrics": {}}
        try:
            model = load_helmholtz_phase_model(sources[name], source=f"CoolProp8/{name}",
                                              eos_index=0)
            minimum = model.fluid.minimum_temperature_K
            gap = model.critical_temperature_K-minimum
            dew = minimum+.6*gap
            rho = CP.PropsSI("Dmolar", "T", dew, "Q", 1, name)
            volume = 2000/rho
            low, high = dew-.01*gap, dew+.01*gap
            begin, end = (low, high) if direction == "heat" else (high, low)
            u0 = 2*CP.PropsSI("Umolar", "T", begin, "Dmolar", rho, name)
            u1 = 2*CP.PropsSI("Umolar", "T", end, "Dmolar", rho, name)
            heat = (u1-u0)/5
            thermo = HelmholtzEquilibriumIsochoricThermodynamics(model, "fluid", volume)

            def heat_flux(_state, heat=heat):
                return HeatFlux(heat)

            system = CoupledSystem(thermo, heat_flux=heat_flux, numerical_form=form,
                energy_scale_J=1000,
                policy=ODESolverPolicy("equilibrium-crossing", method="BDF",
                                       rtol=1e-11, atol=1e-13))
            initial = CoupledState({"fluid": 2}, begin, volume)
            report = system.advance(initial, 5, evaluation_times_s=TIMES)
            reference = CP.AbstractState("HEOS", name)
            phases = []
            for i, state in enumerate(report.trajectory):
                target_u = u0+heat*state.time_s
                reference.update(CP.DmolarUmolar_INPUTS, rho, target_u/2)
                current = thermo.molar_state(state)
                phases.append(isinstance(current, HelmholtzTwoPhaseState))
                q = current.vapor_fraction if phases[-1] else -1
                reference_q = reference.Q() if reference.phase() == CP.iphase_twophase else -1
                row.setdefault("reference_quality_diagnostics", []).append({
                    "time_s": state.time_s, "phase": reference.phase(),
                    "raw_quality": reference.Q(), "normalized_quality": reference_q})
                values = {"n": (state.amounts_mol["fluid"], 2),
                    "T": (state.temperature_K, reference.T()),
                    "U": (thermo.internal_energy_j(state), target_u),
                    "P": (current.pressure_Pa, reference.p()), "Q": (q, reference_q),
                    "heat": (state.boundary_heat_J, heat*state.time_s)}
                row["metrics"].update({f"t{i}:{k}": {"native": float(a), "reference": float(b),
                    "error": abs(float(a-b)), "passed": bool(abs(a-b) <= TOL[k]
                        +(3e-7*abs(b) if k == "P" else 0))} for k, (a, b) in values.items()})
            row["crossed_phase_boundary"] = phases[0] != phases[-1]
            row["energy_balance_residual_J"] = report.energy_balance_residual_J
            row["passed"] = (row["crossed_phase_boundary"]
                             and all(m["passed"] for m in row["metrics"].values()))
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        (args.output/"partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        elapsed = time.monotonic()-started
        print(f"Phase crossing {len(rows)}/24 {name}/{form}/{direction}: "
              f"{sum(r['passed'] for r in rows)} passed; {elapsed:.1f}s; "
              f"ETA {elapsed/len(rows)*(24-len(rows)):.1f}s", flush=True)
    result = {"reference": "CoolProp8.0.0 DU flash plus exact U(t) balance", "units": len(rows),
        "units_passed": sum(r["passed"] for r in rows), "expected_comparisons": 864,
        "executed_comparisons": sum(len(r["metrics"]) for r in rows),
        "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
