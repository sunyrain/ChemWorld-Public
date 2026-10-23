"""54 fixed units: same-EOS devices and actual shared material/energy dynamics."""

import argparse
import json
import time
from pathlib import Path

from scipy.integrate import solve_ivp

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model
from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    MaterialPort,
    ProcessContribution,
)
from chemworld.physchem.helmholtz_processes import HelmholtzPressureStage, HelmholtzStream
from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics
from chemworld.physchem.solver_backend import ODESolverPolicy

NAMES = ("Water", "CarbonDioxide", "R134a", "Nitrogen", "Methanol", "Ammonia")
TIMES = (0, 1, 2, 3, 4, 5)
STATIC = {"T": ("temperature_K", "T", 2e-5), "P": ("pressure_Pa", "P", 2e-3),
    "rho": ("molar_density_mol_m3", "Dmolar", 2e-5), "H": ("enthalpy_J_mol", "Hmolar", 2e-4),
    "U": ("internal_energy_J_mol", "Umolar", 2e-4), "S": ("entropy_J_mol_K", "Smolar", 2e-6)}
DYNAMIC_TOL = {"amount": 1e-8, "T": 2e-5, "U": 2e-3, "feed_H": 2e-3, "fed_amount": 1e-8}


def metric(a, b, atol, rtol=0):
    return {"native": float(a), "reference": float(b), "error": abs(float(a-b)),
            "passed": bool(abs(a-b) <= atol+rtol*abs(b))}


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
    for name in NAMES:
        model = load_helmholtz_phase_model(sources[name], source=f"CoolProp8/{name}", eos_index=0)
        minimum, critical = model.fluid.minimum_temperature_K, model.critical_temperature_K
        for mode in ("throttle", "compressor", "expander"):
            row = {"kind": "device", "fluid": name, "mode": mode,
                   "expected_comparisons": 7, "metrics": {}}
            stage_result = None
            feed_h = None
            try:
                if mode == "throttle":
                    T = minimum+.65*(critical-minimum)
                    psat = CP.PropsSI("P", "T", T, "Q", 0, name)
                    p_in, p_out, efficiency = 1.3*psat, .7*psat, 1
                else:
                    T = critical+.2*(model.fluid.maximum_temperature_K-critical)
                    p_in = .3*model.critical_pressure_Pa
                    p_out = (.6 if mode == "compressor" else .15)*model.critical_pressure_Pa
                    efficiency = .8
                reference_h = CP.PropsSI("Hmolar", "T", T, "P", p_in, name)
                reference_s = CP.PropsSI("Smolar", "T", T, "P", p_in, name)
                feed_h = reference_h
                if mode != "throttle":
                    hs = CP.PropsSI("Hmolar", "P", p_out, "Smolar", reference_s, name)
                    feed_h += ((hs-reference_h)/efficiency if mode == "compressor"
                               else efficiency*(hs-reference_h))
                inlet = model.solve_tp(temperature_K=T, pressure_Pa=p_in)
                if not inlet.success:
                    raise ValueError("inlet:"+inlet.failure_reason)
                device = HelmholtzPressureStage(model, mode, p_out, efficiency)
                stage_result = device.evaluate(HelmholtzStream(inlet.state, .02))
                row["diagnostics"] = dict(stage_result.diagnostics)
                if not stage_result.success:
                    raise ValueError(stage_result.failure_reason)
                state = stage_result.outlet.state
                row["metrics"] = {k: metric(getattr(state, native),
                    CP.PropsSI(ref, "P", p_out, "Hmolar", feed_h, name), atol, 3e-7)
                    for k, (native, ref, atol) in STATIC.items()}
                row["metrics"]["shaft_W"] = metric(stage_result.shaft_work_W,
                                                     .02*(feed_h-reference_h), 2e-5, 3e-7)
                row["passed"] = all(m["passed"] for m in row["metrics"].values())
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            for form in ("ode", "residual"):
                dynamic = {"kind": "finite_receiver", "fluid": name, "mode": mode,
                           "form": form, "expected_comparisons": 30, "metrics": {}}
                try:
                    if stage_result is None or not stage_result.success:
                        raise ValueError("upstream pressure stage failed")
                    volume = 2000/(.05*model.fluid.reducing_molar_density_mol_m3)
                    initial_T = (critical+model.fluid.maximum_temperature_K)/2
                    thermo = HelmholtzIsochoricThermodynamics(model.fluid, "fluid", volume)
                    reference = CP.AbstractState("HEOS", name)
                    reference.specify_phase(CP.iphase_gas)

                    def rhs(_t, y, reference=reference, volume=volume, feed_h=feed_h):
                        T, n, _enthalpy = y
                        rho = n/(volume*1e-3)
                        reference.update(CP.DmolarT_INPUTS, rho, T)
                        un = reference.umolar()+rho*reference.first_partial_deriv(
                            CP.iUmolar, CP.iDmolar, CP.iT)
                        return [(.02*feed_h-.02*un)/(n*reference.cvmolar()), .02, .02*feed_h]

                    independent = solve_ivp(rhs, (0, 5), [initial_T, 2, 0], t_eval=TIMES,
                                            method="LSODA", rtol=1e-11, atol=1e-11)
                    if not independent.success:
                        raise ValueError("independent dynamics:"+independent.message)
                    streams = (stage_result.outlet.separate() if mode == "throttle"
                               else (stage_result.outlet,))
                    ports = tuple(MaterialPort(f"inlet-{i}", "fluid", "device", "tank", "in")
                                  for i in range(len(streams)))
                    pieces = [stream.contribution(port.port_id)
                              for stream, port in zip(streams, ports, strict=True)]
                    contribution = ProcessContribution(
                        port_rates_mol_s={k: v for c in pieces
                                          for k, v in c.port_rates_mol_s.items()},
                        port_enthalpies_J_mol={k: v for c in pieces
                                              for k, v in c.port_enthalpies_J_mol.items()})

                    def process(_state, contribution=contribution):
                        return contribution

                    system = CoupledSystem(thermo, (CoupledProcess("device-feed", process),),
                        ports=ports, numerical_form=form, energy_scale_J=1000,
                        policy=ODESolverPolicy("pressure-device-feed", method="BDF",
                                               rtol=1e-11, atol=1e-13))
                    report = system.advance(CoupledState({"fluid": 2}, initial_T, volume), 5,
                                              evaluation_times_s=TIMES)
                    for i, (state, expected) in enumerate(zip(report.trajectory,
                                                          independent.y.T, strict=True)):
                        T, n, energy = expected
                        reference.update(CP.DmolarT_INPUTS, n/(volume*1e-3), T)
                        values = {"amount": (state.amounts_mol["fluid"], n),
                            "T": (state.temperature_K, T),
                            "U": (thermo.internal_energy_j(state), n*reference.umolar()),
                            "feed_H": (sum(state.exchange_energy_J.values()), energy),
                            "fed_amount": (sum(state.exchanged_mol.values()), .02*state.time_s)}
                        dynamic["metrics"].update({f"t{i}:{k}": metric(a, b, DYNAMIC_TOL[k])
                                                   for k, (a, b) in values.items()})
                    dynamic["energy_balance_residual_J"] = report.energy_balance_residual_J
                    dynamic["passed"] = all(m["passed"] for m in dynamic["metrics"].values())
                except Exception as exc:
                    dynamic.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                rows.append(dynamic)
            (args.output / "partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            elapsed = time.monotonic()-started
            print(f"Pressure consumers {len(rows)}/54 {name}/{mode}: "
                  f"{sum(r['passed'] for r in rows)} passed; {elapsed:.1f}s; "
                  f"ETA {elapsed/len(rows)*(54-len(rows)):.1f}s", flush=True)
    result = {"reference": "CoolProp8.0.0 + independent SciPy LSODA", "units": len(rows),
        "units_passed": sum(r["passed"] for r in rows), "expected_comparisons": 1206,
        "executed_comparisons": sum(len(r["metrics"]) for r in rows),
        "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
