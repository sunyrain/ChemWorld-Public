# ruff: noqa: B023 -- unit callbacks run immediately, never after the loop advances.
"""Fixed pure molecular consumer comparison; run through the locked environment."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from run_molecular_potentials import RHO_SCALE, models

from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    MaterialPort,
)
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_flash import flash_at_pressure
from chemworld.physchem.helmholtz_processes import HelmholtzPressureStage, HelmholtzStream
from chemworld.physchem.helmholtz_solvers import HelmholtzAncillary, HelmholtzPhaseModel
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics
from chemworld.physchem.molecular_consumers import MolecularHelmholtzFluid
from chemworld.physchem.solver_backend import ODESolverPolicy

NAMES = ("pressure_Pa", "compressibility_factor", "helmholtz_energy_J_mol",
         "internal_energy_J_mol", "enthalpy_J_mol", "entropy_J_mol_K", "cv_J_mol_K",
         "cp_J_mol_K", "dP_dT_rho_Pa_K", "dP_drho_T_Pa_m3_mol", "speed_of_sound_m_s",
         "gibbs_energy_J_mol")


def bind(model):
    ideal = HelmholtzFluid("A", "", .04, R, 158.4, .31*RHO_SCALE, 84, 1200, 1e9,
        (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2., "a2": 3.}),
         HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5})), (),
        "explicit synthetic monatomic reference, nonzero offsets")
    return MolecularHelmholtzFluid.from_reference(model, ideal)


def phase(model):
    tc, rc, pc = 158.4, .31*RHO_SCALE, .13006*R*120*RHO_SCALE
    return HelmholtzPhaseModel(bind(model), tc, rc, pc, {
        "rhoL": HelmholtzAncillary("rhoLnoexp", tc, rc, 84, tc, (2.,), (.35,)),
        "rhoV": HelmholtzAncillary("rhoV", tc, rc, 84, tc, (-5.,), (.5,), True),
        "pS": HelmholtzAncillary("pV", tc, pc, 84, tc, (-5.,), (1.,), True),
    })


def target(ref, t, rho):
    x = np.array([1.])
    a, at, att, ad, add, adt = (ref.get_Arxy(i, j, t, rho, x)
        for i, j in ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1)))
    tau, delta = 158.4/t, rho/(.31*RHO_SCALE)
    a0 = math.log(delta)+2+3*tau+1.5*math.log(tau)
    u = R*t*(1.5+3*tau+at)
    s = R*(1.5+3*tau+at-a0-a)
    z, stiffness, thermal = 1+ad, 1+2*ad+add, 1+ad-adt
    cv = R*(1.5-att)
    p, h = rho*R*t*z, u+R*t*z
    values = (p, z, R*t*(a0+a), u, h, s, cv, cv+R*thermal**2/stiffness,
              rho*R*thermal, R*t*stiffness,
              math.sqrt(R*t/.04*(stiffness+R*thermal**2/cv)), h-t*s)
    return dict(zip(NAMES, values, strict=True))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    failures, units, checks, passed = [], 0, 0, 0

    def unit(label, names, actual, expected):
        nonlocal units, checks, passed
        records, results = [], {}
        for side, callback in (("native", actual), ("reference", expected)):
            try:
                results[side] = callback()
            except Exception as exc:
                results[side] = f"{type(exc).__name__}: {exc}"
        for name in names:
            absolute = (1e-9 if name == "entropy_violation" else
                        1e-4 if name == "pressure_Pa" else
                        2e-5 if name.endswith("J_mol") or "energy" in name else 2e-7)
            row = {"unit": label, "quantity": name, "atol": absolute, "rtol": 2e-7}
            for side, result in results.items():
                try:
                    if isinstance(result, str):
                        raise ValueError(result)
                    value = float(result[name])
                    if not math.isfinite(value):
                        raise ValueError("nonfinite response")
                    row[side] = value
                except Exception as exc:
                    row[side+"_exception"] = str(exc)
            row["passed"] = "native" in row and "reference" in row and math.isclose(
                row["native"], row["reference"], rel_tol=2e-7, abs_tol=absolute)
            checks += 1
            passed += row["passed"]
            records.append(row)
            if not row["passed"]:
                failures.append(row)
        with (output/"records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"label": label, "comparisons": records})+"\n")
        units += 1
        elapsed = time.perf_counter()-started
        print(f"Consumers {units}/31 units; {units/elapsed:.2f}/s; "
              f"ETA {(31-units)*elapsed/units:.1f}s; failures {len(failures)}", flush=True)

    for case in range(3):
        native, ref, _ = models(case, "pure")
        fluid = bind(native)
        for reduced_rho in (.01, .3, .8):
            rho = reduced_rho*RHO_SCALE
            unit(f"state-{case}-{reduced_rho}", NAMES,
                 lambda: fluid.state(temperature_K=360, molar_density_mol_m3=rho).__dict__,
                 lambda: target(ref, 360, rho))
    native, ref, _ = models(2, "pure")
    model = phase(native)
    for t in (108., 120., 132.):
        def endpoints():
            roots = ref.model.pure_VLE_T(t/120,
                model.ancillaries["rhoL"].value(t)/RHO_SCALE,
                model.ancillaries["rhoV"].value(t)/RHO_SCALE, 100)
            return tuple(target(ref, t, rho*RHO_SCALE) | {"density": rho*RHO_SCALE}
                         for rho in roots)

        def saturation():
            result = model.saturation(t)
            if not result.success:
                raise ValueError(result.failure_reason)
            return {"liquid_density": result.liquid.molar_density_mol_m3,
                    "gas_density": result.gas.molar_density_mol_m3,
                    "pressure_Pa": result.pressure_Pa}

        def source_saturation():
            liquid, gas = endpoints()
            return {"liquid_density": liquid["density"], "gas_density": gas["density"],
                    "pressure_Pa": gas["pressure_Pa"]}

        unit(f"saturation-{t}", ("liquid_density", "gas_density", "pressure_Pa"),
             saturation, source_saturation)
        for q in (.2, .8):
            for coordinate, name in (("H", "enthalpy_J_mol"), ("S", "entropy_J_mol_K")):
                def flash():
                    liquid, gas = endpoints()
                    result = flash_at_pressure(model, pressure_Pa=gas["pressure_Pa"],
                        coordinate=coordinate, value=(1-q)*liquid[name]+q*gas[name])
                    if not result.success:
                        raise ValueError(result.failure_reason)
                    return {"temperature": result.state.temperature_K,
                            "quality": result.state.vapor_fraction}
                unit(f"flash-{t}-{q}-{coordinate}", ("temperature", "quality"),
                     flash, lambda: {"temperature": t, "quality": q})
    for mode in ("throttle", "compressor", "expander"):
        def stage():
            state = model.solve_tp(temperature_K=360,
                                    pressure_Pa=3*model.critical_pressure_Pa).state
            device = HelmholtzPressureStage(model, mode,
                (4 if mode == "compressor" else 2)*model.critical_pressure_Pa,
                1 if mode == "throttle" else .8)
            result = device.evaluate(HelmholtzStream(state, .01))
            if not result.success:
                raise ValueError(result.failure_reason)
            return {"rate": result.outlet.molar_rate_mol_s,
                    "energy_residual_W": result.diagnostics["energy_residual_W"],
                    "entropy_violation": min(0., result.diagnostics["entropy_generation_W_K"])}
        unit(f"stage-{mode}", ("rate", "energy_residual_W", "entropy_violation"), stage,
             lambda: {"rate": .01, "energy_residual_W": 0., "entropy_violation": 0.})
    for form in ("ode", "residual"):
        for mode in ("heat", "feed"):
            def inventory():
                thermal = HelmholtzIsochoricThermodynamics(model.fluid, "A", 2., 5.)
                start = CoupledState({"A": 2.}, 360, 2.)
                options = {"numerical_form": form,
                    "policy": ODESolverPolicy("molecular-consumers", method="BDF",
                                              rtol=1e-11, atol=1e-13)}
                if mode == "heat":
                    options["heat_flux"] = lambda _: HeatFlux(20.)
                else:
                    feed = HelmholtzStream(model.fluid.state(
                        temperature_K=480, molar_density_mol_m3=1000), .01)
                    options["ports"] = (MaterialPort("feed", "A", "source", "vessel", "in"),)
                    options["processes"] = (CoupledProcess("feed",
                        lambda _: feed.contribution("feed")),)
                report = CoupledSystem(thermal, **options).advance(start, 2.)
                final = report.final_state
                return {"amount": final.amounts_mol["A"], "time": final.time_s,
                        "energy_change_J": thermal.internal_energy_j(final)
                        -thermal.internal_energy_j(start)}
            unit(f"inventory-{form}-{mode}", ("amount", "time", "energy_change_J"), inventory,
                 lambda: {"amount": 2. if mode == "heat" else 2.02, "time": 2.,
                    "energy_change_J": 40. if mode == "heat" else
                    .02*target(ref, 480, 1000)["enthalpy_J_mol"]})
    assert units == 31 and checks == 162
    summary = {"mode": "development", "units": units, "comparisons": checks, "passed": passed,
               "failed": len(failures), "failures": failures,
               "elapsed_s": time.perf_counter()-started}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
