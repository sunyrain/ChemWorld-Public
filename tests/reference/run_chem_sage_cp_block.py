"""Fixed synthetic Cp-input qualification against independent numerical integrals."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp

from chemworld.data.chem_sage import ChemSageDatabase
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_material import CefMaterialSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.solid_solutions import solid_enthalpy
from chemworld.world.solid_world import calphad_world_spec

CASES = (
    ("constant", (30, 0, 0, 0), (), None),
    ("linear", (0, .05, 0, 0), (), None),
    ("quadratic", (0, 0, .00008, 0), (), None),
    ("inverse_square", (0, 0, 0, 1.2e7), (), None),
    ("log", (10, 0, 0, 0), ((2, 99),), None),
    ("general_power", (20, 0, 0, 0), ((.3, .5), (5, -1)), None),
    ("near_singular", (20, 0, 0, 0), ((.3, 1e-14), (5, -1+1e-12)), None),
    ("transitions", (30, .001, 0, 0), ((.2, .5),), None),
    ("ferromagnetic", (30, 0, 0, 0), (), (700, 2.2, 1, .4)),
    ("antiferromagnetic", (30, 0, 0, 0), ((.3, .5),), (-400, -.6, .333, .28)),
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True),
                    encoding="utf-8")


def intervals(case):
    _, coefficients, extra, _ = case
    if case[0] == "transitions":
        return ((800., coefficients, extra, 0.),
                (1200., (35., .002, 0., 0.), ((.1, .5),), 2000.),
                (1800., (40., 0., 0., 0.), ((1., 99.),), -500.))
    return ((1800., coefficients, extra, 0.),)


def original_text(case):
    name, _, extra, magnetic = case
    option = (10 if extra else 7)+(12 if magnetic else 0)
    rows = [f"Synthetic fixed Cp qualification {name}", "1 1 0 1", "FE", "55.845",
            "6 1 2 3 4 5 6", "1 1", "ALPHA", f"{option} {len(intervals(case))} 1",
            "1234 32.1"]
    for i, (upper, coefficients, powers, latent) in enumerate(intervals(case)):
        if i:
            rows.append(str(latent))
        rows.append(" ".join(map(str, (upper, *coefficients))))
        if extra:
            rows.append(" ".join(map(str, (len(powers), *(x for pair in powers for x in pair)))))
    if magnetic:
        rows.extend((f"{magnetic[0]} {magnetic[1]}", f"{magnetic[2]} {magnetic[3]}"))
    return "\n".join(rows)+"\n"


def heat_capacity(temperature, coefficients, powers):
    result = sum(c*temperature**p for c, p in zip(coefficients, (0, 1, 2, -2), strict=True))
    return result+sum(c*(np.log(temperature) if p == 99 else temperature**p) for c, p in powers)


def magnetic_response(temperature, magnetic):
    if magnetic is None:
        return 0., 0., 0., 0.
    with mp.workdps(60):
        tc, moment, original_afm, structure = (mp.mpf(str(x)) for x in magnetic)
        afm = -1/original_afm
        critical = tc if tc > 0 else tc/afm
        beta = moment if moment > 0 else moment/afm
        a = mp.mpf(518)/1125+mp.mpf(11692)/15975*(1/structure-1)

        def energy(t):
            tau = t/critical
            if tau < 1:
                shape = 1-(mp.mpf(79)/(140*structure)/tau
                           +mp.mpf(474)/497*(1/structure-1)
                           *(tau**3/6+tau**9/135+tau**15/600))/a
            else:
                shape = -(tau**-5/10+tau**-15/315+tau**-25/1500)/a
            return mp.mpf("8.3145")*t*mp.log(1+beta)*shape

        t = mp.mpf(str(temperature))
        g, slope, second = energy(t), mp.diff(energy, t), mp.diff(energy, t, 2)
        return tuple(map(float, (g, g-t*slope, -slope, -t*second)))


def reference(case, temperature):
    enthalpy, entropy, lower = 1234., 32.1, 298.15
    selected_cp = None
    for i, (upper, coefficients, powers, latent) in enumerate(intervals(case)):
        if i:
            enthalpy += latent
            entropy += latent/lower
        stop = min(temperature, upper)
        enthalpy += quad(lambda t, c=coefficients, p=powers: heat_capacity(t, c, p), lower, stop,
                         epsabs=1e-9, epsrel=5e-13)[0]
        entropy += quad(lambda t, c=coefficients, p=powers: heat_capacity(t, c, p)/t, lower, stop,
                        epsabs=1e-9, epsrel=5e-13)[0]
        if temperature < upper or i == len(intervals(case))-1:
            selected_cp = heat_capacity(temperature, coefficients, powers)
            break
        lower = upper
    g = enthalpy-temperature*entropy
    mag = magnetic_response(temperature, case[3])
    return dict(zip(("G", "H", "S", "Cp"),
                    (float(x+y) for x, y in zip((g, enthalpy, entropy, selected_cp), mag,
                                               strict=True)), strict=True))


def native_model(case, output):
    text = original_text(case)
    (output/(case[0]+".dat")).write_text(text, encoding="utf-8")
    source = InputSource(DataSourceProvenance("synthetic-chemsage-cp-"+case[0], 0,
        "Fixed synthetic mathematical qualification; not an assessed material", "1", "CC0"),
        "synthetic")
    db = ChemSageDatabase.from_string(text, source=source)
    model = CefPhaseModel(db, "ALPHA")
    cloned = CefPhaseModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert cloned.to_dict() == model.to_dict()
    assert cloned.database.text == text
    return model


def caloric(case, model):
    unit = {"case": case[0], "passed": False, "points": [], "cuts": [], "failures": []}
    try:
        lower = 300.
        for upper, _, _, _ in intervals(case):
            for fraction in (.2, .5, .8):
                t = lower+(upper-lower)*fraction
                point = {"T": t, "reference": reference(case, t)}
                unit["points"].append(point)
                p = model.evaluate([{"FE": 1.}], t)
                point["native"] = {"G": p.gibbs_J_mol, "H": p.enthalpy_J_mol,
                    "S": p.entropy_J_mol_K, "Cp": model.temperature_response(
                        [{"FE": 1.}], t).fixed_constitution_heat_capacity_J_mol_K}
                point["errors"] = {k: abs(v-point["reference"][k])
                                   for k, v in point["native"].items()}
                point["passed"] = all(e <= (1e-7 if k in {"G", "H"} else 1e-8)
                    +1e-10*abs(point["reference"][k]) for k, e in point["errors"].items())
            lower = upper
        for upper, _, _, _ in intervals(case)[:-1]:
            left, right = reference(case, upper-1e-8), reference(case, upper)
            p = model.evaluate([{"FE": 1.}], upper)
            native_right = {"G": p.gibbs_J_mol, "H": p.enthalpy_J_mol,
                            "S": p.entropy_J_mol_K}
            rejected = False
            try:
                model.temperature_response([{"FE": 1.}], upper)
            except ValueError:
                rejected = True
            assert rejected and abs(right["G"]-left["G"]) < 1e-5
            assert all(abs(native_right[k]-right[k]) < 1e-7 for k in native_right)
            unit["cuts"].append({"T": upper, "left": left, "right": right,
                                 "native_right": native_right, "derivative_rejected": rejected})
        unit["passed"] = all(p["passed"] for p in unit["points"])
    except Exception:
        unit["failures"].append(traceback.format_exc())
    return unit


def material(case, model):
    unit = {"case": case[0], "passed": False, "endpoints": [], "failures": []}
    try:
        material_spec = CefMaterialSpec(model, ({"FE": 1.},), 540, 660, 600, .01)
        stock = {"FE": .001}
        spec = calphad_world_spec(material_spec, element_amounts_mol=stock, temperature_K=600)
        runtime = ChemWorldRuntime(composition=spec)
        runtime.reset(seed=40)
        split = ChemWorldRuntime.from_snapshot(runtime.snapshot())
        initial_h = solid_enthalpy(runtime._state, material_spec)
        ref_temperature = 600.
        for target in (640., 560.):
            def rhs(t, y, target=target):
                cp = reference(case, float(y[0]))["Cp"]
                if cp <= 0:
                    raise ValueError("nonpositive independent heat capacity")
                return [float(np.clip(.05*(target-y[0]), -2, 2))/(.001*cp+.2)]

            solution = solve_ivp(rhs, (0, 2), [ref_temperature], method="DOP853",
                rtol=1e-10, atol=1e-11, max_step=.05)
            assert solution.success
            ref_temperature = float(solution.y[0, -1])
            action = {"operation": "heat", "target_temperature_K": target, "duration_s": 2}
            result = runtime.execute(action)
            first = split.execute(dict(action, duration_s=1))
            second = split.execute({"operation": "wait", "duration_s": 1})
            h = solid_enthalpy(runtime._state, material_spec)
            point = {"target": target, "reference_T": ref_temperature,
                "native_T": runtime._state.temperature_K, "split_T": split._state.temperature_K,
                "H": h, "split_H": solid_enthalpy(split._state, material_spec),
                "energy_residual": h-initial_h-runtime._state.ledger.energy_jacket_J
                    +runtime._state.ledger.heat_loss_J, "diagnostics": result.info}
            unit["endpoints"].append(point)
            assert all(x.info["transaction_status"] == "committed"
                       for x in (result, first, second))
            assert runtime._state.species_amounts == split._state.species_amounts == stock
            assert abs(point["native_T"]-ref_temperature) <= 2e-4
            assert abs(point["native_T"]-point["split_T"]) <= 1e-4
            assert abs(point["H"]-point["split_H"]) <= 1e-5
            assert abs(point["energy_residual"]) <= 1e-7
        unit["passed"] = True
    except Exception:
        unit["failures"].append(traceback.format_exc())
    return unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("evidence must stay outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"expected_inputs": 10, "expected_points": 36, "expected_endpoints": 20,
              "caloric": [], "materials": [], "preparation_failures": []}
    started = time.perf_counter()
    for i, case in enumerate(CASES):
        try:
            model = native_model(case, args.output)
            report["caloric"].append(caloric(case, model))
            report["materials"].append(material(case, model))
        except Exception:
            report["preparation_failures"].append({"case": case[0],
                                                      "error": traceback.format_exc()})
        elapsed = max(1e-9, time.perf_counter()-started)
        report["elapsed_s"] = elapsed
        report["passed_caloric_inputs"] = sum(u["passed"] for u in report["caloric"])
        report["passed_materials"] = sum(u["passed"] for u in report["materials"])
        report["passed_points"] = sum(p["passed"] for u in report["caloric"] for p in u["points"])
        report["executed_points"] = sum(len(u["points"]) for u in report["caloric"])
        report["executed_endpoints"] = sum(len(u["endpoints"]) for u in report["materials"])
        report["complete_and_passed"] = (report["passed_points"] == 36
            and report["passed_caloric_inputs"] == report["passed_materials"] == 10)
        write(args.output/"summary.json", report)
        print(f"Cp inputs {i+1}/10 rate={(i+1)/elapsed:.3f}/s "
              f"eta={(9-i)*elapsed/(i+1):.1f}s {case[0]} "
              f"caloric={report['passed_caloric_inputs']} materials={report['passed_materials']}",
              flush=True)
    return 0 if report["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
