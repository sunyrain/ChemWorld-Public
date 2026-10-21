"""Seven fixed full-TDB native material histories and isolated thermal reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

CASES = (
    {"name": "AlMg-fcc", "database": "Al-Mg_Zhong.tdb", "phase": "FCC_A1",
     "x": {"AL": .3, "MG": .7}, "sites": [{"AL": .5, "MG": .5}, {"VA": 1}]},
    {"name": "AlMg-liquid", "database": "Al-Mg_Zhong.tdb", "phase": "LIQUID",
     "x": {"AL": .4, "MG": .6}, "sites": [{"AL": .5, "MG": .5}], "type": "liquid"},
    {"name": "CrFe-bcc", "database": "crfe_bcc_magnetic.tdb", "phase": "BCC_A2",
     "x": {"CR": .4, "FE": .6}, "sites": [{"CR": .5, "FE": .5}, {"VA": 1}]},
    {"name": "AlNi-ordered", "database": "alni_dupin_2001.tdb", "phase": "FCC_L12",
     "x": {"AL": .25, "NI": .75},
     "sites": [{"AL": .15, "NI": .85}, {"AL": .55, "NI": .45}, {"VA": 1}]},
    {"name": "Al2Fe-line", "database": "Al-Fe_sundman2009.tdb", "phase": "AL2FE",
     "x": {"AL": 2/3, "FE": 1/3}, "sites": [{"AL": 1}, {"FE": 1}], "underdetermined": True},
    {"name": "AlCuY-liquid", "database": "Al-Cu-Y.tdb", "phase": "LIQUID",
     "x": {"AL": .3, "CU": .4, "Y": .3},
     "sites": [{"AL": 1/3, "CU": 1/3, "Y": 1/3}], "type": "liquid"},
    {"name": "CuO-ionic", "database": "cuo.tdb", "phase": "IONIC_LIQ",
     "x": {"CU": .6, "O": .4},
     "sites": [{"CU+1": .3, "CU+2": .4, "CU+3": .3}, {"O-2": .5, "VA": .5}],
     "type": "liquid"},
)
SETPOINTS = (1130., 1080.)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def equilibrium_function(case, directory):
    import numpy as np
    import pycalphad
    import pycalphad.variables as v

    if pycalphad.__version__ != "0.11.2":
        raise ValueError("reference must be pycalphad 0.11.2")
    db = pycalphad.Database(str(directory/case["database"]))
    components = sorted(case["x"])
    if "VA" in db.elements:
        components.append("VA")

    def evaluate(temperature):
        conditions = {v.T: temperature, v.P: 101325, v.N: 1}
        conditions.update({v.X(e): case["x"][e] for e in sorted(case["x"])[:-1]})
        result = pycalphad.equilibrium(db, components, [case["phase"]], conditions,
                                      output=["HM"], calc_opts={"pdens": 500})
        phases = np.asarray(result.Phase).reshape(-1)
        amounts = np.asarray(result.NP).reshape(-1)
        active = np.flatnonzero((phases == case["phase"]) & (amounts > 1e-8))
        if len(active) != 1 or abs(float(amounts[active[0]])-1) > 1e-8:
            raise ValueError("reference must have one selected phase with fraction one")
        index = active[0]
        answer = {"T": float(temperature), "G": float(np.asarray(result.GM).reshape(-1)[0]),
                  "H": float(np.asarray(result.HM).reshape(-1)[0]),
                  "x": {e: float(np.asarray(result.X.sel(component=e)).reshape(-1)[index])
                        for e in case["x"]},
                  "mu": {e: float(np.asarray(result.MU.sel(component=e)).reshape(-1)[0])
                         for e in case["x"]}}
        if not all(np.isfinite(value) for value in (answer["G"], answer["H"],
                                                   *answer["x"].values(), *answer["mu"].values())):
            raise ValueError("nonfinite independent material properties")
        return answer

    return evaluate


def curves(directory, output):
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.interpolate import PchipInterpolator

    report = {"expected_units": 7, "expected_curve_points": 455, "units": [],
              "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                          for d in importlib.metadata.distributions()))}
    started = time.monotonic()
    for case in CASES:
        unit = {"case": case, "points": [], "failures": []}
        report["units"].append(unit)
        try:
            path = directory/case["database"]
            unit["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            evaluate = equilibrium_function(case, directory)
            for temperature in np.linspace(1050, 1150, 65):
                unit["points"].append(evaluate(float(temperature)))
                write(output, report)
                completed = sum(len(u["points"]) for u in report["units"])
                elapsed = time.monotonic()-started
                print(f"reference curves {completed}/455 rate={completed/elapsed:.2f}/s "
                      f"eta={(455-completed)*elapsed/completed:.1f}s {case['name']}", flush=True)

            def integrate(stride, unit=unit):
                points = unit["points"][::stride]
                temperatures = np.array([p["T"] for p in points])
                enthalpies = np.array([.001*p["H"]+.2*(p["T"]-1100) for p in points])
                if np.any(np.diff(enthalpies) <= 0):
                    raise ValueError("nonmonotone independent caloric branch")
                interpolator = PchipInterpolator(temperatures, enthalpies)
                derivative = interpolator.derivative()
                current, endpoints = 1100., []
                for setpoint in SETPOINTS:
                    def rhs(t, y, target=setpoint):
                        cp = float(derivative(y[0]))
                        if cp <= 0 or not 1050 <= y[0] <= 1150:
                            raise ValueError("reference left its positive caloric branch")
                        return [float(np.clip(.05*(target-y[0]), -2, 2))/cp]

                    solution = solve_ivp(rhs, (0, 3), [current], method="DOP853", rtol=1e-10,
                                         atol=1e-11, max_step=.05)
                    if not solution.success:
                        raise ValueError(solution.message)
                    current = float(solution.y[0, -1])
                    endpoints.append(current)
                return endpoints

            unit["fine_temperatures"] = integrate(1)
            unit["coarse_temperatures"] = integrate(2)
            unit["convergence_error_K"] = max(abs(a-b) for a, b in zip(
                unit["fine_temperatures"], unit["coarse_temperatures"], strict=True))
            if unit["convergence_error_K"] > 1e-4:
                raise ValueError("reference 65/33 node temperature convergence failed")
        except Exception:
            unit["failures"].append(traceback.format_exc())
        write(output, report)


def native(directory, output):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.data.tdb import TdbDatabase
    from chemworld.physchem.calphad import CefPhaseModel
    from chemworld.physchem.calphad_material import CefMaterialSpec
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.solid_solutions import solid_enthalpy, solid_model
    from chemworld.world.solid_world import calphad_world_spec

    report = {"expected_units": 7, "expected_endpoints": 14, "units": []}
    for index, case in enumerate(CASES):
        unit = {"case": case, "endpoints": [], "failures": []}
        report["units"].append(unit)
        write(output, report)
        print(f"native materials {index}/7: starting {case['name']}", flush=True)
        try:
            source = InputSource(DataSourceProvenance(
                "public-tdb-"+case["database"], 0, "pycalphad public TDB "+case["database"],
                "0.11.2", "database terms not established"), "literature")
            database = TdbDatabase.from_file(directory/case["database"], source=source)
            components = (*case["x"], *(("VA",) if "VA" in database.elements else ()))
            model = CefMaterialSpec(CefPhaseModel(
                database, case["phase"], components=components,
                extrapolation="extend", unspecified_endmembers="zero",
            ), tuple(case["sites"]), 1050, 1150, 1100, .01, phase_type=case.get("type", "solid"))
            stock = {e: .001*x for e, x in case["x"].items()}
            spec = calphad_world_spec(model, element_amounts_mol=stock, temperature_K=1100)
            runtime = ChemWorldRuntime(composition=spec)
            runtime.reset(seed=40)
            split = ChemWorldRuntime.from_snapshot(runtime.snapshot())
            initial_H = solid_enthalpy(runtime._state, model)
            for leg, setpoint in enumerate(SETPOINTS):
                action = {"operation": "heat", "target_temperature_K": setpoint, "duration_s": 3}
                result = runtime.execute(action)
                if result.info["transaction_status"] != "committed":
                    raise ValueError(str(result.info))
                for operation in (dict(action, duration_s=1.5),
                                  {"operation": "wait", "duration_s": 1.5}):
                    step = split.execute(operation)
                    if step.info["transaction_status"] != "committed":
                        raise ValueError(str(step.info))
                state = runtime._state
                selected = solid_model(state, spec)
                current = selected.at_tp(state.species_amounts, state.temperature_K)
                constitution = current.constitution
                properties = constitution.properties
                potentials = constitution.chemical_potentials
                h = solid_enthalpy(state, selected)
                endpoint = {"leg": leg, "T": state.temperature_K, "total_H_J": h,
                            "split_T": split._state.temperature_K,
                            "split_total_H_J": solid_enthalpy(split._state, model),
                            "energy_residual_J": h-initial_H-state.ledger.energy_jacket_J
                            + state.ledger.heat_loss_J, "amounts": state.species_amounts,
                            "G": properties.gibbs_J_mol, "H": properties.enthalpy_J_mol,
                            "x": dict(properties.element_fractions),
                            "sites": [dict(s) for s in constitution.site_fractions],
                            "mu_status": potentials.status,
                            "mu": None if potentials.values_J_mol is None
                            else dict(potentials.values_J_mol),
                            "stationarity": constitution.diagnostics,
                            "operation_diagnostics": result.info}
                unit["endpoints"].append(endpoint)
                write(output, report)
                assert state.species_amounts == stock, "element inventories changed"
                assert abs(endpoint["energy_residual_J"]) <= 1e-7, "physical heat balance"
                assert abs(endpoint["T"]-endpoint["split_T"]) <= 1e-4, "split T convergence"
                assert abs(h-endpoint["split_total_H_J"]) <= 1e-5, "split H convergence"
                print(f"native materials {index}/7 leg={leg+1}/2 {case['name']}", flush=True)
        except Exception:
            unit["failures"].append(traceback.format_exc())
        write(output, report)
        print(f"native materials {index+1}/7 failed={bool(unit['failures'])}", flush=True)


def endpoints(directory, output):
    values = json.loads((output/"native.json").read_text(encoding="utf-8"))
    report = {"units": []}
    for unit in values["units"]:
        case = unit["case"]
        result = {"case": case, "endpoints": [], "failures": []}
        report["units"].append(result)
        try:
            evaluate = equilibrium_function(case, directory)
            for endpoint in unit["endpoints"]:
                result["endpoints"].append(evaluate(endpoint["T"]))
        except Exception:
            result["failures"].append(traceback.format_exc())
        write(output/"endpoints.json", report)
        print(f"reference endpoints {len(report['units'])}/7", flush=True)


def compare(output):
    reports = [json.loads((output/name).read_text(encoding="utf-8")) for name in (
        "curves.json", "native.json", "endpoints.json")]
    summary = {"expected_units": 7, "expected_endpoints": 14, "units": []}
    for curves_unit, native_unit, ref_unit in zip(*(r["units"] for r in reports), strict=True):
        case = native_unit["case"]
        unit = {"name": case["name"], "passed": False, "failures": [], "errors": []}
        summary["units"].append(unit)
        try:
            assert not any(u["failures"] for u in (curves_unit, native_unit, ref_unit)), (
                "one or more preparation/native/reference stages failed")
            assert len(native_unit["endpoints"]) == len(ref_unit["endpoints"]) == 2
            for index, (point, ref) in enumerate(zip(native_unit["endpoints"],
                                                    ref_unit["endpoints"], strict=True)):
                errors = {"leg": index, "T": abs(point["T"]-curves_unit[
                    "fine_temperatures"][index]),
                          "G": abs(point["G"]-ref["G"]), "H": abs(point["H"]-ref["H"]),
                          "x": max(abs(point["x"][e]-x) for e, x in ref["x"].items())}
                unit["errors"].append(errors)
                assert errors["T"] <= 2e-4, "independent thermal trajectory"
                assert errors["x"] <= 1e-8, "independent composition"
                assert all(errors[k] <= 1e-4+1e-8*abs(ref[k]) for k in ("G", "H")), (
                    "independent G/H")
                if case.get("underdetermined"):
                    assert point["mu_status"] == "underdetermined" and point["mu"] is None
                else:
                    assert point["mu_status"] == "determined"
                    errors["mu"] = max(abs(point["mu"][e]-value)
                                       for e, value in ref["mu"].items())
                    assert all(abs(point["mu"][e]-value) <= 1e-4+1e-8*abs(value)
                               for e, value in ref["mu"].items()), "independent mu"
            unit["passed"] = True
        except Exception:
            unit["failures"].append(traceback.format_exc())
    summary["passed_units"] = sum(u["passed"] for u in summary["units"])
    summary["passed_endpoints"] = sum(len(u["errors"]) for u in summary["units"] if u["passed"])
    summary["complete_and_passed"] = summary["passed_units"] == 7
    write(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--stage", choices=("curves", "native", "endpoints"))
    args = parser.parse_args()
    if args.stage:
        if args.stage == "curves":
            curves(args.source_directory, args.output/"curves.json")
        elif args.stage == "native":
            native(args.source_directory, args.output/"native.json")
        else:
            endpoints(args.source_directory, args.output)
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("development evidence must be outside the repository")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    import sys

    for stage, interpreter, isolated in (("curves", args.reference_python, True),
                                         ("native", sys.executable, False),
                                         ("endpoints", args.reference_python, True)):
        command = [str(interpreter), *(["-I"] if isolated else []), str(Path(__file__).resolve()),
                   "--stage", stage, "--source-directory", str(args.source_directory),
                   "--output", str(args.output)]
        log_path = args.output/f"{stage}.log"
        started = time.monotonic()
        with (log_path.open("x", encoding="utf-8") as log,
              subprocess.Popen(command, stdout=log, stderr=log, env=environment) as process):
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    matches = re.findall(r"(?:curves|materials|endpoints) (\d+)/(\d+)",
                                         "\n".join(lines))
                    completed, total = map(int, matches[-1]) if matches else (0, 455 if
                                                                           stage == "curves" else 7)
                    elapsed = time.monotonic()-started
                    rate = completed/elapsed
                    eta = (total-completed)/rate if rate else None
                    print(f"{stage} alive pid={process.pid} "
                          f"completed={completed}/{total} rate={rate:.3f}/s eta_s={eta} "
                          f"log_bytes={log_path.stat().st_size} "
                          f"last={lines[-1] if lines else 'startup'}", flush=True)
        print(f"{stage} exit={process.returncode} log_bytes={log_path.stat().st_size}", flush=True)
        if process.returncode:
            write(args.output/"summary.json", {"complete_and_passed": False,
                                              "failed_stage": stage, "exit": process.returncode})
            return 1
    result = compare(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return 0 if result["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
