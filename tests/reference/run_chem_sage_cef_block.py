"""Fixed 93-phase full-DAT CEF/QKTO response qualification on independent models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

FILES = ("2026-Dixon-Na-K-Cl-I.dat", "Kaye_Pd-Ru-Tc-Mo.dat", "KF-NIF2_switched.dat",
         "MQMQA-tern-tests.dat", "Ocadiz-Flores.dat", "Shishin_Fe-Sb-O-S_slag.dat", "Viitala.dat")
PRESSURE = 201325.
PHASE_COUNT, POINT_COUNT = 93, 558
FAMILIES = {"PURE", "IDMX", "QKTO", "SUBL", "SUBLM", "RKMP", "RKMPM"}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True), encoding="utf-8")


def auxiliary_expressions(raw, header, phase, array):
    """Independent reference from original, parsed endmember stoichiometry."""
    import pycalphad.variables as v
    import symengine as se

    expressions = {}
    for index, element in enumerate(header.pure_elements):
        if not (element.startswith("E(") or element in {"EA", "E", "E-", "/-"}):
            continue
        terms = []
        for j, member in enumerate(raw.endmembers):
            coefficient = member.stoichiometry_pure_elements[index]
            if coefficient == 0:
                continue
            if raw.phase_type is None:
                constitution = [[s] for subl in array for s in subl]
            elif getattr(raw, "endmember_constituent_idxs", None):
                constitution = [[raw.constituent_array[i][k-1]]
                                for i, k in enumerate(raw.endmember_constituent_idxs[j])]
            else:
                constitution = [[member.species_name]]
            weight = se.RealDouble(coefficient)
            for i, subl in enumerate(constitution):
                if subl[0] not in array[i]:
                    weight = se.Integer(0)
                    break
                weight *= v.SiteFraction(phase, i, subl[0])
            terms.append(weight)
        if terms:
            expressions["auxiliary:"+element] = sum(terms)
    return expressions


def feasible_profile(array, constraints, phase, tilted):
    import numpy as np
    import pycalphad.variables as v
    from scipy.linalg import qr
    from scipy.optimize import minimize

    labels = [(i, s) for i, subl in enumerate(array) for s in subl]
    variables = [v.SiteFraction(phase, i, s) for i, s in labels]
    target = []
    for i, subl in enumerate(array):
        weights = np.array([1+((i+j) % len(subl)) if tilted else 1
                            for j in range(len(subl))], dtype=float)
        target.extend(weights/weights.sum())
    target = np.asarray(target)
    derivative = [[c.diff(x) for x in variables] for c in constraints]

    def substitute(y):
        return dict(zip(variables, map(float, y), strict=True))

    def values(y):
        substitutions = substitute(y)
        return np.array([float(c.subs(substitutions)) for c in constraints])

    def jacobian(y):
        substitutions = substitute(y)
        return np.array([[float(d.subs(substitutions)) for d in row] for row in derivative])

    result = target
    if np.max(np.abs(values(target)), initial=0) > 1e-12:
        jac = jacobian(target)
        scale = np.maximum(np.max(np.abs(jac), axis=1), 1.)
        _, r, pivots = qr((jac/scale[:, None]).T, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(r))
        rank = int(np.sum(diagonal > 1e-10*max(float(diagonal.max(initial=0)), 1.)))
        indices = np.sort(pivots[:rank])
        answer = minimize(lambda y: float(np.sum((y-target)**2)), target,
                          jac=lambda y: 2*(y-target), method="SLSQP",
                          bounds=[(1e-8, 1)]*len(target),
                          constraints={"type": "eq", "fun": lambda y: values(y)[indices],
                                       "jac": lambda y: jacobian(y)[indices]},
                          options={"maxiter": 200, "ftol": 1e-13})
        if not answer.success:
            raise ValueError("independent feasible-profile projection failed: "+answer.message)
        result = answer.x
    residual = float(np.max(np.abs(values(result)), initial=0))
    if residual > 1e-9 or min(result) < 1e-8:
        raise ValueError("independent profile does not satisfy all original constraints")
    fractions = [{} for _ in array]
    for (i, species), fraction in zip(labels, result, strict=True):
        fractions[i][species] = float(fraction)
    return fractions, residual


def reference(directory, output):
    import pycalphad
    import pycalphad.variables as v
    import symengine as se
    from pycalphad.io.cs_dat import parse_cs_dat

    if pycalphad.__version__ != "0.11.2":
        raise ValueError("reference must be pycalphad 0.11.2")
    report = {"expected_phases": PHASE_COUNT, "expected_points": POINT_COUNT, "units": [],
              "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                          for d in importlib.metadata.distributions()))}
    started = time.perf_counter()
    for name in FILES:
        path = directory/name
        header, solutions, stoichiometric = parse_cs_dat(
            path.read_text(encoding="utf-8-sig").upper())
        database = pycalphad.Database(str(path))
        components = sorted(database.elements)
        if any(s.name == "VA" for s in database.species) and "VA" not in components:
            components.append("VA")
        originals = (*solutions, *stoichiometric)
        counts, seen = Counter(p.phase_name for p in originals), Counter()
        for raw in originals:
            source_name = raw.phase_name
            seen[source_name] += 1
            if (raw.phase_type or "PURE") not in FAMILIES:
                continue
            phase = (source_name if counts[source_name] == 1 else
                     f"{source_name}#OCCURRENCE{seen[source_name]}")
            unit = {"database": name, "phase": phase, "source_label": source_name,
                    "points": [], "failures": [],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report["units"].append(unit)
            try:
                selected_db = database
                if phase != source_name:
                    # The upstream reader drops repeated labels. Add this
                    # original occurrence under its unambiguous ID, changing
                    # no coefficient, lattice or other declared parameter.
                    selected_db = deepcopy(database)
                    occurrence = deepcopy(raw)
                    occurrence.phase_name = phase
                    occurrence.insert(selected_db, header.pure_elements,
                                      header.gibbs_coefficient_idxs,
                                      header.excess_coefficient_idxs)
                    selected_db.process_parameter_queue()
                if raw.phase_type is None and hasattr(raw.endmembers[0], "curie_temperature"):
                    # 0.11.2 inserts PURE G on its elemental lattice, but TC/
                    # BMAGN on the source phase label. Those unmatched records
                    # silently disappear from Model. Repair only the reference
                    # binding, using original source values and actual lattice.
                    from tinydb import where

                    selected_db = deepcopy(selected_db)
                    selection = ((where("phase_name") == phase)
                                 & where("parameter_type").one_of(("TC", "BMAGN")))
                    old = selected_db._parameters.search(selection)
                    unit["reference_magnetic_binding_repair"] = {
                        "original_arrays": [str(p["constituent_array"]) for p in old],
                        "target_array": [[s.name for s in sorted(subl)]
                                         for subl in selected_db.phases[phase].constituents],
                        "reason": "0.11.2 PURE endmember magnetic records use phase label",
                    }
                    selected_db._parameters.remove(selection)
                    for kind, value in (("TC", raw.endmembers[0].curie_temperature),
                                        ("BMAGN", raw.endmembers[0].magnetic_moment)):
                        selected_db.add_parameter(kind, phase,
                            unit["reference_magnetic_binding_repair"]["target_array"], 0,
                            value, force_insert=False)
                    selected_db.process_parameter_queue()
                model = pycalphad.Model(selected_db, components, phase)
                array = [sorted(s.name for s in subl) for subl in model.constituents]
                unit["constituents"] = array
                constraints = model.get_internal_constraints()
                rows = {"G": model.GM,
                        **{f"element:{e}": model.moles(e) for e in model.nonvacant_elements},
                        **{f"site_sum:{i}": constraints[i] for i in range(len(array))},
                        "charge": constraints[len(array)] if len(constraints) > len(array)
                        else se.Integer(0)}
                auxiliary = auxiliary_expressions(raw, header, phase, array)
                rows.update(auxiliary)
                all_constraints = [*constraints, *auxiliary.values()]
                axes = [("temperature", v.T), ("pressure", v.P),
                        *((f"site:{i}:{s}", v.SiteFraction(phase, i, s))
                          for i, subl in enumerate(array) for s in subl)]
                expressions = {"H": model.HM, "S": model.SM, "Cp_fixed": model.CPM}
                for label, expression in rows.items():
                    prefix = "response:"+label
                    expressions[prefix] = expression
                    for axis, variable in axes:
                        first = expression.diff(variable)
                        expressions[prefix+"|"+axis] = first
                        for other_axis, other_variable in axes:
                            expressions[prefix+"|"+axis+"|"+other_axis] = first.diff(other_variable)
                upper_limits = []
                for member in raw.endmembers:
                    previous = .01
                    for interval in member.intervals:
                        upper = 10000. if previous == interval.T_max else interval.T_max
                        previous = interval.T_max
                    upper_limits.append(upper)
                low, high = 300., min(2000., *upper_limits)
                if low >= high:
                    raise ValueError("phase has no required common temperature interval")
                temperatures = [low+(high-low)*fraction for fraction in (.2, .5, .8)]
                unit["temperatures"] = temperatures
                for tilted in (False, True):
                    y, residual = feasible_profile(array, all_constraints, phase, tilted)
                    for temperature in temperatures:
                        substitutions = {v.T: temperature, v.P: PRESSURE, v.N: 1.}
                        substitutions.update({v.SiteFraction(phase, i, species): value
                                              for i, subl in enumerate(y)
                                              for species, value in subl.items()})
                        point = {"T": temperature, "profile": "tilted" if tilted else "uniform",
                                 "sites": y, "profile_constraint_residual": residual,
                                 "values": {}, "failures": []}
                        unit["points"].append(point)
                        for key, expression in expressions.items():
                            value = float(expression.subs(substitutions))
                            if not math.isfinite(value):
                                raise ValueError("nonfinite independent response "+key)
                            point["values"][key] = value
            except Exception:
                unit["failures"].append(traceback.format_exc())
            write(output, report)
            completed = len(report["units"])
            elapsed = max(time.perf_counter()-started, 1e-9)
            print(f"reference phases {completed}/{PHASE_COUNT} rate={completed/elapsed:.2f}/s "
                  f"eta={(PHASE_COUNT-completed)*elapsed/completed:.1f}s {name} {phase}",
                  flush=True)
    return report


def native_values(model, sites, temperature):
    properties = model.evaluate(sites, temperature, PRESSURE)
    response = model.derivatives(sites, temperature, PRESSURE, allow_one_sided=True)
    thermal = model.temperature_response(sites, temperature, PRESSURE, allow_one_sided=True)
    values = {"H": properties.enthalpy_J_mol, "S": properties.entropy_J_mol_K,
              "Cp_fixed": thermal.fixed_constitution_heat_capacity_J_mol_K}
    for i, name in enumerate(response.output_names):
        prefix = "response:"+name
        values[prefix] = float(response.values[i])
        for j, axis in enumerate(response.coordinate_names):
            values[prefix+"|"+axis] = float(response.jacobian[i, j])
            for k, other in enumerate(response.coordinate_names):
                values[prefix+"|"+axis+"|"+other] = float(response.hessian[i, j, k])
    return values, list(response.diagnostics)


def tolerance(key, expected):
    parts = key.split("|")
    axes = parts[1:]
    if key == "H":
        absolute = 1e-5
    elif key in {"S", "Cp_fixed"}:
        absolute = 1e-7
    elif parts[0] == "response:G":
        if axes.count("pressure") == 2:
            absolute = 1e-18
        elif "pressure" in axes and "temperature" in axes:
            absolute = 1e-14
        elif "pressure" in axes:
            absolute = 1e-12
        elif axes == ["temperature", "temperature"]:
            absolute = 1e-10
        elif axes == ["temperature"]:
            absolute = 1e-7
        else:
            absolute = 1e-5
    else:
        absolute = (1e-10, 1e-9, 1e-8)[len(axes)]
    relative = 1e-9 if not axes and parts[0] != "response:G" and key not in {
        "H", "S", "Cp_fixed"} else 1e-8
    return absolute+relative*abs(expected)


def native_model(database, phase, components):
    from chemworld.physchem.calphad import CefPhaseModel

    return CefPhaseModel(database, phase, components=components, activate_suppressed=True)


def native_source(name):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(DataSourceProvenance("public-chemsage-"+name, 0,
        "pycalphad public DAT "+name, "0.11.2", "database terms not established"), "literature")


def native(directory, reference_report, output):
    from chemworld.data.chem_sage import ChemSageDatabase

    report = {"expected_phases": PHASE_COUNT, "expected_points": POINT_COUNT,
              "units": [], "measurements": {}}
    databases = {}
    database_reports = []
    for name in FILES:
        source = native_source(name)
        try:
            database = ChemSageDatabase.from_file(directory/name, source=source)
            clone = ChemSageDatabase.from_dict(json.loads(json.dumps(database.to_dict())))
            assert clone.to_dict() == database.to_dict()
            databases[name] = database
            database_reports.append({"database": name, "constructed": True})
        except Exception:
            databases[name] = traceback.format_exc()
            database_reports.append({"database": name, "constructed": False,
                                     "failure": databases[name]})
        write(output.parent/"databases.json", database_reports)
    started = time.perf_counter()
    for reference_unit in reference_report["units"]:
        name, phase = reference_unit["database"], reference_unit["phase"]
        unit = {"database": name, "phase": phase, "points": [], "failures": [], "passed": False}
        report["units"].append(unit)
        try:
            db = databases[name]
            assert not isinstance(db, str), db
            assert not reference_unit["failures"], reference_unit["failures"]
            components = tuple(db.elements)+(("VA",) if "VA" in db.species else ())
            model = native_model(db, phase, components)
            unit["activate_suppressed"] = True
            clone = type(model).from_dict(json.loads(json.dumps(model.to_dict())))
            assert clone.to_dict() == model.to_dict()
            assert [sorted(subl) for subl in model.constituents] == reference_unit["constituents"]
            assert len(reference_unit["points"]) == 6
            for reference_point in reference_unit["points"]:
                point = {"T": reference_point["T"], "profile": reference_point["profile"],
                         "passed": False, "values": {}, "failures": []}
                unit["points"].append(point)
                try:
                    values, diagnostics = native_values(model, reference_point["sites"], point["T"])
                    point.update(values=values, diagnostics=diagnostics)
                    assert values.keys() == reference_point["values"].keys(), (
                        "response label mismatch")
                    for key, expected in reference_point["values"].items():
                        error = abs(values[key]-expected)
                        metric = key.split("|")[0]
                        row = report["measurements"].setdefault(metric, {"count": 0, "max_abs": 0.})
                        row["count"] += 1
                        row["max_abs"] = max(row["max_abs"], error)
                        if error > tolerance(key, expected):
                            point["failures"].append({"key": key, "native": values[key],
                                "reference": expected, "error": error,
                                "tolerance": tolerance(key, expected)})
                    point["passed"] = not point["failures"]
                except Exception:
                    point["failures"].append(traceback.format_exc())
            unit["passed"] = len(unit["points"]) == 6 and all(p["passed"] for p in unit["points"])
        except Exception:
            unit["failures"].append(traceback.format_exc())
        write(output, report)
        count = len(report["units"])
        elapsed = max(time.perf_counter()-started, 1e-9)
        print(f"native phases {count}/{PHASE_COUNT} rate={count/elapsed:.2f}/s "
              f"eta={(PHASE_COUNT-count)*elapsed/count:.1f}s passed={unit['passed']}", flush=True)
    report["passed_phases"] = sum(u["passed"] for u in report["units"])
    report["passed_points"] = sum(p["passed"] for u in report["units"] for p in u["points"])
    report["failed_points"] = sum(not p["passed"] for u in report["units"] for p in u["points"])
    report["unexecuted_points"] = POINT_COUNT-report["passed_points"]-report["failed_points"]
    report["complete_and_passed"] = report["passed_phases"] == PHASE_COUNT
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--stage", choices=("reference", "native"))
    args = parser.parse_args()
    if args.stage == "reference":
        reference(args.source_directory, args.output/"reference.json")
        return 0
    if args.stage == "native":
        native(args.source_directory, json.loads((args.output/"reference.json").read_text(
            encoding="utf-8")), args.output/"native.json")
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("ChemSage evidence must stay outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    import sys

    for stage, interpreter, isolated in (("reference", args.reference_python, True),
                                         ("native", sys.executable, False)):
        command = [str(interpreter), *(["-I"] if isolated else []), str(Path(__file__).resolve()),
                   "--stage", stage, "--source-directory", str(args.source_directory),
                   "--output", str(args.output)]
        log_path = args.output/(stage+".log")
        with (log_path.open("x", encoding="utf-8") as log,
              subprocess.Popen(command, stdout=log, stderr=log, env=environment) as process):
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    print(f"{stage} alive pid={process.pid} bytes={log_path.stat().st_size} "
                          f"last={lines[-1] if lines else 'startup'}", flush=True)
        print(f"{stage} exit={process.returncode}", flush=True)
        if process.returncode:
            partial = args.output/"native.json"
            saved = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
            passed = sum(p["passed"] for u in saved.get("units", ()) for p in u["points"])
            failed = sum(not p["passed"] for u in saved.get("units", ()) for p in u["points"])
            failure = {"complete_and_passed": False, "failed_stage": stage,
                       "exit": process.returncode, "expected_phases": PHASE_COUNT,
                       "expected_points": POINT_COUNT,
                       "passed_points": passed, "failed_points": failed,
                       "unexecuted_points": POINT_COUNT-passed-failed,
                       "native_report": str(partial), "stage_log": str(log_path)}
            write(args.output/"stage_failure.json", failure)
            write(args.output/"summary.json", failure)
            return 1
    result = json.loads((args.output/"native.json").read_text(encoding="utf-8"))
    summary = {k: v for k, v in result.items() if k != "units"}
    summary["native_report"] = str(args.output/"native.json")
    summary["failed_units"] = [{"database": u["database"], "phase": u["phase"],
                                "failures": u["failures"]}
                               for u in result["units"] if not u["passed"]]
    write(args.output/"summary.json", summary)
    print(json.dumps({k: v for k, v in result.items() if k not in {"units", "measurements"}}),
          flush=True)
    return 0 if result["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
