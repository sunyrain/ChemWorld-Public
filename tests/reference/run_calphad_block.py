"""Fixed whole-TDB phase-energy comparison against isolated pycalphad 0.11.2."""

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
from pathlib import Path

DATABASES = (
    "Al-Mg_Zhong.tdb", "alni_dupin_2001.tdb", "cuo.tdb", "crfe_bcc_magnetic.tdb",
    "Al-Fe_sundman2009.tdb", "Al-Cu-Y.tdb",
)
TEMPERATURES = (500.0, 1000.0, 1500.0)
PRESSURE = 101325.0
REFERENCE_SCRIPT = Path(__file__).resolve()
SECOND_ORDER = False
ALLOW_ONE_SIDED = False
PHASE_FILTER = None
CONTRIBUTIONS = {"ref": "reference", "idmix": "ideal", "xsmix": "excess",
                 "mag": "magnetic", "2st": "two_state", "ein": "einstein",
                 "vol": "volume", "ord": "ordering"}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def persist(path, report):
    # Only this attempt's already-created report is updated as units complete.
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fractions(constituents, tilted):
    result = []
    for index, subl in enumerate(constituents):
        weights = [1 + ((index + j) % len(subl)) if tilted else 1 for j in range(len(subl))]
        result.append({s: weight / sum(weights) for s, weight in zip(subl, weights, strict=True)})
    return result


def progress(stage, completed, total, started):
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else None
    print(f"{stage}: {completed}/{total}, rate={rate:.3f}/s, ETA_s={eta}", flush=True)


def reference(directory, output):
    import pycalphad
    import pycalphad.variables as v
    from symengine import Integer

    if pycalphad.__version__ != "0.11.2":
        raise ValueError("the reference must be pycalphad 0.11.2")
    report = {"complete": False, "reference": "pycalphad 0.11.2", "units": [],
              "response_order": 2 if SECOND_ORDER else 1,
              "allow_one_sided": ALLOW_ONE_SIDED,
              "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                          for d in importlib.metadata.distributions()))}
    write_new(output, report)
    catalog = []
    for name in DATABASES:
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            db = pycalphad.Database(str(path))
            catalog.extend((name, digest, db, phase) for phase in sorted(db.phases)
                           if PHASE_FILTER is None or (name, phase) in PHASE_FILTER)
        except Exception:
            report["units"].append({"database": name, "sha256": digest,
                                    "phase": None, "failure": traceback.format_exc()})
    report["expected_phases"] = len(catalog) + len(report["units"])
    started = time.monotonic()
    for name, digest, db, phase in catalog:
        unit = {"database": name, "sha256": digest, "phase": phase, "points": []}
        try:
            model = pycalphad.Model(db, sorted(db.elements), phase)
            array = [sorted(s.name for s in subl) for subl in model.constituents]
            unit["constituents"] = array
            energy = model.GM
            expressions = {"G": energy, "H": model.HM, "S": model.SM,
                           "dGdP": energy.diff(v.P)}
            for index, subl in enumerate(array):
                for species in subl:
                    expressions[f"site:{index}:{species}"] = energy.diff(
                        v.SiteFraction(phase, index, species),
                    )
            expressions.update({f"element:{el}": model.moles(el)
                                for el in model.nonvacant_elements})
            expressions.update({f"contribution:{key}": value
                                for key, value in model.models.items()})
            constraints = model.get_internal_constraints()
            if SECOND_ORDER:
                axes = [("temperature", v.T), ("pressure", v.P),
                        *((f"site:{i}:{s}", v.SiteFraction(phase, i, s))
                          for i, subl in enumerate(array) for s in subl)]
                responses = {"G": energy,
                             **{f"element:{el}": model.moles(el)
                                for el in model.nonvacant_elements},
                             **{f"site_sum:{i}": constraints[i] for i in range(len(array))},
                             "charge": constraints[len(array)]
                             if len(constraints) > len(array) else Integer(0)}
                expressions = {"Cp_fixed": -v.T*energy.diff(v.T, v.T)}
                for name, expr in responses.items():
                    prefix = "response:" + name
                    expressions[prefix] = expr
                    for axis, variable in axes:
                        first = expr.diff(variable)
                        expressions[prefix+"|"+axis] = first
                        for other_axis, other_variable in axes:
                            expressions[prefix+"|"+axis+"|"+other_axis] = first.diff(other_variable)
            for tilted in (False, True):
                y = fractions(array, tilted)
                for temperature in TEMPERATURES:
                    substitutions = {v.T: temperature, v.P: PRESSURE, v.N: 1.0}
                    substitutions.update({v.SiteFraction(phase, i, species): value
                                          for i, subl in enumerate(y)
                                          for species, value in subl.items()})
                    point = {"profile": "tilted" if tilted else "uniform", "T": temperature,
                             "site_fractions": y, "values": {}, "failures": {}}
                    for key, expr in expressions.items():
                        try:
                            value = float(expr.subs(substitutions))
                            if not math.isfinite(value):
                                raise ValueError("nonfinite upstream property")
                            point["values"][key] = value
                        except Exception:
                            point["failures"][key] = traceback.format_exc()
                    point["constraints"] = [float(expr.subs(substitutions)) for expr in constraints]
                    unit["points"].append(point)
        except Exception:
            unit["failure"] = traceback.format_exc()
        report["units"].append(unit)
        persist(output, report)
        progress("reference phases", len(report["units"]), report["expected_phases"], started)
    report["complete"] = True
    persist(output, report)


def tolerance(key, expected):
    if SECOND_ORDER:
        if key == "Cp_fixed":
            return 1e-6 + 1e-8*abs(expected)
        if key.startswith("response:G"):
            return 1e-5 + 1e-8*abs(expected)
        return 1e-10 + 1e-9*abs(expected)
    if key.startswith("element:"):
        return 1e-12
    if key == "dGdP":
        return 1e-12 + 1e-9 * abs(expected)
    if key == "S" or key.startswith("site:"):
        return 1e-7 + 1e-9 * abs(expected)
    return 1e-6 + 1e-9 * abs(expected)


def native_second_values(model, y, temperature):
    response = model.derivatives(y, temperature, PRESSURE, allow_one_sided=ALLOW_ONE_SIDED)
    cp = model.temperature_response(y, temperature, PRESSURE, allow_one_sided=ALLOW_ONE_SIDED)
    values = {"Cp_fixed": cp.fixed_constitution_heat_capacity_J_mol_K}
    for row, name in enumerate(response.output_names):
        prefix = "response:" + name
        values[prefix] = float(response.values[row])
        for column, axis in enumerate(response.coordinate_names):
            values[prefix+"|"+axis] = float(response.jacobian[row, column])
            for other_column, other_axis in enumerate(response.coordinate_names):
                values[prefix+"|"+axis+"|"+other_axis] = float(
                    response.hessian[row, column, other_column],
                )
    return values, response.diagnostics


def native(reference_data, directory, output):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.data.tdb import TdbDatabase
    from chemworld.physchem.calphad import CefPhaseModel
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    points_per_phase = 2*len(TEMPERATURES)
    report = {"complete_and_passed": False, "expected_phases": reference_data["expected_phases"],
              "response_order": 2 if SECOND_ORDER else 1,
              "allow_one_sided": ALLOW_ONE_SIDED,
              "expected_points": points_per_phase * reference_data["expected_phases"], "units": []}
    write_new(output, report)
    databases = {}
    started = time.monotonic()
    for ref in reference_data["units"]:
        name, phase = ref["database"], ref["phase"]
        unit = {"database": name, "phase": phase, "passed": False, "points": []}
        try:
            if "failure" in ref:
                raise ValueError("reference model failed: " + ref["failure"])
            if name not in databases:
                path = directory / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                    raise ValueError("TDB changed between reference and native evaluation")
                synthetic = name.startswith("synthetic-")
                source = InputSource(DataSourceProvenance(
                    ("synthetic-cef:" if synthetic else "pycalphad-test-tdb:") + name, 0,
                    ("Explicit formulas: workstreams/world_foundation/"
                     "archive/20261023-completed/convergence/41/NATIVE_CONV41B1_EXPERIMENT.md"
                     if synthetic else
                     "https://github.com/pycalphad/pycalphad/blob/0.11.2/"
                     "pycalphad/tests/databases/" + name),
                    "1" if synthetic else "0.11.2",
                    "CC0-1.0" if synthetic else
                    "Database-specific terms not independently established; external only",
                    ref["sha256"],
                ), "synthetic" if synthetic else "literature")
                databases[name] = TdbDatabase.from_file(path, source=source)
            model = CefPhaseModel(databases[name], phase, extrapolation="extend",
                                  unspecified_endmembers="zero")
            if [list(s) for s in model.constituents] != ref["constituents"]:
                raise ValueError("active model constituents differ from upstream")
            for point in ref["points"]:
                row = {"T": point["T"], "profile": point["profile"], "passed": False,
                       "comparisons": {}, "failures": []}
                try:
                    y, temperature = point["site_fractions"], point["T"]
                    if any(abs(c) > 1e-9 for c in point["constraints"]):
                        try:
                            model.evaluate(y, temperature, PRESSURE)
                        except ValueError as exc:
                            if "charge neutrality" not in str(exc):
                                raise
                            row.update(passed=True, status="invalid_constitution_rejected")
                        else:
                            raise AssertionError("nonneutral constitution was accepted")
                    else:
                        if SECOND_ORDER:
                            values, point_diagnostics = native_second_values(model, y, temperature)
                        else:
                            p = model.evaluate(y, temperature, PRESSURE)
                            values = {"G": p.gibbs_J_mol, "H": p.enthalpy_J_mol,
                                      "S": p.entropy_J_mol_K, "dGdP": p.pressure_derivative_m3_mol}
                            values.update({f"element:{k}": val
                                           for k, val in p.element_fractions.items()})
                            values.update({f"site:{i}:{s}": val for (i, s), val
                                           in model.site_partials(
                                               y, temperature, PRESSURE,
                                           ).items()})
                            values.update({"contribution:" + key: p.contributions_J_mol[native_key]
                                           for key, native_key in CONTRIBUTIONS.items()})
                            point_diagnostics = p.diagnostics
                        row["diagnostics"] = list(point_diagnostics)
                        for key, expected in point["values"].items():
                            actual = values[key]
                            error = abs(actual - expected)
                            row["comparisons"][key] = {"native": actual, "reference": expected,
                                                        "absolute_error": error,
                                                        "passed": error <= tolerance(key, expected)}
                        row["reference_failures"] = point["failures"]
                        row["passed"] = (not point["failures"]
                                         and all(q["passed"] for q in row["comparisons"].values()))
                        row["status"] = "compared"
                except Exception:
                    row["failures"].append(traceback.format_exc())
                unit["points"].append(row)
            unit["passed"] = len(unit["points"]) == points_per_phase and all(
                p["passed"] for p in unit["points"]
            )
        except Exception:
            unit["failure"] = traceback.format_exc()
        unit["unexecuted_points"] = points_per_phase - len(unit["points"])
        report["units"].append(unit)
        persist(output, report)
        progress("native phases", len(report["units"]), report["expected_phases"], started)
    report.update(
        passed_phases=sum(u["passed"] for u in report["units"]),
        passed_points=sum(p["passed"] for u in report["units"] for p in u["points"]),
        failed_points=sum(not p["passed"] for u in report["units"] for p in u["points"]),
        unexecuted_points=sum(u["unexecuted_points"] for u in report["units"]),
        complete_and_passed=len(report["units"]) == report["expected_phases"]
        and all(u["passed"] for u in report["units"]),
    )
    persist(output, report)
    return report


def main():
    global ALLOW_ONE_SIDED, SECOND_ORDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--second-order", action="store_true")
    parser.add_argument("--allow-one-sided", action="store_true")
    args = parser.parse_args()
    SECOND_ORDER = args.second_order
    ALLOW_ONE_SIDED = args.allow_one_sided
    if args.reference_only:
        reference(args.source_directory, args.output)
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("CALPHAD evidence must be outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    reference_path = args.output / "reference.json"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [str(args.reference_python), "-I", str(REFERENCE_SCRIPT),
               "--reference-only", "--source-directory", str(args.source_directory),
               "--output", str(reference_path)]
    if SECOND_ORDER:
        command.append("--second-order")
    if ALLOW_ONE_SIDED:
        command.append("--allow-one-sided")
    with (args.output / "reference.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=log, stderr=log, env=environment) as process:
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    count = 0
                    if reference_path.exists():
                        try:
                            partial = json.loads(reference_path.read_text(encoding="utf-8"))
                            count = len(partial["units"])
                        except ValueError:
                            pass
                    print(f"reference alive: pid={process.pid}, phases_completed={count}, "
                          f"log_bytes={log.tell()}", flush=True)
        if process.returncode:
            write_new(args.output / "summary.json", {"complete_and_passed": False,
                      "reference_exit_code": process.returncode})
            return 1
    result = native(json.loads(reference_path.read_text(encoding="utf-8")),
                    args.source_directory, args.output / "summary.json")
    print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return 0 if result["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
