"""Fixed six-database input comparison; upstream runs only in an isolated Python."""

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
    "Cr-Fe-Ni_shallow_bcc.tdb", "Al-Fe_sundman2009.tdb",
)
TEMPERATURES = (300.0, 700.0, 1100.0)
PRESSURE = 101325.0


def write(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def identity(kind, phase, constituents, order, diffusion=None):
    return {"kind": kind, "phase": phase, "constituents": constituents,
            "order": order, "diffusing_species": diffusion}


def parameter_key(value):
    return json.dumps(value, sort_keys=True)


def reference(directory):
    import pycalphad
    import pycalphad.variables as v
    from symengine import Symbol

    if pycalphad.__version__ != "0.11.2":
        raise ValueError("the reference must be pycalphad 0.11.2")
    units = []
    for name in DATABASES:
        path = directory / name
        unit = {"database": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            db = pycalphad.Database(str(path))
            symbols = {Symbol(k): val for k, val in db.symbols.items()}

            def points(expression, symbols=symbols):
                for _ in range(64):
                    updated = expression.xreplace(symbols)
                    if updated == expression:
                        break
                    expression = updated
                results = []
                for temperature in TEMPERATURES:
                    try:
                        value = float(expression.subs({v.T: temperature, v.P: PRESSURE}))
                        if not math.isfinite(value):
                            raise ValueError("nonfinite reference result")
                        results.append({"T": temperature, "value": value})
                    except Exception:
                        results.append({"T": temperature, "error": traceback.format_exc()})
                return results

            original = [p for p in db._parameters.all()
                        if not p.get("_generated_by_symmetry_option", False)]
            unit.update(
                elements=sorted(db.elements),
                species={s.name: {"constituents": dict(s.constituents), "charge": s.charge}
                         for s in sorted(db.species, key=lambda s: s.name)},
                phases={name: {"site_ratios": list(map(float, phase.sublattices)),
                               "constituents": [sorted(s.name for s in subl)
                                                for subl in phase.constituents]}
                        for name, phase in sorted(db.phases.items())},
                functions={name: points(expression) for name, expression in db.symbols.items()},
                parameters=[{
                    "identity": identity(
                        p["parameter_type"], p["phase_name"],
                        [[s.name for s in subl] for subl in p["constituent_array"]],
                        p["parameter_order"], (str(p["diffusing_species"]) or None)
                        if p.get("diffusing_species") is not None else None,
                    ),
                    "points": points(p["parameter"]),
                } for p in original],
                generated_symmetry_parameters=len(db._parameters.all()) - len(original),
            )
        except Exception:
            unit["failure"] = traceback.format_exc()
        units.append(unit)
        print(f"reference declarations {len(units)}/{len(DATABASES)}: {name}", flush=True)
    return {"reference": "pycalphad 0.11.2", "units": units,
            "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                        for d in importlib.metadata.distributions()))}


def native(reference_data, directory):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.data.tdb import TdbDatabase
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    units = []
    started = time.monotonic()
    for ref in reference_data["units"]:
        name = ref["database"]
        expected_points = 3 * (len(ref.get("functions", {})) + len(ref.get("parameters", [])))
        unit = {"database": name, "passed": False, "expected_points": expected_points,
                "points": [], "failures": [], "structure_passed": False}
        try:
            if "failure" in ref:
                raise ValueError("upstream reference failed: " + ref["failure"])
            path = directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                raise ValueError("TDB changed between upstream and native parsing")
            source = InputSource(DataSourceProvenance(
                "pycalphad-test-tdb:" + name, 0,
                "https://github.com/pycalphad/pycalphad/blob/0.11.2/"
                "pycalphad/tests/databases/" + name,
                "0.11.2", "Database-specific terms not independently established; external only",
                ref["sha256"],
            ), "literature")
            db = TdbDatabase.from_file(path, source=source)
            assert sorted(db.elements) == ref["elements"], "element identities"
            assert {s.name: {"constituents": dict(s.constituents), "charge": s.charge}
                    for s in db.species.values()} == ref["species"], "species/charge"
            assert {p.name: {"site_ratios": list(p.site_ratios),
                             "constituents": [list(s) for s in p.constituents]}
                    for p in db.phases.values()} == ref["phases"], "phases/sites"
            assert set(db.functions) == set(ref["functions"]), "function identities"
            native_parameters = {parameter_key(identity(
                p.kind, p.phase, [list(s) for s in p.constituents], p.order, p.diffusing_species,
            )): p for p in db.parameters}
            assert set(native_parameters) == {
                parameter_key(p["identity"]) for p in ref["parameters"]
            }, "parameter identities"
            assert len(native_parameters) == len(ref["parameters"]), "parameter count"
            unit["structure_passed"] = True
            unit["summary"] = db.summary()
            unit["generated_symmetry_parameters_deferred_to_41B"] = ref[
                "generated_symmetry_parameters"
            ]

            def compare(label, values, evaluate, unit=unit):
                for point in values:
                    row = {"declaration": label, "T": point["T"], "status": "failed"}
                    try:
                        value = evaluate(point["T"])
                        if "error" in point:
                            raise ValueError("upstream point failed: " + point["error"])
                        error = abs(value - point["value"])
                        row.update(value=value, reference=point["value"], absolute_error=error)
                        assert error <= 1e-7 + 1e-11 * abs(point["value"]), "value tolerance"
                        row["status"] = "value_passed"
                    except ValueError as exc:
                        if "outside declared intervals" in str(exc):
                            row.update(status="domain_rejected", diagnostic=str(exc))
                        else:
                            row["failure"] = traceback.format_exc()
                    except Exception:
                        row["failure"] = traceback.format_exc()
                    unit["points"].append(row)

            for label, values in ref["functions"].items():
                compare(label, values, lambda t, label=label, db=db:
                        db.evaluate_function(label, t, PRESSURE))
            for entry in ref["parameters"]:
                label = parameter_key(entry["identity"])
                parameter = native_parameters[label]
                compare(label, entry["points"],
                        lambda t, p=parameter, db=db: db.evaluate_parameter(p, t, PRESSURE))
            unit["passed"] = len(unit["points"]) == expected_points and all(
                p["status"] in {"value_passed", "domain_rejected"} for p in unit["points"]
            )
        except Exception:
            unit["failures"].append(traceback.format_exc())
        unit["unexecuted_points"] = expected_points - len(unit["points"])
        units.append(unit)
        elapsed = time.monotonic() - started
        rate = f"{len(units)/elapsed:.3f}/s" if elapsed > 0 else "not resolved"
        eta = (len(DATABASES) - len(units)) * elapsed / len(units)
        print(f"native {len(units)}/{len(DATABASES)} {name}: passed={unit['passed']}, "
              f"rate={rate} ETA={eta:.1f}s",
              flush=True)
    counts = {status: sum(p["status"] == status for u in units for p in u["points"])
              for status in ("value_passed", "domain_rejected", "failed")}
    return {"complete_and_passed": len(units) == 6 and all(u["passed"] for u in units),
            "expected_databases": 6, "passed_databases": sum(u["passed"] for u in units),
            "expected_points": sum(u["expected_points"] for u in units), **counts,
            "unexecuted_points": sum(u["unexecuted_points"] for u in units), "units": units}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.reference_only:
        write(args.output, reference(args.source_directory))
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        raise ValueError("reference results must be outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    reference_path = args.output / "reference.json"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [str(args.reference_python), "-I", str(Path(__file__).resolve()),
               "--reference-only", "--source-directory", str(args.source_directory),
               "--output", str(reference_path)]
    with (args.output / "reference.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=log, stderr=log, env=environment) as process:
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"reference preparation alive: pid={process.pid}, "
                          f"log_bytes={log.tell()}", flush=True)
        if process.returncode:
            write(args.output / "summary.json", {"complete_and_passed": False,
                  "reference_exit_code": process.returncode, "expected_databases": 6,
                  "unexecuted_databases": 6})
            return 1
    result = native(json.loads(reference_path.read_text(encoding="utf-8")), args.source_directory)
    write(args.output / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "units"}), flush=True)
    return 0 if result["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
