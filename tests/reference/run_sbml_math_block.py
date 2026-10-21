"""Fixed 78-input, three-way COPASI/MathML/high-precision development block."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

OPERATORS = ("sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh", "sech",
    "csch", "coth", "arcsin", "arccos", "arctan", "arcsec", "arccsc", "arccot",
    "arcsinh", "arccosh", "arctanh", "arcsech", "arccsch", "arccoth", "exp", "ln")


def inputs():
    for op in OPERATORS:
        points = ((.2, .5, .8) if op in {"arcsin", "arccos", "arctanh", "arcsech"} else
                  (1.2, 2., 4.) if op in {"arccosh", "arcsec", "arccsc", "arccoth"} else
                  (.2, .7, 1.1) if op == "ln" else (-.7, .2, 1.1))
        for i, x in enumerate(points):
            name = f"{op}-{i}.xml"
            xml = f'''<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core"
level="3" version="2"><model id="math_{op}_{i}" timeUnits="second" substanceUnits="mole"
extentUnits="mole" volumeUnits="litre"><listOfParameters><parameter id="x" value="{x!r}"
units="dimensionless" constant="true"/><parameter id="out" units="dimensionless"
constant="false"/></listOfParameters><listOfRules><assignmentRule variable="out"><math
xmlns="http://www.w3.org/1998/Math/MathML"><apply><{op}/><ci>x</ci></apply></math>
</assignmentRule></listOfRules></model></sbml>'''
            yield name, op, x, xml


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def source_files(directory):
    cases = list(inputs())
    if directory.exists():
        if {p.name for p in directory.iterdir()} != {c[0] for c in cases}:
            raise ValueError("existing mathematical input block differs")
        for name, _, _, xml in cases:
            if (directory/name).read_bytes() != xml.encode("utf-8"):
                raise ValueError("refusing to overwrite original mathematical source "+name)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        for name, _, _, xml in cases:
            (directory/name).write_bytes(xml.encode("utf-8"))


def reference(directory, output):
    import COPASI
    import libsbml
    import mpmath as mp

    versions = {key: importlib.metadata.version(key)
                for key in ("python-copasi", "python-libsbml", "mpmath")}
    if versions != {"python-copasi": "4.48.309", "python-libsbml": "5.21.2", "mpmath": "1.3.0"}:
        raise ValueError("reference versions differ from the fixed block")
    mp.mp.dps = 60
    # Independent standard library functions and explicit reciprocal identities.
    functions = {name: getattr(mp, name) for name in ("sin", "cos", "tan", "sec", "csc", "cot",
        "sinh", "cosh", "tanh", "sech", "csch", "coth", "exp")}
    functions.update({name: getattr(mp, name.replace("arc", "a")) for name in (
        "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh")})
    functions.update({"ln": mp.log, "arcsec": lambda x: mp.acos(1/x),
        "arccsc": lambda x: mp.asin(1/x), "arccot": lambda x: mp.atan(1/x),
        "arcsech": lambda x: mp.acosh(1/x), "arccsch": lambda x: mp.asinh(1/x),
        "arccoth": lambda x: mp.atanh(1/x)})
    report = {"versions": versions, "CVersion": COPASI.CVersion.VERSION.getVersion(),
        "expected_inputs": 78, "mpmath_decimal_precision": 60, "units": [],
        "native_imported": any(n == "chemworld" or n.startswith("chemworld.") for n in sys.modules)}
    started = time.perf_counter()
    for name, op, x, _ in inputs():
        text = (directory/name).read_text(encoding="utf-8")
        unit = {"input": name, "operator": op, "x": x, "failures": [],
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        report["units"].append(unit)
        dm = None
        try:
            doc = libsbml.readSBMLFromString(text)
            doc.checkConsistency()
            messages = [doc.getError(i) for i in range(doc.getNumErrors())]
            unit["libsbml_messages"] = [m.getMessage() for m in messages]
            if any(m.getSeverity() >= libsbml.LIBSBML_SEV_ERROR for m in messages):
                raise ValueError("independent SBML validation failed")
            smodel = doc.getModel()
            unit["independent_structure"] = {"parameters": smodel.getNumParameters(),
                "rules": smodel.getNumRules(), "x": smodel.getParameter("x").getValue(),
                "out_constant": smodel.getParameter("out").getConstant(),
                "assignment_target": smodel.getRule(0).getVariable(),
                "mathml": libsbml.writeMathMLToString(smodel.getRule(0).getMath())}
            dm = COPASI.CRootContainer.addDatamodel()
            if not dm.importSBMLFromString(text):
                raise ValueError("COPASI SBML import failed")
            model = dm.getModel()
            if not model.compileIfNecessary():
                raise ValueError("COPASI compilation failed")
            model.applyInitialValues()
            value = model.getModelValue("out")
            unit["copasi_initial"] = float(value.getInitialValue())
            unit["copasi_current"] = float(value.getValue())
            unit["mpmath"] = [float(mp.diff(functions[op], mp.mpf(repr(x)), order))
                              for order in range(4)]
            if not all(math.isfinite(n) for n in (unit["copasi_initial"], unit["copasi_current"],
                                                  *unit["mpmath"])):
                raise ValueError("nonfinite independent mathematical response")
        except Exception:
            unit["failures"].append(traceback.format_exc())
        finally:
            if dm is not None:
                COPASI.CRootContainer.removeDatamodel(dm)
        write(output, report)
        count = len(report["units"])
        elapsed = max(time.perf_counter()-started, 1e-9)
        print(f"reference {count}/78 rate={count/elapsed:.2f}/s "
              f"eta={(78-count)*elapsed/count:.1f}s {name}", flush=True)


def native(directory, reference_report, output):
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.caloric_math import derivative_values
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    report = {"expected_inputs": 78, "expected_comparisons": 468, "units": [],
              "units_of_all_quantities": "dimensionless (derivatives per dimensionless x)"}
    source = InputSource(DataSourceProvenance("synthetic-sbml-math", 0,
        "Self-authored fixed 78-input mathematical qualification", "1", "CC0-1.0"), "synthetic")
    started = time.perf_counter()
    for name, op, x, _ in inputs():
        unit = {"input": name, "operator": op, "x": x, "passed": False, "failures": [],
                "comparisons": []}
        report["units"].append(unit)
        try:
            reference_unit = next(r for r in reference_report["units"] if r["input"] == name)
            if reference_unit["failures"]:
                raise ValueError("independent reference failed: "+repr(reference_unit["failures"]))
            raw = (directory/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference_unit["sha256"]:
                raise ValueError("source bytes changed after independent reference")
            model = SBMLModel(raw.decode("utf-8"), source)
            declaration = model.declaration
            if (len(declaration.items("listOfParameters")) != 2
                    or len(declaration.items("listOfRules")) != 1):
                raise ValueError("native declaration coverage differs")
            math_expression = declaration.items("listOfRules")[0].mathematics
            unit["native"] = derivative_values(
                lambda z, expression=math_expression: expression.evaluate({"x": z}), x, order=3)
            checks = [("native_COPASI", unit["native"][0], reference_unit["copasi_current"], 0),
                      ("COPASI_mpmath", reference_unit["copasi_current"],
                       reference_unit["mpmath"][0], 0)]
            checks += [("native_mpmath_order_"+str(order), value,
                        reference_unit["mpmath"][order], order)
                       for order, value in enumerate(unit["native"])]
            for label, actual, expected, order in checks:
                tolerance = (1e-10+1e-9*abs(expected) if order else 2e-12+2e-11*abs(expected))
                error = abs(actual-expected)
                row = {"quantity": label, "actual": actual, "reference": expected,
                       "error": error, "tolerance": tolerance, "passed": error <= tolerance}
                unit["comparisons"].append(row)
                if not row["passed"]:
                    unit["failures"].append(row)
            unit["passed"] = not unit["failures"]
        except Exception:
            unit["failures"].append(traceback.format_exc())
        write(output, report)
        count = len(report["units"])
        elapsed = max(time.perf_counter()-started, 1e-9)
        print(f"native {count}/78 rate={count/elapsed:.2f}/s "
              f"eta={(78-count)*elapsed/count:.1f}s passed={unit['passed']}", flush=True)
    report["executed_inputs"] = sum("native" in u for u in report["units"])
    report["passed_inputs"] = sum(u["passed"] for u in report["units"])
    report["failed_executed_inputs"] = report["executed_inputs"]-report["passed_inputs"]
    report["unexecuted_inputs"] = 78-report["executed_inputs"]
    report["comparison_count"] = sum(len(u["comparisons"]) for u in report["units"])
    report["complete_and_passed"] = (report["passed_inputs"] == 78
                                     and report["comparison_count"] == 468)
    write(output, report)


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
    root = Path(__file__).resolve().parents[2]
    if any(p.resolve().is_relative_to(root) for p in (args.source_directory, args.output)):
        raise ValueError("scientific inputs and outputs must stay outside the checkout")
    source_files(args.source_directory)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    exits = {}
    for stage, interpreter in (("reference", args.reference_python), ("native", sys.executable)):
        command = [str(interpreter), *(["-I"] if stage == "reference" else []),
                   str(Path(__file__).resolve()), "--stage", stage,
                   "--source-directory", str(args.source_directory), "--output", str(args.output)]
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
        exits[stage] = process.returncode
        write(args.output/"stage_exits.json", exits)
        print(stage+" exit="+str(process.returncode), flush=True)
        if process.returncode:
            break
    path = args.output/"native.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary = {k: v for k, v in result.items() if k != "units"}
    summary["stage_exits"] = exits
    summary["complete_and_passed"] = result.get("complete_and_passed", False) and exits == {
        "reference": 0, "native": 0}
    summary["evidence_directory"] = str(args.output)
    write(args.output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
