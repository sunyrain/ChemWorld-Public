"""Isolated SDK/60-digit references for the fixed twelve complete OMEX inputs.

No ChemWorld imports: this module is launched by the locked parent with SDK Python -I.
"""

import argparse
import importlib.metadata
import json
import math
import re
import time
import traceback
import warnings
from itertools import product
from pathlib import Path

LABELS = ("t", "a", "b", "j", "v")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def normalized_array(value, logical_shape):
    """Map axes by declared cardinalities only; never inspect the simulated values."""
    import numpy as np

    array = np.asarray(value, dtype=float)
    squeezed = np.squeeze(array)
    shape = tuple(logical_shape)
    expected = math.prod(shape)
    if squeezed.size != expected or not np.all(np.isfinite(squeezed)):
        raise ValueError("SDK must return the complete finite declared sample count")
    if squeezed.ndim == 1:
        return squeezed.ravel().tolist(), {"mapping": "SDK flat order", "shape": list(array.shape)}
    # The SDK can collapse adjacent logical repeat axes, e.g. (2,3,7) -> (7,6).
    # Enumerate contiguous partitions of the declaration, without consulting any values.
    candidates = []
    for cuts in product((False, True), repeat=len(shape)-1):
        groups, start = [], 0
        for index, cut in enumerate((*cuts, True)):
            if cut:
                groups.append(shape[start:index+1])
                start = index+1
        lengths = tuple(math.prod(group) for group in groups)
        if (len(lengths) == squeezed.ndim and len(set(lengths)) == len(lengths)
                and sorted(lengths) == sorted(squeezed.shape)):
            candidates.append((groups, tuple(squeezed.shape.index(n) for n in lengths)))
    if len(candidates) != 1:
        raise ValueError("SDK axes cannot be mapped uniquely from declared cardinalities")
    groups, permutation = candidates[0]
    return squeezed.transpose(permutation).ravel().tolist(), {
        "mapping": "declared cardinalities", "shape": list(array.shape),
        "logical_shape": list(shape), "collapsed_groups": groups, "permutation": list(permutation)}


def tellurium_case(case, inputs, output, unit):
    import libsedml
    import numpy as np
    import roadrunner
    import tellurium
    from tellurium.sedml.tesedml import SEDMLCodeFactory
    from tellurium.utils import omex

    unit["versions"] = {"tellurium": tellurium.__version__,
        "roadrunner": roadrunner.__version__, "libsedml": libsedml.getLibSEDMLDottedVersion()}
    extracted = output/"sdk_extracted"
    extracted.mkdir()
    archive = str(inputs/case["id"]/"experiment.omex")
    # Use the SDK's public archive helpers, as executeCombineArchive does. Its factory's
    # direct ZIP constructor refers to an undefined CombineArchive in this SDK release.
    omex.extractCombineArchive(archive, directory=str(extracted))
    locations = omex.getLocationsByFormat(archive, formatKey="sed-ml", method="omex")
    unit["archive_reader"] = "tellurium.utils.omex (executeCombineArchive pipeline)"
    unit["archive_sedml_locations"] = locations
    if locations != ["experiments/experiment.sedml"]:
        raise ValueError("SDK manifest reader must select the original unique master experiment")
    for name in ("models/model.xml", "experiments/experiment.sedml"):
        if (extracted/name).read_bytes() != (inputs/case["id"]/name).read_bytes():
            raise ValueError("SDK extraction changed the original model or experiment")
    sedml_file = extracted/locations[0]
    factory = SEDMLCodeFactory(str(sedml_file), sedmlFile=str(sedml_file),
        workingDir=str(sedml_file.parent), createOutputs=False, saveOutputs=False)
    unit["input_type"] = factory.inputType
    unit["messages"] = [factory.doc.getError(i).getMessage()
                        for i in range(factory.doc.getNumErrors())]
    code = factory.toPython()
    (output/"generated.py").write_text(code, encoding="utf-8")
    executed = output/"executed.py"
    executed.write_text(code, encoding="utf-8")
    # This is the same unmodified SDK program executed by executePython. Retaining its
    # namespace additionally permits actual RoadRunner SBML reexport, including on failure.
    symbols = {}
    try:
        exec(compile(code, str(executed), "exec"), symbols)
    finally:
        for model in factory.doc.getListOfModels():
            runner = symbols.get(model.getId())
            if runner is not None and hasattr(runner, "getCurrentSBML"):
                (output/(model.getId()+"-reexport.xml")).write_text(runner.getCurrentSBML(),
                                                                  encoding="utf-8")
    unit["task_completed"] = True
    arrays = {key: np.asarray(value, dtype=float)
              for key, value in symbols.items() if key in {
                  generator.getId() for generator in factory.doc.getListOfDataGenerators()}}
    # Nonfinite SDK output is preserved losslessly as strings rather than invalid JSON numbers.
    unit["raw_arrays"] = {key: {"shape": list(array.shape),
        "values": array.astype(str).tolist()} for key, array in arrays.items()}
    selected, mappings = [], {}
    for label in LABELS:
        values, mapping = normalized_array(arrays["dg_"+label], case["logical_shape"])
        selected.append(values)
        mappings[label] = mapping
    unit["axis_mapping"] = mappings
    unit["rows"] = [list(row) for row in zip(*selected, strict=True)]


def copasi_case(case, inputs, output, unit):
    import COPASI

    unit["versions"] = {"COPASI": COPASI.CVersion.VERSION.getVersion()}
    dm = COPASI.CRootContainer.addDatamodel()
    report_file = output/"report.tsv"
    try:
        root_task = "scan" if int(case["id"][1:]) >= 7 else "course"
        # Import the root's simple-task dependencies too; taskId=scan omits their settings.
        options = COPASI.SedmlImportOptions("", "m", COPASI.StringStdVector(), "report", "")
        unit["import_selection"] = {"task": "all for m", "model": "m", "report": "report",
                                    "executed_root": root_task}
        imported = dm.openCombineArchive(str(inputs/case["id"]/"experiment.omex"), None, True,
                                         options)
        unit["import_messages"] = COPASI.CCopasiMessage.getAllMessageText(True)
        if not imported:
            raise ValueError("actual COPASI OMEX importer failed")
        (output/"imported.xml").write_text(dm.exportSBMLToString(3, 2), encoding="utf-8")
        tasks = list(dm.getTaskList())
        unit["tasks"] = [{"name": str(task.getObjectName()), "type": int(task.getType()),
            "scheduled": task.isScheduled(), "has_report": task.getReport().getReportDefinition()
            is not None} for task in tasks]
        kind = (COPASI.CTaskEnum.Task_scan if root_task == "scan"
                else COPASI.CTaskEnum.Task_timeCourse)
        selected = [task for task in tasks if task.isScheduled() and task.getType() == kind
            and task.getReport().getReportDefinition() is not None]
        if len(selected) != 1:
            raise ValueError("expected one actual scheduled imported experiment with a report")
        task = selected[0]
        task.getReport().setTarget(str(report_file))
        task.getReport().setAppend(False)
        task.getReport().getReportDefinition().setPrecision(17)
        if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI) or not task.process(True):
            raise ValueError("actual imported COPASI task failed")
        task.restore()
        task.getReport().close()
        unit["task_completed"] = True
        unit["run_messages"] = COPASI.CCopasiMessage.getAllMessageText(True)
        unit["time_course_methods"] = []
        for candidate in tasks:
            if candidate.getType() != COPASI.CTaskEnum.Task_timeCourse:
                continue
            method = candidate.getMethod()
            parameters = {}
            for name in ("Relative Tolerance", "Absolute Tolerance", "Max Internal Steps",
                         "Max Internal Step Size"):
                parameter = method.getParameter(name)
                if parameter is not None:
                    parameters[name] = parameter.getValue()
            unit["time_course_methods"].append({"name": str(method.getObjectName()),
                "type": int(method.getType()), "parameters": parameters})
        (output/"reexport.xml").write_text(dm.exportSBMLToString(3, 2), encoding="utf-8")
    finally:
        unit["last_messages"] = COPASI.CCopasiMessage.getAllMessageText(True)
        COPASI.CRootContainer.removeDatamodel(dm)
        if report_file.exists():
            raw = report_file.read_text(encoding="utf-8")
            unit["raw_report"] = raw
            numbers, widths, malformed = [], {}, []
            for line in raw.splitlines():
                fields = re.split(r"[\s,;]+", line.strip())
                if fields and not fields[-1]:
                    fields.pop()
                try:
                    row = [float(field) for field in fields]
                except ValueError:
                    continue  # textual report headers remain in raw_report
                widths[str(len(row))] = widths.get(str(len(row)), 0)+1
                if len(row) == 5 and all(math.isfinite(value) for value in row):
                    numbers.append(row)
                elif row:
                    malformed.append(line)
            unit["numeric_width_census"], unit["malformed_numeric_rows"] = widths, malformed
            unit["raw_numeric_rows"] = numbers
            if not malformed and len(numbers) == math.prod(case["shape"]):
                unit["rows"] = numbers
                unit["axis_mapping"] = "SDK report order; all five declared columns"


def sdk(stage, manifest, inputs, output):
    report = {"expected_experiments": 12, "expected_rows": 182, "expected_values": 910,
              "units": [], "installed_distributions": {dist.metadata["Name"]: dist.version
                  for dist in importlib.metadata.distributions()}}
    started = time.perf_counter()
    for case in manifest:
        directory = output/case["id"]
        directory.mkdir()
        unit = {"id": case["id"], "rows": [], "failures": [], "task_completed": False}
        report["units"].append(unit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                execute = copasi_case if stage == "copasi" else tellurium_case
                execute(case, inputs, directory, unit)
            except Exception:
                unit["failures"].append(traceback.format_exc())
            finally:
                unit["warnings"] = [str(item.message) for item in caught]
        unit["complete"] = (unit["task_completed"] and not unit["failures"]
                            and len(unit["rows"]) == math.prod(case["shape"]))
        if not unit["complete"]:
            unit["rows"] = []  # incomplete cases retain raw output but are not partly aligned
        write(output/"reference.json", report)
        count = len(report["units"])
        elapsed = time.perf_counter()-started
        print(f"{stage} {count}/12 rows={len(unit['rows'])}/{math.prod(case['shape'])} "
              f"complete={unit['complete']} elapsed={elapsed:.2f}s "
              f"ETA={elapsed/count*(12-count):.2f}s", flush=True)
    report["complete_experiments"] = sum(unit["complete"] for unit in report["units"])
    report["row_count"] = sum(len(unit["rows"]) for unit in report["units"])
    report["value_count"] = 5*report["row_count"]
    report["collection_complete"] = report["complete_experiments"] == 12
    methods = [method for unit in report["units"] for method in unit.get("time_course_methods", [])]
    report["algorithm_parity_passed"] = (stage == "copasi" and bool(methods)
        and all("LSODA" in method["name"].upper() for method in methods)
        and report["collection_complete"])
    write(output/"reference.json", report)


def closed(manifest, output):
    """Independent literal schedule and analytic physics; no native scheduler reuse."""
    import mpmath as mp

    mp.mp.dps = 60
    number = mp.mpf
    report = {"digits": 60, "version": mp.__version__, "units": []}
    for case in manifest:
        index = int(case["id"][1:])
        kind = {2: "M", 3: "F", 9: "M", 10: "F"}.get(index, "D")
        volume = number(1000 if index == 4 else 1)
        second_volume = volume/2
        initial = number(1)
        rows = []

        def course(a_start, k, duration, steps, time_start, *, offset=0, kind=kind,
                   volume=volume, second_volume=second_volume, initial=initial, rows=rows):
            for sample in range(steps+1):
                dt = offset+duration*sample/steps
                if kind == "M":
                    km = number(".4")
                    a = km*mp.lambertw(a_start/km*mp.exp((a_start-k*dt)/km)).real
                elif kind == "F":
                    h = volume/(second_volume*number(".3"))
                    q = 1+h*initial
                    a = (-q/h*mp.lambertw(-h/q*a_start*mp.exp(-h/q*a_start-k*dt/q))).real
                else:
                    a = a_start*mp.exp(-k*dt)
                b = volume*(initial-a)
                j = volume*k*a
                if kind == "M":
                    j /= number(".4")+a
                elif kind == "F":
                    j /= 1+b/(second_volume*number(".3"))
                rows.append([mp.nstr(value, 65) for value in
                    (time_start+duration*sample/steps, a, b, j, volume)])
            return a

        if index <= 5:
            course(initial, number("12" if index == 4 else ".5" if index == 5 else ".2"),
                   number(".05" if index == 4 else "3"), 6, number(0))
        elif index == 6:
            course(initial, number(".2"), number(2), 4, number("5.5"), offset=number(".5"))
        elif 7 <= index <= 10:
            values = {7: (".1", ".2", ".3"), 8: (".05", ".1", ".2"),
                      9: (".1", ".2", ".4"), 10: (".1", ".2", ".4")}[index]
            a = initial
            for iteration, text in enumerate(values):
                reset = index in (7, 9)
                a = course(initial if reset else a, number(text), number(3), 6,
                           number(0 if reset else 3*iteration))
        elif index == 11:
            for text in (".1", ".2"):
                a = initial
                for iteration in range(3):
                    a = course(a, number(text), number(3), 6, number(3*iteration))
        else:
            a = initial
            for iteration, text in enumerate((".1", ".2")):
                k = number(text)
                a = course(a, k, number(1), 2, number(3*iteration))
                a = course(a, number("1.5")*k, number(2), 4, number(3*iteration+1))
        if len(rows) != math.prod(case["shape"]):
            raise ValueError("independent closed-form schedule does not match fixed count")
        report["units"].append({"id": case["id"], "rows_decimal": rows})
        write(output/"reference.json", report)
        print(f"closed {len(report['units'])}/12", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("copasi", "tellurium", "closed"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.inputs/"cases.json").read_text(encoding="utf-8"))
    if args.stage == "closed":
        closed(manifest, args.output)
    else:
        sdk(args.stage, manifest, args.inputs, args.output)


if __name__ == "__main__":
    main()
