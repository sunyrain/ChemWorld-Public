"""Fixed six SED-ML scans through the actual isolated COPASI importer/task."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

CORE = "http://www.sbml.org/sbml/level3/version2/core"
MATH = "http://www.w3.org/1998/Math/MathML"
SEDML = "http://sed-ml.org/sed-ml/level1/version4"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def model_xml(kind):
    initial = ('<listOfInitialAssignments><initialAssignment symbol="A"><math xmlns="'+MATH+'">'
        '<apply><times/><cn>2</cn><ci>k</ci></apply></math></initialAssignment>'
        '</listOfInitialAssignments>') if "I" in kind else ""
    event = ('<listOfEvents><event id="initial_event" useValuesFromTriggerTime="true">'
        '<trigger initialValue="false" persistent="true"><math xmlns="'+MATH+'">'
        '<apply><gt/><ci>k</ci><cn>1</cn></apply></math></trigger><delay><math xmlns="'+MATH+'">'
        '<cn>0.25</cn></math></delay><listOfEventAssignments><eventAssignment variable="A">'
        '<math xmlns="'+MATH+'"><ci>k</ci></math></eventAssignment></listOfEventAssignments>'
        '</event></listOfEvents>') if "E" in kind else ""
    return ('<sbml xmlns="'+CORE+'" level="3" version="2"><model id="diagnostic_'+kind+'" '
        'timeUnits="second" substanceUnits="mole" extentUnits="mole" volumeUnits="litre">'
        '<listOfUnitDefinitions><unitDefinition id="molar"><listOfUnits><unit kind="mole" '
        'exponent="1" scale="0" multiplier="1"/><unit kind="litre" exponent="-1" '
        'scale="0" multiplier="1"/></listOfUnits></unitDefinition><unitDefinition id="per_second">'
        '<listOfUnits><unit kind="second" exponent="-1" scale="0" multiplier="1"/>'
        '</listOfUnits></unitDefinition></listOfUnitDefinitions><listOfCompartments>'
        '<compartment id="c" size="1" units="litre" spatialDimensions="3" constant="true"/>'
        '</listOfCompartments><listOfSpecies><species id="A" compartment="c" '
        'initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" '
        'constant="false"/><species id="B" compartment="c" initialConcentration="0" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        '</listOfSpecies><listOfParameters><parameter id="k" value="2" units="molar" '
        'constant="true"/><parameter id="rho" value="0.1" units="per_second" constant="true"/>'
        '</listOfParameters>'+initial+'<listOfReactions><reaction id="J" reversible="false">'
        '<listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/>'
        '</listOfReactants><listOfProducts><speciesReference species="B" stoichiometry="1" '
        'constant="true"/></listOfProducts><kineticLaw><math xmlns="'+MATH+'">'
        '<apply><times/><ci>rho</ci><ci>c</ci><ci>A</ci></apply></math></kineticLaw></reaction>'
        '</listOfReactions>'+event+'</model></sbml>')


def experiment_xml(kind, reset):
    path = "/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']/@value"
    variables = (("t", 'symbol="urn:sedml:symbol:time"'),
        ("a", 'target="/s:sbml/s:model/s:listOfSpecies/s:species[@id=\'A\']"'),
        ("kin", 'target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id=\'k\']"'))
    generators, datasets = "", ""
    for identity, target in variables:
        generators += ('<dataGenerator id="dg_'+identity+'"><listOfVariables><variable id="v_'+
            identity+'" taskReference="scan" '+target+'/></listOfVariables><math xmlns="'+MATH+
            '"><ci>v_'+identity+'</ci></math></dataGenerator>')
        datasets += ('<dataSet id="d_'+identity+'" label="'+identity+'" dataReference="dg_'+
                     identity+'"/>')
    return ('<sedML xmlns="'+SEDML+'" xmlns:s="'+CORE+'" level="1" version="4">'
        '<listOfModels><model id="m" language="urn:sedml:language:sbml" source="'+kind+'.xml"/>'
        '</listOfModels><listOfSimulations><uniformTimeCourse id="sim" initialTime="0" '
        'outputStartTime="0" outputEndTime="1" numberOfSteps="4">'
        '<algorithm kisaoID="KISAO:0000088">'
        '<listOfAlgorithmParameters><algorithmParameter kisaoID="KISAO:0000209" value="1e-10"/>'
        '<algorithmParameter kisaoID="KISAO:0000211" value="1e-12"/>'
        '<algorithmParameter kisaoID="KISAO:0000467" value="0.02"/></listOfAlgorithmParameters>'
        '</algorithm></uniformTimeCourse></listOfSimulations><listOfTasks>'
        '<task id="course" modelReference="m" simulationReference="sim"/>'
        '<repeatedTask id="scan" resetModel="'+str(reset).lower()+'" range="values" '
        'concatenate="true"><listOfRanges><vectorRange id="values"><value>0</value>'
        '<value>0.5</value><value>3</value></vectorRange></listOfRanges><listOfChanges>'
        '<setValue modelReference="m" range="values" target="'+path+'"><math xmlns="'+MATH+'">'
        '<ci>values</ci></math></setValue></listOfChanges><listOfSubTasks><subTask task="course"/>'
        '</listOfSubTasks></repeatedTask></listOfTasks><listOfDataGenerators>'+generators+
        '</listOfDataGenerators><listOfOutputs><report id="report"><listOfDataSets>'+datasets+
        '</listOfDataSets></report></listOfOutputs></sedML>')


def reference(sources, output):
    import COPASI

    report = {"expected_experiments": 6, "expected_rows": 90, "expected_values": 270,
              "version": COPASI.CVersion.VERSION.getVersion(), "units": []}
    started = time.perf_counter()
    for kind in ("I", "E", "IE"):
        for reset in (True, False):
            identity = kind+"-reset-"+str(reset).lower()
            unit = {"id": identity, "kind": kind, "reset": reset, "failures": [],
                    "task_completed": False, "rows": []}
            report["units"].append(unit)
            dm = None
            path = output/(identity+"-report.tsv")
            try:
                dm = COPASI.CRootContainer.addDatamodel()
                text = (sources/(identity+".sedml")).read_text(encoding="utf-8")
                if not dm.importSEDMLFromString(text, str(sources)):
                    raise ValueError("actual COPASI SED-ML importer failed")
                unit["import_messages"] = COPASI.CCopasiMessage.getAllMessageText(True)
                (output/(identity+"-imported.xml")).write_text(dm.exportSBMLToString(3, 2),
                                                              encoding="utf-8")
                tasks = list(dm.getTaskList())
                unit["tasks"] = [{"name": str(task.getObjectName()), "type": int(task.getType()),
                                  "scheduled": task.isScheduled()} for task in tasks]
                scan = next(task for task in tasks if task.getType() == COPASI.CTaskEnum.Task_scan)
                scan.getReport().setTarget(str(path))
                scan.getReport().setAppend(False)
                if not scan.initialize(COPASI.CCopasiTask.OUTPUT_UI) or not scan.process(True):
                    raise ValueError("actual imported COPASI scan failed")
                scan.restore()
                scan.getReport().close()
                unit["task_completed"] = True
                unit["run_messages"] = COPASI.CCopasiMessage.getAllMessageText(True)
                for task in tasks:
                    if task.getType() != COPASI.CTaskEnum.Task_timeCourse:
                        continue
                    method = task.getMethod()
                    unit["actual_method"] = str(method.getObjectName())
                    unit["actual_method_type"] = int(method.getType())
                    unit["actual_parameters"] = {}
                    for key in ("Relative Tolerance", "Absolute Tolerance", "Max Internal Steps",
                                "Max Internal Step Size"):
                        parameter = method.getParameter(key)
                        if parameter is not None:
                            unit["actual_parameters"][key] = parameter.getValue()
                    series = task.getTimeSeries()
                    unit["last_subtask_series"] = {
                        "labels": [str(series.getTitle(i))
                                   for i in range(series.getNumVariables())],
                        "rows": [[series.getData(i, j) for j in range(series.getNumVariables())]
                                 for i in range(series.getRecordedSteps())]}
                exported = dm.exportSBMLToString(3, 2)
                (output/(identity+"-reexport.xml")).write_text(exported, encoding="utf-8")
            except Exception:
                unit["failures"].append(traceback.format_exc())
            finally:
                if dm is not None:
                    COPASI.CRootContainer.removeDatamodel(dm)
            if path.exists():
                raw = path.read_text(encoding="utf-8")
                unit["raw_report"] = raw
                for line in raw.splitlines():
                    fields = re.split(r"[\s,;]+", line.strip())
                    if fields and not fields[-1]:
                        fields.pop()  # COPASI writes a final separator on every report row.
                    try:
                        numbers = [float(value) for value in fields]
                    except ValueError:
                        continue
                    if len(numbers) == 3 and all(math.isfinite(value) for value in numbers):
                        unit["rows"].append(numbers)
            unit["complete"] = (unit["task_completed"] and len(unit["rows"]) == 15
                                and not unit["failures"])
            write(output/"reference.json", report)
            count = len(report["units"])
            print(f"initialization diagnostic {count}/6 rows={len(unit['rows'])}/15 "
                  f"complete={unit['complete']} elapsed={time.perf_counter()-started:.2f}s",
                  flush=True)
    report["row_count"] = sum(len(unit["rows"]) for unit in report["units"])
    report["value_count"] = 3*report["row_count"]
    report["complete_experiments"] = sum(unit["complete"] for unit in report["units"])
    report["collection_complete"] = (report["complete_experiments"] == 6
                                     and report["row_count"] == 90)
    write(output/"reference.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("parent", "reference"), default="parent")
    parser.add_argument("--source-directory", type=Path)
    args = parser.parse_args()
    if args.stage == "reference":
        reference(args.source_directory, args.output)
        return 0
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root) or args.reference_python is None:
        raise ValueError("reference interpreter and an outside-Git evidence directory are required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output/"inputs"
    sources.mkdir()
    for kind in ("I", "E", "IE"):
        (sources/(kind+".xml")).write_text(model_xml(kind), encoding="utf-8")
        for reset in (True, False):
            path = sources/(kind+"-reset-"+str(reset).lower()+".sedml")
            path.write_text(experiment_xml(kind, reset), encoding="utf-8")
    shutil.copy2(__file__, args.output/"executed_source.py")
    note = root/"workstreams/world_foundation/NATIVE_CONV50C_EXPERIMENT.md"
    shutil.copy2(note, args.output/note.name)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [str(args.reference_python), "-I", str(Path(__file__).resolve()), "--stage",
               "reference", "--source-directory", str(sources), "--output", str(args.output)]
    with ((args.output/"reference.log").open("x", encoding="utf-8") as log,
          subprocess.Popen(command, stdout=log, stderr=log, env=environment) as child):
        while True:
            try:
                child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f"reference diagnostic alive pid={child.pid} bytes={log.tell()}",
                      flush=True)
    write(args.output/"process.json", {"exit": child.returncode, "command": command})
    data_path = args.output/"reference.json"
    report = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else {}
    summary = {"expected_experiments": 6, "expected_rows": 90, "expected_values": 270,
               **{k: v for k, v in report.items() if k != "units"}}
    summary["exit"] = child.returncode
    write(args.output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary.get("collection_complete") and child.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
