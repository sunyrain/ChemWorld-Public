"""Fixed 18 complete SBML trajectories against actual isolated COPASI."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import time
import traceback
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

CORE = "http://www.sbml.org/sbml/level3/version2/core"
MATH = "http://www.w3.org/1998/Math/MathML"
TIMES = (0., .13, .47, .83, 1.17, 1.63, 2.13, 2.37, 2.63, 3.11, 4.)
QUANTITIES = ("N:A", "N:B", "V:c1", "V:c2", "C:A", "C:B", "p")
TIME = '<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    inputs = load_sibling("run_sbml_input_block")
    declarations = list(inputs.cases())
    for index in range(18):
        root = ET.fromstring(declarations[5 if index == 1 else 3 if index == 16 else 0][3])
        model = root[0]
        model.set("id", f"trajectory_{index+1:02d}")

        def select(kind, identity, model=model):
            return next(e for e in model.iter("{"+CORE+"}"+kind) if e.get("id") == identity)

        def math_xml(body):
            return ET.fromstring('<math xmlns="'+MATH+'">'+body+'</math>')

        def set_math(element, body):
            old = element.find("{"+MATH+"}math")
            if old is not None:
                element.remove(old)
            element.insert(0, math_xml(body))

        def rule(kind, variable, body, model=model):
            parent = model.find("{"+CORE+"}listOfRules")
            item = ET.SubElement(parent, "{"+CORE+"}"+kind, variable=variable)
            item.append(math_xml(body))

        def parameter(name, value, unit, model=model):
            ET.SubElement(model.find("{"+CORE+"}listOfParameters"), "{"+CORE+"}parameter",
                          id=name, value=str(value), units=unit, constant="true")

        events = model.find("{"+CORE+"}listOfEvents")
        existing = next(iter(events))
        if index in {2, 3, 16}:
            rule("rateRule", "c1", '<cn>0.002</cn>')
        if index == 3:
            select("species", "A").set("boundaryCondition", "true")
            rule("rateRule", "A", '<cn>0.1</cn>')
        if index == 4:
            rule("assignmentRule", "c1", '<apply><times/><cn>0.01</cn><apply><plus/><cn>1</cn>'
                 '<apply><times/><cn>0.1</cn>'+TIME+'</apply></apply></apply>')
        if index in {5, 6}:
            definition = ET.SubElement(model.find("{"+CORE+"}listOfUnitDefinitions"),
                                       "{"+CORE+"}unitDefinition", id="molar")
            units = ET.SubElement(definition, "{"+CORE+"}listOfUnits")
            for kind, exponent in (("mole", 1), ("litre", -1)):
                ET.SubElement(units, "{"+CORE+"}unit", kind=kind, exponent=str(exponent),
                              scale="0", multiplier="1")
            law = select("reaction", "J").find("{"+CORE+"}kineticLaw")
            law.remove(law.find("{"+CORE+"}listOfLocalParameters"))
            if index == 5:
                parameter("vmax", .2, "molar_rate")
                parameter("Km", 1., "molar")
                set_math(law, '<apply><divide/><apply><times/><ci>c1</ci><ci>vmax</ci><ci>A</ci>'
                         '</apply><apply><plus/><ci>Km</ci><ci>A</ci></apply></apply>')
            else:
                parameter("Ki", .2, "molar")
                set_math(law, '<apply><divide/><apply><times/><ci>c1</ci><ci>k</ci><ci>A</ci>'
                         '</apply><apply><plus/><cn>1</cn><apply><divide/><apply><divide/>'
                         '<ci>B</ci><ci>c2</ci></apply><ci>Ki</ci></apply></apply></apply>')
        if index in {7, 8}:
            existing.set("useValuesFromTriggerTime", "true" if index == 7 else "false")
            assignment = next(a for a in existing.iter("{"+CORE+"}eventAssignment")
                              if a.get("variable") == "A")
            set_math(assignment, '<apply><times/><cn>2</cn><ci>A</ci></apply>')
        if index in {9, 10}:
            trigger = existing.find("{"+CORE+"}trigger")
            trigger.set("persistent", "true" if index == 10 else "false")
            set_math(trigger, '<apply><and/><apply><gt/>'+TIME+'<cn>2</cn></apply>'
                     '<apply><lt/>'+TIME+'<cn>2.25</cn></apply></apply>')
        if index == 11:
            existing.remove(existing.find("{"+CORE+"}delay"))
            parent = existing.find("{"+CORE+"}listOfEventAssignments")
            item = ET.SubElement(parent, "{"+CORE+"}eventAssignment", variable="c1")
            item.append(math_xml('<cn>0.04</cn>'))
        if index in {12, 13}:
            trigger = existing.find("{"+CORE+"}trigger")
            trigger.set("initialValue", "false" if index == 12 else "true")
            set_math(trigger, '<true/>')
        if index in {14, 15}:
            events.clear()

            def event(name, trigger_body, priority, assignments, capture=True, events=events):
                item = ET.SubElement(events, "{"+CORE+"}event", id=name,
                                     useValuesFromTriggerTime=str(capture).lower())
                t = ET.SubElement(item, "{"+CORE+"}trigger",
                                  initialValue="false", persistent="true")
                t.append(math_xml(trigger_body))
                p = ET.SubElement(item, "{"+CORE+"}priority")
                p.append(math_xml(priority))
                group = ET.SubElement(item, "{"+CORE+"}listOfEventAssignments")
                for target, body in assignments.items():
                    a = ET.SubElement(group, "{"+CORE+"}eventAssignment", variable=target)
                    a.append(math_xml(body))

            if index == 14:
                trigger = '<apply><gt/>'+TIME+'<cn>1</cn></apply>'
                event("first", trigger, '<cn>10</cn>', {"p": '<cn>4</cn>'})
                event("second", trigger, '<ci>p</ci>',
                      {"A": '<apply><divide/><ci>B</ci><ci>c2</ci></apply>'}, capture=False)
                event("third", trigger, '<cn>3.5</cn>', {"B": '<cn>0.02</cn>'})
            else:
                event("first", '<apply><gt/>'+TIME+'<cn>1</cn></apply>',
                      '<cn>10</cn>', {"p": '<cn>5</cn>'})
                event("second", '<apply><gt/><ci>p</ci><cn>4</cn></apply>',
                      '<cn>1</cn>', {"A": '<cn>3</cn>'})
        if index == 17:
            parameter("cf", .5, "dimensionless")
            model.set("conversionFactor", "cf")
        scales = {"amount_mol": .001 if index == 1 else 1.,
                  "volume_L": .001 if index == 1 else 1.,
                  "time_s": 60. if index == 1 else 1.}
        yield f"trajectory-{index+1:02d}.xml", scales, index, inputs.source_xml(root)


def progress(stage, report, started):
    n = len(report["units"])
    elapsed = max(time.perf_counter()-started, 1e-9)
    print(f"{stage} {n}/18 rate={n/elapsed:.2f}/s eta={(18-n)*elapsed/n:.1f}s "
          f"last_passed={report['units'][-1].get('passed', False)}", flush=True)


def jumps(rows):
    result = []
    for left, right in pairwise(rows):
        tolerance = 16*max(math.ulp(left["time_source"]), math.ulp(right["time_source"]))
        if (abs(left["time_source"]-right["time_source"]) <= tolerance
                and any(abs(a-b) > 1e-9*max(1e-12, abs(a), abs(b))
                        for a, b in zip(left["values"], right["values"], strict=True))
                and (not result or abs(result[-1]-right["time_source"]) > 1e-7)):
            result.append(right["time_source"])
    return result


def reference(directory, output):
    import importlib.metadata

    import COPASI

    if importlib.metadata.version("python-copasi") != "4.48.309":
        raise ValueError("reference COPASI differs from the fixed version")
    report = {"expected_inputs": 18, "CVersion": COPASI.CVersion.VERSION.getVersion(), "units": []}
    started = time.perf_counter()
    for name, scales, _, _ in cases():
        dm = None
        unit = {"input": name, "failures": [], "passed": False, "scales": scales}
        report["units"].append(unit)
        try:
            text = (directory/name).read_text(encoding="utf-8")
            unit["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            dm = COPASI.CRootContainer.addDatamodel()
            if not dm.importSBMLFromString(text) or not dm.getModel().forceCompile():
                raise ValueError("actual COPASI import/compilation failed")
            model = dm.getModel()
            unit["COPASI_import_messages"] = COPASI.CCopasiMessage.getAllMessageText()
            unit["COPASI_stoichiometries"] = [{"id": r.getSBMLId(),
                "reactants": [e.getMultiplicity() for e in r.getChemEq().getSubstrates()],
                "products": [e.getMultiplicity() for e in r.getChemEq().getProducts()]}
                for r in model.getReactions()]
            units = {"quantity": model.getQuantityUnitName(), "volume": model.getVolumeUnitName(),
                     "time": model.getTimeUnitName()}
            unit["SDK_units"] = units
            q = {"mol": 1., "mmol": .001}[units["quantity"]]
            v = {"l": 1., "ml": .001, "L": 1., "mL": .001}[units["volume"]]
            t = {"s": 1., "min": 60.}[units["time"]]
            factor = model.getQuantity2NumberFactor()
            unit["SDK_quantity2number"] = factor
            task = dm.getTask("Time-Course")
            if not task.setMethodType(COPASI.CTaskEnum.Method_deterministic):
                raise ValueError("COPASI deterministic method unavailable")
            problem = task.getProblem()
            problem.setDuration(4*scales["time_s"]/t)
            problem.setStepNumber(400)
            problem.setTimeSeriesRequested(True)
            problem.setOutputEvent(True)
            problem.setAutomaticStepSize(False)
            method = task.getMethod()
            method.getParameter("Relative Tolerance").setValue(1e-11)
            method.getParameter("Absolute Tolerance").setValue(1e-13)
            if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI) or not task.process(True):
                raise ValueError("COPASI actual time course failed")
            series = task.getTimeSeries()
            labels = [series.getSBMLId(i, dm) for i in range(series.getNumVariables())]
            unit["columns"] = labels
            raw_rows = [{"data": [series.getData(row, col)
                                  for col in range(series.getNumVariables())],
                         "concentration_data": [series.getConcentrationData(row, col)
                                  for col in range(series.getNumVariables())]}
                        for row in range(series.getRecordedSteps())]
            unit["SDK_raw_rows"], unit["COPASI_task_completed"] = raw_rows, True
            objects = {"A": model.getMetabolite("A"), "B": model.getMetabolite("B"),
                       "c1": model.getCompartment("c1"), "c2": model.getCompartment("c2"),
                       "p": model.getModelValue("p")}
            fixed = {}
            for key, obj in objects.items():
                if key not in labels:
                    if obj.getStatus() != COPASI.CModelEntity.Status_FIXED:
                        raise ValueError("COPASI omitted a nonfixed quantity "+key)
                    fixed[key] = obj.getInitialValue()
            unit["SDK_fixed_values"] = fixed
            indices = {key: labels.index(key) for key in objects if key in labels}
            rows = []
            for row in range(series.getRecordedSteps()):
                data = {**fixed, **{key: series.getData(row, col) for key, col in indices.items()}}
                na, nb, va, vb = data["A"]/factor*q, data["B"]/factor*q, data["c1"]*v, data["c2"]*v
                values = [na, nb, va, vb, na/va, nb/vb, data["p"]]
                if not all(math.isfinite(x) for x in values):
                    raise ValueError("COPASI returned a nonfinite quantity")
                rows.append({"time_source": series.getData(row, 0)*t/scales["time_s"],
                             "values": values})
            unit["trajectory"] = rows
            unit["samples"] = []
            for target in TIMES:
                candidates = [row for row in rows if abs(row["time_source"]-target) <= 1e-9]
                if not candidates:
                    raise ValueError("COPASI omitted a fixed output node "+str(target))
                unit["samples"].append(candidates[-1])
            unit["jump_times_source"] = jumps(rows)
            unit["COPASI_reexport"] = dm.exportSBMLToString(3, 2)
            unit["passed"] = True
        except Exception:
            unit["failures"].append(traceback.format_exc())
        finally:
            if dm is not None:
                COPASI.CRootContainer.removeDatamodel(dm)
        driver.write(output, report)
        progress("reference", report, started)


def native(directory, reference_report, output):
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.coordinate_hybrid import (
        CoordinateHybridMemory,
        advance_coordinate_events,
    )
    from chemworld.physchem.coupled_system import CoupledState
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.physchem.sbml_events import SBMLTriggerRoots, sbml_coordinate_events
    from chemworld.physchem.sbml_kinetics import SBMLKinetics, sbml_continuous_system
    from chemworld.physchem.solver_backend import ODESolverPolicy

    source = InputSource(DataSourceProvenance("synthetic-sbml-trajectory-18", 0,
        "Self-authored complete continuous and event trajectories", "1", "CC0-1.0"), "synthetic")
    report = {"expected_inputs": 18, "expected_comparisons": 1386,
              "units": [], "executed_inputs": 0}
    started = time.perf_counter()
    for name, scales, _, _ in cases():
        unit = {"input": name, "passed": False, "failures": [], "comparisons": []}
        report["units"].append(unit)
        try:
            ref = next(u for u in reference_report["units"] if u["input"] == name)
            if not ref["passed"]:
                raise ValueError("independent trajectory unavailable: "+repr(ref["failures"]))
            text = (directory/name).read_text(encoding="utf-8")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != ref["sha256"]:
                raise ValueError("source changed after independent execution")
            frame = SBMLKinetics(SBMLModel(text, source))
            events, roots = sbml_coordinate_events(frame), SBMLTriggerRoots(frame)
            policy = ODESolverPolicy("sbml_block_bdf", "BDF", rtol=1e-10, atol=1e-12,
                                     max_step_s=.02*scales["time_s"])
            system = sbml_continuous_system(frame, policy=policy)
            state = CoupledState({}, 298.15, 1., coordinates=frame.initial_values)
            memory = CoordinateHybridMemory.initialize(events)
            samples, trajectory, executed, diagnostics = [], [], [], []
            for target in TIMES:
                result = advance_coordinate_events(system, state,
                    target*scales["time_s"]-state.time_s, memory, events,
                    roots=roots.roots, trigger_values=roots.trigger_values)
                state, memory = result.final_state, result.memory
                amounts, volumes = frame.quantities(state.coordinates, state.time_s)
                samples.append({"time_source": target, "values": [amounts["A"], amounts["B"],
                    volumes["c1"], volumes["c2"], amounts["A"]/volumes["c1"],
                    amounts["B"]/volumes["c2"],
                    frame.context(state.coordinates, state.time_s)["p"]]})
                trajectory.extend({"time_s": s.time_s, "coordinates": dict(s.coordinates)}
                                  for s in result.trajectory)
                executed.extend(result.events)
                diagnostics.extend(result.segment_diagnostics)
            unit.update(samples=samples, trajectory=trajectory, events=executed,
                        segment_diagnostics=diagnostics, memory=memory.to_dict())
            report["executed_inputs"] += 1
            for actual, expected in zip(samples, ref["samples"], strict=True):
                for quantity, a, b in zip(QUANTITIES, actual["values"],
                                           expected["values"], strict=True):
                    absolute, relative = ((2e-10, 2e-6) if quantity.startswith("N:") else
                        (1e-12, 2e-8) if quantity.startswith("V:") else
                        (2e-8, 2e-6) if quantity.startswith("C:") else (1e-8, 2e-6))
                    tolerance = absolute+relative*abs(b)
                    comparison = {"time_source": actual["time_source"], "quantity": quantity,
                        "actual": a, "reference": b, "error": abs(a-b), "tolerance": tolerance,
                        "passed": abs(a-b) <= tolerance}
                    unit["comparisons"].append(comparison)
                    if not comparison["passed"]:
                        unit["failures"].append(comparison)
            native_times = []
            for event in executed:
                current = event["time_s"]/scales["time_s"]
                if not native_times or abs(current-native_times[-1]) > 1e-7:
                    native_times.append(current)
            reference_times = ref["jump_times_source"]
            unit["native_jump_times_source"] = native_times
            unit["reference_jump_times_source"] = reference_times
            if (len(native_times) != len(reference_times) or any(abs(a-b) > 1e-7
                    for a, b in zip(native_times, reference_times, strict=False))):
                unit["failures"].append("observable event jump times differ")
            unit["passed"] = not unit["failures"]
        except Exception:
            unit["failures"].append(traceback.format_exc())
        driver.write(output, report)
        progress("native", report, started)
    report["passed_inputs"] = sum(u["passed"] for u in report["units"])
    report["unexecuted_inputs"] = 18-report["executed_inputs"]
    report["comparison_count"] = sum(len(u["comparisons"]) for u in report["units"])
    report["complete_and_passed"] = (report["passed_inputs"] == 18
                                      and report["comparison_count"] == 1386)
    driver.write(output, report)


if __name__ == "__main__":
    driver = load_sibling("run_sbml_math_block")
    driver.__file__ = __file__
    driver.inputs, driver.reference, driver.native = cases, reference, native
    raise SystemExit(driver.main())
