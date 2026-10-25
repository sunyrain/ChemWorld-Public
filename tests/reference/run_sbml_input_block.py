"""Twelve complete authored SBML inputs, independent SDK fields and SI units."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import time
import traceback
from pathlib import Path
from xml.etree import ElementTree as ET

CORE = "http://www.sbml.org/sbml/level3/version2/core"
MATH = "http://www.w3.org/1998/Math/MathML"
_FLOAT_FIELDS = {"value", "size", "spatialDimensions", "initialAmount", "initialConcentration",
                 "exponent", "multiplier", "stoichiometry"}
_BOOL_FIELDS = {"constant", "boundaryCondition", "hasOnlySubstanceUnits", "initialValue",
                "persistent", "useValuesFromTriggerTime", "reversible", "fast"}


def source_xml(root):
    """Keep historical fixture bytes without leaking serializer state to consumers."""
    namespaces = dict(ET._namespace_map)
    try:
        ET.register_namespace("", CORE)
        return ET.tostring(root, encoding="unicode")
    finally:
        ET._namespace_map.clear()
        ET._namespace_map.update(namespaces)


def cases():
    template = Path(__file__).parents[1]/"test_sbml.py"
    assignments = [n for n in ast.parse(template.read_text(encoding="utf-8")).body
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MODEL" for t in n.targets)]
    text = ast.literal_eval(assignments[0].value)
    for index in range(12):
        root = ET.fromstring(text)
        model = root[0]
        model.set("id", f"source_{index}")

        def select(kind, identity, model=model):
            return next(e for e in model.iter("{"+CORE+"}"+kind) if e.get("id") == identity)

        if index == 1:
            species = select("species", "A")
            species.attrib.pop("initialConcentration")
            species.set("initialAmount", "0.01")
            species.set("hasOnlySubstanceUnits", "true")
        elif index == 2:
            select("species", "B").set("boundaryCondition", "true")
        elif index == 3:
            select("species", "A").set("boundaryCondition", "true")
            select("species", "A").set("constant", "true")
            for assignment in model.iter("{"+CORE+"}eventAssignment"):
                if assignment.get("variable") == "A":
                    assignment.set("variable", "B")
        elif index == 4:
            select("compartment", "c1").set("constant", "true")
        elif index == 5:
            definitions = model.find("{"+CORE+"}listOfUnitDefinitions")
            for name, kind, scale, multiplier in (("mmol", "mole", -3, 1),
                    ("ml", "litre", -3, 1), ("minute", "second", 0, 60)):
                definition = ET.SubElement(definitions, "{"+CORE+"}unitDefinition", id=name)
                units = ET.SubElement(definition, "{"+CORE+"}listOfUnits")
                ET.SubElement(units, "{"+CORE+"}unit", kind=kind, exponent="1",
                              scale=str(scale), multiplier=str(multiplier))
            model.set("timeUnits", "minute")
            model.set("substanceUnits", "mmol")
            model.set("extentUnits", "mmol")
            model.set("volumeUnits", "ml")
            for compartment in model.iter("{"+CORE+"}compartment"):
                compartment.set("units", "ml")
            for name in ("per_second", "molar_rate"):
                for term in select("unitDefinition", name).iter("{"+CORE+"}unit"):
                    if term.get("kind") == "second":
                        term.set("multiplier", "60")
                    else:
                        term.set("scale", "-3")
        elif index == 6:
            law = model.find(".//{"+CORE+"}kineticLaw")
            law.remove(law.find("{"+CORE+"}listOfLocalParameters"))
        elif index == 7:
            event = select("event", "switch")
            event.set("useValuesFromTriggerTime", "false")
            event.find("{"+CORE+"}trigger").set("initialValue", "true")
            event.find("{"+CORE+"}trigger").set("persistent", "true")
        elif index == 8:
            event = select("event", "switch")
            for tag in ("delay", "priority"):
                event.remove(event.find("{"+CORE+"}"+tag))
        elif index == 9:
            model.find(".//{http://www.w3.org/1999/xhtml}p").text = (
                "Authored \u4e2d\u6587 \u03b1 & <tag>; preserve provenance.")
            model.find(".//{urn:chemworld:test}source").set("code", "\u4e2d\u6587 \u03b1 & <tag>")
        elif index == 10:
            select("parameter", "rp").set("value", "0.3")
            select("species", "A").set("sboTerm", "SBO:0000247")
            assignment = model.find(".//{"+CORE+"}initialAssignment")
            assignment.find(".//{"+MATH+"}cn").text = "9"
        elif index == 11:
            select("compartment", "c1").attrib.pop("size")
            assignments = model.find("{"+CORE+"}listOfInitialAssignments")
            initial = ET.SubElement(assignments, "{"+CORE+"}initialAssignment", symbol="c1")
            expression = ET.SubElement(initial, "{"+MATH+"}math")
            ET.SubElement(expression, "{"+MATH+"}cn").text = "0.015"
        yield f"input-{index:02}.xml", str(index), None, source_xml(root)


def typed_attribute(name, value):
    if name in _FLOAT_FIELDS:
        return float(value)
    if name == "scale":
        return int(value)
    if name in _BOOL_FIELDS:
        if value not in {"true", "false", "0", "1"}:
            raise ValueError("reference produced invalid Boolean attribute")
        return value in {"true", "1"}
    return value


def canonical_xml(text):
    def walk(node):
        tag = node.tag.rsplit("}", 1)[-1]
        attrs = dict(node.attrib)
        if tag == "cn":
            kind = attrs.pop("type", "real")
            first = (node.text or "").strip()
            if kind == "rational":
                number = float(first)/float(node[0].tail.strip())
            elif kind == "e-notation":
                number = float(first+"e"+node[0].tail.strip())
            else:
                number = float(first)
            return ("number", attrs, number)
        return (node.tag, attrs, " ".join((node.text or "").split()),
                tuple((walk(c), " ".join((c.tail or "").split())) for c in node))
    return walk(ET.fromstring(text))


def xml_records(text):
    rows = []

    def walk(node):
        row = {"kind": node.tag.rsplit("}", 1)[-1],
               "attributes": {k: typed_attribute(k, v) for k, v in node.attrib.items()},
               "mathematics": None, "metadata": []}
        rows.append(row)
        for child in node:
            kind = child.tag.rsplit("}", 1)[-1]
            if kind == "math":
                row["mathematics"] = canonical_xml(ET.tostring(child, encoding="unicode"))
            elif kind in {"notes", "annotation", "message"}:
                row["metadata"].append(canonical_xml(ET.tostring(child, encoding="unicode")))
            else:
                walk(child)
    walk(ET.fromstring(text)[0])
    return rows


def reference(directory, output):
    import COPASI
    import libsbml

    report = {"expected_inputs": 12, "units": [],
              "libsbml_version": libsbml.getLibSBMLDottedVersion(),
              "COPASI_version": COPASI.CVersion.VERSION.getVersion()}
    started = time.perf_counter()
    for name, _, _, _ in cases():
        text = (directory/name).read_text(encoding="utf-8")
        unit = {"input": name, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "failures": []}
        report["units"].append(unit)
        dm = None
        try:
            doc = libsbml.readSBMLFromString(text)
            doc.checkConsistency()
            messages = [doc.getError(i) for i in range(doc.getNumErrors())]
            unit["libsbml_messages"] = [m.getMessage() for m in messages]
            if any(m.getSeverity() >= libsbml.LIBSBML_SEV_ERROR for m in messages):
                raise ValueError("independent SBML validation failed")
            exported = libsbml.writeSBMLToString(doc)
            (output.parent/(name+".libsbml.xml")).write_text(exported, encoding="utf-8")
            unit["records"] = xml_records(exported)
            if unit["records"] != xml_records(text):
                raise ValueError("SDK roundtrip changed original declaration fields")
            model = doc.getModel()
            # Independent SI definitions in order mass,length,time,amount,current,temperature.
            bases = {"mole": (1., (0, 0, 0, 1, 0, 0)),
                     "second": (1., (0, 0, 1, 0, 0, 0)),
                     "litre": (.001, (0, 3, 0, 0, 0, 0))}
            signatures = dict(bases)
            unit["declared_unit_signatures"] = {}
            for index in range(model.getNumUnitDefinitions()):
                definition = model.getUnitDefinition(index)
                factor, powers = 1., [0.]*6
                for j in range(definition.getNumUnits()):
                    term = definition.getUnit(j)
                    base, vector = bases[libsbml.UnitKind_toString(term.getKind())]
                    exponent = term.getExponent()
                    factor *= (base*term.getMultiplier()*10.**term.getScale())**exponent
                    powers = [a+exponent*b for a, b in zip(powers, vector, strict=True)]
                signatures[definition.getId()] = factor, tuple(powers)
                unit["declared_unit_signatures"][definition.getId()] = {
                    "si_factor": factor, "dimension": powers}
            unit["model_unit_signatures"] = {field: {"si_factor": signatures[getattr(model,
                "get"+field[0].upper()+field[1:])()][0], "dimension": signatures[getattr(model,
                "get"+field[0].upper()+field[1:])()][1]}
                for field in ("substanceUnits", "extentUnits", "timeUnits", "volumeUnits")}
            dm = COPASI.CRootContainer.addDatamodel()
            if not dm.importSBMLFromString(text) or not dm.getModel().compileIfNecessary():
                raise ValueError("actual COPASI import/compile failed")
            unit["COPASI_imported_and_compiled"] = True
        except Exception:
            unit["failures"].append(traceback.format_exc())
        finally:
            if dm is not None:
                COPASI.CRootContainer.removeDatamodel(dm)
        driver.write(output, report)
        n = len(report["units"])
        elapsed = max(time.perf_counter()-started, 1e-9)
        print(f"reference sources {n}/12 rate={n/elapsed:.2f}/s eta={(12-n)*elapsed/n:.1f}s",
              flush=True)


def native(directory, reference_report, output):
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    source = InputSource(DataSourceProvenance("synthetic-sbml-source-12", 0,
        "Self-authored complete declaration and source unit block", "1", "CC0-1.0"), "synthetic")
    report = {"expected_inputs": 12, "units": [], "executed_inputs": 0, "passed_inputs": 0}
    for name, _, _, _ in cases():
        unit = {"input": name, "passed": False, "failures": []}
        report["units"].append(unit)
        try:
            ref = next(u for u in reference_report["units"] if u["input"] == name)
            if ref["failures"]:
                raise ValueError("independent source validation failed: "+repr(ref["failures"]))
            text = (directory/name).read_text(encoding="utf-8")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != ref["sha256"]:
                raise ValueError("source changed after SDK qualification")
            model = SBMLModel(text, source)
            clone = SBMLModel.from_dict(model.to_dict())
            assert clone.to_xml() == text and clone.to_dict() == model.to_dict()
            rows = []

            def walk(record, rows=rows):
                rows.append({"kind": record.kind, "attributes": dict(record.attributes),
                    "mathematics": canonical_xml(record.mathematics.xml)
                    if record.mathematics else None,
                    "metadata": [canonical_xml(x) for x in record.metadata_xml]})
                for child in record.children:
                    walk(child)

            walk(model.declaration)
            unit["records"] = rows
            report["executed_inputs"] += 1
            # JSON normalizes immutable tuples only; no field is discarded.
            if driver.json.loads(driver.json.dumps(rows)) != ref["records"]:
                raise ValueError("complete native declarations differ from SDK records")
            unit["unit_comparisons"] = []
            for group, expected in (("declared_unit_signatures", ref["declared_unit_signatures"]),
                                     ("model_unit_signatures", ref["model_unit_signatures"])):
                for key, signature in expected.items():
                    actual = model.unit_signature(key) if group.startswith("declared") else (
                        model.model_unit(key))
                    vector = actual.dimension
                    axes = [vector.mass, vector.length, vector.time, vector.amount,
                            vector.current, vector.temperature]
                    error = abs(actual.si_factor-signature["si_factor"])
                    tolerance = 1e-13+1e-12*abs(signature["si_factor"])
                    good = axes == signature["dimension"] and error <= tolerance
                    unit["unit_comparisons"].append({"group": group, "name": key,
                        "actual": actual.si_factor, "reference": signature["si_factor"],
                        "actual_dimensions": axes, "reference_dimensions": signature["dimension"],
                        "si_factor_error": error, "tolerance": tolerance, "passed": good})
                    if not good:
                        raise ValueError("independent SI source unit differs: "+key)
            unit["passed"] = True
        except Exception:
            unit["failures"].append(traceback.format_exc())
        driver.write(output, report)
        print(f"native sources {len(report['units'])}/12 passed={unit['passed']}", flush=True)
    report["passed_inputs"] = sum(u["passed"] for u in report["units"])
    report["unexecuted_inputs"] = 12-report["executed_inputs"]
    report["complete_and_passed"] = report["passed_inputs"] == 12
    driver.write(output, report)


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("sbml_block_parent",
        Path(__file__).with_name("run_sbml_math_block.py"))
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.__file__ = __file__
    driver.inputs = cases
    driver.reference, driver.native = reference, native
    raise SystemExit(driver.main())
