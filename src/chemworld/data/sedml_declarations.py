"""Closed SED-ML L1V4 declaration profile, without a simulator or file loader."""

import re
from dataclasses import dataclass, field
from math import isfinite
from xml.etree import ElementTree as ET

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.sedml_xml import ExperimentXMLNode, experiment_xml
from chemworld.physchem.mathml import MathExpression

SEDML_NAMESPACE = "http://sed-ml.org/sed-ml/level1/version4"
MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML"
SEDML_INPUT_FORMAT = "chemworld_sedml_l1v4_input_v1"
_SID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_BASE = {"id", "name", "metaid"}
# required attributes, optional attributes, child elements, required children
_RULES = {
    "sedML": ("level version", "", "listOfModels listOfSimulations listOfTasks "
        "listOfDataGenerators listOfOutputs listOfAlgorithmParameters", ""),
    "model": ("id language source", "", "listOfChanges", ""),
    "changeAttribute": ("target newValue", "", "", ""),
    "computeChange": ("target", "", "listOfVariables listOfParameters math", "math"),
    "uniformTimeCourse": ("id initialTime outputStartTime outputEndTime",
        "numberOfSteps numberOfPoints", "algorithm", "algorithm"),
    "algorithm": ("kisaoID", "", "listOfAlgorithmParameters", ""),
    "algorithmParameter": ("kisaoID value", "", "", ""),
    "task": ("id modelReference simulationReference", "", "", ""),
    "repeatedTask": ("id range resetModel", "concatenate",
        "listOfRanges listOfChanges listOfSubTasks", "listOfRanges listOfSubTasks"),
    "setValue": ("target modelReference", "range", "listOfVariables listOfParameters math", "math"),
    "subTask": ("task", "order", "listOfChanges", ""),
    "uniformRange": ("id start end type", "numberOfSteps numberOfPoints", "", ""),
    "vectorRange": ("id", "", "value", "value"),
    "value": ("", "", "", ""),
    "functionalRange": ("id range", "", "listOfVariables listOfParameters math", "math"),
    "variable": ("id", "target symbol taskReference modelReference", "", ""),
    "parameter": ("id value", "", "", ""),
    "dataGenerator": ("id", "", "listOfVariables listOfParameters math", "math"),
    "report": ("id", "", "listOfDataSets", "listOfDataSets"),
    "dataSet": ("id label dataReference", "", "", ""),
    "listOfModels": ("", "", "model", "model"),
    "listOfSimulations": ("", "", "uniformTimeCourse", "uniformTimeCourse"),
    "listOfTasks": ("", "", "task repeatedTask", ""),
    "listOfChanges": ("", "", "changeAttribute computeChange setValue", ""),
    "listOfVariables": ("", "", "variable", "variable"),
    "listOfParameters": ("", "", "parameter", "parameter"),
    "listOfAlgorithmParameters": ("", "", "algorithmParameter", "algorithmParameter"),
    "listOfRanges": ("", "", "uniformRange vectorRange functionalRange", ""),
    "listOfSubTasks": ("", "", "subTask", "subTask"),
    "listOfDataGenerators": ("", "", "dataGenerator", "dataGenerator"),
    "listOfOutputs": ("", "", "report", "report"),
    "listOfDataSets": ("", "", "dataSet", "dataSet"),
}
_CHILD_REPEAT = {key for key in _RULES if key.startswith("listOf")} | {"vectorRange"}


def finite_number(value):
    if not isinstance(value, str) or not _NUMBER.fullmatch(value) or not isfinite(float(value)):
        raise ValueError("experiment numeric literals must be finite ASCII numbers")
    return float(value)


def integer(value, *, minimum=None):
    if not isinstance(value, str) or not _INTEGER.fullmatch(value):
        raise ValueError("experiment count/order must be an ASCII integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError("experiment count is below its supported minimum")
    return result


def boolean(value):
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("experiment Boolean must use XML Boolean syntax")
    return value in {"true", "1"}


def records(node, container):
    groups = [n for n in node.children if n.tag == "{"+SEDML_NAMESPACE+"}"+container]
    return () if not groups else tuple(n for n in groups[0].children
        if n.tag not in {"{"+SEDML_NAMESPACE+"}notes", "{"+SEDML_NAMESPACE+"}annotation"})


def mathematics(node):
    matches = [n for n in node.children if n.tag == "{"+MATHML_NAMESPACE+"}math"]
    if len(matches) != 1:
        raise ValueError("experiment calculation requires one MathML expression")
    result = MathExpression(ET.tostring(matches[0].element(), encoding="unicode"))
    if result.function_names or result.function_arguments is not None or result.number_units:
        raise ValueError("experiment calculations do not load functions or SBML number-unit syntax")
    return result


def step_count(attributes):
    fields = set(attributes) & {"numberOfSteps", "numberOfPoints"}
    if len(fields) != 1:
        raise ValueError("supply exactly one of numberOfSteps or deprecated numberOfPoints")
    return integer(attributes[next(iter(fields))], minimum=1)


def _validate(node):
    name, attributes = node.local_name, node.attributes
    if node.tag != "{"+SEDML_NAMESPACE+"}"+name or name not in _RULES:
        raise ValueError("unknown or unsupported experiment element "+node.tag)
    required, optional, children, mandatory = (set(v.split()) for v in _RULES[name])
    allowed = required | optional | _BASE
    if name == "sedML":
        allowed.add("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
    if set(attributes)-allowed or required-set(attributes):
        raise ValueError("unknown or missing experiment attributes for "+name)
    for field_name in ("id", "modelReference", "simulationReference", "taskReference",
                       "dataReference", "task", "range"):
        if field_name in attributes and not _SID.fullmatch(attributes[field_name]):
            raise ValueError("invalid experiment SId/reference")
    if name in {"uniformTimeCourse", "uniformRange"}:
        step_count(attributes)
    for field_name in ("initialTime", "outputStartTime", "outputEndTime", "start", "end"):
        if field_name in attributes:
            finite_number(attributes[field_name])
    if name == "value":
        finite_number(node.text.strip())
    elif node.text.strip():
        raise ValueError("unexpected text in experiment element "+name)
    if name == "parameter":
        finite_number(attributes["value"])
    if name == "uniformTimeCourse" and not (
            finite_number(attributes["initialTime"]) <= finite_number(attributes["outputStartTime"])
            <= finite_number(attributes["outputEndTime"])):
        raise ValueError("experiment output grid must follow its initial source time")
    if name == "uniformRange" and attributes["type"] not in {"linear", "log"}:
        raise ValueError("unknown experiment range spacing")
    if name in {"model", "repeatedTask", "subTask"}:
        changes = records(node, "listOfChanges")
        allowed_changes = {"changeAttribute", "computeChange"} if name == "model" else {"setValue"}
        if any(change.local_name not in allowed_changes for change in changes):
            raise ValueError("experiment changes differ from their model/task context")
    for key in ("resetModel", "concatenate"):
        if key in attributes:
            boolean(attributes[key])
    if name == "subTask" and "order" in attributes:
        integer(attributes["order"])
    if name == "variable" and not (set(attributes) & {"symbol", "target"}):
        raise ValueError("experiment variable needs an explicit symbol or target")
    counts = {}
    for child in node.children:
        if child.tag in {"{"+SEDML_NAMESPACE+"}notes", "{"+SEDML_NAMESPACE+"}annotation"}:
            child_name = child.local_name
        else:
            child_name = "math" if child.tag == "{"+MATHML_NAMESPACE+"}math" else child.local_name
            if child_name not in children:
                raise ValueError("unsupported child of experiment "+name)
            if child_name == "math":
                mathematics(node)
            else:
                _validate(child)
        counts[child_name] = counts.get(child_name, 0)+1
        if counts[child_name] > 1 and (name not in _CHILD_REPEAT
                                      or child_name in {"notes", "annotation"}):
            raise ValueError("duplicate experiment child "+child_name)
        if child.tail.strip():
            raise ValueError("unexpected experiment mixed content")
    if mandatory-set(counts):
        raise ValueError("missing required experiment children of "+name)
    if name == "report":
        labels = [n.attributes["label"] for n in records(node, "listOfDataSets")]
        if len(set(labels)) != len(labels):
            raise ValueError("report dataset labels must be unambiguous")


def _unique_identifiers(root):
    seen, metadata = set(), set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node.tag in {"{"+SEDML_NAMESPACE+"}notes", "{"+SEDML_NAMESPACE+"}annotation"}:
            continue
        if node.tag.startswith("{"+SEDML_NAMESPACE+"}"):
            for field_name, known in (("id", seen), ("metaid", metadata)):
                value = node.attributes.get(field_name)
                if value is not None:
                    if value in known:
                        raise ValueError("experiment "+field_name+" must be globally unique")
                    known.add(value)
        pending.extend(node.children)


@dataclass(frozen=True)
class SEDMLDocument:
    xml: str
    source: InputSource
    _declaration: ExperimentXMLNode = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, InputSource):
            raise ValueError("experiment input requires explicit source provenance")
        declaration = experiment_xml(self.xml)
        if (declaration.tag != "{"+SEDML_NAMESPACE+"}sedML"
                or declaration.attributes.get("level") != "1"
                or declaration.attributes.get("version") != "4"):
            raise ValueError("only the declared SED-ML Level 1 Version 4 profile is supported")
        _validate(declaration)
        _unique_identifiers(declaration)
        object.__setattr__(self, "_declaration", declaration)

    @property
    def declaration(self):
        return self._declaration

    def to_xml(self):
        return self.xml

    def to_dict(self):
        return {"format": SEDML_INPUT_FORMAT, "xml": self.xml, "source": self.source.to_dict()}

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {"format", "xml", "source"}
                or payload["format"] != SEDML_INPUT_FORMAT):
            raise ValueError("unknown experiment input fields or format")
        return cls(payload["xml"], InputSource.from_dict(payload["source"]))
