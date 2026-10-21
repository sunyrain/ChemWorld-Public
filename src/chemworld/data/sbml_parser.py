"""Closed SBML L3V2 declaration grammar, separate from live model execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from xml.etree import ElementTree as ET

from chemworld.data.xml_inputs import bounded_xml
from chemworld.physchem.mathml import MATHML_NS, SBML_NS, MathExpression

_SID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_REAL_LITERAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_METADATA = {"id": str, "name": str, "metaid": str, "sboTerm": str}
_SID_FIELDS = {"id", "compartment", "substanceUnits", "timeUnits", "volumeUnits", "areaUnits",
               "lengthUnits", "extentUnits", "conversionFactor", "units", "symbol", "variable"}
_LISTS = {
    "listOfFunctionDefinitions": "functionDefinition", "listOfUnitDefinitions": "unitDefinition",
    "listOfUnits": "unit", "listOfCompartments": "compartment", "listOfSpecies": "species",
    "listOfParameters": "parameter", "listOfInitialAssignments": "initialAssignment",
    "listOfRules": ("assignmentRule", "rateRule"), "listOfConstraints": "constraint",
    "listOfReactions": "reaction", "listOfReactants": "speciesReference",
    "listOfProducts": "speciesReference", "listOfModifiers": "modifierSpeciesReference",
    "listOfLocalParameters": "localParameter", "listOfEvents": "event",
    "listOfEventAssignments": "eventAssignment",
}
_MODEL_LISTS = ("listOfFunctionDefinitions", "listOfUnitDefinitions", "listOfCompartments",
    "listOfSpecies", "listOfParameters", "listOfInitialAssignments", "listOfRules",
    "listOfConstraints", "listOfReactions", "listOfEvents")
# allowed attributes, required attributes, allowed children, required children
_SCHEMA = {
    "model": ({"id": str, **dict.fromkeys(("substanceUnits", "timeUnits", "volumeUnits",
        "areaUnits", "lengthUnits", "extentUnits", "conversionFactor"), str)},
        (), _MODEL_LISTS, ()),
    "functionDefinition": ({"id": str}, ("id",), ("math",), ("math",)),
    "unitDefinition": ({"id": str}, ("id",), ("listOfUnits",), ("listOfUnits",)),
    "unit": ({"kind": str, "exponent": float, "scale": int, "multiplier": float},
             ("kind", "exponent", "scale", "multiplier"), (), ()),
    "compartment": ({"id": str, "spatialDimensions": float, "size": float, "units": str,
                     "constant": bool}, ("id", "constant"), (), ()),
    "species": ({"id": str, "compartment": str, "initialAmount": float,
        "initialConcentration": float, "substanceUnits": str, "hasOnlySubstanceUnits": bool,
        "boundaryCondition": bool, "constant": bool, "conversionFactor": str},
        ("id", "compartment", "hasOnlySubstanceUnits", "boundaryCondition", "constant"), (), ()),
    "parameter": ({"id": str, "value": float, "units": str, "constant": bool},
                  ("id", "constant"), (), ()),
    "initialAssignment": ({"symbol": str}, ("symbol",), ("math",), ("math",)),
    "assignmentRule": ({"variable": str}, ("variable",), ("math",), ("math",)),
    "rateRule": ({"variable": str}, ("variable",), ("math",), ("math",)),
    "constraint": ({}, (), ("math", "message"), ("math",)),
    "reaction": ({"id": str, "reversible": bool, "fast": bool, "compartment": str},
        ("reversible",), ("listOfReactants", "listOfProducts", "listOfModifiers", "kineticLaw"),
        ("kineticLaw",)),
    "speciesReference": ({"id": str, "species": str, "stoichiometry": float, "constant": bool},
        ("species", "constant"), (), ()),
    "modifierSpeciesReference": ({"id": str, "species": str}, ("species",), (), ()),
    "kineticLaw": ({}, (), ("math", "listOfLocalParameters"), ("math",)),
    "localParameter": ({"id": str, "value": float, "units": str}, ("id", "value"), (), ()),
    "event": ({"id": str, "useValuesFromTriggerTime": bool}, ("useValuesFromTriggerTime",),
        ("trigger", "delay", "priority", "listOfEventAssignments"), ("trigger",)),
    "trigger": ({"initialValue": bool, "persistent": bool}, ("initialValue", "persistent"),
        ("math",), ("math",)),
    "delay": ({}, (), ("math",), ("math",)),
    "priority": ({}, (), ("math",), ("math",)),
    "eventAssignment": ({"variable": str}, ("variable",), ("math",), ("math",)),
}


@dataclass(frozen=True)
class SBMLRecord:
    kind: str
    attributes: tuple[tuple[str, str | float | bool], ...]
    children: tuple[SBMLRecord, ...] = ()
    mathematics: MathExpression | None = None
    metadata_xml: tuple[str, ...] = ()

    def get(self, name, default=None):
        return dict(self.attributes).get(name, default)

    def of(self, kind):
        return tuple(child for child in self.children if child.kind == kind)

    def items(self, list_name):
        groups = self.of(list_name)
        return groups[0].children if groups else ()


def _kind(element):
    prefix = "{"+SBML_NS+"}"
    if not element.tag.startswith(prefix):
        raise ValueError("unsupported SBML namespace or extension outside annotation")
    return element.tag[len(prefix):]


def _value(text, expected, field):
    try:
        if expected is bool:
            if text not in {"true", "false", "1", "0"}:
                raise ValueError("invalid Boolean")
            return text in {"true", "1"}
        if expected in {int, float}:
            text = text.strip()
            if expected is int and not re.fullmatch(r"[+-]?[0-9]+", text):
                raise ValueError("invalid integer")
            if expected is float and not _REAL_LITERAL.fullmatch(text):
                raise ValueError("invalid XML floating point number")
            number = expected(text)
            if not isfinite(number):
                raise ValueError("nonfinite numeric attribute")
            return number
        if field in _SID_FIELDS and not _SID.fullmatch(text):
            raise ValueError("invalid SId")
        return text
    except (OverflowError, ValueError) as exc:
        raise ValueError("invalid SBML attribute "+field) from exc


def _record(element):
    kind = _kind(element)
    if kind not in _SCHEMA and kind not in _LISTS:
        raise ValueError("unsupported SBML declaration "+kind)
    if (element.text or "").strip() or any((child.tail or "").strip() for child in element):
        raise ValueError("unexpected text in SBML declaration "+kind)
    if kind in _LISTS:
        attrs, required, child_kinds, required_children = {}, (), _LISTS[kind], ()
        child_kinds = (child_kinds,) if isinstance(child_kinds, str) else child_kinds
    else:
        attrs, required, child_kinds, required_children = _SCHEMA[kind]
    attrs = {**_METADATA, **attrs}
    if set(element.attrib)-attrs.keys() or set(required)-element.attrib.keys():
        raise ValueError("unknown or missing SBML attributes on "+kind)
    attributes = tuple((key, _value(value, attrs[key], key))
                       for key, value in element.attrib.items())
    children, seen, metadata, mathematics = [], set(), [], None
    for child in element:
        child_kind = "math" if child.tag == "{"+MATHML_NS+"}math" else _kind(child)
        if child_kind in {"notes", "annotation", "message"}:
            if child_kind == "message" and kind != "constraint":
                raise ValueError("message only belongs to an SBML constraint")
            if child_kind in seen:
                raise ValueError("duplicate SBML metadata "+child_kind)
            metadata.append(ET.tostring(child, encoding="unicode"))
        elif child_kind not in child_kinds:
            raise ValueError("unsupported SBML child "+child_kind+" in "+kind)
        elif kind not in _LISTS and child_kind in seen:
            raise ValueError("duplicate SBML child "+child_kind)
        elif child_kind == "math":
            mathematics = MathExpression(ET.tostring(child, encoding="unicode"))
        else:
            children.append(_record(child))
        seen.add(child_kind)
    if set(required_children)-seen or (kind in _LISTS and not children):
        raise ValueError("missing required SBML children in "+kind)
    result = SBMLRecord(kind, attributes, tuple(children), mathematics, tuple(metadata))
    if kind == "reaction" and result.get("fast", False):
        raise ValueError("SBML fast reaction semantics are not supported")
    if kind == "speciesReference" and not result.get("constant"):
        raise ValueError("dynamic SBML stoichiometry requires its explicit model")
    if kind == "species" and {"initialAmount", "initialConcentration"} <= dict(attributes).keys():
        raise ValueError("SBML species cannot declare both initial amount and concentration")
    if mathematics is not None and (mathematics._root.operation == "lambda") != (
            kind == "functionDefinition"):
        raise ValueError("SBML lambda is required only in functionDefinition")
    return result


def walk_records(root):
    yield root
    for child in root.children:
        yield from walk_records(child)


def acyclic_order(graph, *, label):
    active, ordered, done = set(), [], set()

    def visit(node):
        if node in active:
            raise ValueError("cyclic SBML "+label+" dependency at "+node)
        if node in done:
            return
        active.add(node)
        for other in sorted(graph[node]):
            if other in graph:
                visit(other)
        active.remove(node)
        done.add(node)
        ordered.append(node)

    try:
        for node in graph:
            visit(node)
    except RecursionError as exc:
        raise ValueError("SBML dependency depth exceeds the supported finite model") from exc
    return tuple(ordered)


def parse_sbml(text):
    root = bounded_xml(text)
    if _kind(root) != "sbml" or root.get("level") != "3" or root.get("version") != "2":
        raise ValueError("SBML input must be Level 3 Version 2 Core")
    if set(root.attrib) != {"level", "version"}:
        raise ValueError("unsupported SBML package or root attributes")
    if (root.text or "").strip() or any((child.tail or "").strip() for child in root):
        raise ValueError("unexpected text outside SBML model")
    if len(root) != 1 or _kind(root[0]) != "model":
        raise ValueError("SBML requires exactly one model")
    model = _record(root[0])
    metaids = [r.get("metaid") for r in walk_records(model) if r.get("metaid") is not None]
    if len(metaids) != len(set(metaids)):
        raise ValueError("duplicate SBML metaid")
    return model
