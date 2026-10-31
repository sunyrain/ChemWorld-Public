"""Apply finite SED-ML model preprocessing to original immutable SBML inputs."""

import hashlib
from collections.abc import Callable, Container, Mapping
from math import isfinite
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import cast
from xml.etree import ElementTree as ET

from chemworld.data.sbml import SBMLModel
from chemworld.data.sbml_parser import acyclic_order
from chemworld.data.scientific_inputs import InputSource
from chemworld.data.sedml_declarations import SEDMLDocument, finite_number, mathematics, records
from chemworld.data.sedml_xml import ExperimentXMLNode, closed_xpath, local_archive_path
from chemworld.data.xml_inputs import bounded_xml
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_kinetics import SBMLKinetics

_LANGUAGES = {"urn:sedml:language:sbml", "urn:sedml:language:sbml.level-3.version-2"}
_NUMERIC_TARGETS = {"parameter": {"value"}, "localParameter": {"value"},
                    "compartment": {"size"}, "species": {"initialAmount", "initialConcentration"}}
_SBML = "{http://www.sbml.org/sbml/level3/version2/core}"


def _model_target(
    tree: ET.Element, target: str, namespaces: Mapping[str, str],
    *, require_attribute: bool = False,
) -> tuple[ET.Element, str | None]:
    node, attribute = closed_xpath(tree, target, namespaces, require_attribute=require_attribute)
    model = cast(ET.Element, tree.find(_SBML+"model"))
    known: list[ET.Element] = []
    for group in ("listOfParameters", "listOfCompartments", "listOfSpecies", "listOfReactions"):
        container = model.find(_SBML+group)
        if container is not None:
            known.extend(n for n in container if n.tag in {
                _SBML+"parameter", _SBML+"compartment", _SBML+"species", _SBML+"reaction"})
    reactions = model.find(_SBML+"listOfReactions")
    for reaction in (() if reactions is None else reactions):
        law = reaction.find(_SBML+"kineticLaw/"+_SBML+"listOfLocalParameters")
        if law is not None:
            known.extend(n for n in law if n.tag == _SBML+"localParameter")
    if node not in known:
        raise ValueError("experiment target is not an actual declared core model entity")
    return node, attribute


def model_reference(source: str, identities: Container[str]) -> str | None:
    if source.startswith("#"):
        if source[1:] not in identities:
            raise ValueError("experiment inherits an unknown model identity")
        return source[1:]
    return source if source in identities else None


def source_quantity(model: SBMLModel, target: str, namespaces: Mapping[str, str]) -> float:
    """Read the specified source attribute or the model's initialized symbol basis."""
    tree = bounded_xml(model.to_xml())
    node, attribute = _model_target(tree, target, namespaces)
    if attribute is not None:
        return finite_number(node.get(attribute))
    if node.tag.rsplit("}", 1)[-1] not in {"species", "compartment", "parameter", "reaction"}:
        raise ValueError("model variable does not select a supported initialized quantity")
    identity = node.get("id")
    frame = SBMLKinetics(model)
    context = frame.context(frame.initial_values, 0.)
    if identity not in context:
        raise ValueError("model variable does not select a known initialized symbol")
    return context[cast(str, identity)]


def calculation(
    node: ExperimentXMLNode, *, variable_value: Callable[[ExperimentXMLNode], float],
    extra: Mapping[str, float] | None = None,
) -> float:
    """One shared scalar MathExpression with explicit lexical input bindings."""
    scope = {} if extra is None else dict(extra)
    for parameter in records(node, "listOfParameters"):
        scope[parameter.attributes["id"]] = finite_number(parameter.attributes["value"])
    for variable in records(node, "listOfVariables"):
        scope[variable.attributes["id"]] = variable_value(variable)
    expression = mathematics(node)
    if set(expression.symbols)-scope.keys():
        raise ValueError("experiment calculation has undeclared mathematical symbols")
    result = expression.evaluate(scope)
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not isfinite(result):
        raise ValueError("experiment calculation must produce a finite numeric scalar")
    return float(result)


def _derived(model: SBMLModel, xml: str, document: SEDMLDocument, identity: str) -> SBMLModel:
    parent = model.source.provenance
    provenance = DataSourceProvenance(parent.source_id+":sedml:"+identity, parent.priority,
        parent.citation+"; SED-ML preprocessing from "+document.source.provenance.source_id,
        parent.version, parent.license, hashlib.sha256(xml.encode("utf-8")).hexdigest())
    return SBMLModel(xml, InputSource(provenance, model.source.data_kind),
                     dict(model.unit_defaults), model.items_per_mole)


def _numeric_target(tree: ET.Element, change: ExperimentXMLNode) -> tuple[ET.Element, str]:
    node, attribute = _model_target(tree, change.attributes["target"], change.namespaces,
                                    require_attribute=True)
    if attribute not in _NUMERIC_TARGETS.get(node.tag.rsplit("}", 1)[-1], set()):
        raise ValueError("experiment preprocessing only changes declared numeric model inputs")
    return node, cast(str, attribute)


def resolve_models(
    document: SEDMLDocument, resources: Mapping[str, SBMLModel], *, document_location: str = "",
) -> Mapping[str, SBMLModel]:
    """Resolve all model/compute dependencies before any native experiment begins.

    Resources are explicitly supplied SBMLModel objects. No filesystem/network
    loader or arbitrary XPath implementation is reachable from source strings.
    """
    if not isinstance(document, SEDMLDocument) or not isinstance(resources, Mapping):
        raise ValueError("model preprocessing needs declared experiment and local model inputs")
    base = ""
    if document_location:
        location = local_archive_path(document_location)
        base = str(PurePosixPath(location).parent)
        if base == ".":
            base = ""
    supplied: dict[str, SBMLModel] = {}
    for location, model in resources.items():
        path = local_archive_path(location)
        if (path.casefold() in {key.casefold() for key in supplied}
                or not isinstance(model, SBMLModel)):
            raise ValueError("duplicate local model aliases or non-SBML resources")
        supplied[path] = model
    declarations = {node.attributes["id"]: node for node in records(document.declaration,
                                                                    "listOfModels")}
    if not 1 <= len(declarations) <= 128:
        raise ValueError("experiment model count must stay within the declared finite budget")
    dependencies: dict[str, set[str]] = {}
    parents: dict[str, str | None] = {}
    for identity, node in declarations.items():
        if node.attributes["language"] not in _LANGUAGES:
            raise ValueError("experiment model language/version is unsupported")
        parent = model_reference(node.attributes["source"], declarations)
        parents[identity] = parent
        dependencies[identity] = {parent} if parent is not None else set()
        if (parent is None
                and local_archive_path(node.attributes["source"], base=base) not in supplied):
            raise ValueError("experiment model source is not an explicitly supplied local resource")
        for change in records(node, "listOfChanges"):
            for variable in records(change, "listOfVariables"):
                reference = variable.attributes.get("modelReference")
                if (reference not in declarations or "target" not in variable.attributes
                        or set(variable.attributes) & {"taskReference", "symbol"}):
                    raise ValueError("model preprocessing variables need explicit model targets")
                if reference != identity:
                    dependencies[identity].add(cast(str, reference))
    resolved: dict[str, SBMLModel] = {}
    order = acyclic_order(dependencies, label="SED-ML model preprocessing")
    for identity in order:
        declaration = declarations[identity]
        parent = parents[identity]
        current = (resolved[parent] if parent is not None else
                   supplied[local_archive_path(declaration.attributes["source"], base=base)])
        for change in records(declaration, "listOfChanges"):
            tree = bounded_xml(current.to_xml())
            target, attribute = _numeric_target(tree, change)
            if change.local_name == "changeAttribute":
                value = finite_number(change.attributes["newValue"])
            else:
                def variable_value(
                    variable: ExperimentXMLNode, current: SBMLModel = current,
                    identity: str = identity,
                ) -> float:
                    reference = variable.attributes["modelReference"]
                    source = current if reference == identity else resolved[reference]
                    return source_quantity(source, variable.attributes["target"],
                                           variable.namespaces)

                value = calculation(change, variable_value=variable_value)
            target.set(attribute, format(value, ".17g"))
            # SBML's own closed declaration validator checks every derived input.
            current = _derived(current, ET.tostring(tree, encoding="unicode"), document, identity)
        resolved[identity] = current
    return MappingProxyType(resolved)
