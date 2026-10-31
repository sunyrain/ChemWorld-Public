"""Original SBML declarations and provenance, with closed native JSON decoding.

This immutable input is configuration, not a second model state or simulator.
Units that are absent in the source remain absent unless the author supplies
explicit defaults; symbolic species never acquire an invented chemical formula.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from chemworld.data.sbml_parser import SBMLRecord, acyclic_order, parse_sbml, walk_records
from chemworld.data.sbml_units import (
    SBML_UNIT_KINDS,
    SBMLUnitSignature,
    SBMLUnitTerm,
    base_unit,
    unit_definition,
)
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.mathml import MathExpression

SBML_INPUT_FORMAT = "chemworld_sbml_l3v2_input_v1"
_DEFAULT_FIELDS = {"substanceUnits", "timeUnits", "volumeUnits", "areaUnits", "lengthUnits",
                   "extentUnits"}
_MATH_ENTITIES = {"compartment", "species", "parameter", "reaction", "speciesReference"}


def _unique(records: Iterable[SBMLRecord], label: str) -> dict[str, SBMLRecord]:
    records = tuple(records)
    identities = [r.get("id") for r in records if r.get("id") is not None]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate SBML "+label+" identifier")
    return {r.get("id"): r for r in records if r.get("id") is not None}


def _validate(model: SBMLRecord) -> tuple[dict[str, SBMLRecord], dict[str, SBMLRecord]]:
    records = tuple(walk_records(model))
    global_records = [r for r in records if r.kind not in {"localParameter", "unitDefinition"}]
    identities = _unique(global_records, "global")
    symbols = {name for name, r in identities.items() if r.kind in _MATH_ENTITIES}
    functions = {r.get("id"): cast(MathExpression, r.mathematics)
                 for r in model.items("listOfFunctionDefinitions")}
    units = _unique(model.items("listOfUnitDefinitions"), "unit")
    if units.keys() & SBML_UNIT_KINDS:
        raise ValueError("SBML unit definitions cannot shadow base unit kinds")
    for unit in units.values():
        for term in unit.items("listOfUnits"):
            SBMLUnitTerm(**{k: term.get(k) for k in ("kind", "exponent", "scale", "multiplier")})
    known_units = units.keys() | SBML_UNIT_KINDS
    reactions = model.items("listOfReactions")
    kinetic_locals = {}
    local: dict[str, SBMLRecord] | set[str]
    for reaction in reactions:
        law = reaction.of("kineticLaw")[0]
        local = _unique(law.items("listOfLocalParameters"), "local parameter")
        kinetic_locals[id(law)] = set(local)
        for group in ("listOfReactants", "listOfProducts", "listOfModifiers"):
            for reference in reaction.items(group):
                target = identities.get(reference.get("species"))
                if target is None or target.kind != "species":
                    raise ValueError("reaction references undefined SBML species")
                if reference.get("stoichiometry", 0.) < 0:
                    raise ValueError("SBML stoichiometry cannot be negative")
    for record in records:
        for attribute in (*_DEFAULT_FIELDS, "units"):
            name = record.get(attribute)
            if name is not None and name not in known_units:
                raise ValueError("undefined SBML unit "+name)
        factor = record.get("conversionFactor")
        if factor is not None:
            target = identities.get(factor)
            if target is None or target.kind != "parameter" or not target.get("constant"):
                raise ValueError("SBML conversionFactor must name a constant global parameter")
        compartment = record.get("compartment")
        if compartment is not None:
            target = identities.get(compartment)
            if target is None or target.kind != "compartment":
                raise ValueError("undefined SBML compartment "+compartment)
        if record.kind == "compartment" and (
                record.get("size", 0.) < 0 or not 0 <= record.get("spatialDimensions", 3.) <= 3):
            raise ValueError("invalid SBML compartment size/spatial dimensions")
        if record.kind == "species" and any(record.get(key, 0.) < 0
                for key in ("initialAmount", "initialConcentration")):
            raise ValueError("initial SBML species quantity cannot be negative")
        math = record.mathematics
        if math is not None:
            local = kinetic_locals.get(id(record), set())
            if set(math.symbols)-symbols-local-{"$time"}:
                raise ValueError("undefined SBML mathematical symbol in "+record.kind)
            if set(math.function_names)-functions.keys():
                raise ValueError("undefined SBML mathematical function")
            if any(len(functions[name].function_arguments) != arity
                   for name, arity in math.function_calls):
                raise ValueError("SBML function call argument count mismatch")
            if set(math.number_units)-known_units:
                raise ValueError("undefined unit on SBML numeric literal")
    acyclic_order({name: set(math.function_names) for name, math in functions.items()},
                  label="function")
    rules = model.items("listOfRules")
    initial = model.items("listOfInitialAssignments")
    rule_targets = [r.get("variable") for r in rules]
    initial_targets = [r.get("symbol") for r in initial]
    if (len(set(rule_targets)) != len(rule_targets)
            or len(set(initial_targets)) != len(initial_targets)):
        raise ValueError("duplicate SBML rule/initial-assignment target")
    assigned = {r.get("variable") for r in rules if r.kind == "assignmentRule"}
    if assigned & set(initial_targets):
        raise ValueError("SBML initial assignment conflicts with assignment rule")
    assignable = {name: r for name, r in identities.items()
                  if r.kind in {"compartment", "species", "parameter", "speciesReference"}}
    for rule in (*rules, *initial):
        target = rule.get("symbol" if rule.kind == "initialAssignment" else "variable")
        if target not in assignable:
            raise ValueError("SBML rule/initial assignment references an invalid target")
        if rule.kind != "initialAssignment" and assignable[target].get("constant"):
            raise ValueError("SBML rule cannot modify a constant target")
        if (rule.kind != "initialAssignment" and assignable[target].kind == "species"
                and not assignable[target].get("boundaryCondition")
                and any(reference.get("species") == target for reaction in reactions
                        for group in ("listOfReactants", "listOfProducts")
                        for reference in reaction.items(group))):
            raise ValueError("SBML species cannot be owned by both a rule and a reaction")
    for event in model.items("listOfEvents"):
        targets = []
        for assignment in event.items("listOfEventAssignments"):
            target = assignment.get("variable")
            targets.append(target)
            if (target not in assignable or assignable[target].get("constant")
                    or target in assigned):
                raise ValueError("SBML event assignment has an invalid or rule-owned target")
        if len(set(targets)) != len(targets):
            raise ValueError("SBML event has duplicate simultaneous assignment targets")
    derived = {r.get("variable"): set(cast(MathExpression, r.mathematics).symbols)
               for r in rules if r.kind == "assignmentRule"}
    derived.update({r.get("id"): set(cast(MathExpression,
                                       r.of("kineticLaw")[0].mathematics).symbols)
                    -kinetic_locals[id(r.of("kineticLaw")[0])]
                    for r in reactions if r.get("id") is not None})
    acyclic_order(derived, label="derived assignment/reaction")
    initialized = {**derived, **{r.get("symbol"): set(cast(MathExpression, r.mathematics).symbols)
                                for r in initial}}
    acyclic_order(initialized, label="initial assignment")
    return identities, units


@dataclass(frozen=True)
class SBMLModel:
    raw_text: str
    source: InputSource
    unit_defaults: Mapping[str, str] = field(default_factory=dict)
    items_per_mole: float | None = None
    declaration: SBMLRecord = field(init=False, repr=False)
    _identities: Mapping[str, SBMLRecord] = field(init=False, repr=False, compare=False)
    _units: Mapping[str, SBMLRecord] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, InputSource):
            raise ValueError("SBML requires explicit input provenance")
        if not isinstance(self.unit_defaults, (dict, MappingProxyType)):
            raise ValueError("SBML unit defaults must be an explicit object")
        if (set(self.unit_defaults)-_DEFAULT_FIELDS
                or any(not isinstance(u, str) for u in self.unit_defaults.values())):
            raise ValueError("unknown SBML unit default or non-string unit")
        if self.items_per_mole is not None:
            base_unit("item", items_per_mole=self.items_per_mole)
        declaration = parse_sbml(self.raw_text)
        identities, units = _validate(declaration)
        if any(u not in units.keys() | SBML_UNIT_KINDS for u in self.unit_defaults.values()):
            raise ValueError("SBML author default references an undefined unit")
        object.__setattr__(self, "unit_defaults", MappingProxyType(dict(self.unit_defaults)))
        object.__setattr__(self, "declaration", declaration)
        object.__setattr__(self, "_identities", MappingProxyType(identities))
        object.__setattr__(self, "_units", MappingProxyType(units))

    @classmethod
    def from_file(
        cls, path: str | PathLike[str], *, source: InputSource,
        unit_defaults: Mapping[str, str] | None = None, items_per_mole: float | None = None,
    ) -> SBMLModel:
        path = Path(path)
        if path.stat().st_size > 20*1024*1024:
            raise ValueError("SBML input exceeds 20 MiB")
        return cls(path.read_text(encoding="utf-8-sig"), source,
                   {} if unit_defaults is None else unit_defaults, items_per_mole)

    def to_xml(self) -> str:
        return self.raw_text

    def to_dict(self) -> dict[str, Any]:
        return {"format": SBML_INPUT_FORMAT, "raw_text": self.raw_text,
                "source": self.source.to_dict(), "unit_defaults": dict(self.unit_defaults),
                "items_per_mole": self.items_per_mole}

    @classmethod
    def from_dict(cls, payload: Any) -> SBMLModel:
        if (not isinstance(payload, dict) or set(payload) != {
                "format", "raw_text", "source", "unit_defaults", "items_per_mole"}
                or payload["format"] != SBML_INPUT_FORMAT):
            raise ValueError("unknown SBML input format or fields")
        return cls(payload["raw_text"], InputSource.from_dict(payload["source"]),
                   payload["unit_defaults"], payload["items_per_mole"])

    def unit_signature(self, name: str) -> SBMLUnitSignature:
        if name not in self._units:
            return base_unit(name, items_per_mole=self.items_per_mole)
        terms = tuple(SBMLUnitTerm(**{key: r.get(key) for key in (
            "kind", "exponent", "scale", "multiplier")})
            for r in self._units[name].items("listOfUnits"))
        return unit_definition(terms, items_per_mole=self.items_per_mole)

    def model_unit(self, attribute: str) -> SBMLUnitSignature:
        if attribute not in _DEFAULT_FIELDS:
            raise ValueError("unknown SBML model unit field")
        name = self.declaration.get(attribute, self.unit_defaults.get(attribute))
        if name is None:
            raise ValueError("missing SBML unit requires an explicit author default: "+attribute)
        return self.unit_signature(name)
