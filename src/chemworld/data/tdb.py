"""Declarative CALPHAD TDB input with provenance and explicit consumer limits.

Parsing is not phase-model execution. Physical consumers must request their
supported parameter types and phase options instead of silently dropping data.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import inf, isfinite
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeVar, overload

from chemworld.data.scientific_inputs import InputSource, _keys
from chemworld.physchem.tdb_expressions import (
    TDB_GAS_CONSTANT_J_MOL_K,
    TdbExpression,
    TdbScalar,
    scalar_value,
)

TDB_SCHEMA_VERSION = "chemworld-tdb-1"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?"
_COMMANDS = (
    "ELEMENT", "SPECIES", "PHASE", "CONSTITUENT", "FUNCTION", "PARAMETER",
    "TYPE_DEFINITION", "TEMPERATURE_LIMITS", "DEFINE_SYSTEM_DEFAULT", "DEFAULT_COMMAND",
    "DATABASE_INFORMATION", "ASSESSED_SYSTEMS", "REFERENCE_FILE", "LIST_OF_REFERENCES",
    "ADD_REFERENCES", "VERSION_DATA", "VERSION_DATE", "DIFFUSION", "ZEROVOLUME_SPECIES",
)
_METADATA = set(_COMMANDS) - {
    "ELEMENT", "SPECIES", "PHASE", "CONSTITUENT", "FUNCTION", "PARAMETER",
    "TYPE_DEFINITION", "TEMPERATURE_LIMITS",
}
_PARAMETER_TYPES = frozenset((
    "G", "L", "TC", "NT", "BMAGN", "GD", "THETA", "V0", "VA", "VC", "VK",
    "VISC", "ELRS", "THCD", "SIGM", "XI", "MQ", "MF", "DQ", "DF", "VS",
))
_PHASE_OPTIONS = frozenset("YFBLGAI")
_T = TypeVar("_T")
_ConstituentArray = tuple[tuple[str, ...], ...]
_PhaseRow = tuple[tuple[float, ...], frozenset[str], str]
AuxiliaryBalances = tuple[tuple[str, tuple[tuple[_ConstituentArray, float], ...]], ...]


class ElementDeclaration(Protocol):
    """Read-only element fields shared by TDB and ChemSage source records."""

    @property
    def name(self) -> str: ...

    @property
    def molar_mass_g_mol(self) -> float: ...

    @property
    def reference_phase(self) -> str | None: ...

    @property
    def enthalpy_298_J_mol(self) -> float | None: ...  # noqa: N802 -- declared unit-bearing field

    @property
    def entropy_298_J_mol_K(self) -> float | None: ...  # noqa: N802 -- declared unit-bearing field


class TdbParseError(ValueError):
    """An invalid complete input; no partially usable database is returned."""


def _expand(candidate: str, choices: Collection[str]) -> str:
    if candidate in choices:
        return candidate
    pieces = candidate.replace("-", "_").upper().split("_")
    matches = [choice for choice in choices
               if len(choice.split("_")) == len(pieces)
               and all(full.startswith(part) for full, part
                       in zip(choice.split("_"), pieces, strict=True))]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous TDB keyword {candidate}: {matches}")
    return matches[0]


def _number(text: str) -> float:
    if not re.fullmatch(_NUMBER, text):
        raise ValueError(f"invalid TDB real number: {text}")
    result = float(text)
    if not isfinite(result):
        raise ValueError("TDB numbers must be finite")
    return result


def _name(text: str) -> str:
    if not re.fullmatch(r"[A-Z0-9_+*/.\-]+", text) or not text:
        raise ValueError(f"invalid TDB identifier: {text}")
    return text


def _phase_name(text: str) -> tuple[str, frozenset[str]]:
    pieces = text.split(":")
    if len(pieces) > 2:
        raise ValueError("invalid TDB phase option suffix")
    name = _name(pieces[0])
    options = frozenset(pieces[1]) if len(pieces) == 2 else frozenset()
    if options - _PHASE_OPTIONS:
        raise ValueError(f"unknown TDB phase options: {sorted(options - _PHASE_OPTIONS)}")
    return name, options


def _commands(text: str) -> Iterator[tuple[int, str]]:
    # TDB has one command terminator per physical line; trailing text is a
    # comment even if it resembles another command (commercial/pycalphad rule).
    pending, start, count = [], None, 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("$", 1)[0].strip()
        if not line:
            continue
        if start is None:
            start = line_number
        pending.append(line.split("!", 1)[0])
        if "!" in line:
            count += 1
            if count > 100000:
                raise TdbParseError("TDB exceeds 100000 commands")
            command = " ".join(pending).strip().upper()
            if command:
                yield start, command
            pending, start = [], None
    if pending:
        raise TdbParseError(f"TDB line {start}: missing command terminator !")


@dataclass(frozen=True)
class TdbSegment:
    lower_K: float
    upper_K: float
    expression: TdbExpression


@dataclass(frozen=True)
class TdbFunction:
    name: str
    segments: tuple[TdbSegment, ...]
    reference: str = ""

    def segment(self, temperature_K: TdbScalar, *, extrapolation: str = "reject") -> TdbSegment:
        if extrapolation not in {"reject", "extend"}:
            raise ValueError("TDB extrapolation must be reject or extend")
        temperature = scalar_value(temperature_K)
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("TDB temperature must be finite and positive")
        for segment in self.segments:
            if segment.lower_K <= temperature < segment.upper_K:
                return segment
        if extrapolation == "extend":
            return self.segments[0] if temperature < self.segments[0].lower_K else self.segments[-1]
        raise ValueError(f"TDB {self.name}: temperature {temperature} K outside declared intervals")


def _function(name: str, text: str) -> TdbFunction:
    parts = text.split(";")
    if len(parts) < 2:
        raise ValueError("TDB function requires a temperature expression and ; limit N")
    first = re.fullmatch(rf"\s*({_NUMBER}|,+)\s+(.+)", parts[0])
    if first is None:
        raise ValueError("TDB function requires a lower temperature limit")
    lower = 0.01 if first[1].startswith(",") else _number(first[1])
    expression = first[2]
    segments, reference = [], ""
    for index, part in enumerate(parts[1:], 1):
        ending = re.fullmatch(rf"\s*,*\s*({_NUMBER})?\s*([YN])\b\s*(.*)", part)
        if ending is None:
            raise ValueError("TDB function requires Y or terminal N after each interval")
        upper = inf if ending[1] is None else _number(ending[1])
        if lower < 0 or lower >= upper:
            raise ValueError("TDB temperature intervals must be nonnegative and increasing")
        segments.append(TdbSegment(lower, upper, TdbExpression(expression)))
        if ending[2] == "N":
            if index != len(parts) - 1:
                raise ValueError("TDB expression after terminal N")
            reference = ending[3].strip()
            if reference and not re.fullmatch(r"[A-Z0-9:_\-]+", reference):
                raise ValueError("invalid TDB reference key")
            break
        if index == len(parts) - 1 or upper == inf or not ending[3].strip():
            raise ValueError("TDB Y interval requires another expression and limit")
        lower, expression = upper, ending[3]
    return TdbFunction(name, tuple(segments), reference)


@dataclass(frozen=True)
class TdbElement:
    name: str
    reference_phase: str
    molar_mass_g_mol: float
    enthalpy_298_J_mol: float
    entropy_298_J_mol_K: float


@dataclass(frozen=True)
class TdbSpecies:
    name: str
    constituents: tuple[tuple[str, float], ...]
    charge: float = 0

    @property
    def number_of_atoms(self) -> float:
        return sum(amount for element, amount in self.constituents if element not in {"VA", "/-"})


@dataclass(frozen=True)
class TdbPhase:
    name: str
    site_ratios: tuple[float, ...]
    constituents: tuple[tuple[str, ...], ...]
    options: frozenset[str]
    type_characters: str
    model_hints: Mapping[str, object]


@dataclass(frozen=True)
class TdbParameter:
    kind: str
    phase: str
    constituents: tuple[tuple[str, ...], ...]
    order: int
    function: TdbFunction
    diffusing_species: str | None = None
    exponents: tuple[int, ...] = ()
    source_index: int | None = None


@dataclass(frozen=True)
class TdbTypeDefinition:
    character: str
    text: str
    feature: str


def _formula(name: str, formula: str) -> TdbSpecies:
    pieces = formula.split("/")
    if len(pieces) > 2 or (len(pieces) == 2 and not re.fullmatch(r"[+-]?\d+", pieces[1])):
        raise ValueError("TDB species charge must be an integer after /")
    atoms: dict[str, float] = {}
    offset = 0
    for match in re.finditer(rf"([A-Z]{{1,2}})({_NUMBER})?", pieces[0]):
        if match.start() != offset:
            raise ValueError(f"invalid TDB species formula: {formula}")
        element, amount = match[1], _number(match[2]) if match[2] else 1.0
        if amount <= 0 or element in atoms:
            raise ValueError("TDB formula needs unique elements and positive stoichiometry")
        atoms[element], offset = amount, match.end()
    if not atoms or offset != len(pieces[0]):
        raise ValueError(f"invalid TDB species formula: {formula}")
    return TdbSpecies(name, tuple(sorted(atoms.items())), int(pieces[1]) if len(pieces) == 2 else 0)


def _constituents(text: str) -> _ConstituentArray:
    result = []
    for sublattice in text.split(":"):
        species = tuple(sorted(_name(s.rstrip("%")) for s in re.split(r"[,\s]+", sublattice.strip())
                               if s))
        if (not species or len(species) != len(set(species))
                or ("*" in species and len(species) > 1)):
            raise ValueError("TDB sublattice requires unique species or a single wildcard")
        result.append(species)
    return tuple(result)


@dataclass(frozen=True)
class TdbDatabase:
    source: InputSource
    text: str
    elements: Mapping[str, ElementDeclaration]
    species: Mapping[str, TdbSpecies]
    phases: Mapping[str, TdbPhase]
    functions: Mapping[str, TdbFunction]
    parameters: tuple[TdbParameter, ...]
    type_definitions: tuple[TdbTypeDefinition, ...]
    metadata: tuple[tuple[str, str], ...]

    @classmethod
    def from_string(cls, text: str, *, source: InputSource) -> TdbDatabase:
        if not isinstance(source, InputSource):
            raise ValueError("TDB requires the existing explicit InputSource provenance")
        if not isinstance(text, str) or not text.strip() or len(text) > 20 * 1024 * 1024:
            raise ValueError("TDB requires nonempty text up to 20 MiB")
        return _parse(text, source)

    @classmethod
    def from_file(cls, path: str | PathLike[str], *, source: InputSource) -> TdbDatabase:
        file_path = Path(path)
        if file_path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("TDB exceeds 20 MiB")
        return cls.from_string(file_path.read_text(encoding="utf-8-sig"), source=source)

    def to_dict(self) -> dict[str, Any]:
        # The original declaration is the single serialized input; parsing it
        # again validates every association instead of trusting parallel caches.
        return {"schema_version": TDB_SCHEMA_VERSION, "source": self.source.to_dict(),
                "text": self.text}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TdbDatabase:
        _keys(payload, ("schema_version", "source", "text"))
        if payload["schema_version"] != TDB_SCHEMA_VERSION:
            raise ValueError("unsupported TDB schema version")
        return cls.from_string(payload["text"], source=InputSource.from_dict(payload["source"]))

    def _evaluate(
        self, function: TdbFunction, temperature_K: TdbScalar, pressure_Pa: TdbScalar,
        extrapolation: str, overrides: Mapping[str, float] | None,
    ) -> TdbScalar:
        if not isfinite(scalar_value(pressure_Pa)) or scalar_value(pressure_Pa) <= 0:
            raise ValueError("TDB pressure must be finite and positive")
        values: dict[str, TdbScalar] = {
            "T": temperature_K, "P": pressure_Pa, "R": TDB_GAS_CONSTANT_J_MOL_K,
        }
        overrides = {} if overrides is None else dict(overrides)
        if overrides.keys() - self.functions.keys():
            raise ValueError("TDB overrides must name declared functions")
        if any(type(v) not in (int, float) or not isfinite(v) for v in overrides.values()):
            raise ValueError("TDB function overrides require finite real scalars")
        cache: dict[str, TdbScalar] = dict(overrides)

        def evaluate(curve: TdbFunction) -> TdbScalar:
            segment = curve.segment(temperature_K, extrapolation=extrapolation)
            resolved = dict(values)
            for name in segment.expression.symbols:
                if name in values:
                    continue
                if name not in cache:
                    cache[name] = evaluate(self.functions[name])
                resolved[name] = cache[name]
            return segment.expression.evaluate(resolved)

        return evaluate(function)

    @overload
    def evaluate_function(
        self, name: str, temperature_K: float, pressure_Pa: float = 101325.0, *,
        extrapolation: str = "reject", overrides: Mapping[str, float] | None = None,
    ) -> float: ...

    @overload
    def evaluate_function(
        self, name: str, temperature_K: TdbScalar, pressure_Pa: TdbScalar = 101325.0, *,
        extrapolation: str = "reject", overrides: Mapping[str, float] | None = None,
    ) -> TdbScalar: ...

    def evaluate_function(
        self, name: str, temperature_K: TdbScalar, pressure_Pa: TdbScalar = 101325.0, *,
        extrapolation: str = "reject", overrides: Mapping[str, float] | None = None,
    ) -> TdbScalar:
        name = name.upper()
        if name not in self.functions:
            raise ValueError(f"unknown TDB function {name}")
        if overrides and name in overrides:
            # Validate overrides through the same path even for a root override.
            self._evaluate(TdbFunction("override", (TdbSegment(0, inf, TdbExpression(name)),)),
                           temperature_K, pressure_Pa, extrapolation, overrides)
            return overrides[name]
        return self._evaluate(self.functions[name], temperature_K, pressure_Pa,
                              extrapolation, overrides)

    @overload
    def evaluate_parameter(
        self, parameter: TdbParameter, temperature_K: float, pressure_Pa: float = 101325.0, *,
        extrapolation: str = "reject", overrides: Mapping[str, float] | None = None,
    ) -> float: ...

    @overload
    def evaluate_parameter(
        self, parameter: TdbParameter, temperature_K: TdbScalar,
        pressure_Pa: TdbScalar = 101325.0, *,
        extrapolation: str = "reject", overrides: Mapping[str, float] | None = None,
    ) -> TdbScalar: ...

    def evaluate_parameter(
        self, parameter: TdbParameter, temperature_K: TdbScalar,
        pressure_Pa: TdbScalar = 101325.0, *,
        extrapolation: str = "reject", overrides: Mapping[str, float] | None = None,
    ) -> TdbScalar:
        if parameter not in self.parameters:
            raise ValueError("TDB parameter does not belong to this database")
        return self._evaluate(parameter.function, temperature_K, pressure_Pa,
                              extrapolation, overrides)

    def consumer_diagnostics(
        self, *, parameter_types: Collection[str], phase_options: Collection[str],
        type_features: Collection[str], metadata_commands: Collection[str] = (),
    ) -> tuple[str, ...]:
        """List every unsupported declaration; parsing never implies consumption."""
        return (
            *(f"unconsumed parameter type {kind}" for kind in sorted(
                {p.kind for p in self.parameters} - set(parameter_types))),
            *(f"unconsumed phase option {option}" for option in sorted(
                {o for p in self.phases.values() for o in p.options} - set(phase_options))),
            *(f"unconsumed type definition {d.character}: {d.text}"
              for d in self.type_definitions if d.feature not in set(type_features)),
            *(f"undefined type character {character}" for character in sorted(
                {c for p in self.phases.values() for c in p.type_characters}
                - {d.character for d in self.type_definitions})),
            *(f"nonexecuted database command {kind}: {body}" for kind, body in self.metadata
              if kind in {"DEFAULT_COMMAND", "DEFINE_SYSTEM_DEFAULT", "DIFFUSION",
                          "ZEROVOLUME_SPECIES"} and kind not in set(metadata_commands)),
        )

    def summary(self) -> dict[str, object]:
        return {"elements": len(self.elements), "species": len(self.species),
                "phases": len(self.phases), "functions": len(self.functions),
                "parameters": len(self.parameters),
                "parameter_types": dict(sorted(Counter(p.kind for p in self.parameters).items())),
                "phase_options": sorted({o for p in self.phases.values() for o in p.options}),
                "source_id": self.source.provenance.source_id}


def _parse(text: str, source: InputSource) -> TdbDatabase:
    elements: dict[str, TdbElement] = {}
    species: dict[str, TdbSpecies] = {}
    phases: dict[str, _PhaseRow] = {}
    constituents: dict[str, _ConstituentArray] = {}
    functions: dict[str, TdbFunction] = {}
    parameters: list[TdbParameter] = []
    typedefs: list[TdbTypeDefinition] = []
    metadata: list[tuple[str, str]] = []

    def insert(mapping: dict[str, _T], key: str, value: _T) -> None:
        if key in mapping:
            raise ValueError(f"duplicate TDB declaration {key}")
        mapping[key] = value

    for line, command in _commands(text):
        try:
            words = command.split(None, 1)
            kind = _expand(words[0], _COMMANDS)
            body = words[1] if len(words) == 2 else ""
            tokens = body.split()
            if kind in _METADATA:
                metadata.append((kind, body))
            elif kind == "TEMPERATURE_LIMITS":
                if len(tokens) != 2 or not 0 <= _number(tokens[0]) < _number(tokens[1]):
                    raise ValueError("TDB temperature limits require two increasing values")
                metadata.append((kind, body))
            elif kind == "ELEMENT":
                if len(tokens) != 5:
                    raise ValueError("TDB ELEMENT needs name, reference phase, mass, H298 and S298")
                name = _name(tokens[0])
                mass, h, s = map(_number, tokens[2:])
                if mass < 0:
                    raise ValueError("TDB element mass cannot be negative")
                insert(elements, name, TdbElement(name, tokens[1], mass, h, s))
                insert(species, name, TdbSpecies(name, ((name, 1.0),)))
            elif kind == "SPECIES":
                if len(tokens) != 2:
                    raise ValueError("TDB SPECIES needs name and formula")
                insert(species, _name(tokens[0]), _formula(tokens[0], tokens[1]))
            elif kind == "FUNCTION":
                if len(tokens) < 2:
                    raise ValueError("TDB FUNCTION needs name and temperature expression")
                name, expression = body.split(None, 1)
                TdbExpression(name)  # A function name must be an admissible arithmetic symbol.
                if name in {"T", "P", "R"}:
                    raise ValueError("TDB function cannot shadow T, P or R")
                insert(functions, _name(name), _function(name, expression))
            elif kind == "PHASE":
                if len(tokens) < 4:
                    raise ValueError("TDB PHASE needs name, type characters and site ratios")
                name, options = _phase_name(tokens[0])
                if not re.fullmatch(r"\d+", tokens[2]):
                    raise ValueError("TDB sublattice count must be a positive integer")
                count = int(tokens[2])
                ratios = tuple(map(_number, tokens[3:]))
                if not 1 <= count <= 64 or len(ratios) != count or any(r <= 0 for r in ratios):
                    raise ValueError("TDB PHASE needs 1-64 positive declared site ratios")
                insert(phases, name, (ratios, options, tokens[1]))
            elif kind == "CONSTITUENT":
                match = re.fullmatch(r"(\S+)\s+:\s*(.*?)\s*:\s*", body)
                if match is None:
                    raise ValueError("TDB CONSTITUENT needs : delimited sublattices")
                name, _ = _phase_name(match[1])
                insert(constituents, name, _constituents(match[2]))
            elif kind == "PARAMETER":
                match = re.fullmatch(r"([A-Z][A-Z0-9_]*)\(([^,()]+),([^()]+)\)\s+(.+)", body)
                if match is None:
                    raise ValueError("invalid TDB PARAMETER header")
                param_type, phase_text, array_order, expression = match.groups()
                param_type = _expand(param_type, _PARAMETER_TYPES)
                phase_diffusion = phase_text.split("&")
                if len(phase_diffusion) > 2:
                    raise ValueError("invalid TDB diffusing species")
                name, _ = _phase_name(phase_diffusion[0].strip())
                order_parts = array_order.split(";")
                if len(order_parts) > 2 or (len(order_parts) == 2
                        and not re.fullmatch(r"\s*\d+\s*", order_parts[1])):
                    raise ValueError("TDB parameter order must be a nonnegative integer")
                array = _constituents(order_parts[0])
                order = int(order_parts[1]) if len(order_parts) == 2 else 0
                diffusion = _name(phase_diffusion[1]) if len(phase_diffusion) == 2 else None
                parameters.append(TdbParameter(param_type, name, array, order,
                                  _function(f"{param_type}({name},{array};{order})", expression),
                                  diffusion))
            elif kind == "TYPE_DEFINITION":
                if len(tokens) < 2 or len(tokens[0]) != 1:
                    raise ValueError("TDB TYPE_DEFINITION requires one character and description")
                definition = body.split(None, 1)[1]
                feature = "unsupported"
                if tokens[1:] == ["SEQ", "*"]:
                    feature = "sequence"
                elif len(tokens) >= 5 and not {"IF", "THEN"}.intersection(tokens):
                    with suppress(ValueError):
                        feature = _expand(
                            tokens[4], ("MAGNETIC", "DISORDERED_PART", "NEVER_DISORDER"),
                        )
                typedefs.append(TdbTypeDefinition(tokens[0], definition, feature))
        except (ValueError, IndexError) as exc:
            raise TdbParseError(f"TDB line {line}: {exc}") from exc

    try:
        return _finish(source, text, elements, species, phases, constituents, functions,
                       parameters, typedefs, metadata)
    except (ValueError, KeyError, RecursionError) as exc:
        raise TdbParseError(f"TDB associations: {exc}") from exc


def _finish(
    source: InputSource, text: str, elements: dict[str, TdbElement],
    species: dict[str, TdbSpecies], phase_rows: Mapping[str, _PhaseRow],
    constituents: Mapping[str, _ConstituentArray], functions: dict[str, TdbFunction],
    parameters: Sequence[TdbParameter], typedefs: Sequence[TdbTypeDefinition],
    metadata: Sequence[tuple[str, str]],
) -> TdbDatabase:
    if set(phase_rows) != set(constituents):
        raise ValueError("TDB phases and constituent declarations must match")
    for spec in species.values():
        if {el for el, _ in spec.constituents} - elements.keys():
            raise ValueError(f"TDB species {spec.name} uses undeclared elements")
    hints: dict[str, dict[str, object]] = {name: {} for name in phase_rows}
    seen_types = set()
    for definition in typedefs:
        if definition.character in seen_types:
            raise ValueError(f"duplicate TDB type character {definition.character}")
        seen_types.add(definition.character)
        words = definition.text.replace(",", "").split()
        matching = [name for name, row in phase_rows.items() if definition.character in row[2]]
        if definition.feature == "MAGNETIC":
            if len(words) != 6:
                raise ValueError("TDB magnetic definition needs AFM and structure factors")
            afm, structure = _number(words[4]), _number(words[5])
            if afm > 0 or not 0 < structure < 1:
                raise ValueError("TDB magnetic factors need AFM<=0 and 0<structure<1")
            for name in matching:
                hints[name].update(magnetic_afm_factor=afm, magnetic_structure_factor=structure)
        elif definition.feature in {"DISORDERED_PART", "NEVER_DISORDER"}:
            if len(words) != 5:
                raise ValueError("TDB order/disorder definition needs two phases")
            ordered, disordered = words[2], words[4]
            if ordered == disordered or ordered not in hints or disordered not in hints:
                raise ValueError("TDB order/disorder requires two declared distinct phases")
            for name in (ordered, disordered):
                hints[name].update(ordered_phase=ordered, disordered_phase=disordered,
                                   never_disorder=definition.feature == "NEVER_DISORDER")
    phases = {}
    for name, (ratios, options, chars) in phase_rows.items():
        array = constituents[name]
        if len(ratios) != len(array) or any(set(subl) - species.keys() for subl in array):
            raise ValueError(f"TDB phase {name} has undeclared species or wrong sublattice count")
        if any("*" in subl for subl in array):
            raise ValueError("TDB wildcards are parameters, not phase constituents")
        if "Y" in options and len(ratios) != 2:
            raise ValueError("TDB ionic-liquid model requires two sublattices")
        phases[name] = TdbPhase(name, ratios, array, options, chars, MappingProxyType(hints[name]))
    seen_parameters = set()
    for parameter in parameters:
        phase = phases.get(parameter.phase)
        if phase is None:
            raise ValueError(f"TDB parameter uses undeclared phase {parameter.phase}")
        array = parameter.constituents
        valid_ionic_short = "Y" in phase.options and len(array) == 1
        targets = (phase.constituents[1],) if valid_ionic_short else phase.constituents
        if len(array) != len(targets) or any(
            set(subl) - set(target) and subl != ("*",)
            for subl, target in zip(array, targets, strict=True)
        ):
            raise ValueError(f"TDB parameter has invalid constituents in {parameter.phase}")
        if valid_ionic_short and any(species[s].charge != 0 for s in array[0] if s != "*"):
            raise ValueError("TDB single-sublattice ionic parameters require neutral species")
        if parameter.diffusing_species and parameter.diffusing_species not in species:
            raise ValueError("TDB parameter uses undeclared diffusing species")
        key = (parameter.kind, parameter.phase, array, parameter.order, parameter.diffusing_species)
        if key in seen_parameters:
            raise ValueError(f"duplicate TDB parameter {key}")
        seen_parameters.add(key)
    # Validate every branch, not only the branch used by the first evaluation.
    dependencies = {}
    for name, function in functions.items():
        deps = {symbol for segment in function.segments for symbol in segment.expression.symbols}
        dependencies[name] = deps - {"T", "P", "R"}
    for curve in [*functions.values(), *(p.function for p in parameters)]:
        missing = {s for seg in curve.segments for s in seg.expression.symbols}
        missing -= {"T", "P", "R"} | functions.keys()
        if missing:
            raise ValueError(f"unresolved TDB symbols in {curve.name}: {sorted(missing)}")
    visited = set()

    def visit(name: str, ancestors: tuple[str, ...] = ()) -> None:
        if name in ancestors or len(ancestors) >= 64:
            raise ValueError(
                f"cyclic or excessively deep TDB function references: {(*ancestors, name)}",
            )
        if name in visited:
            return
        for dependency in dependencies[name]:
            visit(dependency, (*ancestors, name))
        visited.add(name)

    for name in functions:
        visit(name)
    return TdbDatabase(source, text, MappingProxyType(elements), MappingProxyType(species),
                       MappingProxyType(phases), MappingProxyType(functions), tuple(parameters),
                       tuple(typedefs), tuple(metadata))
