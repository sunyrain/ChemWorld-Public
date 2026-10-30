"""Strict CHEMKIN gas-mechanism subset converted to the native network.

The importer accepts the traditional ``ELEMENTS``/``SPECIES``/``THERMO``/
``REACTIONS`` sections with NASA7 records and elementary, third-body,
Lindemann, or Troe reactions.  It converts into the existing Cantera-shaped
declaration and then uses the same native validation path. Unsupported
continuation directives reject the complete import instead of being ignored.
"""

from __future__ import annotations

import re
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.reaction_network import ReactionNetworkSpec

_SECTION = re.compile(
    r"^\s*(ELEMENTS?|ELEM|SPECIES|THERMO|REACTIONS|TRANSPORT)\b", re.IGNORECASE
)
_INCLUDE = re.compile(r"^\s*INCLUDE\s+(?P<path>.+?)\s*$", re.IGNORECASE)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_ARRHENIUS = re.compile(
    rf"^(?P<equation>.+?)\s+(?P<A>{_NUMBER})\s+(?P<b>{_NUMBER})\s+(?P<Ea>{_NUMBER})\s*$"
)
_REACTION_LABEL = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_.-]*\s*:")
_REACTION_SECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_EFFICIENCY = re.compile(
    rf"(?P<species>[A-Za-z][A-Za-z0-9_().+\-]*)\s*/\s*(?P<value>{_NUMBER})\s*/"
)


def _strip_comment(line: str) -> str:
    return line.split("!", 1)[0].rstrip()


def _include_target(value: str) -> str:
    target = value.strip()
    if len(target) >= 2 and (
        (target[0] == target[-1] and target[0] in "'\"")
        or (target[0] == "<" and target[-1] == ">")
    ):
        target = target[1:-1].strip()
    if not target:
        raise ValueError("CHEMKIN INCLUDE needs a file path")
    return target


def _expand_includes(
    text: str,
    base_dir: Path,
    *,
    stack: tuple[Path, ...] = (),
) -> tuple[str, tuple[str, ...]]:
    """Expand local CHEMKIN INCLUDE directives with cycle detection."""

    expanded: list[str] = []
    included: list[str] = []
    for raw in text.splitlines():
        match = _INCLUDE.match(_strip_comment(raw))
        if match is None:
            expanded.append(raw)
            continue
        target = Path(_include_target(match["path"]))
        path = (target if target.is_absolute() else base_dir / target).resolve()
        if path in stack:
            chain = " -> ".join(str(item) for item in (*stack, path))
            raise ValueError(f"cyclic CHEMKIN INCLUDE: {chain}")
        if not path.is_file():
            raise ValueError(f"CHEMKIN INCLUDE file does not exist: {path}")
        try:
            child = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read CHEMKIN INCLUDE file: {path}") from exc
        child_text, child_includes = _expand_includes(
            child,
            path.parent,
            stack=(*stack, path),
        )
        expanded.extend(child_text.splitlines())
        included.extend((str(path), *child_includes))
    return "\n".join(expanded), tuple(included)


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _sections(
    text: str,
    *,
    require_mechanism: bool = True,
) -> tuple[list[str], list[str], list[str], list[str], str]:
    species: list[str] = []
    thermo: list[str] = []
    reactions: list[str] = []
    transport: list[str] = []
    reaction_header = "REACTIONS"
    current: str | None = None
    seen_sections: set[str] = set()
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        header = _SECTION.match(line)
        if header:
            current = header[1].upper()
            # CHEMKIN gas files have one declaration for each core section.
            # Combining duplicate REACTIONS/THERMO blocks would silently apply
            # one block's units or coefficients to another block, so reject the
            # complete source before any partial native network is produced.
            section_name = (
                "ELEMENTS" if current in {"ELEMENT", "ELEMENTS", "ELEM"} else current
            )
            if section_name in seen_sections:
                raise ValueError(f"CHEMKIN duplicate {section_name} section")
            seen_sections.add(section_name)
            if current == "TRANSPORT" and line.strip().upper() != "TRANSPORT":
                raise ValueError("CHEMKIN TRANSPORT section does not accept arguments")
            if current == "REACTIONS":
                reaction_header = line.strip()
            continue
        if line.strip().upper() == "END":
            current = None
            continue
        if current == "SPECIES":
            species.extend(line.split())
        elif current == "THERMO":
            thermo.append(line)
        elif current == "REACTIONS":
            reactions.append(line)
        elif current == "TRANSPORT":
            transport.append(line)
        elif current in {"ELEMENT", "ELEMENTS", "ELEM"}:
            continue
        else:
            raise ValueError(f"CHEMKIN declaration is outside a supported section: {line}")
    if require_mechanism and (not species or not reactions):
        raise ValueError("CHEMKIN input needs non-empty SPECIES and REACTIONS sections")
    if len(set(species)) != len(species):
        raise ValueError("CHEMKIN SPECIES declarations must be unique")
    return species, thermo, reactions, transport, reaction_header


def _transport_records(lines: list[str], species_names: list[str]) -> list[dict[str, Any]]:
    """Validate and retain the strict seven-field CHEMKIN transport subset.

    A CHEMKIN transport record is ``species geometry epsilon sigma dipole
    polarizability rotational-relaxation``.  The values are retained in their
    source units (K, Angstrom, Debye, Angstrom², and dimensionless) so this
    import does not imply a native transport model or silently discard source
    data.  Vendor extensions and wrapped records fail closed.
    """

    geometry_names = {0: "atom", 1: "linear", 2: "nonlinear"}
    declared = set(species_names)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in lines:
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(
                "CHEMKIN TRANSPORT records need species plus six fields"
            )
        species = fields[0]
        if species not in declared:
            raise ValueError(f"CHEMKIN TRANSPORT species is not declared: {species}")
        if species in seen:
            raise ValueError(f"duplicate CHEMKIN TRANSPORT record: {species}")
        try:
            geometry_value = _number(fields[1])
            values = [_number(value) for value in fields[2:]]
        except ValueError as exc:
            raise ValueError(f"CHEMKIN TRANSPORT record is not numeric: {line}") from exc
        if not isfinite(geometry_value) or not geometry_value.is_integer():
            raise ValueError("CHEMKIN TRANSPORT geometry must be integer 0, 1, or 2")
        geometry = int(geometry_value)
        if geometry not in geometry_names:
            raise ValueError("CHEMKIN TRANSPORT geometry must be 0, 1, or 2")
        if any(not isfinite(value) for value in values):
            raise ValueError("CHEMKIN TRANSPORT values must be finite")
        epsilon, sigma, dipole, polarizability, rotational_relaxation = values
        if epsilon <= 0.0 or sigma <= 0.0:
            raise ValueError("CHEMKIN TRANSPORT epsilon and sigma must be positive")
        if dipole < 0.0 or polarizability < 0.0 or rotational_relaxation < 0.0:
            raise ValueError("CHEMKIN TRANSPORT optional properties must be nonnegative")
        records.append(
            {
                "species": species,
                "geometry": geometry,
                "geometry_name": geometry_names[geometry],
                "epsilon_over_k_K": epsilon,
                "sigma_A": sigma,
                "dipole_D": dipole,
                "polarizability_A3": polarizability,
                "rotational_relaxation": rotational_relaxation,
            }
        )
        seen.add(species)
    return records


def _source_section_provenance(text: str) -> list[dict[str, Any]]:
    """Return machine-readable source section headers and their line positions."""

    provenance: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        match = _SECTION.match(line)
        if match is None:
            continue
        token = match[1].upper()
        name = "ELEMENTS" if token in {"ELEMENT", "ELEMENTS", "ELEM"} else token
        parts = line.split()
        entry = {
            "name": name,
            "line": line_number,
            "header": line,
            "arguments": parts[1:],
        }
        if name == "REACTIONS":
            entry["section_name"] = _reaction_header_details(line)["section_name"]
        provenance.append(entry)
    return provenance


def _fields(line: str, count: int) -> list[float]:
    values: list[float] = []
    for index in range(count):
        field = line[index * 15 : (index + 1) * 15]
        if not field.strip():
            raise ValueError("CHEMKIN NASA7 coefficient field is empty")
        values.append(_number(field))
    return values


def _thermo_records(lines: list[str], species_names: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(lines):
        header = lines[index]
        if len(header) < 75:
            index += 1
            continue
        name = header[:18].strip()
        if not name or name not in species_names:
            index += 1
            continue
        try:
            low, high, middle = (
                _number(header[45:55]),
                _number(header[55:65]),
                _number(header[65:75]),
            )
            high_first = _fields(lines[index + 1], 5)
            high_last_low_first = _fields(lines[index + 2], 5)
            low_last = _fields(lines[index + 3], 4)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid CHEMKIN NASA7 record for {name}") from exc
        if not low < middle < high:
            raise ValueError(f"CHEMKIN NASA7 temperatures must be ordered for {name}")
        composition: dict[str, int] = {}
        for match in re.finditer(r"([A-Z][a-z]?)[ ]*(\d+)", header[24:44]):
            element, amount = match.groups()
            if element in composition:
                raise ValueError(f"duplicate CHEMKIN element in {name}: {element}")
            composition[element] = int(amount)
        if not composition:
            raise ValueError(f"CHEMKIN NASA7 record has no elemental composition: {name}")
        records[name] = {
            "name": name,
            "composition": composition,
            "thermo": {
                "model": "NASA7",
                "temperature-ranges": [low, middle, high],
                "data": [
                    high_last_low_first[2:] + low_last,
                    high_first + high_last_low_first[:2],
                ],
                "reference-pressure": "1 atm",
                "note": "converted from fixed-width CHEMKIN NASA7",
            },
        }
        index += 4
    missing = set(species_names) - set(records)
    if missing:
        raise ValueError("CHEMKIN NASA7 records missing species: " + ", ".join(sorted(missing)))
    return records


_REACTION_ENERGY_UNITS = {
    "CAL/MOLE": "cal",
    "KCAL/MOLE": "kcal",
    "KJOULES/MOLE": "kJ",
    "JOULES/MOLE": "J",
}
_REACTION_QUANTITY_UNITS = {"MOLE", "MOLES", "MOLECULES"}
_REACTION_MW_MODES = {"MWON", "MWOFF"}


def _reaction_header_details(header: str) -> dict[str, Any]:
    """Parse one strict global or single-named REACTIONS header."""

    raw_tokens = " ".join(header.split()).split()
    tokens = [token.upper() for token in raw_tokens]
    if not tokens or tokens[0] != "REACTIONS":
        raise ValueError("CHEMKIN REACTIONS header is malformed")
    arguments = tokens[1:]
    raw_arguments = raw_tokens[1:]
    known_unit_tokens = (
        set(_REACTION_ENERGY_UNITS) | _REACTION_QUANTITY_UNITS | _REACTION_MW_MODES
    )
    section_name: str | None = None
    if arguments and arguments[0] not in known_unit_tokens:
        section_name = raw_arguments.pop(0)
        arguments.pop(0)
        if not _REACTION_SECTION_NAME.fullmatch(section_name):
            raise ValueError(
                f"CHEMKIN unknown REACTIONS header token: {section_name}; "
                "named reaction sections are restricted to one identifier"
            )
    unknown_tokens = [token for token in arguments if token not in known_unit_tokens]
    if unknown_tokens:
        raise ValueError(
            "CHEMKIN unknown REACTIONS header token(s): "
            + ", ".join(unknown_tokens)
            + "; named reaction sections are strict"
        )
    energy_tokens = [token for token in arguments if token in _REACTION_ENERGY_UNITS]
    if not energy_tokens:
        raise ValueError(
            "CHEMKIN REACTIONS must declare supported activation-energy units; "
            "named reaction sections require one energy token"
        )
    if len(energy_tokens) > 1:
        raise ValueError("CHEMKIN REACTIONS cannot repeat or mix activation-energy units")
    quantity_tokens = [token for token in arguments if token in _REACTION_QUANTITY_UNITS]
    if not quantity_tokens:
        raise ValueError(
            "CHEMKIN REACTIONS must declare MOLE, MOLES, or MOLECULES; "
            "named reaction sections require one quantity token"
        )
    if len(quantity_tokens) > 1:
        raise ValueError("CHEMKIN REACTIONS cannot repeat or mix quantity units")
    mw_tokens = [token for token in arguments if token in _REACTION_MW_MODES]
    if len(mw_tokens) > 1:
        raise ValueError("CHEMKIN REACTIONS cannot repeat or mix MW modes")
    energy = _REACTION_ENERGY_UNITS[energy_tokens[0]]
    quantity_token = quantity_tokens[0]
    return {
        "section_name": section_name,
        "molecular_weight_mode": mw_tokens[0] if mw_tokens else None,
        "units": {
            "volume": "cm^3",
            "quantity": "molecule" if quantity_token == "MOLECULES" else "mol",
            "time": "s",
            "energy": energy,
            "activation-energy": f"{energy}/mol",
            "pressure": "atm",
            "temperature": "K",
        },
    }


def _reaction_units(header: str) -> dict[str, str]:
    return _reaction_header_details(header)["units"]


def _normalize_equation(equation: str) -> str:
    result = re.sub(r"\s+", " ", equation.strip())
    result = result.replace("<=>", "<=>").replace("=>", "=>")
    if "<=>" not in result and "=>" not in result:
        result = result.replace("=", "<=>")
    return result


def _irreversible_equation(equation: str) -> tuple[str, str]:
    arrow = "<=>" if "<=>" in equation else "=>" if "=>" in equation else None
    if arrow is None:
        raise ValueError("CHEMKIN REV needs a reaction equation")
    sides = equation.split(arrow)
    if len(sides) != 2 or not all(side.strip() for side in sides):
        raise ValueError("CHEMKIN REV needs one reaction arrow")
    left, right = (side.strip() for side in sides)
    return f"{left} => {right}", f"{right} => {left}"


def _reaction_nodes(lines: list[str], units: dict[str, str]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    reverse_node: dict[str, Any] | None = None
    for line in lines:
        if _REACTION_LABEL.match(line):
            raise ValueError(
                "CHEMKIN reaction labels are unsupported; named reaction sections are unsupported"
            )
        parsed = _ARRHENIUS.match(line)
        if parsed:
            equation = parsed["equation"].strip()
            arrow_positions = [
                position
                for marker in ("<=>", "=>", "=")
                if (position := equation.find(marker)) >= 0
            ]
            if ":" in equation[: min(arrow_positions, default=len(equation))]:
                raise ValueError("CHEMKIN named reaction sections are unsupported")
            current = {
                "equation": _normalize_equation(equation),
                "rate-constant": {
                    "A": _number(parsed["A"]),
                    "b": _number(parsed["b"]),
                    "Ea": _number(parsed["Ea"]),
                },
            }
            nodes.append(current)
            reverse_node = None
            continue
        if current is None:
            raise ValueError("CHEMKIN continuation appears before a reaction")
        upper = line.strip().upper()
        if upper == "DUPLICATE":
            current["duplicate"] = True
            if reverse_node is not None:
                reverse_node["duplicate"] = True
        elif upper.startswith("REV"):
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError("CHEMKIN REV continuation needs slash-delimited A, b, Ea")
            parts = values[1].split()
            if len(parts) != 3:
                raise ValueError("CHEMKIN REV continuation needs A, b, Ea")
            if reverse_node is not None:
                raise ValueError("CHEMKIN reaction cannot declare multiple REV continuations")
            if current.get("type") not in {None, "three-body"}:
                raise ValueError(
                    "CHEMKIN REV is unsupported for pressure-dependent or falloff rates"
                )
            if "orders" in current:
                raise ValueError("CHEMKIN REV cannot be combined with FORD custom orders")
            if "reverse-orders" in current:
                raise ValueError("CHEMKIN REV cannot be combined with RORD custom orders")
            forward_equation, reverse_equation = _irreversible_equation(current["equation"])
            current["equation"] = forward_equation
            try:
                reverse_rate = {
                    "A": _number(parts[0]),
                    "b": _number(parts[1]),
                    "Ea": _number(parts[2]),
                }
            except ValueError as exc:
                raise ValueError("CHEMKIN REV values must be finite numbers") from exc
            reverse_node = dict(current)
            reverse_node.update(
                {
                    "equation": reverse_equation,
                    "rate-constant": reverse_rate,
                    "note": "explicit reverse Arrhenius from CHEMKIN REV",
                }
            )
            if "efficiencies" in current:
                reverse_node["efficiencies"] = dict(current["efficiencies"])
            nodes.append(reverse_node)
        elif upper.startswith("PLOG"):
            if reverse_node is not None:
                raise ValueError("CHEMKIN REV cannot be combined with PLOG")
            if "reverse-orders" in current:
                raise ValueError("CHEMKIN RORD cannot be combined with PLOG")
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError("CHEMKIN PLOG continuation needs slash-delimited P, A, b, Ea")
            parts = values[1].split()
            if len(parts) != 4:
                raise ValueError("CHEMKIN PLOG continuation needs P, A, b, Ea")
            if "M" in current["equation"]:
                raise ValueError("CHEMKIN PLOG with a third-body collider is unsupported")
            try:
                pressure, prefactor, exponent, activation = map(_number, parts)
            except ValueError as exc:
                raise ValueError("CHEMKIN PLOG values must be finite numbers") from exc
            if pressure <= 0:
                raise ValueError("CHEMKIN PLOG pressure must be positive")
            if "rate-constants" not in current:
                current.pop("rate-constant")
            current.setdefault("rate-constants", [])
            current["type"] = "pressure-dependent-Arrhenius"
            current["rate-constants"].append(
                {"P": pressure, "A": prefactor, "b": exponent, "Ea": activation}
            )
        elif upper.startswith("TCHEB") or upper.startswith("PCHEB"):
            if reverse_node is not None:
                raise ValueError("CHEMKIN REV cannot be combined with CHEB")
            if "reverse-orders" in current:
                raise ValueError("CHEMKIN RORD cannot be combined with CHEB")
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError("CHEMKIN CHEB range needs slash-delimited endpoints")
            parts = values[1].split()
            if len(parts) != 2:
                raise ValueError("CHEMKIN CHEB range needs two endpoints")
            try:
                bounds = [_number(value) for value in parts]
            except ValueError as exc:
                raise ValueError("CHEMKIN CHEB range must be numeric") from exc
            if bounds[0] <= 0 or bounds[0] >= bounds[1]:
                raise ValueError("CHEMKIN CHEB range must be positive and ordered")
            current["temperature-range" if upper.startswith("TCHEB") else "pressure-range"] = bounds
        elif upper.startswith("CHEB"):
            if reverse_node is not None:
                raise ValueError("CHEMKIN REV cannot be combined with CHEB")
            if "reverse-orders" in current:
                raise ValueError("CHEMKIN RORD cannot be combined with CHEB")
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError("CHEMKIN CHEB continuation needs slash-delimited values")
            parts = values[1].split()
            try:
                numbers = [_number(value) for value in parts]
            except ValueError as exc:
                raise ValueError("CHEMKIN CHEB values must be numeric") from exc
            shape = current.get("_cheb_shape")
            if shape is None:
                if len(numbers) != 2 or any(
                    value <= 0 or not value.is_integer() for value in numbers
                ):
                    raise ValueError("CHEMKIN CHEB header needs positive integer dimensions")
                if "M" in current["equation"]:
                    raise ValueError("CHEMKIN CHEB with a third-body collider is unsupported")
                current.pop("rate-constant")
                current["type"] = "Chebyshev"
                current["_cheb_shape"] = (int(numbers[0]), int(numbers[1]))
                current["data"] = []
            else:
                rows, columns = shape
                if len(numbers) != columns:
                    raise ValueError("CHEMKIN CHEB coefficient row has the wrong width")
                data = current["data"]
                if len(data) >= rows:
                    raise ValueError("CHEMKIN CHEB has too many coefficient rows")
                data.append(numbers)
        elif upper.startswith("FORD"):
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError(
                    "CHEMKIN FORD continuation needs slash-delimited species and order"
                )
            parts = values[1].split()
            if len(parts) != 2:
                raise ValueError("CHEMKIN FORD continuation needs species and order")
            if "<=>" in current["equation"]:
                raise ValueError("CHEMKIN FORD custom orders require an irreversible reaction")
            if current.get("type") not in {None, "three-body"}:
                raise ValueError(
                    "CHEMKIN FORD is unsupported for pressure-dependent or falloff rates"
                )
            try:
                order = _number(parts[1])
            except ValueError as exc:
                raise ValueError("CHEMKIN FORD order must be finite") from exc
            orders = current.setdefault("orders", {})
            if parts[0] in orders:
                raise ValueError(f"duplicate CHEMKIN FORD order: {parts[0]}")
            orders[parts[0]] = order
            current["nonreactant-orders"] = True
            if order < 0:
                current["negative-orders"] = True
        elif upper.startswith("RORD"):
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError(
                    "CHEMKIN RORD continuation needs slash-delimited species and order"
                )
            parts = values[1].split()
            if len(parts) != 2:
                raise ValueError("CHEMKIN RORD continuation needs species and order")
            if "<=>" not in current["equation"]:
                raise ValueError("CHEMKIN RORD custom orders require a reversible reaction")
            if reverse_node is not None:
                raise ValueError("CHEMKIN RORD cannot be combined with REV")
            if current.get("type") not in {None, "three-body"}:
                raise ValueError(
                    "CHEMKIN RORD is unsupported for pressure-dependent or falloff rates"
                )
            try:
                order = _number(parts[1])
            except ValueError as exc:
                raise ValueError("CHEMKIN RORD order must be finite") from exc
            if order < 0:
                raise ValueError("CHEMKIN RORD order must be nonnegative")
            orders = current.setdefault("reverse-orders", {})
            if parts[0] in orders:
                raise ValueError(f"duplicate CHEMKIN RORD order: {parts[0]}")
            orders[parts[0]] = order
            current["nonreactant-orders"] = True
        elif upper.startswith("LOW"):
            if reverse_node is not None:
                raise ValueError("CHEMKIN REV cannot be combined with falloff")
            if "reverse-orders" in current:
                raise ValueError("CHEMKIN RORD cannot be combined with falloff")
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError("CHEMKIN LOW continuation needs slash-delimited A, b, Ea")
            parts = values[1].split()
            if len(parts) != 3:
                raise ValueError("CHEMKIN LOW continuation needs A, b, Ea")
            current["type"] = "falloff"
            current["high-P-rate-constant"] = current.pop("rate-constant")
            current["low-P-rate-constant"] = {
                "A": _number(parts[0]),
                "b": _number(parts[1]),
                "Ea": _number(parts[2]),
            }
        elif upper.startswith("TROE"):
            if reverse_node is not None:
                raise ValueError("CHEMKIN REV cannot be combined with falloff")
            if "reverse-orders" in current:
                raise ValueError("CHEMKIN RORD cannot be combined with falloff")
            values = re.search(r"/\s*(.*?)\s*/", line)
            if values is None:
                raise ValueError("CHEMKIN TROE continuation needs slash-delimited values")
            parts = values[1].split()
            if len(parts) != 4:
                raise ValueError("CHEMKIN TROE requires A, T1, T2 and T3")
            current["Troe"] = {
                "A": _number(parts[0]),
                "T1": _number(parts[1]),
                "T2": _number(parts[2]),
                "T3": _number(parts[3]),
            }
            current["type"] = "falloff"
        else:
            pairs = list(_EFFICIENCY.finditer(line))
            if not pairs or "M" not in current["equation"]:
                raise ValueError(f"unsupported CHEMKIN continuation: {line}")
            if "type" not in current:
                # CHEMKIN uses a separate efficiency continuation for an
                # elementary third-body reaction.  The Cantera-shaped
                # declaration needs the explicit reaction type once that
                # continuation is present; without it the native validator
                # correctly rejects the otherwise ambiguous ``efficiencies``
                # field.
                current["type"] = "three-body"
            efficiencies = current.setdefault("efficiencies", {})
            for pair in pairs:
                species = pair["species"]
                if species in efficiencies:
                    raise ValueError(f"duplicate CHEMKIN efficiency: {species}")
                efficiencies[species] = _number(pair["value"])
            if reverse_node is not None:
                reverse_node["type"] = current["type"]
                reverse_node["efficiencies"] = dict(efficiencies)
    for node in nodes:
        if node.get("type") != "Chebyshev":
            continue
        shape = node.pop("_cheb_shape", None)
        if shape is None or len(node.get("data", ())) != shape[0]:
            raise ValueError("CHEMKIN CHEB coefficient matrix has incomplete rows")
        if "temperature-range" not in node or "pressure-range" not in node:
            raise ValueError("CHEMKIN CHEB requires TCHEB and PCHEB ranges")
    return nodes


def convert_chemkin(
    mechanism_text: str,
    thermo_text: str | None = None,
    *,
    source: str = "in-memory CHEMKIN input",
) -> ReactionNetworkSpec:
    """Convert one complete CHEMKIN gas mechanism or reject unsupported input."""

    (
        species_names,
        embedded_thermo_lines,
        reaction_lines,
        transport_lines,
        reaction_header,
    ) = _sections(
        mechanism_text
    )
    if thermo_text is None:
        if not embedded_thermo_lines:
            raise ValueError(
                "CHEMKIN input needs a companion NASA7 thermo text or embedded THERMO section"
            )
        thermo_lines = embedded_thermo_lines
        thermo_source = "embedded THERMO section"
    else:
        # Companion exporters may provide a complete THERMO section while
        # callers may provide only fixed-width records.  Add a wrapper only to
        # the latter form so duplicate section headers remain a hard error.
        has_thermo_header = any(
            re.match(r"^\s*THERMO\b", _strip_comment(line), re.IGNORECASE)
            for line in thermo_text.splitlines()
        )
        companion_text = thermo_text if has_thermo_header else "THERMO\n" + thermo_text + "\nEND\n"
        _, thermo_lines, _, _, _ = _sections(companion_text, require_mechanism=False)
        thermo_source = "companion NASA7 text"
    records = _thermo_records(thermo_lines, species_names)
    transport_records = _transport_records(transport_lines, species_names)
    reaction_header_details = _reaction_header_details(reaction_header)
    units = reaction_header_details["units"]
    reaction_nodes = _reaction_nodes(reaction_lines, units)
    payload = {
        "description": "strict CHEMKIN subset converted to native Cantera-shaped declarations",
        "generator": "chemworld.chemkin_import",
        "units": units,
        "phases": [
            {
                "name": "chemkin-gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "species": species_names,
                "reactions": "all",
            }
        ],
        "species": [records[name] for name in species_names],
        "reactions": reaction_nodes,
    }
    network = convert_mechanism(payload, source=source)
    provenance = _source_section_provenance(mechanism_text)
    metadata = dict(network.metadata)
    metadata.update(
        {
            "import_format": "chemkin-gas-mechanism-v1",
            "source_mechanism": source,
            "thermo_source": thermo_source,
            "duplicate_reaction_count": sum(
                bool(reaction.get("duplicate")) for reaction in reaction_nodes
            ),
            "source_quantity_basis": units["quantity"],
            "source_avogadro_per_mol": AVOGADRO if units["quantity"] == "molecule" else None,
            "source_section_provenance": provenance,
            "reaction_section_name": reaction_header_details["section_name"],
            "reaction_molecular_weight_mode": reaction_header_details[
                "molecular_weight_mode"
            ],
            "transport_section_present": any(
                section["name"] == "TRANSPORT" for section in provenance
            ),
            "transport_records": transport_records,
            "unsupported_source_boundaries": [
                "unrecognized named reaction section headers",
                "native transport runtime and mixture transport",
            ],
        }
    )
    return replace(network, metadata=metadata)


def import_chemkin(
    mechanism_path: str | Path,
    thermo_path: str | Path | None = None,
) -> ReactionNetworkSpec:
    """Read a local CHEMKIN mechanism and optional companion NASA7 thermo file.

    When ``thermo_path`` is omitted, the mechanism must contain a standard
    embedded ``THERMO`` section.  This keeps the strict all-or-reject policy
    while accepting the single-file form emitted by CHEMKIN exporters. Local
    ``INCLUDE`` directives are expanded recursively relative to the including
    file, with missing files and cycles rejected before conversion.
    """

    mechanism_path = Path(mechanism_path)
    if mechanism_path.suffix.lower() not in {".inp", ".ckin", ".ck", ".txt"}:
        raise ValueError("unsupported CHEMKIN mechanism extension")
    mechanism_text, mechanism_includes = _expand_includes(
        mechanism_path.read_text(encoding="utf-8"),
        mechanism_path.parent,
        stack=(mechanism_path.resolve(),),
    )
    if thermo_path is not None:
        thermo_path = Path(thermo_path)
        thermo_text, thermo_includes = _expand_includes(
            thermo_path.read_text(encoding="utf-8"),
            thermo_path.parent,
            stack=(thermo_path.resolve(),),
        )
        source = f"{mechanism_path.name}; thermo={thermo_path.name}"
    else:
        thermo_text = None
        thermo_includes = ()
        source = f"{mechanism_path.name}; embedded THERMO"
    network = convert_chemkin(
        mechanism_text,
        thermo_text,
        source=source,
    )
    included = (*mechanism_includes, *thermo_includes)
    if included:
        metadata = dict(network.metadata)
        metadata["included_source_files"] = included
        network = replace(network, metadata=metadata)
    return network


def chemkin_import_capabilities() -> dict[str, Any]:
    """Machine-readable CHEMKIN subset and explicit rejected directives."""

    return {
        "source_specification": "CHEMKIN fixed-width gas mechanism subset v1",
        "source_section_provenance": {
            "metadata_key": "source_section_provenance",
            "fields": ["name", "line", "header", "arguments", "section_name"],
            "duplicate_core_sections": "reject before conversion",
        },
        "transport": {
            "metadata_key": "transport_records",
            "section_present_key": "transport_section_present",
            "record_fields": [
                "species",
                "geometry",
                "geometry_name",
                "epsilon_over_k_K",
                "sigma_A",
                "dipole_D",
                "polarizability_A3",
                "rotational_relaxation",
            ],
            "source_units": {
                "epsilon_over_k_K": "K",
                "sigma_A": "angstrom",
                "dipole_D": "debye",
                "polarizability_A3": "angstrom^3",
                "rotational_relaxation": "1",
            },
            "policy": (
                "strict validation and source retention; bounded neutral nonpolar consumer "
                "is gas_transport_from_chemkin; nonzero dipole is rejected"
            ),
        },
        "formats": [
            "local mechanism text + companion NASA7 thermo text",
            "single-file local CHEMKIN with embedded NASA7 THERMO section",
        ],
        "phase_models": ["ideal-gas"],
        "thermo_models": ["NASA7"],
        "rate_models": [
            "elementary Arrhenius",
            "third-body",
            "Lindemann",
            "Troe",
            "PLOG pressure-dependent Arrhenius",
            "CHEB pressure-dependent polynomial",
        ],
        "options": [
            "duplicate reactions",
            "thermochemical reverse",
            "D-exponent numbers",
            "recursive local INCLUDE files",
            "MOLECULES concentration units converted with exact Avogadro scaling",
            "explicit REV reverse Arrhenius pairs",
            "FORD forward custom reaction orders",
            "RORD reverse custom reaction orders",
        ],
        "unsupported": [
            "unrecognized named reaction section headers",
            "native transport beyond bounded neutral nonpolar mixture consumer",
            "surface, plasma, ionic, and nonideal phases",
        ],
        "named_section_policy": {
            "REACTIONS_header": (
                "optional one section identifier followed by exactly one CAL/MOLE, "
                "KCAL/MOLE, KJOULES/MOLE, or JOULES/MOLE token, exactly one "
                "MOLE, MOLES, or MOLECULES token, and at most one MWON/MWOFF token"
            ),
            "section_name": "retained as reaction_section_name and provenance section_name",
            "duplicate_or_unknown_header_tokens": "reject whole mechanism",
            "reaction_labels": "reject whole mechanism",
        },
        "whole_mechanism_policy": "all species and reactions converted or reject",
    }
