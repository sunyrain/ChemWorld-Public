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
from pathlib import Path
from typing import Any

from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.reaction_network import ReactionNetworkSpec

_SECTION = re.compile(r"^\s*(ELEMENTS?|ELEM|SPECIES|THERMO|REACTIONS)\b", re.IGNORECASE)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_ARRHENIUS = re.compile(
    rf"^(?P<equation>.+?)\s+(?P<A>{_NUMBER})\s+(?P<b>{_NUMBER})\s+(?P<Ea>{_NUMBER})\s*$"
)
_EFFICIENCY = re.compile(
    rf"(?P<species>[A-Za-z][A-Za-z0-9_().+\-]*)\s*/\s*(?P<value>{_NUMBER})\s*/"
)


def _strip_comment(line: str) -> str:
    return line.split("!", 1)[0].rstrip()


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _sections(
    text: str,
    *,
    require_mechanism: bool = True,
) -> tuple[list[str], list[str], list[str], str]:
    species: list[str] = []
    thermo: list[str] = []
    reactions: list[str] = []
    reaction_header = "REACTIONS"
    current: str | None = None
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        header = _SECTION.match(line)
        if header:
            current = header[1].upper()
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
        elif current in {"ELEMENT", "ELEMENTS", "ELEM"}:
            continue
        else:
            raise ValueError(f"CHEMKIN declaration is outside a supported section: {line}")
    if require_mechanism and (not species or not reactions):
        raise ValueError("CHEMKIN input needs non-empty SPECIES and REACTIONS sections")
    if len(set(species)) != len(species):
        raise ValueError("CHEMKIN SPECIES declarations must be unique")
    return species, thermo, reactions, reaction_header


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


def _reaction_units(header: str) -> dict[str, str]:
    upper = header.upper()
    if "MOLECULES" in upper:
        raise ValueError("CHEMKIN MOLECULES concentration units are unsupported")
    if "KCAL/MOLE" in upper:
        energy = "kcal"
    elif "CAL/MOLE" in upper:
        energy = "cal"
    elif "KJOULES/MOLE" in upper:
        energy = "kJ"
    elif "JOULES/MOLE" in upper:
        energy = "J"
    else:
        raise ValueError("CHEMKIN REACTIONS must declare supported activation-energy units")
    if "MOLES" not in upper:
        raise ValueError("CHEMKIN REACTIONS must declare MOLES concentration units")
    return {
        "volume": "cm^3",
        "quantity": "mol",
        "time": "s",
        "energy": energy,
        "pressure": "atm",
        "temperature": "K",
    }


def _normalize_equation(equation: str) -> str:
    result = re.sub(r"\s+", " ", equation.strip())
    result = result.replace("<=>", "<=>").replace("=>", "=>")
    if "<=>" not in result and "=>" not in result:
        result = result.replace("=", "<=>")
    return result


def _reaction_nodes(lines: list[str], units: dict[str, str]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        parsed = _ARRHENIUS.match(line)
        if parsed:
            current = {
                "equation": _normalize_equation(parsed["equation"]),
                "rate-constant": {
                    "A": _number(parsed["A"]),
                    "b": _number(parsed["b"]),
                    "Ea": _number(parsed["Ea"]),
                },
            }
            nodes.append(current)
            continue
        if current is None:
            raise ValueError("CHEMKIN continuation appears before a reaction")
        upper = line.strip().upper()
        if upper == "DUPLICATE":
            current["duplicate"] = True
        elif upper.startswith("REV") or upper.startswith("PLOG"):
            raise ValueError("CHEMKIN REV/PLOG continuations are unsupported")
        elif upper.startswith("CHEB") or upper.startswith("FORD") or upper.startswith("RORD"):
            raise ValueError("CHEMKIN CHEB/FORD/RORD continuations are unsupported")
        elif upper.startswith("LOW"):
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
            efficiencies = current.setdefault("efficiencies", {})
            for pair in pairs:
                species = pair["species"]
                if species in efficiencies:
                    raise ValueError(f"duplicate CHEMKIN efficiency: {species}")
                efficiencies[species] = _number(pair["value"])
    return nodes


def convert_chemkin(
    mechanism_text: str,
    thermo_text: str | None = None,
    *,
    source: str = "in-memory CHEMKIN input",
) -> ReactionNetworkSpec:
    """Convert one complete CHEMKIN gas mechanism or reject unsupported input."""

    species_names, embedded_thermo_lines, reaction_lines, reaction_header = _sections(
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
        _, thermo_lines, _, _ = _sections(
            "THERMO\n" + thermo_text + "\nEND\n",
            require_mechanism=False,
        )
        thermo_source = "companion NASA7 text"
    records = _thermo_records(thermo_lines, species_names)
    units = _reaction_units(reaction_header)
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
    metadata = dict(network.metadata)
    metadata.update(
        {
            "import_format": "chemkin-gas-mechanism-v1",
            "source_mechanism": source,
            "thermo_source": thermo_source,
            "duplicate_reaction_count": sum(
                bool(reaction.get("duplicate")) for reaction in reaction_nodes
            ),
            "unsupported_source_boundaries": [
                "CHEMKIN REV/PLOG/CHEB/FORD/RORD continuations",
                "CHEMKIN MOLECULES concentration units",
                "external INCLUDE and named reaction sections",
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
    while accepting the single-file form emitted by CHEMKIN exporters.
    """

    mechanism_path = Path(mechanism_path)
    if mechanism_path.suffix.lower() not in {".inp", ".ckin", ".ck", ".txt"}:
        raise ValueError("unsupported CHEMKIN mechanism extension")
    if thermo_path is not None:
        thermo_path = Path(thermo_path)
        thermo_text = thermo_path.read_text(encoding="utf-8")
        source = f"{mechanism_path.name}; thermo={thermo_path.name}"
    else:
        thermo_text = None
        source = f"{mechanism_path.name}; embedded THERMO"
    return convert_chemkin(
        mechanism_path.read_text(encoding="utf-8"),
        thermo_text,
        source=source,
    )


def chemkin_import_capabilities() -> dict[str, Any]:
    """Machine-readable CHEMKIN subset and explicit rejected directives."""

    return {
        "source_specification": "CHEMKIN fixed-width gas mechanism subset v1",
        "formats": [
            "local mechanism text + companion NASA7 thermo text",
            "single-file local CHEMKIN with embedded NASA7 THERMO section",
        ],
        "phase_models": ["ideal-gas"],
        "thermo_models": ["NASA7"],
        "rate_models": ["elementary Arrhenius", "third-body", "Lindemann", "Troe"],
        "options": ["duplicate reactions", "thermochemical reverse", "D-exponent numbers"],
        "unsupported": [
            "REV/PLOG/CHEB/FORD/RORD continuations",
            "MOLECULES concentration units",
            "external INCLUDE and named reaction sections",
            "surface, plasma, ionic, and nonideal phases",
        ],
        "whole_mechanism_policy": "all species and reactions converted or reject",
    }
