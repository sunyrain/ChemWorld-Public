"""Strict, read-only audit for the supported PHREEQC ``.dat`` source subset.

PHREEQC databases are not native ChemWorld databases: their reaction constants
and analytical expressions do not, by themselves, provide the complete
standard-state thermochemistry required by :class:`AqueousDatabase`.  This
module therefore parses only enough of the public text format to make source
authority and unit coverage reviewable.  It never guesses charge, balances a
reaction, or converts a source file into a native database.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

PHREEQC_SOURCE_AUDIT_SCHEMA = "chemworld-phreeqc-dat-audit-v1"
PHREEQC_SOURCE_JSON_SCHEMA = "chemworld-phreeqc-source-json-v1"

_SECTIONS = ("SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES", "PHASES")
_DELTA_H_FACTORS = {
    "J/mol": 1.0,
    "J": 1.0,
    "kJ/mol": 1_000.0,
    "kJ": 1_000.0,
    "kcal/mol": 4_184.0,
    "kcal": 4_184.0,
}
_TERMINATORS = {"END"}
_SOURCE_KEYS = (
    "source_id",
    "citation",
    "source_url",
    "license_id",
    "license_url",
    "redistribution",
)
_REDISTRIBUTION = {"permitted", "restricted", "unknown", "synthetic"}
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _source_report(source: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if source is None:
        return {
            "provided": False,
            "fields": {},
            "missing": list(_SOURCE_KEYS),
            "valid": False,
        }, ["source metadata is required for an authoritative import"]
    values = dict(source)
    missing = [
        key
        for key in _SOURCE_KEYS
        if not isinstance(values.get(key), str) or not values[key].strip()
    ]
    unknown = sorted(set(values) - set(_SOURCE_KEYS))
    errors = []
    if missing:
        errors.append("source metadata missing: " + ", ".join(missing))
    if unknown:
        errors.append("source metadata has unknown fields: " + ", ".join(unknown))
    if values.get("redistribution") not in _REDISTRIBUTION:
        errors.append("source metadata has unsupported redistribution policy")
    return {
        "provided": True,
        "fields": {key: values.get(key) for key in _SOURCE_KEYS},
        "missing": missing,
        "unknown": unknown,
        "valid": not errors,
    }, errors


def _parse_master_species(
    lines: list[tuple[int, str]], errors: list[str],
) -> list[dict[str, Any]]:
    records = []
    for line_number, line in lines:
        fields = line.split()
        if len(fields) < 5 or fields[0].startswith("-"):
            errors.append(f"SOLUTION_MASTER_SPECIES line {line_number} is not a five-field record")
            continue
        try:
            alkalinity = float(fields[2])
            molecular_weight = float(fields[4])
        except ValueError:
            errors.append(f"SOLUTION_MASTER_SPECIES line {line_number} has nonnumeric values")
            continue
        if len(fields) != 5:
            errors.append(
                f"SOLUTION_MASTER_SPECIES line {line_number} has unsupported extra fields"
            )
            continue
        records.append(
            {
                "line": line_number,
                "species_id": fields[0],
                "master_species": fields[1],
                "alkalinity": alkalinity,
                "formula": fields[3],
                "molecular_weight": molecular_weight,
                "molecular_weight_unit": "g/mol",
            }
        )
    return records


def _parse_reaction_sections(
    section: str, lines: list[tuple[int, str]], errors: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_name: str | None = None
    reaction_pattern = re.compile(r"=", re.ASCII)

    def finish() -> None:
        nonlocal current
        if current is not None:
            if "log_k" not in current:
                errors.append(f"{section} line {current['line']} is missing log_k")
            records.append(current)
            current = None

    for line_number, raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("-") and reaction_pattern.search(line):
            finish()
            current = {
                "line": line_number,
                "reaction": line,
                "section": section,
                "log_k": None,
                "delta_h": None,
                "options": {},
            }
            if pending_name is not None:
                current["name"] = pending_name
                pending_name = None
            continue
        if section == "PHASES" and current is None and not line.startswith("-"):
            pending_name = line
            continue
        if current is None:
            errors.append(f"{section} line {line_number} has options before a reaction")
            continue
        if not line.startswith("-"):
            errors.append(f"{section} line {line_number} is not a recognized option")
            continue
        fields = line[1:].split()
        if not fields:
            continue
        key = fields[0].lower()
        values = fields[1:]
        current["options"].setdefault(key, []).append(" ".join(values))
        if key == "log_k":
            if len(values) != 1:
                errors.append(f"{section} line {line_number} log_k needs one number")
            else:
                try:
                    current["log_k"] = float(values[0])
                except ValueError:
                    errors.append(f"{section} line {line_number} log_k is not numeric")
        elif key == "delta_h":
            if len(values) != 2 or not re.fullmatch(_FLOAT, values[0]):
                errors.append(
                    f"{section} line {line_number} delta_h needs value and explicit unit"
                )
                continue
            unit = values[1]
            value = float(values[0])
            entry: dict[str, Any] = {"value": value, "unit": unit}
            if unit in _DELTA_H_FACTORS:
                entry["value_J_mol"] = value * _DELTA_H_FACTORS[unit]
                entry["canonical_unit"] = "J/mol"
            else:
                errors.append(f"{section} line {line_number} has unsupported delta_h unit {unit}")
            current["delta_h"] = entry
    finish()
    return records


def audit_phreeqc_dat(
    text: str, *, source: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Audit a strict PHREEQC source subset without constructing a database.

    ``SOLUTION_MASTER_SPECIES``, ``SOLUTION_SPECIES`` and ``PHASES`` sections
    are recognized.  Reaction records must contain ``-log_k`` and any
    ``-delta_h`` must carry an explicit, supported unit.  Unknown sections and
    parser errors are retained in the report and make ``valid`` false.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("PHREEQC source text must be a nonempty string")
    errors: list[str] = []
    unknown_sections: list[str] = []
    section_lines: dict[str, list[tuple[int, str]]] = {name: [] for name in _SECTIONS}
    section: str | None = None
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        stripped = line.strip()
        if not stripped:
            continue
        heading = stripped.upper()
        if heading in _SECTIONS:
            section = heading
            continue
        if heading in _TERMINATORS:
            section = None
            continue
        if not line[:1].isspace() and heading.isupper() and " " not in heading:
            section = None
            unknown_sections.append(heading)
            continue
        if section is not None:
            section_lines[section].append((line_number, line))

    source_report, source_errors = _source_report(source)
    errors.extend(source_errors)
    master = _parse_master_species(section_lines["SOLUTION_MASTER_SPECIES"], errors)
    reactions = []
    for name in ("SOLUTION_SPECIES", "PHASES"):
        reactions.extend(_parse_reaction_sections(name, section_lines[name], errors))

    master_ids = [item["species_id"] for item in master]
    reaction_ids = [item["reaction"] for item in reactions]
    duplicates = sorted(
        {item for item in master_ids if master_ids.count(item) > 1}
        | {item for item in reaction_ids if reaction_ids.count(item) > 1}
    )
    if duplicates:
        errors.append("duplicate source record identities: " + ", ".join(duplicates))
    required = {"SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES"}
    missing_sections = sorted(name for name in required if not section_lines[name])
    if missing_sections:
        errors.append("missing required sections: " + ", ".join(missing_sections))
    units = {
        "log_k": {"declared": "dimensionless", "canonical": "1"},
        "delta_h": {
            "accepted": sorted(_DELTA_H_FACTORS),
            "canonical": "J/mol",
            "source_semantics": "PHREEQC delta_h unit token is per mole",
            "implicit_conversion": False,
        },
        "molecular_weight": {"declared": "g/mol", "canonical": "g/mol"},
    }
    return {
        "schema_version": PHREEQC_SOURCE_AUDIT_SCHEMA,
        "valid": not errors and not unknown_sections,
        "source": source_report,
        "sections": {name: len(section_lines[name]) for name in _SECTIONS},
        "unknown_sections": unknown_sections,
        "master_species": master,
        "reactions": reactions,
        "units": units,
        "errors": errors,
        "unsupported": [
            "direct construction of AqueousDatabase from PHREEQC text",
            "charge inference from species names",
            "reaction balancing or standard-state thermochemistry imputation",
            "global phase stability and phase-split solving",
        ],
    }


def phreeqc_dat_to_source_json(
    text: str, *, source: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert an audited PHREEQC subset to a lossless normalized source JSON.

    The result is a source record, not an :class:`AqueousDatabase`.  Conversion
    fails closed when the audit is invalid, and the explicit native-conversion
    boundary prevents callers from treating ``log_k``/``delta_h`` text as
    complete standard-state thermochemistry.
    """

    report = audit_phreeqc_dat(text, source=source)
    if not report["valid"]:
        details = "; ".join(report["errors"] or report["unknown_sections"])
        raise ValueError("PHREEQC source cannot be normalized: " + details)
    return {
        "schema": PHREEQC_SOURCE_JSON_SCHEMA,
        "provider": "phreeqc",
        "source": report["source"]["fields"],
        "sections": report["sections"],
        "master_species": report["master_species"],
        "reactions": report["reactions"],
        "units": report["units"],
        "native_conversion": {
            "eligible": False,
            "reason": "PHREEQC source lacks complete native standard-state thermochemistry",
            "required_fields": [
                "species.standard_state.thermo",
                "species.standard_state.reference_basis_id",
                "activity_model.parameters",
            ],
        },
    }


__all__ = [
    "PHREEQC_SOURCE_AUDIT_SCHEMA",
    "PHREEQC_SOURCE_JSON_SCHEMA",
    "audit_phreeqc_dat",
    "phreeqc_dat_to_source_json",
]
