"""Fail-closed PHREEQC source-to-native eligibility boundary.

The PHREEQC text parser intentionally stops at a normalized source record.
This module accepts an explicit supplement that supplies the complete native
database payload and a record-level provenance map.  The supplement is not
inferred from reaction text: it is validated by the existing
``AqueousDatabase.from_dict`` codec before a native database is returned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.phreeqc_source_audit import PHREEQC_SOURCE_JSON_SCHEMA

PHREEQC_NATIVE_SUPPLEMENT_SCHEMA = "chemworld-phreeqc-native-supplement-v1"


def _payload(value: Mapping[str, Any] | str, label: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> list[str]:
    errors = []
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        errors.append(f"{label} missing: {', '.join(missing)}")
    if unknown:
        errors.append(f"{label} unknown: {', '.join(unknown)}")
    return errors


def phreeqc_native_eligibility(
    source_json: Mapping[str, Any] | str,
    supplement: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Return a machine-readable eligibility report for a native supplement.

    A report is returned for every validation failure; no partial database is
    returned.  ``eligible`` is true only when the explicit native payload
    round-trips through the native codec and every parsed source record has a
    nonempty provenance mapping to native species/reactions.
    """

    errors: list[str] = []
    source: Mapping[str, Any] | None = None
    native: AqueousDatabase | None = None
    try:
        source = _payload(source_json, "PHREEQC source JSON")
    except ValueError as exc:
        errors.append(str(exc))
    try:
        data = _payload(supplement, "PHREEQC native supplement")
    except ValueError as exc:
        data = {}
        errors.append(str(exc))

    expected_top = {"schema", "source_schema", "source_id", "database", "record_map"}
    errors.extend(_exact_keys(data, expected_top, "native supplement"))
    if data.get("schema") != PHREEQC_NATIVE_SUPPLEMENT_SCHEMA:
        errors.append("native supplement has unknown schema")
    if data.get("source_schema") != PHREEQC_SOURCE_JSON_SCHEMA:
        errors.append("native supplement has unknown source schema")

    if source is not None:
        if source.get("schema") != PHREEQC_SOURCE_JSON_SCHEMA:
            errors.append("source JSON has unknown schema")
        if source.get("provider") != "phreeqc":
            errors.append("source JSON provider must be phreeqc")
    source_meta = None if source is None else source.get("source")
    if source_meta is not None and not isinstance(source_meta, Mapping):
        errors.append("source JSON source metadata must be an object")
        source_meta = None
    source_id = None if source_meta is None else source_meta.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        errors.append("source JSON has no authoritative source_id")
    if data.get("source_id") != source_id:
        errors.append("native supplement source_id does not match source JSON")

    source_records = source or {}
    master = source_records.get("master_species", [])
    reactions = source_records.get("reactions", [])
    if not isinstance(master, list) or not isinstance(reactions, list):
        errors.append("source JSON master_species and reactions must be arrays")
        master = []
        reactions = []

    database_payload = data.get("database")
    if not isinstance(database_payload, Mapping):
        errors.append("native supplement database must be a complete object")
    else:
        try:
            native = AqueousDatabase.from_dict(database_payload)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"native database payload is not complete or valid: {exc}")

    record_map = data.get("record_map")
    if not isinstance(record_map, Mapping):
        errors.append("native supplement record_map must be an object")
        record_map = {}
    else:
        errors.extend(_exact_keys(record_map, {"master_species", "reactions"}, "record_map"))

    native_species = set() if native is None else {item.species_id for item in native.species}
    native_reactions = set() if native is None else {item.reaction_id for item in native.reactions}

    def check_map(kind: str, records: list[Mapping[str, Any]], target_ids: set[str]) -> None:
        mapping = record_map.get(kind)
        if not isinstance(mapping, Mapping):
            errors.append(f"record_map.{kind} must be an object")
            return
        expected_keys = {
            str(item.get("species_id" if kind == "master_species" else "reaction"))
            for item in records
        }
        errors.extend(_exact_keys(mapping, expected_keys, f"record_map.{kind}"))
        for key, targets in mapping.items():
            if not isinstance(targets, list) or not targets or not all(
                isinstance(target, str) and target in target_ids for target in targets
            ):
                errors.append(f"record_map.{kind}[{key!r}] must name existing native records")

    check_map("master_species", master, native_species)
    check_map("reactions", reactions, native_reactions)

    if native is not None and source_id not in {item.source_id for item in native.sources}:
        errors.append("native database does not declare the PHREEQC source_id")
    if native is not None:
        missing_parameters = native.activity_model.missing_parameters()
        if missing_parameters:
            errors.append(
                "native activity model missing parameters: "
                + ", ".join(missing_parameters)
            )

    return {
        "schema": PHREEQC_NATIVE_SUPPLEMENT_SCHEMA,
        "source_schema": PHREEQC_SOURCE_JSON_SCHEMA,
        "source_id": source_id,
        "database_id": None if native is None else native.database_id,
        "eligible": not errors,
        "errors": errors,
        "coverage": {
            "source_master_species": len(master),
            "source_reactions": len(reactions),
            "native_species": len(native_species),
            "native_reactions": len(native_reactions),
        },
        "database_payload": None if native is None else native.to_dict(),
    }


def phreeqc_source_to_native(
    source_json: Mapping[str, Any] | str,
    supplement: Mapping[str, Any] | str,
) -> AqueousDatabase:
    """Return a native database only after strict supplement eligibility passes."""

    report = phreeqc_native_eligibility(source_json, supplement)
    if not report["eligible"]:
        raise ValueError(
            "PHREEQC native supplement is not eligible: "
            + "; ".join(report["errors"])
        )
    return AqueousDatabase.from_dict(report["database_payload"])


__all__ = [
    "PHREEQC_NATIVE_SUPPLEMENT_SCHEMA",
    "phreeqc_native_eligibility",
    "phreeqc_source_to_native",
]
