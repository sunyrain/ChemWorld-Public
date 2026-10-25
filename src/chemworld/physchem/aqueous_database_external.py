"""Strict JSON adapter for explicitly normalized Reaktoro/PHREEQC declarations.

This adapter accepts a small, versioned JSON envelope produced by an external
provider export. It is deliberately not a parser for provider source files:
thermochemistry, activity parameters, units, and provenance must already be
fully explicit in the native nested records. Unknown fields are rejected.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from chemworld.physchem.aqueous_database import AqueousDatabase

EXTERNAL_AQUEOUS_DATABASE_SCHEMA = "chemworld-aqueous-external-json-v1"
_EXTERNAL_PROVIDERS = ("phreeqc", "reaktoro")
_UNITS = {
    "temperature": "K",
    "pressure": "Pa",
    "amount": "mol",
    "enthalpy": "J/mol",
    "entropy": "J/(mol*K)",
    "heat_capacity": "J/(mol*K)",
    "molar_volume": "m^3/mol",
    "molality": "mol/kg",
}
_TOP_LEVEL_KEYS = {
    "schema",
    "provider",
    "database_id",
    "units",
    "sources",
    "species",
    "reactions",
    "activity_model",
}
_SOURCE_KEYS = {
    "source_id", "citation", "source_url", "license_id", "license_url", "redistribution"
}
_SPECIES_KEYS = {"name", "formula", "charge", "phase", "source_id", "standard_state"}
_REACTION_KEYS = {"name", "stoichiometry", "source_id"}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{label} has noncanonical fields (" + "; ".join(details) + ")")


def _load_payload(payload: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("external aqueous database JSON is invalid") from exc
    else:
        # Treat a Python mapping exactly like a decoded JSON document.  This
        # makes tuple-valued arrays from a native exporter follow the same
        # JSON array normalization as the text path, while rejecting values
        # that could never be represented in the declared interchange format.
        try:
            payload = json.loads(json.dumps(payload, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("external aqueous database is not JSON-compatible") from exc
    return _mapping(payload, "external aqueous database")


def _canonical_native_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Map external names/formulae to the native database schema."""
    sources = []
    for index, raw in enumerate(data["sources"]):
        item = _mapping(raw, f"sources[{index}]")
        _exact_keys(item, _SOURCE_KEYS, f"sources[{index}]")
        sources.append(dict(item))

    species = []
    for index, raw in enumerate(data["species"]):
        item = _mapping(raw, f"species[{index}]")
        _exact_keys(item, _SPECIES_KEYS, f"species[{index}]")
        species.append(
            {
                "species_id": item["name"],
                "composition": dict(_mapping(item["formula"], f"species[{index}].formula")),
                "charge": item["charge"],
                "phase": item["phase"],
                "source_id": item["source_id"],
                "standard_state": dict(
                    _mapping(item["standard_state"], f"species[{index}].standard_state")
                ),
            }
        )

    reactions = []
    for index, raw in enumerate(data["reactions"]):
        item = _mapping(raw, f"reactions[{index}]")
        _exact_keys(item, _REACTION_KEYS, f"reactions[{index}]")
        reactions.append(
            {
                "reaction_id": item["name"],
                "stoichiometry": dict(
                    _mapping(item["stoichiometry"], f"reactions[{index}].stoichiometry")
                ),
                "source_id": item["source_id"],
            }
        )

    return {
        "model": "chemworld-aqueous-database-v1",
        "database_id": data["database_id"],
        "units": dict(data["units"]),
        "sources": sources,
        "species": species,
        "reactions": reactions,
        "activity_model": dict(
            _mapping(data["activity_model"], "activity_model")
        ),
    }


def aqueous_database_from_external_json(
    payload: Mapping[str, Any] | str,
) -> AqueousDatabase:
    """Import one strict normalized provider JSON envelope.

    ``provider`` identifies the upstream exporter only. This function does not
    parse PHREEQC ``.dat`` files or Reaktoro database files, infer units, fit
    parameters, or synthesize missing thermochemistry.
    """

    data = _load_payload(payload)
    _exact_keys(data, _TOP_LEVEL_KEYS, "external aqueous database")
    if data["schema"] != EXTERNAL_AQUEOUS_DATABASE_SCHEMA:
        raise ValueError("unknown external aqueous database schema")
    if data["provider"] not in _EXTERNAL_PROVIDERS:
        raise ValueError("provider must be exactly 'phreeqc' or 'reaktoro'")
    if data["units"] != _UNITS:
        raise ValueError("external aqueous database requires canonical SI units")
    if (
        not isinstance(data["sources"], list)
        or not isinstance(data["species"], list)
        or not isinstance(data["reactions"], list)
    ):
        raise ValueError("sources, species and reactions must be JSON arrays")

    native = _canonical_native_payload(data)
    try:
        database = AqueousDatabase.from_dict(native)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"external payload contains unknown or invalid fields: {exc}") from exc
    # JSON normalization accounts for tuple-to-array conversion while still
    # detecting fields discarded by a nested codec.
    normalized = json.loads(json.dumps(database.to_dict(), allow_nan=False))
    expected = json.loads(json.dumps(native, allow_nan=False))
    if normalized != expected:
        raise ValueError("external payload contains unknown or lossy fields")
    return database


def aqueous_database_to_external_json(
    database: AqueousDatabase, *, provider: str
) -> dict[str, Any]:
    """Export an existing native database to the strict provider envelope."""

    if not isinstance(database, AqueousDatabase):
        raise TypeError("database must be AqueousDatabase")
    if provider not in _EXTERNAL_PROVIDERS:
        raise ValueError("provider must be exactly 'phreeqc' or 'reaktoro'")
    native = database.to_dict()
    return {
        "schema": EXTERNAL_AQUEOUS_DATABASE_SCHEMA,
        "provider": provider,
        "database_id": native["database_id"],
        "units": dict(_UNITS),
        "sources": native["sources"],
        "species": [
            {
                "name": item["species_id"],
                "formula": item["composition"],
                "charge": item["charge"],
                "phase": item["phase"],
                "source_id": item["source_id"],
                "standard_state": item["standard_state"],
            }
            for item in native["species"]
        ],
        "reactions": [
            {
                "name": item["reaction_id"],
                "stoichiometry": item["stoichiometry"],
                "source_id": item["source_id"],
            }
            for item in native["reactions"]
        ],
        "activity_model": native["activity_model"],
    }


__all__ = [
    "EXTERNAL_AQUEOUS_DATABASE_SCHEMA",
    "aqueous_database_from_external_json",
    "aqueous_database_to_external_json",
]
