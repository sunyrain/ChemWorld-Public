"""Fail-closed bridge from audited Reaktoro source to a native database.

The Reaktoro source audit deliberately stops at a normalized, provider-shaped
record.  This module accepts a separate native supplement only when it is a
complete canonical external database envelope, carries the same source
identity, and explicitly maps every source species and reaction record to a
native record.  It never derives activity parameters or standard-state
thermochemistry from the Reaktoro source.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_database_external import aqueous_database_from_external_json
from chemworld.physchem.reaktoro_source_audit import (
    REAKTORO_SOURCE_JSON_SCHEMA,
    audit_reaktoro_source,
)

REAKTORO_NATIVE_SUPPLEMENT_SCHEMA = "chemworld-reaktoro-native-supplement-v1"
REAKTORO_NATIVE_CAPABILITY_SCHEMA = "chemworld-reaktoro-native-capabilities-v1"
# Reaktoro's ``FormationReaction.Reactants`` spelling assigns positive
# coefficients to reactants and negative coefficients to products.  The
# native AqueousReaction contract uses the opposite sign convention
# (negative reactants, positive products), so the bridge must negate every
# source coefficient before comparing a mapped reaction.
REACTION_STOICHIOMETRY_BASIS = (
    "source-positive-reactants-negative-products-to-native-signed-v1"
)

_SOURCE_KEYS = {
    "source_id",
    "citation",
    "source_url",
    "license_id",
    "license_url",
    "redistribution",
}
_NORMALIZED_SOURCE_KEYS = {
    "schema_version",
    "provider",
    "source",
    "elements",
    "species",
    "native_conversion",
}
_SUPPLEMENT_KEYS = {
    "schema",
    "source_schema",
    "source_id",
    "database",
    "record_map",
    "reaction_stoichiometry_basis",
}
_RECORD_MAP_KEYS = {"species", "reactions"}


def reaktoro_native_capabilities() -> dict[str, Any]:
    """Return the bridge's explicit support and fail-closed boundaries.

    This is a provider-free contract.  It describes what a caller must
    supply to make the bridge eligible and records unsupported source
    families without attempting to interpret their upstream schemas.
    """

    return {
        "schema": REAKTORO_NATIVE_CAPABILITY_SCHEMA,
        "bridge": {
            "supplement_schema": REAKTORO_NATIVE_SUPPLEMENT_SCHEMA,
            "status": "supported-with-explicit-supplement",
            "source_native_eligibility": False,
        },
        "provider": {
            "accepted": ["reaktoro"],
            "status": "supported-exact",
            "policy": "reject other provider envelopes",
        },
        "activity_model": {
            "status": "required-explicit",
            "source": "supplement.database.activity_model",
            "synthesized": False,
        },
        "standard_state": {
            "status": "required-explicit-per-species",
            "source": "supplement.database.species[].standard_state",
            "synthesized": False,
        },
        "record_map": {
            "status": "required-complete-nonoverlapping",
            "species": "exact-source-and-native-coverage",
            "reactions": "exact-source-and-native-coverage",
            "identity": ["formula", "charge", "phase", "source_id"],
        },
        "reaction_stoichiometry": {
            "status": "supported-explicit",
            "basis": REACTION_STOICHIOMETRY_BASIS,
            "source_to_native": "negate-coefficients",
        },
        "unsupported": {
            "HKF": {
                "status": "unsupported-fail-closed",
                "classification": "eos-pressure-corrections",
                "policy": "do-not-interpret-unconfirmed-source-schema",
            },
            "WaterHKF": {
                "status": "unsupported-fail-closed",
                "classification": "eos-pressure-corrections",
                "policy": "do-not-interpret-source-water-model-in-bridge",
            },
            "MineralHKF": {
                "status": "unsupported-fail-closed",
                "classification": "eos-pressure-corrections",
                "policy": "do-not-interpret-source-mineral-model-in-bridge",
            },
            "EOS": {
                "status": "unsupported-fail-closed",
                "classification": "eos-pressure-corrections",
                "policy": "do-not-interpret-unconfirmed-source-schema",
            },
            "Interactions": {
                "status": "unsupported-fail-closed",
                "classification": "interaction-parameters",
                "policy": "do-not-interpret-unconfirmed-source-schema",
            },
            "PhaseModels": {
                "status": "unsupported-fail-closed",
                "classification": "phase-model-metadata",
                "policy": "do-not-interpret-unconfirmed-source-schema",
            },
            "unknown_supplement_fields": {
                "status": "unsupported-fail-closed",
                "classification": "supplement-schema",
                "policy": "reject-noncanonical-fields",
            },
        },
    }


def _load_json(value: Mapping[str, Any] | str, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is invalid JSON") from exc
    try:
        value = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not JSON-compatible") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual, key=str)
    unknown = sorted(actual - expected, key=str)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(map(str, missing)))
        if unknown:
            details.append("unknown " + ", ".join(map(str, unknown)))
        raise ValueError(f"{label} has noncanonical fields (" + "; ".join(details) + ")")


def _canonical_normalized_source(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Require a byte-shape-equivalent output of ``audit_reaktoro_source``."""

    source = _load_json(payload, "normalized Reaktoro source")
    _exact_keys(source, _NORMALIZED_SOURCE_KEYS, "normalized Reaktoro source")
    if source["schema_version"] != REAKTORO_SOURCE_JSON_SCHEMA:
        raise ValueError("normalized Reaktoro source has unknown schema")
    if source["provider"] != "reaktoro":
        raise ValueError("normalized Reaktoro source provider must be reaktoro")

    # Reconstruct the audit input from the canonical fields and compare the
    # complete result.  This prevents callers from changing the source audit
    # decision or smuggling an unvalidated thermo subtree into the bridge.
    species = source["species"]
    if not isinstance(species, list):
        raise ValueError("normalized Reaktoro source species must be an array")
    raw_species = []
    for index, item in enumerate(species):
        if not isinstance(item, Mapping):
            raise ValueError(f"normalized Reaktoro source species[{index}] must be an object")
        _exact_keys(
            item,
            {"name", "formula", "charge", "phase", "standard_thermo_model"},
            f"normalized Reaktoro source species[{index}]",
        )
        thermo = dict(item["standard_thermo_model"])
        model_name = thermo.get("model")
        if model_name == "StandardThermoModel" and "formation_reaction" in thermo:
            # The legacy normalized form keeps this lower-case subtree under
            # a required model marker.
            thermo = {
                "model": "StandardThermoModel",
                "formation_reaction": thermo["formation_reaction"],
            }
        elif model_name == "FormationReaction" and "FormationReaction" in thermo:
            thermo = {"FormationReaction": thermo["FormationReaction"]}
        else:
            # Official direct variants are reconstructed through their
            # StandardThermoModel wrapper.  FormationReaction is a legal
            # sibling of that wrapper and must stay outside it.
            body = {key: value for key, value in thermo.items() if key != "model"}
            sibling = body.pop("FormationReaction", None)
            if not body and sibling is not None:
                thermo = {"FormationReaction": sibling}
            else:
                thermo = {"StandardThermoModel": body}
                if sibling is not None:
                    thermo["FormationReaction"] = sibling
        raw_species.append(
            {
                "name": item["name"],
                "formula": item["formula"],
                "charge": item["charge"],
                "phase": item["phase"],
                "thermo": thermo,
            }
        )
    raw = {
        "source": source["source"],
        "elements": source["elements"],
        "species": raw_species,
    }
    audit = audit_reaktoro_source(raw)
    if not audit["valid"] or audit["normalized"] != source:
        errors = "; ".join(audit["errors"])
        detail = f": {errors}" if errors else ""
        raise ValueError("normalized Reaktoro source is not canonical audit output" + detail)
    return source


def _parse_reaction_stoichiometry(value: Any) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for species, coefficient in value.items():
            if (
                not isinstance(species, str)
                or not species
                or isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))
                or not float(coefficient)
                or not math.isfinite(float(coefficient))
            ):
                return None
            result[species] = float(coefficient)
        return result or None
    if isinstance(value, str) and value.strip():
        result = {}
        for token in value.split():
            if ":" not in token:
                return None
            coefficient_text, species = token.split(":", 1)
            try:
                coefficient = float(coefficient_text)
            except ValueError:
                return None
            if (
                not species
                or not coefficient
                or not math.isfinite(coefficient)
                or species in result
            ):
                return None
            result[species] = coefficient
        return result or None
    return None


def _reaction_specs(source: Mapping[str, Any]) -> dict[str, dict[str, float] | None]:
    specs: dict[str, dict[str, float] | None] = {}

    def find_stoichiometry(value: Any) -> dict[str, float] | None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in {"reactants", "reaction", "stoichiometry"}:
                    parsed = _parse_reaction_stoichiometry(child)
                    if parsed is not None:
                        return parsed
                parsed = find_stoichiometry(child)
                if parsed is not None:
                    return parsed
        elif isinstance(value, list):
            for child in value:
                parsed = find_stoichiometry(child)
                if parsed is not None:
                    return parsed
        return None

    def walk(value: Any, path: str, species_name: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key).lower() in {"formationreaction", "formation_reaction"}:
                    specs[f"{species_name}::{child_path}"] = find_stoichiometry(child)
                walk(child, child_path, species_name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", species_name)

    for item in source["species"]:
        walk(item["standard_thermo_model"], "standard_thermo_model", item["name"])
    return specs


def _record_map(
    value: Any,
    expected: set[str],
    target_ids: set[str],
    label: str,
) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    _exact_keys(value, expected, label)
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for key in sorted(expected):
        targets = value[key]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, str) or not target for target in targets)
        ):
            raise ValueError(f"{label}[{key!r}] must be a nonempty string array")
        if len(set(targets)) != len(targets):
            raise ValueError(f"{label}[{key!r}] repeats a native identity")
        missing = set(targets) - target_ids
        if missing:
            raise ValueError(
                f"{label}[{key!r}] names undeclared native records: "
                + ", ".join(sorted(missing))
            )
        overlap = seen.intersection(targets)
        if overlap:
            raise ValueError(
                f"{label} maps one native record more than once: "
                + ", ".join(sorted(overlap))
            )
        seen.update(targets)
        result[key] = list(targets)
    if seen != target_ids:
        missing = sorted(target_ids - seen)
        extra = sorted(seen - target_ids)
        details = []
        if missing:
            details.append("unmapped native records " + ", ".join(missing))
        if extra:
            details.append("unknown native records " + ", ".join(extra))
        raise ValueError(f"{label} is incomplete (" + "; ".join(details) + ")")
    return result


def _phase_compatible(source_phase: Any, native_phase: str) -> bool:
    if not isinstance(source_phase, str):
        return False
    phase = source_phase.strip().lower()
    if phase == "aqueous":
        return native_phase == "aqueous"
    if phase == "solvent":
        return native_phase == "solvent"
    if phase in {"gas", "gaseous", "vapor"}:
        return native_phase == "gas"
    if phase in {"mineral", "solid"}:
        return native_phase == "mineral"
    return False


def _validate_record_identity(
    source: Mapping[str, Any],
    database: AqueousDatabase,
    record_map: Mapping[str, list[str]],
    reaction_specs: Mapping[str, dict[str, float] | None],
    source_id: str,
    errors: list[str],
) -> None:
    native_species = {item.species_id: item for item in database.species}
    source_species = {item["name"]: item for item in source["species"]}
    for source_name, targets in record_map["species"].items():
        source_item = source_species[source_name]
        for target in targets:
            native = native_species[target]
            if native.source_id != source_id:
                errors.append(
                    f"record_map.species[{source_name!r}] target {target!r} has wrong source_id"
                )
            if dict(native.composition) != dict(source_item["formula"]):
                errors.append(
                    f"record_map.species[{source_name!r}] target {target!r} formula disagrees"
                )
            if native.charge != source_item["charge"]:
                errors.append(
                    f"record_map.species[{source_name!r}] target {target!r} charge disagrees"
                )
            if not _phase_compatible(source_item["phase"], native.phase):
                errors.append(
                    f"record_map.species[{source_name!r}] target {target!r} phase disagrees"
                )
    native_reactions = {item.reaction_id: item for item in database.reactions}
    for source_key, targets in record_map["reactions"].items():
        specification = reaction_specs[source_key]
        if specification is None:
            errors.append(
                f"record_map.reactions[{source_key!r}] lacks explicit source stoichiometry"
            )
            continue
        if len(targets) != 1:
            errors.append(
                f"record_map.reactions[{source_key!r}] requires one native target "
                "for stoichiometry comparison"
            )
            continue
        mapped: dict[str, float] = {}
        for source_species, coefficient in specification.items():
            native_targets = record_map["species"].get(source_species)
            if native_targets is None or len(native_targets) != 1:
                errors.append(
                    f"record_map.reactions[{source_key!r}] cannot compare aliased species "
                    f"{source_species!r}"
                )
                mapped = {}
                break
            # Reaktoro FormationReaction.Reactants and native
            # AqueousReaction use opposite signed conventions.  Only this
            # explicitly declared basis permits the conversion.
            mapped[native_targets[0]] = -coefficient
        if mapped and dict(native_reactions[targets[0]].stoichiometry) != mapped:
            errors.append(
                f"record_map.reactions[{source_key!r}] stoichiometry disagrees with native "
                f"reaction {targets[0]!r}"
            )
        for target in targets:
            if native_reactions[target].source_id != source_id:
                errors.append(
                    f"record_map.reactions[{source_key!r}] target {target!r} has wrong source_id"
                )


def reaktoro_native_eligibility(
    normalized_source: Mapping[str, Any] | str,
    supplement: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Return eligibility for an explicit normalized-source native supplement."""

    errors: list[str] = []
    source: dict[str, Any] | None = None
    database: AqueousDatabase | None = None
    try:
        source = _canonical_normalized_source(normalized_source)
    except ValueError as exc:
        errors.append(str(exc))
    if source is not None:
        for item in source["species"]:
            thermo = item["standard_thermo_model"]
            if isinstance(thermo, Mapping) and "HKF" in thermo:
                errors.append(
                    "native bridge HKF StandardThermoModel is unsupported-fail-closed"
                )
            if isinstance(thermo, Mapping) and "WaterHKF" in thermo:
                errors.append(
                    "native bridge WaterHKF StandardThermoModel is unsupported-fail-closed"
                )
            if isinstance(thermo, Mapping) and "MineralHKF" in thermo:
                errors.append(
                    "native bridge MineralHKF StandardThermoModel is unsupported-fail-closed"
                )
            if isinstance(thermo, Mapping):
                formation = thermo.get("FormationReaction")
                if isinstance(formation, Mapping):
                    reaction_model = formation.get("ReactionStandardThermoModel")
                    if (
                        isinstance(reaction_model, Mapping)
                        and "PressureCorrection" in reaction_model
                    ):
                        errors.append(
                            "native bridge PressureCorrection reaction standard model "
                            "is unsupported-fail-closed"
                        )

    try:
        data = _load_json(supplement, "Reaktoro native supplement")
    except ValueError as exc:
        data = {}
        errors.append(str(exc))

    try:
        _exact_keys(data, _SUPPLEMENT_KEYS, "Reaktoro native supplement")
    except ValueError as exc:
        errors.append(str(exc))
    if data.get("schema") != REAKTORO_NATIVE_SUPPLEMENT_SCHEMA:
        errors.append("Reaktoro native supplement has unknown schema")
    if data.get("source_schema") != REAKTORO_SOURCE_JSON_SCHEMA:
        errors.append("Reaktoro native supplement has unknown source schema")
    if data.get("reaction_stoichiometry_basis") != REACTION_STOICHIOMETRY_BASIS:
        errors.append(
            "Reaktoro native supplement requires explicit reaction stoichiometry basis "
            + REACTION_STOICHIOMETRY_BASIS
        )

    source_id = None if source is None else source.get("source", {}).get("source_id")
    if not isinstance(source_id, str) or not source_id:
        errors.append("normalized Reaktoro source has no authoritative source_id")
        source_id = None
    if data.get("source_id") != source_id:
        errors.append("Reaktoro native supplement source_id does not match normalized source")

    database_payload = data.get("database")
    if database_payload is not None:
        try:
            database_envelope = _load_json(
                database_payload, "Reaktoro native database payload"
            )
            if database_envelope.get("provider") != "reaktoro":
                errors.append("Reaktoro native database payload provider must be reaktoro")
            else:
                database = aqueous_database_from_external_json(database_envelope)
        except (TypeError, ValueError) as exc:
            errors.append(f"Reaktoro native database payload is not complete or valid: {exc}")
    else:
        errors.append("Reaktoro native supplement database is required")

    record_map_data = data.get("record_map")
    record_map: dict[str, list[str]] | None = None
    if not isinstance(record_map_data, Mapping):
        errors.append("Reaktoro native supplement record_map must be an object")
    else:
        try:
            _exact_keys(record_map_data, _RECORD_MAP_KEYS, "record_map")
            if source is not None and database is not None:
                source_species = {item["name"] for item in source["species"]}
                native_species = {item.species_id for item in database.species}
                source_reaction_specs = _reaction_specs(source)
                native_reactions = {item.reaction_id for item in database.reactions}
                record_map = {
                    "species": _record_map(
                        record_map_data.get("species"),
                        source_species,
                        native_species,
                        "record_map.species",
                    ),
                    "reactions": _record_map(
                        record_map_data.get("reactions"),
                        set(source_reaction_specs),
                        native_reactions,
                        "record_map.reactions",
                    ),
                }
        except ValueError as exc:
            errors.append(str(exc))

    if source is not None and database is not None and record_map is not None and source_id:
        _validate_record_identity(
            source,
            database,
            record_map,
            _reaction_specs(source),
            source_id,
            errors,
        )
        if source_id not in {item.source_id for item in database.sources}:
            errors.append("native database does not declare the Reaktoro source_id")

    return {
        "schema": REAKTORO_NATIVE_SUPPLEMENT_SCHEMA,
        "source_schema": REAKTORO_SOURCE_JSON_SCHEMA,
        "source_id": source_id,
        "database_id": None if database is None else database.database_id,
        "eligible": not errors,
        "errors": errors,
        "coverage": {
            "source_species": 0 if source is None else len(source["species"]),
            "source_reactions": 0 if source is None else len(_reaction_specs(source)),
            "native_species": 0 if database is None else len(database.species),
            "native_reactions": 0 if database is None else len(database.reactions),
        },
        "database_payload": None if database is None else database.to_dict(),
    }


def reaktoro_source_to_native(
    normalized_source: Mapping[str, Any] | str,
    supplement: Mapping[str, Any] | str,
) -> AqueousDatabase:
    """Construct a native database only after strict supplement eligibility passes."""

    report = reaktoro_native_eligibility(normalized_source, supplement)
    if not report["eligible"]:
        raise ValueError(
            "Reaktoro native supplement is not eligible: " + "; ".join(report["errors"])
        )
    return AqueousDatabase.from_dict(report["database_payload"])


__all__ = [
    "REACTION_STOICHIOMETRY_BASIS",
    "REAKTORO_NATIVE_CAPABILITY_SCHEMA",
    "REAKTORO_NATIVE_SUPPLEMENT_SCHEMA",
    "reaktoro_native_capabilities",
    "reaktoro_native_eligibility",
    "reaktoro_source_to_native",
]
