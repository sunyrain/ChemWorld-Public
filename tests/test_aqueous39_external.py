"""Strict normalized Reaktoro/PHREEQC JSON adapter tests for NATIVE-CONV-39."""

from __future__ import annotations

import copy
import json

import pytest

from chemworld import physchem
from chemworld.physchem.aqueous_database_external import (
    EXTERNAL_AQUEOUS_DATABASE_SCHEMA,
    aqueous_database_from_external_json,
    aqueous_database_to_external_json,
)
from reference.aqueous39_fixtures import database


def test_external_provider_json_roundtrip_is_lossless_for_both_provider_labels() -> None:
    original = database()
    for provider in ("reaktoro", "phreeqc"):
        exported = aqueous_database_to_external_json(original, provider=provider)
        restored = aqueous_database_from_external_json(exported)
        assert restored.to_dict() == original.to_dict()
        assert exported["schema"] == EXTERNAL_AQUEOUS_DATABASE_SCHEMA
        assert exported["provider"] == provider


def test_external_json_accepts_text_and_rejects_unknown_or_ambiguous_fields() -> None:
    payload = aqueous_database_to_external_json(database(), provider="phreeqc")

    unknown = copy.deepcopy(payload)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match=r"unknown.*unexpected"):
        aqueous_database_from_external_json(unknown)

    ambiguous = copy.deepcopy(payload)
    ambiguous["species"][0]["composition"] = ambiguous["species"][0]["formula"]
    with pytest.raises(ValueError, match="unknown composition"):
        aqueous_database_from_external_json(ambiguous)

    unknown_nested = copy.deepcopy(payload)
    unknown_nested["species"][0]["standard_state"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown or invalid fields"):
        aqueous_database_from_external_json(unknown_nested)

    restored = aqueous_database_from_external_json(
        json.dumps(payload, allow_nan=False)
    )
    assert restored.database_id == database().database_id


def test_external_json_rejects_provider_units_and_lossy_thermo_declarations() -> None:
    payload = aqueous_database_to_external_json(database(), provider="reaktoro")

    invalid_provider = copy.deepcopy(payload)
    invalid_provider["provider"] = "unknown"
    with pytest.raises(ValueError, match="provider"):
        aqueous_database_from_external_json(invalid_provider)

    invalid_units = copy.deepcopy(payload)
    invalid_units["units"]["pressure"] = "bar"
    with pytest.raises(ValueError, match="canonical SI units"):
        aqueous_database_from_external_json(invalid_units)

    lossy = copy.deepcopy(payload)
    thermo = lossy["species"][0]["standard_state"]["thermo"]
    thermo["unexpected"] = "ignored-by-an-upstream-codec"
    with pytest.raises(ValueError, match="unknown or lossy fields"):
        aqueous_database_from_external_json(lossy)

    with pytest.raises(ValueError, match="invalid"):
        aqueous_database_from_external_json("not-json")


def test_external_adapter_is_available_through_lazy_public_facade() -> None:
    payload = physchem.aqueous_database_to_external_json(database(), provider="reaktoro")
    restored = physchem.aqueous_database_from_external_json(payload)
    assert restored.to_dict() == database().to_dict()
