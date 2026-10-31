"""Fail-closed PHREEQC source-to-native supplement tests."""

from __future__ import annotations

import copy
import json

import pytest

from chemworld import physchem
from chemworld.physchem.phreeqc_native_eligibility import (
    PHREEQC_NATIVE_SUPPLEMENT_SCHEMA,
    phreeqc_native_eligibility,
    phreeqc_source_to_native,
)
from chemworld.physchem.phreeqc_source_audit import phreeqc_dat_to_source_json
from reference.aqueous39_fixtures import database

SOURCE = """
SOLUTION_MASTER_SPECIES
    H H+ 0.0 H 1.00794
    O O-2 0.0 O 15.9994
SOLUTION_SPECIES
    H+ = H+
        -log_k 0.0
    OH- + H+ = H2O
        -log_k 14.0
        -delta_h -13.34 kcal
PHASES
    Calcite
        CaCO3 + H+ = Ca+2 + HCO3-
        -log_k 1.849
        -delta_h -2.297 kJ/mol
END
"""

SOURCE_META = {
    "source_id": "synthetic-aqueous39",
    "citation": "synthetic fixture with explicit PHREEQC source metadata",
    "source_url": "urn:chemworld:synthetic:aqueous39",
    "license_id": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "redistribution": "synthetic",
}


def source_json() -> dict:
    return phreeqc_dat_to_source_json(SOURCE, source=SOURCE_META)


def supplement() -> dict:
    native = database().to_dict()
    return {
        "schema": PHREEQC_NATIVE_SUPPLEMENT_SCHEMA,
        "source_schema": "chemworld-phreeqc-source-json-v1",
        "source_id": "synthetic-aqueous39",
        "database": native,
        "record_map": {
            "master_species": {
                "H": ["H2O", "H+", "OH-", "HCO3-"],
                "O": ["H2O", "OH-", "CO2*", "HCO3-", "CO3--"],
            },
            "reactions": {
                "H+ = H+": ["water"],
                "OH- + H+ = H2O": ["water", "acid1"],
                "CaCO3 + H+ = Ca+2 + HCO3-": ["calcite"],
            },
        },
    }


def test_complete_explicit_supplement_roundtrips_to_native_database() -> None:
    report = phreeqc_native_eligibility(source_json(), supplement())
    assert report["eligible"] is True
    assert report["errors"] == []
    json.dumps(report, allow_nan=False)
    restored = phreeqc_source_to_native(source_json(), supplement())
    assert restored.to_dict() == database().to_dict()
    assert physchem.phreeqc_source_to_native(source_json(), supplement()).database_id == (
        "synthetic-aqueous39"
    )


@pytest.mark.parametrize("mutation", ("thermo", "activity"))
def test_missing_explicit_native_thermo_or_activity_is_hard_rejected(mutation: str) -> None:
    payload = supplement()
    if mutation == "thermo":
        del payload["database"]["species"][1]["standard_state"]["thermo"]
    else:
        del payload["database"]["activity_model"]["parameters"]["A"]
    report = phreeqc_native_eligibility(source_json(), payload)
    assert report["eligible"] is False
    assert report["errors"]
    with pytest.raises(ValueError, match="not eligible"):
        phreeqc_source_to_native(source_json(), payload)


def test_missing_record_provenance_is_rejected_even_with_valid_native_database() -> None:
    payload = supplement()
    payload["record_map"] = copy.deepcopy(payload["record_map"])
    del payload["record_map"]["reactions"]["CaCO3 + H+ = Ca+2 + HCO3-"]
    report = phreeqc_native_eligibility(source_json(), payload)
    assert report["eligible"] is False
    assert any("record_map.reactions missing" in error for error in report["errors"])


def test_malformed_source_metadata_is_rejected_without_attribute_error() -> None:
    malformed = source_json()
    malformed["source"] = []
    report = phreeqc_native_eligibility(malformed, supplement())
    assert report["eligible"] is False
    assert any("source metadata must be an object" in error for error in report["errors"])
