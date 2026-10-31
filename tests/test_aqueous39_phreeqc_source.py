"""Strict PHREEQC source-authority and unit-contract audit tests."""

from __future__ import annotations

import pytest

from chemworld import physchem
from chemworld.physchem.phreeqc_source_audit import (
    PHREEQC_SOURCE_AUDIT_SCHEMA,
    PHREEQC_SOURCE_JSON_SCHEMA,
    audit_phreeqc_dat,
    phreeqc_dat_to_source_json,
)

SOURCE = """
SOLUTION_MASTER_SPECIES
    H   H+    0.0   H   1.00794
    O   O-2   0.0   O   15.9994
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

METADATA = {
    "source_id": "phreeqc-test",
    "citation": "synthetic PHREEQC source fixture",
    "source_url": "https://example.invalid/phreeqc.dat",
    "license_id": "synthetic",
    "license_url": "https://example.invalid/license",
    "redistribution": "synthetic",
}


def test_phreeqc_source_audit_preserves_sections_and_normalizes_declared_heat_units() -> None:
    report = audit_phreeqc_dat(SOURCE, source=METADATA)
    assert report["schema_version"] == PHREEQC_SOURCE_AUDIT_SCHEMA
    assert report["valid"] is True
    assert report["sections"] == {
        "SOLUTION_MASTER_SPECIES": 2,
        "SOLUTION_SPECIES": 5,
        "PHASES": 4,
    }
    assert report["reactions"][1]["delta_h"] == {
        "value": -13.34,
        "unit": "kcal",
        "value_J_mol": pytest.approx(-13.34 * 4184.0),
        "canonical_unit": "J/mol",
    }
    assert report["reactions"][2]["delta_h"]["value_J_mol"] == pytest.approx(-2297.0)
    assert report["units"]["delta_h"]["implicit_conversion"] is False


def test_phreeqc_source_audit_requires_authority_and_explicit_delta_h_units() -> None:
    missing = audit_phreeqc_dat(SOURCE)
    assert missing["valid"] is False
    assert "source metadata is required for an authoritative import" in missing["errors"]

    no_unit = SOURCE.replace("-delta_h -13.34 kcal", "-delta_h -13.34")
    report = audit_phreeqc_dat(no_unit, source=METADATA)
    assert report["valid"] is False
    assert any("delta_h needs value and explicit unit" in error for error in report["errors"])


def test_phreeqc_source_audit_rejects_unknown_sections_and_public_facade_exports_it() -> None:
    report = audit_phreeqc_dat(SOURCE + "\nSURFACE_SPECIES\n    X = Y\n", source=METADATA)
    assert report["valid"] is False
    assert report["unknown_sections"] == ["SURFACE_SPECIES"]
    assert "direct construction of AqueousDatabase from PHREEQC text" in report["unsupported"]
    assert physchem.audit_phreeqc_dat(SOURCE, source=METADATA)["valid"] is True


def test_phreeqc_source_to_normalized_json_is_lossless_and_stays_out_of_native_database() -> None:
    normalized = phreeqc_dat_to_source_json(SOURCE, source=METADATA)
    assert normalized["schema"] == PHREEQC_SOURCE_JSON_SCHEMA
    assert normalized["provider"] == "phreeqc"
    assert normalized["source"] == METADATA
    assert normalized["native_conversion"]["eligible"] is False
    assert normalized["native_conversion"]["required_fields"]
    assert normalized["reactions"][1]["delta_h"]["value_J_mol"] == pytest.approx(
        -13.34 * 4184.0
    )
    assert physchem.phreeqc_dat_to_source_json(SOURCE, source=METADATA) == normalized

    with pytest.raises(ValueError, match="cannot be normalized"):
        phreeqc_dat_to_source_json(SOURCE.replace("-log_k 14.0", "-log_k missing"), source=METADATA)
