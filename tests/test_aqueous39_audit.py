"""Read-only coverage and provenance audit for NATIVE-CONV-39."""

from __future__ import annotations

import json
from dataclasses import replace

from chemworld import physchem
from chemworld.physchem.aqueous_database import AqueousDataSource
from chemworld.physchem.aqueous_database_audit import (
    aqueous_database_capabilities,
    audit_aqueous_database,
)
from reference.aqueous39_fixtures import database


def test_database_audit_reports_sources_domains_and_phase_coverage() -> None:
    report = audit_aqueous_database(database())

    assert report["valid"] is True
    assert report["counts"]["phases"] == {
        "aqueous": 8,
        "gas": 1,
        "mineral": 2,
        "solvent": 1,
    }
    assert report["activity_model"]["family"] == "Davies"
    assert report["activity_model"]["missing_parameters"] == []
    assert report["sources"]["undeclared_used_ids"] == []
    assert report["sources"]["all_have_license_and_citation"] is True
    source_record = report["sources"]["records"][0]
    assert source_record["source_id"] == "synthetic-aqueous39"
    assert source_record["used"] is True
    assert source_record["citation"]
    assert source_record["license_url"]
    assert source_record["usage"]["activity_models"] == ["synthetic-aqueous39-Davies"]
    assert "H2O" in source_record["usage"]["species"]
    assert "water" in source_record["usage"]["reactions"]
    assert report["sources"]["usage"] == {
        "synthetic-aqueous39": source_record["usage"]
    }
    assert report["standard_states"]["reference_basis_ids"] == ["synthetic-shared-gauge"]
    assert json.loads(json.dumps(report, allow_nan=False))["valid"] is True


def test_database_audit_keeps_unreferenced_source_provenance_visible() -> None:
    db = database()
    extra = AqueousDataSource(
        "unused-source",
        "Unused source fixture",
        "urn:chemworld:unused-source",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "synthetic",
    )
    report = audit_aqueous_database(replace(db, sources=(*db.sources, extra)))

    assert report["valid"] is True
    assert report["sources"]["unreferenced_ids"] == ["unused-source"]
    unused = next(
        item for item in report["sources"]["records"] if item["source_id"] == "unused-source"
    )
    assert unused["used"] is False
    assert not any(unused["usage"].values())


def test_database_audit_exposes_missing_activity_parameters_without_fallback() -> None:
    db = database()
    incomplete_activity = replace(
        db.activity_model,
        parameters={
            key: value
            for key, value in db.activity_model.parameters.items()
            if key != "A"
        },
    )
    incomplete = replace(db, activity_model=incomplete_activity)

    report = audit_aqueous_database(incomplete)
    assert report["valid"] is False
    assert report["activity_model"]["missing_parameters"]


def test_database_capabilities_keep_external_and_global_boundaries_explicit() -> None:
    capabilities = aqueous_database_capabilities()

    assert set(capabilities["phase_models"]) == {"aqueous", "solvent", "gas", "mineral"}
    assert set(capabilities["activity_models"]) == {
        "ideal",
        "DebyeHuckelLimiting",
        "DebyeHuckel",
        "Davies",
        "SIT",
        "Pitzer",
    }
    assert "direct Reaktoro/PHREEQC source-file import" in capabilities["unsupported"]
    assert capabilities["qualification_boundaries"]["out_of_domain"]
    assert capabilities["external_json_contract"] == {
        "schema": "chemworld-aqueous-external-json-v1",
        "providers": ["phreeqc", "reaktoro"],
        "normalization": "canonical SI units and lossless native declarations",
        "source_files": "not parsed; provider export must produce this envelope",
    }


def test_database_audit_is_available_through_lazy_public_facade() -> None:
    assert physchem.audit_aqueous_database(database())["database_id"] == "synthetic-aqueous39"
    assert physchem.aqueous_database_capabilities()["source_specification"] == (
        "chemworld-aqueous-database-v1"
    )
