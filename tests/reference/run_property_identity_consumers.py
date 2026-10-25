"""Qualify reviewed components through the unchanged original F2 consumers."""

import argparse
import json
import re
import runpy
import time
import traceback
from collections import Counter
from pathlib import Path

import chemicals
import review_crc84_identity as crc84
from collect_gsrs_identities import public_identity
from collect_identity_depositors import deposit_fields
from collect_identity_material_pages import TextPage
from collect_sander_identity_evidence import IdentityPage
from collect_srs_identity_sources import FIELDS as SRS_FIELDS
from convert_racemic_identities import stereochemical_layers
from review_printed_identity import (
    ROUTE as DOCUMENT_ROUTE,
)
from review_printed_identity import (
    check_document_identity,
    document_negative_variants,
    read_documents,
)
from run_electro_optical_block import dielectric_source_unit, native_card, optical_source_unit

from chemworld.data.electro_optical import ElectroOpticalDatabase
from chemworld.data.identifiers import UnknownIdentifierError
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.specs import ComponentSpec


def check_identity(
    component,
    gsrs,
    pubchem,
    racemic=None,
    chebi=None,
    nci=None,
    sander=None,
    key_structures=None,
    depositors=None,
    depositor_encodings=None,
    cameo=None,
    remaining_deposits=None,
    remaining_encodings=None,
    institutional=None,
    srs=None,
    documents=None,
    crc84_documents=None,
    source_units=None,
):
    scope = component.metadata["identity_scope"]
    external = component.metadata["external_identity"]
    cas = component.cas_number
    if scope.get("evidence_route") == crc84.ROUTE:
        crc84.check_identity(component, crc84_documents, pubchem, srs, source_units)
        check_roundtrip(component)
        return
    if scope.get("evidence_route") == DOCUMENT_ROUTE:
        check_document_identity(component, documents, pubchem)
        check_roundtrip(component)
        return
    if scope.get("evidence_route") == "reviewed_srs_former_CAS_gsrs":
        check_srs_identity(component, gsrs, pubchem, srs)
        check_roundtrip(component)
        return
    if scope.get("evidence_route") == "reviewed_institutional_compound_identity":
        check_institutional_identity(component, gsrs, pubchem, nci, institutional)
        check_roundtrip(component)
        return
    if scope.get("evidence_route") == "reviewed_original_deposit_geometry":
        check_original_deposit_geometry(
            component, pubchem, nci, remaining_deposits, remaining_encodings
        )
        check_roundtrip(component)
        return
    if scope.get("evidence_route") == "reviewed_pubchem_depositor":
        check_depositor_identity(component, gsrs, pubchem, nci, depositors, depositor_encodings)
        check_roundtrip(component)
        return
    if scope.get("evidence_route") == "reviewed_CAMEO_stoichiometric_adduct":
        check_adduct_identity(component, pubchem, cameo)
        check_roundtrip(component)
        return
    if scope.get("evidence_route") in (
        "reviewed_nci_cir_pubchem",
        "reviewed_CAS_annotation_with_CIR_structure",
    ):
        check_nci_identity(component, gsrs, pubchem, nci)
        check_roundtrip(component)
        return
    if scope.get("evidence_route") == "reviewed_sander_pubchem":
        check_sander_identity(component, gsrs, pubchem, nci, sander, key_structures)
        check_roundtrip(component)
        return
    original = next(r for r in gsrs["results"] if r["source_CAS"] == cas)
    assert len(original["candidates"]) == 1
    record = gsrs["identities"][external["GSRS_uuid"]]
    assert original["candidates"][0]["uuid"] == record["uuid"]
    assert record["status"] == "approved" and record["definitionLevel"] == "COMPLETE"
    assert record["substanceClass"] == "chemical"
    assert record["version"] == external["GSRS_version"] and not record["deprecated"]
    assert any(
        c["code"] == cas and c["codeSystem"] == "CAS" and c["type"] == scope["source_CAS_code_type"]
        for c in record["codes"]
    )
    assert any(
        c["code"] == scope["GSRS_primary_CAS"]
        and c["codeSystem"] == "CAS"
        and c["type"] == "PRIMARY"
        for c in record["codes"]
    )
    structure = record["structure"]
    assert structure["stereochemistry"] == scope["stereochemistry"]
    assert structure["formula"] == component.formula
    assert structure["charge"] == component.charge
    assert scope["source_CAS_code_type"] in {"PRIMARY", "ALTERNATIVE", "SUPERSEDED"}
    original = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
    if "PubChem_candidates" in external:
        assert external["PubChem_candidates"] == [
            candidate_identity(c) for c in original["candidates"]
        ]
    else:
        assert len(original["candidates"]) == 1
    candidate = next(c for c in original["candidates"] if c["CID"] == external["PubChem_CID"])
    transformation = component.metadata.get("identity_transformation")
    if transformation is None:
        assert structure["_inchi"] == component.inchi
        assert structure["_inchiKey"] == component.inchi_key
        if scope["stereochemistry"] == "RACEMIC":
            # Connectivity-level racemates do not declare an absolute enantiomer.
            assert "/t" not in component.inchi
        elif scope["stereochemistry"] == "ABSOLUTE":
            assert structure["stereoCenters"] == structure["definedStereo"]
    else:
        check_racemic_encoding(component, record, racemic)
    if "Sander_CAS" in external:
        assert sander is not None
        reference = sander[external["Sander_CAS"]]
        assert reference["CAS"] == cas
        assert reference["formula"] == component.formula
        assert reference["representative_InChIKey"] == structure["_inchiKey"]
        assert "trans" in reference["name"]
        assert candidate["IUPACName"] not in component.aliases
        assert component.metadata["name_conflict"]["PubChem_IUPACName"] == candidate["IUPACName"]
    assert candidate["CID"] == external["PubChem_CID"]
    if external.get("PubChem_structure_relationship") == "unspecified_double_bond_geometry":
        # Explicit supplementary evidence replaces a generic candidate, never a
        # silently stripped stereo layer on the actual material component.
        assert chebi is not None
        reference = chebi[external["ChEBI_ID"]]
        assert cas in reference["CAS"]
        assert reference["name"] in component.aliases
        assert reference["formula"] == component.formula
        assert reference["inchi"] == component.inchi
        assert reference["inchi_key"] == component.inchi_key
        assert "/b" in component.inchi and "/b" not in candidate["InChI"]
        assert candidate["InChI"] == component.inchi.split("/b")[0]
        assert candidate["InChIKey"] != component.inchi_key
    else:
        assert candidate["InChI"] == structure["_inchi"]
        assert candidate["InChIKey"] == structure["_inchiKey"]
    assert candidate["MolecularFormula"] == component.formula
    assert bool(candidate["CAS_matches"]) == external["PubChem_source_CAS_annotation"]
    check_roundtrip(component)


def check_roundtrip(component):
    assert component.default_phase == "unknown"
    restored = ComponentSpec.from_dict(json.loads(json.dumps(component.to_dict())))
    assert restored.to_dict() == component.to_dict()


def check_racemic_encoding(component, record, racemic):
    """Retain all molecular and relative layers of an explicitly sourced racemate."""
    assert racemic is not None and racemic["reference_version"] == "InChI 1.07.5"
    scope = component.metadata["identity_scope"]
    transformation = component.metadata["identity_transformation"]
    structure = record["structure"]
    assert scope["stereochemistry"] == "RACEMIC"
    assert structure["stereoCenters"] == structure["definedStereo"] > 0
    assert transformation["mode"] == "racemic"
    assert transformation["reference_version"] == racemic["reference_version"]
    assert transformation["reference_url"] == racemic["reference_url"]
    conversions = [r for r in racemic["results"] if r["source_CAS"] == component.cas_number]
    assert len(conversions) == 2
    for converted in conversions:
        assert converted["status"] == "passed" and converted["returncode"] == 0
        assert converted["GSRS_uuid"] == record["uuid"]
        assert converted["GSRS_version"] == record["version"]
        assert converted["source_formula"] == component.formula
        assert converted["source_InChI"] == structure["_inchi"]
        assert converted["source_InChIKey"] == structure["_inchiKey"]
    absolute = next(r for r in conversions if r["mode"] == "absolute")
    encoded = next(r for r in conversions if r["mode"] == "racemic")
    assert absolute["inchi"] == structure["_inchi"]
    assert absolute["inchi_key"] == structure["_inchiKey"]
    assert transformation["representative_InChI"] == absolute["inchi"]
    assert transformation["representative_InChIKey"] == absolute["inchi_key"]
    assert transformation["options"] == encoded["options"]
    assert encoded["options"] == ["/Key", "/AuxNone", "/NoLabels", "/SRac"]
    assert component.inchi == encoded["inchi"]
    assert component.inchi_key == encoded["inchi_key"] != absolute["inchi_key"]
    assert component.inchi.startswith("InChI=1/")
    layers = component.inchi.split("/")[1:]
    assert "s3" in layers and not any(s.startswith("m") for s in layers)
    assert stereochemical_layers(component.inchi) == stereochemical_layers(absolute["inchi"])


def read_fenchone_encoding(directory, current_directory):
    encoded = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert encoded["block"] == "SRS-fenchone-InChI-encoding"
    assert encoded["total"] == encoded["completed"] == encoded["passed"] == 2
    assert encoded["failed"] == 0
    cas, uuid = "18492-37-0", "2266125a-3987-4cda-b319-dabefe16d6fc"
    raw = json.loads((current_directory / (uuid + ".json")).read_text(encoding="utf-8"))
    assert (directory / (cas + ".mol")).read_text(encoding="utf-8") == raw["structure"]["molfile"]
    pubchem_path = Path(encoded["current_PubChem_source"])
    pubchem = json.loads(pubchem_path.read_text(encoding="utf-8"))
    query = next(r for r in pubchem["structure_queries"] if r["source_CAS"] == cas)
    properties = json.loads(
        (pubchem_path.parent / (cas + "-properties.json")).read_text(encoding="utf-8")
    )
    assert query["query_CAS"] == "1195-79-5" and query["status"] == "received"
    assert query["candidates"] == properties["PropertyTable"]["Properties"]
    for row in encoded["results"]:
        assert row["source_CAS"] == cas and row["GSRS_uuid"] == uuid
        output = (directory / (cas + "-" + row["mode"] + ".txt")).read_text(encoding="utf-8")
        assert re.findall(r"^InChI=.*$", output, re.M) == [row["inchi"]]
        assert re.findall(r"^InChIKey=([A-Z-]+)$", output, re.M) == [row["inchi_key"]]
        assert row["current_PubChem_comparisons"] == [{
            "CID": c["CID"], "InChI": c["InChI"], "InChIKey": c["InChIKey"],
            "complete_graph_matches": (
                c["InChI"] == row["inchi"] and c["InChIKey"] == row["inchi_key"]
            ),
            "relative_layers_match": (
                stereochemical_layers(c["InChI"]) == stereochemical_layers(row["inchi"])
            ),
        } for c in query["candidates"]]
    return {"encoding": encoded, "current_PubChem_query": query}


def read_srs_sources(directory, current_directory, fenchone_directory=None):
    sources = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    current = json.loads((current_directory / "summary.json").read_text(encoding="utf-8"))
    assert sources["attempted"] == sources["total"] == 96
    assert current["completed"] == current["total"] == 58
    assert not sources["unattempted_CAS"] and not current["unattempted_CAS"]
    for row in sources["results"]:
        raw = json.loads((directory / row["filename"]).read_text(encoding="utf-8"))
        assert row["http_status"] == 200
        assert [{k: r[k] for k in SRS_FIELDS if k in r} for r in raw] == row["candidates"]
    prior_directory = Path(current["prior_artifact"]).parent
    for uuid, record in current["identities"].items():
        if record.get("error"):
            continue
        path = current_directory / (uuid + ".json")
        if not path.exists():
            path = prior_directory / (uuid + ".json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert public_identity(raw) == {k: v for k, v in record.items() if k != "request"}
    return {
        "sources": sources, "current": current,
        "fenchone": read_fenchone_encoding(fenchone_directory, current_directory)
        if fenchone_directory else None,
    }


def check_srs_identity(component, original_gsrs, pubchem, evidence):
    """Bind a reviewed historical CAS through SRS and a complete GSRS racemate."""
    assert evidence is not None
    scope = component.metadata["identity_scope"]
    external = component.metadata["external_identity"]
    cas = component.cas_number
    assert scope["source_CAS"] == cas
    query = next(r for r in evidence["sources"]["results"] if r["source_CAS"] == cas)
    assert query["status"] == "received_candidates" and len(query["candidates"]) == 1
    source = query["candidates"][0]
    assert source["internalTrackingNumber"] == external["SRS_ITN"]
    assert source["casNumbers"] == external["SRS_CAS_codes"]
    assert source["currentCasNumber"] == scope["GSRS_primary_CAS"] != cas
    assert scope["source_CAS_code_type"] == "Former"
    assert {"casNumber": cas, "casTypeName": "Former"} in source["casNumbers"]
    assert {"casNumber": scope["GSRS_primary_CAS"], "casTypeName": "Current"} in (
        source["casNumbers"]
    )
    assert source["substanceType"] == "Chemical Substance"
    assert parse_formula(source["molecularFormula"]) == component.composition
    assert source["molecularFormula"] == external["SRS_formula"]
    assert source["inchiNotation"] == external["SRS_InChI"]
    current = evidence["current"]
    row = next(r for r in current["results"] if r["current_CAS"] == scope["GSRS_primary_CAS"])
    assert row["status"] == "exact_code_candidates" and len(row["candidates"]) == 1
    assert row["candidates"][0]["uuid"] == external["GSRS_uuid"]
    record = current["identities"][external["GSRS_uuid"]]
    assert record["status"] == "approved" and record["definitionLevel"] == "COMPLETE"
    assert record["substanceClass"] == "chemical" and not record["deprecated"]
    assert record["version"] == external["GSRS_version"]
    assert any(c["codeSystem"] == "CAS" and c["code"] == scope["GSRS_primary_CAS"]
               and c["type"] == "PRIMARY" for c in record["codes"])
    structure = record["structure"]
    assert structure["stereochemistry"] == scope["stereochemistry"] == "RACEMIC"
    assert structure["stereoCenters"] == scope["stereo_centers"]
    assert scope["enantiomer_mole_fractions"] == [0.5, 0.5]
    assert scope["absolute_configuration"] == "not_selected"
    assert structure["formula"] == component.formula
    assert structure["charge"] == component.charge == 0
    if structure["stereoCenters"] == 1:
        assert structure["definedStereo"] == 0
        assert component.metadata.get("identity_transformation") is None
        assert structure["_inchi"] == component.inchi
        assert structure["_inchiKey"] == component.inchi_key
        assert "/t" not in component.inchi and "/s" not in component.inchi
        assert len(component.provenance) == 2
    else:
        assert cas == "18492-37-0" and structure["stereoCenters"] == 2
        assert structure["opticalActivity"] == "( + / - )"
        fenchone = evidence["fenchone"]
        assert fenchone is not None
        check_racemic_encoding(component, record, fenchone["encoding"])
        candidates = fenchone["current_PubChem_query"]["candidates"]
        assert external["current_PubChem_candidates"] == candidates
        assert external["current_PubChem_relationship"] == "unspecified_relative_geometry"
        assert all("/t" not in c["InChI"] and c["InChIKey"] != component.inchi_key
                   for c in candidates)
        assert len(component.provenance) == 3
        assert component.provenance[2].source_path == fenchone["encoding"]["reference_url"]
    if source["inchiNotation"]:
        assert source["inchiNotation"].split("/", 1)[1] == component.inchi.split("/", 1)[1]
    assert record["_name"] == external["GSRS_name"] and record["_name"] in component.aliases
    original = next(r for r in original_gsrs["results"] if r["source_CAS"] == cas)
    assert external["original_GSRS_candidates"] == original["candidates"]
    original = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
    assert external["PubChem_candidates"] == [candidate_identity(c) for c in original["candidates"]]
    assert component.provenance[0].source_path == query["url"]
    assert component.provenance[0].source_key == cas
    assert component.provenance[1].source_key == record["uuid"]


def read_institutional_identity(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    record = next(r for r in summary["results"] if r["filename"] == "ineris-vanadium.pdf")
    assert record["status"] == "received" and record["http_status"] == 200
    assert record["pages"] == 142 and record["CAS_matches"]["13520-90-6"] == [13]
    assert (directory / record["filename"]).read_bytes().startswith(b"%PDF-")
    page = (directory / (record["filename"] + ".txt")).read_text(encoding="utf-8").split("\f")[12]
    assert "VANADIUM ET SES COMPOSÉS" in page and "Page 13 sur 142" in page
    assert "DRC-07-83451-06937D" in page and "mars 2012" in page
    pattern = (
        r"Tribromure de vanadyle\s+(13520-90-6)\s+\(1\)\s+"
        r"(Vanadyl tribromide)\s+(Liquide rouge foncé)\s+(VOBr3)"
    )
    found = re.findall(pattern, page)
    assert len(found) == 1
    cas, name, physical_form, formula = found[0]
    return {
        "CAS": cas, "name": name, "formula": formula, "physical_form": physical_form,
        "url": record["url"], "document": "INERIS DRC-07-83451-06937D",
        "version": "2-3 mars 2012", "page": 13,
    }


def check_institutional_identity(component, gsrs, pubchem, nci, institutional):
    """An exact named compound is usable without fabricating an optional InChI."""
    assert institutional is not None
    scope = component.metadata["identity_scope"]
    external = component.metadata["external_identity"]
    assert component.cas_number == scope["source_CAS"] == institutional["CAS"]
    assert component.formula == institutional["formula"]
    assert component.composition == {"V": 1.0, "O": 1.0, "Br": 3.0}
    assert component.charge == 0 and not component.inchi and not component.inchi_key
    assert scope["material_kind"] == "named_stoichiometric_compound"
    assert scope["stereochemistry"] == "NOT_SPECIFIED_BY_SOURCE"
    assert external["INERIS_identity_fields"] == institutional
    assert institutional["name"] in component.aliases
    assert len(component.provenance) == 1
    provenance = component.provenance[0]
    assert provenance.source_id == "INERIS" and provenance.source_path == institutional["url"]
    assert provenance.source_key == institutional["CAS"]
    for source in (gsrs["results"], nci["results"]):
        row = next(r for r in source if r["source_CAS"] == component.cas_number)
        if "representations" in row:
            assert not row["representations"]["stdinchi"]["candidates"]
        else:
            assert not row["candidates"]
    original = next(r for r in pubchem["original_row_comparison"]
                    if r["source_CAS"] == component.cas_number)
    assert not original["candidates"] and external["PubChem_candidates"] == []


def check_nci_identity(component, gsrs, pubchem, nci):
    """Check an explicitly reviewed material against the saved full CIR response."""
    assert nci is not None
    scope = component.metadata["identity_scope"]
    external = component.metadata["external_identity"]
    cas = component.cas_number
    assert scope["source_CAS"] == cas
    annotation_binding = scope.get("evidence_route") == "reviewed_CAS_annotation_with_CIR_structure"
    if annotation_binding:
        assert scope["CAS_binding_basis"] == "explicit_source_annotation"
        assert external["CIR_resolver_role"] == "structure_only"
        assert "reviewed_racemate_annotation" in external
    if "reviewed_racemate_annotation" in external:
        assert scope["stereochemistry"] == "RACEMIC"
        assert scope["stereo_centers"] == 1
        assert scope["enantiomer_mole_fractions"] == [0.5, 0.5]
        assert scope["absolute_configuration"] == "not_selected"
        assert isinstance(scope["stereocenter_InChI_atom"], int)
    else:
        assert scope["stereochemistry"] == "ACHIRAL"
    assert not next(r for r in gsrs["results"] if r["source_CAS"] == cas)["candidates"]
    original = next(r for r in nci["results"] if r["source_CAS"] == cas)
    assert original["original_units"] == scope["source_units"]
    for representation in ("stdinchi", "names"):
        response = original["representations"][representation]
        assert response["status"] == "received" and response["http_status"] == 200
        assert response["candidates"] == external["NCI_CIR_candidates"][representation]
        assert len(response["candidates"]) == 1
        record = response["candidates"][0]
        assert record["resolver"] == ("name_by_cir" if annotation_binding else "cas_number")
        assert record["notation"] == cas
        if representation == "stdinchi":
            assert record["items"] == [component.inchi]
        else:
            assert cas in record["items"]
    candidates = next(
        r["candidates"] for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas
    )
    assert external["PubChem_candidates"] == [candidate_identity(c) for c in candidates]
    candidate = next(c for c in candidates if c["CID"] == external["PubChem_CID"])
    assert candidate["InChI"] == component.inchi
    assert candidate["InChIKey"] == component.inchi_key
    assert candidate["MolecularFormula"] == component.formula == component.inchi.split("/")[1]
    assert external["PubChem_source_CAS_annotation"] is True
    assert any(cas in entry["CAS"] for entry in candidate["CAS_matches"])
    if "reviewed_racemate_annotation" in external:
        annotation = external["reviewed_racemate_annotation"]
        assert annotation in candidate["CAS_matches"] and cas in annotation["CAS"]
        name = annotation["reference"]["Name"]
        assert name in component.aliases
        assert "\u00b1" in name or "(+/-)" in name
    assert component.charge == 0 and "/q" not in component.inchi and "/p" not in component.inchi
    assert "/t" not in component.inchi
    geometry = scope["double_bond_geometry"]
    if "reviewed_geometry_annotation" in external:
        annotation = external["reviewed_geometry_annotation"]
        assert annotation in candidate["CAS_matches"] and cas in annotation["CAS"]
        name = annotation["reference"]["Name"]
        assert name in component.aliases
        assert geometry in ("E", "Z") and "/b" in component.inchi
        assert re.search(rf"\(\d*{geometry}\)", name)
    elif geometry == "Z":
        assert "/b" in component.inchi and candidate["IUPACName"].startswith("(Z)-")
    else:
        assert scope["double_bond_geometry"] == "not_applicable" and "/b" not in component.inchi


def candidate_identity(candidate):
    fields = ("CID", "MolecularFormula", "InChI", "InChIKey", "IUPACName")
    return {
        **{k: candidate.get(k) for k in fields},
        "CAS_annotated": bool(candidate["CAS_matches"]),
    }


def check_depositor_identity(component, gsrs, pubchem, nci, depositors, encodings=None):
    """Check the named original deposit, not a merged compound synonym list."""
    assert depositors is not None
    scope, external = component.metadata["identity_scope"], component.metadata["external_identity"]
    cas = component.cas_number
    assert scope["source_CAS"] == cas
    assert not next(r for r in gsrs["results"] if r["source_CAS"] == cas)["candidates"]
    query = depositors[cas]
    assert query["status"] == "received" and query["http_status"] == 200
    assert query["source_ID"] == external["depositor_source_ID"]
    assert query["original_units"] == scope["source_units"]
    assert query["url"] == external["depositor_url"]
    assert len(query["records"]) == 1
    deposit = query["records"][0]
    assert deposit["source_name"] == external["depositor_source_name"] == "EPA DSSTox"
    assert deposit["source_ID"] == {"str": external["depositor_source_ID"]}
    assert deposit["SID"] == external["PubChem_SID"]
    assert deposit["standardized_CIDs"] == [external["PubChem_CID"]]
    assert cas in deposit["deposited_names"]
    candidates = next(
        r["candidates"] for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas
    )
    assert candidates == query["original_PubChem_candidates"]
    assert external["PubChem_candidates"] == [candidate_identity(c) for c in candidates]
    candidate = next(c for c in candidates if c["CID"] == external["PubChem_CID"])
    annotation = external["reviewed_depositor_annotation"]
    assert annotation == query["selected_annotation"]
    assert annotation in candidate["CAS_matches"] and cas in annotation["CAS"]
    reference = annotation["reference"]
    assert reference["SourceName"] == deposit["source_name"]
    assert reference["SourceID"] == external["depositor_source_ID"]
    assert reference["Name"] in deposit["deposited_names"]
    assert reference["Name"] in component.aliases
    transformation = component.metadata.get("identity_transformation")
    if transformation is None:
        assert candidate["InChI"] == component.inchi
        assert candidate["InChIKey"] == component.inchi_key
    else:
        assert scope["stereochemistry"] == "RACEMIC" and scope["stereo_centers"] == 2
        assert scope["relative_configuration"] == "R*,R*"
        assert "(R*,R*)-(+-)-" in reference["Name"]
        assert scope["enantiomer_mole_fractions"] == [0.5, 0.5]
        assert scope["absolute_configuration"] == "not_selected"
        assert transformation["representative_InChI"] == candidate["InChI"]
        assert transformation["representative_InChIKey"] == candidate["InChIKey"]
        assert transformation["mode"] == "racemic"
    assert candidate["MolecularFormula"] == component.formula == component.inchi.split("/")[1]
    assert component.charge == 0 and all(s not in component.inchi for s in ("/q", "/p"))
    if scope["stereochemistry"] == "ACHIRAL":
        assert scope["stereo_centers"] == 0 and "/t" not in component.inchi
    else:
        assert encodings is not None and encodings["reference_version"] == "InChI 1.07.5"
        conversions = [r for r in encodings["results"] if r["source_CAS"] == cas]
        assert len(conversions) == 2
        for converted in conversions:
            assert converted["status"] == "encoded" and converted["returncode"] == 0
            assert converted["SID"] == external["PubChem_SID"]["id"]
            assert converted["source_ID"] == external["depositor_source_ID"]
            assert converted["source_formula"] == component.formula
            assert converted["source_url"] == external["depositor_SDF_url"]
            assert converted["source_InChI"] == candidate["InChI"]
        absolute = next(r for r in conversions if r["mode"] == "absolute")
        assert absolute["inchi"] == candidate["InChI"]
        assert absolute["inchi_key"] == candidate["InChIKey"]
        assert absolute["complete_graph_matches_expected_CID"]
        if scope["stereochemistry"] == "ABSOLUTE":
            assert transformation is None and scope["stereo_centers"] == 1
            assert scope["absolute_configuration"] == "4S"
            assert "(4S)" in reference["Name"] and "/t" in component.inchi
        else:
            assert scope["stereochemistry"] == "RACEMIC" and transformation is not None
            encoded = next(r for r in conversions if r["mode"] == "racemic")
            assert (
                component.inchi == encoded["inchi"] and component.inchi_key == encoded["inchi_key"]
            )
            assert (
                encoded["options"]
                == transformation["options"]
                == ["/Key", "/AuxNone", "/NoLabels", "/SRac"]
            )
            assert transformation["reference_version"] == encodings["reference_version"]
            assert transformation["reference_url"] == encodings["reference_url"]
            assert "/s3" in component.inchi and component.inchi.startswith("InChI=1/")
            assert not any(s.startswith("m") for s in component.inchi.split("/")[1:])
            assert stereochemical_layers(component.inchi) == stereochemical_layers(
                candidate["InChI"]
            )
    geometry = scope["double_bond_geometry"]
    if geometry in ("E", "Z"):
        name = scope["reviewed_geometry_name"]
        assert name in deposit["deposited_names"] and name in component.aliases
        assert re.search(rf"\(\d*{geometry}\)", name)
        assert "/b" in component.inchi
    else:
        assert geometry == "not_applicable" and "/b" not in component.inchi
    # The unsuccessful or name-only CIR lookup remains provenance. It is not
    # upgraded to a CAS resolver by the successful depositor review.
    original_nci = next(r for r in nci["results"] if r["source_CAS"] == cas)
    assert external["NCI_CIR_candidates"] == {
        key: original_nci["representations"][key]["candidates"] for key in ("stdinchi", "names")
    }


def read_deposit_queries(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    reviewed = []
    for query in summary["queries"]:
        if query["status"] != "received":
            continue
        raw = json.loads((directory / query["filename"]).read_text(encoding="utf8"))
        fields = [
            deposit_fields(
                r,
                query["source_CAS"],
                query.get("selected_annotation", {})
                .get("reference", {})
                .get("Name", "Tributylsilane"),
                query.get("source_name", "EPA DSSTox"),
            )
            for r in raw["PC_Substances"]
        ]
        assert fields == query["records"]
        reviewed.append(query)
    return reviewed


def read_depositors(directory):
    queries = read_deposit_queries(directory)
    reviewed = {query["source_CAS"]: query for query in queries}
    assert len(reviewed) == len(queries)
    return reviewed


def check_original_deposit_geometry(component, pubchem, nci, deposits, encodings):
    """Retain original geometry when a standardized compound has discarded it."""
    scope, external = component.metadata["identity_scope"], component.metadata["external_identity"]
    cas = component.cas_number
    selected = {
        "1678-82-6": [384453231],
        "6236-88-0": [135038489, 435377934],
        "17082-12-1": [135075622, 472271540],
    }
    assert cas in selected and scope["source_CAS"] == cas
    assert scope["stereochemistry"] == "ACHIRAL" and scope["stereo_centers"] == 0
    assert scope["absolute_configuration"] == "not_applicable"
    assert "enantiomer_mole_fractions" not in scope
    assert component.inchi.startswith("InChI=1S/")
    assert component.charge == 0 and component.formula == component.inchi.split("/")[1]
    assert not any(
        layer.startswith(("m", "s", "q", "p")) for layer in component.inchi.split("/")[2:]
    )
    assert external["reviewed_SIDs"] == selected[cas]
    assert external["structure_basis"] == "unmodified_original_deposit_SDF"
    original = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
    assert external["PubChem_candidates"] == [candidate_identity(c) for c in original["candidates"]]
    nci_record = next(r for r in nci["results"] if r["source_CAS"] == cas)
    assert external["NCI_CIR_candidates"] == {
        k: nci_record["representations"][k]["candidates"] for k in ("stdinchi", "names")
    }
    candidates = [q for q in deposits if q["source_CAS"] == cas]
    assert external["original_deposit_candidates"] == candidates
    assert encodings["reference_version"] == "InChI 1.07.5"
    assert encodings["block"] == "remaining-depositor-default-SDF-encodings"
    assert encodings["total"] == encodings["completed"] == encodings["encoded"] == 36
    for sid in selected[cas]:
        query = next(q for q in candidates if any(r["SID"]["id"] == sid for r in q["records"]))
        assert query["status"] == "received" and query["http_status"] == 200
        assert query["original_units"] == scope["source_units"]
        deposit = next(r for r in query["records"] if r["SID"]["id"] == sid)
        assert deposit["source_name_matches"] and deposit["source_CAS_in_deposit_names"]
        assert deposit["source_ID"]["str"] == query["source_ID"]
        annotation = query["selected_annotation"]
        reference = annotation["reference"]
        assert cas in annotation["CAS"]
        assert reference["SourceName"] == deposit["source_name"]
        assert reference["SourceID"] == query["source_ID"]
        assert reference["Name"] in deposit["deposited_names"]
        assert any(annotation in c["CAS_matches"] for c in original["candidates"])
        converted = next(r for r in encodings["results"] if r["SID"] == sid)
        assert converted["source_CAS"] == cas and converted["source_ID"] == query["source_ID"]
        assert converted["status"] == "encoded" and converted["returncode"] == 0
        assert converted["options"] == ["/Key", "/AuxNone", "/NoLabels"]
        assert (
            converted["inchi"] == component.inchi and converted["inchi_key"] == component.inchi_key
        )
        assert converted["formula_from_InChI"] == component.formula
        assert not converted["original_candidate_matches"]
        assert component.inchi not in [c["InChI"] for c in original["candidates"]]
        if cas == "17082-12-1":
            assert "(E)-" in reference["Name"]
        else:
            assert "trans" in reference["Name"]
    if cas == "17082-12-1":
        assert scope["double_bond_geometry"] == "E" and "/b14-13+" in component.inchi
        assert "/t" not in component.inchi
    else:
        assert scope["double_bond_geometry"] == "not_applicable"
        assert scope["relative_ring_geometry"] == "trans"
        atoms = [9, 10] if cas == "1678-82-6" else [8, 9]
        assert scope["defined_relative_ring_atoms"] == atoms
        assert f"/t{atoms[0]}-,{atoms[1]}-" in component.inchi


def read_cameo(directory):
    source = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    entry = next(r for r in source["results"] if r["source_CAS"] == "353-42-4")
    assert entry["status"] == "received" and entry["http_status"] == 200
    page = TextPage()
    page.feed((directory / entry["filename"]).read_text(encoding="utf8"))
    name = "BORON TRIFLUORIDE DIMETHYL ETHERATE"
    stoichiometry = "Boron trifluoride compound with methyl ether (1:1)"
    assert name in page.lines and "353-42-4" in page.lines and stoichiometry in page.lines
    return {
        "url": entry["url"],
        "CAS": "353-42-4",
        "name": name,
        "formula": page.lines[page.lines.index("Chemical Formula:") + 1],
        "stoichiometric_name": stoichiometry,
    }


def check_adduct_identity(component, pubchem, cameo):
    assert cameo is not None
    scope, external = component.metadata["identity_scope"], component.metadata["external_identity"]
    assert component.cas_number == scope["source_CAS"] == cameo["CAS"]
    assert scope["material_kind"] == "stoichiometric_adduct"
    assert scope["formula_unit_stoichiometry"] == {"C2H6O": 1, "BF3": 1}
    assert scope["stereochemistry"] == "ACHIRAL"
    assert component.formula == "C2H6BF3O" and cameo["formula"] == "C2H6O.BF3"
    composition = {}
    for formula, coefficient in scope["formula_unit_stoichiometry"].items():
        for element, count in parse_formula(formula).items():
            composition[element] = composition.get(element, 0) + coefficient * count
    assert composition == component.composition
    candidates = next(
        r["candidates"]
        for r in pubchem["original_row_comparison"]
        if r["source_CAS"] == component.cas_number
    )
    assert external["PubChem_candidates"] == [candidate_identity(c) for c in candidates]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["CID"] == external["PubChem_CID"] == 2724165
    assert candidate["InChI"] == component.inchi and candidate["InChIKey"] == component.inchi_key
    assert component.inchi.split("/")[1] == cameo["formula"]
    annotation = external["reviewed_CAMEO_annotation"]
    assert annotation in candidate["CAS_matches"] and component.cas_number in annotation["CAS"]
    assert annotation["reference"]["Name"] == cameo["name"]
    assert annotation["reference"]["URL"] == cameo["url"]
    assert external["CAMEO_identity_fields"] == cameo
    assert cameo["name"] in component.aliases and cameo["stoichiometric_name"] in component.aliases
    assert component.charge == 0


def check_sander_identity(component, gsrs, pubchem, nci, sander, key_structures):
    """Check explicitly reviewed Z materials, retaining unsuccessful old lookups."""
    scope, external = component.metadata["identity_scope"], component.metadata["external_identity"]
    cas = component.cas_number
    assert scope["source_CAS"] == cas == external["Sander_CAS"]
    assert scope["stereochemistry"] == "ACHIRAL" and scope["double_bond_geometry"] == "Z"
    assert not next(r for r in gsrs["results"] if r["source_CAS"] == cas)["candidates"]
    reference = sander[cas]
    assert reference["CAS"] == cas
    assert reference["name"] in component.aliases
    assert reference["formula"] == component.formula
    assert reference["representative_InChIKey"] == component.inchi_key
    assert reference["source"]["original_units"] == scope["source_units"]
    assert reference["source"]["identity_fields"] == external["Sander_identity_fields"]
    assert reference["source"]["http_status"] == 200
    assert reference["source"]["status"] == "complete_identity_fields"
    assert reference["structure_attribution"] == "STRUCTURE (FROM NIST):"
    original = next(r for r in pubchem["original_row_comparison"] if r["source_CAS"] == cas)
    assert external["PubChem_candidates"] == [candidate_identity(c) for c in original["candidates"]]
    nci_record = next(r for r in nci["results"] if r["source_CAS"] == cas)
    assert external["NCI_CIR_candidates"] == {
        k: nci_record["representations"][k]["candidates"] for k in ("stdinchi", "names")
    }
    if "PubChem_key_query" in external:
        query = next(r for r in key_structures["structure_queries"] if r["source_CAS"] == cas)
        assert external["PubChem_key_query"] == query
        assert query["status"] == "received" and query["http_status"] == 200
        assert query["query_InChIKey"] == component.inchi_key and query["all_keys_match"]
        assert query["original_units"] == scope["source_units"]
        assert len(query["candidates"]) == 1
        candidate = query["candidates"][0]
        annotation = next(
            r for r in key_structures["annotation_queries"] if r["CID"] == candidate["CID"]
        )
        assert external["PubChem_key_annotations"] == annotation
        assert annotation["status"] == "received" and annotation["http_status"] == 200
        assert annotation["record_CID"] == candidate["CID"] and annotation["record_matches"]
        assert any(cas in entry["CAS"] for entry in annotation["annotations"])
    else:
        candidate = next(c for c in original["candidates"] if c["CID"] == external["PubChem_CID"])
        assert any(cas in entry["CAS"] for entry in candidate["CAS_matches"])
    assert external["PubChem_source_CAS_annotation"] is True
    assert candidate["CID"] == external["PubChem_CID"]
    assert candidate["InChI"] == component.inchi
    assert candidate["InChIKey"] == component.inchi_key
    assert candidate["MolecularFormula"] == component.formula == component.inchi.split("/")[1]
    assert candidate["IUPACName"] in component.aliases
    assert candidate["IUPACName"].startswith(("(Z)-", "(3Z)-"))
    assert component.charge == 0 and all(
        layer not in component.inchi for layer in ("/t", "/q", "/p")
    )
    assert "/b" in component.inchi


def read_chebi(directory):
    sources = json.loads((directory / "summary.json").read_text(encoding="utf8"))["sources"]
    records = {}
    for source in sources:
        assert source["http_status"] == 200 and source["status"] == "received"
        cid = str(source["ChEBI_ID"])
        lines = (directory / (cid + ".txt")).read_text(encoding="utf8").splitlines()
        records[cid] = {
            "source": source,
            "name": lines[lines.index("ASCII Name") + 1],
            "formula": lines[lines.index("Formula") + 1],
            "inchi": lines[lines.index("InChI") + 1],
            "inchi_key": lines[lines.index("InChIKey") + 1],
            "CAS": [line.removeprefix("CAS:") for line in lines if line.startswith("CAS:")],
        }
    return records


def read_sander(directory):
    source = json.loads((directory / "summary.json").read_text(encoding="utf8"))["sources"][0]
    # The collector's console print failed after both files were saved. Preserve
    # that failure and independently check the saved identity fields here.
    assert source["http_status"] == 200
    lines = (directory / "932-44-5.txt").read_text(encoding="utf8").splitlines()
    formula = "".join(lines[lines.index("FORMULA:") + 1 : lines.index("CAS RN")])
    assert lines[lines.index("CAS RN") + 2] == "932-44-5"
    end = lines.index("-2-propylcyclopentane")
    name = "".join(lines[end - 2 : end + 1]).removeprefix("\u2192 ")
    assert name == "1-methyl-trans-2-propylcyclopentane"
    assert "NIST" in lines[lines.index("STRUCTURE") : lines.index("InChIKey")]
    return {
        "932-44-5": {
            "source": source,
            "CAS": "932-44-5",
            "name": name,
            "formula": formula,
            "representative_InChIKey": lines[lines.index("InChIKey") + 2],
            "structure_attribution": "NIST as cited by Sander; not independently measured",
        }
    }


def read_sander_remaining(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    records = {}
    for source in summary["results"]:
        if source["status"] != "complete_identity_fields":
            continue
        cas = source["source_CAS"]
        page = IdentityPage()
        page.feed((directory / f"{cas}.html").read_text(encoding="utf8"))
        fields = page.identity()
        assert fields == source["identity_fields"]
        assert fields["CASRN"] == cas
        records[cas] = {
            "source": source,
            "CAS": cas,
            "name": fields["name"],
            "formula": fields["FORMULA"],
            "representative_InChIKey": fields["InChIKey"],
            "structure_attribution": fields["structure_attribution"],
        }
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--gsrs", type=Path, required=True)
    parser.add_argument("--pubchem", type=Path, required=True)
    parser.add_argument("--racemic-summary", type=Path)
    parser.add_argument("--chebi-directory", type=Path)
    parser.add_argument("--nci-summary", type=Path)
    parser.add_argument("--sander-directory", type=Path)
    parser.add_argument("--sander-remaining-directory", type=Path)
    parser.add_argument("--key-structures", type=Path)
    parser.add_argument("--depositor-directory", type=Path)
    parser.add_argument("--material-depositor-directory", type=Path)
    parser.add_argument("--depositor-encodings", type=Path)
    parser.add_argument("--material-pages-directory", type=Path)
    parser.add_argument("--remaining-depositor-directory", type=Path)
    parser.add_argument("--remaining-depositor-encodings", type=Path)
    parser.add_argument("--institutional-directory", type=Path)
    parser.add_argument("--srs-directory", type=Path)
    parser.add_argument("--srs-current-directory", type=Path)
    parser.add_argument("--srs-fenchone-directory", type=Path)
    parser.add_argument("--primary-documents-directory", type=Path)
    parser.add_argument("--printed-encoding-directory", type=Path)
    parser.add_argument("--indexed-identities-directory", type=Path)
    parser.add_argument("--crc84-directory", type=Path)
    parser.add_argument("--crc84-encoding-directory", type=Path)
    parser.add_argument("--crc84-additional-encoding-directory", type=Path)
    parser.add_argument(
        "--components",
        type=Path,
        default=Path("examples/fixtures/crc_electro_optical_components.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-components", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=14)
    parser.add_argument("--expected-negatives", type=int, default=2)
    parser.add_argument("--expected-identity-negatives", type=int, default=0)
    parser.add_argument("--block", default="reviewed-identity-consumers-1")
    parser.add_argument("--check-source-gaps", action="store_true")
    args = parser.parse_args()
    if args.check_source_gaps:
        example = runpy.run_path(str(
            Path(__file__).resolve().parents[2] / "examples/crc_electro_optical_inputs.py"
        ))
        inspect_source_identity = example["inspect_source_identity"]
        reviewed_property_input = example["reviewed_property_input"]
        source_gap_error = example["SourceIdentityGapError"]

    def read(path):
        return json.loads(path.read_text(encoding="utf8"))

    prior, gsrs, pubchem, mapping = map(
        read,
        (
            args.prior_summary,
            args.gsrs,
            args.pubchem,
            args.components,
        ),
    )
    assert prior["total"] == 17268 and prior["block"] == "sources-consumers"
    gaps = [r for r in prior["results"] if "component_gap" in r.get("metrics", {})]
    assert len(gaps) == 285
    racemic = read(args.racemic_summary) if args.racemic_summary else None
    chebi = read_chebi(args.chebi_directory) if args.chebi_directory else None
    nci = read(args.nci_summary) if args.nci_summary else None
    sander = read_sander(args.sander_directory) if args.sander_directory else None
    if args.sander_remaining_directory:
        sander = {**(sander or {}), **read_sander_remaining(args.sander_remaining_directory)}
    key_structures = read(args.key_structures) if args.key_structures else None
    depositors = read_depositors(args.depositor_directory) if args.depositor_directory else None
    if args.material_depositor_directory:
        additional = read_depositors(args.material_depositor_directory)
        assert not set(additional).intersection(depositors or {})
        depositors = {**(depositors or {}), **additional}
    depositor_encodings = read(args.depositor_encodings) if args.depositor_encodings else None
    cameo = read_cameo(args.material_pages_directory) if args.material_pages_directory else None
    remaining_deposits = (
        read_deposit_queries(args.remaining_depositor_directory)
        if args.remaining_depositor_directory
        else None
    )
    remaining_encodings = (
        read(args.remaining_depositor_encodings) if args.remaining_depositor_encodings else None
    )
    institutional = (
        read_institutional_identity(args.institutional_directory)
        if args.institutional_directory else None
    )
    srs = (
        read_srs_sources(
            args.srs_directory, args.srs_current_directory, args.srs_fenchone_directory
        )
        if args.srs_directory else None
    )
    documents = (
        read_documents(args.primary_documents_directory, args.printed_encoding_directory,
                       args.indexed_identities_directory)
        if args.primary_documents_directory else None
    )
    crc84_documents = (
        crc84.read_documents(args.crc84_directory, args.crc84_encoding_directory,
                             args.crc84_additional_encoding_directory)
        if args.crc84_directory else None
    )
    source_units = {}
    for row in gaps:
        source_units.setdefault(row["inputs"]["raw"]["CAS"], []).append(row["unit"])

    def qualify(component):
        check_identity(
            component,
            gsrs,
            pubchem,
            racemic,
            chebi,
            nci,
            sander,
            key_structures,
            depositors,
            depositor_encodings,
            cameo,
            remaining_deposits,
            remaining_encodings,
            institutional,
            srs,
            documents,
            crc84_documents,
            source_units,
        )

    components = {c["cas_number"]: ComponentSpec.from_dict(c) for c in mapping["components"]}
    assert len(components) == len(mapping["components"]) == args.expected_components
    assert sum(r["inputs"]["raw"]["CAS"] in components for r in gaps) == args.expected_rows
    args.output.mkdir(parents=True, exist_ok=False)
    database = ElectroOpticalDatabase(Path(chemicals.__file__).parent)
    rows, negative_checks = [], []
    started = time.perf_counter()
    for original in gaps:
        inputs = original["inputs"]
        cas, source = inputs["raw"]["CAS"], inputs["source"]
        row = {
            "unit": original["unit"],
            "inputs": inputs,
            "original_gap": original["metrics"]["component_gap"],
        }
        if cas not in components:
            row.update(status="unresolved", reason="no_reviewed_explicit_component")
            if args.check_source_gaps:
                try:
                    diagnostic = inspect_source_identity(mapping, cas, original["unit"])
                    assert diagnostic["status"] == "source_gap"
                    assert original["unit"] in diagnostic["source_units"]
                    assert diagnostic["category"] and diagnostic["reason"]
                    assert inspect_source_identity(
                        json.loads(json.dumps(mapping)), cas, original["unit"]
                    ) == diagnostic
                    record = database.record(cas, method=source)
                    raw_fields = dict(zip(record.field_names, record.raw_fields, strict=True))
                    assert raw_fields == inputs["raw"]
                    field = "Permittivity" if source == "CRC_PERMITTIVITY" else "RI"
                    assert record.value(field).value == float(inputs["raw"][field])
                    try:
                        reviewed_property_input(
                            database, mapping, cas, method=source, source_point=True
                        )
                    except source_gap_error as exc:
                        assert exc.diagnostic == diagnostic
                    else:
                        raise AssertionError("unresolved fixture material accepted")
                    try:
                        native_card(database, record, source_point=True)
                    except UnknownIdentifierError:
                        pass
                    else:
                        raise AssertionError("original identifier path silently supplied component")
                    row.update(identity_diagnostic=diagnostic, gap_check="passed")
                except Exception:
                    row.update(gap_check="failed", failure=traceback.format_exc())
        else:
            component = components[cas]
            try:
                qualify(component)
                assert original["unit"] in component.metadata["identity_scope"]["source_units"]
                record = database.record(cas, method=source)
                assert (
                    dict(zip(record.field_names, record.raw_fields, strict=True)) == inputs["raw"]
                )
                # F2 source_units enumerates rows from zero; database data rows start at one.
                assert record.row_number == inputs["row"] + 1
                card = native_card(database, record, component=component, source_point=True)
                assert card.package.component.to_dict() == component.to_dict()
                if args.check_source_gaps:
                    reviewed = reviewed_property_input(
                        database, mapping, cas, method=source, source_point=True
                    )
                    assert reviewed.to_dict() == card.to_dict()
                consumer = (
                    dielectric_source_unit if source == "CRC_PERMITTIVITY" else optical_source_unit
                )
                metrics = consumer(database, record, inputs["raw"], component=component)
                assert metrics["input_card"] == 1 and metrics["standalone_correlation"] == 0
                row.update(status="passed", component=component.to_dict(), metrics=metrics)
            except Exception:
                row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 25 == 0 or len(rows) == 285:
            print(
                f"{len(rows)}/285 original rows: {dict(Counter(r['status'] for r in rows))}",
                flush=True,
            )
    pairs = (
        ("590-11-4", "590-12-5"),
        ("506-23-0", "544-73-0"),
        ("4628-21-1", "4894-61-5"),
        ("7318-67-4", "7319-00-8"),
        ("20972-43-4", "21650-65-7"),
    )
    bindings = [
        (a, b)
        for pair in pairs
        for a, b in (pair, pair[::-1])
        if a in components and b in components
    ]
    assert len(bindings) == args.expected_negatives
    for source_cas, wrong_cas in bindings:
        check = {"source_CAS": source_cas, "wrong_component_CAS": wrong_cas}
        try:
            database.property_input(
                source_cas,
                components[wrong_cas],
                method="CRC_PERMITTIVITY" if source_cas == "20972-43-4" else "CRC_RI",
                source_point=True,
            )
        except ValueError as exc:
            assert str(exc) == "electro-optical source identity does not match component"
            check.update(status="passed", rejection=str(exc))
        else:
            check.update(status="failed", reason="different_CAS_accepted")
        negative_checks.append(check)
    identity_negatives = []
    for component in components.values():
        transformation = component.metadata.get("identity_transformation")
        if (
            transformation is None
            or component.metadata["identity_scope"].get("evidence_route")
            in {"reviewed_pubchem_depositor", "reviewed_srs_former_CAS_gsrs",
                DOCUMENT_ROUTE, crc84.ROUTE}
        ):
            continue
        for mode in ("absolute_representative", "relative_single_enantiomer"):
            data = component.to_dict()
            if mode == "absolute_representative":
                data["inchi"] = transformation["representative_InChI"]
                data["inchi_key"] = transformation["representative_InChIKey"]
            else:
                data["inchi"] = data["inchi"].replace("/s3", "/s2")
            check = {"source_CAS": component.cas_number, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    for cas, wrong_cas in (("20972-43-4", "21650-65-7"), ("21650-65-7", "20972-43-4")):
        if cas not in components or wrong_cas not in components:
            continue
        data = components[cas].to_dict()
        data.update(inchi=components[wrong_cas].inchi, inchi_key=components[wrong_cas].inchi_key)
        check = {"source_CAS": cas, "wrong_identity": "opposite_double_bond_geometry"}
        try:
            qualify(ComponentSpec.from_dict(data))
        except AssertionError:
            check["status"] = "passed"
        else:
            check.update(status="failed", reason="wrong_material_identity_accepted")
        identity_negatives.append(check)
    for component in components.values():
        scope = component.metadata["identity_scope"]
        if scope.get("evidence_route") not in (
            "reviewed_nci_cir_pubchem",
            "reviewed_CAS_annotation_with_CIR_structure",
        ):
            continue
        variants = []
        if scope["double_bond_geometry"] in ("E", "Z"):
            data = component.to_dict()
            data["inchi"] = component.inchi.split("/b")[0]
            variants.append(("missing_double_bond_geometry", data))
        if "reviewed_geometry_annotation" in component.metadata["external_identity"]:
            data = component.to_dict()
            base, geometry = data["inchi"].split("/b")
            assert geometry.endswith(("-", "+")) and "/" not in geometry
            data["inchi"] = base + "/b" + geometry[:-1] + ("+" if geometry[-1] == "-" else "-")
            variants.append(("opposite_double_bond_geometry", data))
        if "reviewed_racemate_annotation" in component.metadata["external_identity"]:
            for mode in (
                "racemate_scope_lost",
                "unequal_enantiomer_fractions",
                "absolute_enantiomer",
            ):
                data = json.loads(json.dumps(component.to_dict()))
                target_scope = data["metadata"]["identity_scope"]
                if mode == "racemate_scope_lost":
                    target_scope["stereochemistry"] = "ACHIRAL"
                elif mode == "unequal_enantiomer_fractions":
                    target_scope["enantiomer_mole_fractions"] = [0.75, 0.25]
                else:
                    data["inchi"] += f"/t{target_scope['stereocenter_InChI_atom']}-/m0/s1"
                variants.append((mode, data))
        if component.cas_number == "590-13-6":
            for candidate in component.metadata["external_identity"]["PubChem_candidates"]:
                if candidate["CID"] == 643834:
                    continue
                data = json.loads(json.dumps(component.to_dict()))
                data.update(inchi=candidate["InChI"], inchi_key=candidate["InChIKey"])
                data["metadata"]["external_identity"]["PubChem_CID"] = candidate["CID"]
                variants.append((f"wrong_source_candidate_{candidate['CID']}", data))
        for mode, data in variants:
            check = {"source_CAS": component.cas_number, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    reviewed_sander = [
        c
        for c in components.values()
        if c.metadata["identity_scope"].get("evidence_route") == "reviewed_sander_pubchem"
    ]
    for index, component in enumerate(reviewed_sander):
        variants = []
        for mode in (
            "missing_double_bond_geometry",
            "opposite_double_bond_geometry",
            "wrong_Sander_CAS",
        ):
            data = json.loads(json.dumps(component.to_dict()))
            if mode == "missing_double_bond_geometry":
                data["inchi"] = data["inchi"].split("/b")[0]
            elif mode == "opposite_double_bond_geometry":
                base, geometry = data["inchi"].split("/b")
                assert geometry.endswith("-") and "/" not in geometry
                data["inchi"] = base + "/b" + geometry[:-1] + "+"
            else:
                data["metadata"]["external_identity"]["Sander_CAS"] = reviewed_sander[
                    (index + 1) % len(reviewed_sander)
                ].cas_number
            variants.append((mode, data))
        if component.cas_number == "1501-60-6":
            data = json.loads(json.dumps(component.to_dict()))
            data["inchi"] = data["metadata"]["external_identity"]["NCI_CIR_candidates"]["stdinchi"][
                0
            ]["items"][0]
            assert data["inchi"] != component.inchi
            variants.append(("wrong_NCI_connectivity", data))
        for mode, data in variants:
            check = {"source_CAS": component.cas_number, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    reviewed_depositors = [
        c
        for c in components.values()
        if c.metadata["identity_scope"].get("evidence_route") == "reviewed_pubchem_depositor"
    ]
    for index, component in enumerate(reviewed_depositors):
        variants = []
        data = json.loads(json.dumps(component.to_dict()))
        data["metadata"]["external_identity"]["depositor_source_ID"] = reviewed_depositors[
            (index + 1) % len(reviewed_depositors)
        ].metadata["external_identity"]["depositor_source_ID"]
        variants.append(("wrong_depositor_source_ID", data))
        if component.metadata["identity_scope"]["stereochemistry"] == "ABSOLUTE":
            for mode in ("missing_absolute_configuration", "opposite_absolute_configuration"):
                data = json.loads(json.dumps(component.to_dict()))
                data["inchi"] = (
                    data["inchi"].split("/t")[0]
                    if mode.startswith("missing")
                    else data["inchi"].replace("/m0", "/m1")
                )
                variants.append((mode, data))
        transformation = component.metadata.get("identity_transformation")
        if transformation is not None:
            for mode in (
                "absolute_representative",
                "relative_single_enantiomer",
                "unequal_enantiomer_fractions",
            ):
                data = json.loads(json.dumps(component.to_dict()))
                if mode == "absolute_representative":
                    data["inchi"] = transformation["representative_InChI"]
                    data["inchi_key"] = transformation["representative_InChIKey"]
                elif mode == "relative_single_enantiomer":
                    data["inchi"] = data["inchi"].replace("/s3", "/s2")
                else:
                    data["metadata"]["identity_scope"]["enantiomer_mole_fractions"] = [0.75, 0.25]
                variants.append((mode, data))
        if component.metadata["identity_scope"]["double_bond_geometry"] in ("E", "Z"):
            for mode in ("missing_double_bond_geometry", "opposite_double_bond_geometry"):
                data = json.loads(json.dumps(component.to_dict()))
                base, geometry = data["inchi"].split("/b")
                assert geometry.endswith(("-", "+")) and "/" not in geometry
                data["inchi"] = (
                    base
                    if mode.startswith("missing")
                    else (base + "/b" + geometry[:-1] + ("+" if geometry[-1] == "-" else "-"))
                )
                variants.append((mode, data))
        if component.cas_number == "31849-78-2":
            for candidate in component.metadata["external_identity"]["PubChem_candidates"]:
                if candidate["CID"] == 5463114:
                    continue
                data = json.loads(json.dumps(component.to_dict()))
                data.update(inchi=candidate["InChI"], inchi_key=candidate["InChIKey"])
                data["metadata"]["external_identity"]["PubChem_CID"] = candidate["CID"]
                variants.append((f"wrong_source_candidate_{candidate['CID']}", data))
        for mode, data in variants:
            check = {"source_CAS": component.cas_number, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    if "353-42-4" in components:
        component = components["353-42-4"]
        for mode in (
            "wrong_adduct_stoichiometry",
            "free_ether_fragment",
            "undefined_mixture_scope",
        ):
            data = json.loads(json.dumps(component.to_dict()))
            if mode == "wrong_adduct_stoichiometry":
                data["metadata"]["identity_scope"]["formula_unit_stoichiometry"]["BF3"] = 2
            elif mode == "free_ether_fragment":
                data["inchi"] = "InChI=1S/C2H6O/c1-3-2/h1-2H3"
            else:
                data["metadata"]["identity_scope"]["material_kind"] = "mixture_unknown_composition"
            check = {"source_CAS": component.cas_number, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    for cas in ("1678-82-6", "6236-88-0", "17082-12-1"):
        if cas not in components:
            continue
        component = components[cas]
        modes = [
            "wrong_original_deposit_SID",
            "generic_standardized_CID",
            "opposite_original_geometry",
        ]
        if cas != "17082-12-1":
            modes.append("imposed_absolute_configuration")
        for mode in modes:
            data = json.loads(json.dumps(component.to_dict()))
            if mode == "wrong_original_deposit_SID":
                data["metadata"]["external_identity"]["reviewed_SIDs"][0] = 315702911
            elif mode == "generic_standardized_CID":
                candidate = data["metadata"]["external_identity"]["PubChem_candidates"][0]
                data["inchi"], data["inchi_key"] = candidate["InChI"], candidate["InChIKey"]
            elif mode == "opposite_original_geometry":
                data["inchi"] = (
                    data["inchi"].replace("/b14-13+", "/b14-13-")
                    if cas == "17082-12-1"
                    else data["inchi"][:-1] + "+"
                )
            else:
                data["metadata"]["identity_scope"]["absolute_configuration"] = "R,R"
            check = {"source_CAS": cas, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    if "13520-90-6" in components:
        for mode in (
            "neighbor_dibromide_CAS", "dibromide_formula", "oxygen_free_formula",
            "unknown_mixture", "charged_material", "invented_InChI", "wrong_source_page",
            "invented_absolute_stereo",
        ):
            data = json.loads(json.dumps(components["13520-90-6"].to_dict()))
            if mode == "neighbor_dibromide_CAS":
                data["cas_number"] = "13520-89-3"
            elif mode in ("dibromide_formula", "oxygen_free_formula"):
                data["formula"] = "VOBr2" if mode == "dibromide_formula" else "VBr3"
                data["molecular_weight_g_mol"] = None
            elif mode == "unknown_mixture":
                data["metadata"]["identity_scope"]["material_kind"] = "mixture_unknown_composition"
            elif mode == "charged_material":
                data["charge"] = 1
            elif mode == "invented_InChI":
                data["inchi"] = "InChI=1S/V"
            elif mode == "wrong_source_page":
                data["metadata"]["external_identity"]["INERIS_identity_fields"]["page"] = 12
            else:
                data["metadata"]["identity_scope"]["stereochemistry"] = "ABSOLUTE"
            check = {"source_CAS": "13520-90-6", "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    for component in components.values():
        route = component.metadata["identity_scope"].get("evidence_route")
        if route != "reviewed_srs_former_CAS_gsrs":
            continue
        for mode in ("wrong_SRS_ITN", "wrong_current_CAS", "incorrect_CAS_relation",
                     "wrong_current_GSRS_uuid", "lost_racemate_scope", "unequal_enantiomers",
                     "imposed_absolute_configuration", "wrong_complete_structure"):
            data = json.loads(json.dumps(component.to_dict()))
            scope = data["metadata"]["identity_scope"]
            external = data["metadata"]["external_identity"]
            if mode == "wrong_SRS_ITN":
                external["SRS_ITN"] = "not-the-source-ITN"
            elif mode == "wrong_current_CAS":
                scope["GSRS_primary_CAS"] = component.cas_number
            elif mode == "incorrect_CAS_relation":
                scope["source_CAS_code_type"] = "Incorrect"
            elif mode == "wrong_current_GSRS_uuid":
                external["GSRS_uuid"] = "not-the-source-uuid"
            elif mode == "lost_racemate_scope":
                scope["stereochemistry"] = "ACHIRAL"
            elif mode == "unequal_enantiomers":
                scope["enantiomer_mole_fractions"] = [0.75, 0.25]
            elif mode == "imposed_absolute_configuration":
                scope["absolute_configuration"] = "R"
            else:
                data["inchi"] = "InChI=1S/CH4/h1H4"
            check = {"source_CAS": component.cas_number, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
        if component.cas_number == "18492-37-0":
            for mode in (
                "absolute_representative", "missing_relative_layer", "wrong_relative_geometry",
                "relative_not_racemic", "generic_InChIKey", "wrong_encoding_option",
            ):
                data = json.loads(json.dumps(component.to_dict()))
                transformation = data["metadata"]["identity_transformation"]
                if mode == "absolute_representative":
                    data["inchi"] = transformation["representative_InChI"]
                    data["inchi_key"] = transformation["representative_InChIKey"]
                elif mode == "missing_relative_layer":
                    data["inchi"] = re.sub(r"/t[^/]+", "", data["inchi"])
                elif mode == "wrong_relative_geometry":
                    data["inchi"] = data["inchi"].replace("/t7-,10+", "/t7-,10-")
                elif mode == "relative_not_racemic":
                    data["inchi"] = data["inchi"].replace("/s3", "/s2")
                elif mode == "generic_InChIKey":
                    data["inchi_key"] = "LHXDLQBQYFFVNW-UHFFFAOYSA-N"
                else:
                    transformation["options"][-1] = "/SRel"
                check = {"source_CAS": component.cas_number, "wrong_identity": mode}
                try:
                    qualify(ComponentSpec.from_dict(data))
                except AssertionError:
                    check["status"] = "passed"
                else:
                    check.update(status="failed", reason="wrong_material_identity_accepted")
                identity_negatives.append(check)
    for component in components.values():
        route = component.metadata["identity_scope"].get("evidence_route")
        if route not in {DOCUMENT_ROUTE, crc84.ROUTE}:
            continue
        variants = (crc84.negative_variants(component) if route == crc84.ROUTE
                    else document_negative_variants(component))
        for mode, data in variants:
            check = {"source_CAS": component.cas_number, "wrong_identity": mode}
            try:
                qualify(ComponentSpec.from_dict(data))
            except AssertionError:
                check["status"] = "passed"
            else:
                check.update(status="failed", reason="wrong_material_identity_accepted")
            identity_negatives.append(check)
    states = dict(Counter(r["status"] for r in rows))
    reviewed_passed = (
        states.get("passed", 0) == args.expected_rows
        and len(identity_negatives) == args.expected_identity_negatives
        and all(c["status"] == "passed" for c in negative_checks + identity_negatives)
    )
    gap_checks = dict(Counter(r["gap_check"] for r in rows if "gap_check" in r))
    gaps_passed = (
        args.check_source_gaps and gap_checks == {"passed": 285 - args.expected_rows}
        and len(mapping["source_gaps"]) == len({
            r["inputs"]["raw"]["CAS"] for r in rows if r["status"] == "unresolved"
        })
        and mapping["unresolved_source_rows"] == 285 - args.expected_rows
    )
    result = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": args.block,
        "total": 285,
        "completed": len(rows),
        "reviewed_CAS_count": len(components),
        "reviewed_source_rows": args.expected_rows,
        "states": states,
        "new_verified_source_mappings": states.get("passed", 0),
        "reviewed_block_passed": reviewed_passed,
        "complete_and_passed": states.get("passed", 0) == 285,
        "source_gap_checks": gap_checks,
        "functional_acceptance_passed": reviewed_passed and gaps_passed,
        "comparison_count": sum(len(r.get("metrics", {}).get("comparisons", [])) for r in rows),
        "negative_checks": negative_checks,
        "identity_negative_checks": identity_negatives,
        "expected_identity_negative_count": args.expected_identity_negatives,
        "supplementary_identity_sources": chebi,
        "supplementary_sander_sources": sander,
        "NCI_identity_source": str(args.nci_summary) if args.nci_summary else None,
        "supplementary_key_structures": key_structures,
        "supplementary_depositor_sources": depositors,
        "supplementary_depositor_encodings": depositor_encodings,
        "supplementary_CAMEO_source": cameo,
        "remaining_original_deposits": remaining_deposits,
        "remaining_original_encodings": remaining_encodings,
        "institutional_identity_source": institutional,
        "SRS_identity_source": str(args.srs_directory) if args.srs_directory else None,
        "SRS_current_GSRS_source": (
            str(args.srs_current_directory) if args.srs_current_directory else None
        ),
        "SRS_fenchone_encoding": (
            str(args.srs_fenchone_directory) if args.srs_fenchone_directory else None
        ),
        "primary_document_identities": documents,
        "CRC84_document_identities": crc84_documents,
        "results": rows,
        "source_counts": dict(Counter(r["inputs"]["source"] for r in gaps)),
        "qualified_source_counts": dict(
            Counter(r["inputs"]["source"] for r in rows if r["status"] == "passed")
        ),
        "scope": (
            "Explicit reviewed components; unresolved rows remain in the original denominator. "
            "No automatic alias remapping."
        ),
        "seconds": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "results",
                    "negative_checks",
                    "identity_negative_checks",
                    "supplementary_identity_sources",
                    "supplementary_sander_sources",
                    "supplementary_key_structures",
                    "supplementary_depositor_sources",
                    "supplementary_depositor_encodings",
                    "supplementary_CAMEO_source",
                    "remaining_original_deposits",
                    "remaining_original_encodings",
                    "primary_document_identities",
                    "CRC84_document_identities",
                }
            },
            indent=2,
        )
    )
    return not (reviewed_passed and (gaps_passed if args.check_source_gaps else True))


if __name__ == "__main__":
    raise SystemExit(main())
