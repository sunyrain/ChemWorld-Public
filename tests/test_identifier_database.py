"""Native identifier query behavior without a reference-package dependency."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chemworld.data.identifiers import (
    IDENTIFIER_SOURCES,
    AmbiguousIdentifierError,
    ConflictingIdentifierDataError,
    IdentifierConversionError,
    IdentifierDatabase,
    IdentifierSourceFormatError,
    IdentifierSourceUnavailableError,
    MissingIdentifierFieldError,
    UnknownIdentifierError,
    UnknownIdentifierMethodError,
)
from chemworld.physchem import ComponentSpec, MixtureSpec

WATER = (
    "962",
    "7732-18-5",
    "H2O",
    "18.015",
    "O",
    "H2O/h1H2",
    "XLYOFNOQVPJJNP-UHFFFAOYSA-N",
    "oxidane",
    "water",
    "aqua",
    "water",
)
ETHANOL = (
    "702",
    "64-17-5",
    "C2H6O",
    "46.069",
    "CCO",
    "C2H6O/c1-2-3/h3H,2H2,1H3",
    "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    "ethanol",
    "ethanol",
    "ethyl alcohol",
)
ETHER = (
    "8254",
    "115-10-6",
    "C2H6O",
    "46.069",
    "COC",
    "",
    "",
    "methoxymethane",
    "dimethyl ether",
    "ether",
)


def write_source(directory, source_id, rows):
    source = next(item for item in IDENTIFIER_SOURCES if item.source_id == source_id)
    path = directory / source.filename
    if source.kind == "preferences":
        path.write_text(json.dumps(rows), encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(rows)
    return path


@pytest.fixture
def database(tmp_path):
    for source in IDENTIFIER_SOURCES:
        if source.kind == "compound":
            write_source(
                tmp_path,
                source.source_id,
                (WATER, ETHANOL, ETHER) if source.source_id == "pubchem-small" else (),
            )
    return IdentifierDatabase(tmp_path)


def test_lazy_source_boundaries_and_complete_catalog(database):
    assert len(database.catalog) == 13
    assert database.loaded_sources == ()
    assert database.element("Fe").atomic_number == 26
    assert database.loaded_sources == ()
    rows = database.search("water", source_ids=("pubchem-small",))
    assert rows[0].raw_fields == WATER
    assert database.loaded_sources == ("pubchem-small",)
    assert rows[0].synonyms == ("aqua", "water")
    assert rows[0].to_dict()["source"]["version"] == "chemicals-identifiers-1.5.2"
    assert "not established" in rows[0].source.license


def test_inspect_search_keeps_source_failures_and_structural_conflicts(database):
    report = database.inspect_search(
        "C2H6O", kind="formula", source_ids=("pubchem-small", "absent"))
    assert len(report["candidates"]) == 2 and report["identity_conflict"]
    assert "smiles" in report["conflicting_fields"]
    assert report["sources"][1]["status"] == "unknown_method"
    assert report["selected_record_id"] is None
    assert json.loads(json.dumps(report)) == report
    assert database.inspect_search("absent", source_ids=("pubchem-small",))["sources"][0][
        "status"] == "missing_identity"
    empty = IdentifierDatabase(database.source_directory / "absent")
    report = empty.inspect_search("water", source_ids=("pubchem-small",))
    assert report["sources"][0]["status"] == "source_unavailable"
    assert report["candidates"] == []
    with pytest.raises(TypeError):
        database.inspect_search("water", source_ids="pubchem-small")


@pytest.mark.parametrize(
    "identity,kind",
    [
        ("  WATER  ", "auto"),
        ("aqua", "name"),
        ("7732-18-5", "auto"),
        (7732185, "auto"),
        ("7732-18-5", "source_key"),
        (962, "pubchem"),
        ("PubChem=962", "auto"),
        ("H2O", "formula"),
        ("Formula=OH2", "auto"),
        ("O", "smiles"),
        ("SMILES=O", "auto"),
        ("InChI=1/H2O/h1H2", "auto"),
        ("H2O/h1H2", "inchi"),
        ("InChI=1S/H2O/h1H2", "inchi"),
        ("InChIKey=XLYOFNOQVPJJNP-UHFFFAOYSA-N", "auto"),
        ("XLYOFNOQVPJJNP-UHFFFAOYSA-N", "inchikey"),
    ],
)
def test_query_kinds_recover_source_row(database, identity, kind):
    assert database.resolve(identity, kind=kind).raw_fields == WATER


def test_formula_isomers_and_duplicate_rows_are_not_dropped(database, tmp_path):
    write_source(tmp_path, "example-user", (ETHANOL, ETHANOL))
    matches = database.search("C2H6O", kind="formula")
    assert len(matches) == 4
    assert len({r.record_id for r in matches}) == 4
    with pytest.raises(AmbiguousIdentifierError) as caught:
        database.resolve("C2H6O", kind="formula")
    assert len(caught.value.candidates) == 4
    assert database.resolve("CCO", kind="smiles").cas_number == "64-17-5"


def test_same_cas_conflict_requires_explicit_source_and_preserves_candidates(database, tmp_path):
    other = (*WATER[:3], "18.02", *WATER[4:])
    write_source(tmp_path, "example-user", (other,))
    with pytest.raises(ConflictingIdentifierDataError) as caught:
        database.resolve("water")
    assert caught.value.fields == ("molecular_weight_g_mol",)
    assert len(caught.value.candidates) == 2
    candidates = database.field_candidates("water", "molecular_weight_g_mol")
    assert {c.value for c in candidates} == {18.015, 18.02}
    assert database.resolve("water", source_priority=("example-user",)).raw_fields == other
    assert len(database.search("water")) == 2


def test_preference_files_are_only_used_when_requested(database, tmp_path):
    write_source(
        tmp_path,
        "organic-preferences",
        {"preferred_cas": ["64-17-5"], "unpreferred_cas": ["115-10-6"]},
    )
    with pytest.raises(AmbiguousIdentifierError):
        database.resolve("C2H6O", kind="formula")
    assert "organic-preferences" not in database.loaded_sources
    chosen = database.resolve("C2H6O", kind="formula", preference_sources=("organic-preferences",))
    assert chosen.cas_number == "64-17-5"
    assert len(database.search("C2H6O", kind="formula")) == 2


def test_element_symbol_and_smiles_mean_different_identities(database, tmp_path):
    carbon = ("5462310", "7440-44-0", "C", "12.011", "[C]", "C", "", "carbon", "atomic carbon")
    methane = ("297", "74-82-8", "CH4", "16.043", "C", "CH4/h1H4", "", "methane", "methane")
    write_source(tmp_path, "pubchem-small", (carbon, methane))
    assert database.resolve("C").formula == "C"
    assert database.resolve("6").formula == "C"
    assert database.resolve("SMILES=C").formula == "CH4"


def test_missing_source_unknown_species_method_and_field_are_distinct(database, tmp_path):
    assert database.search("not present") == ()
    with pytest.raises(UnknownIdentifierError):
        database.resolve("not present")
    with pytest.raises(UnknownIdentifierMethodError):
        database.search("water", kind="unknown")
    with pytest.raises(UnknownIdentifierMethodError):
        database.records("not-a-source")
    with pytest.raises(UnknownIdentifierMethodError):
        database.field_candidates("water", "critical_temperature_K")
    with pytest.raises(IdentifierSourceUnavailableError):
        database.mixture("air")
    write_source(tmp_path, "example-user", (("1",),))
    fresh = IdentifierDatabase(tmp_path)
    with pytest.raises(IdentifierSourceFormatError):
        fresh.records("example-user")
    assert fresh.loaded_sources == ()


@pytest.mark.parametrize(
    "identity,kind",
    [
        (True, "cas"),
        (1.0, "pubchem"),
        (-1, "pubchem"),
        ("7732-18-4", "auto"),
        ("", "name"),
        ("064-17-5", "cas"),
        (None, "auto"),
    ],
)
def test_invalid_query_rejected_before_io(tmp_path, identity, kind):
    db = IdentifierDatabase(tmp_path)
    with pytest.raises(ValueError):
        db.search(identity, kind=kind)
    assert db.loaded_sources == ()


def test_compound_conversion_uses_original_native_contract(database):
    component = database.component("water")
    assert component.cas_number == "7732-18-5"
    assert component.composition == {"H": 2, "O": 1}
    assert component.charge == 0
    assert component.aliases == ("water", "oxidane", "aqua")
    assert component.inchi == "InChI=1/H2O/h1H2"
    assert ComponentSpec.from_dict(json.loads(json.dumps(component.to_dict()))) == component
    assert database.ids_to_cas(["water", "ethanol"]) == ("7732-18-5", "64-17-5")
    assert database.molecular_weight("CCO") == 46.069
    with pytest.raises(AmbiguousIdentifierError):
        database.cas_from_any("C2H6O")


@pytest.mark.parametrize(
    "field,value,error",
    [
        (2, "", MissingIdentifierFieldError),
        (2, "TcO2", IdentifierConversionError),
        (2, "H0", IdentifierConversionError),
        (3, "nan", IdentifierConversionError),
        (3, "1000", IdentifierConversionError),
        (1, "7732-18-4", IdentifierConversionError),
    ],
)
def test_unusable_source_record_remains_queryable_with_explicit_conversion_error(
    database,
    tmp_path,
    field,
    value,
    error,
):
    row = list(WATER)
    row[field] = value
    write_source(tmp_path, "example-user", (row,))
    record = database.resolve("water", source_ids=("example-user",))
    assert record.raw_fields[field] == value
    with pytest.raises(error):
        record.to_component()


def test_fake_ids_preferences_and_dippr_are_separate_from_valid_cas(database, tmp_path):
    key = "2099995000-00-0"
    write_source(tmp_path, "fake-registry", ((key, "smiles", "[OH-].[Y+3]", "", ""),))
    write_source(tmp_path, "inorganic-preferences", {"preferred_cas": [], "unpreferred_cas": [key]})
    write_source(tmp_path, "dippr-2014", (("7732-18-5",), ("64-17-5",)))
    assert database.fake_records(key)[0].raw_fields[2] == "[OH-].[Y+3]"
    assert database.preferences("inorganic-preferences").unpreferred_keys == (key,)
    assert database.dippr_compounds() == {"7732-18-5", "64-17-5"}
    with pytest.raises(ValueError, match="CAS"):
        database.search(key, kind="cas")


def test_mixture_basis_is_explicit_and_other_basis_is_derived(database, tmp_path):
    raw = (
        "Example",
        "synthetic fixture",
        "2",
        "7732-18-5",
        "64-17-5",
        "water",
        "ethanol",
        "0.4",
        "0.6",
        "0.9",
        "0.1",
        "Example-Mix",
    )
    source = write_source(tmp_path, "mixtures", (("Primary Name", "Source", "N components"), raw))
    before = source.read_bytes()
    record = database.mixture(["example mix"])
    assert record.raw_fields == raw
    mass = database.mixture_spec(
        "example", basis="mass", phase="liquid", temperature_K=298.15, pressure_Pa=101325
    )
    mole = database.mixture_spec(
        "example", basis="mole", phase="liquid", temperature_K=298.15, pressure_Pa=101325
    )
    assert mass.mass_fractions == pytest.approx((0.4, 0.6))
    assert mole.mole_fractions == pytest.approx((0.9, 0.1))
    assert mass.mole_fractions != mole.mole_fractions
    assert MixtureSpec.from_dict(json.loads(json.dumps(mass.to_dict()))) == mass
    assert source.read_bytes() == before


def test_no_reference_import_or_file_access_on_construction(tmp_path):
    program = """
import sys
from chemworld.data.identifiers import IdentifierDatabase
database = IdentifierDatabase(sys.argv[1])
assert database.loaded_sources == ()
assert 'chemicals' not in sys.modules and 'periodictable' not in sys.modules
assert len(database.catalog) == 13
print('native identifier reader OK')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(tmp_path / "missing")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "native identifier reader OK" in result.stdout


def test_query_does_not_modify_source_files(database):
    before = {p.name: p.read_bytes() for p in Path(database.source_directory).iterdir()}
    database.component("ethanol")
    assert {p.name: p.read_bytes() for p in Path(database.source_directory).iterdir()} == before


def test_ionic_source_formula_drives_native_charge(database, tmp_path):
    ion = (
        "223",
        "14798-03-9",
        "H4N+",
        "18.03846",
        "[NH4+]",
        "H3N/h1H3/p+1",
        "QGZKDVFQNNGYKY-UHFFFAOYSA-O",
        "azanium",
        "ammonium",
    )
    write_source(tmp_path, "cation", (ion,))
    actual = database.component("ammonium", source_ids=("cation",))
    assert actual.charge == 1
    assert actual.composition == {"H": 4, "N": 1}
    assert actual.cas_number == "14798-03-9"


def test_malformed_preferences_and_conflicting_preferences_are_explicit(database, tmp_path):
    write_source(tmp_path, "organic-preferences", {"preferred_cas": "bad", "unpreferred_cas": []})
    with pytest.raises(IdentifierSourceFormatError):
        database.preferences("organic-preferences")
    write_source(
        tmp_path,
        "organic-preferences",
        {"preferred_cas": ["64-17-5"], "unpreferred_cas": ["64-17-5"]},
    )
    with pytest.raises(ValueError, match="contradict"):
        database.resolve("ethanol", preference_sources=("organic-preferences",))


def test_explicit_nonstandard_inchi_prefix_is_preserved(database, tmp_path):
    row = list(WATER)
    row[5] = "InChI=1/H2O/h1H2"
    write_source(tmp_path, "example-user", (row,))
    assert database.component("water", source_ids=("example-user",)).inchi == row[5]
