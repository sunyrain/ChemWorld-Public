"""Source-preserving constant lookup and native scientific input consumption."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from chemworld.data.constant_catalog import CONSTANT_SOURCES
from chemworld.data.constants import (
    AmbiguousConstantIdentityError,
    ConflictingConstantDataError,
    ConstantDatabase,
    ConstantFieldNotFoundError,
    ConstantIdentityNotFoundError,
    ConstantSourceFormatError,
    ConstantSourceUnavailableError,
    ConstantUnitUnavailableError,
    ConstantValueUnavailableError,
    NonScalarConstantError,
    SourceCorrelation,
    UnknownConstantMethodError,
)
from chemworld.data.scientific_inputs import QuantityColumn, ScientificTable
from chemworld.foundation.dimensions import (
    DimensionVector,
    dimension_definition,
    unmapped_supported_unit_dimensions,
)


def put(root, method, content):
    source = next(s for s in CONSTANT_SOURCES if s.source_id == method)
    path = root / source.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def db(tmp_path):
    put(tmp_path, "HEOS", "CAS\tname\tTc\tPc\tPt\tomega\tunknown\n"
        "7732185\tWater\t647.096\t22064000\t\t0.3443\t12\n"
        "64-17-5\tEthanol\tNaN\tbad\t0\t0.646\t13\n")
    put(tmp_path, "IUPAC", "CAS\tTc\tTc_error\n7732-18-5\t647.14\t0.1\n")
    put(tmp_path, "VDI", json.dumps({"7732-18-5": {
        "Name": "water", "MW": 18.015, "Tc": 647.096,
        "T": [300, 647.096, 400], "P": [1000, 22064000, 1e5],
        "sigma": [.07, 0, None], "Beta": [.001, None, .002],
    }}))
    return ConstantDatabase(tmp_path)


def test_fractional_dimensions_roundtrip_and_physical_relations():
    def vector(name):
        return dimension_definition(name).vector
    assert vector("solubility_parameter")**2 == vector("pressure")
    assert vector("parachor") == vector("molar_volume") * vector("surface_tension")**.25
    assert vector("electric_dipole_moment") == vector("electric_charge") * vector("length")
    assert vector("thermal_expansivity") * vector("temperature") == DimensionVector()
    for name in ("parachor", "solubility_parameter", "pressure"):
        assert DimensionVector.from_dict(json.loads(json.dumps(vector(name).to_dict()))) == (
            vector(name)
        )
    assert all(type(v) is int for v in vector("pressure").to_dict().values())
    assert DimensionVector.from_dict({"mass": "0.25"}).mass == .25
    assert unmapped_supported_unit_dimensions() == ()


@pytest.mark.parametrize("bad", [True, None, "1", float("nan"), float("inf"), -float("inf")])
def test_dimensions_reject_invalid_exponents(bad):
    with pytest.raises((TypeError, ValueError)):
        DimensionVector(mass=bad)
    with pytest.raises((TypeError, ValueError)):
        DimensionVector(mass=1)**bad


def test_lazy_source_reads_keep_identity_raw_values_and_files(db, monkeypatch):
    before = {p: p.read_bytes() for p in db.source_directory.rglob("*") if p.is_file()}
    assert db.loaded_sources == ()
    assert len(db.catalog) == 43
    record = db.record("7732-18-5", method="HEOS")
    assert record.source_key == "7732185"
    assert record.cas_number == "7732-18-5"
    assert record.raw("Pc") == "22064000"
    assert db.record("Water", method="HEOS") is record
    assert db.record(7732185, method="HEOS") is record
    assert db.record(record.record_id, method="HEOS") is record
    assert db.loaded_sources == ("HEOS",)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("unexpected cached read"))
    assert record.value("Pc").quantity("bar").value == 220.64
    assert db.record("Water", method="HEOS") is record
    monkeypatch.undo()
    assert all(p.read_bytes() == value for p, value in before.items())


def test_distinct_failures_and_source_error_not_promoted_to_sigma(db):
    for error, action in (
        (UnknownConstantMethodError, lambda: db.records("absent")),
        (ConstantSourceUnavailableError, lambda: db.records("PSRK")),
        (ConstantIdentityNotFoundError, lambda: db.record("absent", method="HEOS")),
        (ConstantFieldNotFoundError, lambda: db.value("Water", "absent", method="HEOS")),
        (ConstantValueUnavailableError, lambda: db.value("Water", "Pt", method="HEOS").quantity()),
        (ConstantUnitUnavailableError,
         lambda: db.value("Water", "unknown", method="HEOS").quantity()),
    ):
        with pytest.raises(error):
            action()
    assert db.value("Ethanol", "Tc", method="HEOS").quality == "invalid"
    assert db.value("Ethanol", "Pc", method="HEOS").quality == "invalid"
    assert db.value("Ethanol", "Pt", method="HEOS").quantity().value == 0
    value = db.value("7732-18-5", "Tc_error", method="IUPAC").input_value()
    assert value.value == .1 and value.uncertainty is None
    assert "coverage unspecified" in value.uncertainty_note


def test_conflicts_require_explicit_selection_and_duplicates_survive(db):
    assert db.methods("7732-18-5", "Tc", sources=("HEOS", "IUPAC")) == ("HEOS", "IUPAC")
    with pytest.raises(ConflictingConstantDataError) as error:
        db.resolve("7732-18-5", "Tc", methods=("HEOS", "IUPAC"))
    assert len(error.value.candidates) == 2
    assert db.resolve("7732-18-5", "Tc", methods=("HEOS", "IUPAC"),
                      source_priority=("IUPAC",)).value == 647.14
    put(db.source_directory, "PSRK", "CAS\tTc\tTc\n7732-18-5\t640\t641\n"
        "7732-18-5\t650\t651\n")
    assert len(db.records_for("7732-18-5", method="PSRK")) == 2
    with pytest.raises(AmbiguousConstantIdentityError):
        db.record("7732-18-5", method="PSRK")
    assert db.record("PSRK:1", method="PSRK").raw("Tc__2") == "641"


def test_inspection_separates_source_identity_field_value_unit_and_conflicts(db):
    report = db.inspect_value("7732-18-5", "Tc", methods=("HEOS", "IUPAC", "PSRK", "absent"))
    assert [r["status"] for r in report["methods"]] == [
        "matched", "matched", "source_unavailable", "unknown_method"]
    assert report["value_conflict"] and not report["identity_conflict"]
    assert report["selected_record_id"] is None
    assert json.loads(json.dumps(report)) == report
    cases = (("Water", "Pt", "missing_value"), ("Water", "unknown", "missing_unit"),
             ("Ethanol", "Tc", "invalid_value"), ("Water", "absent", "missing_field"))
    for identity, field, diagnostic in cases:
        item = db.inspect_value(identity, field, methods=("HEOS",))["methods"][0]
        assert diagnostic in item["records"][0]["diagnostics"]
    assert db.inspect_value("absent", "Tc", methods=("HEOS",))["methods"][0][
        "status"] == "missing_identity"
    series = db.inspect_value("water", "P", methods=("VDI",))
    assert series["methods"][0]["records"][0]["raw_value"] == (1000, 22064000, 1e5)
    point = db.inspect_value("water", "P", methods=("VDI",), point=2)
    assert point["methods"][0]["records"][0]["value"]["value"] == 1e5
    put(db.source_directory, "PSRK", "CAS\tTc\n7732-18-5\t640\n7732-18-5\t650\n")
    report = db.inspect_value("7732-18-5", "Tc", methods=("PSRK",))
    assert report["methods"][0]["status"] == "ambiguous_records"
    assert report["value_conflict"]
    with pytest.raises(TypeError):
        db.inspect_value("Water", "Tc", methods="HEOS")


def test_xml_original_units_and_correlations_are_preserved(tmp_path):
    put(tmp_path, "CHEMSEP_XML", '<compounds><compound><CAS value="7732-18-5"/>'
        '<CriticalVolume value="0.056" units="m3/kmol"/>'
        '<DipoleMoment value="3.33564095198152e-30" units="Coulomb.m"/>'
        '<Parachor value="0.04" units="kg0.25.m3/s0.5/kmol"/>'
        '<Odd value="12" units="unknown"/>'
        '<VaporPressure units="Pa"><eqno value="101"/><A value="1"/></VaporPressure>'
        '</compound></compounds>')
    db = ConstantDatabase(tmp_path)
    record = db.record("7732-18-5", method="CHEMSEP_XML")
    assert record.value("Vc").quantity().value == pytest.approx(5.6e-5)
    assert record.value("dipole_moment").quantity("Debye").value == pytest.approx(1)
    assert record.value("Parachor").quantity().value == pytest.approx(4e-5)
    assert isinstance(record.raw("VaporPressure"), SourceCorrelation)
    assert 'value="101"' in record.raw("VaporPressure").xml
    assert record.descriptor("VaporPressure").continuation == "22C-G"
    with pytest.raises(NonScalarConstantError):
        record.value("VaporPressure")
    with pytest.raises(ConstantUnitUnavailableError):
        record.value("Odd").quantity()
    assert db.candidates("7732-18-5", "VaporPressure", methods=("CHEMSEP_XML",)) == ()


def test_vdi_order_null_and_critical_zero_survive_json_csv_input(db):
    series = db.saturation_series("water", "sigma")
    assert series.temperatures_K == (300, 647.096, 400)
    assert series.valid_points() == ((300, 647.096), (.07, 0))
    assert series.values[2].quality == "missing"
    assert db.candidates("water", "P", methods=("VDI",)) == ()
    with pytest.raises(NonScalarConstantError):
        db.value("water", "P", method="VDI")
    columns = (QuantityColumn("T", "K", "condition", (1, 1000), "absolute"),
               QuantityColumn("sigma", "N/m", "observation", (0, 1), "saturation"))
    table = db.saturation_table("water", columns=columns,
        field_bindings={"T": "T", "sigma": "sigma"}, dataset_id="water",
        group_id="compound:water", split="validation")
    assert table.rows[-1].values["sigma"].quality == "missing"
    assert table.rows[1].values["sigma"].value == 0
    payload = table.to_dict()
    assert ScientificTable.from_json(json.dumps(payload)).to_dict() == payload
    metadata = {k: v for k, v in payload.items() if k != "rows"}
    assert ScientificTable.from_csv(table.to_csv(), metadata).to_dict() == payload


@pytest.mark.parametrize("content", ["[]", '{"x": {"T": 1}}', '{"x": {"T": []}}',
    '{"x": {"T": [300, 400], "P": [10]}}', "{invalid"])
def test_invalid_vdi_layout_reports_source_format(tmp_path, content):
    put(tmp_path, "VDI", content)
    with pytest.raises(ConstantSourceFormatError):
        ConstantDatabase(tmp_path).records("VDI")


@pytest.mark.parametrize("content", ["", "CAS\tTc\n7732185\n"])
def test_bad_text_source_is_distinguished_from_missing_method(tmp_path, content):
    put(tmp_path, "HEOS", content)
    with pytest.raises(ConstantSourceFormatError):
        ConstantDatabase(tmp_path).records("HEOS")


def test_parameter_table_requires_valid_values_and_explicit_bindings(db):
    args = {"method": "HEOS", "columns": (
        QuantityColumn("pressure", "bar", "parameter", (0, 1000), "critical"),
    ), "field_bindings": {"pressure": "Pc"}, "dataset_id": "constants", "group_id": "water"}
    table = db.parameter_table(("Water",), **args)
    assert table.rows[0].values["pressure"].value == 220.64
    assert table.sources[0].provenance.version == "chemicals-constant-data-1.5.2"
    with pytest.raises(ValueError, match="valid numeric"):
        db.parameter_table(("Ethanol",), **args)
    with pytest.raises(ValueError, match="explicit source field"):
        db.parameter_table(("Water",), **{**args, "field_bindings": {}})


def test_historical_defaults_do_not_become_observations_or_replace_weights(tmp_path):
    from chemworld.physchem.elements import molecular_weight

    before = molecular_weight({"H": 2, "O": 1})
    put(tmp_path, "OPENBABEL_ELEMENTS", "# Historical header\n"
        "118\tOg\t0.0\t1.6\t0.0\t2.0\t6\t294\t0\t0\t0\t1\t1\t1\tOganesson\n")
    record = ConstantDatabase(tmp_path).record("Og", method="OPENBABEL_ELEMENTS")
    assert record.value("covalent_radius").quality == "source_default_or_ambiguous"
    assert record.value("ionization_energy").input_value().quality == "missing"
    with pytest.raises(ConstantValueUnavailableError):
        record.value("covalent_radius").quantity()
    with pytest.raises(ConstantUnitUnavailableError):
        record.value("historical_mass").quantity()
    assert molecular_weight({"H": 2, "O": 1}) == before


def test_import_does_not_import_reference_library_or_load_data():
    script = """
import builtins, sys
original = builtins.__import__
def checked(name, *a, **kw):
    assert name.split('.')[0] not in {'chemicals', 'thermo', 'pandas'}, name
    return original(name, *a, **kw)
builtins.__import__ = checked
from chemworld.data.constants import ConstantDatabase
assert ConstantDatabase('nonexistent-explicit-input').loaded_sources == ()
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_density_input_binds_real_model_field_and_changes_particle_geometry(tmp_path):
    # Only the density is literature-like input here; kinetics remain synthetic.
    # Exercise the real card-16 binding and real crystal constitutive consumer.
    put(tmp_path, "CRC_INORGANIC", "CAS\tChemical\trho\n7647-14-5\tSodium chloride\t2170\n")
    path = Path(__file__).parent / "reference/run_constant_foundation_block.py"
    runner = runpy.run_path(str(path))
    result = runner["crystal_consumer"](ConstantDatabase(tmp_path), 2170)
    assert result["particle_moles_error"] < 1e-20
    assert result["receipt"]["parameters"][0]["value"] == 2170
