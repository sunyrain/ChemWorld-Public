"""Actual TDB input behavior, declarative safety and complete associations."""

import json
import sys

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.tdb import TdbDatabase, TdbParseError
from chemworld.physchem.caloric_math import FirstDerivative
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.tdb_expressions import TdbExpression


def source():
    return InputSource(DataSourceProvenance(
        "synthetic-tdb", 0, "Explicit test equations", "1", "CC0-1.0",
    ), "synthetic")


BASE = """
$ Deliberately declared out of order, and case insensitive.
par G(ALPHA,FE:VA;0) 100 REF#; 500 Y REF#+10; 1200 N TEST !
constit ALPHA : FE, NI : VA : !
phase ALPHA % 2 1 3 !
fun REF 100 20+2*T-3*T*LN(T); 1200 N !
element FE BCC_A2 55.845 4489 27.28 !
element NI FCC_A1 58.6934 4787 29.8 !
element VA VACUUM 0 0 0 !
type_def % SEQ * !
"""


def database(text=BASE):
    return TdbDatabase.from_string(text, source=source())


def test_declarations_roundtrip_provenance_and_native_analytic_derivative(tmp_path):
    before = set(sys.modules)
    db = database()
    assert db.summary() == {
        "elements": 3, "species": 3, "phases": 1, "functions": 1, "parameters": 1,
        "parameter_types": {"G": 1}, "phase_options": [], "source_id": "synthetic-tdb",
    }
    assert db.elements["FE"].molar_mass_g_mol == 55.845
    assert db.phases["ALPHA"].site_ratios == (1, 3)
    assert db.phases["ALPHA"].constituents == (("FE", "NI"), ("VA",))
    t = FirstDerivative(300, 1)
    result = db.evaluate_function("ref", t)
    from math import log

    assert result.value == pytest.approx(20 + 2 * 300 - 900 * log(300))
    assert result.slope == pytest.approx(2 - 3 * (log(300) + 1))
    payload = json.loads(json.dumps(db.to_dict(), allow_nan=False))
    loaded = TdbDatabase.from_dict(payload)
    assert loaded.to_dict() == payload
    assert loaded.evaluate_function("REF", t) == result
    path = tmp_path / "synthetic.tdb"
    path.write_text(BASE, encoding="utf-8-sig")
    assert TdbDatabase.from_file(path, source=source()).to_dict() == db.to_dict()
    assert not {"pycalphad", "symengine", "sympy"}.intersection(set(sys.modules) - before)
    with pytest.raises(TypeError):
        db.phases["INJECTED"] = db.phases["ALPHA"]


def test_piecewise_right_side_exact_cut_and_explicit_extrapolation():
    db = database()
    parameter = db.parameters[0]
    assert db.evaluate_parameter(parameter, 499) == db.evaluate_function("REF", 499)
    assert db.evaluate_parameter(parameter, 500) == db.evaluate_function("REF", 500) + 10
    for temperature in (99, 1200, 1500):
        with pytest.raises(ValueError, match="outside declared"):
            db.evaluate_parameter(parameter, temperature)
        assert db.evaluate_parameter(parameter, temperature, extrapolation="extend") == (
            db.evaluate_function("REF", temperature, extrapolation="extend")
            + (10 if temperature >= 500 else 0)
        )
    with pytest.raises(ValueError, match="extrapolation"):
        db.evaluate_function("REF", 300, extrapolation="guess")
    assert db.evaluate_parameter(parameter, 300, overrides={"REF": 42}) == 42
    assert db.evaluate_function("REF", 300, overrides={"REF": 42}) == 42
    for overrides in ({"T": 1}, {"REF": float("inf")}, {"REF": True}):
        with pytest.raises(ValueError):
            db.evaluate_function("REF", 300, overrides=overrides)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('x')", "T.real", "T[0]", "[T]", "T if T else 1",
    "(lambda: 1)()", "LN(T, 1)", "LN(x=T)", "SUM(T)", "True", "1E999",
    "T > 1", "{'T':1}", "(T := 2)", "T//2", "T%2",
])
def test_expression_rejects_code_and_nonmathematical_syntax(expression):
    with pytest.raises(ValueError):
        TdbExpression(expression)


def test_arithmetic_pressure_constants_real_domain_and_parameter_references():
    from math import exp, log

    curve = database("FUNCTION Q 1 EXP(-2)+LOG(T)+R*T*LN(P)+T^2; N !")
    assert curve.evaluate_function("Q", 10, 2) == pytest.approx(
        exp(-2) + log(10) + 8.3145 * 10 * log(2) + 100,
    )
    for pressure in (-1, 0, float("nan")):
        with pytest.raises(ValueError, match="pressure"):
            curve.evaluate_function("Q", 10, pressure)
    with pytest.raises(ValueError):
        TdbExpression("(-1)**.5").evaluate({})
    with pytest.raises(ValueError):
        TdbExpression("1/0").evaluate({})
    with pytest.raises(ValueError):
        TdbExpression("EXP(1000)").evaluate({})


@pytest.mark.parametrize(("suffix", "message"), [
    ("FUNCTION BAD 1 MISSING; 1000 N !", "unresolved"),
    ("FUNCTION BAD 1 BAD; 1000 N !", "cyclic"),
    ("FUNCTION XX 1 YY; 1000 N !\nFUNCTION YY 1 XX; 1000 N !", "cyclic"),
    ("FUNCTION REF 1 0; 1000 N !", "duplicate"),
    ("FUNCTION BAD 1 0; 1000 Y !", "another expression"),
    ("FUNCTION BAD 1000 0; 500 N !", "increasing"),
    ("FUNCTION BAD 1 0; 500 N; 1000 N !", "terminal N"),
    ("FUNCTION BAD 1 0; 500 N", "terminator"),
    ("FUNCTION T 1 0; 1000 N !", "shadow"),
    ("PARAMETER G(ALPHA,XX:VA;0) 1 0; 1000 N !", "invalid constituents"),
    ("PARAMETER G(UNKNOWN,FE;0) 1 0; 1000 N !", "undeclared phase"),
    ("SPECIES FAKE XX1 !", "undeclared elements"),
    ("PARAMETER G(ALPHA,FE:VA;0) 1 0; 1000 N !", "duplicate TDB parameter"),
    ("PHASE BAD % 2 1 !", "site ratios"),
    ("PHASE BAD:Z % 1 1 !", "unknown TDB phase options"),
    ("COMMAND_UNKNOWN 1 !", "unknown or ambiguous"),
])
def test_whole_database_rejects_incomplete_invalid_or_ambiguous_input(suffix, message):
    with pytest.raises(TdbParseError, match=message):
        database(BASE + "\n" + suffix)


def test_all_branches_are_validated_and_command_has_original_line_number():
    with pytest.raises(TdbParseError, match="MISSING"):
        database(BASE + "\nFUNCTION BAD 1 0; 500 Y MISSING; 1000 N !")
    with pytest.raises(TdbParseError, match="line 3"):
        database("$ comment\n\nFUNCTION F 1 T.real; 1000 N !")
    db = database("FUNCTION F 1 3; 1000 N ! FUNCTION MALICIOUS 1 2; 1000 N !\n")
    assert tuple(db.functions) == ("F",)


def test_ionic_species_types_parameter_catalog_and_explicit_consumer_diagnostics():
    text = """
    ELEMENT /- ELECTRON_GAS 0 0 0 !
    ELEMENT VA VACUUM 0 0 0 !
    ELEMENT CU FCC_A1 63.546 5004 33.15 !
    ELEMENT O GAS 15.999 4341 102.52 !
    SPECIES CU+2 CU1/+2 !
    SPECIES O-2 O1/-2 !
    SPECIES O2 O2 !
    PHASE IONIC:Y % 2 1 1 !
    CONSTITUENT IONIC :CU+2:O-2,VA,O2: !
    TYPE_DEF % SEQ * !
    PARAMETER G(IONIC,CU+2:O-2;0) 1 -1000; 6000 N !
    PARAMETER G(IONIC,O2;0) 1 -500; 6000 N !
    PARAMETER VISC(IONIC,CU+2:VA;0) 1 2; 6000 N !
    PARAMETER MQ(IONIC&CU,CU+2:VA;0) 1 1; 6000 N !
    """
    db = database(text)
    assert db.species["CU+2"].charge == 2
    assert db.species["O-2"].charge == -2
    assert db.species["VA"].number_of_atoms == db.species["/-"].number_of_atoms == 0
    assert db.species["O2"].number_of_atoms == 2
    assert db.parameters[-1].diffusing_species == "CU"
    assert db.consumer_diagnostics(parameter_types={"G"}, phase_options=set(),
                                   type_features={"sequence"}) == (
        "unconsumed parameter type MQ", "unconsumed parameter type VISC",
        "unconsumed phase option Y",
    )


def test_late_magnetic_ordering_and_unsupported_definitions_remain_visible():
    text = BASE + """
    PHASE ORDERED:I %&@ 2 1 3 !
    CONSTITUENT ORDERED :FE,NI:VA: !
    TYPE_DEF & GES A_P_D ORDERED MAGNETIC -1 .4 !
    TYPE_DEF @ GES A_P_D ORDERED DIS_PART ALPHA ,,, !
    TYPE_DEF ? IF SOMETHING THEN UNKNOWN !
    """
    db = database(text)
    hints = db.phases["ORDERED"].model_hints
    assert hints["magnetic_structure_factor"] == .4
    assert hints["magnetic_afm_factor"] == -1
    assert hints["ordered_phase"] == "ORDERED"
    assert hints["disordered_phase"] == "ALPHA"
    assert db.phases["ALPHA"].model_hints["ordered_phase"] == "ORDERED"
    assert db.consumer_diagnostics(parameter_types={"G"}, phase_options={"I"},
                                   type_features={"sequence", "MAGNETIC", "DISORDERED_PART"}) == (
        "unconsumed type definition ?: IF SOMETHING THEN UNKNOWN",
    )


def test_missing_source_unknown_fields_and_no_mutable_parsed_state():
    with pytest.raises(ValueError, match="InputSource"):
        TdbDatabase.from_string(BASE, source=None)
    payload = database().to_dict()
    payload["extra"] = "not a second state"
    with pytest.raises(ValueError, match="unknown"):
        TdbDatabase.from_dict(payload)
    assert TdbExpression("T+P").symbols == ("P", "T")
    with pytest.raises(ValueError, match="nesting"):
        TdbExpression("+" * 100 + "T")


def test_all_public_parameter_kinds_are_preserved_and_commands_never_executed():
    from chemworld.data import TdbDatabase as PublicDatabase

    kinds = ("G", "L", "TC", "NT", "BMAGN", "GD", "THETA", "V0", "VA", "VC", "VK",
             "VISC", "ELRS", "THCD", "SIGM", "XI", "MQ", "MF", "DQ", "DF", "VS")
    text = BASE + "\n".join(
        f"PARAMETER {kind}(ALPHA,NI:VA;0) 1 42; 1000 N !" for kind in kinds
    ) + "\nDEFAULT_COMMAND REJECT_PHASE ALPHA !"
    db = PublicDatabase.from_string(text, source=source())
    assert tuple(p.kind for p in db.parameters[1:]) == kinds
    assert all(db.evaluate_parameter(p, 300) == 42 for p in db.parameters[1:])
    assert "ALPHA" in db.phases
    diagnostics = db.consumer_diagnostics(parameter_types=kinds, phase_options=(),
                                          type_features={"sequence"})
    assert diagnostics == ("nonexecuted database command DEFAULT_COMMAND: REJECT_PHASE ALPHA",)


def test_prefix_ambiguity_and_boolean_temperatures_do_not_become_physical_numbers():
    with pytest.raises(TdbParseError, match="ambiguous"):
        database(BASE + "\nPARAMETER V(ALPHA,NI:VA;0) 1 1; 1000 N !")
    with pytest.raises(ValueError, match="boolean"):
        database().evaluate_function("REF", True)
