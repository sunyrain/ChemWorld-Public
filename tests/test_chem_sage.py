"""ChemSage declarations and native Gibbs consumers, with hostile-input guards."""

import json
from dataclasses import FrozenInstanceError, replace
from math import exp, expm1, log

import pytest

from chemworld.data.chem_sage import ChemSageDatabase
from chemworld.data.chem_sage_functions import basis_expression, endmember_function
from chemworld.data.chem_sage_parser import ChemSageParseError, parse_chem_sage
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.caloric_math import derivative_values, exponential_minus_one
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_material import CefMaterialSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.tdb_expressions import TdbExpression

QKTO = """Synthetic binary QKTO
2 2 0 2 0
FE NI
55.845 58.69
6 1 2 3 4 5 6
1 1
LIQUID
QKTO
FE
1 1 1 0
2000 -1000 10 -20 0 0 0
1 1
NI
1 1 0 1
2000 -2000 5 -25 0 0 0
1 2
2 1 2 1 1 1000
0
"""
CP = """Synthetic phase Cp with a transition
1 1 0 1
FE
55.845
6 1 2 3 4 5 6
1 1
ALPHA
7 2 1
0 30
800 30 0 0 0
5000
1500 35 0 0 0
"""
MIXED_VALENCE = """Synthetic mixed-valence Fe-O declarations
2 2 0 3 0
FE O
55.845 15.999
6 1 2 3 4 5 6
1 1
SLAG
SUBQ
2 3
FEO
1 1 1 1
2000 0 0 0 0 0 0
1 1 0 0 0
2.4
FE2O3
1 1 2 3
2000 0 0 0 0 0 0
2 3 0 0 0
2.4
2 1
FE FE
O
2 3
1 2
2
1
1 2
1 1
1 1 3 3 4 4 4 4
1 2 3 3 4 6 4 4
2 2 3 3 6 6 4 4
0
"""


def test_qkto_original_indices_groups_coefficients_and_immutable_records():
    result = parse_chem_sage(QKTO)
    assert result.header.solution_sizes == (0, 2)
    assert result.header.components == ("FE", "NI")
    assert len(result.phases) == 1
    phase = result.phases[0]
    assert phase.phase_type == "QKTO" and not phase.gas_slot
    assert [m.chemical_group for m in phase.endmembers] == [1, 2]
    assert phase.endmembers[0].intervals[0].coefficients == (-1000, 10, -20, 0, 0, 0)
    assert phase.interactions[0].indices == (1, 2)
    assert phase.interactions[0].exponents == (1, 1)
    assert phase.interactions[0].coefficients == (1000,)
    with pytest.raises(FrozenInstanceError):
        phase.name = "changed"


def test_cp_reference_and_transition_belong_to_the_right_segment():
    result = parse_chem_sage(CP)
    phase = result.phases[0]
    assert phase.phase_type == "PURE"
    member = phase.endmembers[0]
    assert member.enthalpy_298_J_mol == 0 and member.entropy_298_J_mol_K == 30
    assert member.intervals[0].transition_enthalpy_J_mol == 0
    assert member.intervals[1].transition_enthalpy_J_mol == 5000
    assert member.intervals[1].coefficients == (35, 0, 0, 0)


@pytest.mark.parametrize("text", ["", QKTO[:-2], QKTO.replace("55.845", "NaN"),
                                  QKTO.replace("2 2 0 2 0", "2 -2 0 2 0"),
                                  QKTO.replace("QKTO", "__import__('os')", 1).replace(
                                      "\nQKTO\n", "\nexec\n"),
                                  QKTO.replace("2 2 0 2 0", "999999999999999999 2 0 2 0")])
def test_truncated_nonfinite_executable_and_unbounded_inputs_are_rejected(text):
    with pytest.raises(ChemSageParseError):
        parse_chem_sage(text)


def test_status_and_footer_are_retained_and_not_executed(tmp_path):
    marker = tmp_path/"never-created"
    text = QKTO.replace("\nLIQUID\n", "\n"+"LIQUID".ljust(25)+"!\n")
    text += f"Comment __import__('pathlib').Path({str(marker)!r}).touch()\n"
    result = parse_chem_sage(text)
    assert result.phases[0].status == "dormant"
    assert "__import__" in result.trailer
    assert not marker.exists()


def test_same_element_with_distinct_valences_is_not_a_duplicate_species():
    data = parse_chem_sage(MIXED_VALENCE).phases[0].mqmqa
    assert data.cations == ("FE", "FE")
    assert data.cation_charges == (2, 3)
    assert data.pair_indices == ((1, 1), (2, 1))
    with pytest.raises(ChemSageParseError, match="unique charged"):
        parse_chem_sage(MIXED_VALENCE.replace("\n2 3\n1 2\n", "\n2 2\n1 2\n"))


def test_valence_in_mqmqa_label_is_not_an_elemental_atom_count():
    db = database(MIXED_VALENCE.replace("\nFE FE\n", "\nFE2+ FE3+\n"))
    for name, charge in (("FE2+[2+]", 2), ("FE3+[3+]", 3)):
        assert db.species[name].constituents == (("FE", 1.),)
        assert db.species[name].charge == charge


def test_sublattice_atom_multiplier_and_index_rows_keep_original_identity():
    text = """Synthetic indexed two-sublattice input
2 2 0 2 0
FE NI
55.845 58.69
6 1 2 3 4 5 6
1 1
ALPHA:2
SUBL
FE:VA
1 1 1 0
2000 0 0 0 0 0 0
NI:VA
1 1 0 1
2000 0 0 0 0 0 0
2
.5 .5
2 1
FE NI
VA
1 2
1 1
0
"""
    phase = parse_chem_sage(text).phases[0]
    assert phase.name == "ALPHA:2"
    assert phase.site_ratios == (1, 1)
    assert phase.constituents == (("FE", "NI"), ("VA",))
    assert phase.endmember_indices == ((1, 1), (2, 1))


def function_values(curve, temperature):
    return derivative_values(lambda t: curve.segment(t).expression.evaluate({"T": t}),
                             temperature, order=3)


def test_compiled_cp_reference_latent_heat_and_right_segment_are_physical():
    parsed = parse_chem_sage(CP)
    curve = endmember_function(parsed.phases[0].endmembers[0], parsed.header, "CP_TEST")
    for t in (298.15, 600., 900.):
        g, d1, d2, d3 = function_values(curve, t)
        if t < 800:
            h, s, cp = 30*(t-298.15), 30+30*log(t/298.15), 30
        else:
            h = 30*(800-298.15)+5000+35*(t-800)
            s, cp = 30+30*log(800/298.15)+5000/800+35*log(t/800), 35
        assert g == pytest.approx(h-t*s, abs=1e-9)
        assert g-t*d1 == pytest.approx(h, abs=1e-9)
        assert -d1 == pytest.approx(s, abs=1e-11)
        assert -t*d2 == pytest.approx(cp, abs=1e-11)
        assert -d2-t*d3 == pytest.approx(0, abs=1e-11)
    assert curve.segment(800).lower_K == 800
    left_h, left_s = 30*(800-298.15), 30+30*log(800/298.15)
    g, d1, *_ = function_values(curve, 800.)
    assert g == pytest.approx(left_h-800*left_s, abs=1e-8)
    assert g-800*d1 == pytest.approx(left_h+5000, abs=1e-8)


@pytest.mark.parametrize("power", [0., -1., 1., 2., -2., .5, 1e-14, -1+1e-12, 99.])
def test_general_cp_powers_and_logarithm_use_the_same_analytic_gibbs_kernel(power):
    parsed = parse_chem_sage(CP)
    member = parsed.phases[0].endmembers[0]
    interval = replace(member.intervals[0], upper_K=1500, coefficients=(0.,)*4,
                       extra_powers=((.3, power),))
    member = replace(member, option=10, intervals=(interval,))
    curve = endmember_function(member, parsed.header, "POWER_TEST")
    for t in (298.15, 400., 900.):
        g, d1, d2, d3 = function_values(curve, t)
        cp = .3*log(t) if power == 99 else .3*t**power
        slope = .3/t if power == 99 else .3*power*t**(power-1)
        assert -t*d2 == pytest.approx(cp, rel=1e-11, abs=1e-10)
        assert -d2-t*d3 == pytest.approx(slope, rel=1e-10, abs=1e-10)
        if t == 298.15:
            assert g-t*d1 == pytest.approx(0, abs=1e-6)
            assert -d1 == pytest.approx(30, abs=1e-8)


def test_stable_exponential_primitive_preserves_small_values_and_all_three_derivatives():
    for x in (0., 1e-18, -1e-18, -100., 5.):
        values = derivative_values(exponential_minus_one, x, order=3)
        assert values[0] == expm1(x)
        assert values[1:] == pytest.approx((exp(x),)*3, rel=1e-14, abs=0)
    with pytest.raises(ValueError, match="unsupported"):
        TdbExpression("EXPM1(T)")


def test_pressure_coefficients_need_source_units_and_volume_options_are_not_ignored():
    with pytest.raises(ValueError, match="explicit source unit"):
        basis_expression((1, 7, 8), (1, 2, 3))
    expression = TdbExpression(basis_expression((1, 7, 8), (1, 2, 3),
                                               excess_pressure_unit="bar"))
    assert expression.evaluate({"P": 200000}) == 1+2*2+3*4
    parsed = parse_chem_sage(CP)
    with pytest.raises(ValueError, match="mechanical"):
        endmember_function(replace(parsed.phases[0].endmembers[0], option=9), parsed.header,
                           "UNSUPPORTED_VOLUME")


def database(text=QKTO):
    source = InputSource(DataSourceProvenance("synthetic-chemsage", 0, "Analytic input", "1",
                                            "CC0"), "synthetic")
    return ChemSageDatabase.from_string(text, source=source)


def test_qkto_reuses_actual_cef_gibbs_caloric_hessian_and_original_input_codec():
    db = database()
    assert db.text == QKTO and db.elements["FE"].enthalpy_298_J_mol is None
    assert db.phase_records["LIQUID"].phase_type == "QKTO"
    cloned_db = ChemSageDatabase.from_dict(json.loads(json.dumps(db.to_dict())))
    assert cloned_db.to_dict() == db.to_dict()
    model = CefPhaseModel(db, "LIQUID")
    sites, t = [{"FE": .3, "NI": .7}], 700.
    p = model.evaluate(sites, t)
    expected = .3*(-1000+10*t-20*t*log(t))+.7*(-2000+5*t-25*t*log(t))
    expected += 8.3145*t*(.3*log(.3)+.7*log(.7))+1000*.3*.7
    assert p.gibbs_J_mol == pytest.approx(expected, abs=1e-9)
    assert p.contributions_J_mol["excess"] == pytest.approx(210)
    assert model.temperature_response(sites, t).fixed_constitution_heat_capacity_J_mol_K == (
        pytest.approx(23.5))
    response = model.derivatives(sites, t)
    assert response.output_names[-2:] == ("site_sum:0", "charge")
    assert model.chemical_potentials(sites, t).status == "determined"
    clone = CefPhaseModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert clone.to_dict() == model.to_dict() and clone.evaluate(sites, t) == p
    material = CefMaterialSpec(model, ({"FE": .5, "NI": .5},), 500, 900, 600, .01,
                               phase_type="liquid")
    assert CefMaterialSpec.from_dict(json.loads(json.dumps(material.to_dict()))).to_dict() == (
        material.to_dict())


def test_cp_source_enters_the_same_cef_model_and_no_volume_or_mqmqa_fallback_is_possible():
    model = CefPhaseModel(database(CP), "ALPHA")
    p = model.evaluate([{"FE": 1}], 900.)
    assert p.enthalpy_J_mol == pytest.approx(30*(800-298.15)+5000+35*100, abs=1e-8)
    thermal = model.temperature_response([{"FE": 1}], 900)
    assert thermal.fixed_constitution_heat_capacity_J_mol_K == pytest.approx(35)
    with pytest.raises(ValueError, match="one-sided"):
        model.temperature_response([{"FE": 1}], 800)
    mq = database(MIXED_VALENCE)
    assert mq.phase_records["SLAG"].mqmqa.cation_charges == (2, 3)
    assert {s.charge for s in mq.species.values() if "FE[" in s.name} == {2, 3}
    with pytest.raises(ValueError, match="quadruplet"):
        CefPhaseModel(mq, "SLAG")


def test_source_suppression_requires_explicit_serialized_activation():
    text = QKTO.replace("\nLIQUID\n", "\n"+"LIQUID".ljust(25)+"!\n")
    db = database(text)
    with pytest.raises(ValueError, match="suppression"):
        CefPhaseModel(db, "LIQUID")
    model = CefPhaseModel(db, "LIQUID", activate_suppressed=True)
    clone = CefPhaseModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert clone.activate_suppressed
    with pytest.raises(ValueError, match="boolean"):
        replace(model, activate_suppressed="false")


def test_duplicate_source_labels_keep_each_occurrence_and_its_actual_coefficients():
    second = CP[CP.index("ALPHA\n"):].replace("800 30", "800 40")
    text = CP.replace("1 1 0 1", "1 1 0 2")+second
    db = database(text)
    assert tuple(db.phases) == ("ALPHA#OCCURRENCE1", "ALPHA#OCCURRENCE2")
    assert [p.name for p in db.phase_records.values()] == ["ALPHA", "ALPHA"]
    values = [CefPhaseModel(db, name).temperature_response(
        [{"FE": 1.}], 600).fixed_constitution_heat_capacity_J_mol_K for name in db.phases]
    assert values == pytest.approx((30, 40))
    assert ChemSageDatabase.from_dict(db.to_dict()).to_dict() == db.to_dict()


def test_magnetic_pure_phase_binds_parameters_to_real_elemental_lattice():
    text = """Synthetic magnetic PURE phase
1 1 0 1
FE
55.845
6 1 2 3 4 5 6
1 1
FE_BCC(S)
19 1 1
1234 32.1
1800 30 0 0 0
700 2.2
1 .4
"""
    model = CefPhaseModel(database(text), "FE_BCC(S)")
    magnetic = [p for p in model.database.parameters if p.kind in {"TC", "BMAGN"}]
    assert len(magnetic) == 2 and all(p.constituents == (("FE",),) for p in magnetic)
    assert model.evaluate([{"FE": 1.}], 600).contributions_J_mol["magnetic"] < -1


@pytest.mark.parametrize("change", [
    {"schema_version": "python:some.module"}, {"excess_pressure_unit": "inferred"},
    {"__class__": "os.system"},
])
def test_input_codec_rejects_unknown_schema_pressure_and_executable_metadata(change):
    payload = database().to_dict()
    payload.update(change)
    with pytest.raises(ValueError):
        ChemSageDatabase.from_dict(payload)
    model = CefPhaseModel(database(), "LIQUID").to_dict()
    model["database"] = payload
    with pytest.raises(ValueError):
        CefPhaseModel.from_dict(model)


def test_auxiliary_electron_balance_is_constrained_without_counting_it_as_an_atom():
    text = """Synthetic internal electron balance
3 2 0 2 0
FE O E(ALPHA)
55.845 15.999 .00054858
6 1 2 3 4 5 6
1 1
ALPHA
SUBL
FE:O
1 1 1 1 -1
2000 0 0 0 0 0 0
VA:O
1 1 0 1 2
2000 0 0 0 0 0 0
2
1 1
2 1
FE VA
O
1 2
1 1
0
"""
    model = CefPhaseModel(database(text), "ALPHA")
    good = ({"FE": 2/3, "VA": 1/3}, {"O": 1})
    assert model.evaluate(good, 600).element_fractions == pytest.approx({"FE": .4, "O": .6})
    assert model.constraint_labels[-1] == "auxiliary:E(ALPHA)"
    assert len(model.derivatives(good, 600).output_names) == 7
    with pytest.raises(ValueError, match="auxiliary"):
        model.evaluate(({"FE": .5, "VA": .5}, {"O": 1}), 600)
    result = model.stationary_constitution({"FE": .4, "O": .6},
                                           ({"FE": .5, "VA": .5}, {"O": 1}), 600)
    assert result.success, result.diagnostics
    assert result.chemical_potentials.status == "underdetermined"
    assert set(result.properties.element_fractions) == {"FE", "O"}


def test_qkto_real_native_thermal_consumer_uses_finite_energy_and_original_model():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.solid_solutions import solid_enthalpy
    from chemworld.world.solid_world import calphad_world_spec

    model = CefPhaseModel(database(), "LIQUID")
    material = CefMaterialSpec(model, ({"FE": .5, "NI": .5},), 500, 900, 600, .01,
                               phase_type="liquid")
    world = calphad_world_spec(material, element_amounts_mol={"FE": .0003, "NI": .0007},
                               temperature_K=600)
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=40)
    before = solid_enthalpy(runtime._state, material)
    step = runtime.execute({"operation": "heat", "target_temperature_K": 610, "duration_s": 1})
    assert step.info["transaction_status"] == "committed", step.info
    assert runtime._state.temperature_K == pytest.approx(610-10*exp(-.05/.2235), abs=2e-6)
    assert runtime._state.species_amounts == {"FE": .0003, "NI": .0007}
    assert solid_enthalpy(runtime._state, material)-before == pytest.approx(
        runtime._state.ledger.energy_jacket_J, abs=1e-8)
    assert ChemWorldRuntime.from_snapshot(runtime.snapshot())._state.to_dict() == (
        runtime._state.to_dict())


def ternary_model(groups, interaction="2 1 2 2 1 1000"):
    text = """Synthetic ternary QKTO groups
3 2 0 3 0
FE NI CR
55.845 58.69 51.996
6 1 2 3 4 5 6
1 1
LIQUID
QKTO
"""
    for i, (element, group) in enumerate(zip(("FE", "NI", "CR"), groups, strict=True)):
        counts = " ".join("1" if j == i else "0" for j in range(3))
        text += f"{element}\n1 1 {counts}\n2000 0 0 0 0 0 0\n1 {group}\n"
    text += interaction+"\n0\n"
    return CefPhaseModel(database(text), "LIQUID")


@pytest.mark.parametrize("groups,expected", [((1, 1, 1), 24), ((1, 2, 1), 42),
                                            ((1, 2, 2), 12)])
def test_kohler_and_both_toop_choices_are_different_physical_extrapolations(groups, expected):
    model = ternary_model(groups)
    sites = [{"FE": .2, "NI": .3, "CR": .5}]
    assert model.evaluate(sites, 700).contributions_J_mol["excess"] == pytest.approx(expected)
    if groups == (1, 1, 1):
        reversed_model = ternary_model(groups, "2 2 1 2 1 1000")
        assert reversed_model.evaluate(sites, 700).contributions_J_mol["excess"] == (
            pytest.approx(36))
        pure = model.evaluate([{"FE": 0., "NI": 0., "CR": 1.}], 700)
        assert pure.gibbs_J_mol == 0


def test_ternary_qkto_exponents_are_offset_once_and_not_replaced_by_rk_order():
    model = ternary_model((1, 1, 1), "3 1 2 3 2 1 1 1000")
    sites = [{"FE": .2, "NI": .3, "CR": .5}]
    assert model.evaluate(sites, 700).contributions_J_mol["excess"] == pytest.approx(6)
