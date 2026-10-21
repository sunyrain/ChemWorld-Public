"""Analytic CEF contracts, including ionic normalization and real contributions."""

import json
from math import exp, log

import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.tdb import TdbDatabase
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.data_conflicts import DataSourceProvenance

R = 8.3145
ELEMENTS = """
ELEMENT FE BCC_A2 55.845 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
ELEMENT CR BCC_A2 51.996 0 0 !
ELEMENT CO HCP_A3 58.933 0 0 !
ELEMENT O GAS 15.999 0 0 !
ELEMENT VA VACUUM 0 0 0 !
TYPE_DEF % SEQ * !
"""
BINARY = """
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :FE,NI: !
PARAMETER G(ALPHA,FE;0) 100 -1000+10*T-20*T*LN(T); 2000 N !
PARAMETER G(ALPHA,NI;0) 100 -2000+5*T-25*T*LN(T); 2000 N !
PARAMETER L(ALPHA,FE,NI;0) 100 1000+T; 2000 N !
PARAMETER L(ALPHA,FE,NI;1) 100 -500+2*T; 2000 N !
"""


def model(text=BINARY, phase="ALPHA", **kwargs):
    source = InputSource(DataSourceProvenance("synthetic-cef", 0, "Analytic CEF test", "1", "CC0"),
                         "synthetic")
    return CefPhaseModel(TdbDatabase.from_string(ELEMENTS + text, source=source), phase, **kwargs)


def test_binary_g_h_s_site_response_and_exact_model_roundtrip():
    x, t = .3, 700
    m = model()
    y = [{"FE": x, "NI": 1-x}]
    p = m.evaluate(y, t)
    gf, gn = -1000+10*t-20*t*log(t), -2000+5*t-25*t*log(t)
    df, dn = 10-20*(log(t)+1), 5-25*(log(t)+1)
    l0, l1 = 1000+t, -500+2*t
    mixing = x*log(x)+(1-x)*log(1-x)
    g = x*gf+(1-x)*gn+R*t*mixing+x*(1-x)*(l0+l1*(2*x-1))
    dg = x*df+(1-x)*dn+R*mixing+x*(1-x)*(1+2*(2*x-1))
    assert p.gibbs_J_mol == pytest.approx(g, abs=1e-9)
    assert p.entropy_J_mol_K == pytest.approx(-dg, abs=1e-11)
    assert p.enthalpy_J_mol == pytest.approx(g-t*dg, abs=1e-9)
    assert sum(p.contributions_J_mol.values()) == pytest.approx(g)
    assert dict(p.element_fractions) == {"FE": x, "NI": 1-x}
    partials = m.site_partials(y, t)
    dx = gf-gn+R*t*log(x/(1-x))+l0*(1-2*x)+l1*(-6*x*x+6*x-1)
    assert partials[0, "FE"]-partials[0, "NI"] == pytest.approx(dx, abs=1e-9)
    clone = CefPhaseModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert clone.to_dict() == m.to_dict()
    assert clone.evaluate(y, t) == p


def test_vacancies_and_multisublattice_atom_basis():
    m = model("""
    PHASE ALPHA % 2 2 3 !
    CONSTITUENT ALPHA :FE:O,VA: !
    PARAMETER G(ALPHA,FE:O;0) 100 -5000; 2000 N !
    PARAMETER G(ALPHA,FE:VA;0) 100 -1000; 2000 N !
    """)
    p = m.evaluate([{"FE": 1}, {"O": .4, "VA": .6}], 500)
    normalization = 2+3*.4
    assert p.element_fractions["FE"] == pytest.approx(2/normalization)
    assert p.element_fractions["O"] == pytest.approx(1.2/normalization)
    assert p.enthalpy_J_mol == pytest.approx((-.4*5000-.6*1000)/normalization)
    assert p.entropy_J_mol_K == pytest.approx(-3*R*(.4*log(.4)+.6*log(.6))/normalization)


def test_ternary_implicit_symmetry_and_muggianu_extension():
    text = """
    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :FE,NI,CR,CO: !
    PARAMETER L(ALPHA,FE,NI,CR;0) 100 900; 2000 N !
    """
    m = model(text, unspecified_endmembers="zero")
    y = [{"FE": .2, "NI": .3, "CR": .1, "CO": .4}]
    assert m.evaluate(y, 500).contributions_J_mol["excess"] == pytest.approx(900*.2*.3*.1)
    m = model(text + "\nPARAMETER L(ALPHA,FE,NI,CR;1) 100 600; 2000 N !",
              unspecified_endmembers="zero")
    # TDB canonical order CR, FE, NI; correction is the omitted CO fraction / 3.
    expected = .2*.3*.1*(900*(.1+.4/3)+600*(.2+.4/3))
    assert m.evaluate(y, 500).contributions_J_mol["excess"] == pytest.approx(expected)


def test_reciprocal_parameters_select_the_declared_mixing_sublattice():
    text = """
    PHASE ALPHA % 2 1 1 !
    CONSTITUENT ALPHA :FE,NI:CR,CO: !
    PARAMETER L(ALPHA,FE,NI:CR,CO;0) 100 100; 2000 N !
    PARAMETER L(ALPHA,FE,NI:CR,CO;1) 100 200; 2000 N !
    PARAMETER L(ALPHA,FE,NI:CR,CO;2) 100 300; 2000 N !
    """
    p = model(text, unspecified_endmembers="zero").evaluate(
        [{"FE": .2, "NI": .8}, {"CO": .3, "CR": .7}], 500,
    )
    assert p.contributions_J_mol["excess"] == pytest.approx(
        .2*.8*.3*.7*(100+200*(.3-.7)+300*(.2-.8))/2,
    )


def test_fixed_charged_phase_needs_real_neutrality():
    m = model("""
    SPECIES FE+2 FE1/+2 !
    SPECIES O-2 O1/-2 !
    PHASE ALPHA:I % 2 1 1 !
    CONSTITUENT ALPHA :FE+2:O-2,VA: !
    PARAMETER G(ALPHA,FE+2:O-2;0) 100 -10000; 2000 N !
    PARAMETER G(ALPHA,FE+2:VA;0) 100 0; 2000 N !
    """)
    assert m.evaluate([{"FE+2": 1}, {"O-2": 1, "VA": 0}], 500).enthalpy_J_mol == -5000
    with pytest.raises(ValueError, match="charge"):
        m.evaluate([{"FE+2": 1}, {"O-2": .8, "VA": .2}], 500)


def test_ionic_variable_valence_vacancy_and_neutral_short_parameter():
    m = model("""
    SPECIES FE+1 FE1/+1 !
    SPECIES FE+2 FE1/+2 !
    SPECIES O-2 O1/-2 !
    SPECIES O2 O2 !
    PHASE ALPHA:Y % 2 1 1 !
    CONSTITUENT ALPHA :FE+1,FE+2:O-2,VA,O2: !
    PARAMETER G(ALPHA,FE+1:O-2;0) 100 -1000; 2000 N !
    PARAMETER G(ALPHA,FE+2:O-2;0) 100 -2000; 2000 N !
    PARAMETER G(ALPHA,FE+1:VA;0) 100 -500; 2000 N !
    PARAMETER G(ALPHA,FE+2:VA;0) 100 -800; 2000 N !
    PARAMETER G(ALPHA,O2;0) 100 -100; 2000 N !
    """)
    p = m.evaluate([{"FE+1": .4, "FE+2": .6}, {"O-2": .5, "VA": .2, "O2": .3}], 500)
    normalization = 1.32+1.6*(.5+2*.3)
    reference = -1000*.4*.5-2000*.6*.5-500*.4*.2*1.6-800*.6*.2*1.6-100*.3*1.6
    assert p.enthalpy_J_mol == pytest.approx(reference/normalization, abs=1e-9)
    assert p.element_fractions["FE"] == pytest.approx(1.32/normalization)
    assert p.element_fractions["O"] == pytest.approx(1.76/normalization)
    entropy = -R*(1.32*(.4*log(.4)+.6*log(.6))
                  +1.6*(.5*log(.5)+.2*log(.2)+.3*log(.3)))/normalization
    assert p.entropy_J_mol_K == pytest.approx(entropy)


ORDERED = """
PHASE DIS % 1 1 !
CONSTITUENT DIS :FE,NI: !
PARAMETER G(DIS,FE;0) 100 0; 2000 N !
PARAMETER G(DIS,NI;0) 100 0; 2000 N !
PHASE ORD %@ 2 .5 .5 !
CONSTITUENT ORD :FE,NI:FE,NI: !
TYPE_DEF @ GES A_P_D ORD DIS_PART DIS !
PARAMETER G(ORD,FE:FE;0) 100 0; 2000 N !
PARAMETER G(ORD,NI:NI;0) 100 0; 2000 N !
PARAMETER G(ORD,FE:NI;0) 100 -1000; 2000 N !
PARAMETER G(ORD,NI:FE;0) 100 -1000; 2000 N !
"""


@pytest.mark.parametrize("never", [False, True])
def test_ordered_energy_and_never_disorder_keep_one_atomic_basis(never):
    text = ORDERED.replace("DIS_PART", "NEVER_DISORDER") if never else ORDERED
    m = model(text, phase="ORD")
    p = m.evaluate([{"FE": .8, "NI": .2}, {"FE": .2, "NI": .8}], 500)
    assert p.enthalpy_J_mol == pytest.approx(-680 if never else -180, abs=1e-9)
    assert p.entropy_J_mol_K == pytest.approx(-R*(.8*log(.8)+.2*log(.2)))
    if not never:
        randomized = m.evaluate([{"FE": .5, "NI": .5}, {"FE": .5, "NI": .5}], 500)
        assert randomized.gibbs_J_mol == pytest.approx(-R*500*log(2))
        assert randomized.contributions_J_mol["ordering"] == 0


@pytest.mark.parametrize(("option", "array", "count"), [
    ("F", "FE:FE:FE:NI", 3), ("B", "FE:FE:NI:NI", 1),
])
def test_four_sublattice_symmetry_is_derived_without_mutating_input(option, array, count):
    text = ORDERED.split("PHASE ORD")[0] + f"""
    PHASE ORD:{option} %@ 4 .25 .25 .25 .25 !
    CONSTITUENT ORD :FE,NI:FE,NI:FE,NI:FE,NI: !
    TYPE_DEF @ GES A_P_D ORD DIS_PART DIS !
    PARAMETER G(ORD,{array};0) 100 1000; 2000 N !
    """
    m = model(text, phase="ORD", unspecified_endmembers="zero")
    assert m.generated_parameter_count == count
    assert len([p for p in m.database.parameters if p.phase == "ORD"]) == 1
    p = m.evaluate([{"FE": .5, "NI": .5}] * 4, 500)
    assert p.contributions_J_mol["ordering"] == 0
    assert p.gibbs_J_mol == pytest.approx(-R*500*log(2))


def pure(extra="", *, magnetic=None):
    text = """
    PHASE ALPHA %& 1 1 !
    CONSTITUENT ALPHA :FE: !
    PARAMETER G(ALPHA,FE;0) 100 0; 2000 N !
    """
    if magnetic is None:
        text = text.replace("%&", "%")
    else:
        text += f"\nTYPE_DEF & GES A_P_D ALPHA MAGNETIC {magnetic} .4 !\n"
    return model(text + extra)


def test_explicit_two_state_zero_and_einstein_log_temperature_are_not_missing():
    assert pure().evaluate([{"FE": 1}], 500).gibbs_J_mol == 0
    p = pure("\nPARAMETER GD(ALPHA,FE;0) 100 0; 2000 N !").evaluate([{"FE": 1}], 500)
    assert p.gibbs_J_mol == pytest.approx(-R*500*log(2))
    assert p.enthalpy_J_mol == pytest.approx(0, abs=1e-9)
    assert p.entropy_J_mol_K == pytest.approx(R*log(2))
    p = pure(f"\nPARAMETER THETA(ALPHA,FE;0) 100 {log(400)}; 2000 N !").evaluate([{"FE": 1}], 500)
    assert p.gibbs_J_mol == pytest.approx(1.5*R*400+3*R*500*log(1-exp(-400/500)))
    assert p.enthalpy_J_mol == pytest.approx(1.5*R*400+3*R*400/(exp(400/500)-1))
    zero_log = pure("\nPARAMETER THETA(ALPHA,FE;0) 100 0; 2000 N !").evaluate([{"FE": 1}], 500)
    assert zero_log.gibbs_J_mol == pytest.approx(1.5*R+3*R*500*log(1-exp(-1/500)))
    tiny = pure("\nPARAMETER THETA(ALPHA,FE;0) 100 -20; 2000 N !").evaluate([{"FE": 1}], 500)
    assert tiny.gibbs_J_mol == pytest.approx(3*R*500*(-20-log(500)), abs=1e-8)
    assert tiny.enthalpy_J_mol == pytest.approx(3*R*500, abs=1e-8)


@pytest.mark.parametrize(("afm", "tc", "nt"), [(-1, 1000, 0), (-1, -1000, 0), (0, 1000, 500)])
def test_magnetic_models_are_finite_nonzero_and_keep_critical_energy(afm, tc, nt):
    m = pure(f"""
    PARAMETER TC(ALPHA,FE;0) 100 {tc}; 2000 N !
    PARAMETER NT(ALPHA,FE;0) 100 {nt}; 2000 N !
    PARAMETER BMAGN(ALPHA,FE;0) 100 2; 2000 N !
    """, magnetic=afm)
    for temperature in (300, 700, 1000, 1200):
        p = m.evaluate([{"FE": 1}], temperature)
        assert p.gibbs_J_mol < 0 and p.entropy_J_mol_K < 0
    critical = m.evaluate([{"FE": 1}], 1000)
    assert critical.gibbs_J_mol != 0
    assert any("critical" in note for note in critical.diagnostics)


def test_volume_first_derivative_and_unimplemented_high_pressure_do_not_cancel():
    text = """
    PARAMETER V0(ALPHA,FE;0) 100 1E-5; 2000 N !
    PARAMETER VA(ALPHA,FE;0) 100 .001*T; 2000 N !
    """
    p = pure(text).evaluate([{"FE": 1}], 500, 201325)
    assert p.pressure_derivative_m3_mol == pytest.approx(1e-5*exp(.5))
    assert p.gibbs_J_mol == pytest.approx(exp(.5))
    m = pure(text + """
    PARAMETER VK(ALPHA,FE;0) 100 1; 2000 N !
    PARAMETER VC(ALPHA,FE;0) 100 -1; 2000 N !
    """)
    with pytest.raises(ValueError, match="high-pressure"):
        m.evaluate([{"FE": 1}], 500, 201325)
    with pytest.raises(ValueError, match="high-pressure"):
        m.evaluate([{"FE": 1}], 500)
    assert m.evaluate([{"FE": 1}], 500, pressure_derivative=False).gibbs_J_mol == 0


def test_invalid_constitutions_domains_missing_parameters_and_endpoint_partials():
    m = model()
    for fractions in ([{"FE": .1, "NI": .8}], [{"FE": -1, "NI": 2}], [{"FE": True, "NI": 0}],
                      [{"FE": 1}], [{"FE": float("nan"), "NI": .5}]):
        with pytest.raises(ValueError):
            m.evaluate(fractions, 500)
    with pytest.raises(ValueError, match="outside declared"):
        m.evaluate([{"FE": .5, "NI": .5}], 99)
    assert m.evaluate([{"FE": 1, "NI": 0}], 500).gibbs_J_mol < 0
    with pytest.raises(ValueError, match="diverges"):
        m.site_partials([{"FE": 1, "NI": 0}], 500)
    with pytest.raises(ValueError, match="endmember"):
        model("PHASE ALPHA % 1 1 !\nCONSTITUENT ALPHA :FE,NI: !")


def test_unassigned_type_marker_is_visible_and_defined_unknown_physics_is_rejected():
    text = BINARY.replace("PHASE ALPHA %", "PHASE ALPHA %'")
    p = model(text).evaluate([{"FE": .3, "NI": .7}], 700)
    assert p.gibbs_J_mol == model().gibbs([{"FE": .3, "NI": .7}], 700)
    assert any("unassigned" in note for note in p.diagnostics)
    with pytest.raises(ValueError, match="unsupported type definition"):
        model(text + "\nTYPE_DEF ' GES A_P_D ALPHA HKF_ELECTROSTATIC 1 !")


def test_defined_gibbs_does_not_promise_a_finite_derivative_at_a_branch_boundary():
    m = model("""
    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :FE: !
    PARAMETER G(ALPHA,FE;0) 100 (T-100)**.5; 2000 N !
    """)
    assert m.gibbs([{"FE": 1}], 100) == 0
    with pytest.raises(ValueError, match="real domain"):
        m.evaluate([{"FE": 1}], 100)


def test_second_response_has_analytic_cp_mixed_partials_and_constrained_composition_curvature():
    import numpy as np

    m, x, t = model(), .3, 700
    y = [{"FE": x, "NI": 1-x}]
    response = m.temperature_response(y, t)
    assert response.fixed_constitution_heat_capacity_J_mol_K == pytest.approx(20*x+25*(1-x))
    d = m.derivatives(y, t)
    assert d.coordinate_names == ("temperature", "pressure", "site:0:FE", "site:0:NI")
    assert d.coordinate_units == ("K", "Pa", "dimensionless", "dimensionless")
    from chemworld.foundation.units import unit_spec

    for unit in (*d.coordinate_units, *d.output_units):
        unit_spec(unit)
    assert d.values[0] == pytest.approx(response.gibbs_J_mol)
    assert -t*d.hessian[0, 0, 0] == pytest.approx(20*x+25*(1-x))
    expected_mixed = (10-20*(log(t)+1))-(5-25*(log(t)+1))+R*log(x/(1-x))
    expected_mixed += 1-2*x+2*(-6*x*x+6*x-1)
    assert d.hessian[0, 0, 2]-d.hessian[0, 0, 3] == pytest.approx(expected_mixed)
    expected_curvature = R*t/(x*(1-x))-2*(1000+t)+(-500+2*t)*(-12*x+6)
    direction = np.array([0, 0, 1, -1])
    assert direction @ d.hessian[0] @ direction == pytest.approx(expected_curvature)
    assert d.jacobian[d.output_names.index("element:FE"), 2] == pytest.approx(1-x)
    assert d.jacobian[d.output_names.index("element:FE"), 3] == pytest.approx(-x)
    assert not d.hessian.flags.writeable


def test_second_response_refuses_implicit_critical_and_segment_side_or_zero_site_hessian():
    m = pure("""
    PARAMETER TC(ALPHA,FE;0) 100 1000; 2000 N !
    PARAMETER BMAGN(ALPHA,FE;0) 100 2; 2000 N !
    """, magnetic=-1)
    with pytest.raises(ValueError, match="one-sided"):
        m.temperature_response([{"FE": 1}], 1000)
    p = m.temperature_response([{"FE": 1}], 1000, allow_one_sided=True)
    assert p.diagnostics and p.fixed_constitution_heat_capacity_J_mol_K > 0
    segmented = model("""
    PHASE ALPHA % 1 1 !
    CONSTITUENT ALPHA :FE: !
    PARAMETER G(ALPHA,FE;0) 100 -20*T*LN(T); 500 Y -25*T*LN(T); 2000 N !
    """)
    with pytest.raises(ValueError, match="one-sided"):
        segmented.temperature_response([{"FE": 1}], 500)
    assert segmented.temperature_response(
        [{"FE": 1}], 500, allow_one_sided=True,
    ).fixed_constitution_heat_capacity_J_mol_K == pytest.approx(25)
    with pytest.raises(ValueError, match="interior"):
        model().derivatives([{"FE": 1, "NI": 0}], 500)


def test_elemental_potentials_follow_euler_and_composition_tangent_not_site_partials():
    m, x, t = model(), .3, 700
    y = [{"FE": x, "NI": 1-x}]
    p = m.evaluate(y, t)
    partials = m.site_partials(y, t)
    direction = partials[0, "FE"]-partials[0, "NI"]
    mu = m.chemical_potentials(y, t)
    assert mu.status == "determined" and mu.internal_minimum
    assert mu.values_J_mol["FE"] == pytest.approx(p.gibbs_J_mol+(1-x)*direction)
    assert mu.values_J_mol["NI"] == pytest.approx(p.gibbs_J_mol-x*direction)
    assert x*mu.values_J_mol["FE"]+(1-x)*mu.values_J_mol["NI"] == pytest.approx(p.gibbs_J_mol)


def stoichiometric():
    return model("""
    PHASE ALPHA % 2 1 1 !
    CONSTITUENT ALPHA :FE:O: !
    PARAMETER G(ALPHA,FE:O;0) 100 -1000; 2000 N !
    """)


def test_stoichiometric_mu_is_underdetermined_and_unreachable_stock_is_not_repaired():
    m = stoichiometric()
    sites = [{"FE": 1}, {"O": 1}]
    mu = m.chemical_potentials(sites, 500)
    assert mu.status == "underdetermined" and mu.values_J_mol is None
    assert mu.matrix_rank < mu.unknowns
    solved = m.stationary_constitution({"FE": .5, "O": .5}, sites, 500)
    assert solved.success and solved.chemical_potentials.status == "underdetermined"
    assert solved.relaxed_heat_capacity_J_mol_K == pytest.approx(0, abs=1e-9)
    bad = m.stationary_constitution({"FE": .6, "O": .4}, sites, 500)
    assert not bad.success and bad.site_fractions is None and bad.properties is None
    assert sites == [{"FE": 1}, {"O": 1}]


def test_ordered_nonstationary_and_saddle_constitutions_do_not_get_unqualified_mu():
    from chemworld.physchem.calphad_stationary import CefStationaryPolicy

    m = model(ORDERED, phase="ORD")
    off = m.chemical_potentials([{"FE": .8, "NI": .2}, {"FE": .2, "NI": .8}], 500)
    assert off.status == "nonstationary" and off.values_J_mol is None
    uniform = [{"FE": .5, "NI": .5}]*2
    saddle = m.chemical_potentials(uniform, 110)
    assert saddle.status == "unstable" and saddle.values_J_mol is None
    assert saddle.minimum_internal_curvature_J_mol < 0
    permitted = m.chemical_potentials(uniform, 110, require_local_minimum=False)
    assert permitted.status == "determined" and not permitted.internal_minimum
    result = m.stationary_constitution(
        {"FE": .5, "NI": .5}, uniform, 110, policy=CefStationaryPolicy(maximum_saddle_escapes=0),
    )
    assert not result.success and result.failure_reason == "internal_saddle"
    assert result.site_fractions is None
    escaped = m.stationary_constitution({"FE": .5, "NI": .5}, uniform, 110)
    assert escaped.success and escaped.diagnostics["saddle_escapes"]
    assert escaped.diagnostics["internal_minimum"]
    assert escaped.diagnostics["total_root_iterations"] <= 50
    assert escaped.properties.gibbs_J_mol < m.gibbs(uniform, 110)


def test_local_ordering_root_and_relaxed_cp_match_independent_scalar_stationarity():
    from scipy.optimize import brentq

    m, t = model(ORDERED, phase="ORD"), 110
    original = [{"FE": .8, "NI": .2}, {"FE": .2, "NI": .8}]
    result = m.stationary_constitution({"FE": .5, "NI": .5}, original, t)
    assert result.success, result.diagnostics
    y = brentq(lambda x: -4000*(x-.5)+R*t*log(x/(1-x)), .500001, .999999, xtol=1e-14)
    assert result.site_fractions[0]["FE"] == pytest.approx(y, abs=1e-9)
    assert result.site_fractions[1]["FE"] == pytest.approx(1-y, abs=1e-9)
    dy = -R*log(y/(1-y))/(-4000+R*t/(y*(1-y)))
    expected_cp = -4000*(y-.5)*dy
    assert result.relaxed_heat_capacity_J_mol_K == pytest.approx(expected_cp, rel=1e-8)
    assert result.diagnostics["fixed_constitution_heat_capacity_J_mol_K"] == pytest.approx(
        0, abs=1e-9,
    )
    assert result.chemical_potentials.status == "determined"
    assert result.diagnostics["root"]["algorithm"] == "scaled_projected_newton_with_gradient"
    assert original == [{"FE": .8, "NI": .2}, {"FE": .2, "NI": .8}]


def test_failed_local_root_returns_no_accepted_constitution_or_mutated_initial_guess():
    from chemworld.physchem.calphad_stationary import CefStationaryPolicy
    from chemworld.physchem.residual_system import RootPolicy

    m = model(ORDERED, phase="ORD")
    initial = [{"FE": .8, "NI": .2}, {"FE": .2, "NI": .8}]
    result = m.stationary_constitution(
        {"FE": .5, "NI": .5}, initial, 110,
        policy=CefStationaryPolicy(root=RootPolicy(maximum_iterations=1)),
    )
    assert not result.success and result.failure_reason == "iteration_limit"
    assert result.site_fractions is result.properties is result.chemical_potentials is None
    assert initial == [{"FE": .8, "NI": .2}, {"FE": .2, "NI": .8}]


def test_magnetic_moment_sign_cusp_refuses_fake_higher_derivatives():
    m = pure("""
    PARAMETER TC(ALPHA,FE;0) 100 1000; 2000 N !
    PARAMETER BMAGN(ALPHA,FE;0) 100 T-500; 2000 N !
    """, magnetic=-1)
    assert m.gibbs([{"FE": 1}], 500) == 0
    for allow in (False, True):
        with pytest.raises(ValueError, match="sign boundary"):
            m.temperature_response([{"FE": 1}], 500, allow_one_sided=allow)
    constant_zero = pure("""
    PARAMETER TC(ALPHA,FE;0) 100 1000; 2000 N !
    PARAMETER BMAGN(ALPHA,FE;0) 100 0; 2000 N !
    """, magnetic=-1)
    assert constant_zero.temperature_response(
        [{"FE": 1}], 500,
    ).fixed_constitution_heat_capacity_J_mol_K == 0


def test_text_false_does_not_authorize_a_one_sided_response():
    with pytest.raises(ValueError, match="boolean"):
        model().temperature_response([{"FE": .3, "NI": .7}], 500, allow_one_sided="false")


def test_pressure_hessian_and_thermal_cross_derivative_keep_their_small_physical_scale():
    m = model("""
    PHASE ALPHA:G % 1 1 !
    CONSTITUENT ALPHA :FE: !
    PARAMETER G(ALPHA,FE;0) 100 R*T*LN(P/101325); 2000 N !
    """)
    t, pressure = 700, 200000
    d = m.derivatives([{"FE": 1}], t, pressure)
    assert d.jacobian[0, 1] == pytest.approx(R*t/pressure, rel=1e-13, abs=1e-16)
    assert d.hessian[0, 1, 1] == pytest.approx(-R*t/pressure**2, rel=1e-13, abs=1e-20)
    assert d.hessian[0, 0, 1] == pytest.approx(R/pressure, rel=1e-13, abs=1e-18)
