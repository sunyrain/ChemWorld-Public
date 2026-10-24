"""Finite original-G phase boundaries, honest failures and actual-stock selection."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.caloric_math import derivative_values, exponential, logarithm
from chemworld.physchem.calphad_equilibrium import CalphadEquilibriumResult, calphad_tp_equilibrium
from chemworld.physchem.calphad_phase_diagram import (
    CalphadDiagramResult,
    CalphadDiagramSpec,
    CriticalPointRequest,
    _DiagramEvaluator,
    scan_calphad_phase_diagram,
)
from chemworld.physchem.calphad_phase_trace import locate_critical, locate_crossing, trace_branch
from chemworld.runtime.calphad_phase_diagram import select_calphad_diagram_inventory_point
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget
from reference.calphad42_diagram_cases import diagram_spec, equilibrium_spec
from reference.calphad42_diagram_reference import R, binodal
from reference.calphad_equilibrium_models import BINARY, ELEMENTS, native_spec, solution


@pytest.fixture(scope="module")
def diagrams():
    return {name: scan_calphad_phase_diagram(diagram_spec(name))
            for name in ("REGULAR", "LOOP", "CROSSING")}


@pytest.mark.parametrize("family", ["REGULAR", "LOOP", "CROSSING"])
def test_finite_map_and_real_boundary_graph(diagrams, family):
    result = diagrams[family]
    assert result.complete
    assert len(result.scan_points) == 25
    for point in result.boundaries:
        assert point["compositions"] == pytest.approx(binodal(family, point["temperature_K"]),
                                                     abs=2e-5)
    for branch in result.branches:
        assert branch["complete"] and not branch["breaks"]
        assert all(path["edges"] and path["termination"] for path in branch["paths"])


def test_critical_endpoints_reentrance_and_nonunique_crossing(diagrams):
    regular, loop, crossing = (diagrams[k] for k in ("REGULAR", "LOOP", "CROSSING"))
    assert regular.endpoints[0]["temperature_K"] == pytest.approx(12000/(2*R), abs=.005)
    assert [p["temperature_K"] for p in loop.endpoints] == pytest.approx([550, 850], abs=.005)
    assert [p["split_side"] for p in loop.endpoints] == [1, -1]
    for result in (regular, loop):
        for p in result.endpoints:
            assert p["derivatives_J_mol"][4] > 0
            assert p["candidate_equilibrium"]["success"]
    assert regular.windows[0]["intervals"][0]["temperature_bounds_K"] == pytest.approx(
        [450, 656.8567394], abs=.005)
    assert loop.windows[0]["intervals"][0]["temperature_bounds_K"] == pytest.approx(
        [586.4991277, 797.1026349], abs=.005)
    point = crossing.endpoints[0]
    assert point["kind"] == "nonunique_first_order" and point["phase_fractions"] is None
    assert point["temperature_K"] == pytest.approx(700, abs=.005)
    assert point["enthalpy_interval_J_mol"][1]-point["enthalpy_interval_J_mol"][0] == \
        pytest.approx(10000)
    assert not crossing.windows[0]["intervals"]


def test_warm_seeds_preserve_default_and_only_guide_original_search():
    spec = equilibrium_spec("REGULAR")
    args = (spec, 600, 101325, {"FE": .25, "NI": .75})
    cold = calphad_tp_equilibrium(*args)
    assert cold.to_dict() == calphad_tp_equilibrium(*args, initial_phases=None).to_dict()
    assert cold.to_dict() == calphad_tp_equilibrium(*args, initial_phases=()).to_dict()
    warm = calphad_tp_equilibrium(*args, initial_phases=cold.phases)
    assert warm.success and warm.total_gibbs_J == pytest.approx(cold.total_gibbs_J)
    assert warm.diagnostics["rounds"][-1]["negative_minima"] == 0
    assert all(row["accepted"] for row in warm.diagnostics["warm_seed_attempts"])
    bogus = replace(cold.phases[0], phase_name="UNDECLARED", amount_mol_atoms=10000)
    ignored = calphad_tp_equilibrium(*args, initial_phases=(bogus,))
    assert ignored.success
    assert sum(p.amount_mol_atoms for p in ignored.phases) == pytest.approx(1)
    assert not ignored.diagnostics["warm_seed_attempts"][0]["accepted"]


def stock(x=.25):
    return PhaseLedger({"feed": PhaseRecord("feed", "vessel", "solid", .02,
                                           {"FE": 2*x, "NI": 2*(1-x)}),
                        "other": PhaseRecord("other", "elsewhere", "solid", .01, {"CR": 3})})


def select(ledger, diagram, index):
    return select_calphad_diagram_inventory_point(
        ledger, diagram, index, vessel_id="vessel", source_phase_ids=("feed",),
        phase_targets=(CalphadPhaseTarget("left", "ALPHA", .007),
                       CalphadPhaseTarget("right", "ALPHA", .009)))


def test_json_query_continuation_recomputes_actual_stock_and_preserves_owner(diagrams):
    encoded = json.loads(json.dumps(diagrams["REGULAR"].to_dict()))
    clone = CalphadDiagramResult.from_dict(encoded)
    # Cached analysis amounts cannot become material stock.
    index = next(i for i, p in enumerate(clone.scan_points)
                 if p["temperature_K"] == 600 and p["composition"] == .25)
    clone.scan_points[index]["equilibrium"]["phases"][0]["amount_mol_atoms"] = 1e9
    ledger = stock()
    before = ledger.to_dict()
    result = select(ledger, clone, index)
    assert result.success
    assert result.ledger.total_amounts_mol() == pytest.approx({"FE": .5, "NI": 1.5, "CR": 3})
    assert ledger.to_dict() == before
    assert result.ledger.phases["other"] == ledger.phases["other"]
    assert result.to_dict()["energy_scope"] == "TP_material_boundary; no_heat_duty_or_time_advance"
    with pytest.raises(ValueError, match="composition"):
        select(stock(.3), clone, index)
    with pytest.raises(ValueError):
        CalphadDiagramResult.from_dict({**encoded, "physical_time_s": 1})
    with pytest.raises(ValueError):
        CalphadDiagramSpec.from_dict({**encoded["specification"], "invented": 1})


def test_budget_failure_cannot_be_installed():
    spec = diagram_spec("REGULAR")
    limited = scan_calphad_phase_diagram(replace(spec, policy=replace(spec.policy,
                                                                   maximum_evaluations=1)))
    assert not limited.complete
    index = next(i for i, p in enumerate(limited.scan_points) if not p["success"])
    with pytest.raises(ValueError, match="unresolved"):
        select(stock(), limited, index)
    assert any(a["equilibrium"]["failure_reason"] == "diagram_TP_budget_exhausted"
               for a in limited.attempts)


def test_resolved_loop_constitutions_seed_actual_inventory_closure(diagrams):
    diagram = diagrams["LOOP"]
    index = next(i for i, p in enumerate(diagram.scan_points)
                 if p["temperature_K"] == 600 and p["composition"] == .25)
    cold = calphad_tp_equilibrium(diagram.specification.equilibrium, 600, 101325,
                                 {"FE": .5, "NI": 1.5})
    assert not cold.success
    assert cold.failure_reason == "negative_tangent_minimum_without_new_column"
    ledger = stock()
    result = select(ledger, diagram, index)
    assert result.success
    assert result.ledger.total_amounts_mol() == pytest.approx({"FE": .5, "NI": 1.5, "CR": 3})
    assert [p.element_fractions["FE"] for p in result.equilibrium.phases] == pytest.approx(
        binodal("LOOP", 600), abs=2e-6)
    assert result.equilibrium.diagnostics["rounds"][-1]["negative_minima"] == 0
    assert result.ledger.phases["other"] == ledger.phases["other"]
    assert set(ledger.phases) == {"feed", "other"}


def test_failed_temperature_cannot_be_bridged(monkeypatch):
    import chemworld.physchem.calphad_phase_diagram as api
    original = api.calphad_tp_equilibrium

    def fault(spec, temperature, *args, **kwargs):
        if temperature == 650:
            return CalphadEquilibriumResult(False, failure_reason="injected_T650_failure")
        return original(spec, temperature, *args, **kwargs)

    monkeypatch.setattr(api, "calphad_tp_equilibrium", fault)
    evaluator = _DiagramEvaluator(diagram_spec("REGULAR"))
    branch = trace_branch(evaluator, (600, .5), (), 1, 0)
    assert not branch["complete"]
    assert branch["breaks"]
    for path in branch["paths"]:
        assert all(not min(a, b) < 650 < max(a, b) for a, b in path["edges"])
    assert any(a["equilibrium"]["failure_reason"] == "injected_T650_failure"
               for a in evaluator.attempts)


@pytest.mark.parametrize("cut", [625, 721])
def test_tdb_segment_cut_including_terminal_critical_link_is_not_bridged(cut):
    text = BINARY.replace("100 12000; 2500 N", f"100 12000; {cut} Y 12000; 2500 N")
    spec = replace(diagram_spec("REGULAR"),
                   equilibrium=native_spec({"tdb": text, "phases": ["ALPHA"]}))
    evaluator = _DiagramEvaluator(spec)
    endpoint = locate_critical(evaluator, spec.critical_requests[0], 0)
    assert endpoint["success"]
    branch = trace_branch(evaluator, (600, .5), (endpoint,), 1, 0)
    assert not branch["complete"]
    assert any(b["reason"] == "TDB_segment_boundary" for b in branch["breaks"])
    assert all(not min(a, b) < cut <= max(a, b)
               for path in branch["paths"] for a, b in path["edges"])


def test_equal_g_but_different_element_potentials_is_not_same_composition_crossing():
    text = (ELEMENTS + solution("ALPHA", ("FE", "NI"), (0, 0)) + "\n" +
            solution("BETA", ("FE", "NI"),
                     ("10000*(1-T/700)+10", "10000*(1-T/700)-10")))
    spec = replace(diagram_spec("CROSSING"),
                   equilibrium=native_spec({"tdb": text, "phases": ["ALPHA", "BETA"]}))
    result = locate_crossing(_DiagramEvaluator(spec), spec.crossing_requests[0], 0)
    assert not result["success"]
    assert result["failure_reason"] == "crossing_candidates_do_not_share_common_tangent"
    assert result["element_potential_error_J_mol"] > 1


def test_negative_fourth_derivative_cannot_be_a_stable_critical_point():
    text = (ELEMENTS + solution("ALPHA", ("FE", "NI"), (0, 0)) + "\n" +
            "PARAMETER L(ALPHA,FE,NI;0) 100 2*8.3145*700+10000; 2500 N !\n" +
            "PARAMETER L(ALPHA,FE,NI;2) 100 10000; 2500 N !\n")
    request = CriticalPointRequest("ALPHA", 695, .48, (680, 720), (.35, .65))
    spec = replace(diagram_spec("REGULAR"),
                   equilibrium=native_spec({"tdb": text, "phases": ["ALPHA"]}))
    result = locate_critical(_DiagramEvaluator(spec), request, 0)
    assert not result["success"]
    assert result["failure_reason"] == "critical_requires_positive_fourth_derivative"
    assert result["derivatives_J_mol"][4] < 0


def test_higher_order_temperature_crossing_is_not_labelled_first_order():
    text = (ELEMENTS + solution("ALPHA", ("FE", "NI"), (0, 0)) + "\n" +
            solution("BETA", ("FE", "NI"), ("0.001*(T-700)^3", "0.001*(T-700)^3")))
    spec = replace(diagram_spec("CROSSING"),
                   equilibrium=native_spec({"tdb": text, "phases": ["ALPHA", "BETA"]}))
    request = replace(spec.crossing_requests[0], temperature_K=700.)
    result = locate_crossing(_DiagramEvaluator(spec), request, 0)
    assert not result["success"]
    assert result["failure_reason"] == "crossing_has_no_resolved_first_order_enthalpy_jump"


def test_another_stable_phase_preempts_candidate_critical():
    text = BINARY + solution("BETA", ("FE", "NI"), (-10000, -10000))
    spec = replace(diagram_spec("REGULAR"),
                   equilibrium=native_spec({"tdb": text, "phases": ["ALPHA", "BETA"]}))
    result = locate_critical(_DiagramEvaluator(spec), spec.critical_requests[0], 0)
    assert not result["success"]
    assert result["failure_reason"] == "critical_candidate_common_tangent_unconfirmed"


def test_original_derivative_jets_extend_to_four_without_changing_lower_orders():
    expected = (16, 32, 48, 48, 24)
    for order in range(5):
        assert derivative_values(lambda x: x**4, 2, order=order) == expected[:order+1]
    for order in (-1, 5, True, 1.5):
        with pytest.raises(ValueError):
            derivative_values(lambda x: x**4, 2, order=order)
    assert derivative_values(lambda x: logarithm(exponential(x*x)), 2, order=4) == \
        pytest.approx((4, 4, 2, 0, 0), abs=1e-10)


def test_binary_coordinate_scope_rejects_multisublattice():
    from reference.calphad_equilibrium_models import SUBLATTICE
    original = native_spec({"tdb": SUBLATTICE, "phases": ["ORDERED"]})
    with pytest.raises(ValueError, match="single binary"):
        replace(diagram_spec("REGULAR"), equilibrium=original)
