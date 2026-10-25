"""Functional boundaries for complete surfaces, invariant pools and read-only queries."""

from copy import deepcopy

import numpy as np
import pytest

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration, example_world
from chemworld.examples.native_surface_steady import (
    branching_declaration,
    branching_mechanism,
    branching_seeds,
)
from chemworld.physchem.catalysis_analysis import (
    CatalyticAnalysisConditions,
    catalytic_product_selectivity,
    catalytic_steady_states,
    map_catalytic_conditions,
    solve_catalytic_steady_state,
)
from chemworld.physchem.catalysis_steady_chart import build_catalytic_steady_chart
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.catalysis_analysis import analyze_catalytic_world


def boundary(temperature=700, pressure=5e4):
    return CatalyticAnalysisConditions(
        temperature, {"A": pressure, "B": 0, "C": 0}, {"surface": 1e-6}
    )


def trace_pool():
    payload = authored_declaration()
    payload["reactions"] = [
        {"equation": equation, "rate-constant": {"A": rate, "b": 0, "Ea": 0}}
        for equation, rate in (("A + Pt => HA", 0.5), ("HA => HB", 200), ("HB => B + Pt", 1e18))
    ]
    spec = convert_surface_mechanism(
        payload,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    ).surface
    conditions = CatalyticAnalysisConditions(700, {"A": 5e4, "B": 0}, {"surface": 1e-6})
    return spec, conditions


def test_connected_pool_resolves_sub_epsilon_trace_and_physical_spectrum():
    spec, conditions = trace_pool()
    seed = dict.fromkeys(spec.adsorbates, 1 / 3)
    root = solve_catalytic_steady_state(spec, conditions, seed, target_species="B")
    assert root["status"] == "converged", root
    adsorption = 0.5 * 5e4 / (8.31446261815324 * 700)
    vacant = 1 / (1 + adsorption / 200 + adsorption / 1e18)
    expected = {"Pt": vacant, "HA": adsorption * vacant / 200, "HB": adsorption * vacant / 1e18}
    for key, value in expected.items():
        assert root["coverages"][key] == pytest.approx(value, rel=1e-7, abs=0)
    assert 0 < root["coverages"]["HB"] < np.finfo(float).eps
    assert root["stability"] == "stable"
    # The fast physical rate is retained; a log-coordinate residual spectrum
    # would scale it by the tiny trace coverage and be scientifically wrong.
    assert min(v["real"] for v in root["eigenvalues_s_inv"]) == pytest.approx(-1e18, rel=1e-6)
    assert max(v["real"] for v in root["eigenvalues_s_inv"]) == pytest.approx(
        -200 - adsorption, rel=1e-5
    )
    assert root["steady_initialization"]["physical_time_advanced_s"] == 0
    assert root["steady_initialization"]["experiment_resource_debit"] is None
    assert seed == dict.fromkeys(spec.adsorbates, 1 / 3)


def test_relaxation_budget_failure_has_no_root_and_direct_option_is_explicit():
    spec, conditions = trace_pool()
    seed = dict.fromkeys(spec.adsorbates, 1 / 3)
    refused = solve_catalytic_steady_state(
        spec, conditions, seed, target_species="B", relaxation_maximum_evaluations=1
    )
    assert refused["status"] == "not_converged"
    assert refused["coverages"] is None
    assert refused["steady_initialization"]["rate_evaluations"] == 1
    assert "evaluation_budget_exhausted" in refused["failure_reason"]
    direct = solve_catalytic_steady_state(
        spec, conditions, seed, target_species="B", relaxation_time_s=0
    )
    assert direct["steady_initialization"]["method"] == "direct_root"
    assert direct["steady_initialization"]["rate_evaluations"] == 0


@pytest.mark.parametrize("disconnected", [False, True])
def test_full_invariants_and_independent_coordinates(disconnected):
    spec = branching_mechanism(disconnected=disconnected)
    seeds = branching_seeds(disconnected=disconnected)
    chart = build_catalytic_steady_chart(spec, boundary().capacities_mol, seeds[0])
    result = catalytic_steady_states(
        spec, boundary(), seeds, target_species="B", product_species=("B", "C"), element_basis="H"
    )
    assert result["converged_attempts"] == 3
    assert result["distinct_roots"] == 1
    root = result["roots"][0]
    assert root["steady_chart"]["conserved_dimension"] == (3 if disconnected else 2)
    assert root["steady_chart"]["reachable_dimension"] == (4 if disconnected else 3)
    assert chart.same_class(root["coverages"])
    assert root["coverages"]["I"] == 0.1
    assert root["selectivity"]["status"] == "available"
    assert sum(root["selectivity"]["fractions"].values()) == pytest.approx(1.0)
    alternate = solve_catalytic_steady_state(
        spec,
        boundary(),
        seeds[0],
        target_species="B",
        coordinate_order=tuple(reversed(spec.adsorbates)),
    )
    assert alternate["status"] == "converged"
    assert root["coverages"] == pytest.approx(alternate["coverages"], abs=1e-10)


def test_multiseed_rejects_different_inert_and_disconnected_totals():
    spec = branching_mechanism(disconnected=True)
    seed = branching_seeds(disconnected=True)[0]
    for changed in (
        {**seed, "E": seed["E"] - 0.01, "I": 0.11},
        {**seed, "E": seed["E"] - 0.01, "F": seed["F"] + 0.01},
    ):
        with pytest.raises(ValueError, match="compatibility class"):
            catalytic_steady_states(spec, boundary(), [seed, changed], target_species="B")


def zero_mechanism():
    payload = branching_declaration()
    payload["reactions"] = payload["reactions"][:1]
    return convert_surface_mechanism(
        payload,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    ).surface


def test_genuine_reachable_zero_mode_remains_marginal():
    spec, seed = zero_mechanism(), branching_seeds()[0]
    root = solve_catalytic_steady_state(
        spec,
        boundary(pressure=0),
        seed,
        target_species="B",
        product_species=("B", "C"),
        element_basis="H",
    )
    assert root["status"] == "converged"
    assert root["steady_chart"]["reachable_dimension"] == 1
    assert root["eigenvalues_s_inv"] == [{"real": 0.0, "imag": 0.0}]
    assert root["stability"] == "marginal_or_unresolved"
    assert root["coverages"] == seed
    assert root["selectivity"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "rates,products,status",
    [
        ({"A": -2.0, "B": 1.0, "C": 1.0}, ("B", "C"), "available"),
        ({"A": -2.0, "B": 1.0, "C": 1.0}, ("B",), "incomplete"),
        ({"A": -1.0, "B": -1.0, "C": 2.0}, ("B", "C"), "unavailable"),
        ({"A": 0.0, "B": 0.0, "C": 0.0}, ("B", "C"), "unavailable"),
    ],
)
def test_selectivity_preserves_signed_rates_and_explicit_denominator(rates, products, status):
    result = catalytic_product_selectivity(
        branching_mechanism(), rates, product_species=products, element_basis="H"
    )
    assert result["status"] == status
    assert result["signed_gas_rates_mol_s"] == rates
    assert result["denominator_element_mol_s"] == 2 * sum(rates[k] for k in products)
    if status != "available":
        assert result["fractions"] is None


def test_condition_map_retains_bad_point_and_successful_warm_start():
    spec, seeds = branching_mechanism(), branching_seeds()
    result = map_catalytic_conditions(
        spec, [boundary(500), boundary(200), boundary(900)], seeds, target_species="B"
    )
    assert [r["status"] for r in result["points"]] == ["converged", "not_converged", "converged"]
    assert result["points"][1]["attempts"][0]["failure_reason"]
    assert result["points"][2]["seed_count"] == 4


@pytest.mark.parametrize("isothermal", [False, True])
def test_current_world_query_has_no_state_budget_or_clock_effect(isothermal):
    composition = example_world(isothermal=isothermal)
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=34)
        action = runtime.execute({"operation": "wait", "duration_s": 1})
        assert action.info["transaction_status"] == "committed"
        before = deepcopy(runtime.snapshot().to_dict())
        result = analyze_catalytic_world(composition, runtime._state, target_species="B")
        assert result["status"] == "converged", result
        assert result["world_query"]["world_advanced"] is False
        assert result["gas_amounts_mol"] == pytest.approx(runtime._state.species_amounts)
        assert runtime.snapshot().to_dict() == before
        with pytest.raises(ValueError, match="attached"):
            analyze_catalytic_world(
                composition, runtime._state, target_species="B", catalyst_id="nonexistent"
            )
    finally:
        runtime.close()


def test_chart_rejects_bad_permutation_and_overfilled_initial_state():
    spec, seed = branching_mechanism(), branching_seeds()[0]
    with pytest.raises(ValueError, match="permutation"):
        build_catalytic_steady_chart(spec, boundary().capacities_mol, seed, coordinate_order=["E"])
    with pytest.raises(ValueError):
        build_catalytic_steady_chart(spec, boundary().capacities_mol, {**seed, "E": 1.0})
    chart = build_catalytic_steady_chart(spec, boundary().capacities_mol, seed)
    with pytest.raises(ValueError, match="finite independent"):
        chart.reconstruct([np.nan] * chart.rank)


def test_selectivity_rejects_undeclared_basis_or_product():
    spec = branching_mechanism()
    for products, basis in ((("B",), "C"), (("B", "B"), "H"), (("missing",), "H")):
        with pytest.raises(ValueError):
            catalytic_product_selectivity(
                spec, {"A": -1.0, "B": 1.0, "C": 0.0}, product_species=products, element_basis=basis
            )
