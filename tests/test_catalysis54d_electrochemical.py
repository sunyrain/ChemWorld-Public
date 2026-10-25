"""Electrochemical steady maps and controls use the original declared rates."""

import pytest

from chemworld.examples.chemostatted_catalysis import example_boundary, example_mechanism
from chemworld.physchem import ChemostattedAnalysisConditions
from chemworld.physchem.catalysis_analysis import (
    chemostatted_rate_control,
    map_chemostatted_catalytic_conditions,
    solve_chemostatted_catalytic_steady_state,
)
from reference.chemostatted_catalysis54_reference import equilibrium_potential


def boundary_conditions(*, pH=2.0, potential=-0.2, temperature=298.15):
    return ChemostattedAnalysisConditions(
        example_boundary(pH=pH, temperature=temperature),
        potential,
        {"hydrogen": 101325.0},
        {"metal": 1e-6},
    )


def test_open_steady_state_exposes_current_affinity_and_real_liquid_rates():
    spec = example_mechanism("her")
    result = solve_chemostatted_catalytic_steady_state(
        spec, boundary_conditions(), {"H_ads": 0.1}, target_species="hydrogen"
    )
    assert result["status"] == "converged"
    assert 0 < result["coverages"]["H_ads"] < 1
    assert result["turnover_rate_s_inv"] > 0
    assert result["faradaic_current_A"] == pytest.approx(
        result["electrical_readout"]["applied_current_A"], rel=1e-10
    )
    assert set(result["liquid_rates_mol_s"]) == {"proton", "chloride"}
    assert set(result["electrical_readout"]["affinities_J_mol"]) == {
        "electron-proton-capture",
        "hydrogen-recombination",
    }
    high_coverage = solve_chemostatted_catalytic_steady_state(
        spec, boundary_conditions(potential=-0.4), {"H_ads": 0.9}, target_species="hydrogen"
    )
    assert high_coverage["status"] == "converged"
    assert 0 < high_coverage["coverages"]["H_ads"] < 1


def test_open_volmer_root_matches_independent_nernst_potential():
    potential = equilibrium_potential("volmer", 298.15, 2.0, theta=0.5)
    result = solve_chemostatted_catalytic_steady_state(
        example_mechanism("volmer"),
        boundary_conditions(potential=potential),
        {"H_ads": 0.5},
        target_species="hydrogen",
    )
    assert result["status"] == "converged"
    assert result["coverages"]["H_ads"] == pytest.approx(0.5, abs=1e-7)
    assert result["faradaic_current_A"] == pytest.approx(0.0, abs=1e-12)
    assert result["electrical_readout"]["affinities_J_mol"][
        "electron-proton-capture"
    ] == pytest.approx(0.0, abs=1e-7)


def test_condition_map_keeps_out_of_domain_potential_as_a_visible_failure():
    spec = example_mechanism("her")
    result = map_chemostatted_catalytic_conditions(
        spec,
        [
            boundary_conditions(pH=2),
            boundary_conditions(pH=3),
            boundary_conditions(pH=3, potential=0.2),
        ],
        [{"H_ads": 0.1}],
        target_species="hydrogen",
    )
    assert result["points_total"] == 3
    assert result["points_converged"] == 2
    assert [row["status"] for row in result["points"]] == [
        "converged",
        "converged",
        "rejected",
    ]
    assert result["points"][1]["conditions"]["pH"] == pytest.approx(3.0)
    assert result["points"][2]["roots"] == []


def test_electrochemical_rate_and_selectivity_controls_require_explicit_ph_pair():
    spec = example_mechanism("her")
    low = boundary_conditions(pH=2)
    high = boundary_conditions(pH=3)
    result = chemostatted_rate_control(
        spec,
        low,
        {"H_ads": 0.1},
        target_species="hydrogen",
        pH_pair=(low, high),
        product_species=("hydrogen",),
        element_basis="H",
    )
    assert result["status"] == "available"
    assert result["potential_sensitivity"]["unit"] == "V^-1"
    assert result["potential_sensitivity"]["status"] == "available"
    assert result["pH_sensitivity"]["status"] == "available"
    assert result["selectivity_control"]["pH"]["status"] == "available"

    unavailable = chemostatted_rate_control(
        spec, low, {"H_ads": 0.1}, target_species="hydrogen"
    )
    assert unavailable["pH_sensitivity"]["status"] == "unavailable"
    assert "explicit electroneutral boundary pair" in unavailable["pH_sensitivity"][
        "failure_reason"
    ]
