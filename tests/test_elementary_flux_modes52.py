"""Exact original-coordinate modes, current native controls and honest limits."""

import json
from fractions import Fraction

import pytest

from chemworld.examples.sbml_flux_modes import flux_model, run_example
from chemworld.physchem.elementary_flux_modes import (
    FluxModePolicy,
    enumerate_elementary_flux_modes,
    sbml_elementary_flux_modes,
)
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from reference.elementary_flux_modes52_cases import STRUCTURAL, source_scope_model
from reference.elementary_flux_modes52_reference import exchanges, minimal_balanced


def enumerate_case(case, **kwargs):
    return enumerate_elementary_flux_modes(
        case["matrix"],
        reaction_ids=tuple(f"r{i}" for i in range(len(case["irreversible"]))),
        irreversible=case["irreversible"],
        internal_species_ids=tuple(f"I{i}" for i in range(len(case["matrix"]))),
        external_matrix=case.get("external", ()),
        external_species_ids=tuple(f"E{i}" for i in range(len(case.get("external", ())))),
        **kwargs,
    )


@pytest.mark.parametrize("case", STRUCTURAL, ids=lambda c: c["name"])
def test_exact_complete_modes_match_independent_expected_supports(case):
    result = enumerate_case(case)
    actual = sorted(
        (tuple(int(v) for v in m["vector"]), m["bidirectional"]) for m in result["modes"]
    )
    expected = sorted(zip(case["vectors"], case["bidirectional"], strict=True))
    assert actual == expected
    assert result["complete"] and result["status"] == "complete"
    assert result["rank"] == case["rank"]
    assert result["lineality_dimension"] == case["lineality"]
    assert result["evaluated_supports"] == result["total_supports"] == case["total"]
    assert result["extreme_rays_computed"] is False
    for mode in result["modes"]:
        assert minimal_balanced(case["matrix"], mode["vector"])
        assert list(mode["external_exchange"].values()) == exchanges(
            case.get("external", ()), mode["vector"]
        )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", STRUCTURAL, ids=lambda c: c["name"])
def test_native_source_units_preserve_modes_and_expose_external_exchange(case):
    model = flux_model(
        case["matrix"],
        irreversible=case["irreversible"],
        external_matrix=case.get("external", ()),
        scaled=case.get("scaled", False),
    )
    frame = SBMLKinetics(model, controlled_parameters=("cf",))
    result = sbml_elementary_flux_modes(frame, frame.initial_values)
    assert sorted(tuple(int(v) for v in m["vector"]) for m in result["modes"]) == sorted(
        case["vectors"]
    )
    assert result["chemical_composition_scope"] == "unknown_formal_entities"
    assert all(
        abs(v) <= 1e-12
        for row in result["native_normalized_mode_balance"].values()
        for v in row.values()
    )
    if case.get("scaled"):
        assert Fraction(result["current_row_factors"]["I0"]) == Fraction(0.001)


def test_actual_time_zero_and_later_conversion_controls_use_sole_native_state():
    result = run_example()
    assert all(r["transaction_status"] == "committed" for r in result["operations"])
    assert len(result["before"]["modes"]) == 2
    assert len(result["after"]["modes"]) == 3
    assert result["after"]["current_controls"] == {"cf": 0.0}
    assert result["before"]["time_s"] == result["after"]["time_s"] == 1.0
    assert result["snapshots_unchanged_by_analysis"]


def test_zero_current_kinetic_rates_do_not_disable_structural_columns():
    frame = SBMLKinetics(flux_model(zero_rates=True))
    result = sbml_elementary_flux_modes(frame, frame.initial_values)
    assert not any(result["current_reaction_rates_mol_s"].values())
    assert len(result["modes"]) == 2


@pytest.mark.parametrize("budget", [0, 1, 3, 5])
def test_budget_exhaustion_preserves_partial_modes_without_a_no_mode_claim(budget):
    result = enumerate_case(STRUCTURAL[2], policy=FluxModePolicy(budget))
    assert result["status"] == "incomplete" and not result["complete"]
    assert result["evaluated_supports"] == budget
    assert not result["no_modes_proven"]
    assert result["total_supports"] == 6
    assert result["stop_reason"] == "support_evaluation_budget_exhausted"
    if budget == 5:
        assert len(result["modes"]) == 2


def test_arithmetic_budget_is_incomplete_and_exact_float_near_rank_is_never_clipped():
    result = enumerate_elementary_flux_modes(
        ((1, 2),),
        reaction_ids=("a", "b"),
        irreversible=(False, False),
        internal_species_ids=("X",),
        policy=FluxModePolicy(max_integer_bits=1),
    )
    assert result["status"] == "incomplete" and result["rank"] is None
    assert "arithmetic size budget" in result["stop_reason"]
    assert result["rank_evaluations"] == 1
    assert not result["no_modes_proven"]
    assert enumerate_case(STRUCTURAL[-1])["no_modes_proven"]


def test_no_internal_rows_retains_independent_exchange_columns_and_direction():
    result = enumerate_elementary_flux_modes(
        (),
        reaction_ids=("in", "out"),
        irreversible=(True, False),
        internal_species_ids=(),
        external_matrix=((-1, 2),),
        external_species_ids=("external",),
    )
    assert [m["vector"] for m in result["modes"]] == [["1", "0"], ["0", "1"]]
    assert [m["bidirectional"] for m in result["modes"]] == [False, True]
    assert [m["external_exchange"]["external"] for m in result["modes"]] == ["-1", "2"]
    assert not any(m["internal_cycle"] for m in result["modes"])


@pytest.mark.parametrize("defect", ["rate_rule", "assigned_species", "event", "time_dependent"])
def test_scope_rejection_is_at_analysis_not_invalid_source_parsing(defect):
    frame = SBMLKinetics(source_scope_model(defect))
    with pytest.raises(ValueError):
        sbml_elementary_flux_modes(frame, frame.initial_values)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1/3"])
def test_invalid_exact_matrix_inputs_are_rejected(value):
    with pytest.raises(ValueError):
        enumerate_elementary_flux_modes(
            ((value,),), reaction_ids=("r",), irreversible=(True,), internal_species_ids=("X",)
        )


def test_fraction_inputs_do_not_go_through_float_and_inputs_are_read_only():
    matrix = [[Fraction(1, 3), Fraction(-1, 7)]]
    before = [row[:] for row in matrix]
    result = enumerate_elementary_flux_modes(
        matrix, reaction_ids=("a", "b"), irreversible=(True, True), internal_species_ids=("X",)
    )
    assert result["modes"][0]["vector"] == ["3", "7"]
    assert result["internal_matrix"] == [["1/3", "-1/7"]]
    assert matrix == before
