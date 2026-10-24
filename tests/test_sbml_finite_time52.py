"""Finite-time perturbation metrics and original-world ownership contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.examples.sbml_calibration import pool_model
from chemworld.physchem.sbml_finite_time import sbml_finite_time_stability
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_finite_time import analyze_sbml_finite_time_world
from chemworld.world.sbml_world import sbml_world
from reference.sbml_finite_time52_cases import model
from reference.sbml_finite_time52_reference import reference


def analyze(kind="closed", *, scaled=False, offsets=(0.2, 0.5), scales=None, **options):
    spec = SBMLSystemSpec(model(kind, scaled=scaled))
    names = tuple(spec.frame._species)
    return sbml_finite_time_stability(
        spec,
        spec.frame.initial_values,
        start_time_s=0,
        sample_offsets_s=offsets,
        amount_scales_mol=scales or dict.fromkeys(names, 1.0),
        **options,
    )


@pytest.mark.parametrize("kind", ("closed", "nonnormal", "growth", "nonlinear"))
def test_independent_finite_time_growth_and_original_amounts(kind):
    result = analyze(kind)
    assert result["status"] == "complete"
    for row in result["records"]:
        expected = reference(kind, row["offset_s"], result["amount_scales_mol"])
        assert row["singular_growth_factors"] == pytest.approx(expected["growth"], rel=2e-7)
        assert [row["coordinates"]["N:" + s] for s in result["species_ids"]] == pytest.approx(
            expected["amounts"], rel=2e-7
        )
    if kind in {"closed", "nonlinear"}:
        assert result["reachable_dimension"] == 1
        assert result["records"][-1]["conserved_zero_modes_excluded"] == 1


def test_stable_non_normal_system_has_true_transient_amplification():
    row = analyze("nonnormal")["records"][0]
    assert row["maximum_amplification"] > 1
    assert row["finite_time_rates_s_inverse"][0] > 0
    assert not np.allclose(row["finite_time_rates_s_inverse"], [-0.5, -1.2])


def test_explicit_metric_changes_non_normal_growth_but_source_units_do_not():
    equal = analyze("nonnormal")
    weighted = analyze("nonnormal", scales={"X": 0.2, "Y": 2.0})
    scaled = analyze("nonnormal", scaled=True)
    assert equal["records"][0]["maximum_amplification"] != pytest.approx(
        weighted["records"][0]["maximum_amplification"]
    )
    assert equal["records"][0]["singular_growth_factors"] == pytest.approx(
        scaled["records"][0]["singular_growth_factors"], rel=2e-7
    )


@pytest.mark.parametrize(
    "options",
    (
        {"offsets": (0,)},
        {"offsets": (1, 0.5)},
        {"offsets": ()},
        {"scales": {"A": 1}},
        {"scales": {"A": 1, "B": 0}},
        {"maximum_directions": 0},
        {"max_integer_bits": 0},
    ),
)
def test_bad_time_metric_and_budget_are_rejected(options):
    with pytest.raises(ValueError):
        analyze(**options)


def test_boundary_initial_conditions_are_not_filled_with_small_inventory():
    spec = SBMLSystemSpec(pool_model())
    with pytest.raises(ValueError, match="strictly positive"):
        sbml_finite_time_stability(
            spec,
            spec.frame.initial_values,
            start_time_s=0,
            sample_offsets_s=(1,),
            amount_scales_mol={"A": 1, "B": 1},
        )


def test_unresolved_small_singular_values_have_no_clipped_logarithm():
    result = analyze("birth", offsets=(120,))
    assert result["status"] == "complete"
    row = result["records"][0]
    assert row["resolved_singular_values"] == [False]
    assert row["finite_time_rates_s_inverse"] == [None]
    assert row["maximum_amplification"] is None
    assert row["singular_growth_factors"][0] < row["singular_resolution_estimate"]


def test_world_after_actual_control_wait_and_json_is_read_only():
    composition = sbml_world(
        pool_model(kind="reversible", initial_b=0.2), controlled_parameters=("k1",)
    )
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=52)
        for action in (
            {"operation": "set_model_parameters", "values": {"k1": 0.7}},
            {"operation": "wait", "duration_s": 0.1},
        ):
            assert runtime.execute(action).info["transaction_status"] == "committed"
        before = runtime._state.to_dict()
        options = {"sample_offsets_s": (0.2, 0.5), "amount_scales_mol": {"A": 1, "B": 1}}
        result = analyze_sbml_finite_time_world(runtime._state, composition, **options)
        assert result["initial_coordinates"]["P:k1"] == 0.7
        assert result["start_time_s"] == 0.1
        assert result["records"][0]["finite_time_rates_s_inverse"] == pytest.approx(
            [-0.9], rel=2e-7
        )
        assert runtime._state.to_dict() == before
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        assert analyze_sbml_finite_time_world(restored._state, composition, **options) == result
        assert json.loads(json.dumps(result, allow_nan=False)) == result
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_residual_model_is_not_silently_treated_as_explicit_ode():
    spec = replace(SBMLSystemSpec(model("closed")), numerical_form="residual")
    with pytest.raises(ValueError, match="explicit"):
        sbml_finite_time_stability(
            spec,
            spec.frame.initial_values,
            start_time_s=0,
            sample_offsets_s=(1,),
            amount_scales_mol={"A": 1, "B": 1},
        )
