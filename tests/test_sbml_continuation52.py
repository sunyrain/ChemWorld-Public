"""Branch traversal, exact current-state ownership and explicit unsupported boundaries."""

import json
from dataclasses import replace
from itertools import pairwise
from math import sqrt

import numpy as np
import pytest

from chemworld.data.sbml import SBMLModel
from chemworld.examples.sbml_continuation import example_report
from chemworld.physchem.residual_system import RootPolicy
from chemworld.physchem.sbml_continuation import sbml_steady_branch
from reference.sbml_continuation52_cases import CASES, build
from reference.sbml_continuation52_reference import reference_checks


def analyze(case=None, **overrides):
    spec, values, options = build(case or {"name": "scalar"})
    return sbml_steady_branch(spec, values, start_time_s=0, **{**options, **overrides})


@pytest.mark.parametrize("case", (CASES[0], CASES[5], CASES[8], CASES[10]))
def test_independent_branch_spectrum_and_conservation(case):
    result = analyze(case)
    checks = reference_checks(
        result, closed=case.get("closed", False), scaled=case.get("scaled", False)
    )
    assert all(c["passed"] for c in checks), checks


def test_fold_tangent_turns_without_reversing_the_branch_orientation():
    result = analyze()
    rows = result["records"]
    assert len(result["fold_brackets"]) == 2
    assert any(r["stability"] == "unstable" for r in rows)
    assert all(np.dot(a["tangent"], b["tangent"]) > 0 for a, b in pairwise(rows))
    assert all(abs(np.linalg.norm(r["tangent"]) - 1) < 1e-12 for r in rows)
    assert all(r["tangent_residual"] < 1e-10 for r in rows)


def test_original_other_control_and_boundary_pool_are_held():
    spec, values, options = build({"name": "held", "coupled": True})
    values.update({"P:k": 0.8, "N:H": 1.5})
    before = dict(values)
    report = sbml_steady_branch(spec, values, start_time_s=1.2, **options)
    assert report["status"] == "complete"
    assert values == before
    for row in report["records"]:
        assert row["coordinates"]["P:k"] == 0.8
        assert row["coordinates"]["N:H"] == 1.5
        assert row["coordinates"]["N:Y"] == pytest.approx(2 * row["coordinates"]["N:X"])


def test_resolved_parameter_sign_is_retained_across_a_fold_point():
    spec, values, options = build(CASES[0])
    values.update({"N:X": 0.5, "P:mu": 0.625})
    x_fold = 1 - sqrt(1.5) / 3
    p_fold = x_fold**3 - 3 * x_fold**2 + 2.5 * x_fold
    # Analytic projection of the fold onto the initial tangent hyperplane.
    step = ((x_fold - 0.5) + 0.25 * (p_fold - 0.625)) / sqrt(1 + 0.25**2)
    report = sbml_steady_branch(
        spec,
        values,
        start_time_s=0,
        **{
            **options,
            "step_size": step,
            "maximum_points": 3,
            "root_policy": RootPolicy(tolerance=1e-15),
        },
    )
    assert report["status"] == "complete"
    assert report["records"][1]["coordinates"]["N:X"] == pytest.approx(x_fold, abs=1e-12)
    assert report["fold_brackets"][0]["indices"] == [0, 2]


@pytest.mark.parametrize(
    "field,value",
    (
        ("amount_scales_mol", {"X": 0}),
        ("amount_scales_mol", {"H": 1}),
        ("parameter_scale", 0),
        ("parameter_bounds", [1, 0]),
        ("parameter_bounds", [0.5, 1]),
        ("minimum_step", 1),
        ("initial_direction", True),
        ("maximum_points", 0),
        ("maximum_evaluations", 0),
        ("minimum_tangent_alignment", 1),
        ("root_policy", False),
    ),
)
def test_invalid_metrics_domains_and_budgets_raise(field, value):
    with pytest.raises(ValueError):
        analyze(**{field: value})


def test_complete_state_and_positive_interior_are_required():
    spec, values, options = build(CASES[0])
    del values["N:H"]
    with pytest.raises(ValueError, match="complete"):
        sbml_steady_branch(spec, values, start_time_s=0, **options)
    values = dict(spec.frame.initial_values)
    values["N:X"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        sbml_steady_branch(spec, values, start_time_s=0, **options)


def test_parameter_may_not_change_conversion_factor():
    spec, values, options = build(CASES[0])
    source = SBMLModel(
        spec.model.raw_text.replace(
            '<model id="fold_network"', '<model conversionFactor="mu" id="fold_network"'
        ),
        spec.model.source,
    )
    spec = replace(spec, model=source)
    with pytest.raises(ValueError, match="conversionFactor"):
        sbml_steady_branch(spec, values, start_time_s=0, **options)


def test_exact_rank_cannot_be_silently_truncated_in_bad_metric():
    with pytest.raises(ValueError, match="unresolved"):
        analyze(CASES[4], amount_scales_mol={"X": 1e-30, "Y": 1e30})


def test_residual_form_is_not_silently_treated_as_explicit_ode():
    spec, values, options = build(CASES[0])
    spec = replace(spec, numerical_form="residual")
    with pytest.raises(ValueError, match="explicit"):
        sbml_steady_branch(spec, values, start_time_s=0, **options)


@pytest.mark.parametrize("case", CASES[12:16])
def test_partial_and_singular_results_retain_failed_attempts(case):
    report = analyze(case)
    assert report["status"] in {"partial", "failed"}
    assert report["stop_reason"]
    assert report["rejected_attempts"] > 0
    assert report["attempts"][-1]["reason"]
    assert report["completed_points"] < report["requested_points"]
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    if case["failure"] == "evaluation":
        assert report["costs"]["variational_rate_calls"] == 2
        assert report["completed_points"] == 1


def test_true_world_control_wait_and_complete_json_query_are_read_only():
    result = example_report()
    assert result["world_unchanged"] and result["json_replay_exact"]
    assert result["transactions"] == ["committed", "committed"]
    assert result["analysis"]["initial_coordinates"]["P:k"] == 0.8
    assert result["analysis"]["start_time_s"] == 0.1
