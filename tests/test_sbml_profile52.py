"""Profile likelihood consumes original histories and honest fixed-grid inference."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.sbml_calibration import SBMLCalibration
from chemworld.data.sbml_profile import GAUSSIAN_KNOWN_SIGMA, sbml_profile_likelihood
from chemworld.examples.sbml_calibration import fixed_inputs
from chemworld.examples.sbml_profile import A_GRID, affine_inputs
from reference.sbml_profile52_reference import affine_baseline, affine_point, affine_sse, pool_point


def profile(inputs=None, parameter="a", grid=A_GRID, **kwargs):
    return sbml_profile_likelihood(
        SBMLCalibration(*(inputs or affine_inputs())),
        parameter,
        grid,
        noise_model=GAUSSIAN_KNOWN_SIGMA,
        **kwargs,
    )


@pytest.mark.parametrize("scaled,coordinate", [(False, "linear"), (True, "log")])
def test_public_affine_profile_matches_independent_weighted_constrained_optima(scaled, coordinate):
    result = profile(affine_inputs(scaled=scaled, coordinate=coordinate))
    baseline = affine_baseline()
    assert result["baseline"]["objective"] == pytest.approx(baseline["sse"], abs=2e-5)
    for point in result["points"]:
        sse, nuisance = affine_point("a", point["value"])
        assert point["weighted_SSE"] == pytest.approx(sse, abs=2e-5)
        assert point["delta_chi2"] == pytest.approx(sse - baseline["sse"], abs=2e-5)
        assert point["fit"]["estimate"]["b"] == pytest.approx(nuisance, abs=2e-5)
        assert point["fit"]["estimate"]["a"] == point["value"]
        assert point["fit"]["identifiability"]["parameter_count"] == 1
        assert all(len(row) == 1 for row in point["fit"]["identifiability"]["weighted_jacobian"])
    assert {c["side"] for c in result["crossing_brackets"]} == {"left", "right"}
    assert result["confidence_interval"] is None
    assert result["status"] == "complete_discrete_profile"
    json.dumps(result, allow_nan=False)


def test_all_fixed_runs_original_history_without_empty_optimizer_or_tangents():
    result = profile(fixed_inputs(), "k1", (0.5, 0.55, 0.6))
    for point in result["points"]:
        fit = point["fit"]
        assert point["weighted_SSE"] == pytest.approx(
            pool_point("decay", point["value"])[0], abs=2e-3
        )
        assert fit["optimizer"]["nfev"] == 0
        assert fit["jacobian_runs"] == []
        assert fit["resources"]["forward_groups"] == 2
        assert fit["identifiability"]["status"] == "no_free_parameters"
        assert all(run["stage"] == "fixed_evaluation" for run in fit["runs"])


def test_validation_poison_cannot_change_profile_or_resources():
    table, parameters, experiments = affine_inputs()
    poisoned = replace(
        table,
        rows=tuple(
            replace(
                row,
                values={
                    name: replace(value, value=value.value + 100, uncertainty=0.0001)
                    if name != "time"
                    else value
                    for name, value in row.values.items()
                },
            )
            if row.split == "validation"
            else row
            for row in table.rows
        ),
    )
    clean, poison = (
        profile((table, parameters, experiments)),
        profile((poisoned, parameters, experiments)),
    )
    assert clean["resources"] == poison["resources"]
    assert clean["crossing_brackets"] == poison["crossing_brackets"]
    for a, b in zip(clean["points"], poison["points"], strict=True):
        assert a["delta_chi2"] == b["delta_chi2"]
        assert a["fit"]["estimate"] == b["fit"]["estimate"]
        assert all(run["split"] == "train" for run in b["fit"]["runs"])
        assert all(
            group["group_id"] == "train"
            for run in b["fit"]["jacobian_runs"]
            for group in run["groups"]
        )


def test_nonconverged_baseline_retains_real_sse_and_negative_delta():
    result = profile(grid=(1.0,), max_nfev=1)
    assert result["baseline"]["status"] == "optimizer_not_converged"
    assert result["baseline"]["objective"] == pytest.approx(affine_sse(0.3, 0.2))
    point = result["points"][0]
    assert point["weighted_SSE"] == pytest.approx(affine_sse(1, 0.2))
    assert point["delta_chi2"] < 0
    assert any("improves_baseline" in reason for reason in result["limitations"])
    assert result["status"] == "limited_discrete_profile"


def test_budget_keeps_all_planned_points_and_no_unexecuted_objective():
    result = profile(max_total_forward_groups=1)
    assert result["resources"]["forward_groups"] == 1
    assert result["counts"] == {"planned": 5, "attempted": 0, "converged": 0, "not_attempted": 5}
    assert all(point["weighted_SSE"] is None for point in result["points"])
    assert result["baseline"]["failures"]


def test_rank_deficient_flat_grid_and_missing_crossings_are_not_confidence_limits():
    result = profile(fixed_inputs(kind="confounded"), "k1", (0.25, 0.45, 0.65))
    assert "baseline_rank_deficient" in result["limitations"]
    assert "profile_flat_on_declared_grid" in result["limitations"]
    assert "threshold_not_bracketed_on_both_sides" in result["limitations"]
    assert result["confidence_interval"] is None


def test_fixed_fit_projects_analytical_derivative_and_preserves_old_default_validation():
    inputs = affine_inputs()
    fixed = SBMLCalibration(*inputs).fit(fixed_parameters={"a": 1}, evaluate_validation=False)
    jac = np.asarray(fixed["identifiability"]["weighted_jacobian"])
    np.testing.assert_allclose(
        jac[:, 0],
        np.array([0, 0.3, 0.8, 1.4, 2.2]) * 1.99 / np.array([0.04, 0.06, 0.05, 0.08, 0.05]),
    )
    assert "validation" not in fixed
    default = SBMLCalibration(*inputs).fit()
    assert default["validation"]["n_valid"] == 5
    assert any(run["stage"] == "heldout" for run in default["runs"])


@pytest.mark.parametrize("fixed", [{"unknown": 1}, {"a": True}, {"a": float("nan")}, {"a": 4}, []])
def test_invalid_fixed_values_do_not_consume_blueprint(fixed):
    calibration = SBMLCalibration(*affine_inputs())
    with pytest.raises(ValueError):
        calibration.fit(fixed_parameters=fixed)
    assert not calibration._fit_started
    assert not calibration.runs


@pytest.mark.parametrize(
    "grid", [(), (1, 1), (1, 0.5), (float("nan"),), (float("inf"),), (True,), (0,), (4,), None]
)
def test_invalid_grids_are_refused_before_any_execution(grid):
    blueprint = SBMLCalibration(*affine_inputs())
    with pytest.raises(ValueError):
        sbml_profile_likelihood(blueprint, "a", grid, noise_model=GAUSSIAN_KNOWN_SIGMA)
    assert not blueprint.runs


@pytest.mark.parametrize(
    "options",
    [
        {"noise_model": "unknown_sigma"},
        {"delta_chi2_threshold": 0},
        {"delta_chi2_threshold": True},
        {"max_nfev": True},
        {"max_total_forward_groups": 0},
    ],
)
def test_invalid_assumptions_and_budgets(options):
    kwargs = {"noise_model": GAUSSIAN_KNOWN_SIGMA, **options}
    with pytest.raises(ValueError):
        sbml_profile_likelihood(SBMLCalibration(*affine_inputs()), "a", A_GRID, **kwargs)


def test_used_blueprint_is_refused_and_fresh_blueprint_remains_unmodified():
    blueprint = SBMLCalibration(*affine_inputs())
    before = [experiment.composition.to_dict() for experiment in blueprint.experiments]
    sbml_profile_likelihood(blueprint, "a", (1,), noise_model=GAUSSIAN_KNOWN_SIGMA)
    assert not blueprint.runs and not blueprint.jacobian_runs and not blueprint._fit_started
    assert [experiment.composition.to_dict() for experiment in blueprint.experiments] == before
    blueprint.fit(max_nfev=1)
    with pytest.raises(ValueError, match="unused"):
        sbml_profile_likelihood(blueprint, "a", (1,), noise_model=GAUSSIAN_KNOWN_SIGMA)
