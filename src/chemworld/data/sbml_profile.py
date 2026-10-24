"""Discrete known-sigma Gaussian profiles of original public SBML histories."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from math import isfinite
from numbers import Real
from typing import Any

from chemworld.data.sbml_calibration import SBMLCalibration

GAUSSIAN_KNOWN_SIGMA = "independent_gaussian_known_sigma"


def sbml_profile_likelihood(
    calibration: SBMLCalibration,
    parameter_id: str,
    grid: Any,
    *,
    noise_model: str,
    delta_chi2_threshold: float = 3.841458820694124,
    max_nfev: int = 30,
    max_total_forward_groups: int = 2000,
    progress: Any = None,
) -> dict[str, Any]:
    """Fit a baseline then reoptimize nuisance parameters at every fixed grid value.

    ``noise_model`` explicitly declares independent Gaussian errors with known
    standard deviations from the training table. The threshold is user supplied
    (default: the usual asymptotic one-parameter 95% chi-square threshold); this
    routine makes no finite-sample coverage claim. It returns discrete crossing
    brackets, never an interpolated confidence interval. All optimization uses
    the original bounded fitter, public histories and analytic tangent columns.
    """
    if not isinstance(calibration, SBMLCalibration):
        raise ValueError("profile requires an original SBMLCalibration blueprint")
    if calibration.runs or calibration.jacobian_runs or calibration._fit_started:
        raise ValueError("profile requires a fresh unused calibration blueprint")
    parameters = {p.parameter_id: p for p in calibration.parameters}
    if parameter_id not in parameters:
        raise ValueError("unknown profile parameter")
    if noise_model != GAUSSIAN_KNOWN_SIGMA:
        raise ValueError(
            "only explicitly declared independent Gaussian known-sigma errors supported"
        )
    if (
        isinstance(delta_chi2_threshold, bool)
        or not isinstance(delta_chi2_threshold, (int, float))
        or not isfinite(delta_chi2_threshold)
        or delta_chi2_threshold <= 0
    ):
        raise ValueError("delta chi-square threshold must be finite and positive")
    for value in (max_nfev, max_total_forward_groups):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "profile optimizer and total forward budgets must be positive integers"
            )
    try:
        values = tuple(grid)
    except TypeError as error:
        raise ValueError("profile grid must be a fixed numeric sequence") from error
    if not values or any(
        isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in values
    ):
        raise ValueError("profile grid must be nonempty and finite numeric values")
    if any(a >= b for a, b in pairwise(values)):
        raise ValueError("profile grid must be strictly increasing without duplicates")
    for value in values:
        parameters[parameter_id].encode(value)
    values = tuple(float(v) for v in values)
    report: dict[str, Any] = {
        "schema_version": "chemworld-sbml-profile-likelihood-1",
        "noise_model": noise_model,
        "objective": "sum(((public_prediction - observation) / known_sigma)**2)",
        "delta_definition": "point_weighted_SSE - baseline_weighted_SSE; no clipping or rescaling",
        "optimization_split": "train",
        "validation_evaluation": "not_requested",
        "parameter_id": parameter_id,
        "parameter_unit": parameters[parameter_id].column.unit,
        "grid": list(values),
        "delta_chi2_threshold": float(delta_chi2_threshold),
        "max_total_forward_groups": max_total_forward_groups,
        "points": [],
        "resources": {},
        "failures": [],
        "limitations": [],
        "confidence_interval": None,
        "boundary": (
            "discrete local nuisance optimization; crossing brackets are not exact confidence "
            "limits, global minima, or finite-sample coverage qualification"
        ),
    }

    def fit_one(label: str, fitted_parameters: Any, fixed: Any = None) -> dict[str, Any] | None:
        used = report["resources"].get("forward_groups", 0)
        remaining = max_total_forward_groups - used
        if remaining <= 0:
            return None

        def forward_progress(record: dict[str, Any]) -> None:
            if progress is not None:
                progress(
                    {
                        **record,
                        "profile_stage": label,
                        "total_forward_groups": used + record["completed_forward_groups"],
                        "total_forward_limit": max_total_forward_groups,
                    }
                )

        fresh = SBMLCalibration(
            calibration.table,
            fitted_parameters,
            calibration.experiments,
            max_forward_groups=min(calibration.max_forward_groups, remaining),
            progress=forward_progress,
        )
        fitted = fresh.fit(
            max_nfev=max_nfev,
            fixed_parameters=fixed,
            evaluate_validation=False,
        )
        for key, value in fitted["resources"].items():
            report["resources"][key] = report["resources"].get(key, 0) + value
        if fitted["status"] != "fitted":
            report["failures"].append(
                {"stage": label, "status": fitted["status"], "failures": fitted["failures"]}
            )
        return fitted

    baseline = fit_one("baseline", calibration.parameters)
    report["baseline"] = baseline
    # A positive total budget guarantees an attempted baseline. Keeping this
    # branch explicit also prevents treating a missing estimate as an optimum.
    baseline_values = baseline.get("estimate") if baseline else None
    baseline_sse = baseline.get("objective") if baseline else None
    for index, value in enumerate(values):
        point: dict[str, Any] = {
            "index": index,
            "value": value,
            "status": "not_attempted",
            "weighted_SSE": None,
            "delta_chi2": None,
            "fit": None,
            "at_parameter_bound": value in parameters[parameter_id].bounds,
        }
        report["points"].append(point)
        if baseline_values is None:
            point["reason"] = "baseline_has_no_estimate"
            continue
        starts = tuple(
            replace(p, initial=baseline_values[p.parameter_id]) for p in calibration.parameters
        )
        fitted = fit_one(f"point_{index}", starts, {parameter_id: value})
        if fitted is None:
            point["reason"] = "total_forward_budget_exhausted"
            continue
        point.update(status=fitted["status"], fit=fitted, weighted_SSE=fitted.get("objective"))
        if point["weighted_SSE"] is not None and baseline_sse is not None:
            point["delta_chi2"] = point["weighted_SSE"] - baseline_sse
        if progress is not None:
            progress(
                {
                    "profile_stage": "point_complete",
                    "completed_points": index + 1,
                    "planned_points": len(values),
                    "status": point["status"],
                    "total_forward_groups": report["resources"].get("forward_groups", 0),
                }
            )

    points = report["points"]
    converged = [p for p in points if p["status"] == "fitted"]
    attempted = [p for p in points if p["fit"] is not None]
    report["counts"] = {
        "planned": len(points),
        "attempted": len(attempted),
        "converged": len(converged),
        "not_attempted": len(points) - len(attempted),
    }
    issues = report["limitations"]
    if not baseline or baseline["status"] != "fitted":
        issues.append("baseline_not_converged")
    if any(p["delta_chi2"] is not None and p["delta_chi2"] < 0 for p in points):
        issues.append("profile_improves_baseline; baseline_is_locally_inadequate")
    if len(converged) != len(points):
        issues.append("not_all_nuisance_points_converged")
    if baseline and baseline.get("identifiability", {}).get("status") == "rank_deficient":
        issues.append("baseline_rank_deficient")
    if any(p["at_parameter_bound"] for p in points):
        issues.append("grid_reaches_parameter_bound")
    if baseline and any(baseline.get("active_bounds", {}).values()):
        issues.append("baseline_has_active_bounds")
    if any(any(p["fit"].get("active_bounds", {}).values()) for p in attempted):
        issues.append("nuisance_has_active_bounds")
    deltas = [p["delta_chi2"] for p in converged if p["delta_chi2"] is not None]
    if len(deltas) > 1 and max(deltas) - min(deltas) <= 1e-6:
        issues.append("profile_flat_on_declared_grid")
    crossings: list[dict[str, Any]] = []
    center = baseline_values[parameter_id] if baseline_values else None
    for left, right in pairwise(points):
        if left["status"] != "fitted" or right["status"] != "fitted":
            continue
        a, b = left["delta_chi2"], right["delta_chi2"]
        if a is None or b is None or a == b:
            continue
        threshold = delta_chi2_threshold
        if min(a, b) <= threshold <= max(a, b):
            side = (
                "left"
                if a > b and left["value"] < center
                else ("right" if b > a and right["value"] > center else "other")
            )
            crossings.append(
                {"bracket": [left["value"], right["value"]], "delta_chi2": [a, b], "side": side}
            )
    report["crossing_brackets"] = crossings
    if not {"left", "right"}.issubset({crossing["side"] for crossing in crossings}):
        issues.append("threshold_not_bracketed_on_both_sides")
    report["status"] = "complete_discrete_profile" if not issues else "limited_discrete_profile"
    json.dumps(report, allow_nan=False)
    return report


__all__ = ["GAUSSIAN_KNOWN_SIGMA", "sbml_profile_likelihood"]
