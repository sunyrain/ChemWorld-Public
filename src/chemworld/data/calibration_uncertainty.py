"""Optional known-sigma covariance from the original bounded fitter's Jacobian.

This postprocessor does not refit, rescale known errors by residual SSE, or alter
an existing report. Fixed inputs and independent known Gaussian response errors
are explicit assumptions, not facts inferred from a small training residual.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chemworld.foundation.units import unit_spec

KNOWN_INDEPENDENT_SIGMA = "independent_gaussian_known_sigma"


def known_sigma_parameter_uncertainty(
    calibration: Any,
    report: Mapping[str, Any],
    *,
    noise_model: str,
    parameter_units: Mapping[str, str],
) -> dict[str, Any]:
    """Return local, conditional covariance for identifiable interior parameters.

    The covariance uses physical parameter coordinates. Rank-deficient, active-bound
    and failed solutions return an explicit unavailable status, never a pseudoinverse
    masquerading as finite uncertainty. No confidence-coverage claim is made.
    """
    if noise_model != KNOWN_INDEPENDENT_SIGMA:
        raise ValueError("parameter uncertainty requires independent Gaussian known sigma")
    parameters = tuple(calibration.parameters)
    names = [p.parameter_id for p in parameters]
    if set(parameter_units) != set(names):
        raise ValueError("parameter units must cover the original parameter whitelist exactly")
    for unit in parameter_units.values():
        unit_spec(unit)
    if report.get("parameters") != [p.to_dict() for p in parameters]:
        raise ValueError("uncertainty report does not belong to these parameter declarations")
    fixed = dict(report.get("fixed_parameters", {}))
    if set(fixed) - set(names):
        raise ValueError("unknown fixed parameter in calibration report")
    free = [p for p in parameters if p.parameter_id not in fixed]
    order = [p.parameter_id for p in free]
    result: dict[str, Any] = {
        "noise_model": noise_model,
        "parameter_order": order,
        "parameter_units": {name: parameter_units[name] for name in order},
        "fixed_parameters": fixed,
        "fixed_parameter_meaning": "conditioned values, not data-estimated zero uncertainty",
        "covariance": None,
        "standard_errors": None,
        "residual_variance_rescaling": False,
        "status": "fit_not_converged",
        "boundary": (
            "Local linearized covariance conditional on exact predictors and fixed parameters; "
            "independent known Gaussian response sigma. No global identification, "
            "confidence-coverage or material-accuracy claim."
        ),
    }
    if report.get("status") != "fitted" or not report.get("optimizer", {}).get("success"):
        return result
    if not free:
        result["status"] = "no_free_parameters"
        return result
    estimate = report["estimate"]
    coordinates = [p.encode(estimate[p.parameter_id]) for p in free]
    if any(report.get("active_bounds", {}).get(name, 0) for name in order) or any(
        x <= 0 or x >= 1 for x in coordinates
    ):
        result["status"] = "active_parameter_bound"
        return result
    identifiability = report.get("identifiability", {})
    if identifiability.get("rank") != len(free):
        result["status"] = "rank_deficient"
        return result
    matrix = np.asarray(identifiability.get("weighted_jacobian"), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(free) or not np.all(np.isfinite(matrix)):
        raise ValueError("finite weighted Jacobian columns must match the free parameters")
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    threshold = float(identifiability["rank_threshold"])
    if len(singular) != len(free) or np.any(singular <= threshold):
        result["status"] = "rank_deficient"
        return result
    derivatives = np.asarray(
        [p.decode_derivative(x) for p, x in zip(free, coordinates, strict=True)]
    )
    factor = derivatives[:, None] * right.T / singular[None, :]
    covariance = factor @ factor.T
    if not np.all(np.isfinite(covariance)):
        result["status"] = "nonfinite_covariance"
        return result
    result.update(
        status="available",
        covariance=covariance.tolist(),
        standard_errors={
            name: float(v) for name, v in zip(order, np.sqrt(np.diag(covariance)), strict=True)
        },
        weighted_jacobian_rank=len(free),
    )
    return result
