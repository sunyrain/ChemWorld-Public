"""Bounded influence calibration with explicit training and untouched holdout data.

The forward callable returns the common molecular-interface result. It may cache
bulk coexistence (which is independent of influence coefficients), but receives
the current influence declaration on every call. Measured uncertainty weights the
loss; it is not a promise about EOS/model discrepancy or holdout accuracy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import MolecularEOSState, ResidualHelmholtzModel
from chemworld.physchem.molecular_interface import (
    MolecularInterfaceResult,
    PolynomialSquareGradientInfluence,
    molecular_planar_interface,
    pure_square_gradient_interface,
)


@dataclass(frozen=True)
class SurfaceTensionObservation:
    observation_id: str
    temperature_K: float
    surface_tension_N_m: float
    standard_uncertainty_N_m: float
    partition: str
    source: str

    def __post_init__(self) -> None:
        if (not self.observation_id or not self.source
                or self.partition not in {"training", "holdout"}
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in
                       (self.temperature_K, self.surface_tension_N_m,
                        self.standard_uncertainty_N_m))):
            raise ValueError("surface tension data need identity, source, split and positive SI")


@dataclass(frozen=True)
class InfluenceFitParameter:
    """A physical coefficient or symmetric pair interaction; scale conditions the fit."""

    name: str
    kind: str
    indices: tuple[int, int]
    lower: float
    upper: float
    scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        if (not self.name or self.kind not in {"pure_coefficient", "interaction"}
                or len(self.indices) != 2 or any(type(i) is not int or i < 0 for i in self.indices)
                or not all(isfinite(x) for x in (self.lower, self.upper, self.scale))
                or self.lower >= self.upper or self.scale <= 0
                or (self.kind == "interaction" and self.indices[0] >= self.indices[1])):
            raise ValueError("fit parameter needs unique index, ordered bounds and positive scale")


@dataclass(frozen=True)
class MolecularInterfaceFitCase:
    """Reusable native forward problem; coexistence and mesh stay fixed during fitting."""

    model: ResidualHelmholtzModel
    left_state: MolecularEOSState
    right_state: MolecularEOSState
    mesh: DiffusionMesh | None = None
    component_density_bounds: tuple[tuple[float, float], ...] = ()
    interior_density_grid: tuple[float, ...] = ()
    max_iterations: int = 500
    residual_tolerance: float = 1e-7
    first_integral_tolerance: float = .05
    initial_profile: tuple[tuple[float, ...], ...] | None = None

    def evaluate(self, observation: SurfaceTensionObservation,
                 influence: PolynomialSquareGradientInfluence) -> MolecularInterfaceResult:
        if abs(observation.temperature_K-self.left_state.temperature_K) > 1e-9:
            raise ValueError("fit observation and coexistence temperatures disagree")
        if self.mesh is None:
            return pure_square_gradient_interface(
                self.model, self.left_state, self.right_state, influence,
                interior_density_grid=self.interior_density_grid)
        return molecular_planar_interface(
            self.model, self.left_state, self.right_state, influence, mesh=self.mesh,
            component_density_bounds=self.component_density_bounds,
            initial_profile=self.initial_profile,
            max_iterations=self.max_iterations, residual_tolerance=self.residual_tolerance,
            first_integral_tolerance=self.first_integral_tolerance)


@dataclass(frozen=True)
class MolecularInfluenceFitResult:
    success: bool
    influence: PolynomialSquareGradientInfluence | None
    parameters: tuple[dict[str, Any], ...] = ()
    observations: tuple[dict[str, Any], ...] = ()
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "influence": None if self.influence is None
                else self.influence.to_dict(),
                "scope": "bounded_weighted_training_fit; holdout_never_enters_loss"}


InterfacePredictor = Callable[
    [SurfaceTensionObservation, PolynomialSquareGradientInfluence], MolecularInterfaceResult]


def _coefficient(influence: PolynomialSquareGradientInfluence,
                 parameter: InfluenceFitParameter) -> float:
    table = (influence.pure_coefficients_J_m5_mol2 if parameter.kind == "pure_coefficient"
             else influence.interaction)
    i, j = parameter.indices
    try:
        return float(table[i][j])
    except IndexError as exc:
        raise ValueError("fit parameter index is outside the influence declaration") from exc


def _updated(template: PolynomialSquareGradientInfluence,
             parameters: Sequence[InfluenceFitParameter],
             values: NDArray[np.float64]) -> PolynomialSquareGradientInfluence:
    pure = [list(row) for row in template.pure_coefficients_J_m5_mol2]
    interaction = [list(row) for row in template.interaction]
    for parameter, value in zip(parameters, values, strict=True):
        i, j = parameter.indices
        if parameter.kind == "pure_coefficient":
            pure[i][j] = float(value*parameter.scale)
        else:
            interaction[i][j] = interaction[j][i] = float(value*parameter.scale)
    candidate = replace(template, pure_coefficients_J_m5_mol2=tuple(tuple(r) for r in pure),
                        interaction=tuple(tuple(r) for r in interaction))
    # The declaration owns full-domain positivity and mixing-matrix rank/inertia.
    return candidate


def _physical_jacobian_evidence(
    values: NDArray[np.float64], residuals: NDArray[np.float64],
    jacobian: NDArray[np.float64], records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the returned dense two-point Jacobian using only executed calls.

    Match actual parameter displacements instead of guessing SciPy's relative
    step, zero-coordinate substitution or bound direction. Rejected later trials
    need not be adjacent to the final Jacobian. Every accepted column must have
    an exactly matching, entirely physical base and difference evaluation.
    """
    samples = [(record, np.asarray(record["parameters_scaled"]),
                np.asarray(record["residuals"])) for record in records]

    def physical(record: dict[str, Any]) -> bool:
        return (bool(record["declaration_valid"])
                and len(record["forward_success"]) == len(residuals)
                and all(record["forward_success"]))

    bases = [record["objective_call"] for record, x, f in samples
             if np.array_equal(x, values) and np.array_equal(f, residuals)
             and physical(record)]
    shape_valid = (jacobian.shape == (len(residuals), len(values))
                   and np.all(np.isfinite(jacobian))
                   and np.all(np.isfinite(values)) and np.all(np.isfinite(residuals)))
    columns = []
    for index in range(len(values)):
        matches = []
        if shape_valid:
            for record, x, f in samples:
                if x.shape != values.shape or f.shape != residuals.shape:
                    continue
                changed = np.flatnonzero(x != values)
                if changed.tolist() != [index] or not np.all(np.isfinite(f)):
                    continue
                step = float(x[index]-values[index])
                if not isfinite(step) or step == 0:
                    continue
                column = (f-residuals)/step
                if np.array_equal(column, jacobian[:, index]):
                    matches.append({"objective_call": record["objective_call"],
                                    "step_scaled": step, "physical": physical(record)})
        columns.append({"parameter_index": index, "matching_calls": matches,
                        "verified": bool(bases and any(m["physical"] for m in matches))})
    verified = bool(shape_valid and bases and all(column["verified"] for column in columns))
    reason = ("" if verified else "invalid_optimizer_jacobian" if not shape_valid else
              "no_physical_final_base" if not bases else "missing_physical_difference_columns")
    return {"verified": verified, "method": "exact_recorded_dense_two_point",
            "base_objective_calls": bases, "columns": columns, "failure_reason": reason,
            "additional_forward_calls": 0}


def fit_molecular_influence(
    template: PolynomialSquareGradientInfluence,
    parameters: Sequence[InfluenceFitParameter],
    observations: Sequence[SurfaceTensionObservation],
    predictor: InterfacePredictor, *, maximum_evaluations: int = 100,
    difference_step: float = 1e-3, optimality_tolerance: float = 1e-7,
    rank_relative_tolerance: float = 1e-7, maximum_condition_number: float = 1e10,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> MolecularInfluenceFitResult:
    """Fit selected coefficients; retain failed trials and every final holdout error.

    Algorithmic success requires a full-rank scaled training Jacobian verified
    against physical recorded differences, successful final forward problems,
    and optimizer termination. Predictive applicability
    is a separate caller-defined decision, never inferred from experimental sigma.
    The covariance is a local, unconstrained linear approximation and is not
    supplied at active bounds or for a rank-deficient model.
    """
    parameters, observations = tuple(parameters), tuple(observations)
    training = tuple(o for o in observations if o.partition == "training")
    if (not parameters or len(training) < len(parameters)
            or len({p.name for p in parameters}) != len(parameters)
            or len({(p.kind, p.indices) for p in parameters}) != len(parameters)
            or len({o.observation_id for o in observations}) != len(observations)):
        raise ValueError("fit needs unique parameters/data and training count >= parameter count")
    if (type(maximum_evaluations) is not int or maximum_evaluations < 1
            or any(not isfinite(v) or v <= 0 for v in
                   (difference_step, optimality_tolerance, rank_relative_tolerance,
                    maximum_condition_number))):
        raise ValueError("fit budgets and tolerances must be positive")
    if any(not template.temperature_range_K[0] <= o.temperature_K
           <= template.temperature_range_K[1] for o in observations):
        raise ValueError("all training and holdout temperatures must be inside influence domain")
    scales = np.array([p.scale for p in parameters])
    lower, upper = (np.array([getattr(p, bound)/p.scale for p in parameters])
                    for bound in ("lower", "upper"))
    initial = np.array([_coefficient(template, p)/p.scale for p in parameters])
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial influence coefficients must lie in the declared fit bounds")
    _updated(template, parameters, initial)
    failures: list[dict[str, Any]] = []
    objective_trace: list[dict[str, Any]] = []
    calls = 0
    forward_calls = 0

    def predict(o: SurfaceTensionObservation, declaration: PolynomialSquareGradientInfluence,
                stage: str) -> tuple[float | None, str]:
        nonlocal forward_calls
        forward_calls += 1
        try:
            result = predictor(o, declaration)
            gamma = result.surface_tension_N_m
            if not result.success or gamma is None or not isfinite(gamma) or gamma <= 0:
                raise ValueError(result.failure_reason or "invalid/nonpositive predicted tension")
            return float(gamma), ""
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append({"stage": stage, "objective_call": calls,
                             "observation_id": o.observation_id, "error": message,
                             "influence": declaration.to_dict()})
            return None, message

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        try:
            declaration = _updated(template, parameters, values)
        except (ValueError, ArithmeticError) as exc:
            failures.append({"stage": "training_candidate", "objective_call": calls,
                             "parameters": (values*scales).tolist(), "error": str(exc)})
            rejected = np.full(len(training), 1e10)
            objective_trace.append({"objective_call": calls,
                "parameters_scaled": values.tolist(), "residuals": rejected.tolist(),
                "declaration_valid": False, "forward_success": [False]*len(training)})
            return rejected
        residuals = []
        forward_success = []
        for o in training:
            gamma, _ = predict(o, declaration, "training_objective")
            forward_success.append(gamma is not None)
            residuals.append(1e10 if gamma is None else
                             (gamma-o.surface_tension_N_m)/o.standard_uncertainty_N_m)
        if progress is not None:
            progress({"stage": "training", "objective_calls": calls,
                      "forward_calls": forward_calls, "failed_calls": len(failures)})
        vector = np.asarray(residuals)
        objective_trace.append({"objective_call": calls,
            "parameters_scaled": values.tolist(), "residuals": vector.tolist(),
            "declaration_valid": True, "forward_success": forward_success})
        return vector

    optimum = least_squares(residual, initial, bounds=(lower, upper),
        max_nfev=maximum_evaluations, diff_step=difference_step, x_scale="jac",
        ftol=optimality_tolerance, xtol=optimality_tolerance, gtol=optimality_tolerance)
    optimizer_values = {"optimizer_scaled_parameters": optimum.x.tolist(),
                        "optimizer_training_residuals": optimum.fun.tolist(),
                        "optimizer_scaled_jacobian": np.asarray(optimum.jac).tolist()}
    try:
        fitted = _updated(template, parameters, optimum.x)
    except (ValueError, ArithmeticError) as exc:
        return MolecularInfluenceFitResult(False, None, failure_reason=str(exc),
            diagnostics={"failed_evaluations": failures, "objective_calls": calls,
                         "forward_calls": forward_calls, "objective_trace": objective_trace,
                         "training_observation_ids": [o.observation_id for o in training],
                         "physical_jacobian_verified": False, **optimizer_values})
    rows: list[dict[str, Any]] = []
    for observation in observations:
        gamma, reason = predict(observation, fitted, "final_"+observation.partition)
        error = None if gamma is None else gamma-observation.surface_tension_N_m
        rows.append({**asdict(observation), "predicted_N_m": gamma, "error_N_m": error,
                     "relative_error": None if error is None else
                     error/observation.surface_tension_N_m,
                     "standardized_residual": None if error is None else
                     error/observation.standard_uncertainty_N_m,
                     "success": gamma is not None, "failure_reason": reason})
    jacobian = np.asarray(optimum.jac)
    jacobian_evidence = _physical_jacobian_evidence(
        optimum.x, optimum.fun, jacobian, objective_trace)
    physical_jacobian_verified = jacobian_evidence["verified"]
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    cutoff = rank_relative_tolerance*singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff)) if singular_values[0] > 0 else 0
    condition = (float(singular_values[0]/singular_values[-1])
                 if singular_values[-1] > 0 else None)
    active = [p.name for p, value, lo, hi in zip(parameters, optimum.x, lower, upper, strict=True)
              if min(value-lo, hi-value) <= 1e-6*max(1, abs(value))]
    identifiable = (physical_jacobian_verified and rank == len(parameters)
                    and condition is not None and condition <= maximum_condition_number)
    covariance = None
    if identifiable and not active and len(training) > len(parameters):
        covariance = ((np.linalg.inv(jacobian.T@jacobian)
                       *float(2*optimum.cost/(len(training)-len(parameters))))
                      *scales[:, None]*scales[None, :]).tolist()
    metrics = {}
    for partition in ("training", "holdout"):
        selected = [row for row in rows if row["partition"] == partition]
        valid = [row for row in selected if row["success"]]
        metrics[partition] = {
            "total": len(selected), "successful": len(valid), "failed": len(selected)-len(valid),
            "complete": len(valid) == len(selected),
            "rmse_N_m": None if not valid else
                float(np.sqrt(np.mean([row["error_N_m"]**2 for row in valid]))),
            "mean_absolute_relative_error": None if not valid else
                float(np.mean([abs(row["relative_error"]) for row in valid])),
            "maximum_absolute_relative_error": None if not valid else
                max(abs(row["relative_error"]) for row in valid),
            "metric_denominator": len(valid),
        }
    reasons = []
    if not optimum.success:
        reasons.append("optimizer: "+str(optimum.message))
    if not physical_jacobian_verified:
        reasons.append("final training Jacobian has no verified physical difference stencil")
    elif not identifiable:
        reasons.append("training sensitivity is rank deficient or ill conditioned")
    if not all(row["success"] for row in rows):
        reasons.append("one or more final training/holdout forward problems failed")
    return MolecularInfluenceFitResult(not reasons, fitted,
        tuple({**asdict(p), "value": float(v*p.scale), "at_bound": p.name in active}
              for p, v in zip(parameters, optimum.x, strict=True)), tuple(rows), "; ".join(reasons),
        {"optimizer_success": bool(optimum.success), "optimizer_message": str(optimum.message),
         "optimizer_evaluations": int(optimum.nfev), "objective_calls": calls,
         "forward_calls": forward_calls, "training_weighted_sum_squares": float(2*optimum.cost),
         "training_degrees_of_freedom": len(training)-len(parameters),
         "scaled_jacobian_singular_values": singular_values.tolist()
             if physical_jacobian_verified else None,
         "jacobian_rank": rank if physical_jacobian_verified else None,
         "scaled_jacobian_condition": condition if physical_jacobian_verified else None,
         "identifiable": identifiable,
         "optimizer_jacobian_singular_values": singular_values.tolist(),
         "optimizer_jacobian_rank": rank, "optimizer_jacobian_condition": condition,
         "physical_jacobian_verified": physical_jacobian_verified,
         "physical_jacobian_evidence": jacobian_evidence,
         "objective_trace": objective_trace,
         "training_observation_ids": [o.observation_id for o in training],
         **optimizer_values,
         "active_bounds": active, "local_physical_parameter_covariance": covariance,
         "covariance_scope": "local_linear_unconstrained; scaled_by_training_residual_variance",
         "failed_evaluations": failures, "metrics": metrics,
         "holdout_policy": "evaluated once after fit; never enters objective or model selection",
         "applicability": "not inferred; caller must predeclare independent predictive tolerances"})
