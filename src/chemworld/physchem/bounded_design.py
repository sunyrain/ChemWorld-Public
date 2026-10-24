"""Finite constrained boundary design; evaluations own their physical equations.

The optimizer never substitutes a penalty for a failed physical evaluation.
Its candidates and counters are numerical records, not experiment state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

import numpy as np
from scipy.optimize import minimize

from chemworld.data.calibration_core import BoundedParameterCoordinates
from chemworld.physchem.residual_system import finite_jacobian


@dataclass(frozen=True)
class DesignVariable(BoundedParameterCoordinates):
    parameter_id: str
    unit: str
    bounds: tuple[float, float]
    initial: float
    coordinate: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", tuple(self.bounds))
        if (
            not isinstance(self.parameter_id, str)
            or not self.parameter_id
            or not isinstance(self.unit, str)
            or not self.unit
            or len(self.bounds) != 2
        ):
            raise ValueError("design variable requires identity, units and bounds")
        self._validate_coordinates()


@dataclass(frozen=True)
class DesignConstraint:
    metric: str
    unit: str
    scale: float
    tolerance: float
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.metric, str)
            or not self.metric
            or not isinstance(self.unit, str)
            or not self.unit
            or any(
                isinstance(v, bool) for v in (self.scale, self.tolerance, self.lower, self.upper)
            )
            or not isfinite(self.scale)
            or self.scale <= 0
            or not isfinite(self.tolerance)
            or self.tolerance < 0
            or (self.lower is None and self.upper is None)
            or any(v is not None and not isfinite(v) for v in (self.lower, self.upper))
            or (self.lower is not None and self.upper is not None and self.lower > self.upper)
        ):
            raise ValueError(
                "design constraint requires explicit finite bounds, scale and tolerance"
            )

    def violation(self, value: float) -> float:
        return max(
            0.0,
            self.lower - value if self.lower is not None else 0.0,
            value - self.upper if self.upper is not None else 0.0,
        )


@dataclass(frozen=True)
class BoundedDesignPolicy:
    maximum_evaluations: int = 600
    maximum_iterations: int = 120
    objective_scale: float = 10.0
    tolerance: float = 1e-10
    relative_difference_step: float = 1e-5

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v < 1 for v in (self.maximum_evaluations, self.maximum_iterations)
        ) or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (self.objective_scale, self.tolerance, self.relative_difference_step)
        ):
            raise ValueError("design requires finite numerical budgets, scales and tolerances")


class DesignEvaluationError(ValueError):
    def __init__(self, reason: str, diagnostics: Mapping[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.diagnostics = dict(diagnostics or {})


class _BudgetExhaustedError(RuntimeError):
    pass


DEFAULT_DESIGN_POLICY = BoundedDesignPolicy()


@dataclass(frozen=True)
class DesignPoint:
    objective: float
    observations: Mapping[str, float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundedDesignResult:
    success: bool
    values: dict[str, float] | None
    point: DesignPoint | None
    attempts: tuple[dict[str, Any], ...]
    trials: tuple[dict[str, Any], ...]
    best_feasible_trial: int | None
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "physical_evaluations": len(self.trials),
            "scope": "best_verified_converged_local_candidate_from_finite_starts",
            "physical_time_s": None,
            "experiment_resource_debit": None,
            "derivatives": "shared_bounded_finite_jacobian_of_actual_evaluations",
        }


def solve_bounded_design(
    variables: Sequence[DesignVariable],
    constraints: Sequence[DesignConstraint],
    evaluate: Callable[[dict[str, float]], DesignPoint],
    *,
    starts: Sequence[Mapping[str, float]] = (),
    policy: BoundedDesignPolicy = DEFAULT_DESIGN_POLICY,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> BoundedDesignResult:
    variables, constraints = tuple(variables), tuple(constraints)
    names = tuple(p.parameter_id for p in variables)
    if not variables or len(set(names)) != len(names):
        raise ValueError("design requires distinct explicit variables")
    supplied = tuple(starts) or ({p.parameter_id: p.initial for p in variables},)
    normalized = []
    for start in supplied:
        if set(start) != set(names):
            raise ValueError("every design start must name exactly all variables")
        normalized.append(np.array([p.encode(start[p.parameter_id]) for p in variables]))
    # One vector supplies objective and every constraint; one Jacobian supplies
    # all numerical gradients. A cached exact candidate is not a surrogate.
    equations: list[tuple[DesignConstraint, float, float, str]] = []
    for c in constraints:
        if c.lower is not None and c.lower == c.upper:
            equations.append((c, 1.0, c.lower, "eq"))
        else:
            if c.lower is not None:
                equations.append((c, 1.0, c.lower, "ineq"))
            if c.upper is not None:
                equations.append((c, -1.0, c.upper, "ineq"))
    trials: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    feasible: list[int] = []
    candidates: list[tuple[float, Any, int]] = []
    cache: list[Any] = [None, None]
    derivatives: list[Any] = [None, None]
    current_start = 0
    exhausted = False

    def physical(q: Any, *, fresh: bool = False) -> DesignPoint:
        key = tuple(map(float, q))
        if not fresh and cache[0] == key:
            return cache[1]
        if len(trials) >= policy.maximum_evaluations:
            raise _BudgetExhaustedError("design_physical_evaluation_budget_exhausted")
        values = {
            p.parameter_id: float(p.decode(float(x))) for p, x in zip(variables, q, strict=True)
        }
        row: dict[str, Any] = {
            "index": len(trials),
            "start": current_start,
            "values": values,
            "fresh_verification": fresh,
            "success": False,
        }
        trials.append(row)
        try:
            point = evaluate(values)
            if not isfinite(point.objective) or any(
                not isfinite(v) for v in point.observations.values()
            ):
                raise DesignEvaluationError("nonfinite_physical_design_result")
            if any(c.metric not in point.observations for c in constraints):
                raise DesignEvaluationError("missing_design_observation")
            violations = [c.violation(point.observations[c.metric]) for c in constraints]
            passed = all(v <= c.tolerance for v, c in zip(violations, constraints, strict=True))
            row.update(
                success=True,
                objective=point.objective,
                observations=dict(point.observations),
                violations=violations,
                feasible=passed,
                diagnostics=dict(point.diagnostics),
            )
            if passed:
                feasible.append(row["index"])
            cache[:] = [key, point]
            return point
        except (ValueError, ArithmeticError) as exc:
            row.update(failure_reason=str(exc), diagnostics=getattr(exc, "diagnostics", {}))
            raise DesignEvaluationError(str(exc), row["diagnostics"]) from exc
        finally:
            if progress is not None:
                progress(
                    {
                        "start": current_start,
                        "starts": len(normalized),
                        "physical_evaluations": len(trials),
                        "maximum_evaluations": policy.maximum_evaluations,
                        "success": row["success"],
                    }
                )

    def vector(q: Any) -> Any:
        point = physical(q)
        return np.array(
            [
                point.objective / policy.objective_scale,
                *(
                    sign * (point.observations[c.metric] - bound) / c.scale
                    for c, sign, bound, _ in equations
                ),
            ]
        )

    def jacobian(q: Any) -> Any:
        key = tuple(map(float, q))
        if derivatives[0] != key:
            matrix = finite_jacobian(
                vector,
                q,
                np.ones(len(variables)),
                lower=np.zeros(len(variables)),
                upper=np.ones(len(variables)),
                relative_step=policy.relative_difference_step,
            )
            derivatives[:] = [key, matrix]
        return derivatives[1]

    selected: list[dict[str, Any]] = []
    for kind in ("eq", "ineq"):
        indexes = np.array([i + 1 for i, equation in enumerate(equations) if equation[3] == kind])
        if len(indexes):
            selected.append(
                {
                    "type": kind,
                    "fun": lambda q, indexes=indexes: vector(q)[indexes],
                    "jac": lambda q, indexes=indexes: jacobian(q)[indexes],
                }
            )
    for current_start, initial in enumerate(normalized):
        cache[:] = [None, None]
        derivatives[:] = [None, None]
        record: dict[str, Any] = {
            "start": current_start,
            "initial": dict(supplied[current_start]),
            "success": False,
            "first_trial": len(trials),
        }
        attempts.append(record)
        try:
            optimum = minimize(
                lambda q: vector(q)[0],
                initial,
                jac=lambda q: jacobian(q)[0],
                bounds=[(0.0, 1.0)] * len(variables),
                constraints=selected,
                method="SLSQP",
                options={"ftol": policy.tolerance, "maxiter": policy.maximum_iterations},
            )
            record.update(
                optimizer_success=bool(optimum.success),
                optimizer_status=int(optimum.status),
                message=str(optimum.message),
                iterations=int(optimum.nit),
            )
            if not optimum.success:
                record["failure_reason"] = "optimizer_did_not_converge"
                continue
            point = physical(optimum.x, fresh=True)
            if not trials[-1]["feasible"]:
                record["failure_reason"] = "converged_candidate_violates_physical_constraints"
                continue
            record.update(success=True, verified_trial=len(trials) - 1)
            candidates.append((point.objective, np.array(optimum.x), current_start))
        except _BudgetExhaustedError as exc:
            record["failure_reason"] = str(exc)
            exhausted = True
            break
        except DesignEvaluationError as exc:
            record["failure_reason"] = "physical_evaluation_failed:" + str(exc)
        finally:
            record["executed_evaluations"] = len(trials) - record["first_trial"]
    final_point, final_values, reason = None, None, "no_verified_converged_candidate"
    if candidates and not exhausted:
        _, q, current_start = min(candidates, key=lambda pair: pair[0])
        try:
            final_point = physical(q, fresh=True)
            if not trials[-1]["feasible"]:
                final_point = None
                reason = "selected_candidate_failed_fresh_constraints"
            else:
                final_values = dict(trials[-1]["values"])
                reason = ""
        except _BudgetExhaustedError:
            exhausted = True
        except DesignEvaluationError as exc:
            reason = "selected_candidate_failed_fresh_physics:" + str(exc)
    if exhausted:
        reason = "design_physical_evaluation_budget_exhausted"
        final_point, final_values = None, None
    best = min(feasible, key=lambda i: trials[i]["objective"]) if feasible else None
    return BoundedDesignResult(
        final_point is not None,
        final_values,
        final_point,
        tuple(attempts),
        tuple(trials),
        best,
        reason,
    )
