"""Scaled residuals and sparse nonlinear roots, without physical-state ownership.

Callbacks use native units: F(t, y, ydot, parameters). Differential and algebraic
rows follow the declared variable order. Numerical differentiation is explicit;
missing analytic parameter/time derivatives never silently become differences.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class ResidualDomainError(ValueError):
    """An explicitly unsupported numerical trial, distinct from a coding error."""


class ResidualNumericalError(RuntimeError):
    """A singular or failed numerical solve; callers retain their original state."""


def readonly(values):
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def vector(values, size, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(result)):
        raise ResidualDomainError(f"{name} is nonfinite")
    return result.copy()


def matrix(values, shape, name):
    result = (
        sparse.csc_matrix(values, copy=True)
        if sparse.issparse(values)
        else np.asarray(values, dtype=float).copy()
    )
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result.data if sparse.issparse(result) else result)):
        raise ResidualDomainError(f"{name} is nonfinite")
    return result


def linear_solve(jacobian, rhs):
    """Retain sparse factorization; do not hide singularity with least squares."""
    try:
        if sparse.issparse(jacobian):
            # Frequency-domain consumers have complex forcing or coefficients;
            # never discard a quadrature component at this shared boundary.
            dtype = np.result_type(jacobian.dtype, getattr(rhs, "dtype", np.asarray(rhs).dtype),
                                   np.float64)
            factor = splu(sparse.csc_matrix(jacobian, dtype=dtype))
            if sparse.issparse(rhs):
                # Algebraic elimination can fill in, but sparse RHS must not
                # require a dense n-by-n allocation before that is known.
                rhs = sparse.csc_matrix(rhs, dtype=dtype)
                answers = [
                    sparse.csc_matrix(factor.solve(rhs[:, i].toarray().ravel())[:, None])
                    for i in range(rhs.shape[1])
                ]
                answer = (
                    sparse.hstack(answers, format="csc")
                    if answers
                    else sparse.csc_matrix((jacobian.shape[0], 0), dtype=dtype)
                )
            else:
                answer = factor.solve(np.asarray(rhs, dtype=dtype))
        else:
            answer = np.linalg.solve(jacobian, rhs.toarray() if sparse.issparse(rhs) else rhs)
    except (np.linalg.LinAlgError, RuntimeError) as exc:
        raise ResidualNumericalError("singular_jacobian") from exc
    if not np.all(np.isfinite(answer.data if sparse.issparse(answer) else answer)):
        raise ResidualNumericalError("nonfinite_linear_solution")
    return answer


@dataclass(frozen=True)
class ResidualVariable:
    name: str
    unit: str
    scale: float = 1.0
    role: str = "differential"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("variables require identity, units and positive finite scale")
        if self.role not in {"differential", "algebraic", "parameter"}:
            raise ValueError("unknown residual variable role")
        if any(v is not None and not isfinite(v) for v in (self.lower, self.upper)):
            raise ValueError("declared variable bounds must be finite")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError("variable lower bound must precede upper bound")


@dataclass(frozen=True)
class ResidualEquation:
    name: str
    unit: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("equations require identity, units and positive finite scale")


def bounds(variables):
    return (
        np.asarray([-np.inf if v.lower is None else v.lower for v in variables]),
        np.asarray([np.inf if v.upper is None else v.upper for v in variables]),
    )


def in_bounds(point, variables):
    low, high = bounds(variables)
    return bool(np.all(np.isfinite(point)) and np.all(point >= low) and np.all(point <= high))


def finite_jacobian(
    function, point, scales, *, lower=None, upper=None, relative_step=6e-6, indices=None
):
    """Central differences, or an explicitly numerical one-sided boundary stencil."""
    point = np.asarray(point, dtype=float)
    baseline = np.asarray(function(point), dtype=float)
    columns = []
    for index in range(len(point)) if indices is None else indices:
        value = point[index]
        step = relative_step * max(abs(value), scales[index])
        room = {
            -1: np.inf if lower is None else max(0.0, value - lower[index]),
            1: np.inf if upper is None else max(0.0, upper[index] - value),
        }
        # Prefer a central stencil; near a bound use the roomier one-sided
        # direction instead of a vanishing, cancellation-prone central step.
        if min(room.values()) >= step:
            signs = (-1, 1)
        else:
            signs = tuple(sorted(room, key=room.get, reverse=True))
        samples = {}
        for sign in signs:
            offset = min(step, room[sign])
            if offset <= 0:
                continue
            trial = point.copy()
            trial[index] += sign * offset
            delta = trial[index] - value
            if delta == 0:
                continue
            if (lower is not None and trial[index] < lower[index]) or (
                upper is not None and trial[index] > upper[index]
            ):
                continue
            try:
                samples[sign] = (np.asarray(function(trial), dtype=float), delta)
            except ResidualDomainError:
                continue
            if min(room.values()) < step:
                break
        if len(samples) == 2:
            forward, hf = samples[1]
            backward, negative_hb = samples[-1]
            hb = -negative_hb
            columns.append(
                (hb / (hf + hb)) * ((forward - baseline) / hf)
                + (hf / (hf + hb)) * ((baseline - backward) / hb)
            )
        elif samples:
            sample, delta = next(iter(samples.values()))
            columns.append((sample - baseline) / delta)
        else:
            raise ResidualDomainError("no admissible finite-difference stencil")
    return np.column_stack(columns) if columns else np.empty((baseline.size, 0))


@dataclass(frozen=True)
class ResidualDerivatives:
    """Derivative policy; optional numerical state pattern lists possible rows per column.

    A pattern is a conservative structural declaration, never inferred from
    values that happen to be zero at one point. Existing defaults stay dense.
    """
    mode: str
    state: Callable | None = None
    rate: Callable | None = None
    parameter: Callable | None = None
    time: Callable | None = None
    relative_step: float = 6e-6
    state_sparsity: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"analytic", "finite_difference", "hybrid"}:
            raise ValueError("declare analytic, finite_difference or hybrid derivatives")
        callbacks = (self.state, self.rate, self.parameter, self.time)
        if self.mode == "analytic" and (self.state is None or self.rate is None):
            raise ValueError("analytic residuals require state and rate Jacobians")
        if self.mode == "finite_difference" and any(c is not None for c in callbacks):
            raise ValueError("finite-difference mode cannot masquerade as analytic callbacks")
        if any(c is not None and not callable(c) for c in callbacks):
            raise ValueError("derivative providers must be callable")
        if not isfinite(self.relative_step) or self.relative_step <= 0:
            raise ValueError("difference step must be finite and positive")
        if self.state_sparsity is not None:
            if self.mode == "analytic" or self.state is not None:
                raise ValueError("state sparsity is only an explicit numerical derivative policy")
            from chemworld.physchem.residual_sparse_difference import validate_column_sparsity
            object.__setattr__(self, "state_sparsity", validate_column_sparsity(
                self.state_sparsity, len(self.state_sparsity)
            ))


@dataclass(frozen=True)
class ResidualSystem:
    system_id: str
    variables: tuple[ResidualVariable, ...]
    equations: tuple[ResidualEquation, ...]
    residual: Callable
    derivatives: ResidualDerivatives
    parameters: tuple[ResidualVariable, ...] = ()
    parameter_values: tuple[float, ...] = ()
    time_scale_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("variables", "equations", "parameters", "parameter_values"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.system_id or not callable(self.residual):
            raise ValueError("residual systems need identity and a pure residual callback")
        if not self.variables or len(self.equations) != len(self.variables):
            raise ValueError("residual systems require one equation per variable")
        names = [v.name for v in (*self.variables, *self.parameters)]
        if len(set(names)) != len(names) or len({e.name for e in self.equations}) != len(
            names
        ) - len(self.parameters):
            raise ValueError("variable/parameter and equation identities must be unique")
        if any(v.role == "parameter" for v in self.variables) or any(
            v.role != "parameter" for v in self.parameters
        ):
            raise ValueError("parameter roles must be separate from physical variables")
        p = vector(self.parameter_values, len(self.parameters), "parameters")
        if not in_bounds(p, self.parameters):
            raise ValueError("parameter outside declared bounds")
        if not isfinite(self.time_scale_s) or self.time_scale_s <= 0:
            raise ValueError("nominal time scale must be finite and positive")
        if self.derivatives.state_sparsity is not None:
            from chemworld.physchem.residual_sparse_difference import validate_column_sparsity
            validate_column_sparsity(self.derivatives.state_sparsity,
                                     len(self.variables), len(self.equations))

    @property
    def differential(self):
        return np.asarray(
            [i for i, v in enumerate(self.variables) if v.role == "differential"], dtype=int
        )

    @property
    def algebraic(self):
        return np.asarray(
            [i for i, v in enumerate(self.variables) if v.role == "algebraic"], dtype=int
        )

    @property
    def scales(self):
        return np.asarray([v.scale for v in self.variables])

    @property
    def residual_scales(self):
        return np.asarray([e.scale for e in self.equations])


class ResidualEvaluator:
    """Per-solve counters and differentiation; never retained in a world snapshot."""

    def __init__(self, system):
        self.system = system
        self.calls = 0
        self.derivative_calls = dict.fromkeys(("state", "rate", "parameter", "time"), 0)
        self.directional_calls = 0
        self.p = readonly(system.parameter_values)

    def value(self, t, y, rate, p=None):
        if not isfinite(t):
            raise ValueError("residual time must be finite")
        self.calls += 1
        p = self.p if p is None else readonly(p)
        n = len(self.system.variables)
        return vector(self.system.residual(float(t), readonly(y), readonly(rate), p), n, "residual")

    def derivative(self, kind, t, y, rate, *, columns=None):
        self.derivative_calls[kind] += 1
        system, providers = self.system, self.system.derivatives
        n = len(system.variables)
        width = len(system.parameters) if kind == "parameter" else 1 if kind == "time" else n
        if columns is not None:
            columns = np.asarray(columns)
            if (
                kind not in {"state", "rate"}
                or columns.ndim != 1
                or not np.issubdtype(columns.dtype, np.integer)
                or np.any(columns < 0)
                or np.any(columns >= width)
                or len(set(columns)) != len(columns)
            ):
                raise ValueError("derivative columns must be unique valid state/rate indices")
            if columns.size == 0:
                return sparse.csc_matrix((n, 0))
        if kind == "parameter" and width == 0:
            return np.empty((n, 0))
        callback = getattr(providers, kind)
        if providers.mode == "analytic" or (providers.mode == "hybrid" and callback is not None):
            if callback is None:
                raise ValueError(f"analytic {kind} derivative is unavailable")
            answer = callback(float(t), readonly(y), readonly(rate), self.p)
            checked = (
                vector(answer, n, "time derivative")
                if kind == "time"
                else matrix(answer, (n, width), f"{kind} Jacobian")
            )
            return checked if columns is None else checked[:, columns]
        if kind == "state":
            point, scales = y, system.scales
            low, high = bounds(system.variables)

            def function(v):
                return self.value(t, v, rate)
        elif kind == "rate":
            point, scales = rate, system.scales / system.time_scale_s
            low = high = None

            def function(v):
                return self.value(t, y, v)
        elif kind == "parameter":
            point, scales = self.p, [p.scale for p in system.parameters]
            low, high = bounds(system.parameters)

            def function(v):
                return self.value(t, y, rate, v)
        else:
            point, scales = [t], [system.time_scale_s]
            low = high = None

            def function(v):
                return self.value(v[0], y, rate)

        numerical = finite_jacobian
        options = {}
        if kind == "state" and providers.state_sparsity is not None:
            from chemworld.physchem.residual_sparse_difference import sparse_finite_jacobian
            numerical = sparse_finite_jacobian
            options["sparsity"] = providers.state_sparsity
        answer = numerical(
            function,
            point,
            scales,
            lower=low,
            upper=high,
            relative_step=providers.relative_step,
            indices=columns,
            **options,
        )
        return answer[:, 0] if kind == "time" else answer

    def state_directional(self, t, y, rate, direction):
        """Compute F_y @ direction without materializing a numerical Jacobian."""
        direction = vector(direction, len(self.system.variables), "state direction")
        if self.system.derivatives.state is not None:
            return self.derivative("state", t, y, rate) @ direction
        if self.system.derivatives.mode == "analytic":
            raise ValueError("analytic state derivative is unavailable")
        self.derivative_calls["state"] += 1
        self.directional_calls += 1
        norm = float(np.max(np.abs(direction / self.system.scales)))
        if norm == 0:
            return np.zeros(len(direction))
        if not isfinite(norm):
            raise ResidualDomainError("nonfinite scaled derivative direction")
        low, high = bounds(self.system.variables)
        minimum, maximum = -np.inf, np.inf
        for value, change, lower, upper in zip(y, direction, low, high, strict=True):
            if change != 0:
                a, b = sorted(((lower - value) / change, (upper - value) / change))
                minimum, maximum = max(minimum, a), min(maximum, b)
        return finite_jacobian(
            lambda alpha: self.value(t, np.asarray(y) + alpha[0] * direction, rate),
            [0.0],
            [1 / norm],
            lower=[minimum],
            upper=[maximum],
            relative_step=self.system.derivatives.relative_step,
        )[:, 0]

    def diagnostics(self):
        return {
            "residual_evaluations": self.calls,
            "derivative_evaluations": dict(self.derivative_calls),
            "derivative_mode": self.system.derivatives.mode,
            "state_directional_evaluations": self.directional_calls,
            "derivative_modes": {
                kind: (
                    "analytic"
                    if getattr(self.system.derivatives, kind) is not None
                    else "unavailable"
                    if self.system.derivatives.mode == "analytic"
                    else "finite_difference"
                )
                for kind in self.derivative_calls
            },
        }


@dataclass(frozen=True)
class RootPolicy:
    tolerance: float = 1e-10
    maximum_iterations: int = 40
    maximum_backtracks: int = 24

    def __post_init__(self) -> None:
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("root tolerance must be positive and finite")
        if any(
            type(n) is not int or n < 1 for n in (self.maximum_iterations, self.maximum_backtracks)
        ):
            raise ValueError("root iteration limits must be positive integers")


DEFAULT_ROOT_POLICY = RootPolicy()


@dataclass(frozen=True)
class RootResult:
    value: np.ndarray | None
    success: bool
    failure_reason: str
    diagnostics: dict

    def __post_init__(self) -> None:
        if self.value is not None:
            object.__setattr__(self, "value", readonly(self.value))


def solve_root(
    function, initial, *, jacobian, variables, residual_scales, policy=DEFAULT_ROOT_POLICY,
    progress=None,
):
    """Projected Newton with a scaled gradient fallback and residual merit test.

    Projection constructs a numerical trial, never repairs an accepted state.
    Each trial is evaluated again; a constrained minimum with nonzero residual
    is a failure, even when its projected gradient vanishes.
    """
    variables = tuple(variables)
    n = len(variables)
    if not n:
        raise ValueError("root solve requires at least one variable")
    x = vector(initial, n, "initial root point")
    rows = vector(residual_scales, n, "residual scales")
    if np.any(rows <= 0):
        raise ValueError("residual scales must be positive")
    columns = np.asarray([v.scale for v in variables])
    calls, backtracks, iteration, norm = 0, 0, 0, None
    projected_steps, gradient_steps = 0, 0

    def evaluate(point):
        nonlocal calls
        calls += 1
        return vector(function(readonly(point)), n, "root residual") / rows

    def finish(success, reason=""):
        return RootResult(
            x if success else None,
            success,
            reason,
            {
                "algorithm": "scaled_projected_newton_with_gradient",
                "iterations": iteration,
                "evaluations": calls,
                "backtracks": backtracks,
                "projected_newton_steps": projected_steps,
                "gradient_steps": gradient_steps,
                "scaled_residual_norm": norm,
                "tolerance": policy.tolerance,
            },
        )

    if not in_bounds(x, variables):
        return finish(False, "initial_outside_bounds")
    low, high = bounds(variables)
    try:
        f = evaluate(x)
        for iteration in range(policy.maximum_iterations + 1):
            norm = float(np.linalg.norm(f, ord=np.inf))
            if progress is not None:
                progress({
                    "event": "root-iteration", "iteration": iteration,
                    "maximum_iterations": policy.maximum_iterations,
                    "root_evaluations": calls, "backtracks": backtracks,
                    "scaled_residual_norm": norm,
                })
            if norm <= policy.tolerance:
                return finish(True)
            if iteration == policy.maximum_iterations:
                return finish(False, "iteration_limit")
            j = matrix(jacobian(readonly(x)), (n, n), "root Jacobian")
            scaled = (
                sparse.diags(1 / rows) @ j @ sparse.diags(columns)
                if sparse.issparse(j)
                else j * columns[None, :] / rows[:, None]
            )
            step = linear_solve(scaled, -f)
            gradient = np.asarray(scaled.T @ f).ravel()
            merit = 0.5 * float(f @ f)
            if not np.all(np.isfinite(gradient)) or not isfinite(merit):
                return finish(False, "nonfinite_scaled_gradient_or_merit")
            for direction_kind in ("newton", "gradient"):
                if direction_kind == "gradient":
                    jg = scaled @ gradient
                    curvature = float(jg @ jg)
                    if not np.all(np.isfinite(jg)) or not isfinite(curvature):
                        return finish(False, "nonfinite_gradient_curvature")
                    if curvature == 0:
                        continue
                    step = -gradient * (float(gradient @ gradient) / curvature)
                target = np.clip(x + columns * step, low, high)
                direction = target - x
                slope = float(gradient @ (direction / columns))
                if slope >= 0:
                    continue
                alpha = 1.0
                for _ in range(policy.maximum_backtracks):
                    # A convex combination stays feasible, including exact-bound roots.
                    trial = target.copy() if alpha == 1 else (1 - alpha) * x + alpha * target
                    try:
                        trial_f = evaluate(trial) if in_bounds(trial, variables) else None
                    except ResidualDomainError:
                        trial_f = None
                    if trial_f is not None and (
                        np.linalg.norm(trial_f, np.inf) <= policy.tolerance
                        or 0.5 * float(trial_f @ trial_f) <= merit + 1e-4 * alpha * slope
                    ):
                        if direction_kind == "gradient":
                            gradient_steps += 1
                        elif not np.array_equal(target, x + columns * step):
                            projected_steps += 1
                        x, f = trial, trial_f
                        break
                    alpha *= 0.5
                    backtracks += 1
                else:
                    continue
                break
            else:
                return finish(False, "line_search_failed")
    except (ResidualDomainError, ResidualNumericalError) as exc:
        return finish(False, str(exc))
    raise RuntimeError("unreachable Newton termination")


@dataclass(frozen=True)
class SteadyResidualResult:
    root: RootResult
    parameter_response: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameter_response is not None:
            object.__setattr__(self, "parameter_response", readonly(self.parameter_response))


def solve_steady_residual(
    system, initial, *, time_s=0.0, policy=DEFAULT_ROOT_POLICY, sensitivity=False, progress=None
):
    evaluator = ResidualEvaluator(system)
    zero = np.zeros(len(system.variables))
    root = solve_root(
        lambda y: evaluator.value(time_s, y, zero),
        initial,
        jacobian=lambda y: evaluator.derivative("state", time_s, y, zero),
        variables=system.variables,
        residual_scales=system.residual_scales,
        policy=policy,
        progress=progress,
    )
    response = None
    if root.success and sensitivity:
        j = evaluator.derivative("state", time_s, root.value, zero)
        dp = evaluator.derivative("parameter", time_s, root.value, zero)
        response = linear_solve(j, -dp.toarray() if sparse.issparse(dp) else -dp)
    return SteadyResidualResult(root, response, evaluator.diagnostics())


__all__ = [
    "ResidualDerivatives",
    "ResidualDomainError",
    "ResidualEquation",
    "ResidualEvaluator",
    "ResidualNumericalError",
    "ResidualSystem",
    "ResidualVariable",
    "RootPolicy",
    "RootResult",
    "SteadyResidualResult",
    "solve_root",
    "solve_steady_residual",
]
