"""Index-1 residual reduction into the existing ODE backend.

This is not an IDA implementation. Differential initial coordinates stay fixed;
one common Newton solver resolves rates and algebraic coordinates. The local
regularity contract requires nonsingular differential mass and algebraic blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from chemworld.physchem.residual_system import (
    Matrix,
    MatrixInput,
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualEvaluator,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    in_bounds,
    linear_solve,
    matrix,
    readonly,
    solve_root,
    vector,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode

RHSFunction = Callable[[float, NDArray[np.float64], NDArray[np.float64]], ArrayLike]
RHSJacobian = Callable[[float, NDArray[np.float64], NDArray[np.float64]], MatrixInput]


class _ODEEvent(Protocol):
    direction: int
    terminal: bool

    def __call__(self, t: float, x: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ResidualSolvePolicy:
    ode: ODESolverPolicy = field(
        default_factory=lambda: ODESolverPolicy(
            "chemworld-index1-reduction-bdf-v1", method="BDF", rtol=1e-8, atol=1e-11
        )
    )
    root: RootPolicy = field(default_factory=lambda: RootPolicy(tolerance=1e-12))
    structure_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.ode.method not in {"BDF", "Radau"} or self.ode.vectorized:
            raise ValueError("residual reduction supports nonvectorized BDF or Radau")
        if not isfinite(self.structure_tolerance) or self.structure_tolerance <= 0:
            raise ValueError("structure tolerance must be positive and finite")


DEFAULT_RESIDUAL_POLICY = ResidualSolvePolicy()


@dataclass(frozen=True)
class ResidualPoint:
    time_s: float
    state: np.ndarray
    rate: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "state", readonly(self.state))
        object.__setattr__(self, "rate", readonly(self.rate))


@dataclass(frozen=True)
class ConsistentInitialResult:
    point: ResidualPoint | None
    success: bool
    failure_reason: str
    diagnostics: dict


@dataclass(frozen=True)
class ResidualEvent:
    name: str
    value: Callable[[float, NDArray[np.float64], NDArray[np.float64]], float]
    direction: int = 0
    terminal: bool = True
    initial_violation_stops: bool = False

    def __post_init__(self) -> None:
        if not self.name or not callable(self.value) or self.direction not in {-1, 0, 1}:
            raise ValueError("residual event needs identity, callback and crossing direction")
        if type(self.terminal) is not bool:
            raise ValueError("event terminal flag must be boolean")
        if type(self.initial_violation_stops) is not bool:
            raise ValueError("initial event violation flag must be boolean")


@dataclass(frozen=True)
class ResidualSolveReport:
    points: tuple[ResidualPoint, ...]
    success: bool
    failure_reason: str
    diagnostics: dict
    events: tuple[tuple[str, ResidualPoint], ...] = ()
    stopping_event: str | None = None

    @property
    def final_point(self) -> ResidualPoint | None:
        return self.points[-1] if self.success and self.points else None


def _sub(
    value: Matrix, rows: NDArray[np.int_], columns: NDArray[np.int_],
) -> Matrix:
    return value[rows][:, columns] if sparse.issparse(value) else value[np.ix_(rows, columns)]


def _maximum(value: Matrix) -> float:
    entries = value.data if sparse.issparse(value) else value
    return float(np.max(np.abs(entries), initial=0))


class _Reduction:
    def __init__(self, system: ResidualSystem, policy: ResidualSolvePolicy) -> None:
        self.system, self.policy = system, policy
        self.evaluator = ResidualEvaluator(system)
        self.d, self.a = system.differential, system.algebraic
        self.n = len(system.variables)
        self.root_calls, self.root_iterations, self.maximum_residual = 0, 0, 0.0
        self.last_root: dict[str, Any] | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.evaluator.diagnostics(),
            "root_calls": self.root_calls,
            "root_iterations": self.root_iterations,
            "maximum_scaled_residual": self.maximum_residual,
            "last_root": self.last_root,
            "algorithm": "index1_residual_reduction",
        }

    def blocks(
        self, t: float, y: NDArray[np.float64], rate: NDArray[np.float64], *,
        linearize: bool = False,
    ) -> tuple[Matrix, Matrix]:
        fy = self.evaluator.derivative("state", t, y, rate, columns=None if linearize else self.a)
        fya = fy[:, self.a] if linearize else fy
        fv = self.evaluator.derivative("rate", t, y, rate)
        scale = self.system.scales / self.system.time_scale_s
        rows = self.system.residual_scales
        normalized = (
            sparse.diags(1 / rows) @ fv @ sparse.diags(scale)
            if sparse.issparse(fv)
            else fv * scale[None, :] / rows[:, None]
        )
        if (
            _maximum(normalized[self.a, :]) > self.policy.structure_tolerance
            or _maximum(normalized[:, self.a]) > self.policy.structure_tolerance
        ):
            raise ResidualNumericalError("unsupported_algebraic_rate_dependence")
        for block in (_sub(fv, self.d, self.d), fya[self.a, :]):
            if block.shape[0]:
                linear_solve(block, np.zeros(block.shape[0]))
        g = (
            sparse.hstack((fv[:, self.d], fya), format="csc")
            if sparse.issparse(fv) or sparse.issparse(fy)
            else np.column_stack((fv[:, self.d], fya))
        )
        return fy, g

    def resolve(
        self, t: float, x: NDArray[np.float64], guess: NDArray[np.float64],
        rate_guess: NDArray[np.float64], *, full_rate: bool = False, linearize: bool = False,
    ) -> tuple[ResidualPoint, Matrix, Matrix]:
        y = np.asarray(guess, dtype=float).copy()
        y[self.d] = x

        def unpack(u: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            state, rate = y.copy(), np.zeros(self.n)
            rate[self.d], state[self.a] = u[: len(self.d)], u[len(self.d) :]
            return state, rate

        def jacobian(u: NDArray[np.float64]) -> Matrix:
            state, rate = unpack(u)
            fya = self.evaluator.derivative("state", t, state, rate, columns=self.a)
            fv = self.evaluator.derivative("rate", t, state, rate)
            return (
                sparse.hstack((fv[:, self.d], fya), format="csc")
                if sparse.issparse(fv) or sparse.issparse(fya)
                else np.column_stack((fv[:, self.d], fya))
            )

        unknowns = tuple(
            ResidualVariable(
                "rate:" + self.system.variables[i].name,
                self.system.variables[i].unit + "/s",
                self.system.variables[i].scale / self.system.time_scale_s,
            )
            for i in self.d
        ) + tuple(self.system.variables[i] for i in self.a)
        self.root_calls += 1
        root = solve_root(
            lambda u: self.evaluator.value(t, *unpack(u)),
            np.r_[rate_guess[self.d], y[self.a]],
            jacobian=jacobian,
            variables=unknowns,
            residual_scales=self.system.residual_scales,
            policy=self.policy.root,
        )
        self.last_root = {**root.diagnostics, "failure_reason": root.failure_reason}
        self.root_iterations += root.diagnostics["iterations"]
        if not root.success:
            raise ResidualNumericalError("consistency:" + root.failure_reason)
        y, rate = unpack(root.value)
        full_matrix = linearize or (
            full_rate and len(self.a) > 0 and self.system.derivatives.state is not None
        )
        fy, g = self.blocks(t, y, rate, linearize=full_matrix)
        self.maximum_residual = max(self.maximum_residual, root.diagnostics["scaled_residual_norm"])
        if full_rate and len(self.a):
            ft = self.evaluator.derivative("time", t, y, rate)
            # Only g_x @ xdot is needed for the algebraic tangent. In an
            # explicitly numerical mode use a scaled directional difference,
            # not every state column for every output point or event root.
            tangent = (
                _sub(fy, self.a, self.d) @ rate[self.d]
                if full_matrix
                else self.evaluator.state_directional(t, y, rate, rate)[self.a]
            )
            rate[self.a] = linear_solve(
                _sub(fy, self.a, self.a) if full_matrix else fy[self.a, :],
                -tangent - ft[self.a],
            )
        return ResidualPoint(t, y, rate), fy, g


def consistent_initial_conditions(
    system: ResidualSystem, state: ArrayLike, *, time_s: float = 0.0,
    rate_guess: ArrayLike | None = None, policy: ResidualSolvePolicy = DEFAULT_RESIDUAL_POLICY,
) -> ConsistentInitialResult:
    if not isfinite(time_s):
        raise ValueError("initial time must be finite")
    initial = vector(state, len(system.variables), "initial state")
    rate = (
        np.zeros(initial.size)
        if rate_guess is None
        else vector(rate_guess, initial.size, "initial rate guess")
    )
    reduction = _Reduction(system, policy)
    if not in_bounds(initial, system.variables):
        return ConsistentInitialResult(
            None, False, "initial_outside_bounds", reduction.diagnostics()
        )
    try:
        point, _, _ = reduction.resolve(time_s, initial[reduction.d], initial, rate, full_rate=True)
        return ConsistentInitialResult(point, True, "", reduction.diagnostics())
    except (ResidualDomainError, ResidualNumericalError) as exc:
        return ConsistentInitialResult(None, False, str(exc), reduction.diagnostics())


def solve_residual(
    system: ResidualSystem,
    initial: ArrayLike,
    *,
    time_span_s: tuple[float, float],
    evaluation_times_s: Sequence[float] | np.ndarray | None = None,
    policy: ResidualSolvePolicy = DEFAULT_RESIDUAL_POLICY,
    events: Sequence[ResidualEvent] = (),
    initial_rate: ArrayLike | None = None,
) -> ResidualSolveReport:
    """Return consistent candidates only; never commit a partial/failed interval."""
    start, end = map(float, time_span_s)
    if not isfinite(start) or not isfinite(end) or end < start:
        raise ValueError("residual time span must be finite and nondecreasing")
    n = len(system.variables)
    y0 = vector(initial, n, "initial state")
    v0 = np.zeros(n) if initial_rate is None else vector(initial_rate, n, "initial rate")
    if not len(system.differential):
        raise ValueError("pure algebraic systems use the steady residual solver")
    grid = None if evaluation_times_s is None else np.asarray(evaluation_times_s, dtype=float)
    if grid is not None and (
        grid.ndim != 1
        or not np.all(np.isfinite(grid))
        or np.any(grid < start)
        or np.any(grid > end)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("evaluation grid must be strictly ordered within the time span")
    events = tuple(events)
    if len({e.name for e in events}) != len(events):
        raise ValueError("event identities must be unique")
    reduction = _Reduction(system, policy)
    ode_diagnostic: dict[str, Any] = {}
    point_cache_hits = 0

    def diagnostics() -> dict[str, Any]:
        return {
            **reduction.diagnostics(),
            "ode": ode_diagnostic,
            "requested_span_s": [start, end],
            "point_cache_hits": point_cache_hits,
        }

    if not in_bounds(y0, system.variables):
        return ResidualSolveReport((), False, "initial_outside_bounds", diagnostics())
    try:
        first, _, _ = reduction.resolve(start, y0[reduction.d], y0, v0, full_rate=True)
        guess, rate_guess = first.state, first.rate
        scale = system.scales[reduction.d]
        cached_point_key: tuple[float, bytes] | None = None
        cached_resolutions: dict[tuple[bool, bool], tuple[ResidualPoint, Matrix, Matrix]] = {}

        def point(
            t: float, x: NDArray[np.float64], *, full_rate: bool = False, linearize: bool = False,
        ) -> tuple[ResidualPoint, Matrix, Matrix]:
            nonlocal cached_point_key, cached_resolutions, point_cache_hits
            # A solver/event batch often requests the exact same point many
            # times. Cache one immutable candidate inside this solve only;
            # never retain a trajectory cache, world state or continuation.
            key = (float(t), np.asarray(x, dtype=float).tobytes())
            if key != cached_point_key:
                cached_point_key, cached_resolutions = key, {}
            flags = (full_rate, linearize)
            if flags in cached_resolutions:
                point_cache_hits += 1
                return cached_resolutions[flags]
            result = reduction.resolve(
                t, x * scale, guess, rate_guess, full_rate=full_rate, linearize=linearize
            )
            cached_resolutions[flags] = result
            return result

        def event_value(spec: ResidualEvent, candidate: ResidualPoint) -> float:
            value = float(spec.value(candidate.time_s, candidate.state, reduction.evaluator.p))
            if not isfinite(value):
                raise ResidualDomainError("nonfinite_event:" + spec.name)
            return value

        def initial_trigger(e: ResidualEvent) -> bool:
            value = event_value(e, first)
            return value == 0 or (
                e.terminal
                and e.initial_violation_stops
                and e.direction != 0
                and e.direction * value > 0
            )

        initial_events = tuple((e.name, first) for e in events if initial_trigger(e))
        initial_stop = next(
            (
                e.name
                for e in events
                if e.terminal and any(name == e.name for name, _ in initial_events)
            ),
            None,
        )
        if start == end or initial_stop is not None:
            return ResidualSolveReport(
                (first,), True, "", diagnostics(), initial_events, initial_stop
            )

        def rhs(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
            candidate, _, _ = point(t, x)
            return candidate.rate[reduction.d] / scale

        def jacobian(t: float, x: NDArray[np.float64]) -> sparse.csc_matrix:
            _, fy, g = point(t, x, linearize=True)
            reduced = linear_solve(g, -fy[:, reduction.d])[: len(reduction.d)]
            # Elimination may introduce fill even when the original blocks are sparse.
            return (
                sparse.diags(1 / scale) @ reduced @ sparse.diags(scale)
                if sparse.issparse(reduced)
                else sparse.csc_matrix(reduced * scale[None, :] / scale[:, None])
            )

        callbacks = []
        for spec in events:

            def callback(t: float, x: NDArray[np.float64], spec: ResidualEvent = spec) -> float:
                candidate, _, _ = point(t, x)
                return event_value(spec, candidate)

            # SciPy reads these runtime attributes on the event function.
            event_callback = cast(_ODEEvent, callback)
            event_callback.direction = spec.direction
            event_callback.terminal = spec.terminal
            callbacks.append(event_callback)
        ode_policy = replace(policy.ode, event_names=tuple(e.name for e in events))
        report = solve_ode(
            rhs,
            first.state[reduction.d] / scale,
            time_span_s=(start, end),
            evaluation_times_s=None if grid is None else sorted({start, end, *grid}),
            policy=ode_policy,
            events=callbacks or None,
            jac=jacobian,
        )
        ode_diagnostic = report.diagnostic.to_dict()
        if not report.success:
            return ResidualSolveReport((), False, "ode:" + report.message, diagnostics())
        event_points = []
        for spec, times, states in zip(
            events, report.raw_result.t_events or (), report.raw_result.y_events or (), strict=True
        ):
            for t, x in zip(times, states, strict=True):
                candidate, _, _ = point(t, x, full_rate=True)
                event_points.append((spec.name, candidate))
        event_points.sort(key=lambda item: item[1].time_s)
        stopping = [
            (name, p)
            for name, p in event_points
            if next(e.terminal for e in events if e.name == name)
        ]
        terminal = min(stopping, key=lambda item: item[1].time_s) if stopping else None
        points = [point(t, x, full_rate=True)[0] for t, x in zip(report.t, report.y.T, strict=True)]
        if terminal is not None:
            if points and points[-1].time_s == terminal[1].time_s:
                points[-1] = terminal[1]
            else:
                points.append(terminal[1])
        if any(
            not in_bounds(p.state, system.variables)
            for p in [*points, *(p for _, p in event_points)]
        ):
            raise ResidualNumericalError("accepted_state_outside_bounds")
        return ResidualSolveReport(
            tuple(points),
            True,
            "",
            diagnostics(),
            tuple(event_points),
            None if terminal is None else terminal[0],
        )
    except (ResidualDomainError, ResidualNumericalError) as exc:
        return ResidualSolveReport((), False, str(exc), diagnostics())


def mass_matrix_system(
    system_id: str,
    variables: Sequence[ResidualVariable],
    equations: Sequence[ResidualEquation],
    mass: MatrixInput,
    rhs: RHSFunction,
    *,
    derivatives: str = "analytic",
    rhs_jacobian: RHSJacobian | None = None,
    rhs_parameter_jacobian: RHSJacobian | None = None,
    rhs_time_derivative: RHSFunction | None = None,
    parameters: Sequence[ResidualVariable] = (),
    parameter_values: Sequence[float] = (),
    time_scale_s: float = 1.0,
) -> ResidualSystem:
    """Constant M ydot = f(t,y,p), including exact zero algebraic rows."""
    n = len(variables)
    owned = matrix(mass, (n, n), "mass matrix")
    if derivatives in {"analytic", "hybrid"}:
        if rhs_jacobian is None and derivatives == "analytic":
            raise ValueError("analytic mass-matrix models need their RHS Jacobian")
        providers = ResidualDerivatives(
            derivatives,
            state=None
            if rhs_jacobian is None
            else lambda t, y, v, p: -matrix(rhs_jacobian(t, y, p), (n, n), "RHS Jacobian"),
            rate=lambda t, y, v, p: owned.copy(),
            parameter=None
            if rhs_parameter_jacobian is None
            else lambda t, y, v, p: (
                -matrix(
                    rhs_parameter_jacobian(t, y, p), (n, len(parameters)), "RHS parameter Jacobian"
                )
            ),
            time=None
            if rhs_time_derivative is None
            else lambda t, y, v, p: -vector(rhs_time_derivative(t, y, p), n, "RHS time derivative"),
        )
    else:
        if any(f is not None for f in (rhs_jacobian, rhs_parameter_jacobian, rhs_time_derivative)):
            raise ValueError("numerical mass-matrix derivatives cannot include analytic callbacks")
        providers = ResidualDerivatives(derivatives)
    return ResidualSystem(
        system_id,
        tuple(variables),
        tuple(equations),
        lambda t, y, v, p: owned @ v - vector(rhs(t, y, p), n, "RHS"),
        providers,
        tuple(parameters),
        tuple(parameter_values),
        time_scale_s,
    )
