"""Auditable ODE solver backend contracts for ChemWorld physics modules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

ODEFunction = Callable[[float, np.ndarray], np.ndarray]
ODEEvent = Callable[[float, np.ndarray], float]


@dataclass(frozen=True)
class ODESolverPolicy:
    """Numerical policy for deterministic, replayable ODE integrations."""

    solver_id: str
    method: str = "LSODA"
    rtol: float = 1.0e-8
    atol: float = 1.0e-12
    backend: str = "scipy.integrate.solve_ivp"
    max_step_s: float | None = None
    first_step_s: float | None = None
    dense_output: bool = False
    vectorized: bool = False
    event_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.solver_id.strip():
            raise ValueError("solver_id cannot be empty")
        if not self.method.strip():
            raise ValueError("method cannot be empty")
        if self.rtol <= 0.0 or not isfinite(self.rtol):
            raise ValueError("rtol must be finite and positive")
        if self.atol <= 0.0 or not isfinite(self.atol):
            raise ValueError("atol must be finite and positive")
        if self.max_step_s is not None and (
            self.max_step_s <= 0.0 or not isfinite(self.max_step_s)
        ):
            raise ValueError("max_step_s must be finite and positive when provided")
        if self.first_step_s is not None and (
            self.first_step_s <= 0.0 or not isfinite(self.first_step_s)
        ):
            raise ValueError("first_step_s must be finite and positive when provided")
        if len(self.event_names) != len(set(self.event_names)):
            raise ValueError("event_names cannot contain duplicates")
        if any(not name.strip() for name in self.event_names):
            raise ValueError("event_names cannot contain empty names")

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(self._hash_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        payload = self._hash_payload()
        payload["policy_hash"] = self.policy_hash
        return payload

    def scipy_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "dense_output": self.dense_output,
            "vectorized": self.vectorized,
        }
        if self.max_step_s is not None:
            kwargs["max_step"] = self.max_step_s
        if self.first_step_s is not None:
            kwargs["first_step"] = self.first_step_s
        return kwargs

    def _hash_payload(self) -> dict[str, object]:
        return {
            "solver_id": self.solver_id,
            "backend": self.backend,
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step_s": self.max_step_s,
            "first_step_s": self.first_step_s,
            "dense_output": self.dense_output,
            "vectorized": self.vectorized,
            "event_names": list(self.event_names),
        }


@dataclass(frozen=True)
class ODESolverDiagnostic:
    """JSON-friendly diagnostics emitted by one ODE solver call."""

    policy: ODESolverPolicy
    success: bool
    message: str
    status: int
    nfev: int
    njev: int | None
    nlu: int | None
    t_start_s: float
    t_end_s: float
    evaluation_count: int
    event_count: int
    final_time_s: float
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.to_dict(),
            "success": self.success,
            "message": self.message,
            "status": self.status,
            "nfev": self.nfev,
            "njev": self.njev,
            "nlu": self.nlu,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "evaluation_count": self.evaluation_count,
            "event_count": self.event_count,
            "final_time_s": self.final_time_s,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ODESolveReport:
    """Thin typed wrapper around SciPy's result plus ChemWorld diagnostics."""

    raw_result: Any
    diagnostic: ODESolverDiagnostic

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.raw_result.t, dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.raw_result.y, dtype=float)

    @property
    def success(self) -> bool:
        return bool(self.diagnostic.success)

    @property
    def message(self) -> str:
        return self.diagnostic.message

    @property
    def terminal_point(self) -> tuple[float, np.ndarray, int] | None:
        """Actual terminal root, even when t_eval did not include that instant."""
        if self.diagnostic.status != 1:
            return None
        candidates = [
            (float(times[-1]), np.asarray(states[-1], dtype=float), index)
            for index, (times, states) in enumerate(
                zip(
                    self.raw_result.t_events,
                    self.raw_result.y_events,
                    strict=True,
                )
            )
            if len(times)
        ]
        return max(candidates, key=lambda item: item[0]) if candidates else None

    def raise_for_failure(self, context: str) -> None:
        if not self.success:
            raise RuntimeError(f"{context} failed: {self.message}")


DEFAULT_REACTION_ODE_POLICY = ODESolverPolicy(
    solver_id="chemworld_reaction_network_lsoda_v1",
    method="LSODA",
    rtol=1.0e-8,
    atol=1.0e-12,
)

DEFAULT_REACTOR_ODE_POLICY = ODESolverPolicy(
    solver_id="chemworld_reactor_lsoda_v1",
    method="LSODA",
    rtol=1.0e-8,
    atol=1.0e-12,
)

RUNTIME_REACTION_KERNEL_ODE_POLICY = ODESolverPolicy(
    solver_id="chemworld_runtime_reaction_kernel_lsoda_v1",
    method="LSODA",
    rtol=1.0e-7,
    atol=1.0e-11,
)

REFERENCE_REACTION_ODE_POLICY = ODESolverPolicy(
    solver_id="chemworld_reference_reaction_rk45_v1",
    method="RK45",
    rtol=1.0e-6,
    atol=1.0e-10,
)


def solve_ode(
    rhs: ODEFunction,
    y0: Sequence[float] | np.ndarray,
    *,
    time_span_s: tuple[float, float],
    evaluation_times_s: Sequence[float] | np.ndarray | None = None,
    policy: ODESolverPolicy = DEFAULT_REACTION_ODE_POLICY,
    events: Sequence[ODEEvent] | None = None,
    jac_sparsity: Any | None = None,
    jac: Callable | None = None,
) -> ODESolveReport:
    """Solve an ODE using a declared ChemWorld solver policy."""

    y0_array = _validate_initial_state(y0)
    start_s, end_s = _validate_time_span(time_span_s)
    t_eval = _validate_evaluation_times(
        evaluation_times_s,
        start_s=start_s,
        end_s=end_s,
    )
    kwargs = policy.scipy_kwargs()
    if jac is not None:
        if policy.method not in {"BDF", "Radau"}:
            raise ValueError("a declared Jacobian requires the BDF or Radau method")
        kwargs["jac"] = jac
    if jac_sparsity is not None:
        if policy.method not in {"BDF", "Radau"}:
            raise ValueError("Jacobian sparsity requires the BDF or Radau method")
        sparsity = csr_matrix(jac_sparsity)
        if sparsity.shape != (y0_array.size, y0_array.size) or not np.all(
            np.isfinite(sparsity.data)
        ):
            raise ValueError("Jacobian sparsity must be finite and match the packed state")
        kwargs["jac_sparsity"] = sparsity.astype(bool)
    if t_eval is not None:
        kwargs["t_eval"] = t_eval
    if events is not None:
        kwargs["events"] = tuple(events)
    event_count = 0 if events is None else len(events)
    if event_count != len(policy.event_names):
        raise ValueError(
            "events must match policy.event_names so event diagnostics remain auditable"
        )

    result = solve_ivp(rhs, (start_s, end_s), y0_array, **kwargs)
    diagnostic = _diagnostic_from_result(
        result,
        policy=policy,
        start_s=start_s,
        end_s=end_s,
        evaluation_count=0 if t_eval is None else len(t_eval),
    )
    return ODESolveReport(raw_result=result, diagnostic=diagnostic)


def _validate_initial_state(y0: Sequence[float] | np.ndarray) -> np.ndarray:
    y0_array = np.asarray(y0, dtype=float)
    if y0_array.ndim != 1:
        raise ValueError("y0 must be a one-dimensional numeric vector")
    if y0_array.size == 0:
        raise ValueError("y0 cannot be empty")
    if not np.all(np.isfinite(y0_array)):
        raise ValueError("y0 must contain only finite values")
    return y0_array


def _validate_time_span(time_span_s: tuple[float, float]) -> tuple[float, float]:
    start_s, end_s = float(time_span_s[0]), float(time_span_s[1])
    if not isfinite(start_s) or not isfinite(end_s):
        raise ValueError("time_span_s must contain finite values")
    if end_s < start_s:
        raise ValueError("time_span_s end cannot be earlier than start")
    return start_s, end_s


def _validate_evaluation_times(
    evaluation_times_s: Sequence[float] | np.ndarray | None,
    *,
    start_s: float,
    end_s: float,
) -> np.ndarray | None:
    if evaluation_times_s is None:
        return None
    t_eval = np.asarray(tuple(evaluation_times_s), dtype=float)
    if t_eval.ndim != 1:
        raise ValueError("evaluation_times_s must be one-dimensional")
    if t_eval.size == 0:
        return None
    if not np.all(np.isfinite(t_eval)):
        raise ValueError("evaluation_times_s must contain only finite values")
    if np.any(t_eval < start_s - 1.0e-12) or np.any(t_eval > end_s + 1.0e-12):
        raise ValueError("evaluation_times_s must lie within time_span_s")
    if np.any(np.diff(t_eval) < -1.0e-12):
        raise ValueError("evaluation_times_s must be sorted")
    return t_eval


def _diagnostic_from_result(
    result: Any,
    *,
    policy: ODESolverPolicy,
    start_s: float,
    end_s: float,
    evaluation_count: int,
) -> ODESolverDiagnostic:
    t_values = np.asarray(getattr(result, "t", (start_s,)), dtype=float)
    final_time = float(t_values[-1]) if t_values.size else start_s
    if int(getattr(result, "status", -999)) == 1:
        final_time = max(
            (float(times[-1]) for times in getattr(result, "t_events", ()) if len(times)),
            default=final_time,
        )
    event_count = sum(len(event_times) for event_times in getattr(result, "t_events", ()) or ())
    success = bool(getattr(result, "success", False))
    message = str(getattr(result, "message", ""))
    return ODESolverDiagnostic(
        policy=policy,
        success=success,
        message=message,
        status=int(getattr(result, "status", -999)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=_optional_int(getattr(result, "njev", None)),
        nlu=_optional_int(getattr(result, "nlu", None)),
        t_start_s=start_s,
        t_end_s=end_s,
        evaluation_count=evaluation_count,
        event_count=event_count,
        final_time_s=final_time,
        failure_reason="" if success else message,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    return None


def solve_residual(system, initial, **kwargs):
    """Residual/index-1 entry sharing this module's ODE backend and diagnostics."""
    from chemworld.physchem.residual_solver import solve_residual as solve

    return solve(system, initial, **kwargs)


__all__ = [
    "DEFAULT_REACTION_ODE_POLICY",
    "DEFAULT_REACTOR_ODE_POLICY",
    "REFERENCE_REACTION_ODE_POLICY",
    "RUNTIME_REACTION_KERNEL_ODE_POLICY",
    "ODEEvent",
    "ODEFunction",
    "ODESolveReport",
    "ODESolverDiagnostic",
    "ODESolverPolicy",
    "solve_ode",
    "solve_residual",
]
