"""Finite trajectory design and receding-horizon control through original sessions.

This is an engineering boundary tool with access to private model state, not an
agent observation or a state estimator. Predictions execute the installed
session on private snapshots. Only the accepted first period reaches the live
session. Constraints supplied by the caller observe declared action endpoints;
continuous physical limits remain owned by the original executor and its events.
User scoring/factory callbacks are deliberately not serialized as executable code.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import isclose, isfinite
from time import perf_counter
from typing import Any

import numpy as np

from chemworld.physchem.bounded_design import (
    BoundedDesignPolicy,
    BoundedDesignResult,
    DesignConstraint,
    DesignEvaluationError,
    DesignPoint,
    DesignVariable,
    solve_bounded_design,
)
from chemworld.runtime.engine import ChemWorldRuntime


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _callback_name(callback: Callable) -> str:
    name = getattr(callback, "__qualname__", type(callback).__name__)
    return f"{getattr(callback, '__module__', '')}.{name}"


def _ledgers(snapshot: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    active = snapshot["world_state"]
    states = {active["vessel_id"]: active, **active["inactive_vessels"]}
    return {name: dict(state["ledger"]) for name, state in states.items()}


def _ledger_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    old, new = _ledgers(before), _ledgers(after)
    return {
        name: {key: value - old.get(name, {}).get(key, 0.0) for key, value in ledger.items()}
        for name, ledger in new.items()
    }


@dataclass(frozen=True)
class TrajectoryAction:
    """An original action; bindings replace its numeric top-level fields.

    ``duration_field`` names elapsed physical duration, not a queue reservation.
    Set it to None for a queued command whose duration is consumed by later wait.
    Shared design values are simply referenced by the same name in several actions.
    """

    action: Mapping[str, Any]
    bindings: Mapping[str, str] = field(default_factory=dict)
    duration_field: str | None = "duration_s"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", deepcopy(dict(self.action)))
        object.__setattr__(self, "bindings", dict(self.bindings))
        if not isinstance(self.action.get("operation"), str) or not self.action["operation"]:
            raise ValueError("trajectory action requires an original operation")
        if any(not k or not isinstance(v, str) or not v for k, v in self.bindings.items()):
            raise ValueError("trajectory bindings require action fields and variable names")
        if "operation" in self.bindings:
            raise ValueError("a numeric design variable cannot select an operation")
        if self.duration_field is not None and (
            not isinstance(self.duration_field, str) or not self.duration_field
        ):
            raise ValueError("duration field must be an explicit field name or None")
        json.dumps(self.action, allow_nan=False)

    def compile(self, values: Mapping[str, float]) -> dict[str, Any]:
        action = deepcopy(dict(self.action))
        for key, name in self.bindings.items():
            value = values[name]
            if isinstance(value, bool) or not isfinite(value):
                raise ValueError("action bindings require finite numeric values")
            action[key] = float(value)
        if self.duration_field is not None and self.duration_field in action:
            duration = action[self.duration_field]
            if isinstance(duration, bool) or not isfinite(duration) or duration <= 0:
                raise ValueError("declared physical action duration must be positive finite")
        return action

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True)
class TrajectoryPeriod:
    period_id: str
    actions: tuple[TrajectoryAction, ...]
    forecast_disturbances: tuple[TrajectoryAction, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "forecast_disturbances", tuple(self.forecast_disturbances))
        if (
            not self.period_id
            or not self.actions
            or any(
                not isinstance(a, TrajectoryAction)
                for a in (*self.actions, *self.forecast_disturbances)
            )
        ):
            raise ValueError("a period needs identity and original action declarations")
        if any(a.bindings for a in self.forecast_disturbances):
            raise ValueError("forecast disturbances must be declared, not optimized")


@dataclass(frozen=True)
class TrajectoryPolicy:
    maximum_actions_per_trial: int = 128
    maximum_prediction_actions: int = 10000
    maximum_horizon_duration_s: float = 14400.0
    duration_tolerance_s: float = 1e-8

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v < 1
            for v in (self.maximum_actions_per_trial, self.maximum_prediction_actions)
        ) or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (self.maximum_horizon_duration_s, self.duration_tolerance_s)
        ):
            raise ValueError("trajectory budgets and duration tolerance must be positive finite")


@dataclass(frozen=True)
class TrajectoryStep:
    period_id: str
    role: str
    action: dict[str, Any]
    duration_field: str | None
    success: bool
    failure_reason: str
    actual_duration_s: float
    wall_time_s: float
    outcome: dict[str, Any]
    ledger_deltas: dict[str, Any]
    snapshot: dict[str, Any]

    def to_dict(self, *, include_snapshot: bool = False) -> dict[str, Any]:
        result = {key: value for key, value in self.__dict__.items() if key != "snapshot"}
        if include_snapshot:
            result["snapshot"] = self.snapshot
        return _json(result)


@dataclass(frozen=True)
class TrajectoryTrace:
    initial_snapshot: dict[str, Any]
    steps: tuple[TrajectoryStep, ...]
    values: dict[str, float]

    @property
    def final_snapshot(self) -> dict[str, Any]:
        return deepcopy(self.steps[-1].snapshot if self.steps else self.initial_snapshot)

    @property
    def actual_duration_s(self) -> float:
        return sum(step.actual_duration_s for step in self.steps)

    def to_dict(self, *, include_snapshots: bool = False) -> dict[str, Any]:
        final = self.final_snapshot
        result = {
            "values": self.values,
            "steps": [s.to_dict(include_snapshot=include_snapshots) for s in self.steps],
            "action_executions": len(self.steps),
            "elapsed_duration_s": self.actual_duration_s,
            "ledger_deltas": _ledger_delta(self.initial_snapshot, final),
            "resource_ledgers_at_end": final["resource_ledgers"],
            "constraint_sampling": "declared_action_endpoints_and_original_physical_events",
            "state_visibility": "private_engineering_model_state",
        }
        if include_snapshots:
            result["initial_snapshot"] = self.initial_snapshot
        return _json(result)


@dataclass(frozen=True)
class TrajectoryDesignSpec:
    variables: tuple[DesignVariable, ...]
    periods: tuple[TrajectoryPeriod, ...]
    score: Callable[[TrajectoryTrace], DesignPoint]
    objective_unit: str
    constraints: tuple[DesignConstraint, ...] = ()
    starts: tuple[Mapping[str, float], ...] = ()
    design_policy: BoundedDesignPolicy = field(default_factory=BoundedDesignPolicy)
    trajectory_policy: TrajectoryPolicy = field(default_factory=TrajectoryPolicy)

    def __post_init__(self) -> None:
        for key in ("variables", "periods", "constraints"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        object.__setattr__(self, "starts", tuple(deepcopy(dict(s)) for s in self.starts))
        names = {v.parameter_id for v in self.variables}
        if not names or len(names) != len(self.variables) or not self.periods:
            raise ValueError("trajectory design needs distinct variables and finite periods")
        if len({p.period_id for p in self.periods}) != len(self.periods):
            raise ValueError("trajectory period identities must be unique")
        if (
            not callable(self.score)
            or not isinstance(self.objective_unit, str)
            or not self.objective_unit
        ):
            raise ValueError("trajectory design needs a scoring callback and objective unit")
        used = {name for p in self.periods for a in p.actions for name in a.bindings.values()}
        if used != names:
            raise ValueError("every design variable must bind an original action field")
        count = sum(len(p.actions) + len(p.forecast_disturbances) for p in self.periods)
        if count > self.trajectory_policy.maximum_actions_per_trial:
            raise ValueError("declared horizon exceeds its action budget")

    def to_dict(self) -> dict[str, Any]:
        return _json(
            {
                "schema": "original-session-trajectory-design-v1",
                "variables": [asdict(v) for v in self.variables],
                "periods": [asdict(p) for p in self.periods],
                "constraints": [asdict(c) for c in self.constraints],
                "starts": self.starts,
                "objective_unit": self.objective_unit,
                "design_policy": asdict(self.design_policy),
                "trajectory_policy": asdict(self.trajectory_policy),
                "score_callback": _callback_name(self.score),
                "callback_serialization": "description_only; explicit_callable_required",
            }
        )


def _execute_action(
    runtime: ChemWorldRuntime,
    declared: TrajectoryAction,
    values: Mapping[str, float],
    period_id: str,
    role: str,
    tolerance: float,
) -> TrajectoryStep:
    action = declared.compile(values)
    before = runtime.snapshot().to_dict()
    start = perf_counter()
    reason = ""
    outcome: dict[str, Any] = {}
    actual = 0.0
    try:
        result = runtime.execute(deepcopy(action))
        info = result.info
        outcome = {
            key: info.get(key)
            for key in (
                "transaction_status",
                "operation_outcome",
                "actual_duration_s",
                "outcome_diagnostics",
                "rollback_reason",
                "error_message",
                "step",
                "remaining_budget",
                "cost_delta",
                "risk_delta",
                "sample_delta",
            )
        }
        outcome.update(terminated=result.terminated, truncated=result.truncated)
        actual = float(info.get("actual_duration_s") or 0.0)
        if not isfinite(actual) or actual < 0:
            reason = "invalid_actual_duration"
        elif info.get("transaction_status") != "committed":
            reason = "action_not_committed"
        elif info.get("operation_outcome") != "success":
            reason = "action_" + str(info.get("operation_outcome"))
        elif result.terminated or result.truncated:
            reason = "session_terminated_or_truncated"
        elif (
            declared.duration_field is not None
            and declared.duration_field in action
            and not isclose(
                actual, action[declared.duration_field], rel_tol=1e-10, abs_tol=tolerance
            )
        ):
            reason = "action_did_not_complete_declared_duration"
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        reason = "action_exception:" + str(exc)
        outcome = {"exception_type": type(exc).__name__, "failure_reason": str(exc)}
    after = runtime.snapshot().to_dict()
    return TrajectoryStep(
        period_id,
        role,
        action,
        declared.duration_field,
        not reason,
        reason,
        actual,
        perf_counter() - start,
        _json(outcome),
        _ledger_delta(before, after),
        after,
    )


@dataclass(frozen=True)
class TrajectoryDesignResult:
    specification: TrajectoryDesignSpec
    optimization: BoundedDesignResult
    initial_snapshot: dict[str, Any]
    trace: TrajectoryTrace | None
    wall_time_s: float
    prediction_actions: int

    @property
    def success(self) -> bool:
        return self.optimization.success and self.trace is not None

    def to_dict(self) -> dict[str, Any]:
        return _json(
            {
                "schema": "original-session-trajectory-result-v1",
                "success": self.success,
                "specification": self.specification.to_dict(),
                "optimization": self.optimization.to_dict(),
                "selected_trajectory": None if self.trace is None else self.trace.to_dict(),
                "solve_wall_time_s": self.wall_time_s,
                "prediction_action_executions": self.prediction_actions,
                "live_experiment_actions": 0,
                "live_experiment_resource_debit": None,
            }
        )


def optimize_trajectory(
    runtime: ChemWorldRuntime,
    specification: TrajectoryDesignSpec,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> TrajectoryDesignResult:
    """Predict on exact private sessions; retain all trials and fresh verification."""
    initial = runtime.snapshot().to_dict()
    start = perf_counter()
    predictions = 0
    last_trace: TrajectoryTrace | None = None
    policy = specification.trajectory_policy

    def evaluate(values: dict[str, float]) -> DesignPoint:
        nonlocal predictions, last_trace
        candidate: ChemWorldRuntime | None = None
        steps: list[TrajectoryStep] = []
        trial_start = perf_counter()
        try:
            candidate = ChemWorldRuntime.from_snapshot(
                initial, domain_modules=runtime.domain_modules
            )
            for period in specification.periods:
                control_duration = 0.0
                for role, actions in (
                    ("control", period.actions),
                    ("forecast_disturbance", period.forecast_disturbances),
                ):
                    for action in actions:
                        if predictions >= policy.maximum_prediction_actions:
                            raise DesignEvaluationError(
                                "trajectory_prediction_action_budget_exhausted"
                            )
                        requested = (
                            action.compile(values).get(action.duration_field)
                            if action.duration_field
                            else None
                        )
                        elapsed = sum(s.actual_duration_s for s in steps)
                        if (
                            requested is not None
                            and elapsed + requested > policy.maximum_horizon_duration_s
                        ):
                            raise DesignEvaluationError(
                                "trajectory_horizon_duration_budget_exhausted"
                            )
                        predictions += 1
                        step = _execute_action(
                            candidate,
                            action,
                            values,
                            period.period_id,
                            role,
                            policy.duration_tolerance_s,
                        )
                        steps.append(step)
                        if role == "control":
                            control_duration += step.actual_duration_s
                        if not step.success:
                            raise DesignEvaluationError(step.failure_reason)
                        if (
                            elapsed + step.actual_duration_s
                            > policy.maximum_horizon_duration_s + policy.duration_tolerance_s
                        ):
                            raise DesignEvaluationError(
                                "trajectory_horizon_duration_budget_exhausted"
                            )
                if control_duration <= 0:
                    raise DesignEvaluationError("period_has_no_physical_time_progress")
            trace = TrajectoryTrace(deepcopy(initial), tuple(steps), dict(values))
            point = specification.score(deepcopy(trace))
            if not isinstance(point, DesignPoint):
                raise DesignEvaluationError("trajectory scorer must return an original DesignPoint")
            if not isfinite(point.objective) or any(
                not isfinite(value) for value in point.observations.values()
            ):
                raise DesignEvaluationError("nonfinite_trajectory_score")
            if any(c.metric not in point.observations for c in specification.constraints):
                raise DesignEvaluationError("missing_trajectory_constraint_observation")
            try:
                json.dumps(_json(point.diagnostics), allow_nan=False)
            except TypeError as exc:
                raise DesignEvaluationError("score diagnostics require JSON data") from exc
            last_trace = trace
            diagnostics = {
                "trajectory": trace.to_dict(),
                "score": _json(point.diagnostics),
                "objective_unit": specification.objective_unit,
                "trial_wall_time_s": perf_counter() - trial_start,
            }
            return DesignPoint(point.objective, deepcopy(dict(point.observations)), diagnostics)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            trace = TrajectoryTrace(deepcopy(initial), tuple(steps), dict(values))
            raise DesignEvaluationError(
                str(exc),
                {
                    "trajectory": trace.to_dict(),
                    "cause": _json(getattr(exc, "diagnostics", {})),
                    "trial_wall_time_s": perf_counter() - trial_start,
                },
            ) from exc
        finally:
            if candidate is not None:
                candidate.close()

    result = solve_bounded_design(
        specification.variables,
        specification.constraints,
        evaluate,
        starts=specification.starts,
        policy=specification.design_policy,
        progress=progress,
    )
    if runtime.snapshot().to_dict() != initial:
        raise ValueError("live session changed during trajectory planning; no plan can be applied")
    return TrajectoryDesignResult(
        specification,
        result,
        deepcopy(initial),
        last_trace if result.success else None,
        perf_counter() - start,
        predictions,
    )


@dataclass(frozen=True)
class TrajectoryExecutionResult:
    success: bool
    failure_reason: str
    trace: TrajectoryTrace
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure_reason": self.failure_reason,
            "actual_execution": self.trace.to_dict(),
            "execution_wall_time_s": self.wall_time_s,
            "actual_committed_actions": sum(
                s.outcome.get("transaction_status") == "committed" for s in self.trace.steps
            ),
            "prefix_retained": True,
        }


def execute_first_period(
    runtime: ChemWorldRuntime, plan: TrajectoryDesignResult
) -> TrajectoryExecutionResult:
    """Execute only an accepted control prefix; never roll back real elapsed physics."""
    initial, start = runtime.snapshot().to_dict(), perf_counter()
    reason = ""
    steps: list[TrajectoryStep] = []
    values = plan.optimization.values or {}
    if not plan.success or plan.trace is None or not plan.trace.steps:
        reason = "unverified_trajectory_plan"
    elif initial != plan.initial_snapshot:
        reason = "stale_trajectory_plan"
    else:
        period_id = plan.trace.steps[0].period_id
        for expected in plan.trace.steps:
            if expected.period_id != period_id or expected.role != "control":
                break
            action = TrajectoryAction(expected.action, duration_field=expected.duration_field)
            step = _execute_action(
                runtime,
                action,
                {},
                period_id,
                "control",
                plan.specification.trajectory_policy.duration_tolerance_s,
            )
            steps.append(step)
            if not step.success:
                reason = step.failure_reason
                break
        if not reason and sum(s.actual_duration_s for s in steps) <= 0:
            reason = "accepted_control_period_has_no_physical_time_progress"
    return TrajectoryExecutionResult(
        not reason,
        reason,
        TrajectoryTrace(initial, tuple(steps), dict(values)),
        perf_counter() - start,
    )


@dataclass(frozen=True)
class RecedingHorizonResult:
    success: bool
    failure_reason: str
    cycles: tuple[dict[str, Any], ...]
    initial_snapshot: dict[str, Any]
    final_snapshot: dict[str, Any]
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return _json(
            {
                "success": self.success,
                "failure_reason": self.failure_reason,
                "cycles": self.cycles,
                "completed_control_periods": sum(
                    c.get("control_completed", False) for c in self.cycles
                ),
                "wall_time_s": self.wall_time_s,
                "actual_ledger_deltas": _ledger_delta(self.initial_snapshot, self.final_snapshot),
                "actual_resource_ledgers": self.final_snapshot["resource_ledgers"],
                "disturbance_rule": "declared_forecasts_only; actual_disturbance_then_replan",
                "callback_serialization": "description_only; explicit_callables_required",
            }
        )


def run_receding_horizon(
    runtime: ChemWorldRuntime,
    specification: TrajectoryDesignSpec | Callable[[int, dict[str, Any]], TrajectoryDesignSpec],
    *,
    control_periods: int,
    disturbance_actions: Callable[[int, dict[str, Any]], Sequence[TrajectoryAction]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RecedingHorizonResult:
    """Replan from actual state after each control period and external disturbance.

    Disturbance callbacks receive a copy, not the live runtime. They return actual
    original actions, which are executed and accounted here. Their unknown future
    output is never called by prediction. A factory can supply a changed forecast.
    """
    if type(control_periods) is not int or control_periods < 1:
        raise ValueError("receding horizon requires a finite positive number of control periods")
    initial, start = runtime.snapshot().to_dict(), perf_counter()
    cycles: list[dict[str, Any]] = []
    reason = ""
    for index in range(control_periods):
        row: dict[str, Any] = {"index": index, "control_completed": False, "disturbances": []}
        cycles.append(row)
        try:
            spec = (
                specification(index, runtime.snapshot().to_dict())
                if callable(specification)
                else specification
            )
            if not isinstance(spec, TrajectoryDesignSpec):
                raise ValueError("trajectory factory must return a TrajectoryDesignSpec")

            def report(item: dict[str, Any], index: int = index) -> None:
                if progress is not None:
                    progress({"control_period": index, "control_periods": control_periods, **item})

            plan = optimize_trajectory(runtime, spec, progress=report)
            row["planning"] = plan.to_dict()
            if not plan.success or plan.trace is None or not plan.trace.steps:
                reason = "planning_failed:" + plan.optimization.failure_reason
                break
            executed = execute_first_period(runtime, plan)
            row["execution"] = executed.to_dict()
            row["control_completed"] = executed.success
            if not executed.success:
                reason = "execution_failed:" + executed.failure_reason
                break
            if disturbance_actions is not None:
                actions = tuple(disturbance_actions(index, runtime.snapshot().to_dict()))
                if len(actions) > spec.trajectory_policy.maximum_actions_per_trial:
                    raise ValueError(
                        "actual disturbance action list exceeds declared action budget"
                    )
                for action in actions:
                    if not isinstance(action, TrajectoryAction) or action.bindings:
                        raise ValueError(
                            "actual disturbances require explicit unbound original actions"
                        )
                    step = _execute_action(
                        runtime,
                        action,
                        {},
                        f"disturbance:{index}",
                        "actual_disturbance",
                        spec.trajectory_policy.duration_tolerance_s,
                    )
                    row["disturbances"].append(step.to_dict())
                    if not step.success:
                        reason = "disturbance_failed:" + step.failure_reason
                        break
            if reason:
                break
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            reason = str(exc)
            row["failure_reason"] = reason
            break
    return RecedingHorizonResult(
        not reason,
        reason,
        tuple(cycles),
        initial,
        runtime.snapshot().to_dict(),
        perf_counter() - start,
    )
