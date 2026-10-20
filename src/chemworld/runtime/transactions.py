"""Transactional state updates for ChemWorld runtime v2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import isclose, isfinite
from typing import Any

from chemworld.foundation import PhysicalConstitution, WorldState
from chemworld.foundation.balances import TransitionBalance
from chemworld.foundation.constitution import CheckResult


@dataclass(frozen=True)
class WorldEvent:
    event_type: str
    operation_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "operation_type": self.operation_type,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class StatePatch:
    patch_type: str
    affected_ledgers: tuple[str, ...]
    state: WorldState | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def apply(self, state: WorldState) -> WorldState:
        if self.patch_type in {"replace_state", "rollback_penalty"} and self.state is not None:
            return self.state
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_type": self.patch_type,
            "affected_ledgers": list(self.affected_ledgers),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class TransactionResult:
    state: WorldState
    events: tuple[WorldEvent, ...]
    patches: tuple[StatePatch, ...]
    transaction_status: str
    rollback_reason: str | None
    constitution_checks: tuple[dict[str, object], ...]
    material_balance: TransitionBalance = field(default_factory=TransitionBalance)

    def to_info(self) -> dict[str, Any]:
        return {
            "world_events": [event.to_dict() for event in self.events],
            "state_patches_summary": [patch.to_dict() for patch in self.patches],
            "transaction_status": self.transaction_status,
            "rollback_reason": self.rollback_reason,
        }


class TransactionManager:
    """Apply patches atomically and roll back failed constitution checks."""

    def __init__(
        self, constitution: PhysicalConstitution,
        material_boundary: Callable[
            [WorldState, WorldState, dict[str, Any]], TransitionBalance
        ] | None = None,
    ) -> None:
        self.constitution = constitution
        self.material_boundary = material_boundary

    def commit(
        self,
        *,
        state: WorldState,
        operation_type: str,
        events: tuple[WorldEvent, ...],
        patches: tuple[StatePatch, ...],
        action: dict[str, Any] | None = None,
        actual_duration_s: float | None = None,
    ) -> TransactionResult:
        candidate = state
        for patch in patches:
            candidate = patch.apply(candidate)
        report = self.constitution.check_state(candidate)
        balance = TransitionBalance()
        transition_checks: list[CheckResult] = []
        if actual_duration_s is not None:
            transition_checks.append(CheckResult(
                "operation_actual_duration",
                isfinite(actual_duration_s) and actual_duration_s >= 0.0 and isclose(
                    candidate.ledger.time_s - state.ledger.time_s,
                    actual_duration_s, rel_tol=1e-9, abs_tol=1e-7,
                ),
                "candidate physical clock agrees with the provider's actual elapsed time",
            ))
        try:
            if self.material_boundary is not None and action is not None:
                if action.get("operation") != operation_type:
                    raise ValueError("action and transaction operation disagree")
                balance = self.material_boundary(state, candidate, action)
            transition_checks.append(self.constitution.check_transition(state, candidate, balance))
            transition_checks.append(
                self.constitution.check_carrier_transition(state, candidate, balance)
            )
        except ValueError as exc:
            transition_checks.append(CheckResult("boundary_exchange_integrity", False, str(exc)))
        if operation_type in {"quench", "cool_crystallize"}:
            thermal = candidate.metadata.get("last_energy_transition", {})
            residual = (
                float(thermal.get("energy_balance_residual_J", float("inf")))
                if isinstance(thermal, dict) and thermal.get("operation") == operation_type
                else float("inf")
            )
            transition_checks.append(CheckResult(
                "operation_energy_conservation", abs(residual) <= 1e-8,
                "signed sensible, jacket, environmental, and phase-change heat close",
                value=residual, tolerance=1e-8,
            ))
        checks = tuple(check.to_dict() for check in (*report.checks, *transition_checks))
        if all(check["passed"] for check in checks):
            return TransactionResult(
                state=candidate,
                events=events,
                patches=patches,
                transaction_status="committed",
                rollback_reason=None,
                constitution_checks=checks,
                material_balance=balance,
            )
        penalty_state = self._penalize_rollback(state)
        failed_checks = [check["name"] for check in checks if not check["passed"]]
        rollback_event = WorldEvent(
            "transaction_rollback",
            operation_type,
            {"failed_checks": failed_checks},
        )
        rollback_patch = StatePatch(
            patch_type="rollback_penalty",
            affected_ledgers=("process",),
            state=penalty_state,
            summary={
                "delta_cost": penalty_state.ledger.cost - state.ledger.cost,
                "delta_risk": penalty_state.ledger.risk - state.ledger.risk,
                "delta_sample_consumed_L": (
                    penalty_state.ledger.sample_consumed_L
                    - state.ledger.sample_consumed_L
                ),
                "failed_checks": failed_checks,
            },
        )
        return TransactionResult(
            state=penalty_state,
            events=(*events, rollback_event),
            patches=(*patches, rollback_patch),
            transaction_status="rolled_back",
            rollback_reason="constitution_failed",
            constitution_checks=checks,
            material_balance=balance,
        )

    def rollback(
        self,
        *,
        state: WorldState,
        operation_type: str,
        rollback_reason: str,
        failed_preconditions: tuple[str, ...] = (),
        failed_checks: tuple[str, ...] = (),
        events: tuple[WorldEvent, ...] = (),
    ) -> TransactionResult:
        """Rollback an action before candidate-state mutation is committed."""

        penalty_state = self._penalize_rollback(state)
        payload: dict[str, Any] = {"rollback_reason": rollback_reason}
        if failed_preconditions:
            payload["failed_preconditions"] = list(failed_preconditions)
        if failed_checks:
            payload["failed_checks"] = list(failed_checks)
        rollback_event = WorldEvent("transaction_rollback", operation_type, payload)
        rollback_patch = StatePatch(
            patch_type="rollback_penalty",
            affected_ledgers=("process",),
            state=penalty_state,
            summary={
                "delta_cost": penalty_state.ledger.cost - state.ledger.cost,
                "delta_risk": penalty_state.ledger.risk - state.ledger.risk,
                "delta_sample_consumed_L": (
                    penalty_state.ledger.sample_consumed_L
                    - state.ledger.sample_consumed_L
                ),
                "failed_preconditions": list(failed_preconditions),
                "failed_checks": list(failed_checks),
            },
        )
        checks = tuple(check.to_dict() for check in self.constitution.check_state(state).checks)
        return TransactionResult(
            state=penalty_state,
            events=(*events, rollback_event),
            patches=(rollback_patch,),
            transaction_status="rolled_back",
            rollback_reason=rollback_reason,
            constitution_checks=checks,
        )

    @staticmethod
    def _penalize_rollback(state: WorldState) -> WorldState:
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + 0.03,
            risk=min(1.0, state.ledger.risk + 0.08),
        )
        process = None
        if state.process is not None:
            process = replace(
                state.process,
                cost=ledger.cost,
                risk=ledger.risk,
            )
        return state.replace(ledger=ledger, process=process)


__all__ = ["StatePatch", "TransactionManager", "TransactionResult", "WorldEvent"]
