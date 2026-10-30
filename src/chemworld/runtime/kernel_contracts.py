"""Operation-kernel contracts for the transactional runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any, Protocol

from chemworld.foundation import OperationRecord, WorldState
from chemworld.foundation.balances import TransitionBalance
from chemworld.foundation.outcomes import OPERATION_OUTCOMES
from chemworld.operation_validator import OperationValidation
from chemworld.physchem.model_provider import (
    ContractModelProviderStub as ContractModelProviderStub,
)
from chemworld.physchem.model_provider import (
    ModelProviderResult as ModelProviderResult,
)
from chemworld.physchem.model_provider import (
    PhysicalModelProvider as PhysicalModelProvider,
)
from chemworld.runtime.mechanisms import CompiledMechanism
from chemworld.runtime.profiles import TaskRuntimeProfile
from chemworld.runtime.transactions import StatePatch, TransactionManager, WorldEvent
from chemworld.tasks import TaskSpec

if TYPE_CHECKING:
    from chemworld.runtime.domain_services import ChemWorldDomainServices




@dataclass(frozen=True)
class RuntimeContext:
    task_spec: TaskSpec | None
    profile: TaskRuntimeProfile
    compiled_mechanism: CompiledMechanism
    domain_services: ChemWorldDomainServices
    transaction_manager: TransactionManager
    debug_truth: bool = False


@dataclass(frozen=True)
class KernelPlan:
    operation_type: str
    kernel_id: str
    affected_ledgers: tuple[str, ...]
    required_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "kernel_id": self.kernel_id,
            "affected_ledgers": list(self.affected_ledgers),
            "required_capabilities": list(self.required_capabilities),
        }


@dataclass(frozen=True)
class KernelResult:
    state: WorldState
    operation_record: OperationRecord
    events: tuple[WorldEvent, ...]
    patches: tuple[StatePatch, ...]
    state_delta_summary: dict[str, Any]
    cost_delta: float
    risk_delta: float
    sample_delta: float
    affected_ledgers: tuple[str, ...]
    kernel_id: str
    kernel_version: str
    transaction_status: str
    rollback_reason: str | None
    outcome: str | None = None
    actual_duration_s: float = 0.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    measurement_status: str | None = None
    # Transient evaluator-side specimen; never serialized into public diagnostics.
    measurement_state: WorldState | None = None

    def __post_init__(self) -> None:
        outcome = self.outcome
        if outcome is None:
            outcome = "success" if self.transaction_status == "committed" else (
                "numerical_failure" if self.rollback_reason == "constitution_failed" else "rejected"
            )
        if outcome not in OPERATION_OUTCOMES:
            raise ValueError(f"unknown operation outcome: {outcome}")
        if not isfinite(self.actual_duration_s) or self.actual_duration_s < 0.0:
            raise ValueError("actual_duration_s must be finite and nonnegative")
        if outcome in {"rejected", "numerical_failure"} and self.actual_duration_s != 0.0:
            raise ValueError("rolled back operation cannot report elapsed physical time")
        object.__setattr__(self, "outcome", outcome)

    @property
    def material_balance(self) -> TransitionBalance:
        """The transaction's single exchange record, without copying receipts."""
        return self.operation_record.material_balance

    @property
    def actual_consumption(self) -> dict[str, float]:
        return {
            "cost": self.cost_delta,
            "sample_consumed_L": self.sample_delta,
            "physical_time_s": self.actual_duration_s,
        }








class OperationKernel(Protocol):
    @property
    def operation_type(self) -> str:
        ...

    @property
    def kernel_id(self) -> str:
        ...

    @property
    def kernel_version(self) -> str:
        ...

    @property
    def required_capabilities(self) -> frozenset[str]:
        ...

    def validate(
        self,
        state: WorldState,
        action: dict[str, Any],
        context: RuntimeContext,
    ) -> OperationValidation | None:
        ...

    def plan(
        self,
        state: WorldState,
        action: dict[str, Any],
        context: RuntimeContext,
    ) -> KernelPlan:
        ...

    def apply(
        self,
        state: WorldState,
        action: dict[str, Any],
        context: RuntimeContext,
    ) -> KernelResult:
        ...


__all__ = [
    "ContractModelProviderStub",
    "KernelPlan",
    "KernelResult",
    "ModelProviderResult",
    "OperationKernel",
    "PhysicalModelProvider",
    "RuntimeContext",
]
