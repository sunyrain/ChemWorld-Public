"""Operation-kernel contracts for the transactional runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any, Protocol

from chemworld.foundation import OperationRecord, WorldState
from chemworld.foundation.balances import TransitionBalance
from chemworld.operation_validator import OperationValidation
from chemworld.physchem.maturity import ModelProviderContract
from chemworld.runtime.mechanisms import CompiledMechanism
from chemworld.runtime.profiles import TaskRuntimeProfile
from chemworld.runtime.transactions import StatePatch, TransactionManager, WorldEvent
from chemworld.tasks import TaskSpec

if TYPE_CHECKING:
    from chemworld.runtime.domain_services import ChemWorldDomainServices

OPERATION_OUTCOMES = frozenset({
    "success", "rejected", "numerical_failure", "process_failure", "stopped",
})


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


@dataclass(frozen=True)
class ModelProviderResult:
    """Provider-neutral result used by independently implemented physical models."""

    outputs: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    success: bool = True
    failure_reason: str | None = None
    provenance: tuple[str, ...] = ()
    outcome: str | None = None
    state: WorldState | None = None
    actual_duration_s: float = 0.0
    measurement_state: WorldState | None = None

    def __post_init__(self) -> None:
        outcome = self.outcome or ("success" if self.success else "numerical_failure")
        if outcome not in OPERATION_OUTCOMES:
            raise ValueError(f"unknown operation outcome: {outcome}")
        object.__setattr__(self, "outcome", outcome)
        if not isfinite(self.actual_duration_s) or self.actual_duration_s < 0.0:
            raise ValueError("actual_duration_s must be finite and nonnegative")
        if outcome in {"rejected", "numerical_failure"} and self.actual_duration_s != 0.0:
            raise ValueError("unexecuted provider result cannot report elapsed physical time")
        if self.success != (outcome in {"success", "stopped"}):
            raise ValueError("provider success flag and outcome disagree")
        if self.success and self.failure_reason is not None:
            raise ValueError("successful provider result cannot include failure_reason")
        if not self.success and not str(self.failure_reason or "").strip():
            raise ValueError("failed provider result requires failure_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": dict(self.outputs),
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
            "success": self.success,
            "failure_reason": self.failure_reason,
            "provenance": list(self.provenance),
            "outcome": self.outcome,
            "actual_duration_s": self.actual_duration_s,
        }


class PhysicalModelProvider(Protocol):
    """Minimum contract that parallel model teams implement before runtime wiring."""

    @property
    def model_contract(self) -> ModelProviderContract:
        ...

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        """Return domain violations; an empty tuple means the input is supported."""
        ...

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        ...


@dataclass(frozen=True)
class ContractModelProviderStub:
    """Deterministic contract fixture for model teams before runtime integration.

    The stub deliberately performs no physical calculation. It verifies the
    provider result shape and makes that limitation visible in every result.
    """

    model_contract: ModelProviderContract
    outputs: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...] = (
        "contract stub only; no physical model evaluation was performed",
    )

    def __post_init__(self) -> None:
        self._require_exact_fields(
            label="outputs",
            expected=self.model_contract.output_fields,
            actual=self.outputs,
        )
        self._require_exact_fields(
            label="diagnostics",
            expected=self.model_contract.diagnostic_fields,
            actual=self.diagnostics,
        )
        object.__setattr__(self, "outputs", dict(self.outputs))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @staticmethod
    def _require_exact_fields(
        *,
        label: str,
        expected: tuple[str, ...],
        actual: Mapping[str, Any],
    ) -> None:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"stub {label} do not match provider contract: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            f"missing required input: {field_name}"
            for field_name in self.model_contract.input_fields
            if field_name not in inputs
        )

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return ModelProviderResult(
                outputs={},
                diagnostics=dict.fromkeys(
                    self.model_contract.diagnostic_fields,
                    None,
                ),
                warnings=self.warnings,
                success=False,
                failure_reason="; ".join(violations),
                provenance=(*self.model_contract.provenance, "contract-stub"),
            )
        return ModelProviderResult(
            outputs=dict(self.outputs),
            diagnostics=dict(self.diagnostics),
            warnings=self.warnings,
            provenance=(*self.model_contract.provenance, "contract-stub"),
        )


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
