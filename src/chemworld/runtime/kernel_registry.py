"""Operation-kernel registry and default service-backed kernel."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from chemworld.foundation import WorldState
from chemworld.operation_validator import OperationValidation
from chemworld.runtime.kernel_contracts import (
    KernelPlan,
    KernelResult,
    OperationKernel,
    RuntimeContext,
)
from chemworld.runtime.profiles import TaskRuntimeProfile
from chemworld.runtime.transactions import StatePatch, WorldEvent
from chemworld.world.domain_modules import DomainModuleRegistry


@dataclass(frozen=True)
class ServiceOperationKernel:
    operation_type: str
    module: str
    kernel_version: str = "runtime-v2.0"

    @property
    def kernel_id(self) -> str:
        return f"chemworld.operation.{self.operation_type}"

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset({self.module})

    def validate(
        self,
        state: WorldState,
        action: dict[str, Any],
        context: RuntimeContext,
    ) -> OperationValidation | None:
        del state, action, context
        return None

    def plan(
        self,
        state: WorldState,
        action: dict[str, Any],
        context: RuntimeContext,
    ) -> KernelPlan:
        del state, action
        return KernelPlan(
            operation_type=self.operation_type,
            kernel_id=self.kernel_id,
            affected_ledgers=affected_ledgers(
                self.operation_type, modules=context.domain_services.modules
            ),
            required_capabilities=tuple(sorted(self.required_capabilities)),
        )

    def apply(
        self,
        state: WorldState,
        action: dict[str, Any],
        context: RuntimeContext,
    ) -> KernelResult:
        before = state
        domain_service_id = context.domain_services.service_id_for_operation(self.operation_type)
        affected = affected_ledgers(self.operation_type, modules=context.domain_services.modules)
        provider = context.domain_services.evaluate_operation(state, action)
        preconditions = dict(provider.outputs["preconditions"])
        failed_preconditions = tuple(key for key, passed in preconditions.items() if not passed)
        next_state = before if provider.state is None else provider.state
        diagnostics = dict(provider.diagnostics)
        if provider.failure_reason is not None:
            diagnostics["failure_reason"] = provider.failure_reason
        if provider.outcome in {"rejected", "numerical_failure"}:
            rejection_event = WorldEvent(
                event_type=f"operation_{provider.outcome}",
                operation_type=self.operation_type,
                payload={
                    "kernel_id": self.kernel_id,
                    "domain_service_id": domain_service_id,
                    "affected_ledgers": list(affected),
                    "failed_preconditions": list(failed_preconditions),
                    "diagnostics": diagnostics,
                },
            )
            transaction = context.transaction_manager.rollback(
                state=before,
                operation_type=self.operation_type,
                rollback_reason=(
                    "precondition_failed" if failed_preconditions else str(provider.outcome)
                ),
                failed_preconditions=failed_preconditions,
                events=(rejection_event,),
            )
            record = context.domain_services.record_operation(
                self.operation_type,
                before,
                transaction.state,
                preconditions,
                action,
                constitution_checks=transaction.constitution_checks,
            )
            return KernelResult(
                state=transaction.state,
                operation_record=record,
                events=transaction.events,
                patches=transaction.patches,
                state_delta_summary=record.state_delta_summary,
                cost_delta=transaction.state.ledger.cost - before.ledger.cost,
                risk_delta=transaction.state.ledger.risk - before.ledger.risk,
                sample_delta=(
                    transaction.state.ledger.sample_consumed_L - before.ledger.sample_consumed_L
                ),
                affected_ledgers=("process",),
                kernel_id=self.kernel_id,
                kernel_version=self.kernel_version,
                transaction_status=transaction.transaction_status,
                rollback_reason=transaction.rollback_reason,
                outcome=str(provider.outcome),
                diagnostics=diagnostics,
            )
        patch = StatePatch(
            patch_type="replace_state",
            affected_ledgers=affected,
            state=next_state,
            summary=context.domain_services.operation_recorder.state_delta_summary(
                self.operation_type,
                before,
                next_state,
            ),
        )
        event = WorldEvent(
            event_type="operation_applied",
            operation_type=self.operation_type,
            payload={
                "kernel_id": self.kernel_id,
                "domain_service_id": domain_service_id,
                "affected_ledgers": list(affected),
                "outcome": provider.outcome,
                "actual_duration_s": provider.actual_duration_s,
            },
        )
        transaction = context.transaction_manager.commit(
            state=before,
            operation_type=self.operation_type,
            events=(event,),
            patches=(patch,),
            action=action,
            actual_duration_s=provider.actual_duration_s,
        )
        record = context.domain_services.record_operation(
            self.operation_type,
            before,
            transaction.state,
            preconditions,
            action,
            constitution_checks=transaction.constitution_checks,
            material_balance=transaction.material_balance,
        )
        if transaction.transaction_status != "committed" and record.is_instrument_measurement:
            record = replace(record, measurement_cost=0.0, sample_consumed_L=0.0)
        return KernelResult(
            state=transaction.state,
            operation_record=record,
            events=transaction.events,
            patches=transaction.patches,
            state_delta_summary=record.state_delta_summary,
            cost_delta=transaction.state.ledger.cost - before.ledger.cost,
            risk_delta=transaction.state.ledger.risk - before.ledger.risk,
            sample_delta=(
                transaction.state.ledger.sample_consumed_L - before.ledger.sample_consumed_L
            ),
            affected_ledgers=(
                affected if transaction.transaction_status == "committed" else ("process",)
            ),
            kernel_id=self.kernel_id,
            kernel_version=self.kernel_version,
            transaction_status=transaction.transaction_status,
            rollback_reason=transaction.rollback_reason,
            outcome=(
                str(provider.outcome)
                if transaction.transaction_status == "committed"
                else "numerical_failure"
            ),
            actual_duration_s=transaction.state.ledger.time_s - before.ledger.time_s,
            diagnostics=diagnostics,
            measurement_status=(
                "pending"
                if transaction.transaction_status == "committed"
                and record.is_instrument_measurement
                else None
            ),
            measurement_state=(
                provider.measurement_state
                if transaction.transaction_status == "committed" else None
            ),
        )


class OperationKernelRegistry:
    def __init__(self, kernels: list[OperationKernel]) -> None:
        self._kernels = {kernel.operation_type: kernel for kernel in kernels}
        if len(self._kernels) != len(kernels):
            raise ValueError("Duplicate operation kernels are not allowed")

    @classmethod
    def default(cls, *, modules: DomainModuleRegistry | None = None) -> OperationKernelRegistry:
        modules = modules or DomainModuleRegistry.default()
        contracts = modules.contracts()
        return cls(
            [
                ServiceOperationKernel(
                    operation_type=operation_type,
                    module=contracts[operation_type].module
                    if operation_type in contracts
                    else "general",
                )
                for operation_type in modules.operation_order()
                if operation_type in contracts
            ]
        )

    def validate_profile(self, profile: TaskRuntimeProfile) -> None:
        missing = sorted(profile.required_kernels - set(self._kernels))
        if missing:
            raise ValueError(f"Missing required operation kernels: {missing}")
        kernel_capabilities = frozenset(
            capability
            for operation in profile.required_kernels
            for capability in self._kernels[operation].required_capabilities
        )
        missing_capabilities = sorted(profile.required_capabilities - kernel_capabilities)
        if missing_capabilities:
            raise ValueError(
                f"Missing required operation kernel capabilities: {missing_capabilities}"
            )

    def has(self, operation_type: str) -> bool:
        return operation_type in self._kernels

    def get(self, operation_type: str) -> OperationKernel:
        try:
            return self._kernels[operation_type]
        except KeyError as exc:
            raise ValueError(f"No operation kernel registered for {operation_type!r}") from exc

    def kernel_id_for_operation(self, operation_type: str) -> str:
        return self.get(operation_type).kernel_id

    def to_dict(self) -> dict[str, Any]:
        return {
            operation_type: {
                "kernel_id": kernel.kernel_id,
                "kernel_version": kernel.kernel_version,
                "required_capabilities": sorted(kernel.required_capabilities),
            }
            for operation_type, kernel in sorted(self._kernels.items())
        }


def affected_ledgers(operation_type: str, *, modules=None) -> tuple[str, ...]:
    spec = (modules or DomainModuleRegistry.default()).operation_specs().get(operation_type)
    return ("process",) if spec is None else spec.affected_ledgers


__all__ = [
    "OperationKernelRegistry",
    "ServiceOperationKernel",
    "affected_ledgers",
]
