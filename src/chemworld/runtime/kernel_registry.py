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
from chemworld.world.control_contract import CONTROL_OPERATIONS
from chemworld.world.operations import OPERATION_TYPES, operation_contracts
from chemworld.world.stream_contract import STREAM_OPERATIONS


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
        del state, action, context
        return KernelPlan(
            operation_type=self.operation_type,
            kernel_id=self.kernel_id,
            affected_ledgers=affected_ledgers(self.operation_type),
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
        affected = affected_ledgers(self.operation_type)
        next_state, record = context.domain_services.apply_operation(state, action)
        failed_preconditions = tuple(
            key for key, passed in record.preconditions.items() if not passed
        )
        if failed_preconditions:
            rejection_event = WorldEvent(
                event_type="operation_rejected",
                operation_type=self.operation_type,
                payload={
                    "kernel_id": self.kernel_id,
                    "domain_service_id": domain_service_id,
                    "affected_ledgers": list(affected),
                    "failed_preconditions": list(failed_preconditions),
                },
            )
            transaction = context.transaction_manager.rollback(
                state=before,
                operation_type=self.operation_type,
                rollback_reason="precondition_failed",
                failed_preconditions=failed_preconditions,
                events=(rejection_event,),
            )
            record = context.domain_services.record_operation(
                self.operation_type,
                before,
                transaction.state,
                record.preconditions,
                action,
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
            )
        patch = StatePatch(
            patch_type="replace_state",
            affected_ledgers=affected,
            state=next_state,
            summary=record.state_delta_summary,
        )
        event = WorldEvent(
            event_type="operation_applied",
            operation_type=self.operation_type,
            payload={
                "kernel_id": self.kernel_id,
                "domain_service_id": domain_service_id,
                "affected_ledgers": list(affected),
            },
        )
        transaction = context.transaction_manager.commit(
            state=before,
            operation_type=self.operation_type,
            events=(event,),
            patches=(patch,),
        )
        if transaction.transaction_status != "committed":
            record = context.domain_services.record_operation(
                self.operation_type,
                before,
                transaction.state,
                record.preconditions,
                action,
            )
            if record.is_instrument_measurement:
                record = replace(
                    record,
                    measurement_cost=0.0,
                    sample_consumed_L=0.0,
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
            affected_ledgers=affected,
            kernel_id=self.kernel_id,
            kernel_version=self.kernel_version,
            transaction_status=transaction.transaction_status,
            rollback_reason=transaction.rollback_reason,
        )


class OperationKernelRegistry:
    def __init__(self, kernels: list[OperationKernel]) -> None:
        self._kernels = {kernel.operation_type: kernel for kernel in kernels}
        if len(self._kernels) != len(kernels):
            raise ValueError("Duplicate operation kernels are not allowed")

    @classmethod
    def default(cls) -> OperationKernelRegistry:
        contracts = operation_contracts()
        return cls(
            [
                ServiceOperationKernel(
                    operation_type=operation_type,
                    module=contracts[operation_type].module
                    if operation_type in contracts
                    else "general",
                )
                for operation_type in OPERATION_TYPES
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


def affected_ledgers(operation_type: str) -> tuple[str, ...]:
    material = {
        "add_reagent",
        "add_component",
        "add_catalyst",
        "add_solvent",
        "sample",
        "heat",
        "wait",
        "add_phase",
        "add_extractant",
        "mix",
        "settle",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
        "seed_crystals",
        "cool_crystallize",
        "filter_crystals",
        "resuspend_crystals",
        "evaporate",
        "distill",
        "collect_fraction",
        "run_flow",
        "electrolyze",
    }
    affected = ["process"]
    if operation_type in material:
        affected.append("species")
        affected.append("phases")
    if operation_type in {
        "add_solvent",
        "heat",
        "wait",
        "quench",
        "evaporate",
        "distill",
        "run_flow",
    }:
        affected.append("thermal")
        affected.append("vessels")
    if operation_type in {
        "add_solvent",
        "add_catalyst",
        "heat",
        "wait",
        "mix",
        "seed_crystals",
        "set_flow_rate",
        "set_potential",
        "electrolyze",
        "distill",
        "measure",
    }:
        affected.append("equipment")
    if operation_type == "measure":
        affected.append("observation")
    if operation_type == "resuspend_crystals":
        affected.extend(("samples", "vessels", "equipment"))
    if operation_type in {"create_container", "transfer_material"}:
        affected.extend(("samples", "species", "phases", "vessels", "equipment"))
    if operation_type in {"select_vessel", "route_material"}:
        affected.extend(
            ("inactive_vessels", "samples", "species", "phases", "vessels", "equipment", "thermal")
        )
    if operation_type in CONTROL_OPERATIONS:
        affected.extend(("equipment", "species", "phases", "thermal", "process"))
    if operation_type in STREAM_OPERATIONS:
        affected.extend(
            ("inactive_vessels", "samples", "equipment", "species", "phases", "vessels", "thermal")
        )
    if operation_type == "configure_instrument":
        affected.append("equipment")
    return tuple(dict.fromkeys(affected))


__all__ = [
    "OperationKernelRegistry",
    "ServiceOperationKernel",
    "affected_ledgers",
]
