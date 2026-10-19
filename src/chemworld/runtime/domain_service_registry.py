"""Domain-service contracts and task-scoped registry validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from chemworld.world.control_contract import CONTROL_OPERATIONS
from chemworld.world.operations import OPERATION_TYPES
from chemworld.world.stream_contract import STREAM_OPERATIONS


@dataclass(frozen=True)
class DomainServiceContract:
    service_id: str
    module: str
    service_class: str
    operations: tuple[str, ...]
    capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "module": self.module,
            "service_class": self.service_class,
            "operations": list(self.operations),
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class DomainServiceRegistry:
    contracts: tuple[DomainServiceContract, ...]

    @classmethod
    def default(cls) -> DomainServiceRegistry:
        return cls(
            (
                DomainServiceContract(
                    "primitive_operations",
                    "reaction",
                    "ChemWorldPrimitiveOperationServices",
                    (
                        "add_reagent",
                        "add_component",
                        "add_solvent",
                        "add_catalyst",
                        "sample",
                        "quench",
                        "create_container",
                        "transfer_material",
                        "select_vessel",
                        "route_material",
                        "evaporate",
                        "terminate",
                    ),
                    ("reaction",),
                ),
                DomainServiceContract(
                    "reaction_thermal",
                    "reaction",
                    "ChemWorldReactionThermalServices",
                    ("heat", "wait", *CONTROL_OPERATIONS),
                    ("reaction",),
                ),
                DomainServiceContract(
                    "phase_separation",
                    "separation",
                    "ChemWorldPhaseSeparationServices",
                    (
                        "add_phase",
                        "add_extractant",
                        "mix",
                        "settle",
                        "separate_phase",
                        "wash",
                        "dry",
                        "concentrate",
                        "transfer",
                    ),
                    ("separation",),
                ),
                DomainServiceContract(
                    "crystallization",
                    "crystallization",
                    "ChemWorldCrystallizationServices",
                    ("seed_crystals", "cool_crystallize", "filter_crystals", "resuspend_crystals"),
                    ("crystallization",),
                ),
                DomainServiceContract(
                    "distillation",
                    "distillation",
                    "ChemWorldDistillationServices",
                    ("distill", "collect_fraction"),
                    ("distillation",),
                ),
                DomainServiceContract(
                    "continuous_flow",
                    "continuous_flow",
                    "ChemWorldFlowServices",
                    ("set_flow_rate", "run_flow", *STREAM_OPERATIONS),
                    ("continuous_flow",),
                ),
                DomainServiceContract(
                    "electrochemistry",
                    "electrochemistry",
                    "ChemWorldElectrochemicalServices",
                    ("set_potential", "electrolyze"),
                    ("electrochemistry",),
                ),
                DomainServiceContract(
                    "instrument_cost",
                    "observation",
                    "ChemWorldInstrumentCostServices",
                    ("measure", "configure_instrument"),
                    ("observation",),
                ),
            )
        )

    def validate_contract_integrity(self) -> None:
        owners: dict[str, str] = {}
        duplicates: list[str] = []
        for contract in self.contracts:
            for operation in contract.operations:
                if operation in owners:
                    duplicates.append(operation)
                owners[operation] = contract.service_id
        unknown = sorted(set(owners) - set(OPERATION_TYPES))
        if duplicates or unknown:
            details = {
                "duplicates": sorted(duplicates),
                "unknown": unknown,
            }
            raise ValueError(f"Invalid domain service registry contract: {details}")

    def validate_operation_coverage(self, operations: Iterable[str] | None = None) -> None:
        self.validate_contract_integrity()
        owners = set(self.operation_map())
        expected = set(OPERATION_TYPES) if operations is None else set(operations)
        missing = sorted(expected - owners)
        if missing:
            raise ValueError(
                f"Invalid domain service registry operation coverage: {{'missing': {missing}}}"
            )

    def service_id_for_operation(self, operation: str) -> str:
        return self.contract_for_operation(operation).service_id

    def contract_for_operation(self, operation: str) -> DomainServiceContract:
        for contract in self.contracts:
            if operation in contract.operations:
                return contract
        raise ValueError(f"No domain service registered for operation {operation!r}")

    def service_ids_for_operations(self, operations: Iterable[str]) -> frozenset[str]:
        operation_map = self.operation_map()
        requested = frozenset(operations)
        missing = sorted(requested - set(operation_map))
        if missing:
            raise ValueError(f"Missing domain service operation coverage: {missing}")
        return frozenset(operation_map[operation] for operation in requested)

    def capabilities_for_services(self, service_ids: Iterable[str]) -> frozenset[str]:
        requested = frozenset(service_ids)
        contracts = {contract.service_id: contract for contract in self.contracts}
        missing = sorted(requested - set(contracts))
        if missing:
            raise ValueError(f"Missing required domain services: {missing}")
        return frozenset(
            capability
            for service_id in requested
            for capability in contracts[service_id].capabilities
        )

    def validate_profile(self, profile: Any) -> None:
        required_operations = frozenset(getattr(profile, "required_kernels", ()))
        required_services = frozenset(
            getattr(
                profile,
                "required_domain_services",
                self.service_ids_for_operations(required_operations),
            )
        )
        operation_services = self.service_ids_for_operations(required_operations)
        missing_services = sorted(required_services - operation_services)
        if missing_services:
            raise ValueError(
                f"Profile requires domain services not used by its operations: {missing_services}"
            )

        service_capabilities = self.capabilities_for_services(required_services)
        required_capabilities = frozenset(getattr(profile, "required_capabilities", ()))
        missing_capabilities = sorted(required_capabilities - service_capabilities)
        if missing_capabilities:
            raise ValueError(
                f"Missing required domain service capabilities: {missing_capabilities}"
            )

    def operation_map(self) -> dict[str, str]:
        return {
            operation: contract.service_id
            for contract in self.contracts
            for operation in contract.operations
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "services": {contract.service_id: contract.to_dict() for contract in self.contracts},
            "operation_service_map": self.operation_map(),
        }


__all__ = ["DomainServiceContract", "DomainServiceRegistry"]
