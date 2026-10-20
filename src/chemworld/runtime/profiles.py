"""Task-scoped runtime profile contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from chemworld.runtime.domain_service_registry import DomainServiceRegistry
from chemworld.tasks import TaskSpec
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.parameters import WORLD_FAMILY_VERSION


@dataclass(frozen=True)
class TaskRuntimeProfile:
    world_law_id: str
    allowed_operations: frozenset[str]
    required_kernels: frozenset[str]
    optional_kernels: frozenset[str]
    required_domain_services: frozenset[str]
    required_capabilities: frozenset[str]
    allowed_instruments: frozenset[str]

    @classmethod
    def from_task(
        cls,
        task: TaskSpec | None,
        *,
        modules: DomainModuleRegistry | None = None,
        service_registry: DomainServiceRegistry | None = None,
    ) -> TaskRuntimeProfile:
        modules = (
            modules
            or (service_registry.modules if service_registry else None)
            or DomainModuleRegistry.default()
        )
        operation_types = tuple(modules.operation_specs())
        if task is None:
            allowed = frozenset(operation_types)
            instruments: frozenset[str] = frozenset()
        else:
            allowed = frozenset(task.allowed_operations)
            instruments = frozenset(task.allowed_instruments)
        contracts = modules.contracts()
        service_registry = service_registry or DomainServiceRegistry.from_modules(modules)
        required_domain_services = service_registry.service_ids_for_operations(allowed)
        capabilities = frozenset(
            contracts[operation].module for operation in allowed if operation in contracts
        )
        return cls(
            world_law_id=(WORLD_FAMILY_VERSION if task is None else task.world_law_id),
            allowed_operations=allowed,
            required_kernels=allowed,
            optional_kernels=frozenset(operation_types) - allowed,
            required_domain_services=required_domain_services,
            required_capabilities=capabilities,
            allowed_instruments=instruments,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "world_law_id": self.world_law_id,
            "allowed_operations": sorted(self.allowed_operations),
            "required_kernels": sorted(self.required_kernels),
            "optional_kernels": sorted(self.optional_kernels),
            "required_domain_services": sorted(self.required_domain_services),
            "required_capabilities": sorted(self.required_capabilities),
            "allowed_instruments": sorted(self.allowed_instruments),
        }
        payload["profile_hash"] = self.profile_hash
        return payload

    @property
    def profile_hash(self) -> str:
        payload = {
            "world_law_id": self.world_law_id,
            "allowed_operations": sorted(self.allowed_operations),
            "required_kernels": sorted(self.required_kernels),
            "optional_kernels": sorted(self.optional_kernels),
            "required_domain_services": sorted(self.required_domain_services),
            "required_capabilities": sorted(self.required_capabilities),
            "allowed_instruments": sorted(self.allowed_instruments),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def is_operation_allowed(self, operation_type: str) -> bool:
        return operation_type in self.allowed_operations

    def is_domain_service_required(self, service_id: str) -> bool:
        return service_id in self.required_domain_services


__all__ = ["TaskRuntimeProfile"]
