"""Finite component capabilities; composition is a union, never a kind-set lookup."""

from __future__ import annotations

from chemworld.world.domain_modules import ComponentCapability as ComponentCapability
from chemworld.world.domain_modules import DomainModuleRegistry

COMMON_OPERATIONS = (
    "add_solvent",
    "add_reagent",
    "add_component",
    "sample",
    "quench",
    "create_container",
    "transfer_material",
    "measure",
    "configure_instrument",
    "terminate",
)
NETWORK_OPERATIONS = ("select_vessel", "route_material")
MAX_VESSELS = 16
MAX_CONNECTIONS = 64

COMPONENT_CAPABILITIES = DomainModuleRegistry.default().components()


def component_operations(
    kinds: tuple[str, ...],
    *,
    modules: DomainModuleRegistry | None = None,
    include_common: bool = True,
) -> tuple[str, ...]:
    """Capabilities accumulate independently of the identities of their peers."""
    modules = modules or DomainModuleRegistry.default()
    capabilities = modules.components()
    available = modules.operation_specs()
    declared = {op for kind in kinds for op in capabilities[kind].operations}
    if not declared <= available.keys():
        raise ValueError(
            f"Component operations have no installed owner: {sorted(declared - available.keys())}"
        )
    return tuple(
        dict.fromkeys(
            (
                *(op for op in COMMON_OPERATIONS if include_common and op in available),
                *(op for kind in kinds for op in capabilities[kind].operations),
            )
        )
    )
