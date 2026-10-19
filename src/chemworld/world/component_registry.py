"""Finite component capabilities; composition is a union, never a kind-set lookup."""

from __future__ import annotations

from dataclasses import dataclass

from chemworld.world.control_contract import CONTROL_OPERATIONS
from chemworld.world.operations import (
    CRYSTALLIZATION_OPERATIONS,
    DISTILLATION_OPERATIONS,
    ELECTROCHEMISTRY_OPERATIONS,
    FLOW_OPERATIONS,
    SEPARATION_OPERATIONS,
)


@dataclass(frozen=True)
class ComponentCapability:
    operations: tuple[str, ...]
    template: str | None = None
    priority: int = 0
    minimum_operations: int = 3
    minimum_measurements: int = 0
    minimum_time_s: float = 0.0
    outputs: tuple[str, ...] = ("out",)
    reactive_template: str | None = None


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

COMPONENT_CAPABILITIES = {
    "reaction": ComponentCapability(
        ("add_reagent", "add_component", "add_solvent", "add_catalyst"),
        "reaction-to-assay",
        20,
        4,
    ),
    "thermal": ComponentCapability(("heat", "wait", *CONTROL_OPERATIONS)),
    "phase": ComponentCapability(("add_phase",), "equilibrium-characterization", 10),
    "separation": ComponentCapability(
        SEPARATION_OPERATIONS,
        "partition-discovery",
        30,
        4,
        reactive_template="reaction-to-purification",
    ),
    "crystallization": ComponentCapability(
        CRYSTALLIZATION_OPERATIONS,
        "reaction-to-crystallization",
        40,
        10,
        2,
        2.0,
        ("out",),
    ),
    "distillation": ComponentCapability(
        DISTILLATION_OPERATIONS,
        "reaction-to-distillation",
        50,
        6,
        0,
        2.0,
        ("distillate", "bottoms", "collected_fraction"),
    ),
    "continuous_flow": ComponentCapability(
        FLOW_OPERATIONS,
        "flow-reaction-optimization",
        60,
        6,
        0,
        1.0,
    ),
    "electrochemistry": ComponentCapability(
        ELECTROCHEMISTRY_OPERATIONS,
        "electrochemical-conversion",
        70,
        7,
        1,
        1.0,
    ),
    "observation": ComponentCapability(("measure",), outputs=()),
}


def component_operations(kinds: tuple[str, ...]) -> tuple[str, ...]:
    """Capabilities accumulate independently of the identities of their peers."""
    return tuple(
        dict.fromkeys(
            (
                *COMMON_OPERATIONS,
                *(op for kind in kinds for op in COMPONENT_CAPABILITIES[kind].operations),
            )
        )
    )
