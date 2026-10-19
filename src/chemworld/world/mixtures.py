"""Shared interpretation of the current phase-resolved solvent inventory."""

from __future__ import annotations

from typing import Any

from chemworld.foundation import WorldState, equipment_settings, selected_phase_id
from chemworld.foundation.solvents import HEAT_CAPACITY_RATIOS, SolventInventory


def working_solvents(state: WorldState) -> SolventInventory:
    phases = {} if state.phases is None else state.phases.phases
    selected = selected_phase_id(state.phases)
    if selected in phases and phases[selected].solvents.volume_L > 0:
        return phases[selected].solvents
    total = SolventInventory()
    for phase in phases.values():
        total = total + phase.solvents
    if total.volume_L > 0:
        return total
    # An explicitly constructed initial state has a nominal pure medium until
    # its phase inventory is initialized. This never selects historical physics.
    index = int(equipment_settings(state.equipment, "batch_reactor").get("solvent", 0))
    return SolventInventory.pure(index, max(state.volume_L, 1e-12))


def volumetric_heat_capacity(state: WorldState, base_J_L_K: float) -> float:
    return float(base_J_L_K) * working_solvents(state).linear_property(HEAT_CAPACITY_RATIOS)


def crystallization_medium_modifier(state: WorldState, world: Any, property_name: str) -> float:
    nominal = {
        "solubility": (1.0, 1.5, 0.7, 0.4),
        "nucleation": (1.0, 0.8, 1.5, 1.3),
        "growth": (1.0, 1.2, 0.8, 0.7),
        "occlusion": (1.0, 0.9, 1.2, 1.4),
    }
    endpoints = (
        getattr(world, f"crystallization_solvent_{property_name}_multipliers")
        if state.metadata.get("crystallization_material_family_id")
        == "reaction-crystallization-latent-materials-v1"
        else nominal[property_name]
    )
    return working_solvents(state).log_property(endpoints)
