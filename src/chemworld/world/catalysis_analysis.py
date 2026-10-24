"""Read-only held-current-pressure queries from an attached physical catalyst."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chemworld.foundation import WorldState
from chemworld.physchem.catalysis_analysis import (
    CatalyticAnalysisConditions,
    solve_catalytic_steady_state,
)
from chemworld.physchem.catalysis_network import (
    CATALYTIC_STATE_TYPE,
    CatalyticMechanismSpec,
    capacity_field,
    validate_catalytic_slot,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import RootPolicy
from chemworld.world.composition import WorldCompositionSpec


def analyze_catalytic_world(
    composition: WorldCompositionSpec,
    state: WorldState,
    *,
    target_species: str,
    catalyst_id: str | None = None,
    product_species: Sequence[str] | None = None,
    element_basis: str | None = None,
    policy: RootPolicy | None = None,
) -> dict[str, Any]:
    """Resolve declaration, capacity and current inventory from their original owners.

    The returned steady state assumes held current gas pressures. It neither
    predicts a sealed reactor's endpoint nor executes an action on that reactor.
    """
    attached = [
        c
        for c in composition.components
        if c.kind == "surface_reaction"
        and c.vessel == state.vessel_id
        and (catalyst_id is None or c.id == catalyst_id)
    ]
    if len(attached) != 1 or composition.physics is None:
        raise ValueError("read-only analysis requires one explicitly attached catalytic component")
    component = attached[0]
    mechanism_id = component.parameters.get("surface_id")
    spec = (
        composition.physics.models_of("surface").get(mechanism_id)
        if isinstance(mechanism_id, str)
        else None
    )
    if not isinstance(spec, CatalyticMechanismSpec):
        raise ValueError("attached component does not declare a catalytic mechanism")
    slots = [
        s
        for s in state.state_slots.slots
        if s.spec.state_type == CATALYTIC_STATE_TYPE
        and s.spec.owner_id == component.id
        and s.location_id == state.vessel_id
        and s.spec.mechanism_id == spec.mechanism_id
    ]
    if len(slots) != 1:
        raise ValueError("attached catalyst requires its unambiguous current owned slot")
    slot = slots[0]
    validate_catalytic_slot(slot, spec)
    if state.phases is None:
        raise ValueError("current world requires a physical phase ledger")
    gas_phases = [
        p
        for p in state.phases.phases.values()
        if p.vessel_id == state.vessel_id and p.phase_type == "gas"
    ]
    if len(gas_phases) != 1 or gas_phases[0].volume_L <= 0:
        raise ValueError("held-pressure analysis requires one current physical gas phase")
    phase = gas_phases[0]
    if set(phase.species_amounts_mol) - set(spec.gas_species):
        raise ValueError("current gas phase includes species outside the attached mechanism")
    capacities = {k: slot.values[capacity_field(k)] for k in spec.site_types}
    scales = spec.coverage_amount_scales_mol(capacities)
    conditions = CatalyticAnalysisConditions(
        temperature_K=state.temperature_K,
        gas_partial_pressures_Pa={
            k: phase.species_amounts_mol.get(k, 0.0)
            * R
            * state.temperature_K
            / (phase.volume_L * 1e-3)
            for k in spec.gas_species
        },
        capacities_mol=capacities,
        volume_L=phase.volume_L,
    )
    result = solve_catalytic_steady_state(
        spec,
        conditions,
        {k: slot.values[k] / scales[k] for k in spec.adsorbates},
        target_species=target_species,
        product_species=product_species,
        element_basis=element_basis,
        policy=policy,
    )
    return {
        **result,
        "world_query": {
            "catalyst_id": component.id,
            "slot_id": slot.spec.slot_id,
            "vessel_id": state.vessel_id,
            "gas_phase_id": phase.phase_id,
            "state_time_s": state.ledger.time_s,
            "boundary": "read-only; held-current-gas-pressure surface analysis",
            "world_advanced": False,
        },
    }
