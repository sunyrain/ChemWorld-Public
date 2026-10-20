"""Initial-state factories for ChemWorld scenarios."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from chemworld.foundation import WorldState, upsert_equipment_record
from chemworld.foundation.state import SpeciesLedger


def initial_chemworld_state(
    *,
    species_ids: Iterable[str] | None = None,
    species_roles: Mapping[str, tuple[str, ...]] | None = None,
    initial_amounts_mol: Mapping[str, float] | None = None,
    initial_limiting_species: str | None = None,
    liquid_heat_capacity_J_L_K: float = 4180.0,
) -> WorldState:
    """Return the canonical empty batch-reactor state for scenario generation.

    The material amounts start at zero because ChemWorld tasks begin from an
    empty virtual lab. The mechanism/scenario layer owns the species namespace
    and recipe-scale initial-amount policy; the transactional runtime records charged
    initial amounts in ``SpeciesLedger.initial_amounts_mol``.
    """

    resolved_species_ids = tuple(species_ids or ())
    resolved_initial_amounts = dict.fromkeys(initial_amounts_mol or {}, 0.0)

    state = WorldState(
        species_amounts=dict.fromkeys(resolved_species_ids, 0.0),
        volume_L=0.0,
        temperature_K=298.15,
        pressure_Pa=101_325.0,
        phase="liquid",
        vessel_id="batch_reactor",
        liquid_heat_capacity_J_L_K=liquid_heat_capacity_J_L_K,
        units={
            "amount": "mol",
            "volume": "L",
            "temperature": "K",
            "pressure": "Pa",
            "time": "s",
            "cost": "currency",
            "risk": "risk",
        },
        species=SpeciesLedger(
            species_roles=dict(species_roles or {}),
            initial_amounts_mol=resolved_initial_amounts,
        ),
    )
    return state.replace(
        equipment=upsert_equipment_record(
            state.equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id=state.vessel_id,
            status="idle",
            settings={"solvent": 0, "catalyst": 0, "stirring_speed_rpm": 600.0},
        )
    )


__all__ = ["initial_chemworld_state"]
