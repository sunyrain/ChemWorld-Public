"""A declared seeded-mineral experiment using the existing world and session."""

from collections.abc import Mapping
from typing import cast

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.aqueous_mineral_kinetics import (
    AqueousMineralKineticsSpec,
    prepare_aqueous_mineral_state,
)
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.aqueous_mineral_kinetics import (
    DATABASE_MINERAL_TYPE,
    database_mineral_geometry_slot,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_aqueous_mineral_kinetics_world(
    spec: AqueousMineralKineticsSpec,
    amounts_mol: Mapping[str, float],
    geometries: Mapping[str, MineralGeometry],
    *,
    temperature_K: float = 298.15,
    pressure_Pa: float = 1e5,
    max_volume_L: float = 2.0,
    budget: int = 40,
) -> WorldCompositionSpec:
    """Prepare fast liquid chemistry only; wait owns subsequent finite kinetics."""
    db = spec.equilibrium.database
    if set(geometries) != set(spec.equilibrium.mineral_ids):
        raise ValueError("world needs all original mineral reference geometries")
    amounts = prepare_aqueous_mineral_state(
        spec, amounts_mol, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    liquid = {s.species_id: amounts[s.species_id] for s in db.species if s.phase != "mineral"}
    props = aqueous_phase_properties(
        db, liquid, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    phase_ids = {k: "mineral:" + k for k in spec.equilibrium.mineral_ids}
    phases = {
        "aqueous": PhaseRecord(
            "aqueous",
            "reactor",
            "aqueous",
            props.volume_m3 * 1000,
            liquid,
            selected=True,
            metadata={"carrier_representation": "species_inventory"},
        )
    }
    for species in db.species:
        if species.phase == "mineral":
            key, n = phase_ids[species.species_id], amounts[species.species_id]
            phases[key] = PhaseRecord(
                key,
                "reactor",
                "solid",
                n * cast(float, species.standard_state.molar_volume_m3_mol) * 1000,
                {species.species_id: n},
                metadata={"pure_mineral": True},
            )
    record = EquipmentRecord(
        "declared-mineral-model",
        DATABASE_MINERAL_TYPE,
        "reactor",
        batch_reset="preserve",
        settings={
            "physical_model": spec.to_dict(),
            "liquid_phase_id": "aqueous",
            "mineral_phase_ids": phase_ids,
            "temperature_K": temperature_K,
            "pressure_Pa": pressure_Pa,
        },
    )
    state = WorldState(
        amounts,
        sum(p.volume_L for p in phases.values()),
        temperature_K,
        pressure_Pa,
        "liquid",
        "reactor",
        phases=PhaseLedger(phases),
        equipment=EquipmentLedger({record.equipment_id: record}),
        vessels=VesselLedger(
            {
                "reactor": VesselRecord(
                    "reactor",
                    "declared-mineral-TP-vessel",
                    max_volume_L,
                    temperature_K,
                    pressure_Pa,
                    (),
                    temperature_K,
                    pressure_Pa,
                )
            }
        ),
        state_slots=StateSlots(
            tuple(
                database_mineral_geometry_slot("reactor", phase_ids[k], k, geometries[k])
                for k in spec.equilibrium.mineral_ids
            )
        ),
        units={
            "amount": "mol",
            "volume": "L",
            "temperature": "K",
            "pressure": "Pa",
            "time": "s",
            "cost": "currency",
            "risk": "risk",
        },
    )
    network = ReactionNetworkSpec(
        "declared-mineral-kinetics",
        tuple(
            SpeciesSpec(
                s.species_id,
                formula=hill_formula(dict(s.composition)),
                charge=s.charge,
                phase="solid" if s.phase == "mineral" else "liquid",
            )
            for s in db.species
        ),
        (),
    )
    physics = CompositionPhysics(network, {"reactor": state})
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "declared-mineral-kinetics",
            "components": [
                {
                    "id": "mineral-solution",
                    "kind": "mineral_kinetics",
                    "vessel": "reactor",
                    "parameters": {},
                }
            ],
            "vessels": [{"id": "reactor", "capacity_L": max_volume_L}],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study declared finite mineral rates and original material history",
                "budget": budget,
                "operations": ["wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
