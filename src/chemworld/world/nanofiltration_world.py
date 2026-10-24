"""Three original finite vessels connected to one declared multi-ion pore membrane."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.nanofiltration import NanofiltrationSpec, nanofiltration_inventory
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.nanofiltration import NANOFILTRATION_TYPE, install_nf_liquid
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def nanofiltration_world_spec(
    spec: NanofiltrationSpec,
    feed_amounts: Mapping[str, float],
    *,
    permeate: Mapping[str, float] | None = None,
    retentate: Mapping[str, float] | None = None,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = nanofiltration_inventory(spec, feed_amounts, permeate=permeate, retentate=retentate)
    names = ("reactor", spec.permeate_vessel_id, spec.retentate_vessel_id)
    if len(set(names)) != 3:
        raise ValueError("multi-ion membrane needs three distinct vessels")
    feed_volume = (
        spec.properties.volume(feed_amounts, spec.temperature_K, spec.reservoir_pressure_Pa) * 1000
    )
    capacities = (feed_volume, spec.permeate_capacity_L, spec.retentate_capacity_L)
    states = {}
    for i, name in enumerate(names):
        equipment = (
            {}
            if i
            else {
                "membrane": EquipmentRecord(
                    "membrane",
                    NANOFILTRATION_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict()},
                )
            }
        )
        state = WorldState(
            inventory.amounts_mol[i],
            0,
            spec.temperature_K,
            spec.reservoir_pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-multiion-nanofiltration-tank",
                        capacities[i],
                        spec.properties.temperature_bounds_K[1],
                        spec.inlet_pressure_Pa * 1.1,
                        ("liquid",),
                        spec.temperature_K,
                        spec.reservoir_pressure_Pa,
                    )
                }
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
        states[name] = install_nf_liquid(
            state, spec, inventory.amounts_mol[i], inventory.enthalpies_J[i]
        )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "declared-multiion-nanofiltration",
            tuple(
                SpeciesSpec(
                    s.species_id, hill_formula(dict(s.composition)), charge=s.charge, phase="liquid"
                )
                for s in spec.properties.database.species
            ),
            (),
        ),
        states,
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "declared-multiion-nanofiltration",
            "components": [
                {"id": "separator", "kind": "ro_membrane", "vessel": "reactor", "parameters": {}},
                *(
                    {"id": key, "kind": "material_input", "vessel": name, "parameters": {}}
                    for key, name in zip(("permeate", "retentate"), names[1:], strict=True)
                ),
            ],
            "vessels": [
                {"id": name, "capacity_L": capacity}
                for name, capacity in zip(names, capacities, strict=True)
            ],
            "connections": [
                {
                    "id": key,
                    "unit": "mol",
                    "source": {"component": "separator", "port": "out"},
                    "target": {"component": key, "port": "in"},
                }
                for key in ("permeate", "retentate")
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study declared multi-ion nanofiltration and finite collection",
                "budget": budget,
                "operations": ["set_flow_rate", "run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
