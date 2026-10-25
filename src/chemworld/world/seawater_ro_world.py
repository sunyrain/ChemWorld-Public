"""Original three-vessel membrane composition with empirical TDS mass slots."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.seawater_ro import SeawaterROSpec, seawater_ro_inventory
from chemworld.runtime.seawater_ro import SEAWATER_RO_TYPE, install_seawater_liquid
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def seawater_ro_world_spec(
    spec: SeawaterROSpec,
    feed_mass_kg: Mapping[str, float],
    *,
    budget: int = 100,
    shared_clock: bool = False,
) -> WorldCompositionSpec:
    inventory = seawater_ro_inventory(spec, feed_mass_kg)
    names = ("reactor", spec.permeate_vessel_id, spec.retentate_vessel_id)
    if len(set(names)) != 3:
        raise ValueError("seawater membrane needs distinct vessels")
    capacities = (
        spec.properties.volume(feed_mass_kg, spec.temperature_K, spec.reservoir_pressure_Pa) * 1000,
        spec.permeate_capacity_L,
        spec.retentate_capacity_L,
    )
    states = {}
    for i, name in enumerate(names):
        equipment = (
            {}
            if i
            else {
                "membrane": EquipmentRecord(
                    "membrane",
                    SEAWATER_RO_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict()},
                )
            }
        )
        state = WorldState(
            {"H2O": 0.0},
            0.0,
            spec.temperature_K,
            spec.reservoir_pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-seawater-ro-tank",
                        capacities[i],
                        393.15,
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
        states[name] = install_seawater_liquid(
            state, spec, inventory.masses_kg[i], inventory.enthalpies_J[i]
        )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "declared-seawater-ro", (SpeciesSpec("H2O", "H2O", phase="liquid"),), ()
        ),
        states,
        lab_clock=LabClockSpec(vessel_capacity=3) if shared_clock else None,
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "declared-seawater-ro",
            "components": [
                {"id": "separator", "kind": "ro_membrane", "vessel": "reactor", "parameters": {}},
                *(
                    {"id": key, "kind": "material_input", "vessel": name, "parameters": {}}
                    for key, name in zip(("permeate", "retentate"), names[1:], strict=True)
                ),
                *(
                    [
                        {
                            "id": "clock",
                            "kind": "lab_schedule",
                            "vessel": "reactor",
                            "parameters": {"duration_range_s": [1e-6, 14400]},
                        }
                    ]
                    if shared_clock
                    else []
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
                "objective": "study empirical seawater axial RO with conserved water and TDS mass",
                "budget": budget,
                "operations": ["set_flow_rate", "wait", "terminate"]
                if shared_clock
                else ["set_flow_rate", "run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
