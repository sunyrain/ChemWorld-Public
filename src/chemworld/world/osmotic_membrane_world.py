"""Original two-tank world for passive, finite, isothermal osmosis."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.osmotic_membrane import OsmoticMembraneSpec, osmotic_membrane_inventory
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.osmotic_membrane import OSMOTIC_MEMBRANE_TYPE, install_osmotic_liquid
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def osmotic_membrane_world_spec(
    spec: OsmoticMembraneSpec,
    left_amounts: Mapping[str, float],
    right_amounts: Mapping[str, float],
    *,
    budget: int = 100,
    shared_clock: bool = False,
) -> WorldCompositionSpec:
    """Wait moves signed water; no pump, turbine or electrical output is installed."""
    initial = osmotic_membrane_inventory(spec, left_amounts, right_amounts)
    names = ("reactor", spec.right_vessel_id)
    if names[0] == names[1]:
        raise ValueError("osmotic membrane requires two distinct vessels")
    states = {}
    for i, name in enumerate(names):
        equipment = (
            {}
            if i
            else {
                "membrane": EquipmentRecord(
                    "membrane",
                    OSMOTIC_MEMBRANE_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict()},
                )
            }
        )
        state = WorldState(
            initial.amounts_mol[i],
            0,
            spec.temperature_K,
            spec.pressures_Pa[i],
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-isothermal-osmotic-tank",
                        spec.capacities_L[i],
                        spec.bath.maximum_setpoint_K,
                        max(spec.pressures_Pa) * 1.1,
                        ("liquid",),
                        spec.temperature_K,
                        spec.pressures_Pa[i],
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
        states[name] = install_osmotic_liquid(
            state, spec, i, initial.amounts_mol[i], initial.enthalpies_J[i]
        )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "declared-finite-osmotic-membrane",
            tuple(
                SpeciesSpec(
                    s.species_id, hill_formula(dict(s.composition)), charge=s.charge, phase="liquid"
                )
                for s in spec.properties.database.species
            ),
            (),
        ),
        states,
        lab_clock=LabClockSpec(vessel_capacity=2) if shared_clock else None,
    )
    components = [
        {"id": "separator", "kind": "ro_membrane", "vessel": names[0], "parameters": {}},
        {"id": "draw", "kind": "material_input", "vessel": names[1], "parameters": {}},
    ]
    if shared_clock:
        components.append(
            {
                "id": "clock",
                "kind": "lab_schedule",
                "vessel": names[0],
                "parameters": {"duration_range_s": [1e-6, 14400]},
            }
        )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "declared-finite-osmotic-membrane",
            "components": components,
            "vessels": [
                {"id": name, "capacity_L": capacity}
                for name, capacity in zip(names, spec.capacities_L, strict=True)
            ],
            "connections": [
                {
                    "id": "osmotic-water",
                    "unit": "mol",
                    "source": {"component": "separator", "port": "out"},
                    "target": {"component": "draw", "port": "in"},
                }
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite chemical-potential-driven water transfer",
                "budget": budget,
                "operations": ["wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
