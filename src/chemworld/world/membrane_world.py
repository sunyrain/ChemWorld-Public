"""A bounded native dilute-RO world, independent of a task template."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.membranes import MembraneSpec, membrane_inventory
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.membranes import MEMBRANE_TYPE, install_liquid, set_fouling_inventory
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def membrane_world_spec(
    spec=None,
    *,
    feed_volume_L=1.0,
    concentration_mol_m3=10.0,
    budget=100,
    foulant_concentration_kg_m3=0.1,
    wash_volume_L=0.1,
):
    spec = spec or MembraneSpec()
    inventory = membrane_inventory(
        feed_volume_L,
        concentration_mol_m3,
        fouling=bool(spec.fouling),
        foulant_concentration_kg_m3=foulant_concentration_kg_m3,
        wash_volume_L=wash_volume_L,
    )
    states = {}
    names = (
        "reactor",
        spec.collector_vessel_id,
        *((spec.fouling.wash_vessel_id, spec.fouling.waste_vessel_id) if spec.fouling else ()),
    )
    if len(set(names)) != len(names):
        raise ValueError("each membrane liquid inventory needs its own vessel")
    capacities = (
        feed_volume_L,
        spec.collector_capacity_L,
        *((wash_volume_L, spec.fouling.waste_capacity_L) if spec.fouling else ()),
    )
    for index, name in enumerate(names):
        equipment = (
            {}
            if index
            else {
                "membrane": EquipmentRecord(
                    "membrane",
                    MEMBRANE_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict()},
                )
            }
        )
        state = WorldState(
            inventory.amounts_mol[index],
            feed_volume_L if index == 0 else 0.0,
            spec.temperature_K,
            101325.0,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-ro-chamber",
                        capacities[index],
                        340,
                        2e6,
                        ("liquid",),
                        spec.temperature_K,
                        101325.0,
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
        states[name] = install_liquid(state, inventory.amounts_mol[index])
    if spec.fouling:
        states["reactor"] = set_fouling_inventory(states["reactor"], "membrane", 0.0)
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "dilute-finite-ro",
            (
                SpeciesSpec("water", "H2O", phase="liquid"),
                SpeciesSpec("nacl", "NaCl", phase="liquid"),
                *((SpeciesSpec("foulant", "SiO2", phase="liquid"),) if spec.fouling else ()),
            ),
            (),
        ),
        states,
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "dilute-finite-ro",
            "components": [
                {"id": "separator", "kind": "ro_membrane", "vessel": "reactor", "parameters": {}},
                {
                    "id": "collector",
                    "kind": "material_input",
                    "vessel": spec.collector_vessel_id,
                    "parameters": {},
                },
                {
                    "id": "meter",
                    "kind": "observation",
                    "vessel": "reactor",
                    "parameters": {"instruments": ["membrane_meter"]},
                },
                *(
                    {"id": key, "kind": "material_input", "vessel": vessel, "parameters": {}}
                    for key, vessel in zip(("wash", "waste"), names[2:], strict=False)
                ),
            ],
            "vessels": [
                {"id": v, "capacity_L": s.vessels.vessels[v].max_volume_L}
                for v, s in states.items()
            ],
            "connections": [
                {
                    "id": "permeate",
                    "unit": "mol",
                    "source": {"component": "separator", "port": "out"},
                    "target": {"component": "collector", "port": "in"},
                },
                *(
                    {
                        "id": key,
                        "unit": "mol",
                        "source": {"component": source, "port": "out"},
                        "target": {"component": target, "port": "in"},
                    }
                    for key, source, target in (
                        ("wash_in", "wash", "separator"),
                        ("wash_out", "separator", "waste"),
                    )
                    if spec.fouling
                ),
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite dilute membrane transport",
                "budget": budget,
                "operations": ["set_flow_rate", "run_flow", "wait", "measure", "terminate"],
                "instruments": ["membrane_meter"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
