"""Two finite native liquid inventories in one periodic internal ED cell pair."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.electrodialysis import ElectrodialysisSpec, electrodialysis_inventory
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.electrodialysis import ELECTRODIALYSIS_TYPE, install_ed_liquid
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def electrodialysis_world_spec(
    spec: ElectrodialysisSpec,
    diluate: Mapping[str, float],
    concentrate: Mapping[str, float],
    *,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = electrodialysis_inventory(spec, diluate, concentrate)
    names = ("reactor", spec.concentrate_vessel_id)
    if len(set(names)) != 2:
        raise ValueError("ED requires distinct finite dilute and concentrate tanks")
    states = {}
    for i, name in enumerate(names):
        equipment = (
            {}
            if i
            else {
                "membrane": EquipmentRecord(
                    "membrane",
                    ELECTRODIALYSIS_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={
                        "physical_model": spec.to_dict(),
                        "electrical_control": {
                            "limits_V": [-spec.maximum_voltage_V, spec.maximum_voltage_V],
                            "limits_A": [-spec.maximum_current_A, spec.maximum_current_A],
                            "program": ScalarControlProgram(((0, 0),), "A").to_dict(),
                        },
                    },
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
                        "periodic-ed-reservoir",
                        spec.capacities_L[i],
                        spec.properties.temperature_bounds_K[1],
                        spec.reservoir_pressure_Pa * 1.1,
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
        states[name] = install_ed_liquid(
            state, spec, inventory.amounts_mol[i], inventory.enthalpies_J[i]
        )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "periodic-interior-electrodialysis",
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
            "composition_id": "periodic-interior-electrodialysis",
            "components": [
                {"id": "separator", "kind": "ro_membrane", "vessel": "reactor", "parameters": {}},
                {
                    "id": "concentrate",
                    "kind": "material_input",
                    "vessel": names[1],
                    "parameters": {},
                },
            ],
            "vessels": [
                {"id": n, "capacity_L": v} for n, v in zip(names, spec.capacities_L, strict=True)
            ],
            "connections": [
                {
                    "id": "periodic-ion-water-transfer",
                    "unit": "mol",
                    "source": {"component": "separator", "port": "out"},
                    "target": {"component": "concentrate", "port": "in"},
                }
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite periodic multi-ion electrodialysis",
                "budget": budget,
                "operations": ["configure_electrical_control", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
