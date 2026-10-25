"""Two original finite liquid inventories connected to an explicit passive load."""

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
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.reverse_electrodialysis import (
    ReverseElectrodialysisSpec,
    reverse_electrodialysis_inventory,
)
from chemworld.runtime.electrodialysis import install_ed_liquid
from chemworld.runtime.reverse_electrodialysis import REVERSE_ELECTRODIALYSIS_TYPE
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def reverse_electrodialysis_world_spec(
    spec: ReverseElectrodialysisSpec,
    diluate: Mapping[str, float],
    concentrate: Mapping[str, float],
    *,
    budget: int = 100,
    shared_clock: bool = True,
) -> WorldCompositionSpec:
    pair = spec.cell_pair
    inventory = reverse_electrodialysis_inventory(spec, diluate, concentrate)
    names = ("reactor", pair.concentrate_vessel_id)
    if len(set(names)) != 2:
        raise ValueError("RED requires two distinct physical reservoirs")
    states = {}
    for i, name in enumerate(names):
        equipment = (
            {}
            if i
            else {
                "membrane": EquipmentRecord(
                    "membrane",
                    REVERSE_ELECTRODIALYSIS_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={
                        "physical_model": spec.to_dict(),
                        "electrical_control": {
                            "limits_V": [-pair.maximum_voltage_V, pair.maximum_voltage_V],
                            "limits_A": [-pair.maximum_current_A, pair.maximum_current_A],
                            "limits_R_Ohm": [
                                spec.minimum_load_resistance_Ohm,
                                spec.maximum_load_resistance_Ohm,
                            ],
                            "passive_load_modes": ["open_circuit", "resistance"],
                            "supported_units": ["A", "Ohm"],
                            "load_mode": "open_circuit",
                            "staged_control": False,
                            "program": ScalarControlProgram(((0, 0),), "A").to_dict(),
                        },
                    },
                ),
            }
        )
        state = WorldState(
            inventory.amounts_mol[i],
            0,
            pair.temperature_K,
            pair.reservoir_pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-passive-red-reservoir",
                        pair.capacities_L[i],
                        spec.properties.temperature_bounds_K[1],
                        pair.reservoir_pressure_Pa * 1.1,
                        ("liquid",),
                        pair.temperature_K,
                        pair.reservoir_pressure_Pa,
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
            state, pair, inventory.amounts_mol[i], inventory.enthalpies_J[i]
        )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "periodic-passive-reverse-electrodialysis",
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
        {"id": "concentrate", "kind": "material_input", "vessel": names[1], "parameters": {}},
        {
            "id": "meter",
            "kind": "observation",
            "vessel": names[0],
            "parameters": {"instruments": ["electrical_meter"]},
        },
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
            "composition_id": "periodic-passive-reverse-electrodialysis",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": v} for n, v in zip(names, pair.capacities_L, strict=True)
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
                "objective": "measure finite salt-gradient power delivered to a passive load",
                "budget": budget,
                "operations": ["configure_electrical_control", "wait", "measure", "terminate"],
                "instruments": ["electrical_meter"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
