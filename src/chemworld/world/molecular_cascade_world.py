"""Four finite molecular tanks on the original transport and shared clock."""

from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.molecular_cascade_inventory import (
    MolecularCascadeInventorySpec,
    molecular_cascade_inventory,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.molecular_cascade import CASCADE_TYPE, install_cascade_tank
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_molecular_cascade_world(
    spec: MolecularCascadeInventorySpec,
    amounts: Sequence[Mapping[str, float]],
    temperatures_K: Sequence[float],
    *,
    formulas: Mapping[str, str],
    shared_clock: bool = True,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = molecular_cascade_inventory(spec, amounts, temperatures_K)
    names, anchor = spec.vessel_ids, spec.vessel_ids[2]
    if set(formulas) != set(spec.properties.component_ids):
        raise ValueError("molecular cascade needs explicit original species formulas")
    states = {}
    for i, name in enumerate(names):
        phase = "gas" if i % 2 == 0 else "liquid"
        equipment = (
            {}
            if i != 2
            else {
                "cascade": EquipmentRecord(
                    "cascade",
                    CASCADE_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={
                        "physical_model": spec.to_dict(),
                        "flow_control": {"configured": True},
                    },
                )
            }
        )
        closed = spec.at_tp(i, amounts[i], temperatures_K[i])
        state = WorldState(
            dict(amounts[i]),
            1000 * sum(amounts[i].values()) / closed.molar_density_mol_m3,
            temperatures_K[i],
            spec.cascade.pressure_Pa,
            phase,
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "molecular-moving-volume-tank",
                        spec.tank_capacities_L[i],
                        spec.tank_temperature_bounds_K[i][1],
                        spec.cascade.pressure_Pa * 1.1,
                        (phase,),
                        temperatures_K[i],
                        spec.cascade.pressure_Pa,
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
        states[name] = install_cascade_tank(state, spec, i, amounts[i], inventory.enthalpies_J[i])
    states = {anchor: states[anchor], **{k: v for k, v in states.items() if k != anchor}}
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "molecular-equilibrium-cascade",
            tuple(
                SpeciesSpec(k, formulas[k], phase="liquid") for k in spec.properties.component_ids
            ),
            (),
        ),
        states,
        lab_clock=LabClockSpec(vessel_capacity=4) if shared_clock else None,
    )
    components: list[dict[str, Any]] = [
        {
            "id": name,
            "kind": "finite_transport" if i == 2 else "material_input",
            "vessel": name,
            "parameters": {},
        }
        for i, name in sorted(enumerate(names), key=lambda row: row[0] != 2)
    ]
    if shared_clock:
        components.append(
            {
                "id": "clock",
                "kind": "lab_schedule",
                "vessel": anchor,
                "parameters": {"duration_range_s": [1e-6, 14400]},
            }
        )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "molecular-equilibrium-four-tank-cascade",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": spec.tank_capacities_L[i]}
                for i, n in sorted(enumerate(names), key=lambda row: row[0] != 2)
            ],
            "connections": [
                {
                    "id": f"cascade:{i}",
                    "unit": "mol",
                    "source": {"component": source, "port": "out"},
                    "target": {"component": target, "port": "in"},
                }
                for i, (source, target) in enumerate(
                    ((names[0], anchor), (names[1], anchor), (anchor, names[3]))
                )
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study a finite molecular equilibrium cascade",
                "budget": budget,
                "operations": ["wait", "terminate"]
                if shared_clock
                else ["run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
