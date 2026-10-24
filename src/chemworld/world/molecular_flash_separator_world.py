"""Single-feed original molecular flash station with three finite tank owners."""

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
from chemworld.physchem.molecular_flash_separator_inventory import (
    MolecularFlashSeparatorInventorySpec,
    molecular_flash_separator_inventory,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.molecular_flash_separator import SEPARATOR_TYPE, install_separator_tank
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_molecular_flash_separator_world(
    spec: MolecularFlashSeparatorInventorySpec,
    amounts: Sequence[Mapping[str, float]],
    temperatures_K: Sequence[float],
    *,
    formulas: Mapping[str, str],
    shared_clock: bool = True,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = molecular_flash_separator_inventory(spec, amounts, temperatures_K)
    ns, anchor = spec.vessel_ids, spec.vessel_ids[1]
    if set(formulas) != set(spec.properties.component_ids):
        raise ValueError("molecular separator requires original species formulas")
    states = {}
    for i, name in enumerate(ns):
        phase = spec.phase_id(i)
        equipment = (
            {}
            if i != 1
            else {
                "separator": EquipmentRecord(
                    "separator",
                    SEPARATOR_TYPE,
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
            spec.separator.pressure_Pa,
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
                        spec.separator.pressure_Pa * 1.1,
                        (phase,),
                        temperatures_K[i],
                        spec.separator.pressure_Pa,
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
        states[name] = install_separator_tank(state, spec, i, amounts[i], inventory.enthalpies_J[i])
    states = {anchor: states[anchor], **{k: v for k, v in states.items() if k != anchor}}
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "molecular-flash-separator",
            tuple(
                SpeciesSpec(k, formulas[k], phase="liquid") for k in spec.properties.component_ids
            ),
            (),
        ),
        states,
        lab_clock=LabClockSpec(vessel_capacity=3) if shared_clock else None,
    )
    ordered = sorted(enumerate(ns), key=lambda row: row[0] != 1)
    components: list[dict[str, Any]] = [
        {
            "id": name,
            "kind": "finite_transport" if i == 1 else "material_input",
            "vessel": name,
            "parameters": {},
        }
        for i, name in ordered
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
            "composition_id": "molecular-flash-three-finite-tanks",
            "components": components,
            "vessels": [{"id": n, "capacity_L": spec.tank_capacities_L[i]} for i, n in ordered],
            "connections": [
                {
                    "id": f"flash:{i}",
                    "unit": "mol",
                    "source": {"component": ns[0], "port": "out"},
                    "target": {"component": ns[i], "port": "in"},
                }
                for i in (1, 2)
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite temperature-controlled molecular flashing",
                "budget": budget,
                "operations": ["wait", "terminate"]
                if shared_clock
                else ["run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
