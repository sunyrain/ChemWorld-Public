"""Four original finite liquid tanks coupled by two pumps and one exchanger."""

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.finite_heat_recovery import heat_recovery_inventory
from chemworld.physchem.finite_heat_recovery_specs import FiniteHeatRecoverySpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.finite_heat_recovery import HEAT_RECOVERY_TYPE
from chemworld.runtime.transport_devices import install_tank
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def finite_heat_recovery_world_spec(
    spec: FiniteHeatRecoverySpec,
    amounts: tuple[dict, ...],
    temperatures_K: tuple[float, ...],
    *,
    shared_clock: bool = False,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = heat_recovery_inventory(spec, amounts, temperatures_K)
    names = (
        spec.left_feed_vessel_id,
        "reactor",
        spec.right_feed_vessel_id,
        spec.right_receiver_vessel_id,
    )
    if len(set(names)) != 4:
        raise ValueError("heat-recovery tank names cannot reuse the reactor anchor")
    states = {}
    for i, name in enumerate(names):
        volume = (
            spec.properties.evaluate(amounts[i], temperatures_K[i])[2] * 1000
            if sum(amounts[i].values())
            else 0
        )
        capacity = spec.receiver_capacities_L[i // 2] if i % 2 else max(volume, 1e-6)
        equipment = (
            {}
            if i != 1
            else {
                "heat_recovery": EquipmentRecord(
                    "heat_recovery",
                    HEAT_RECOVERY_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={
                        "physical_model": spec.to_dict(),
                        "flow_controls": {"left": 0.0, "right": 0.0},
                    },
                )
            }
        )
        state = WorldState(
            dict(amounts[i]),
            volume,
            temperatures_K[i],
            spec.properties.pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-heat-recovery-tank",
                        capacity,
                        spec.properties.temperature_bounds_K[1],
                        spec.properties.pressure_Pa * 1.1,
                        ("liquid",),
                        temperatures_K[i],
                        spec.properties.pressure_Pa,
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
        states[name] = install_tank(state, spec, i, amounts[i], inventory.enthalpies_J[i])
    states = {"reactor": states["reactor"], **{k: v for k, v in states.items() if k != "reactor"}}
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "finite-two-stream-heat-recovery",
            tuple(
                SpeciesSpec(p.component.identifier, p.component.formula, phase="liquid")
                for p in spec.properties.packages
            ),
            (),
        ),
        states,
        lab_clock=LabClockSpec(vessel_capacity=4) if shared_clock else None,
    )
    components: list[dict] = [
        {
            "id": name,
            "kind": "finite_transport" if i == 1 else "material_input",
            "vessel": name,
            "parameters": {},
        }
        for i, name in enumerate(names)
    ]
    if shared_clock:
        components.append(
            {
                "id": "clock",
                "kind": "lab_schedule",
                "vessel": "reactor",
                "parameters": {"duration_range_s": [1e-6, 14400]},
            }
        )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-two-stream-heat-recovery",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": s.vessels.vessels[n].max_volume_L}
                for n, s in states.items()
            ],
            "connections": [
                {
                    "id": side,
                    "unit": "mol",
                    "source": {"component": names[i], "port": "out"},
                    "target": {"component": names[i + 1], "port": "in"},
                }
                for side, i in (("left", 0), ("right", 2))
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite two-stream heat recovery",
                "budget": budget,
                "operations": ["set_flow_rate", "wait", "terminate"]
                if shared_clock
                else ["set_flow_rate", "run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
