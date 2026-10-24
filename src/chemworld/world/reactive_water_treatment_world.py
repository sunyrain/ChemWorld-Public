"""Three original finite tanks with pumped inlet and reactive mixed holdup."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.reactive_water_treatment import (
    ReactiveWaterTreatmentSpec,
    reactive_water_treatment_inventory,
)
from chemworld.runtime.reactive_water_treatment import (
    REACTIVE_TREATMENT_TYPE,
    install_reactive_treatment_liquid,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def reactive_water_treatment_world_spec(
    spec: ReactiveWaterTreatmentSpec,
    feed: Mapping[str, float],
    reactor: Mapping[str, float],
    product: Mapping[str, float] | None = None,
    *,
    shared_clock: bool = False,
    budget: int = 100,
) -> WorldCompositionSpec:
    initial = reactive_water_treatment_inventory(spec, feed, reactor, product)
    names = ("reactor", spec.reactor_vessel_id, spec.product_vessel_id)
    if len(set(names)) != 3:
        raise ValueError("reactive treatment needs three distinct original tanks")
    states = {}
    for i, name in enumerate(names):
        equipment = (
            {}
            if i
            else {
                "treatment": EquipmentRecord(
                    "treatment",
                    REACTIVE_TREATMENT_TYPE,
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
            spec.properties.pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-reactive-treatment-tank",
                        spec.capacities_L[i],
                        spec.properties.temperature_bounds_K[1],
                        spec.properties.pressure_Pa * 1.1,
                        ("liquid",),
                        spec.temperature_K,
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
        states[name] = install_reactive_treatment_liquid(
            state, spec, initial.amounts_mol[i], initial.enthalpies_J[i]
        )
    components = [
        {"id": "treatment", "kind": "finite_transport", "vessel": names[0], "parameters": {}}
    ]
    components.extend(
        {"id": n, "kind": "material_input", "vessel": n, "parameters": {}} for n in names[1:]
    )
    if spec.network.reactions:
        components.append(
            {"id": "liquid-chemistry", "kind": "reaction", "vessel": names[1], "parameters": {}}
        )
    if shared_clock:
        components.extend(
            {
                "id": f"{name}:clock",
                "kind": "lab_schedule",
                "vessel": name,
                "parameters": {"duration_range_s": [1e-6, 14400]},
            }
            for name in names
        )
    physics = CompositionPhysics(
        spec.network,
        states,
        lab_clock=LabClockSpec(vessel_capacity=3) if shared_clock else None,
        advance_bindings={names[0]: "finite-pump-transport"},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-reactive-water-treatment",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": c} for n, c in zip(names, spec.capacities_L, strict=True)
            ],
            "connections": [
                {
                    "id": "inlet",
                    "unit": "mol",
                    "source": {"component": "treatment", "port": "out"},
                    "target": {"component": names[1], "port": "in"},
                },
                {
                    "id": "overflow",
                    "unit": "mol",
                    "source": {"component": names[1], "port": "out"},
                    "target": {"component": names[2], "port": "in"},
                },
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite pump-fed mixed-tank chemistry",
                "budget": budget,
                "operations": ["set_flow_rate", "wait", "terminate"]
                if shared_clock
                else ["set_flow_rate", "run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
