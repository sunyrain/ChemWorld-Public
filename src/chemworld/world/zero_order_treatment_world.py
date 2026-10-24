"""Finite source, product and concentrate tanks on the original native runtime."""

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
from chemworld.physchem.water_treatment_economics import empty_treatment_receipt
from chemworld.physchem.zero_order_treatment import treatment_inventory
from chemworld.physchem.zero_order_treatment_specs import ZeroOrderTreatmentSpec
from chemworld.runtime.transport_devices import install_tank
from chemworld.runtime.zero_order_treatment import TREATMENT_TYPE
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def zero_order_treatment_world_spec(
    spec: ZeroOrderTreatmentSpec,
    feed_amounts: Mapping[str, float],
    *,
    collectors: tuple[Mapping[str, float], ...] | None = None,
    shared_clock: bool = False,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = treatment_inventory(spec, feed_amounts, collectors=collectors)
    names = ("reactor", spec.product_vessel_id, *(s.reject_vessel_id for s in spec.stages))
    if len(set(names)) != len(names):
        raise ValueError("treatment outlets cannot reuse the reactor source identity")
    feed_volume = spec.properties.evaluate(dict(feed_amounts), spec.temperature_K)[2] * 1000
    capacities = (feed_volume, spec.product_capacity_L, *(s.reject_capacity_L for s in spec.stages))
    states = {}
    for i, (name, capacity) in enumerate(zip(names, capacities, strict=True)):
        equipment = (
            {}
            if i
            else {
                "treatment": EquipmentRecord(
                    "treatment",
                    TREATMENT_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={
                        "physical_model": spec.to_dict(),
                        "treatment_totals": empty_treatment_receipt(spec),
                    },
                )
            }
        )
        state = WorldState(
            inventory.amounts_mol[i],
            0.0,
            spec.temperature_K,
            spec.properties.pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-zero-order-treatment-tank",
                        capacity,
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
        states[name] = install_tank(
            state, spec, i, inventory.amounts_mol[i], inventory.enthalpies_J[i]
        )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "zero-order-liquid-treatment",
            tuple(
                SpeciesSpec(p.component.identifier, p.component.formula, phase="liquid")
                for p in spec.properties.packages
            ),
            (),
        ),
        states,
        lab_clock=LabClockSpec(vessel_capacity=len(names)) if shared_clock else None,
    )
    components = [
        {"id": "treatment", "kind": "finite_transport", "vessel": "reactor", "parameters": {}}
    ]
    components.extend(
        {"id": name, "kind": "material_input", "vessel": name, "parameters": {}}
        for name in names[1:]
    )
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
            "composition_id": "declared-zero-order-water-treatment",
            "components": components,
            "vessels": [
                {"id": name, "capacity_L": capacity}
                for name, capacity in zip(names, capacities, strict=True)
            ],
            "connections": [
                {
                    "id": name,
                    "unit": "mol",
                    "source": {"component": "treatment", "port": "out"},
                    "target": {"component": name, "port": "in"},
                }
                for name in names[1:]
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study declared empirical liquid splits and actual utility costs",
                "budget": budget,
                "operations": ["set_flow_rate", "wait", "terminate"]
                if shared_clock
                else ["set_flow_rate", "run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
