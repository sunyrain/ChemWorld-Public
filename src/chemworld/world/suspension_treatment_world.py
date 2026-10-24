"""Three original finite tanks with prescribed jar conditioning and empirical clarification."""

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.suspension_treatment import (
    suspension_treatment_inventory,
)
from chemworld.physchem.suspension_treatment_specs import (
    SuspensionTankInventory,
    SuspensionTreatmentSpec,
)
from chemworld.runtime.suspension_treatment import (
    SUSPENSION_TREATMENT_TYPE,
    install_suspension_treatment_material,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def suspension_treatment_world_spec(
    spec: SuspensionTreatmentSpec,
    feed: SuspensionTankInventory,
    product: SuspensionTankInventory | None = None,
    sludge: SuspensionTankInventory | None = None,
    *,
    shared_clock: bool = False,
    budget: int = 100,
) -> WorldCompositionSpec:
    initial = suspension_treatment_inventory(spec, feed, product, sludge)
    names = ("reactor", spec.product_vessel_id, spec.sludge_vessel_id)
    if len(set(names)) != 3:
        raise ValueError("suspension treatment needs three distinct original tanks")
    states = {}
    for i, name in enumerate(names):
        equipment = (
            {}
            if i
            else {
                "treatment": EquipmentRecord(
                    "treatment",
                    SUSPENSION_TREATMENT_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict()},
                )
            }
        )
        state = WorldState(
            {},
            0,
            spec.temperature_K,
            spec.material.liquid.pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-suspension-treatment-tank",
                        spec.capacities_L[i],
                        spec.material.liquid.temperature_bounds_K[1],
                        spec.material.liquid.pressure_Pa * 1.1,
                        ("liquid", "solid"),
                        spec.temperature_K,
                        spec.material.liquid.pressure_Pa,
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
        states[name] = install_suspension_treatment_material(
            state, spec, initial.amounts_mol[i], initial.enthalpies_J[i]
        )
    components = [
        {"id": "treatment", "kind": "finite_transport", "vessel": names[0], "parameters": {}}
    ]
    components.extend(
        {"id": n, "kind": "material_input", "vessel": n, "parameters": {}} for n in names[1:]
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
        ReactionNetworkSpec(
            "suspension-chemical-identities",
            (
                *(
                    SpeciesSpec(p.component.identifier, p.component.formula, phase="liquid")
                    for p in spec.material.liquid.packages
                ),
                SpeciesSpec(
                    spec.material.particle.species_id, spec.material.particle.formula, phase="solid"
                ),
            ),
            (),
        ),
        states,
        lab_clock=LabClockSpec(vessel_capacity=3) if shared_clock else None,
        advance_bindings={names[0]: "finite-pump-transport"},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-suspension-water-treatment",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": c} for n, c in zip(names, spec.capacities_L, strict=True)
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
                "objective": "study finite suspension treatment from a measured jar endpoint",
                "budget": budget,
                "operations": ["set_flow_rate", "wait", "terminate"]
                if shared_clock
                else ["set_flow_rate", "run_flow", "wait", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
