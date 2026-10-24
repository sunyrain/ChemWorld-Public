"""Five finite original vessels; no independent second sorbent process driver."""

from collections.abc import Mapping, Sequence

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.aqueous_adsorption import (
    AqueousAdsorptionSpec,
    aqueous_adsorption_inventory,
)
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.aqueous_adsorption import (
    AQUEOUS_ADSORPTION_TYPE,
    MEDIA_TYPE,
    install_adsorption_liquid,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def aqueous_adsorption_world_spec(
    spec: AqueousAdsorptionSpec,
    amounts: Sequence[Mapping[str, float]],
    *,
    initial_loading_mol_per_kg: Mapping[str, float] | None = None,
    shared_clock: bool = False,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = aqueous_adsorption_inventory(
        spec, amounts, initial_loading_mol_per_kg=initial_loading_mol_per_kg
    )
    names = (
        "reactor",
        spec.contact_vessel_id,
        spec.product_vessel_id,
        spec.rinse_vessel_id,
        spec.waste_vessel_id,
    )
    if len(set(names)) != 5:
        raise ValueError("aqueous adsorption needs five distinct tanks")
    states = {}
    for i, name in enumerate(names):
        equipment = {}
        if i == 0:
            equipment["aqueous_adsorption"] = EquipmentRecord(
                "aqueous_adsorption",
                AQUEOUS_ADSORPTION_TYPE,
                name,
                batch_reset="preserve",
                settings={"physical_model": spec.to_dict()},
            )
        if i == 1:
            equipment[spec.media_equipment_id] = EquipmentRecord(
                spec.media_equipment_id,
                MEDIA_TYPE,
                name,
                batch_reset="preserve",
                settings={"sorbent_model": spec.sorbent.to_dict()},
            )
        state = WorldState(
            inventory.amounts_mol[i],
            0.0,
            spec.temperature_K,
            spec.pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-aqueous-adsorption-tank",
                        spec.capacities_L[i],
                        spec.viscosity.temperature_bounds_K[1],
                        spec.pressure_Pa * 1.1,
                        ("liquid",),
                        spec.temperature_K,
                        spec.pressure_Pa,
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
        states[name] = install_adsorption_liquid(
            state,
            spec,
            inventory.amounts_mol[i],
            inventory.enthalpies_J[i],
            inventory.sorbent_slot if i == 1 else None,
        )
    components = [
        {"id": "adsorption", "kind": "finite_transport", "vessel": names[0], "parameters": {}}
    ]
    components.extend(
        {"id": n, "kind": "material_input", "vessel": n, "parameters": {}} for n in names[1:]
    )
    if shared_clock:
        components.extend(
            {
                "id": f"{n}:clock",
                "kind": "lab_schedule",
                "vessel": n,
                "parameters": {"duration_range_s": [1e-6, 14400]},
            }
            for n in names
        )
    network = ReactionNetworkSpec(
        "finite-neutral-aqueous-adsorption",
        tuple(
            SpeciesSpec(s.species_id, hill_formula(dict(s.composition)), phase="liquid")
            for s in spec.database.species
        ),
        (),
    )
    physics = CompositionPhysics(
        network,
        states,
        lab_clock=LabClockSpec(vessel_capacity=5) if shared_clock else None,
        advance_bindings={names[0]: "finite-pump-transport"},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-aqueous-adsorption",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": c} for n, c in zip(names, spec.capacities_L, strict=True)
            ],
            "connections": [
                {
                    "id": edge,
                    "unit": "mol",
                    "source": {"component": source, "port": "out"},
                    "target": {"component": target, "port": "in"},
                }
                for edge, source, target in (
                    ("process-inlet", "adsorption", names[1]),
                    ("process-outlet", names[1], names[2]),
                    ("rinse-inlet", names[3], names[1]),
                    ("rinse-outlet", names[1], names[4]),
                )
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite adsorption and finite wash regeneration",
                "budget": budget,
                "operations": ["set_flow_rate", "wait", "select_vessel", "terminate"]
                if shared_clock
                else ["set_flow_rate", "run_flow", "wait", "select_vessel", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
