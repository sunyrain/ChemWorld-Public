"""Six finite original vessels; no independent second sorbent process driver."""

from collections.abc import Mapping, Sequence

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.cation_exchange import (
    CationExchangeSpec,
    cation_exchange_inventory,
)
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.cation_exchange import (
    CATION_EXCHANGE_TYPE,
    MEDIA_TYPE,
    install_exchange_liquid,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def cation_exchange_world_spec(
    spec: CationExchangeSpec,
    amounts: Sequence[Mapping[str, float]],
    *,
    initial_calcium_equivalent_fraction: float = 0.0,
    shared_clock: bool = False,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = cation_exchange_inventory(
        spec, amounts, initial_calcium_equivalent_fraction=initial_calcium_equivalent_fraction
    )
    names = (
        "reactor",
        spec.contact_vessel_id,
        spec.product_vessel_id,
        spec.regenerant_vessel_id,
        spec.rinse_vessel_id,
        spec.waste_vessel_id,
    )
    if len(set(names)) != 6:
        raise ValueError("cation exchange needs six distinct tanks")
    states = {}
    for i, name in enumerate(names):
        equipment = {}
        if i == 0:
            equipment["cation_exchange"] = EquipmentRecord(
                "cation_exchange",
                CATION_EXCHANGE_TYPE,
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
                settings={"resin_model": "gaines-thomas-Na-Ca-v1"},
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
                        "finite-cation-exchange-tank",
                        spec.capacities_L[i],
                        spec.properties.viscosity.temperature_bounds_K[1],
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
        states[name] = install_exchange_liquid(
            state,
            spec,
            inventory.amounts_mol[i],
            inventory.enthalpies_J[i],
            inventory.resin_slot if i == 1 else None,
        )
    components = [
        {"id": "exchange", "kind": "finite_transport", "vessel": names[0], "parameters": {}}
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
        "finite-Na-Ca-Cl-exchange",
        tuple(
            SpeciesSpec(
                s.species_id, hill_formula(dict(s.composition)), phase="liquid", charge=s.charge
            )
            for s in spec.database.species
        ),
        (),
    )
    physics = CompositionPhysics(
        network,
        states,
        lab_clock=LabClockSpec(vessel_capacity=6) if shared_clock else None,
        advance_bindings={names[0]: "finite-pump-transport"},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-cation-exchange",
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
                    ("process-inlet", "exchange", names[1]),
                    ("process-outlet", names[1], names[2]),
                    ("regenerant-inlet", names[3], names[1]),
                    ("rinse-inlet", names[4], names[1]),
                    ("waste-outlet", names[1], names[5]),
                )
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite cation exchange, salt regeneration and rinse",
                "budget": budget,
                "operations": ["set_flow_rate", "wait", "select_vessel", "terminate"]
                if shared_clock
                else ["set_flow_rate", "run_flow", "wait", "select_vessel", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
