"""Finite aqueous mesh cells on the original world/phase/shared-clock contracts."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.aqueous_reactive_transport import (
    AqueousReactiveTransportSpec,
    prepare_aqueous_transport_state,
)
from chemworld.physchem.aqueous_transport_flow import inventory_properties
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.aqueous_reactive_transport import (
    AQUEOUS_TRANSPORT_TYPE,
    install_aqueous_cell,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_aqueous_reactive_transport_world(
    spec: AqueousReactiveTransportSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
    *,
    budget: int = 100,
) -> WorldCompositionSpec:
    initial = prepare_aqueous_transport_state(spec, amounts_by_vessel)
    states = {}
    members = (
        spec.anchor_vessel_id,
        *(v for v in spec.member_vessel_ids if v != spec.anchor_vessel_id),
    )
    for name in members:
        volume = (
            spec.vessel_capacity_m3(name)
            if spec.flow is None
            else inventory_properties(spec, initial[name]).volume_m3
        ) * 1000
        device = name + ":aqueous-transport"
        state = WorldState(
            initial[name],
            volume,
            spec.temperature_K,
            spec.pressure_Pa,
            "liquid",
            name,
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
            },
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-spatial-aqueous-cell",
                        spec.vessel_capacity_m3(name) * 1000,
                        spec.temperature_K,
                        spec.pressure_Pa,
                        ("liquid",),
                        spec.temperature_K,
                        spec.pressure_Pa,
                    )
                }
            ),
            equipment=EquipmentLedger(
                {
                    device: EquipmentRecord(
                        device,
                        AQUEOUS_TRANSPORT_TYPE,
                        name,
                        batch_reset="preserve",
                        settings={"physical_model": spec.to_dict()},
                    )
                }
                if name == spec.anchor_vessel_id
                else {}
            ),
        )
        states[name] = install_aqueous_cell(state, initial[name], volume)
    network = ReactionNetworkSpec(
        "finite-aqueous-spatial-species",
        tuple(
            SpeciesSpec(
                s.species_id, hill_formula(dict(s.composition)), phase="liquid", charge=s.charge
            )
            for s in spec.equilibrium.database.species
        ),
        (),
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-aqueous-spatial",
            "vessels": [
                {"id": v, "capacity_L": spec.vessel_capacity_m3(v) * 1000} for v in members
            ],
            "components": [
                {
                    "id": v + ":node",
                    "vessel": v,
                    "kind": "aqueous_reactive_transport"
                    if v == spec.anchor_vessel_id
                    else "material_input",
                    "parameters": {},
                }
                for v in members
            ]
            + [
                {
                    "id": v + ":clock",
                    "vessel": v,
                    "kind": "lab_schedule",
                    "parameters": {"duration_range_s": [1e-6, 14400]},
                }
                for v in members
            ],
            "connections": [
                {
                    "id": identity,
                    "unit": "mol",
                    "source": {"component": source + ":node", "port": "out"},
                    "target": {"component": target + ":node", "port": "in"},
                }
                for identity, source, target in spec.topology
            ],
            "physics": CompositionPhysics(
                network,
                states,
                lab_clock=LabClockSpec(heater_ids=(), analyzers=(), vessel_capacity=len(states)),
                advance_bindings={spec.anchor_vessel_id: "finite-aqueous-reactive-transport"},
            ).to_dict(),
            "task": {
                "objective": "study finite aqueous reaction and electrodiffusion",
                "budget": budget,
                "operations": ["wait", "terminate", "select_vessel"],
                "instruments": [],
            },
        }
    )
