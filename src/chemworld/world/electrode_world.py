"""An explicit finite isothermal working-electrode world, with task-independent physics."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.transient_electrode import TransientElectrodeSpec
from chemworld.runtime.transient_electrode import (
    ELECTRODE_TYPE,
    initial_charge_slot,
    material_phases,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def electrode_world_spec(spec=None, *, concentration_O=1, concentration_R=1, budget=40):
    spec = spec or TransientElectrodeSpec()
    vessel = "reactor"
    physical = spec.initial_state(concentration_O, concentration_R)
    record = EquipmentRecord(
        "working-electrode",
        ELECTRODE_TYPE,
        vessel,
        batch_reset="preserve",
        settings={
            "physical_model": spec.to_dict(),
            "electrical_control": {
                "limits_V": [-spec.maximum_potential_V, spec.maximum_potential_V],
                "program": ScalarControlProgram(((0, 0),), "V").to_dict(),
            },
        },
    )
    state = WorldState(
        {
            spec.oxidized_species: concentration_O * spec.volume_l / 1000,
            spec.reduced_species: concentration_R * spec.volume_l / 1000,
        },
        spec.volume_l,
        spec.temperature_K,
        101325,
        "liquid",
        vessel,
        units={
            "amount": "mol",
            "volume": "L",
            "temperature": "K",
            "pressure": "Pa",
            "time": "s",
            "cost": "currency",
            "risk": "risk",
        },
        equipment=EquipmentLedger({record.equipment_id: record}),
        vessels=VesselLedger(
            {
                vessel: VesselRecord(
                    vessel, "finite-layer-cell", 0.01, 400, 5e5, (), spec.temperature_K, 101325
                )
            }
        ),
        phases=material_phases(spec, physical.amounts_mol, vessel),
    )
    state = state.replace(
        state_slots=StateSlots((initial_charge_slot(state, record, physical.coordinates),))
    )
    # One-electron Fe(III)/Fe(II) identities; this is a synthetic parameter set,
    # not calibrated aqueous iron chemistry or a counterelectrode model.
    network = ReactionNetworkSpec(
        "supported-planar-redox",
        (
            SpeciesSpec(spec.oxidized_species, "Fe", charge=3),
            SpeciesSpec(spec.reduced_species, "Fe", charge=2),
        ),
        (),
    )
    physics = CompositionPhysics(network, {vessel: state})
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-transient-electrode",
            "components": [
                {
                    "id": "electrode",
                    "kind": "transient_electrode",
                    "vessel": vessel,
                    "parameters": {},
                },
                {
                    "id": "meter",
                    "kind": "observation",
                    "vessel": vessel,
                    "parameters": {"instruments": ["electrical_meter"]},
                },
            ],
            "vessels": [{"id": vessel, "capacity_L": 0.01}],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite electrochemical response",
                "budget": budget,
                "operations": ["configure_electrical_control", "wait", "measure", "terminate"],
                "instruments": ["electrical_meter"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
