"""Explicit finite vacuum-concentrator composition with an owned receiving bottle."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.concentration import MODEL_ID, install_concentration
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def concentration_world_spec(spec, initial, *, heater=None, capacity_L=0.1):
    vessel = "concentrator"
    network = ReactionNetworkSpec(
        "authored-finite-concentrator",
        tuple(
            SpeciesSpec(p.component.identifier, p.component.formula)
            for p in spec.properties.packages
        ),
        (),
    )
    equipment = EquipmentLedger(
        {
            "concentrator:heater": EquipmentRecord(
                "concentrator:heater",
                "thermal_device",
                vessel,
                batch_reset="preserve",
                settings={
                    "thermal_device": (
                        ThermalDeviceSpec(
                            maximum_power_W=spec.equipment.max_heater_power_W,
                            setpoint_K=spec.equipment.maximum_temperature_K,
                            enabled=False,
                        )
                        if heater is None
                        else heater
                    ).to_dict()
                },
            ),
            "vacuum_concentrator": EquipmentRecord(
                "vacuum_concentrator",
                "finite_vacuum_concentrator",
                vessel,
                batch_reset="preserve",
                settings={"transport_enabled": 0, "concentration_model": MODEL_ID},
            ),
        }
    )
    amounts = {k: sum(n[k] for n in initial.amounts_mol) for k in spec.properties.component_ids}
    state = WorldState(
        amounts,
        0.001,
        300,
        spec.pressure_Pa,
        "liquid",
        vessel,
        equipment=equipment,
        units={
            "amount": "mol",
            "volume": "L",
            "temperature": "K",
            "pressure": "Pa",
            "time": "s",
            "cost": "currency",
            "risk": "risk",
        },
        vessels=VesselLedger(
            {
                vessel: VesselRecord(
                    vessel, "finite-concentrator", capacity_L, 450, 5e5, (), 300, spec.pressure_Pa
                )
            }
        ),
    )
    state = install_concentration(state, spec, initial)
    physics = CompositionPhysics(network, {vessel: state}, concentration_models={vessel: spec})
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-concentrator",
            "vessels": [{"id": vessel, "capacity_L": capacity_L}],
            "connections": [],
            "components": [
                {
                    "id": "concentrator:material",
                    "kind": "material_input",
                    "vessel": vessel,
                    "parameters": {},
                },
                {
                    "id": "concentrator:thermal",
                    "kind": "thermal",
                    "vessel": vessel,
                    "parameters": {
                        "temperature_range_K": [250, 430],
                        "duration_range_s": [1, 14400],
                    },
                },
                {
                    "id": "concentrator:separation",
                    "kind": "vacuum_concentration",
                    "vessel": vessel,
                    "parameters": {},
                },
                {
                    "id": "concentrator:probe",
                    "kind": "observation",
                    "vessel": vessel,
                    "parameters": {"instruments": ["phase_state"]},
                },
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study chemical response",
                "budget": 40,
                "operations": [
                    "concentrate",
                    "heat",
                    "wait",
                    "measure",
                    "pause_control",
                    "resume_control",
                    "terminate",
                ],
                "instruments": ["phase_state"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
