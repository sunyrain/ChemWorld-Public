"""Explicit single-apparatus finite equilibrium-column composition."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.dynamic_column import MODEL_ID, install_column
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def dynamic_column_world_spec(spec, initial, *, heater=None, capacity_L=0.1):
    vessel = "column"
    network = ReactionNetworkSpec(
        "authored-finite-column",
        tuple(
            SpeciesSpec(p.component.identifier, p.component.formula)
            for p in spec.properties.packages
        ),
        (),
    )
    equipment = EquipmentLedger(
        {
            "column:heater": EquipmentRecord(
                "column:heater",
                "thermal_device",
                vessel,
                batch_reset="preserve",
                settings={
                    "thermal_device": (ThermalDeviceSpec() if heater is None else heater).to_dict()
                },
            ),
            "distillation_column": EquipmentRecord(
                "distillation_column",
                "dynamic_distillation_column",
                vessel,
                batch_reset="preserve",
                settings={"reflux_ratio": 1.0, "full_reflux": 0, "distillation_model": MODEL_ID},
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
                    vessel, "finite-column", capacity_L, 450, 5e5, (), 300, spec.pressure_Pa
                )
            }
        ),
    )
    state = install_column(state, spec, initial)
    physics = CompositionPhysics(network, {vessel: state}, models={"column": {vessel: spec}})
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-column",
            "vessels": [{"id": vessel, "capacity_L": capacity_L}],
            "connections": [],
            "components": [
                {
                    "id": "column:material",
                    "kind": "material_input",
                    "vessel": vessel,
                    "parameters": {},
                },
                {
                    "id": "column:thermal",
                    "kind": "thermal",
                    "vessel": vessel,
                    "parameters": {
                        "temperature_range_K": [250, 430],
                        "duration_range_s": [1, 14400],
                    },
                },
                {
                    "id": "column:separation",
                    "kind": "dynamic_distillation",
                    "vessel": vessel,
                    "parameters": {},
                },
                {
                    "id": "column:probe",
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
                    "distill",
                    "collect_fraction",
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
