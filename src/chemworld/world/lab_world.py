"""Two independently authored bottles sharing one heater and chromatograph."""

from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.coupled_system import ConstantCvThermodynamics, SpeciesInternalEnergy
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.world.chromatography_world import chromatography_world_spec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def shared_lab_world():
    """Development fixture, not a chemical parameter fit or autonomous campaign."""
    source = chromatography_world_spec()
    physics = source.physics
    # Distinct existing recipes, not copies of one sample. A finite heater
    # changes sample temperatures; chromatography remains an offline method.
    states = {key: physics.initial_states[key] for key in ("reactor", "sample")}
    reactor = states["reactor"]
    reactor = reactor.replace(
        species_amounts=dict.fromkeys(reactor.species_amounts, 1e-05),
        phases=None,
        equipment=EquipmentLedger(
            {
                **reactor.equipment.equipment,
                "heater": EquipmentRecord(
                    "heater",
                    "thermal_device",
                    "reactor",
                    settings={"thermal_device": ThermalDeviceSpec(enabled=False).to_dict()},
                ),
            }
        ),
    )
    states["reactor"] = reactor
    models = {
        key: ThermalSystemSpec(
            ConstantCvThermodynamics(
                {s.species_id: SpeciesInternalEnergy(0.0, 30.0) for s in physics.network.species},
                20.0,
            ),
            pressure_law="fixed_pressure",
            minimum_temperature_K=290.0,
        )
        for key in states
    }
    composition = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "native-shared-lab",
        "vessels": [{"id": key, "capacity_L": 0.1} for key in states],
        "components": [
            {
                "id": key + ":" + kind,
                "vessel": key,
                "kind": kind,
                "parameters": {"instruments": ["programmed_hplc"]} if kind == "observation" else {},
            }
            for key in states
            for kind in ("thermal", "observation", "lab_schedule")
        ],
        "physics": CompositionPhysics(
            physics.network,
            states,
            thermal_models=models,
            lab_clock=LabClockSpec("heater", "programmed_hplc"),
        ).to_dict(),
        "task": {
            "objective": "characterize two vessels with shared finite apparatus",
            "budget": 80,
            "operations": [
                "select_vessel",
                "queue_lab_heat",
                "queue_lab_measure",
                "wait",
                "terminate",
            ],
            "instruments": ["programmed_hplc"],
            "termination": "explicit-stop-or-budget",
        },
    }
    return WorldCompositionSpec.from_dict(composition)
