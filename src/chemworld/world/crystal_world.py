"""A declared finite liquid/solid crystallizer using existing runtime operations."""

from chemworld.foundation import WorldState
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.heat_transfer_units import (
    CondensedBackgroundSpec,
    ThermalDeviceSpec,
    ThermalSystemSpec,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.full_process_contract import population_settings
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def crystal_world_spec(model, thermodynamics, component_formulas):
    target, impurity = model.target_component, model.impurity_component
    if impurity is None or set(component_formulas) != {target, impurity}:
        raise ValueError("research fixture needs authored target and impurity identities")
    name = "reactor"
    network = ReactionNetworkSpec(
        "finite-crystal-research",
        tuple(
            SpeciesSpec(key, formula, phase="liquid") for key, formula in component_formulas.items()
        ),
        (),
    )
    liquid, solid = {target: 0.02, impurity: 0.002}, {target: 0.001, impurity: 0.0001}
    count = 0.001 / model.particle_moles(100e-6)
    equipment = EquipmentLedger(
        {
            "crystallizer": EquipmentRecord(
                "crystallizer",
                "crystallizer",
                name,
                settings={
                    **population_settings(((count, 100e-6),)),
                    "population_constituents": [[0.0003, 0.0001, 0.001]],
                    "seed_target_mol": 0.0003,
                    "dissolved_seed_target_mol": 0.0002,
                },
            ),
            "heater": EquipmentRecord(
                "heater",
                "thermal_device",
                name,
                batch_reset="preserve",
                settings={
                    "thermal_device": ThermalDeviceSpec(
                        maximum_power_W=5,
                        maximum_cooling_power_W=5,
                        setpoint_K=298.15,
                        minimum_setpoint_K=250,
                        enabled=False,
                    ).to_dict()
                },
            ),
        }
    )
    state = WorldState(
        {key: liquid[key] + solid[key] for key in liquid},
        0.02,
        298.15,
        101325,
        "liquid",
        name,
        quenched=True,
        liquid_heat_capacity_J_L_K=2000,
        equipment=equipment,
        phases=PhaseLedger(
            {
                "mother_liquor": PhaseRecord(
                    "mother_liquor",
                    name,
                    "liquid",
                    0.02,
                    liquid,
                    selected=True,
                    settled=True,
                    solvents=SolventInventory.pure(0, 0.02),
                ),
                "solid": PhaseRecord("solid", name, "solid", 0, solid, settled=True),
            }
        ),
        vessels=VesselLedger(
            {
                name: VesselRecord(
                    name,
                    "finite-crystallizer",
                    0.1,
                    430,
                    500000,
                    ("liquid", "solid"),
                    298.15,
                    101325,
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
    thermal = ThermalSystemSpec(
        thermodynamics,
        pressure_law="fixed_pressure",
        environment_temperature_K=298.15,
        loss_conductance_W_K=0.1,
        minimum_temperature_K=250,
        maximum_temperature_K=430,
        stop_temperature_K=425,
        condensed_background=CondensedBackgroundSpec(2000),
    )
    physics = CompositionPhysics(
        network, {name: state}, thermal_models={name: thermal}, crystal_models={name: model}
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-crystal-research",
            "components": [
                {"id": "crystal", "kind": "crystal_thermal", "vessel": name, "parameters": {}},
                {
                    "id": "thermal",
                    "kind": "thermal",
                    "vessel": name,
                    "parameters": {
                        "temperature_range_K": [250, 425],
                        "duration_range_s": [1, 14400],
                    },
                },
                {
                    "id": "probe",
                    "kind": "observation",
                    "vessel": name,
                    "parameters": {"instruments": ["phase_state"]},
                },
            ],
            "vessels": [{"id": name, "capacity_L": 0.1}],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study chemical response",
                "budget": 120,
                "operations": [
                    "seed_crystals",
                    "cool_crystallize",
                    "filter_crystals",
                    "resuspend_crystals",
                    "create_container",
                    "transfer_material",
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
