"""A finite synthetic solid specimen on the existing native thermal device."""

from math import isfinite

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.solid_solutions import SolidSolutionSpec
from chemworld.runtime.solid_solutions import SOLID_TYPE, install_solid
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def solid_world_spec(model=None, *, x_A=0.5, amount_mol=0.001, temperature_K=300, budget=100):
    if not isfinite(x_A) or not 0 <= x_A <= 1:
        raise ValueError("solid composition must lie in [0, 1]")
    model = model or SolidSolutionSpec()
    amounts = {"A": amount_mol * x_A, "B": amount_mol * (1 - x_A)}
    equilibrium = model.at_tp(amounts, temperature_K)
    heater = ThermalDeviceSpec(
        conductance_W_K=0.05,
        maximum_power_W=2,
        maximum_cooling_power_W=2,
        setpoint_K=temperature_K,
        minimum_setpoint_K=model.minimum_temperature_K,
        maximum_setpoint_K=model.maximum_temperature_K,
    )
    state = WorldState(
        amounts,
        amount_mol * model.molar_volume_L_mol,
        temperature_K,
        model.pressure_Pa,
        "solid",
        "reactor",
        vessels=VesselLedger(
            {
                "reactor": VesselRecord(
                    "reactor",
                    "solid-specimen-holder",
                    0.01,
                    model.maximum_temperature_K,
                    200000,
                    ("solid",),
                    temperature_K,
                    model.pressure_Pa,
                )
            }
        ),
        equipment=EquipmentLedger(
            {
                "solid_material": EquipmentRecord(
                    "solid_material",
                    SOLID_TYPE,
                    "reactor",
                    settings={"physical_model": model.to_dict()},
                    batch_reset="preserve",
                ),
                "heater": EquipmentRecord(
                    "heater",
                    "thermal_device",
                    "reactor",
                    settings={"thermal_device": heater.to_dict()},
                    batch_reset="preserve",
                ),
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
    state = install_solid(state, model, equilibrium)
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "synthetic-binary-solid",
            (SpeciesSpec("A", "Al", phase="solid"), SpeciesSpec("B", "Cu", phase="solid")),
            (),
        ),
        {"reactor": state},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "binary-solid",
            "components": [
                {"id": "solid", "kind": "solid_solution", "vessel": "reactor", "parameters": {}},
                {
                    "id": "heater",
                    "kind": "thermal",
                    "vessel": "reactor",
                    "parameters": {
                        "temperature_range_K": [
                            model.minimum_temperature_K,
                            model.maximum_temperature_K,
                        ],
                        "duration_range_s": [1, 600],
                    },
                },
                {
                    "id": "meter",
                    "kind": "observation",
                    "vessel": "reactor",
                    "parameters": {"instruments": ["solid_phase_meter"]},
                },
            ],
            "vessels": [{"id": "reactor", "capacity_L": 0.01}],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite binary solid phase equilibrium",
                "budget": budget,
                "operations": [
                    "heat",
                    "wait",
                    "pause_control",
                    "resume_control",
                    "measure",
                    "terminate",
                ],
                "instruments": ["solid_phase_meter"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
