"""Original finite thermal world for neutral multiphase CEF specimens."""

from collections.abc import Mapping
from typing import cast

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.calphad_caloric_equilibrium import solve_calphad_ph
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.calphad_thermal import CalphadThermalSpec, install_calphad_thermal_state
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_calphad_thermal_world(
    spec: CalphadThermalSpec,
    amounts_mol: Mapping[str, float],
    *,
    enthalpy_J: float,
    device: ThermalDeviceSpec,
    max_volume_L: float = 0.1,
    budget: int = 100,
) -> WorldCompositionSpec:
    result = solve_calphad_ph(
        spec.caloric,
        amounts_mol,
        enthalpy_J,
        support_heat_capacity_J_K=spec.support_heat_capacity_J_K,
        support_reference_temperature_K=spec.reference_temperature_K,
    )
    if not result.success:
        raise ValueError("initial CALPHAD PH failed: " + result.failure_reason)
    records = {
        "calphad-model": EquipmentRecord(
            "calphad-model",
            "calphad_material",
            "reactor",
            batch_reset="preserve",
            settings={"model_id": "calphad_material", "vessel": "reactor"},
        ),
        "calphad-heater": EquipmentRecord(
            "calphad-heater",
            "thermal_device",
            "reactor",
            batch_reset="preserve",
            settings={"thermal_device": device.to_dict()},
        ),
    }
    state = WorldState(
        dict(amounts_mol),
        0.001,
        cast(float, result.temperature_K),
        spec.pressure_Pa,
        "solid",
        "reactor",
        equipment=EquipmentLedger(records),
        vessels=VesselLedger(
            {
                "reactor": VesselRecord(
                    "reactor",
                    "constant-pressure-calphad",
                    max_volume_L,
                    spec.maximum_temperature_K,
                    spec.pressure_Pa,
                    (),
                    cast(float, result.temperature_K),
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
    state = install_calphad_thermal_state(state, spec, result)
    network = ReactionNetworkSpec(
        "calphad-finite-heat",
        tuple(SpeciesSpec(e, symbol, phase="solid") for e, symbol in spec.element_symbols.items()),
        (),
    )
    physics = CompositionPhysics(
        network, {"reactor": state}, models={"calphad_material": {"reactor": spec}}
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "calphad-multiphase-thermal",
            "components": [
                {"id": "material", "kind": "calphad_phase", "vessel": "reactor", "parameters": {}},
                {
                    "id": "thermal",
                    "kind": "thermal",
                    "vessel": "reactor",
                    "parameters": {
                        "temperature_range_K": [
                            spec.minimum_temperature_K,
                            spec.maximum_temperature_K,
                        ],
                        "duration_range_s": [1, 600],
                    },
                },
                {
                    "id": "sensor",
                    "kind": "observation",
                    "vessel": "reactor",
                    "parameters": {"instruments": ["phase_state"]},
                },
            ],
            "vessels": [{"id": "reactor", "capacity_L": max_volume_L}],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study same-reference latent heat and actual phase stock",
                "budget": budget,
                "operations": [
                    "heat",
                    "wait",
                    "pause_control",
                    "resume_control",
                    "measure",
                    "terminate",
                ],
                "instruments": ["phase_state"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
