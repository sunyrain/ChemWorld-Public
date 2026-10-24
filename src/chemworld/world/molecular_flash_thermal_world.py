"""An authored molecular two-phase vessel on the original world composition."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_caloric_flash import (
    MolecularCaloricFlashSeed,
    solve_molecular_caloric_flash,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.molecular_flash_thermal import (
    MolecularFlashThermalSpec,
    install_molecular_caloric_flash,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_molecular_flash_thermal_world(
    spec: MolecularFlashThermalSpec,
    feed_amounts_mol: Mapping[str, float],
    *,
    enthalpy_J: float,
    seed: MolecularCaloricFlashSeed,
    formulas: Mapping[str, str],
    device: ThermalDeviceSpec,
    max_volume_L: float = 0.1,
    budget: int = 100,
) -> WorldCompositionSpec:
    ids = spec.equilibrium.model.component_ids
    if set(formulas) != set(ids):
        raise ValueError("world species require their explicit formulas")
    result = solve_molecular_caloric_flash(
        spec.equilibrium,
        feed_amounts_mol,
        coordinate="H",
        value=enthalpy_J,
        seed=seed,
        support_heat_capacity_J_K=spec.support_heat_capacity_J_K,
        support_reference_temperature_K=spec.support_reference_temperature_K,
    )
    if not result.success:
        raise ValueError("initial molecular world closure failed: " + result.failure_reason)
    record = EquipmentRecord(
        "molecular-heater",
        "thermal_device",
        "reactor",
        batch_reset="preserve",
        settings={"thermal_device": device.to_dict()},
    )
    state = WorldState(
        dict(feed_amounts_mol),
        0.001,
        seed.temperature_K,
        spec.pressure_Pa,
        "liquid",
        "reactor",
        equipment=EquipmentLedger({record.equipment_id: record}),
        vessels=VesselLedger(
            {
                "reactor": VesselRecord(
                    "reactor",
                    "constant-pressure-molecular",
                    max_volume_L,
                    spec.equilibrium.temperature_range_K[1],
                    spec.pressure_Pa,
                    (),
                    seed.temperature_K,
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
    state = install_molecular_caloric_flash(state, spec, result)
    network = ReactionNetworkSpec(
        "authored-molecular-caloric", tuple(SpeciesSpec(k, formulas[k]) for k in ids), ()
    )
    physics = CompositionPhysics(network, {"reactor": state}, models={"flash": {"reactor": spec}})
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "molecular-caloric-finite-thermal",
            "components": [
                {
                    "id": "reactor:thermal",
                    "kind": "thermal",
                    "vessel": "reactor",
                    "parameters": {
                        "temperature_range_K": list(spec.equilibrium.temperature_range_K),
                        "duration_range_s": [1, 600],
                    },
                },
                {
                    "id": "reactor:vle",
                    "kind": "vapor_liquid_equilibrium",
                    "vessel": "reactor",
                    "parameters": {},
                },
                {
                    "id": "reactor:sensor",
                    "kind": "observation",
                    "vessel": "reactor",
                    "parameters": {"instruments": ["phase_state"]},
                },
            ],
            "vessels": [{"id": "reactor", "capacity_L": max_volume_L}],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite heat on actual molecular phases",
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
