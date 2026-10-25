"""Author a general molecular equilibrium vessel on the original world runtime."""

from collections.abc import Mapping
from typing import cast

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_multiphase import (
    MolecularMultiphaseSeed,
    solve_molecular_multiphase,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.molecular_multiphase_thermal import (
    MolecularMultiphaseThermalSpec,
    install_molecular_multiphase_thermal,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_molecular_multiphase_world(
    spec: MolecularMultiphaseThermalSpec,
    feed_amounts_mol: Mapping[str, float],
    *,
    energy_J: float,
    seed: MolecularMultiphaseSeed,
    formulas: Mapping[str, str],
    device: ThermalDeviceSpec,
    max_volume_L: float,
    max_pressure_Pa: float,
    budget: int = 100,
) -> WorldCompositionSpec:
    """Initial energy is H at fixed P, or U at fixed V, including declared support."""
    equilibrium = spec.inventory.equilibrium
    ids = equilibrium.model.component_ids
    if set(formulas) != set(ids):
        raise ValueError("world species require their explicit formulas")
    result = solve_molecular_multiphase(
        equilibrium, feed_amounts_mol,
        inputs={spec.boundary: spec.boundary_value, spec.coordinate: energy_J}, seed=seed,
        support_heat_capacity_J_K=spec.support_heat_capacity_J_K,
        support_reference_temperature_K=spec.support_reference_temperature_K,
    )
    if not result.success:
        raise ValueError("initial molecular world closure failed: " + result.failure_reason)
    temperature, pressure = cast(float, result.temperature_K), cast(float, result.pressure_Pa)
    heater = EquipmentRecord(
        "molecular-heater", "thermal_device", "reactor", batch_reset="preserve",
        settings={"thermal_device": device.to_dict()},
    )
    state = WorldState(
        dict(feed_amounts_mol), cast(float, result.volume_m3) * 1000,
        temperature, pressure, "liquid", "reactor",
        equipment=EquipmentLedger({heater.equipment_id: heater}),
        vessels=VesselLedger({"reactor": VesselRecord(
            "reactor", "molecular-multiphase", max_volume_L,
            equilibrium.temperature_bounds_K[1], max_pressure_Pa, (), temperature, pressure,
        )}),
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa",
               "time": "s", "cost": "currency", "risk": "risk"},
    )
    state = install_molecular_multiphase_thermal(state, spec, result)
    network = ReactionNetworkSpec(
        "authored-molecular-multiphase", tuple(SpeciesSpec(k, formulas[k]) for k in ids), ()
    )
    physics = CompositionPhysics(network, {"reactor": state}, models={"flash": {"reactor": spec}})
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "molecular-multiphase-finite-thermal",
        "components": [
            {"id": "reactor:thermal", "kind": "thermal", "vessel": "reactor", "parameters": {
                "temperature_range_K": list(equilibrium.temperature_bounds_K),
                "duration_range_s": [1, 600],
            }},
            {"id": "reactor:vle", "kind": "vapor_liquid_equilibrium", "vessel": "reactor",
             "parameters": {}},
            {"id": "reactor:sensor", "kind": "observation", "vessel": "reactor",
             "parameters": {"instruments": ["phase_state"]}},
        ],
        "vessels": [{"id": "reactor", "capacity_L": max_volume_L}],
        "connections": [], "physics": physics.to_dict(),
        "task": {
            "objective": "explore molecular phase changes with finite heat",
            "budget": budget,
            "operations": ["heat", "wait", "pause_control", "resume_control", "measure",
                           "terminate"],
            "instruments": ["phase_state"], "termination": "explicit-stop-or-budget",
        },
    })
