"""Author finite gas nodes on the original world composition and shared clock."""

from collections.abc import Mapping
from math import isclose

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.gas_network_pressure import gas_node_volume, pressure_control_for
from chemworld.physchem.gas_network_surface import gas_phase_network, network_with_gas_surfaces
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    gas_material_topology,
    gas_node_pressure,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_thermal_wall import GAS_WALL_TYPE
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.runtime.gas_reactor_network import (
    GAS_NETWORK_TYPE,
    gas_accounting_slot,
    install_gas_node,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def gas_reactor_network_world(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
    *,
    budget: int = 100,
    anchor_thermal_device: ThermalDeviceSpec | None = None,
    surface_amounts_by_owner: Mapping[str, Mapping[str, float]] | None = None,
) -> WorldCompositionSpec:
    """Explicit material inputs, finite sources/sinks, one original shared owner."""
    surface_amounts_by_owner = surface_amounts_by_owner or {}
    if set(surface_amounts_by_owner) != {a.owner_id for a in spec.surfaces}:
        raise ValueError("initial surface amounts must cover the declared original catalyst owners")
    surface_slots = tuple(a.make_slot(surface_amounts_by_owner[a.owner_id]) for a in spec.surfaces)
    network = network_with_gas_surfaces(network, spec.surfaces)
    gas = gas_phase_network(network) if spec.surfaces else network
    pack_gas_reactor_network_state(spec, network, amounts_by_vessel, surface_slots=surface_slots)
    if anchor_thermal_device is not None and (
        not spec.energy_nodes
        or anchor_thermal_device.enabled
        or anchor_thermal_device.mode != "finite_power"
    ):
        raise ValueError(
            "gas shared heater requires thermal nodes and starts disabled in finite-power mode"
        )
    heater_id = f"{spec.anchor_vessel_id}:heater"
    wall = spec.thermal_wall
    energy_nodes = {e.vessel_id: e for e in spec.energy_nodes}
    states = {}
    for node in spec.nodes:
        name = node.vessel_id
        amounts = {s: float(amounts_by_vessel[name].get(s, 0)) for s in gas.species_ids}
        temperature = spec.initial_temperatures_K[name]
        control = pressure_control_for(node, spec.pressure_control)
        volume = gas_node_volume(node, sum(amounts.values()), temperature, control)
        if control and not isclose(volume, node.volume_L, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("initial piston n/T/P must match its declared reference volume")
        capacity = control.maximum_volume_L if control else node.volume_L
        pressure = gas_node_pressure(node, sum(amounts.values()), temperature, control)
        anchor = name == spec.anchor_vessel_id
        device = f"{name}:gas-network"
        state = WorldState(
            amounts,
            volume,
            temperature,
            pressure,
            "gas",
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
                        "finite-gas-vessel",
                        capacity,
                        energy_nodes[name].maximum_temperature_K
                        if spec.energy_nodes
                        else spec.temperature_K,
                        node.maximum_pressure_Pa,
                        ("gas",),
                        temperature,
                        pressure,
                    )
                }
            ),
            equipment=EquipmentLedger(
                (
                    {
                        device: EquipmentRecord(
                            device,
                            GAS_NETWORK_TYPE,
                            name,
                            batch_reset="preserve",
                            settings={"physical_model": spec.to_dict()},
                        ),
                        **(
                            {
                                wall.owner_id: EquipmentRecord(
                                    wall.owner_id, GAS_WALL_TYPE, name, batch_reset="preserve"
                                )
                            }
                            if wall is not None
                            else {}
                        ),
                        **(
                            {
                                heater_id: EquipmentRecord(
                                    heater_id,
                                    "thermal_device",
                                    name,
                                    batch_reset="preserve",
                                    settings={"thermal_device": anchor_thermal_device.to_dict()},
                                )
                            }
                            if anchor_thermal_device is not None
                            else {}
                        ),
                    }
                    if anchor
                    else {}
                )
                | {
                    a.owner_id: EquipmentRecord(
                        a.owner_id, "fixed_catalyst", name, batch_reset="preserve"
                    )
                    for a in spec.surfaces
                    if a.vessel_id == name
                }
            ),
            state_slots=StateSlots(
                ((gas_accounting_slot(spec, network, device),) if anchor else ())
                + ((wall.make_slot(name),) if anchor and wall is not None else ())
                + tuple(s for s in surface_slots if s.location_id == name)
            ),
        )
        states[name] = install_gas_node(state, node, temperature, amounts, pressure_control=control)
    # The active vessel is descriptive; all nodes advance as one physical system.
    states = {spec.anchor_vessel_id: states[spec.anchor_vessel_id], **states}
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": f"finite-gas-network:{network.network_id}",
            "vessels": [
                {
                    "id": n.vessel_id,
                    "capacity_L": spec.pressure_control.maximum_volume_L
                    if spec.pressure_control is not None
                    and n.vessel_id == spec.pressure_control.vessel_id
                    else n.volume_L,
                }
                for n in spec.nodes
            ],
            "components": [
                {
                    "id": f"{n.vessel_id}:node",
                    "vessel": n.vessel_id,
                    "kind": "gas_reactor_network"
                    if n.vessel_id == spec.anchor_vessel_id
                    else "material_input",
                    "parameters": {},
                }
                for n in spec.nodes
            ]
            + [
                {
                    "id": f"{n.vessel_id}:clock",
                    "vessel": n.vessel_id,
                    "kind": "lab_schedule",
                    "parameters": {"duration_range_s": [1e-6, 14400]},
                }
                for n in spec.nodes
            ]
            + (
                [
                    {
                        "id": "gas-chemistry",
                        "vessel": spec.anchor_vessel_id,
                        "kind": "reaction",
                        "parameters": {},
                    }
                ]
                if network.reactions
                else []
            ),
            "connections": [
                {
                    "id": identity,
                    "unit": "mol",
                    "source": {"component": f"{source}:node", "port": "out"},
                    "target": {"component": f"{target}:node", "port": "in"},
                }
                for identity, source, target in gas_material_topology(spec)
            ],
            "physics": CompositionPhysics(
                network,
                states,
                lab_clock=LabClockSpec(
                    heater_ids=(heater_id,) if anchor_thermal_device else (),
                    analyzers=(),
                    vessel_capacity=len(states),
                ),
                advance_bindings={spec.anchor_vessel_id: "finite-isothermal-gas-network"},
            ).to_dict(),
            "task": {
                "objective": "study finite gas reaction and transport",
                "budget": budget,
                "operations": ["wait", "terminate"]
                + (["queue_lab_heat"] if anchor_thermal_device else [])
                + (["select_vessel"] if len(states) > 1 else []),
                "instruments": [],
            },
        }
    )
