"""One finite gas network participates in the existing shared-clock transaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isclose
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import SharedAdvanceSystem
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
)
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.gas_network_pressure import (
    GasPressureControlSpec,
    gas_node_volume,
    pressure_control_for,
)
from chemworld.physchem.gas_network_surface import GasSurfaceContext, gas_phase_network
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    advance_gas_reactor_network,
    gas_accounting_coordinates,
    gas_inventory_key,
    gas_material_topology,
    gas_node_pressure,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_thermal_wall import GAS_WALL_TYPE
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

GAS_NETWORK_TYPE = "finite_isothermal_gas_network"
GAS_ACCOUNTING_TYPE = "gas_network_accounting_v1"

if TYPE_CHECKING:
    from chemworld.world.composition import WorldCompositionSpec


def gas_network_record(state: WorldState) -> EquipmentRecord | None:
    return attached_device_record(state, GAS_NETWORK_TYPE)


def gas_surface_slots(state: WorldState, spec: GasReactorNetworkSpec) -> tuple[StateSlot, ...]:
    owners = {a.owner_id for a in spec.surfaces}
    return tuple(
        slot
        for local in (state, *state.inactive_vessels.values())
        for slot in local.state_slots.slots
        if slot.spec.owner_id in owners
    )


def gas_accounting_slot(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    device: str,
    values: Mapping[str, float] | None = None,
) -> StateSlot:
    coordinates = gas_accounting_coordinates(spec, network)
    return StateSlot(
        StateSlotSpec(
            f"{device}:accounting",
            "equipment",
            device,
            GAS_ACCOUNTING_TYPE,
            tuple(StateFieldSpec(c.name, c.unit, minimum=None) for c in coordinates),
            mechanism_id=network.network_id,
            batch_reset="preserve",
        ),
        spec.anchor_vessel_id,
        {c.name: 0.0 for c in coordinates} if values is None else dict(values),
    )


def install_gas_node(
    state: WorldState,
    node: GasReactorNodeSpec,
    temperature: float,
    amounts: Mapping[str, float],
    *,
    pressure_control: GasPressureControlSpec | None = None,
) -> WorldState:
    """Update the original phase inventory; counters are never material stocks."""
    if not (state.process is not None):
        raise ValueError("gas node update requires process state")
    volume = gas_node_volume(node, sum(amounts.values()), temperature, pressure_control)
    return state.replace(
        species_amounts=dict(amounts),
        volume_L=volume,
        temperature_K=temperature,
        pressure_Pa=gas_node_pressure(node, sum(amounts.values()), temperature, pressure_control),
        phases=PhaseLedger(
            {
                "gas": PhaseRecord(
                    "gas",
                    state.vessel_id,
                    "gas",
                    volume,
                    dict(amounts),
                    settled=True,
                    selected=True,
                    metadata={"carrier_representation": "species_inventory"},
                )
            }
        ),
        process=replace(state.process, last_observation_record=None),
    )


def unpack_gas_network(
    state: WorldState,
    network: ReactionNetworkSpec,
) -> tuple[EquipmentRecord, GasReactorNetworkSpec, CoupledState]:
    record = gas_network_record(state)
    if record is None:
        raise ValueError("no gas network is installed")
    spec = GasReactorNetworkSpec.from_dict(record.settings["physical_model"])
    shared_network = network
    slots = gas_surface_slots(state, spec)
    surface = GasSurfaceContext(spec, network, slots) if spec.surfaces else None
    if surface is not None:
        network = surface.gas
    if state.vessel_id != spec.anchor_vessel_id:
        raise ValueError("gas network device must belong to its declared anchor")
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    wall_slots = [
        s
        for node in spec.nodes
        for local in (locals_.get(node.vessel_id),)
        if local is not None
        for s in local.state_slots.slots
        if s.spec.state_type == GAS_WALL_TYPE
    ]
    wall_slot = None
    if spec.thermal_wall is not None:
        wall = spec.thermal_wall
        if len(wall_slots) != 1:
            raise ValueError("thermal wall needs exactly one original equipment slot")
        wall_slot = wall_slots[0]
        wall.validate_slot(wall_slot, spec.anchor_vessel_id)
        equipment = (
            None if state.equipment is None else state.equipment.equipment.get(wall.owner_id)
        )
        if (
            equipment is None
            or equipment.equipment_type != GAS_WALL_TYPE
            or equipment.attached_vessel_id != spec.anchor_vessel_id
            or equipment.status == "parked"
        ):
            raise ValueError("thermal wall requires its fixed original anchor equipment")
    elif wall_slots:
        raise ValueError("undeclared thermal wall slot cannot be ignored")
    expected = gas_accounting_slot(spec, network, record.equipment_id)
    owned = [s for s in state.state_slots.slots if s.spec.state_type == GAS_ACCOUNTING_TYPE]
    if len(owned) != 1 or owned[0].spec != expected.spec or owned[0].location_id != state.vessel_id:
        raise ValueError("gas accounting must have exactly one declared equipment owner")
    amounts = {}
    for node in spec.nodes:
        local = locals_.get(node.vessel_id)
        if local is None:
            raise ValueError("gas network is missing a finite node")
        if local.phases is None:
            raise ValueError("gas node requires its original phase ledger")
        control = pressure_control_for(node, spec.pressure_control)
        expected_volume = gas_node_volume(
            node, sum(local.species_amounts.values()), local.temperature_K, control
        )
        volume_matches = (
            local.volume_L == node.volume_L
            if control is None
            else isclose(local.volume_L, expected_volume, rel_tol=1e-10, abs_tol=1e-12)
        )
        if (
            set(local.species_amounts) != set(network.species_ids)
            or set(local.phases.phases) != {"gas"}
            or local.phases.phases["gas"].phase_id != "gas"
            or local.phases.phases["gas"].vessel_id != node.vessel_id
            or local.phases.phases["gas"].phase_type != "gas"
            or local.phases.phases["gas"].solvents.volume_L != 0
            or local.phases.phases["gas"].species_amounts_mol != local.species_amounts
            or local.phases.phases["gas"].volume_L != local.volume_L
            or not volume_matches
            or (not spec.energy_nodes and local.temperature_K != spec.temperature_K)
        ):
            raise ValueError(
                "gas node volume, temperature or phase inventory differs from declaration"
            )
        expected_pressure = gas_node_pressure(
            node, sum(local.species_amounts.values()), local.temperature_K, spec.pressure_control
        )
        if not isclose(local.pressure_Pa, expected_pressure, rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError("gas pressure differs from the original finite inventory")
        for slot in local.state_slots.slots:
            if wall_slot is not None and slot == wall_slot:
                continue
            if node.vessel_id == spec.anchor_vessel_id and slot == owned[0]:
                continue
            if (
                surface is not None
                and node.vessel_id in surface.slots
                and slot == surface.slots[node.vessel_id]
            ):
                equipment = (
                    None
                    if local.equipment is None
                    else local.equipment.equipment.get(slot.spec.owner_id)
                )
                if (
                    equipment is None
                    or equipment.equipment_type != "fixed_catalyst"
                    or equipment.attached_vessel_id != node.vessel_id
                ):
                    raise ValueError("catalytic slot requires its actual attached fixed catalyst")
                continue
            if (
                slot.spec.owner_kind != "equipment"
                or slot.spec.time_evolution != "static"
                or slot.spec.heat_capacity_field is not None
                or slot.spec.energy_interactions
            ):
                raise ValueError("gas network cannot silently ignore additional coupled state")
        amounts[node.vessel_id] = dict(local.species_amounts)
    packed = pack_gas_reactor_network_state(
        spec,
        shared_network,
        amounts,
        time_s=state.ledger.time_s,
        counters=owned[0].values,
        temperatures_by_vessel_K={
            n.vessel_id: locals_[n.vessel_id].temperature_K for n in spec.nodes
        },
        surface_slots=slots,
        wall_temperature_K=None if wall_slot is None else wall_slot.values["temperature_K"],
    )
    device = gas_anchor_thermal_device(state, spec)
    gas_reactor_network_system(
        spec, shared_network, thermal_device=device, surface_slots=slots
    ).validate_state(packed)
    return record, spec, packed


def gas_anchor_thermal_device(
    state: WorldState, spec: GasReactorNetworkSpec
) -> ThermalDeviceSpec | None:
    """Only the authored fixed anchor may own one original bounded thermal device."""
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    devices = []
    for node in spec.nodes:
        equipment = cast(EquipmentLedger, locals_[node.vessel_id].equipment)
        for record in equipment.equipment.values():
            if record.equipment_type != "thermal_device":
                continue
            if (
                not spec.energy_nodes
                or node.vessel_id != spec.anchor_vessel_id
                or record.attached_vessel_id != spec.anchor_vessel_id
                or record.status == "parked"
            ):
                raise ValueError("gas network heater must remain fixed to its thermal anchor")
            devices.append(ThermalDeviceSpec.from_dict(record.settings["thermal_device"]))
    if len(devices) > 1:
        raise ValueError("gas network supports one anchor thermal device")
    return devices[0] if devices else None


class GasReactorNetworkRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: WorldCompositionSpec):
        self.composition = composition

    def supports(self, state: WorldState) -> bool:
        return gas_network_record(state) is not None

    def _wall_binding(self, state: WorldState) -> bool:
        """Bind the new wall model; heater settings and legacy contracts stay independent."""
        physics = self.composition.physics
        original = None if physics is None else physics.initial_states.get(state.vessel_id)
        source = None if original is None else gas_network_record(original)
        current = gas_network_record(state)
        declared = (
            None
            if source is None
            else GasReactorNetworkSpec.from_dict(source.settings["physical_model"])
        )
        actual = (
            None
            if current is None
            else GasReactorNetworkSpec.from_dict(current.settings["physical_model"])
        )
        wall_model = bool(
            (declared and (declared.thermal_wall or declared.shaft_machine))
            or (actual and (actual.thermal_wall or actual.shaft_machine))
        )
        if wall_model and (
            source is None
            or current is None
            or source.equipment_id != current.equipment_id
            or declared != actual
        ):
            shaft_model = bool(
                (declared and declared.shaft_machine) or (actual and actual.shaft_machine)
            )
            raise ValueError(
                "shaft gas model differs from original composition"
                if shaft_model
                else "thermal wall gas model differs from original composition"
            )
        return wall_model

    def _unpack_bound(
        self, state: WorldState
    ) -> tuple[EquipmentRecord, GasReactorNetworkSpec, CoupledState]:
        self._wall_binding(state)
        if not (self.composition.physics is not None):
            raise ValueError("gas network requires composition physics")
        return unpack_gas_network(state, self.composition.physics.network)

    def validate_state(self, state: WorldState) -> None:
        if self._wall_binding(state):
            if not (self.composition.physics is not None):
                raise ValueError("gas network requires composition physics")
            unpack_gas_network(state, self.composition.physics.network)

    def temperature_bounds_k(self, state: WorldState) -> tuple[float, float] | None:
        record = gas_network_record(state)
        if record is None:
            return None
        spec = GasReactorNetworkSpec.from_dict(record.settings["physical_model"])
        equipment = cast(EquipmentLedger, state.equipment)
        if not spec.energy_nodes or not any(
            e.equipment_type == "thermal_device" and e.attached_vessel_id == state.vessel_id
            for e in equipment.equipment.values()
        ):
            return None
        energy = next(e for e in spec.energy_nodes if e.vessel_id == state.vessel_id)
        return energy.minimum_temperature_K, energy.maximum_temperature_K

    def shared_system(self, state: WorldState, strategy_id: str) -> SharedAdvanceSystem:
        record = gas_network_record(state)
        if record is None:
            raise ValueError("gas shared system requires its installed device")
        spec = GasReactorNetworkSpec.from_dict(record.settings["physical_model"])
        components = {c.id: c for c in self.composition.components}
        edges = {e["id"]: e for e in self.composition.connections}
        for identity, source, target in gas_material_topology(spec):
            actual = edges.get(identity)
            if (
                actual is None
                or actual["unit"] != "mol"
                or actual.get("scheduling") == "manual"
                or actual["source"]["port"] != "out"
                or actual["target"]["port"] != "in"
                or components[actual["source"]["component"]].vessel != source
                or components[actual["target"]["component"]].vessel != target
            ):
                raise ValueError(
                    "gas flow identity must map to its actual declared material connection"
                )
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            tuple(n.vessel_id for n in spec.nodes),
            tuple(identity for identity, _, _ in gas_material_topology(spec)),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        if self.composition.physics is None:
            return set()
        result: set[str] = set()
        for local in self.composition.physics.initial_states.values():
            record = gas_network_record(local)
            if record is not None:
                result.update(
                    n.vessel_id
                    for n in GasReactorNetworkSpec.from_dict(
                        record.settings["physical_model"]
                    ).nodes
                )
        return result

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def validate_world(self) -> None:
        if not (self.composition.physics is not None):
            raise ValueError("gas network requires composition physics")
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        declared = {
            c.vessel for c in self.composition.components if c.kind == "gas_reactor_network"
        }
        if installed != declared:
            raise ValueError("gas network components must match installed anchor devices")
        claimed: set[str] = set()
        for vessel in installed:
            state = states[vessel].replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            _, spec, _ = self._unpack_bound(state)
            names = {n.vessel_id for n in spec.nodes}
            if names & claimed:
                raise ValueError("gas networks cannot overlap physical ownership")
            claimed.update(names)
            self.shared_system(state, "finite-isothermal-gas-network")
            for node in spec.nodes:
                vessels = states[node.vessel_id].vessels
                if not (vessels is not None):
                    raise ValueError("gas network node requires vessel hardware")
                hardware = vessels.vessels[node.vessel_id]
                control = pressure_control_for(node, spec.pressure_control)
                if (
                    (control.maximum_volume_L if control else node.volume_L) > hardware.max_volume_L
                    or node.maximum_pressure_Pa > hardware.max_pressure_Pa
                    or (
                        next(
                            e.maximum_temperature_K
                            for e in spec.energy_nodes
                            if e.vessel_id == node.vessel_id
                        )
                        if spec.energy_nodes
                        else spec.temperature_K
                    )
                    > hardware.max_temperature_K
                ):
                    raise ValueError("gas model exceeds declared vessel hardware")

    def advance(
        self,
        state: WorldState,
        action: Mapping[str, Any],
        *,
        heat: bool = False,
        jacket_program: Any = None,
    ) -> ModelProviderResult:
        record = gas_network_record(state)
        declared = (
            None
            if record is None
            else GasReactorNetworkSpec.from_dict(record.settings["physical_model"])
        )
        if jacket_program is not None or (
            (heat or "target_temperature_K" in action)
            and (declared is None or not declared.energy_nodes)
        ):
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="gas_network_requires_declared_isotherm",
            )
        if not (self.composition.physics is not None):
            raise ValueError("gas network requires composition physics")
        network = self.composition.physics.network
        try:
            working = state
            if heat:
                from chemworld.runtime.coupled_thermal import connected_device

                device_record, device = connected_device(state)
                device = replace(
                    device, setpoint_K=float(action["target_temperature_K"]), enabled=True
                )
                equipment = dict(cast(EquipmentLedger, state.equipment).equipment)
                equipment[device_record.equipment_id] = replace(
                    device_record,
                    settings={**device_record.settings, "thermal_device": device.to_dict()},
                )
                working = state.replace(equipment=EquipmentLedger(equipment))
            _, spec, packed = self._unpack_bound(working)
            slots = gas_surface_slots(working, spec)
            surface = GasSurfaceContext(spec, network, slots) if spec.surfaces else None
            gas = gas_phase_network(network) if surface is not None else network
            thermal_device = gas_anchor_thermal_device(working, spec)
            result, diagnostic = advance_gas_reactor_network(
                spec,
                network,
                packed,
                float(action["duration_s"]),
                thermal_device=thermal_device,
                surface_slots=slots,
            )
            final, dt = result.final_state, result.actual_duration_s
            thermal_deltas = {
                name: {
                    "energy_jacket_J": detail["bath_heat_J"],
                    "heat_reaction_J": -detail["chemical_energy_change_J"],
                    **({"heat_loss_J": detail["heat_loss_J"]} if spec.energy_nodes else {}),
                    **(
                        {"boundary_work_J": -detail["boundary_work_on_J"]}
                        if spec.pressure_control
                        else {}
                    ),
                }
                for name, detail in diagnostic["nodes"].items()
            }
            if spec.thermal_wall is not None:
                # The wall is fixed anchor equipment. Its external heat belongs
                # to that local thermal ledger, while gas-only node diagnostics
                # keep their own first law. The shared resource ledger receives
                # the already aggregated loss exactly once below.
                thermal_deltas[spec.anchor_vessel_id]["heat_loss_J"] += diagnostic["thermal_wall"][
                    "heat_loss_J"
                ]
            locals_ = {working.vessel_id: working, **working.inactive_vessels}
            updated = {}
            for node in spec.nodes:
                local = install_gas_node(
                    locals_[node.vessel_id],
                    node,
                    diagnostic["nodes"][node.vessel_id]["temperature_K"],
                    {
                        s: final.amounts_mol[gas_inventory_key(node.vessel_id, s)]
                        for s in gas.species_ids
                    },
                    pressure_control=spec.pressure_control,
                )
                if spec.thermal_wall is not None:
                    wall = spec.thermal_wall
                    local = local.replace(
                        state_slots=StateSlots(
                            tuple(
                                wall.project(final.coordinates, s)
                                if s.spec.owner_id == wall.owner_id
                                else s
                                for s in local.state_slots.slots
                            )
                        )
                    )
                updated[node.vessel_id] = local.replace(
                    **(
                        {
                            "state_slots": StateSlots(
                                tuple(
                                    surface.project(final, s)
                                    if s.spec.owner_id in {a.owner_id for a in spec.surfaces}
                                    else s
                                    for s in local.state_slots.slots
                                )
                            )
                        }
                        if surface is not None
                        else {}
                    ),
                    ledger=advance_ledger(
                        local.ledger,
                        dt,
                        energy_deltas=thermal_deltas[node.vessel_id],
                    ),
                )
            candidate = updated.pop(state.vessel_id)
            previous = next(
                s for s in state.state_slots.slots if s.spec.state_type == GAS_ACCOUNTING_TYPE
            )
            candidate = candidate.replace(
                state_slots=StateSlots(
                    tuple(
                        replace(
                            previous,
                            values={key: final.coordinates[key] for key in previous.values},
                        )
                        if s.spec.slot_id == previous.spec.slot_id
                        else s
                        for s in candidate.state_slots.slots
                    )
                ),
                inactive_vessels={**state.inactive_vessels, **updated},
                vessel_elapsed_s={
                    **state.vessel_elapsed_s,
                    **{
                        n.vessel_id: state.vessel_elapsed_s.get(n.vessel_id, 0) + dt
                        for n in spec.nodes
                    },
                },
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    energy_deltas={
                        "energy_jacket_J": diagnostic["bath_heat_J"],
                        "heat_reaction_J": sum(
                            row["heat_reaction_J"] for row in thermal_deltas.values()
                        ),
                        **({"heat_loss_J": diagnostic["heat_loss_J"]} if spec.energy_nodes else {}),
                        **(
                            {"flow_shaft_work_J": diagnostic["shaft_work_J"]}
                            if spec.shaft_machine
                            else {}
                        ),
                        **(
                            {"boundary_work_J": -diagnostic["boundary_work_on_J"]}
                            if spec.pressure_control
                            else {}
                        ),
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="gas_network_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            # The original thermal table owns heat only. Mechanical work is
            # already in each resource ledger (positive OUT) and accounting
            # slot (positive IN); do not confuse either sign with heat.
            vessel_thermal_deltas={
                name: {key: value for key, value in row.items() if key != "boundary_work_J"}
                for name, row in thermal_deltas.items()
            },
            diagnostics={
                "gas_network": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "energy_scope": (
                    "finite two-tank gas shaft device; original inlet/outlet H; "
                    "signed shaft work counted once in original resource ledger"
                    if spec.shaft_machine
                    else "finite constant-load piston; original U/Cv; "
                    "actual V and signed mechanical work"
                    if spec.pressure_control
                    else "finite constant-volume gas; independent node temperatures; "
                    "actual source H; "
                    "bounded anchor heater and signed environmental loss"
                    if spec.energy_nodes
                    else "finite constant-volume ideal gas; common isotherm; "
                    "transported enthalpy and signed bath heat"
                ),
            },
        )
