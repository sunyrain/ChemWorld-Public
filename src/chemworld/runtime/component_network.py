"""Instance-scoped execution and conservative, declared material-port routing.

Inactive vessels are held unchanged. Time is advanced explicitly by operating
the selected vessel; the global ledger records total executed resource use.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState
from chemworld.foundation.material_properties import install_material_properties
from chemworld.foundation.samples import SampleLedger, StoredSample
from chemworld.foundation.solvents import SolventAccounting
from chemworld.foundation.state import Ledger
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    PhaseLedger,
    SpeciesLedger,
    ThermalLedger,
    VesselLedger,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.runtime.material_routing import (
    _active,
    _contents,
    _install,
    _merge,
    _occupied,
    _portion,
)
from chemworld.world.component_registry import NETWORK_OPERATIONS, component_operations

if TYPE_CHECKING:
    from chemworld.runtime.kernel_contracts import ModelProviderResult
    from chemworld.runtime.material_routing import ChemWorldMaterialRoutingServices
    from chemworld.world.composition import CompiledWorldComposition, WorldCompositionSpec


def _local(state: WorldState) -> WorldState:
    # WorldState normalizes these optional constructor inputs in __post_init__.
    thermal = cast(ThermalLedger, state.thermal).vessels[state.vessel_id]
    return state.replace(
        inactive_vessels={},
        lab_clock=None,
        vessel_elapsed_s={},
        ledger=Ledger(
            time_s=state.vessel_elapsed_s.get(state.vessel_id, 0),
            risk=state.ledger.risk,
            energy_jacket_J=thermal.energy_jacket_J,
            heat_reaction_J=thermal.heat_reaction_J,
            heat_loss_J=thermal.heat_loss_J,
        ),
        samples=SampleLedger(
            active_lineage=state.samples.active_lineage,
            active_reference_sources=state.samples.active_reference_sources,
            active_reference_shares=state.samples.active_reference_shares,
            batch_generation=state.samples.batch_generation,
        ),
        solvent_accounting=SolventAccounting(),
    )


def _activate(
    global_state: WorldState, local: WorldState, inactive: dict[str, WorldState]
) -> WorldState:
    return local.replace(
        inactive_vessels=inactive,
        vessel_elapsed_s=global_state.vessel_elapsed_s,
        ledger=global_state.ledger,
        lab_clock=global_state.lab_clock,
        terminated=global_state.terminated,
        solvent_accounting=global_state.solvent_accounting,
        samples=replace(
            global_state.samples,
            active_lineage=local.samples.active_lineage,
            active_reference_sources=local.samples.active_reference_sources,
            active_reference_shares=local.samples.active_reference_shares,
        ),
    )


def initialize_network(state: WorldState, compiled: CompiledWorldComposition) -> WorldState:
    from chemworld.world.composition import (
        component_instruments,
        physical_instruments,
        physical_operations,
    )

    spec = compiled.spec
    components = {c.id: c for c in spec.components}
    network = {
        "vessels": [v["id"] for v in spec.vessels],
        "thermal_systems": [] if spec.physics is None else sorted(
            set(spec.physics.models_of("thermal")) | set(spec.physics.models_of("flash"))
            | set(spec.physics.models_of("liquid"))),
        "operations": {
            v["id"]: list(
                physical_operations(spec, v["id"])
                if spec.physics is not None
                else component_operations(
                    tuple(c.kind for c in spec.components if c.vessel == v["id"]),
                    modules=spec.modules,
                )
            )
            for v in spec.vessels
        },
        "bounds": {
            name: {f"{op}:{key}": list(bounds) for (op, key), bounds in fields.items()}
            for name, fields in compiled.compatibility.vessel_field_bounds.items()
        },
        "instruments": {
            v["id"]: list(
                physical_instruments(spec, v["id"])
                if spec.physics is not None
                else component_instruments(
                    spec,
                    v["id"],
                    fallback=compiled.task_spec.allowed_instruments,
                )
            )
            for v in spec.vessels
        },
        "connections": [
            {
                **connection,
                "source_vessel": components[connection["source"]["component"]].vessel,
                "target_vessel": components[connection["target"]["component"]].vessel,
            }
            for connection in spec.connections
        ],
    }
    locals_: dict[str, WorldState] = {}
    template_vessel = cast(VesselLedger, state.vessels).vessels[state.vessel_id]
    for index, vessel in enumerate(spec.vessels):
        name = vessel["id"]
        if spec.physics is not None:
            # Native entities are authored separately. Never clone the first
            # reactor's material, equipment, phase, or state slots into another.
            local = deepcopy(spec.physics.initial_states[name])
            locals_[name] = local.replace(metadata={**local.metadata, "component_network": network})
            continue
        # Only the first vessel receives the scenario's initial charge. Other
        # vessels start empty; constructing a graph never duplicates inventory.
        amounts = state.species_amounts if index == 0 else dict.fromkeys(state.species_amounts, 0.0)
        phases = PhaseLedger(
            {
                key: replace(
                    phase,
                    vessel_id=name,
                    volume_L=phase.volume_L if index == 0 else 0.0,
                    species_amounts_mol=phase.species_amounts_mol
                    if index == 0
                    else dict.fromkeys(phase.species_amounts_mol, 0.0),
                )
                for key, phase in cast(PhaseLedger, state.phases).phases.items()
            }
        )
        equipment = EquipmentLedger(
            {
                key: replace(value, attached_vessel_id=name)
                for key, value in cast(EquipmentLedger, state.equipment).equipment.items()
            }
        )
        if index:
            equipment = install_material_properties(equipment, {}, name)
        locals_[name] = state.replace(
            vessel_id=name,
            species_amounts=amounts,
            volume_L=state.volume_L if index == 0 else 0.0,
            phases=phases,
            species=state.species if index == 0 else replace(
                cast(SpeciesLedger, state.species), initial_amounts_mol={}
            ),
            vessels=VesselLedger(
                {name: replace(template_vessel, vessel_id=name, max_volume_L=vessel["capacity_L"])}
            ),
            equipment=equipment,
            state_slots=StateSlots(
                tuple(
                    replace(
                        slot,
                        location_id=name,
                        spec=replace(slot.spec, owner_id=name)
                        if slot.spec.owner_kind == "vessel"
                        else slot.spec,
                    )
                    for slot in state.state_slots.slots
                )
            )
            if index == 0
            else StateSlots(),
            thermal=None,
            metadata={**state.metadata, "component_network": network},
            samples=replace(
                state.samples,
                active_lineage=(f"batch-{state.samples.batch_generation:04d}-{name}",),
            ),
        )
    first = spec.vessels[0]["id"]
    return locals_.pop(first).replace(
        inactive_vessels=locals_,
        vessel_elapsed_s={v["id"]: 0.0 for v in spec.vessels},
    )


def select_vessel(state: WorldState, action: dict[str, Any]) -> WorldState:
    names = state.metadata["component_network"]["vessels"]
    name = names[_index(action.get("vessel"), len(names))]
    if name == state.vessel_id:
        return state
    inactive = dict(state.inactive_vessels)
    target = inactive.pop(name)
    inactive[state.vessel_id] = _local(state)
    return _activate(state, target, inactive)


def reconnect_device(state: WorldState, identity: str, destination: str | None) -> WorldState:
    """Move one actual device and owned state; materials remain in place."""
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    owners = [key for key, local in locals_.items()
              if identity in cast(EquipmentLedger, local.equipment).equipment]
    if len(owners) != 1 or (destination is not None and destination not in locals_):
        raise ValueError("equipment needs exactly one owner and a declared destination")
    source = owners[0]
    if source == destination or destination is None:
        local = locals_[source]
        equipment = cast(EquipmentLedger, local.equipment).equipment
        record = equipment[identity]
        locals_[source] = local.replace(equipment=EquipmentLedger({
            **equipment,
            identity: replace(
                record, attached_vessel_id=source if destination is None else destination,
                status="parked" if destination is None
                else "idle" if record.status == "parked" else record.status,
            ),
        }))
        active = locals_.pop(state.vessel_id)
        return active.replace(inactive_vessels=locals_)
    src, dst = locals_[source], locals_[destination]
    source_equipment = cast(EquipmentLedger, src.equipment).equipment
    record = source_equipment[identity]
    slots = tuple(
        slot for slot in src.state_slots.slots
        if slot.spec.owner_kind == "equipment" and slot.spec.owner_id == identity
    )
    locals_[source] = src.replace(
        equipment=EquipmentLedger(
            {k: v for k, v in source_equipment.items() if k != identity}
        ),
        state_slots=StateSlots(tuple(slot for slot in src.state_slots.slots if slot not in slots)),
    )
    locals_[destination] = dst.replace(
        equipment=EquipmentLedger({**cast(EquipmentLedger, dst.equipment).equipment,
                                   identity: replace(
                                       record, attached_vessel_id=destination,
                                       status=("idle" if record.status == "parked"
                                               else record.status),
                                   )}),
        state_slots=StateSlots((*dst.state_slots.slots,
                               *(replace(slot, location_id=destination) for slot in slots))),
    )
    active = locals_.pop(state.vessel_id)
    return active.replace(inactive_vessels=locals_)


def _index(value: Any, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise ValueError("Unknown network address")
    return value


def _whole(state: WorldState) -> StoredSample:
    phases = cast(PhaseLedger, state.phases).phases
    return _active(state.replace(volume_L=sum(p.volume_L for p in phases.values())))


def _outlet_equipment(
    equipment: EquipmentLedger,
    amounts: dict[str, float],
    roles: dict[str, tuple[str, ...]],
    volume: float,
) -> EquipmentLedger:
    catalyst = sum(amounts.get(key, 0) for key, value in roles.items() if "catalyst" in value)
    return EquipmentLedger(
        {
            key: replace(
                record,
                settings={
                    **record.settings,
                    "catalyst_amount_mol": catalyst,
                    "solvent_volume_L": volume,
                },
            )
            if key == "batch_reactor"
            else replace(record, status="stale")
            if key.startswith("instrument:")
            else record
            for key, record in equipment.equipment.items()
        }
    )


def _withdraw(
    state: WorldState, port: str, fraction: float, target: str
) -> tuple[WorldState, StoredSample]:
    whole = _whole(state)
    if not _occupied(whole):
        raise ValueError("Source port is empty")
    if port == "out":
        incoming = _portion(whole, fraction, target)
        retained = _portion(whole, 1 - fraction, state.vessel_id)
        after = _install(state, retained)
    else:
        phase_ledger = cast(PhaseLedger, state.phases)
        species = cast(SpeciesLedger, state.species)
        phase = phase_ledger.phases.get(port)
        if phase is None or phase.phase_type != "liquid" or phase.volume_L <= 0:
            raise ValueError("Requested liquid outlet has no available inventory")
        total_moles = sum(state.species_amounts.values())
        share = (
            sum(phase.species_amounts_mol.values()) / total_moles
            if total_moles
            else phase.volume_L / whole.volume_L
        )
        basis = _portion(whole, share * fraction, target)
        contents = _contents(basis)
        retained_slots, incoming_slots = state.state_slots.split_material(
            fraction, state.vessel_id, target, phase_id=port,
            destination_phase_id="reactor_liquid",
        )
        incoming_phase = replace(
            phase,
            phase_id="reactor_liquid",
            vessel_id=target,
            selected=False,
            settled=False,
            volume_L=phase.volume_L * fraction,
            species_amounts_mol={
                key: value * fraction for key, value in phase.species_amounts_mol.items()
            },
        )
        contents = replace(
            contents,
            phases=PhaseLedger({"reactor_liquid": incoming_phase}),
            properties={
                "batch_reactor": {
                    **contents.properties.get("batch_reactor", {}),
                    "catalyst_amount_mol": sum(
                        n
                        for k, n in incoming_phase.species_amounts_mol.items()
                        if "catalyst" in species.species_roles.get(k, ())
                    ),
                    "solvent_volume_L": incoming_phase.volume_L,
                }
            },
            state_slots=incoming_slots,
            metrics={},
        )
        incoming = replace(
            basis,
            volume_L=incoming_phase.volume_L,
            species_amounts_mol=incoming_phase.species_amounts_mol,
            contents=contents,
        )
        phases = {
            **phase_ledger.phases,
            port: replace(
                phase,
                volume_L=phase.volume_L * (1 - fraction),
                species_amounts_mol={
                    k: v * (1 - fraction) for k, v in phase.species_amounts_mol.items()
                },
            ),
        }
        retained = _portion(whole, 1 - share * fraction, state.vessel_id)
        selected = next((p for p in phases.values() if p.selected), None)
        after = state.replace(
            state_slots=retained_slots,
            phases=PhaseLedger(phases),
            species_amounts=PhaseLedger(phases).total_amounts_mol(),
            volume_L=selected.volume_L
            if selected is not None
            else sum(p.volume_L for p in phases.values()),
            species=_contents(retained).species,
            equipment=_outlet_equipment(
                state.equipment or EquipmentLedger(),
                PhaseLedger(phases).total_amounts_mol(),
                species.species_roles,
                sum(p.volume_L for p in phases.values()),
            ),
            process=replace(
                state.process,
                metrics=_contents(retained).metrics,
                last_observation_record=None,
            )
            if state.process
            else None,
        )
    reference = _contents(retained)
    return after.replace(
        samples=replace(
            after.samples,
            active_lineage=retained.lineage,
            active_reference_sources=reference.reference_sources,
            active_reference_shares=reference.reference_shares,
        )
    ), incoming


def resolve_material_connection(
    state: WorldState, action: Mapping[str, Any], *, source_ports: Container[str] | None = None,
) -> tuple[dict[str, Any], float]:
    """Resolve the compiled material edge and common transfer constraints once."""
    network = state.metadata["component_network"]
    edge = network["connections"][_index(action.get("connection"), len(network["connections"]))]
    if state.lab_clock is not None and edge.get("scheduling") != "manual":
        raise ValueError("shared routing requires an explicitly manual connection")
    source, target = edge["source_vessel"], edge["target_vessel"]
    if source != state.vessel_id:
        raise ValueError("Select the source vessel before routing its output")
    if source == target or target not in state.inactive_vessels:
        raise ValueError("material connection requires a distinct installed destination")
    if source_ports is not None and edge["source"]["port"] not in source_ports:
        raise ValueError("select a supported source material outlet")
    fraction = action.get("transfer_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (float, int))
        or not isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("Transfer fraction must be in (0, 1]")
    if action.get("mixing") not in (0, 1) or isinstance(action.get("mixing"), bool):
        raise ValueError("Mixing must be empty_only=0 or allow=1")
    return edge, fraction


def route_material(state: WorldState, action: dict[str, Any]) -> WorldState:
    edge, fraction = resolve_material_connection(state, action)
    source, target = edge["source_vessel"], edge["target_vessel"]
    destination = state.inactive_vessels[target]
    original = _whole(destination)
    if _occupied(original) and action["mixing"] != 1:
        raise ValueError("Occupied destination requires explicit mixing")
    if _occupied(original) and original.quenched != state.quenched:
        raise ValueError("Mixing requires matching reaction stop state")
    after, incoming = _withdraw(state, edge["source"]["port"], float(fraction), target)
    if original.volume_L + incoming.volume_L > original.capacity_L + 1e-12:
        raise ValueError("Destination vessel capacity exceeded")
    merged = _merge(original, incoming)
    contents = _contents(merged)
    updated = _install(destination, merged).replace(
        samples=replace(
            destination.samples,
            active_lineage=merged.lineage,
            active_reference_sources=contents.reference_sources,
            active_reference_shares=contents.reference_shares,
        )
    )
    transfer = {
        "connection": edge["id"],
        "source": source,
        "destination": target,
        "source_port": edge["source"]["port"],
        "fraction": float(fraction),
        "volume_L": incoming.volume_L,
        "lineage": list(incoming.lineage),
        "time_s": state.ledger.time_s,
        "mixing": action["mixing"],
    }
    return after.replace(
        inactive_vessels={**state.inactive_vessels, target: updated},
        samples=replace(
            after.samples,
            transfers=(*state.samples.transfers, transfer),
            transfer_tools_used=state.samples.transfer_tools_used + 1,
        ),
        ledger=state.ledger.with_updates(cost=state.ledger.cost + 0.005),
    )


def network_error(
    state: WorldState, operation: str, action: dict[str, Any],
    material_router: ChemWorldMaterialRoutingServices | None = None,
) -> str | None:
    try:
        if state.terminated:
            raise ValueError("Network operations require an open episode")
        if operation == "select_vessel":
            select_vessel(state, action)
        elif operation == "route_material":
            (route_material if material_router is None else material_router.route_material)(
                state, action
            )
    except (ArithmeticError, RuntimeError, KeyError, ValueError, TypeError, IndexError) as exc:
        return str(exc)
    return None


def available_connections(
    state: WorldState, material_router: ChemWorldMaterialRoutingServices | None = None,
) -> tuple[int, ...]:
    network = state.metadata.get("component_network", {})
    result = []
    for index, edge in enumerate(network.get("connections", [])):
        if state.lab_clock is not None and edge.get("scheduling") != "manual":
            continue
        if edge["source_vessel"] != state.vessel_id:
            continue
        target = state.inactive_vessels[edge["target_vessel"]]
        if material_router is not None and material_router.uses_native_packets(state):
            if (
                network_error(
                    state,
                    "route_material",
                    {
                        "connection": index,
                        "transfer_fraction": 0.01,
                        "mixing": 1,
                    },
                    material_router,
                )
                is None
            ):
                result.append(index)
            continue
        port = edge["source"]["port"]
        phases = cast(PhaseLedger, state.phases).phases
        source_volume = (
            sum(p.volume_L for p in phases.values())
            if port == "out"
            else phases[port].volume_L
            if port in phases
            else 0.0
        )
        capacity = cast(VesselLedger, target.vessels).vessels[target.vessel_id].max_volume_L
        room = capacity - sum(p.volume_L for p in cast(PhaseLedger, target.phases).phases.values())
        fraction = min(0.5, room / source_volume) if source_volume > 0 and room > 0 else 0.0
        if (
            fraction > 0
            and network_error(
                state,
                "route_material",
                {
                    "connection": index,
                    "transfer_fraction": fraction,
                    "mixing": 1,
                },
                material_router,
            )
            is None
        ):
            result.append(index)
    return tuple(result)


class ComponentNetworkServices:
    def __init__(
        self, material_router: ChemWorldMaterialRoutingServices,
        composition: WorldCompositionSpec | None = None,
    ) -> None:
        self.material_router = material_router
        self.composition = composition

    select_vessel = staticmethod(select_vessel)

    def route_material(
        self, state: WorldState, action: dict[str, Any],
    ) -> WorldState | ModelProviderResult:
        return self.material_router.route_material(state, action)

    def reconnect_heater(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        """Move one disabled finite heater, never a material or process clock."""
        from chemworld.runtime.coupled_thermal import reconnect_thermal_device
        from chemworld.world.composition import physical_operations

        if self.composition is None or self.composition.physics is None or state.lab_clock:
            raise ValueError("manual heater connection requires native local-clock apparatus")
        locals_ = {state.vessel_id: _local(state), **state.inactive_vessels}
        devices = [record for local in locals_.values()
                   for record in cast(EquipmentLedger, local.equipment).equipment.values()
                   if record.equipment_type == "thermal_device"]
        if len(devices) != 1 or devices[0].settings["thermal_device"]["enabled"]:
            raise ValueError("reconnect exactly one disabled heater; pause control first")
        # Reuse the existing finite vessel-address decoder without selecting it.
        destination = select_vessel(state, action).vessel_id
        physics = self.composition.physics
        bounds = physics.declared_temperature_bounds_k(
            locals_[destination], self.composition.modules)
        if bounds is None:
            raise ValueError("destination has no declared native thermal domain")
        low, high = bounds
        if not low <= devices[0].settings["thermal_device"]["setpoint_K"] <= high:
            raise ValueError("retained heater setpoint exceeds destination thermal domain")
        candidate = reconnect_thermal_device(state, devices[0].equipment_id, destination)
        states = {candidate.vessel_id: _local(candidate), **candidate.inactive_vessels}
        spec = replace(self.composition, physics=replace(physics, initial_states=states))
        network = {**candidate.metadata["component_network"], "operations": {
            name: list(physical_operations(spec, name)) for name in states}}
        return candidate.replace(
            metadata={**candidate.metadata, "component_network": network},
            inactive_vessels={name: local.replace(metadata={
                **local.metadata, "component_network": network})
                for name, local in candidate.inactive_vessels.items()},
        )


def public_network(state: WorldState) -> dict[str, Any] | None:
    network = state.metadata.get("component_network")
    if network is None:
        return None
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    capacities = {}
    for name, local in locals_.items():
        capacities[name] = cast(VesselLedger, local.vessels).vessels[name].max_volume_L
    return {
        "active_vessel": state.vessel_id,
        "vessels": [
            {
                "index": index,
                "id": name,
                "elapsed_s": state.vessel_elapsed_s[name],
                "capacity_L": capacities[name],
            }
            for index, name in enumerate(network["vessels"])
        ],
        "connections": [{"index": i, **edge} for i, edge in enumerate(network["connections"])],
        "clock_model": ("shared_lab_clock" if state.lab_clock is not None else
                        "explicit_local_advancement; inactive vessels held unchanged"),
        "terminal_scope": "episode; final assay samples the selected vessel",
    }


def operation_available(state: WorldState, operation: str) -> bool:
    network = state.metadata.get("component_network")
    if operation in NETWORK_OPERATIONS and network is None:
        return False
    return (
        network is None
        or operation in NETWORK_OPERATIONS
        or operation in network["operations"][state.vessel_id]
    )
