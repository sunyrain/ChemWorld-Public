"""Instance-scoped execution and conservative, declared material-port routing.

Inactive vessels are held unchanged. Time is advanced explicitly by operating
the selected vessel; the global ledger records total executed resource use.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from chemworld.foundation import WorldState
from chemworld.foundation.samples import SampleLedger, StoredSample
from chemworld.foundation.solvents import SolventAccounting
from chemworld.foundation.state import Ledger
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, VesselLedger
from chemworld.runtime.material_routing import (
    _active,
    _contents,
    _install,
    _merge,
    _occupied,
    _portion,
)
from chemworld.world.component_registry import NETWORK_OPERATIONS, component_operations


def _local(state: WorldState) -> WorldState:
    assert state.thermal is not None
    thermal = state.thermal.vessels[state.vessel_id]
    return state.replace(
        inactive_vessels={},
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
        terminated=global_state.terminated,
        solvent_accounting=global_state.solvent_accounting,
        samples=replace(
            global_state.samples,
            active_lineage=local.samples.active_lineage,
            active_reference_sources=local.samples.active_reference_sources,
            active_reference_shares=local.samples.active_reference_shares,
        ),
    )


def initialize_network(state: WorldState, compiled: Any) -> WorldState:
    spec = compiled.spec
    components = {c.id: c for c in spec.components}
    network = {
        "vessels": [v["id"] for v in spec.vessels],
        "operations": {
            v["id"]: list(
                component_operations(tuple(c.kind for c in spec.components if c.vessel == v["id"]))
            )
            for v in spec.vessels
        },
        "bounds": {
            name: {f"{op}:{key}": list(bounds) for (op, key), bounds in fields.items()}
            for name, fields in compiled.compatibility.vessel_field_bounds.items()
        },
        "instruments": {
            v["id"]: list(
                dict.fromkeys(
                    instrument
                    for c in spec.components
                    if c.vessel == v["id"] and c.kind == "observation"
                    for instrument in c.parameters.get(
                        "instruments", compiled.task_spec.allowed_instruments
                    )
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
    assert state.phases is not None and state.vessels is not None and state.equipment is not None
    assert state.species is not None
    template_vessel = state.vessels.vessels[state.vessel_id]
    for index, vessel in enumerate(spec.vessels):
        name = vessel["id"]
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
                for key, phase in state.phases.phases.items()
            }
        )
        locals_[name] = state.replace(
            vessel_id=name,
            species_amounts=amounts,
            volume_L=state.volume_L if index == 0 else 0.0,
            phases=phases,
            species=state.species if index == 0 else replace(state.species, initial_amounts_mol={}),
            vessels=VesselLedger(
                {name: replace(template_vessel, vessel_id=name, max_volume_L=vessel["capacity_L"])}
            ),
            equipment=EquipmentLedger(
                {
                    key: replace(value, attached_vessel_id=name)
                    for key, value in state.equipment.equipment.items()
                }
            ),
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


def _index(value: Any, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise ValueError("Unknown network address")
    return value


def _whole(state: WorldState) -> StoredSample:
    assert state.phases is not None
    return _active(state.replace(volume_L=sum(p.volume_L for p in state.phases.phases.values())))


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
        assert state.phases is not None and state.species is not None
        phase = state.phases.phases.get(port)
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
            equipment=_outlet_equipment(
                EquipmentLedger(
                    {k: v for k, v in contents.equipment.equipment.items() if k == "batch_reactor"}
                ),
                incoming_phase.species_amounts_mol,
                state.species.species_roles,
                incoming_phase.volume_L,
            ),
            metrics={},
        )
        incoming = replace(
            basis,
            volume_L=incoming_phase.volume_L,
            species_amounts_mol=incoming_phase.species_amounts_mol,
            contents=contents,
        )
        phases = {
            **state.phases.phases,
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
            phases=PhaseLedger(phases),
            species_amounts=PhaseLedger(phases).total_amounts_mol(),
            volume_L=selected.volume_L
            if selected is not None
            else sum(p.volume_L for p in phases.values()),
            species=_contents(retained).species,
            equipment=_outlet_equipment(
                state.equipment or EquipmentLedger(),
                PhaseLedger(phases).total_amounts_mol(),
                state.species.species_roles,
                sum(p.volume_L for p in phases.values()),
            ),
            process=replace(
                state.process,
                metrics=_contents(retained).metrics,
                last_observation={},
                last_observed_mask={},
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


def route_material(state: WorldState, action: dict[str, Any]) -> WorldState:
    network = state.metadata["component_network"]
    edge = network["connections"][_index(action.get("connection"), len(network["connections"]))]
    source, target = edge["source_vessel"], edge["target_vessel"]
    if source != state.vessel_id:
        raise ValueError("Select the source vessel before routing its output")
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


def network_error(state: WorldState, operation: str, action: dict[str, Any]) -> str | None:
    try:
        if state.terminated:
            raise ValueError("Network operations require an open episode")
        if operation == "select_vessel":
            select_vessel(state, action)
        elif operation == "route_material":
            route_material(state, action)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return str(exc)
    return None


def available_connections(state: WorldState) -> tuple[int, ...]:
    network = state.metadata.get("component_network", {})
    result = []
    for index, edge in enumerate(network.get("connections", [])):
        if edge["source_vessel"] != state.vessel_id:
            continue
        target = state.inactive_vessels[edge["target_vessel"]]
        assert state.phases is not None and target.phases is not None and target.vessels is not None
        port = edge["source"]["port"]
        source_volume = (
            sum(p.volume_L for p in state.phases.phases.values())
            if port == "out"
            else state.phases.phases[port].volume_L
            if port in state.phases.phases
            else 0.0
        )
        capacity = target.vessels.vessels[target.vessel_id].max_volume_L
        room = capacity - sum(p.volume_L for p in target.phases.phases.values())
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
            )
            is None
        ):
            result.append(index)
    return tuple(result)


def public_network(state: WorldState) -> dict[str, Any] | None:
    network = state.metadata.get("component_network")
    if network is None:
        return None
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    capacities = {}
    for name, local in locals_.items():
        assert local.vessels is not None
        capacities[name] = local.vessels.vessels[name].max_volume_L
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
        "clock_model": "explicit_local_advancement; inactive vessels held unchanged",
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
