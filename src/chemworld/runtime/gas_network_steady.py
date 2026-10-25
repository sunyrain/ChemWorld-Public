"""Read-only steady analysis of the selected original gas/surface network."""

from __future__ import annotations

from typing import Any

from chemworld.physchem.gas_network_steady import gas_network_steady_state
from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec
from chemworld.runtime.gas_reactor_network import (
    GasReactorNetworkRuntime,
    gas_anchor_thermal_device,
    gas_network_record,
    gas_surface_slots,
    unpack_gas_network,
)
from chemworld.runtime.scheduling import local_states


def analyze_gas_network_steady_world(
    state: Any, composition: Any, **options: Any
) -> dict[str, Any]:
    """Use current original n/T and slots, without executing a live operation."""
    if composition.physics is None:
        raise ValueError("steady gas analysis requires the original physical composition")
    if (
        state.terminated
        or state.quenched
        or (state.lab_clock is not None and state.lab_clock.stopped)
    ):
        raise ValueError("cannot analyze a stopped world")
    if state.lab_clock is not None and (state.lab_clock.queued or state.lab_clock.running):
        raise ValueError("queued or running controls are not fixed steady boundaries")
    locals_ = local_states(state)
    found = []
    for local in locals_.values():
        record = gas_network_record(local)
        if record is not None:
            spec = GasReactorNetworkSpec.from_dict(record.settings["physical_model"])
            if state.vessel_id in {node.vessel_id for node in spec.nodes}:
                found.append(local)
    if len(found) != 1:
        raise ValueError("selected vessel must belong to exactly one original gas network")
    anchor = found[0].replace(
        inactive_vessels={key: value for key, value in locals_.items() if key != found[0].vessel_id}
    )
    network = composition.physics.network
    _, spec, packed = unpack_gas_network(anchor, network)
    shared = GasReactorNetworkRuntime(composition).shared_system(
        anchor, "finite-isothermal-gas-network"
    )
    result = gas_network_steady_state(
        spec,
        network,
        packed,
        thermal_device=gas_anchor_thermal_device(anchor, spec),
        surface_slots=gas_surface_slots(anchor, spec),
        **options,
    )
    result["world_consumer"] = {
        "selected_vessel": state.vessel_id,
        "anchor_vessel": anchor.vessel_id,
        "start_time_s": state.ledger.time_s,
        "connection_ids": list(shared.connections),
        "live_operations_executed": 0,
        "live_resources_charged": False,
    }
    return result


__all__ = ["analyze_gas_network_steady_world"]
