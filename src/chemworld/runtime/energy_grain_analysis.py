"""Frozen-condition QSD query using the original gas world's live inventory."""

from __future__ import annotations

from typing import Any

from chemworld.physchem.energy_grain_binding import compile_energy_grain_network
from chemworld.physchem.energy_grain_reduction import energy_grain_qsd
from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec, R
from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, gas_node_view
from chemworld.runtime.gas_reactor_network import gas_network_record, unpack_gas_network
from chemworld.runtime.scheduling import local_states


def analyze_energy_grain_world_state(state: Any, composition: Any) -> dict[str, Any]:
    """Read the selected original node; hold its current T and bath for analysis.

    Flow, heating and queued controls are not extrapolated. The query does not
    reset the well distribution or charge resources, even if it is far from QSD.
    """
    if composition.physics is None:
        raise ValueError("QSD query requires the original physical composition")
    network = composition.physics.network
    declaration = network.metadata.get("energy_grain_declaration")
    if not isinstance(declaration, dict):
        raise ValueError("QSD query requires the original energy-grain declaration")
    grain_spec = EnergyGrainMasterEquationSpec.from_dict(declaration)
    expected = compile_energy_grain_network(grain_spec)
    if (
        network.species != expected.species
        or network.reactions != expected.reactions
        or network.metadata.get("gas_thermochemistry") != expected.metadata["gas_thermochemistry"]
    ):
        raise ValueError("live reaction network differs from its original energy-grain declaration")
    locals_ = local_states(state)
    found = []
    for local in locals_.values():
        record = gas_network_record(local)
        if record is not None:
            spec = GasReactorNetworkSpec.from_dict(record.settings["physical_model"])
            if state.vessel_id in {node.vessel_id for node in spec.nodes}:
                found.append(local)
    if len(found) != 1:
        raise ValueError("selected vessel requires one original gas-network owner")
    anchor = found[0].replace(
        inactive_vessels={k: v for k, v in locals_.items() if k != found[0].vessel_id}
    )
    _, spec, packed = unpack_gas_network(anchor, network)
    node = next(n for n in spec.nodes if n.vessel_id == state.vessel_id)
    view = gas_node_view(packed, node, network, pressure_control=spec.pressure_control)
    report = energy_grain_qsd(
        grain_spec,
        temperature_K=view.temperature_K,
        bath_concentrations_mol_L={
            b.species_id: view.amounts_mol[b.species_id] / view.volume_L
            for b in grain_spec.bath_gases
        },
        grain_amounts_mol={
            g.species_id: view.amounts_mol[g.species_id]
            for well in grain_spec.wells
            for g in well.grains
        },
    )
    report["world_query"] = {
        "selected_vessel_id": state.vessel_id,
        "anchor_vessel_id": anchor.vessel_id,
        "state_time_s": state.ledger.time_s,
        "volume_L": view.volume_L,
        "total_gas_pressure_Pa": sum(view.amounts_mol.values())
        * 1000
        * R
        * view.temperature_K
        / view.volume_L,
        "bath_amounts_mol": {
            b.species_id: view.amounts_mol[b.species_id] for b in grain_spec.bath_gases
        },
        "chemistry_enabled": node.chemistry_enabled,
        "declared_flow_connections": len(spec.connections),
        "dynamic_temperature": bool(spec.energy_nodes),
        "variable_volume": spec.pressure_control is not None,
        "queued_or_running_controls": bool(
            state.lab_clock is not None and (state.lab_clock.queued or state.lab_clock.running)
        ),
        "boundary": "current node conditions held; future flow, heating and controls not predicted",
        "world_advanced": False,
        "live_resources_charged": False,
        "thermochemistry_modified": False,
    }
    return report
