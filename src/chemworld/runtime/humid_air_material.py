"""Selected humid gas/condensate packets through the original material route.

Both endpoints retain their existing n/T/V, support, clock and resource owners.
Only the existing transaction commits candidates. No pressure-flow kinetics,
extra phase stock, integration loop or external source is introduced here.
"""

from dataclasses import replace

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.balances import MaterialPacket
from chemworld.foundation.state_ledgers import PhaseLedger, ProcessLedger
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.humid_air_thermal import HumidAirIsochoricThermodynamics


def humid_air_model(state, physics):
    model = physics.models_of("thermal").get(state.vessel_id)
    return (
        model
        if model is not None and isinstance(model.thermodynamics, HumidAirIsochoricThermodynamics)
        else None
    )


def _replace_amounts(state, amounts):
    if state.phase not in {"fluid", "gas"} or len(state.phases.phases) != 1:
        raise ValueError("humid-air material requires one original aggregate inventory")
    return state.replace(
        species_amounts=amounts,
        phases=PhaseLedger(
            {
                key: replace(phase, species_amounts_mol=amounts)
                for key, phase in state.phases.phases.items()
            }
        ),
        process=replace(state.process or ProcessLedger(), last_observation_record=None),
    )


def route_humid_air_packet(state, physics, action):
    from chemworld.runtime.component_network import resolve_material_connection
    from chemworld.runtime.thermal_material_routing import _energy, _set_energy

    edge, fraction = resolve_material_connection(state, action, source_ports={"out"})
    target = state.inactive_vessels[edge["target_vessel"]]
    source_model, target_model = humid_air_model(state, physics), humid_air_model(target, physics)
    if source_model is None or target_model is None:
        raise ValueError("humid-air packets require two same-G8 thermal endpoints")
    if source_model.thermodynamics.components != target_model.thermodynamics.components:
        raise ValueError("humid-air packet endpoints must share component identities and weights")
    if action["mixing"] != 1:
        raise ValueError("occupied humid-air destination requires explicit mixing")
    # Validates original support ownership and all four component amounts.
    source_u, target_u = _energy(state, source_model), _energy(target, target_model)
    outlet = source_model.thermodynamics.material_outlet(
        CoupledState(state.species_amounts, state.temperature_K, state.volume_L), fraction
    )
    amounts, energy = outlet["amounts_mol"], outlet["enthalpy_J"]
    packet = MaterialPacket(state.vessel_id, target.vessel_id, tuple(amounts.items()))
    source_amounts, target_amounts = packet.apply(state.species_amounts, target.species_amounts)
    source = _set_energy(_replace_amounts(state, source_amounts), source_model, source_u - energy)
    destination = _set_energy(
        _replace_amounts(target, target_amounts), target_model, target_u + energy
    )
    # A phase-selective packet has a separate retained fraction for each species.
    reference = {
        key: initial
        * (amounts[key] / state.species_amounts[key] if state.species_amounts[key] else 0.0)
        for key, initial in state.species.initial_amounts_mol.items()
    }
    source = source.replace(
        species=replace(
            state.species,
            initial_amounts_mol={
                key: initial - reference[key]
                for key, initial in state.species.initial_amounts_mol.items()
            },
        )
    )
    destination = destination.replace(
        species=species_with_added_initial_amounts(target.species, reference)
    )
    source_error = _energy(source, source_model) - source_u + energy
    target_error = _energy(destination, target_model) - target_u - energy
    if abs(source_error) > 1e-7 or abs(target_error) > 1e-7:
        raise ValueError("humid-air material packet violates the declared energy balance")
    receipt = {
        **packet.to_dict(),
        **{k: v for k, v in outlet.items() if k != "amounts_mol"},
        "source_energy_residual_J": source_error,
        "destination_energy_residual_J": target_error,
        "source_thermal_boundary": "adiabatic_finite_phase_packet",
        "scope": "zero_time_controlled_packet; no_pressure_flow_or_pump_model",
    }
    transfer = {
        "connection": edge["id"],
        "source": state.vessel_id,
        "destination": target.vessel_id,
        "source_port": "out",
        "fraction": fraction,
        "material_phase": outlet["material_phase"],
        "amounts_mol": amounts,
        "time_s": state.ledger.time_s,
        "mixing": action["mixing"],
    }
    return source.replace(
        inactive_vessels={**state.inactive_vessels, target.vessel_id: destination},
        ledger=state.ledger.with_updates(cost=state.ledger.cost + 0.005),
        samples=replace(
            state.samples,
            transfers=(*state.samples.transfers, transfer),
            transfer_tools_used=state.samples.transfer_tools_used + 1,
        ),
        metadata={**state.metadata, "last_material_packet": receipt},
    )
