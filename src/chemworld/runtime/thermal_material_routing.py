"""Finite advective material packets using the same owned inventories and U(n,T).

Gas volumes are vessel geometry. A packet carries inlet enthalpy and any declared
mobile carrier internal energy; it never carries equipment. This zero-duration
mixing operation does not simulate a valve, pipe or pressure-driven flow.
"""

from dataclasses import replace
from math import fsum

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.balances import MaterialPacket
from chemworld.foundation.state_ledgers import PhaseLedger, ProcessLedger
from chemworld.physchem.coupled_system import ConstantCvThermodynamics
from chemworld.physchem.enthalpy import caloric_temperature
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.runtime.coupled_thermal import local_thermodynamics, pressure_for_state


def feed_packet(state, physics, action):
    source_id = physics.feed_sources.get(action.get("component"))
    if source_id is None or source_id not in state.inactive_vessels:
        raise ValueError("feed must identify an inactive finite source")
    source = state.inactive_vessels[source_id]
    packet = MaterialPacket.representative(
        source_id, state.vessel_id, source.species_amounts, amount_mol=action.get("amount_mol")
    )
    return source_id, dict(packet.amounts_mol)


def _replace_inventory(state, amounts, slots):
    if state.phase != "gas" or len(state.phases.phases) != 1:
        raise ValueError("advective gas packets require one gas bulk phase")
    return state.replace(
        species_amounts=amounts,
        phases=PhaseLedger(
            {
                key: replace(phase, species_amounts_mol=amounts)
                for key, phase in state.phases.phases.items()
            }
        ),
        state_slots=slots,
        process=replace(state.process or ProcessLedger(), last_observation_record=None),
    )


def _energy(state, model):
    thermo, packed = local_thermodynamics(state, model)
    return thermo.internal_energy_j(packed)


def _set_energy(state, model, energy):
    thermo, packed = local_thermodynamics(state, model)
    if isinstance(thermo, ConstantCvThermodynamics):
        reference = replace(packed, temperature_K=thermo.reference_temperature_K)
        capacity = thermo.heat_capacity_j_k(reference)
        if capacity <= 0:
            raise ValueError("material mixing needs positive heat capacity")
        temperature = (
            thermo.reference_temperature_K
            + (energy - thermo.internal_energy_j(reference)) / capacity
        )
    else:
        lower, upper = model.minimum_temperature_K, model.stop_temperature_K
        temperature = caloric_temperature(
            lambda t: thermo.internal_energy_j(replace(packed, temperature_K=t)),
            energy, (lower, upper), energy_basis="internal_energy", rtol=1e-12,
            algorithm="residual",
            heat_capacity_at_temperature=lambda t: thermo.heat_capacity_j_k(
                replace(packed, temperature_K=t)
            ),
        )
    if not model.minimum_temperature_K <= temperature < model.stop_temperature_K:
        raise ValueError("material packet would leave the supported thermal operating range")
    result = state.replace(temperature_K=temperature)
    pressure = pressure_for_state(result, model)
    limit = min(model.maximum_pressure_Pa, state.vessels.vessels[state.vessel_id].max_pressure_Pa)
    if pressure >= limit:
        raise ValueError("material packet would exceed vessel pressure capacity")
    return result.replace(pressure_Pa=pressure)


def exchange_packet(source, target, physics, amounts, *, fraction=0.0, fixed_source=False):
    """Return two local candidates and the executed packet's thermal receipt."""
    from chemworld.runtime.aqueous_mineral import aqueous_record, exchange_aqueous_packet

    if aqueous_record(target) is not None:
        return exchange_aqueous_packet(source, target, physics, amounts,
                                      fraction=fraction, fixed_source=fixed_source)
    if target.vessel_id in physics.models_of("liquid"):
        from chemworld.runtime.liquid_equilibrium import exchange_liquid_packet

        return exchange_liquid_packet(source, target, physics, amounts,
                                      fraction=fraction, fixed_source=fixed_source)
    if target.vessel_id in physics.models_of("flash"):
        from chemworld.runtime.flash_thermal import exchange_flash_packet

        return exchange_flash_packet(source, target, physics, amounts,
                                      fraction=fraction, fixed_source=fixed_source)
    from chemworld.runtime.condensed_material_routing import (
        background_model,
        exchange_condensed_packet,
    )

    if background_model(target,physics) is not None:
        return exchange_condensed_packet(source,target,physics,amounts,
                                         fraction=fraction,fixed_source=fixed_source)
    if source.vessel_id == target.vessel_id or source.phase != "gas" or target.phase != "gas":
        raise ValueError("gas packet requires two distinct gas vessels")
    if target.vessel_id not in physics.models_of("thermal"):
        raise ValueError("receiving vessel requires declared thermodynamics")
    model = physics.models_of("thermal")[target.vessel_id]
    if model.pressure_law != "ideal_gas":
        raise ValueError("gas packet requires an ideal-gas receiving boundary")
    if set(source.species_amounts) != set(target.species_amounts):
        raise ValueError("gas packet species must share the receiving inventory layout")
    target_phase = next(iter(target.phases.phases))
    source_slots, moved = source.state_slots.split_material(
        fraction, source.vessel_id, target.vessel_id, destination_phase_id=target_phase
    )
    target_slots = target.state_slots.receive_material(moved)
    packet = MaterialPacket(source.vessel_id, target.vessel_id, tuple(amounts.items()))
    source_amounts, target_amounts = packet.apply(source.species_amounts, target.species_amounts)
    # Reference state and species data must agree before transporting energy.
    if not fixed_source:
        source_model = physics.models_of("thermal")[source.vessel_id]
        if source_model.thermodynamics.species != model.thermodynamics.species or (
            source_model.thermodynamics.reference_temperature_K
            != model.thermodynamics.reference_temperature_K
        ):
            raise ValueError("material endpoints must share thermodynamic reference data")
    thermal = model.thermodynamics
    enthalpy = fsum(
        amount
        * (
            thermal.species_energy_j_mol(key, source.temperature_K)
            + R_J_PER_MOL_K * source.temperature_K
        )
        for key, amount in amounts.items()
    )
    carrier_energy = moved.heat_capacity() * (
        source.temperature_K - thermal.reference_temperature_K
    )
    carrier_energy += fsum(
        amount * thermal.species_energy_j_mol(key, source.temperature_K)
        for key, amount in moved.species_amounts().items()
    )
    energy = enthalpy + carrier_energy
    old_target_energy = _energy(target, model)
    target_candidate = _set_energy(
        _replace_inventory(target, target_amounts, target_slots), model, old_target_energy + energy
    )
    reference_fraction = fsum(amounts.values()) / fsum(source.species_amounts.values())
    reference_amounts = {
        key: value * reference_fraction for key, value in source.species.initial_amounts_mol.items()
    }
    target_candidate = target_candidate.replace(
        species=species_with_added_initial_amounts(target.species, reference_amounts)
    )
    source_candidate = _replace_inventory(source, source_amounts, source_slots)
    if fixed_source:
        source_candidate = source_candidate.replace(
            pressure_Pa=fsum(source_amounts.values())
            * R_J_PER_MOL_K
            * source.temperature_K
            / (source.volume_L / 1000)
        )
        source_residual = None
    else:
        old_source_energy = _energy(source, source_model)
        source_candidate = _set_energy(source_candidate, source_model, old_source_energy - energy)
        source_residual = _energy(source_candidate, source_model) - old_source_energy + energy
    source_candidate = source_candidate.replace(
        species=replace(
            source.species,
            initial_amounts_mol={
                key: amount * (1 - reference_fraction)
                for key, amount in source.species.initial_amounts_mol.items()
            },
        )
    )
    receipt = {
        **packet.to_dict(),
        "inlet_temperature_K": source.temperature_K,
        "gas_enthalpy_J": enthalpy,
        "carrier_internal_energy_J": carrier_energy,
        "destination_energy_residual_J": _energy(target_candidate, model)
        - old_target_energy
        - energy,
        "source_energy_residual_J": source_residual,
        "source_thermal_boundary": "fixed_temperature_stock; utility heat/work not modelled"
        if fixed_source
        else "adiabatic_packet_withdrawal",
    }
    if abs(receipt["destination_energy_residual_J"]) > 1e-7 or (
        source_residual is not None and abs(source_residual) > 1e-7
    ):
        raise ValueError("advective packet violates declared energy balance")
    return source_candidate, target_candidate, receipt


def add_finite_feed(state, physics, action):
    source_id, amounts = feed_packet(state, physics, action)
    source, target, receipt = exchange_packet(
        state.inactive_vessels[source_id], state, physics, amounts, fixed_source=True
    )
    return target.replace(
        inactive_vessels={**state.inactive_vessels, source_id: source},
        ledger=state.ledger.with_updates(
            cost=state.ledger.cost + 0.03 * fsum(amounts.values()) / 0.01
        ),
        metadata={**target.metadata, "last_material_packet": receipt},
    )


def route_gas_packet(state, physics, action):
    from chemworld.runtime.component_network import resolve_material_connection
    from chemworld.runtime.mineral_kinetics import kinetic_record, route_mineral_phase

    if kinetic_record(state) is not None:
        return route_mineral_phase(state, physics, action)
    from chemworld.runtime.condensed_material_routing import background_model

    edge, fraction = resolve_material_connection(state, action, source_ports={"out"})
    target = state.inactive_vessels[edge["target_vessel"]]
    target_occupied = (target.volume_L > 0 if background_model(target,physics) is not None
                       else fsum(target.species_amounts.values()) > 1e-12)
    if target_occupied and action["mixing"] != 1:
        raise ValueError("occupied destination requires explicit mixing")
    empty = (state.volume_L <= 0 if background_model(state,physics) is not None
             else fsum(state.species_amounts.values()) <= 0)
    if empty:
        raise ValueError("source gas inventory is empty")
    amounts = {key: value * fraction for key, value in state.species_amounts.items()}
    source, target, receipt = exchange_packet(state, target, physics, amounts, fraction=fraction)
    transfer = {
        "connection": edge["id"],
        "source": state.vessel_id,
        "destination": target.vessel_id,
        "source_port": "out",
        "fraction": fraction,
        "amounts_mol": amounts,
        "time_s": state.ledger.time_s,
        "mixing": action["mixing"],
    }
    return source.replace(
        inactive_vessels={**state.inactive_vessels, target.vessel_id: target},
        ledger=state.ledger.with_updates(cost=state.ledger.cost + 0.005),
        samples=replace(
            state.samples,
            transfers=(*state.samples.transfers, transfer),
            transfer_tools_used=state.samples.transfer_tools_used + 1,
        ),
        metadata={**state.metadata, "last_material_packet": receipt},
    )
