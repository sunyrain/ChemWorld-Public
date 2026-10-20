"""Finite advective material packets using the same owned inventories and U(n,T).

Gas volumes are vessel geometry. A packet carries inlet enthalpy and any declared
mobile carrier internal energy; it never carries equipment. This zero-duration
mixing operation does not simulate a valve, pipe or pressure-driven flow.
"""

from dataclasses import replace
from math import isfinite

from scipy.optimize import brentq

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.state_ledgers import PhaseLedger, ProcessLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.coupled_system import ConstantCvThermodynamics
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.runtime.coupled_thermal import local_thermodynamics, pressure_for_state


def feed_packet(state, physics, action):
    source_id = physics.feed_sources.get(action.get("component"))
    if source_id is None or source_id not in state.inactive_vessels:
        raise ValueError("feed must identify an inactive finite source")
    source = state.inactive_vessels[source_id]
    amount = action.get("amount_mol")
    available = sum(source.species_amounts.values())
    if (
        isinstance(amount, bool)
        or not isinstance(amount, (float, int))
        or not isfinite(amount)
        or not 0 <= amount <= available
        or available <= 0
    ):
        raise ValueError("feed amount exceeds finite source inventory")
    return source_id, {
        key: value / available * amount for key, value in source.species_amounts.items()
    }


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
        def residual(temperature):
            return thermo.internal_energy_j(replace(packed, temperature_K=temperature)) - energy

        lower, upper = model.minimum_temperature_K, model.stop_temperature_K
        if residual(lower) > 0 or residual(upper) <= 0:
            raise ValueError("packet energy leaves the supported caloric operating range")
        temperature = float(brentq(residual, lower, upper, xtol=1e-10, rtol=1e-12))
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
    if target.vessel_id in physics.liquid_models:
        from chemworld.runtime.liquid_equilibrium import exchange_liquid_packet

        return exchange_liquid_packet(source, target, physics, amounts,
                                      fraction=fraction, fixed_source=fixed_source)
    if target.vessel_id in physics.flash_models:
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
    if target.vessel_id not in physics.thermal_models:
        raise ValueError("receiving vessel requires declared thermodynamics")
    model = physics.thermal_models[target.vessel_id]
    if model.pressure_law != "ideal_gas":
        raise ValueError("gas packet requires an ideal-gas receiving boundary")
    if set(source.species_amounts) != set(target.species_amounts):
        raise ValueError("gas packet species must share the receiving inventory layout")
    moved = source.state_slots.material().portion(fraction, target.vessel_id)
    target_phase = next(iter(target.phases.phases))
    moved = StateSlots(
        tuple(replace(slot, spec=replace(slot.spec, phase_id=target_phase)) for slot in moved.slots)
    )
    source_slots = StateSlots(
        (
            *source.state_slots.stationary().slots,
            *source.state_slots.material().portion(1 - fraction, source.vessel_id).slots,
        )
    )
    target_slots = StateSlots(
        (*target.state_slots.stationary().slots, *target.state_slots.material().merge(moved).slots)
    )
    source_amounts = {
        key: value - amounts.get(key, 0) for key, value in source.species_amounts.items()
    }
    if any(value < 0 for value in source_amounts.values()):
        raise ValueError("material packet exceeds its source")
    target_amounts = {
        key: value + amounts.get(key, 0) for key, value in target.species_amounts.items()
    }
    # Reference state and species data must agree before transporting energy.
    if not fixed_source:
        source_model = physics.thermal_models[source.vessel_id]
        if source_model.thermodynamics.species != model.thermodynamics.species or (
            source_model.thermodynamics.reference_temperature_K
            != model.thermodynamics.reference_temperature_K
        ):
            raise ValueError("material endpoints must share thermodynamic reference data")
    thermal = model.thermodynamics
    enthalpy = sum(
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
    carrier_energy += sum(
        amount * thermal.species_energy_j_mol(key, source.temperature_K)
        for key, amount in moved.species_amounts().items()
    )
    energy = enthalpy + carrier_energy
    old_target_energy = _energy(target, model)
    target_candidate = _set_energy(
        _replace_inventory(target, target_amounts, target_slots), model, old_target_energy + energy
    )
    reference_fraction = sum(amounts.values()) / sum(source.species_amounts.values())
    reference_amounts = {
        key: value * reference_fraction for key, value in source.species.initial_amounts_mol.items()
    }
    target_candidate = target_candidate.replace(
        species=species_with_added_initial_amounts(target.species, reference_amounts)
    )
    source_candidate = _replace_inventory(source, source_amounts, source_slots)
    if fixed_source:
        source_candidate = source_candidate.replace(
            pressure_Pa=sum(source_amounts.values())
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
        "source": source.vessel_id,
        "destination": target.vessel_id,
        "amounts_mol": amounts,
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
            cost=state.ledger.cost + 0.03 * sum(amounts.values()) / 0.01
        ),
        metadata={**target.metadata, "last_material_packet": receipt},
    )


def route_gas_packet(state, physics, action):
    from chemworld.runtime.mineral_kinetics import kinetic_record, route_mineral_phase

    if kinetic_record(state) is not None:
        return route_mineral_phase(state, physics, action)
    from chemworld.runtime.condensed_material_routing import background_model

    connections = state.metadata["component_network"]["connections"]
    index = action.get("connection")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(connections):
        raise ValueError("unknown material connection")
    edge = connections[index]
    if edge["source_vessel"] != state.vessel_id or edge["source"]["port"] != "out":
        raise ValueError("select the source vessel's representative material outlet")
    fraction = action.get("transfer_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("material fraction must be in (0, 1]")
    target = state.inactive_vessels[edge["target_vessel"]]
    if isinstance(action.get("mixing"), bool) or action.get("mixing") not in (0, 1):
        raise ValueError("declare empty-only or allowed mixing")
    target_occupied = (target.volume_L > 0 if background_model(target,physics) is not None
                       else sum(target.species_amounts.values()) > 1e-12)
    if target_occupied and action["mixing"] != 1:
        raise ValueError("occupied destination requires explicit mixing")
    empty = (state.volume_L <= 0 if background_model(state,physics) is not None
             else sum(state.species_amounts.values()) <= 0)
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
