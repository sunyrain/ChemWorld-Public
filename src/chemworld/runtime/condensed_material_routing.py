"""Finite liquid packets for an explicitly declared negligible-PV carrier.

The carrier remains an L stock. Molecular solutes and mobile bound loadings
carry their actual U; no gas RT enthalpy or molecular carrier is fabricated.
"""

from dataclasses import replace
from math import isclose, isfinite

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import PhaseLedger, ProcessLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.runtime.thermal_material_routing import _energy, _set_energy


def background_model(state, physics):
    if physics is None:
        return None
    model = physics.thermal_models.get(state.vessel_id)
    return model if model is not None and model.condensed_background is not None else None


def source_models(physics, source_id):
    return [
        m
        for m in physics.thermal_models.values()
        if m.condensed_background is not None
        and m.condensed_background.source_vessel_id == source_id
    ]


def background_source_energy(state, physics):
    models = source_models(physics, state.vessel_id)
    if not models:
        raise ValueError("carrier stock has no declared caloric reference")
    model = models[0]
    return (
        model.condensed_background.volumetric_cv_J_L_K
        * state.volume_L
        * (state.temperature_K - model.thermodynamics.reference_temperature_K)
    )


def validate_background_sources(physics, composition):
    declared = set()
    for target_id, model in physics.thermal_models.items():
        bg = model.condensed_background
        if bg is None or bg.source_vessel_id is None:
            continue
        name = bg.source_vessel_id
        if name not in physics.initial_states or name == target_id:
            raise ValueError("carrier source must be a distinct declared finite vessel")
        source = physics.initial_states[name]
        if (
            source.phase != "liquid"
            or len(source.phases.phases) != 1
            or any(source.species_amounts.values())
            or source.state_slots.slots
            or name
            in (
                physics.thermal_models.keys()
                | physics.flash_models.keys()
                | physics.liquid_models.keys()
                | physics.column_models.keys()
                | physics.concentration_models.keys()
            )
            or any(
                c.vessel == name and composition.modules.components()[c.kind].processes
                for c in composition.components
            )
        ):
            raise ValueError("carrier supply must be a static, pure, finite liquid stock")
        phase = next(iter(source.phases.phases.values()))
        if not isclose(
            source.temperature_K, bg.feed_temperature_K, abs_tol=1e-10
        ) or phase.solvents != SolventInventory.pure(bg.solvent_index, source.volume_L):
            raise ValueError("carrier supply must match its declared temperature and identity")
        signature = (
            bg.volumetric_cv_J_L_K,
            bg.feed_temperature_K,
            bg.solvent_index,
            model.thermodynamics.reference_temperature_K,
        )
        for other in source_models(physics, name):
            ob = other.condensed_background
            if signature != (
                ob.volumetric_cv_J_L_K,
                ob.feed_temperature_K,
                ob.solvent_index,
                other.thermodynamics.reference_temperature_K,
            ):
                raise ValueError("shared carrier source needs consistent caloric references")
        declared.add(name)
    return declared


def validate_carrier_input(before, after, action, physics):
    """Authenticate internal receipt against the installed property/source model."""
    from chemworld.runtime.solvent_transport import internal_carrier_input

    receipt = internal_carrier_input(before, after, action)
    if receipt is None:
        return None
    model = background_model(before, physics)
    if model is None or model.condensed_background.source_vessel_id != receipt["source"]:
        raise ValueError("no finite carrier source producer is installed for this input")
    bg = model.condensed_background
    source, target = (
        before.inactive_vessels[receipt["source"]],
        after.inactive_vessels[receipt["source"]],
    )
    energy = background_source_energy(source, physics) - background_source_energy(target, physics)
    if (
        bg.solvent_index != receipt["solvent_index"]
        or not isclose(source.temperature_K, bg.feed_temperature_K, abs_tol=1e-10)
        or not isclose(receipt["inlet_energy_J"], energy, abs_tol=1e-6)
        or not isclose(_energy(after, model) - _energy(before, model), energy, abs_tol=1e-6)
    ):
        raise ValueError("finite carrier inlet disagrees with its source/receiver energy")
    return receipt


def _install(state, amounts, slots, volume, solvents):
    if state.phase != "liquid" or len(state.phases.phases) != 1:
        raise ValueError("condensed packets require one declared liquid compartment")
    key, phase = next(iter(state.phases.phases.items()))
    return state.replace(
        species_amounts=amounts,
        volume_L=volume,
        state_slots=slots,
        phases=PhaseLedger(
            {key: replace(phase, species_amounts_mol=amounts, volume_L=volume, solvents=solvents)}
        ),
        process=replace(state.process or ProcessLedger(), last_observation_record=None),
    )


def add_background(state, physics, *, volume_L, solvent_index, cost_delta):
    model = background_model(state, physics)
    if model is None or solvent_index != model.condensed_background.solvent_index:
        raise ValueError(
            "carrier addition must use its explicitly declared property/stock identity"
        )
    background = model.condensed_background
    if not isfinite(volume_L) or volume_L <= 0:
        raise ValueError("carrier input must be a positive finite volume")
    source_id = background.source_vessel_id
    source = None
    taken = SolventInventory.pure(solvent_index, volume_L)
    remaining = None
    requested = volume_L
    if source_id is not None:
        source = state.inactive_vessels.get(source_id)
        if source is None:
            raise ValueError("finite carrier input requires its inactive source vessel")
        source_phase = next(iter(source.phases.phases.values()))
        taken, remaining = source_phase.solvents.split(volume_L)
        volume_L = taken.volume_L
        if volume_L <= 0:
            raise ValueError("finite carrier supply is exhausted")
    volume = state.volume_L + volume_L
    if volume > state.vessels.vessels[state.vessel_id].max_volume_L:
        raise ValueError("finite carrier addition exceeds the receiving vessel")
    phase = next(iter(state.phases.phases.values()))
    inlet_energy = (
        background.volumetric_cv_J_L_K
        * volume_L
        * (background.feed_temperature_K - model.thermodynamics.reference_temperature_K)
    )
    initial = _energy(state, model)
    candidate = _set_energy(
        _install(
            state,
            dict(state.species_amounts),
            state.state_slots,
            volume,
            phase.solvents + taken,
        ),
        model,
        initial + inlet_energy,
    )
    residual = _energy(candidate, model) - initial - inlet_energy
    if abs(residual) > 1e-6:
        raise ValueError("finite carrier mixing failed its declared energy balance")
    if source is not None:
        source = _install(
            source, dict(source.species_amounts), source.state_slots, remaining.volume_L, remaining
        )
        candidate = candidate.replace(
            inactive_vessels={**state.inactive_vessels, source_id: source}
        )
    receipt = {
        "source": source_id or "supply:carrier",
        "boundary_kind": "internal" if source_id else "external",
        "destination": state.vessel_id,
        "requested_volume_L": requested,
        "volume_L": volume_L,
        "solvent_index": solvent_index,
        "inlet_temperature_K": background.feed_temperature_K,
        "inlet_energy_J": inlet_energy,
        "energy_model": "declared_condensed_negligible_PV",
        "energy_residual_J": residual,
    }
    history = list(state.metadata.get("carrier_packet_history", ()))
    history.append(receipt)
    return candidate.replace(
        ledger=state.ledger.with_updates(
            cost=state.ledger.cost + cost_delta * volume_L / requested
        ),
        metadata={**candidate.metadata, "carrier_packet_history": history},
    )


def exchange_condensed_packet(source, target, physics, amounts, *, fraction, fixed_source=False):
    if fixed_source:
        raise ValueError("condensed carrier supply uses its finite L stock, not a molar gas feed")
    source_model, target_model = (
        background_model(source, physics),
        background_model(target, physics),
    )
    if source_model is None or target_model is None or source.vessel_id == target.vessel_id:
        raise ValueError("condensed packet needs two distinct authored thermal compartments")
    a, b = source_model.condensed_background, target_model.condensed_background
    if (
        a.solvent_index != b.solvent_index
        or a.volumetric_cv_J_L_K != b.volumetric_cv_J_L_K
        or source_model.thermodynamics.species != target_model.thermodynamics.species
        or source_model.thermodynamics.reference_temperature_K
        != (target_model.thermodynamics.reference_temperature_K)
        or not isclose(source.pressure_Pa, target.pressure_Pa, rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError(
            "liquid endpoints must share caloric/carrier and fixed-pressure references"
        )
    if (
        not 0 < fraction <= 1
        or set(amounts) != set(source.species_amounts)
        or any(
            not isclose(n, source.species_amounts[k] * fraction, rel_tol=0, abs_tol=1e-12)
            for k, n in amounts.items()
        )
    ):
        raise ValueError("liquid packet must debit its actual representative source fraction")
    source_phase = next(iter(source.phases.phases.values()))
    target_phase = next(iter(target.phases.phases.values()))
    volume = source.volume_L * fraction
    if (
        volume <= 0
        or target.volume_L + volume > target.vessels.vessels[target.vessel_id].max_volume_L
    ):
        raise ValueError("liquid packet leaves its finite source or receiving volume domain")
    moved = source.state_slots.material().portion(fraction, target.vessel_id)
    moved = StateSlots(
        tuple(
            replace(slot, spec=replace(slot.spec, phase_id=target_phase.phase_id))
            for slot in moved.slots
        )
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
    thermal = target_model.thermodynamics
    fluid_energy = sum(
        n * thermal.species_energy_j_mol(k, source.temperature_K) for k, n in amounts.items()
    )
    background_energy = (
        a.volumetric_cv_J_L_K * volume * (source.temperature_K - thermal.reference_temperature_K)
    )
    carrier_energy = moved.heat_capacity() * (
        source.temperature_K - thermal.reference_temperature_K
    )
    carrier_energy += sum(
        n * thermal.species_energy_j_mol(k, source.temperature_K)
        for k, n in moved.species_amounts().items()
    )
    packet_energy = fluid_energy + background_energy + carrier_energy
    old_source, old_target = _energy(source, source_model), _energy(target, target_model)
    source_candidate = _set_energy(
        _install(
            source,
            {k: n - amounts[k] for k, n in source.species_amounts.items()},
            source_slots,
            source.volume_L - volume,
            source_phase.solvents.scale(1 - fraction),
        ),
        source_model,
        old_source - packet_energy,
    )
    target_candidate = _set_energy(
        _install(
            target,
            {k: n + amounts[k] for k, n in target.species_amounts.items()},
            target_slots,
            target.volume_L + volume,
            target_phase.solvents + source_phase.solvents.scale(fraction),
        ),
        target_model,
        old_target + packet_energy,
    )
    source_candidate = source_candidate.replace(
        species=replace(
            source_candidate.species,
            initial_amounts_mol={
                k: n * (1 - fraction) for k, n in source.species.initial_amounts_mol.items()
            },
        )
    )
    target_candidate = target_candidate.replace(
        species=species_with_added_initial_amounts(
            target.species, {k: n * fraction for k, n in source.species.initial_amounts_mol.items()}
        )
    )
    residuals = (
        _energy(source_candidate, source_model) - old_source + packet_energy,
        _energy(target_candidate, target_model) - old_target - packet_energy,
    )
    if max(map(abs, residuals)) > 1e-6:
        raise ValueError("finite liquid packet failed its source/receiver energy balance")
    receipt = {
        "source": source.vessel_id,
        "destination": target.vessel_id,
        "amounts_mol": dict(amounts),
        "volume_L": volume,
        "inlet_temperature_K": source.temperature_K,
        "solute_energy_J": fluid_energy,
        "background_energy_J": background_energy,
        "mobile_carrier_energy_J": carrier_energy,
        "packet_energy_J": packet_energy,
        "energy_model": "declared_condensed_negligible_PV",
        "source_energy_residual_J": residuals[0],
        "destination_energy_residual_J": residuals[1],
    }
    return source_candidate, target_candidate, receipt


def reconcile_condensed_initialization(previous, fresh, transition, physics):
    """Mix new material with retained body/carrier energy at a lifecycle boundary."""
    scope = transition.receipt["scope"]
    if scope not in {"material", "vessel"}:
        return transition

    def locals_of(state):
        return {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}

    old, templates, current = locals_of(previous), locals_of(fresh), locals_of(transition.state)
    receipts = {}
    for name in transition.receipt["vessels"]:
        if name in physics.crystal_models:
            continue
        state = current[name]
        model = background_model(state, physics)
        if model is None:
            continue
        thermal = model.thermodynamics
        initial_t, retained_t = templates[name].temperature_K, old[name].temperature_K
        reference = thermal.reference_temperature_K
        introduced = sum(
            n * thermal.species_energy_j_mol(k, initial_t) for k, n in state.species_amounts.items()
        )
        introduced += (
            model.condensed_background.volumetric_cv_J_L_K
            * state.volume_L
            * (initial_t - reference)
        )
        retained = 0.0
        for slot in state.state_slots.slots:
            keep = slot.spec.owner_kind == "equipment" or (
                scope == "material" and slot.spec.owner_kind == "vessel"
            )
            temperature = retained_t if keep else initial_t
            energy = (
                0.0
                if slot.spec.heat_capacity_field is None
                else slot.values[slot.spec.heat_capacity_field] * (temperature - reference)
            )
            energy += sum(
                slot.values[f.name] * thermal.species_energy_j_mol(f.species_id, temperature)
                for f in slot.spec.fields
                if f.species_id
            )
            if keep:
                retained += energy
            else:
                introduced += energy
        body = thermal.support_heat_capacity_J_K * (
            (retained_t if scope == "material" else initial_t) - reference
        )
        if scope == "material":
            retained += body
        else:
            introduced += body
        expected = introduced + retained
        state = _set_energy(state, model, expected)
        current[name] = state
        receipts[name] = {
            "retained_internal_energy_J": retained,
            "introduced_internal_energy_J": introduced,
            "expected_internal_energy_J": expected,
            "actual_internal_energy_J": _energy(state, model),
            "temperature_K": state.temperature_K,
        }
    if not receipts:
        return transition
    active = current.pop(transition.state.vessel_id)
    return replace(
        transition,
        state=active.replace(inactive_vessels=current),
        receipt={**transition.receipt, "condensed_thermal_reconciliation": receipts},
    )
