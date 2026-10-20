"""Isothermal liquid closure consumed by existing operations and coordinator."""

from dataclasses import replace
from math import isclose, isfinite

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.equilibrium import entrain_liquid_amounts
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult


def install_liquids(state, model, result, *, settled=False):
    phases = {}
    for index, amounts in enumerate(result.phases or (result.feed_amounts_mol,)):
        key = f"liquid_{index}"
        phases[key] = PhaseRecord(
            key, state.vessel_id, "liquid", model.volume_l(amounts), amounts, settled=settled,
            metadata={"carrier_representation": "species_inventory"})
    volume = sum(p.volume_L for p in phases.values())
    if volume > state.vessels.vessels[state.vessel_id].max_volume_L:
        raise ValueError("liquid equilibrium candidate exceeds its finite vessel capacity")
    return state.replace(
        species_amounts=dict(result.feed_amounts_mol), phases=PhaseLedger(phases),
        volume_L=volume, temperature_K=model.temperature_K, pressure_Pa=model.pressure_Pa,
        phase="liquid", process=replace(state.process, last_observation_record=None))


def validate_liquid_state(state, model, *, require_equilibrium=True):
    if (state.state_slots.slots or state.phase != "liquid"
            or not isclose(state.temperature_K, model.temperature_K, abs_tol=1e-9)
            or not isclose(state.pressure_Pa, model.pressure_Pa, abs_tol=1e-6)
            or set(state.species_amounts) != set(model.activity_model.component_ids)):
        raise ValueError("strict liquid state must match its declared isothermal material model")
    for phase in state.phases.phases.values():
        if (phase.phase_type != "liquid" or phase.solvents.volume_L
                or phase.metadata.get("carrier_representation") != "species_inventory"
                or not isclose(phase.volume_L, model.volume_l(phase.species_amounts_mol),
                               abs_tol=1e-10)):
            raise ValueError("strict liquid phases require registered amounts and declared holdup")
    if require_equilibrium:
        result = model.solve(state.species_amounts)
        actual = sorted(state.phases.phases.values(), key=lambda p: tuple(
            p.species_amounts_mol[k] / max(1e-300, sum(p.species_amounts_mol.values()))
            for k in model.activity_model.component_ids))
        expected = sorted(result.phases, key=lambda p: tuple(
            p[k] / sum(p.values()) for k in model.activity_model.component_ids))
        if len(actual) != len(expected) and result.phase_status != "empty":
            raise ValueError("strict liquid initial state has the wrong number of phases")
        if any(abs(phase.species_amounts_mol[k] - amounts[k]) > 1e-8 * max(
            1, sum(state.species_amounts.values())) for phase, amounts in zip(
                actual, expected, strict=False)
               for k in amounts):
            raise ValueError("strict liquid initial phase inventories are not at equilibrium")


def exchange_liquid_packet(source, target, physics, amounts, *, fraction=0, fixed_source=False):
    model = physics.liquid_models[target.vessel_id]
    validate_liquid_state(source, model, require_equilibrium=False)
    validate_liquid_state(target, model, require_equilibrium=False)
    if not fixed_source and physics.liquid_models.get(source.vessel_id) != model:
        raise ValueError("liquid packet endpoints require identical material/reference models")
    available = sum(source.species_amounts.values())
    moved = sum(amounts.values())
    fraction = moved / available
    if not 0 <= fraction <= 1 or any(abs(amounts[k] - fraction * source.species_amounts[k])
                                    > 1e-12 for k in amounts):
        raise ValueError("liquid packet must be a representative finite source withdrawal")
    remaining = {k: v - amounts.get(k, 0) for k, v in source.species_amounts.items()}
    received = {k: v + amounts.get(k, 0) for k, v in target.species_amounts.items()}
    target_candidate = install_liquids(target, model, model.solve(received))
    if fixed_source:
        source_candidate = source.replace(
            species_amounts=remaining, volume_L=source.volume_L * (1 - fraction),
            phases=PhaseLedger({k: replace(p, volume_L=p.volume_L * (1 - fraction),
                species_amounts_mol={s: v * (1 - fraction)
                                    for s, v in p.species_amounts_mol.items()})
                               for k, p in source.phases.phases.items()}))
    else:
        source_candidate = install_liquids(source, model, model.solve(remaining))
    references = {k: v * fraction for k, v in source.species.initial_amounts_mol.items()}
    target_candidate = target_candidate.replace(
        species=species_with_added_initial_amounts(target.species, references))
    source_candidate = source_candidate.replace(species=replace(source.species,
        initial_amounts_mol={k: v * (1 - fraction)
                            for k, v in source.species.initial_amounts_mol.items()}))
    return source_candidate, target_candidate, {
        "source": source.vessel_id, "destination": target.vessel_id, "amounts_mol": amounts,
        "model": "strict_isothermal_liquid", "not_covered": ["mixing_heat", "PV_work"]}


class LiquidEquilibriumRuntime:
    def __init__(self, composition):
        self.composition = composition

    def supports(self, state):
        return (self.composition is not None and self.composition.physics is not None
                and state.vessel_id in self.composition.physics.liquid_models)

    def model(self, state):
        return self.composition.physics.liquid_models[state.vessel_id]

    def advance(self, state, action, *, heat=False, jacket_program=None, mixing=False):
        if heat or jacket_program is not None:
            return ModelProviderResult({}, success=False, outcome="rejected",
                                       failure_reason="liquid_model_is_isothermal")
        try:
            model = self.model(state)
            validate_liquid_state(state, model, require_equilibrium=False)
            result = model.solve(state.species_amounts)
            duration = float(action["duration_s"])
            candidate = install_liquids(state, model, result).replace(ledger=advance_ledger(
                state.ledger, duration,
                cost_delta=.01 + duration / 3600 * .015 if mixing else 0))
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult({}, success=False, outcome="numerical_failure",
                                       failure_reason="liquid_equilibrium_failed",
                                       diagnostics={"solver_error": str(exc)})
        return ModelProviderResult({}, state=candidate, actual_duration_s=duration, diagnostics={
            "liquid_equilibrium": result.to_dict(), "not_covered": ["mixing_heat", "PV_work"]})

    def wash(self, state, action):
        model = self.model(state)
        source_id = self.composition.physics.feed_sources.get(model.wash_feed_id)
        if source_id is None or source_id not in state.inactive_vessels:
            raise ValueError("strict wash requires a declared finite liquid source")
        source = state.inactive_vessels[source_id]
        volume = float(action["wash_volume_L"])
        if not isfinite(volume) or not 0 < volume <= source.volume_L:
            raise ValueError("wash exceeds its finite source volume")
        additions = {k: v * volume / source.volume_L for k, v in source.species_amounts.items()}
        source_candidate, mixed, receipt = exchange_liquid_packet(
            source, state, self.composition.physics, additions, fixed_source=True)
        # A miscible wash remains one liquid. It cannot manufacture a second
        # phase in order to discard arbitrary material.
        phases = list(mixed.phases.phases.values())
        removed = dict.fromkeys(model.activity_model.component_ids, 0.0)
        if len(phases) == 2:
            discarded = phases[1 - model.retained_phase]
            retained = phases[model.retained_phase]
            removed, _, _ = entrain_liquid_amounts(
                discarded.species_amounts_mol, retained.species_amounts_mol,
                model.entrainment_fraction)
        remaining = {k: v - removed[k] for k, v in mixed.species_amounts.items()}
        candidate = install_liquids(mixed, model, model.solve(remaining), settled=True)
        history = list(state.metadata.get("removed_phase_inventory_history", ()))
        if sum(removed.values()) > 0:
            history.append({"operation": "wash", "inventories": {"wash_waste": {
                "species_amounts_mol": removed, "volume_L": model.volume_l(removed),
                "unit": "mol", "carrier_representation": "species_inventory"}}})
        return candidate.replace(
            inactive_vessels={**state.inactive_vessels, source_id: source_candidate},
            ledger=advance_ledger(state.ledger, 0, cost_delta=.02 + .25 * volume),
            metadata={**candidate.metadata, "removed_phase_inventory_history": history,
                      "last_liquid_wash": {"source": source_id, "amounts_mol": additions,
                                           "wash_volume_L": volume, "removed_mol": removed,
                                           "packet": receipt}})
