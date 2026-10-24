"""Internal PDMP delegate of the original stochastic advance strategy."""

from dataclasses import replace
from typing import Any

from chemworld.physchem.stochastic_hybrid import (
    HYBRID_ALGORITHM,
    StochasticHybridSpec,
    advance_stochastic_hybrid,
    validate_hybrid_conservation,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult


class HybridStochasticRuntime:
    def __init__(self, composition: Any) -> None:
        self.composition = composition

    @property
    def models(self) -> dict[str, StochasticHybridSpec]:
        return ({} if self.composition is None or self.composition.physics is None else
                self.composition.physics.models_of("stochastic_hybrid"))

    def supports(self, state: Any) -> bool:
        continuation = state.process.stochastic_continuation
        return state.vessel_id in self.models or (
            continuation is not None and continuation.get("algorithm") == HYBRID_ALGORITHM)

    def validate_world(self) -> None:
        if not self.models:
            return
        physics = self.composition.physics
        if (physics.network.stochastic is not None
                or not set(self.models) <= physics.initial_states.keys()):
            raise ValueError("hybrid model cannot share an all-integer SSA declaration")
        for vessel in self.models:
            state = physics.initial_states[vessel]
            self.validate(state)
            if any(vessel in bindings for kind, bindings in physics.models.items()
                   if kind != "stochastic_hybrid") or state.state_slots.slots:
                raise ValueError("hybrid model cannot silently ignore another physical model")
            if any(c.vessel == vessel and c.kind not in {
                    "stochastic", "observation", "lab_schedule"}
                   for c in self.composition.components):
                raise ValueError("hybrid first slice excludes material dosing and other controls")
            if any(r.equipment_type not in {"batch_reactor", "instrument"}
                   for r in state.equipment.equipment.values()):
                raise ValueError("hybrid first slice has no continuous equipment coupling")
            from chemworld.world.composition import physical_instruments

            for name in physical_instruments(self.composition, vessel):
                instrument = self.composition.modules.instrument_specs()[name]
                if (instrument.contract.sample_consumption_L or instrument.contract.destructive
                        or instrument.discrete_sampling):
                    raise ValueError("hybrid first slice excludes consuming samples")

    def validate(self, state: Any) -> StochasticHybridSpec:
        spec = self.models.get(state.vessel_id)
        if spec is None:
            raise ValueError("hybrid state requires its physical partition declaration")
        if state.temperature_K != spec.temperature_K or state.volume_L != spec.volume_L:
            raise ValueError("hybrid state changed its declared fixed volume or temperature")
        if (len(state.phases.phases) != 1 or state.phase != "liquid"
                or next(iter(state.phases.phases.values())).species_amounts_mol
                != state.species_amounts
                or state.state_slots.slots or state.samples.samples):
            raise ValueError("hybrid requires one authoritative phase and no hidden material stock")
        continuation = state.process.stochastic_continuation
        if continuation is None or continuation.get("time_s") != state.ledger.time_s:
            raise ValueError("hybrid process and original world clocks disagree")
        network = self.composition.physics.network
        advance_stochastic_hybrid(network, spec, state.species_amounts, continuation, 0)
        # The declaration, not editable hidden continuation, owns each closed stock.
        template = self.composition.physics.initial_states[state.vessel_id]
        validate_hybrid_conservation(network, spec, template.species_amounts, state.species_amounts)
        return spec

    def reconcile_initialization(self, transition: Any) -> Any:
        # Ordinary session reset installs a fresh original template directly. A
        # material/vessel/equipment lifecycle boundary must not erase pending risks.
        affected = set(transition.receipt.get("vessels", {}))
        scope = transition.receipt.get("scope")
        if scope == "equipment":
            # Equipment receipts identify their actual owning vessel, not a
            # material-reset `vessels` map. Other independent owners may replace
            # apparatus without resetting this hybrid's Poisson continuation.
            affected.update(transition.receipt[key]["attached_vessel_id"]
                            for key in ("removed_device", "introduced_device"))
        if (affected & set(self.models) or (self.models and
                scope == "batch")):
            raise ValueError("hybrid partial lifecycle reset requires a fresh session")
        for local in (transition.state, *transition.state.inactive_vessels.values()):
            if self.supports(local):
                self.validate(local)
        return transition

    def advance(self, state: Any, action: dict[str, Any], *, heat: bool = False,
                 jacket_program: Any = None) -> ModelProviderResult:
        if (heat or jacket_program is not None or action.get("operation", "wait") != "wait"
                or state.quenched or state.terminated):
            return ModelProviderResult({}, diagnostics={}, success=False, outcome="rejected",
                failure_reason="hybrid_fixed_boundary_only")
        try:
            spec = self.validate(state)
        except (ValueError, KeyError, TypeError) as exc:
            return ModelProviderResult({}, success=False, outcome="rejected",
                failure_reason="hybrid_contract_violation", diagnostics={"reason": str(exc)})
        try:
            result = advance_stochastic_hybrid(self.composition.physics.network, spec,
                state.species_amounts, state.process.stochastic_continuation,
                float(action["duration_s"]))
            candidate = state.replace(species_amounts=result.amounts_mol,
                process=replace(state.process, last_observation_record=None,
                                stochastic_continuation=result.continuation),
                ledger=advance_ledger(state.ledger, result.actual_duration_s))
            self.validate(candidate)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult({}, success=False, outcome="numerical_failure",
                failure_reason="hybrid_advance_failed", diagnostics={"solver_error": str(exc),
                    "failed_candidate_work": getattr(exc, "work", {})})
        return ModelProviderResult({}, state=candidate, actual_duration_s=result.actual_duration_s,
            diagnostics={"solver": HYBRID_ALGORITHM, "events_executed": len(result.events),
                         "stochastic_work": result.work,
                         "energy_scope": "isothermal kinetics only; heat balance not modelled"})
