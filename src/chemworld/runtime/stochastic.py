"""SSA adapter of the existing physical advance coordinator and transaction."""

from copy import deepcopy
from dataclasses import replace

from chemworld.physchem.stochastic_kinetics import (
    StochasticSpec,
    advance_stochastic,
    counts_to_mol,
    initial_continuation,
    mol_to_counts,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult


def install_stochastic_state(state, *, seed, algorithm="gillespie-direct-v1"):
    mol_to_counts(state.species_amounts)
    return state.replace(
        process=replace(
            state.process,
            stochastic_continuation=initial_continuation(
                seed, state.ledger.time_s, algorithm=algorithm,
            ),
        )
    )


class StochasticRuntime:
    supported_operations = ("wait", "add_component")

    def __init__(self, composition=None):
        self.composition = composition
        from chemworld.runtime.stochastic_adaptive import AdaptiveHybridStochasticRuntime
        from chemworld.runtime.stochastic_hybrid import HybridStochasticRuntime
        from chemworld.runtime.stochastic_sde import SDEStochasticRuntime

        self.hybrid = HybridStochasticRuntime(composition)
        self.sde = SDEStochasticRuntime(composition)
        self.adaptive = AdaptiveHybridStochasticRuntime(composition)

    def supports(self, state):
        return (state.process.stochastic_continuation is not None
                or state.vessel_id in self.hybrid.models or state.vessel_id in self.sde.models
                or state.vessel_id in self.adaptive.models)

    def validate_world(self):
        if self.composition is None or self.composition.physics is None:
            return
        physics = self.composition.physics
        self.hybrid.validate_world()
        self.sde.validate_world()
        self.adaptive.validate_world()
        declared = {c.vessel for c in self.composition.components if c.kind == "stochastic"}
        installed = {key for key, s in physics.initial_states.items() if self.supports(s)}
        discrete = (installed - self.hybrid.models.keys() - self.sde.models.keys()
                    - self.adaptive.models.keys())
        if declared != installed or bool(discrete) != (physics.network.stochastic is not None):
            raise ValueError("SSA network, components and owned process state must agree")
        if installed and installed != set(physics.initial_states) and physics.lab_clock is None:
            raise ValueError(
                "SSA network cannot implicitly advance an undeclared continuous vessel"
            )
        for vessel in installed:
            state = physics.initial_states[vessel]
            if (vessel in self.hybrid.models or vessel in self.sde.models
                    or vessel in self.adaptive.models):
                continue
            self.validate(state)
            competing = (
                physics.models_of("thermal"),
                physics.models_of("flash"),
                physics.models_of("liquid"),
                physics.models_of("column"),
                physics.models_of("concentration"),
                physics.models_of("crystal"),
            )
            if any(vessel in models for models in competing) or state.state_slots.slots:
                raise ValueError("mixed continuous ODE/SSA system is not supported")
            if any(
                c.vessel == vessel and c.kind not in {
                    "stochastic", "observation", "material_input", "lab_schedule",
                }
                for c in self.composition.components
            ):
                raise ValueError("SSA components cannot enable unsupported continuous controls")
            equipment_types = (
                {"batch_reactor", "instrument"} if physics.lab_clock else {"batch_reactor"}
            )
            if any(r.equipment_type not in equipment_types
                   for r in state.equipment.equipment.values()):
                raise ValueError("SSA slice has no continuous equipment coupling")
            from chemworld.world.composition import physical_instruments

            for name in physical_instruments(self.composition, vessel):
                instrument = self.composition.modules.instrument_specs()[name]
                if (
                    instrument.contract.sample_consumption_L or instrument.contract.destructive
                ) and not instrument.discrete_sampling:
                    raise ValueError(
                        "SSA requires an explicit integer sampler, not fractional withdrawal"
                    )

    def reconcile_initialization(self, transition):
        return self.reconcile_initialization_from(transition, None)

    def reconcile_initialization_from(self, transition, previous):
        if self.composition is None or self.composition.physics is None:
            return transition
        transition = self.hybrid.reconcile_initialization(transition)
        transition = self.sde.reconcile_initialization(transition, previous=previous)
        transition = self.adaptive.reconcile_initialization(transition, previous=previous)

        def reset_local(state):
            template = self.composition.physics.initial_states[state.vessel_id]
            if (self.hybrid.supports(template) or self.sde.supports(template)
                    or self.adaptive.supports(template)):
                return state
            if self.supports(template) and not self.supports(state):
                continuation = deepcopy(template.process.stochastic_continuation)
                continuation.update(time_s=state.ledger.time_s, pending=None, signature=None)
                state = state.replace(
                    process=replace(state.process, stochastic_continuation=continuation)
                )
            return state

        state = reset_local(transition.state)
        return replace(
            transition,
            state=state.replace(
                inactive_vessels={
                    key: reset_local(local) for key, local in state.inactive_vessels.items()
                }
            ),
        )

    def validate(self, state):
        if self.adaptive.supports(state):
            return self.adaptive.validate(state)
        if self.sde.supports(state):
            return self.sde.validate(state)
        if self.hybrid.supports(state):
            return self.hybrid.validate(state)
        network = self.composition.physics.network
        spec = StochasticSpec(**network.stochastic)
        if state.temperature_K != spec.temperature_K or state.volume_L != spec.volume_L:
            raise ValueError("SSA requires its declared constant temperature and volume")
        if state.quenched or state.terminated:
            raise ValueError("cannot advance stopped stochastic chemistry")
        if (
            len(state.phases.phases) != 1
            or state.phase != "liquid"
            or next(iter(state.phases.phases.values())).species_amounts_mol != state.species_amounts
        ):
            raise ValueError("SSA requires a single authoritative well-mixed inventory")
        continuation = state.process.stochastic_continuation
        if continuation is None or continuation["time_s"] != state.ledger.time_s:
            raise ValueError("SSA process and session clocks disagree")
        counts = mol_to_counts(state.species_amounts)
        advance_stochastic(network, counts, continuation, 0)
        return counts

    def validate_state(self, state):
        if self.adaptive.supports(state):
            self.adaptive.validate(state)
        elif self.sde.supports(state):
            self.sde.validate(state)
        elif self.hybrid.supports(state):
            self.hybrid.validate(state)

    def reconcile_material_change(self, before, after):
        if self.sde.supports(before):
            return self.sde.reconcile_material_change(before, after)
        return after

    def advance(self, state, action, *, heat=False, jacket_program=None):
        if self.adaptive.supports(state):
            return self.adaptive.advance(state, action, heat=heat, jacket_program=jacket_program)
        if self.sde.supports(state):
            return self.sde.advance(state, action, heat=heat, jacket_program=jacket_program)
        if self.hybrid.supports(state):
            return self.hybrid.advance(state, action, heat=heat, jacket_program=jacket_program)
        if heat or jacket_program is not None:
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="ssa_constant_boundary_only",
            )
        try:
            counts = self.validate(state)
        except ValueError as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="ssa_contract_violation",
                diagnostics={"reason": str(exc)},
            )
        try:
            result = advance_stochastic(
                self.composition.physics.network,
                counts,
                state.process.stochastic_continuation,
                float(action["duration_s"]),
            )
            candidate = state.replace(
                species_amounts=counts_to_mol(result.counts),
                process=replace(
                    state.process,
                    last_observation_record=None,
                    stochastic_continuation=result.continuation,
                ),
                ledger=advance_ledger(state.ledger, result.actual_duration_s),
            )
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="ssa_advance_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            diagnostics={
                "solver": self.composition.physics.network.stochastic["algorithm"],
                "events_executed": (result.continuation["events_committed"]
                                    - state.process.stochastic_continuation["events_committed"]),
                "stochastic_work": result.work,
                "energy_scope": "isothermal kinetics only; heat balance not modelled",
            },
        )
