"""SSA adapter of the existing physical advance coordinator and transaction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState
from chemworld.foundation.lifecycle import LifecycleTransition
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, ProcessLedger
from chemworld.physchem.stochastic_kinetics import (
    StochasticSpec,
    advance_stochastic,
    counts_to_mol,
    initial_continuation,
    mol_to_counts,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

if TYPE_CHECKING:
    from chemworld.physchem.reactor_shared import JacketTemperatureProgram
    from chemworld.physchem.stochastic_adaptive_hybrid import AdaptiveHybridSpec
    from chemworld.physchem.stochastic_hybrid import StochasticHybridSpec
    from chemworld.physchem.stochastic_sde import StochasticSDESpec
    from chemworld.world.composition import WorldCompositionSpec
    from chemworld.world.physical_world import CompositionPhysics


def install_stochastic_state(
    state: WorldState, *, seed: int, algorithm: str = "gillespie-direct-v1",
) -> WorldState:
    mol_to_counts(state.species_amounts)
    return state.replace(
        process=replace(
            cast(ProcessLedger, state.process),
            stochastic_continuation=initial_continuation(
                seed, state.ledger.time_s, algorithm=algorithm,
            ),
        )
    )


class StochasticRuntime:
    supported_operations = ("wait", "add_component")

    def __init__(self, composition: WorldCompositionSpec | None = None) -> None:
        self.composition = composition
        from chemworld.runtime.stochastic_adaptive import AdaptiveHybridStochasticRuntime
        from chemworld.runtime.stochastic_hybrid import HybridStochasticRuntime
        from chemworld.runtime.stochastic_sde import SDEStochasticRuntime

        self.hybrid = HybridStochasticRuntime(composition)
        self.sde = SDEStochasticRuntime(composition)
        self.adaptive = AdaptiveHybridStochasticRuntime(composition)

    def supports(self, state: WorldState) -> bool:
        return (cast(ProcessLedger, state.process).stochastic_continuation is not None
                or state.vessel_id in self.hybrid.models or state.vessel_id in self.sde.models
                or state.vessel_id in self.adaptive.models)

    def validate_world(self) -> None:
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
                   for r in cast(EquipmentLedger, state.equipment).equipment.values()):
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

    def reconcile_initialization(self, transition: LifecycleTransition) -> LifecycleTransition:
        return self.reconcile_initialization_from(transition, None)

    def reconcile_initialization_from(
        self, transition: LifecycleTransition, previous: WorldState | None,
    ) -> LifecycleTransition:
        if self.composition is None or self.composition.physics is None:
            return transition
        physics = self.composition.physics
        transition = self.hybrid.reconcile_initialization(transition)
        transition = self.sde.reconcile_initialization(transition, previous=previous)
        transition = self.adaptive.reconcile_initialization(transition, previous=previous)

        def reset_local(state: WorldState) -> WorldState:
            template = physics.initial_states[state.vessel_id]
            if (self.hybrid.supports(template) or self.sde.supports(template)
                    or self.adaptive.supports(template)):
                return state
            memory = cast(ProcessLedger, template.process).stochastic_continuation
            if memory is not None and not self.supports(state):
                continuation = deepcopy(memory)
                continuation.update(time_s=state.ledger.time_s, pending=None, signature=None)
                state = state.replace(
                    process=replace(cast(ProcessLedger, state.process),
                                    stochastic_continuation=continuation)
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

    def validate(
        self, state: WorldState,
    ) -> dict[str, int] | AdaptiveHybridSpec | StochasticSDESpec | StochasticHybridSpec:
        if self.adaptive.supports(state):
            return self.adaptive.validate(state)
        if self.sde.supports(state):
            return self.sde.validate(state)
        if self.hybrid.supports(state):
            return self.hybrid.validate(state)
        network = (self.composition.physics.network
                   if self.composition is not None and self.composition.physics is not None
                   else None)
        if network is None or network.stochastic is None:
            raise ValueError("SSA requires a declared stochastic physical network")
        spec = StochasticSpec(**network.stochastic)
        if state.temperature_K != spec.temperature_K or state.volume_L != spec.volume_L:
            raise ValueError("SSA requires its declared constant temperature and volume")
        if state.quenched or state.terminated:
            raise ValueError("cannot advance stopped stochastic chemistry")
        phases = cast(PhaseLedger, state.phases).phases
        if (
            len(phases) != 1
            or state.phase != "liquid"
            or next(iter(phases.values())).species_amounts_mol != state.species_amounts
        ):
            raise ValueError("SSA requires a single authoritative well-mixed inventory")
        continuation = cast(ProcessLedger, state.process).stochastic_continuation
        if continuation is None or continuation["time_s"] != state.ledger.time_s:
            raise ValueError("SSA process and session clocks disagree")
        counts = mol_to_counts(state.species_amounts)
        advance_stochastic(network, counts, continuation, 0)
        return counts

    def validate_state(self, state: WorldState) -> None:
        if self.adaptive.supports(state):
            self.adaptive.validate(state)
        elif self.sde.supports(state):
            self.sde.validate(state)
        elif self.hybrid.supports(state):
            self.hybrid.validate(state)

    def reconcile_material_change(self, before: WorldState, after: WorldState) -> WorldState:
        if self.sde.supports(before):
            return self.sde.reconcile_material_change(before, after)
        return after

    def advance(
        self, state: WorldState, action: dict[str, Any], *, heat: bool = False,
        jacket_program: JacketTemperatureProgram | None = None,
    ) -> ModelProviderResult:
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
            # All continuous/hybrid delegates have already returned above.
            counts = cast(dict[str, int], self.validate(state))
        except ValueError as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="ssa_contract_violation",
                diagnostics={"reason": str(exc)},
            )
        try:
            # The discrete validation established the physical declaration and memory.
            composition = cast("WorldCompositionSpec", self.composition)
            physics = cast("CompositionPhysics", composition.physics)
            process = cast(ProcessLedger, state.process)
            continuation = cast(dict[str, Any], process.stochastic_continuation)
            result = advance_stochastic(
                physics.network,
                counts,
                continuation,
                float(action["duration_s"]),
            )
            candidate = state.replace(
                species_amounts=counts_to_mol(result.counts),
                process=replace(
                    process,
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
                "solver": continuation["algorithm"],
                "events_executed": (result.continuation["events_committed"]
                                    - continuation["events_committed"]),
                "stochastic_work": result.work,
                "energy_scope": "isothermal kinetics only; heat balance not modelled",
            },
        )
