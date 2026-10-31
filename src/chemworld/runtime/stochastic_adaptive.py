"""Adaptive chemical delegate of the original stochastic physical coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState
from chemworld.foundation.lifecycle import LifecycleTransition
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, ProcessLedger
from chemworld.physchem.stochastic_adaptive_advance import (
    AdaptiveContinuation,
    advance_adaptive_hybrid,
    rebind_adaptive_continuation,
    validate_adaptive_continuation,
)
from chemworld.physchem.stochastic_adaptive_hybrid import (
    ADAPTIVE_HYBRID_ALGORITHM,
    AdaptiveHybridSpec,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

if TYPE_CHECKING:
    from chemworld.physchem.reactor_shared import JacketTemperatureProgram
    from chemworld.world.composition import WorldCompositionSpec
    from chemworld.world.physical_world import CompositionPhysics


class AdaptiveHybridStochasticRuntime:
    def __init__(self, composition: WorldCompositionSpec | None) -> None:
        self.composition = composition

    @property
    def models(self) -> Mapping[str, AdaptiveHybridSpec]:
        if self.composition is None or self.composition.physics is None:
            return {}
        return {
            key: value
            for key, value in self.composition.physics.models_of("stochastic_hybrid").items()
            if isinstance(value, AdaptiveHybridSpec)
        }

    def supports(self, state: WorldState) -> bool:
        memory = cast(ProcessLedger, state.process).stochastic_continuation
        return state.vessel_id in self.models or (
            memory is not None and memory.get("algorithm") == ADAPTIVE_HYBRID_ALGORITHM
        )

    def validate(self, state: WorldState, *, numerical: bool = True) -> AdaptiveHybridSpec:
        spec = self.models.get(state.vessel_id)
        if spec is None:
            raise ValueError("adaptive process requires its actual physical model declaration")
        if state.temperature_K != spec.temperature_K or state.volume_L != spec.volume_L:
            raise ValueError("adaptive process changed its declared fixed temperature or volume")
        if (
            len(cast(PhaseLedger, state.phases).phases) != 1
            or state.phase != "liquid"
            or next(iter(cast(PhaseLedger, state.phases).phases.values())).species_amounts_mol
            != state.species_amounts
            or state.state_slots.slots
            or state.samples.samples
        ):
            raise ValueError("adaptive process requires one authoritative liquid mol inventory")
        memory = cast(ProcessLedger, state.process).stochastic_continuation
        if memory is None or memory.get("time_s") != state.ledger.time_s:
            raise ValueError("adaptive process and original world clocks disagree")
        composition = cast("WorldCompositionSpec", self.composition)
        physics = cast("CompositionPhysics", composition.physics)
        validate_adaptive_continuation(
            physics.network,
            spec,
            state.species_amounts,
            memory,
            numerical=numerical,
        )
        return spec

    def validate_world(self) -> None:
        if not self.models:
            return
        composition = cast("WorldCompositionSpec", self.composition)
        physics = cast("CompositionPhysics", composition.physics)
        if (
            physics.network.stochastic is not None
            or not set(self.models) <= physics.initial_states.keys()
        ):
            raise ValueError("adaptive process cannot compete with an integer SSA declaration")
        from chemworld.world.composition import physical_instruments

        for identity in self.models:
            state = physics.initial_states[identity]
            self.validate(state)
            if any(
                identity in models
                for kind, models in physics.models.items()
                if kind != "stochastic_hybrid"
            ):
                raise ValueError("adaptive process cannot ignore another physical model")
            for component in composition.components:
                if component.vessel != identity:
                    continue
                if component.kind not in {
                    "stochastic",
                    "thermal",
                    "material_input",
                    "observation",
                    "lab_schedule",
                }:
                    raise ValueError("adaptive component enables an unsupported coupling")
                if component.kind == "thermal" and set(component.parameters) - {"duration_range_s"}:
                    raise ValueError("adaptive kinetics do not include thermal evolution")
            if any(
                r.equipment_type not in {"batch_reactor", "instrument"}
                for r in cast(EquipmentLedger, state.equipment).equipment.values()
            ):
                raise ValueError("adaptive process has no continuous equipment coupling")
            for name in physical_instruments(composition, identity):
                instrument = composition.modules.instrument_specs()[name]
                if (
                    name != "sde_amount_meter"
                    or instrument.discrete_sampling
                    or instrument.contract.destructive
                    or instrument.contract.sample_consumption_L
                ):
                    raise ValueError(
                        "adaptive fixed-volume world needs its continuous nonconsuming gauge"
                    )

    def reconcile_initialization(
        self, transition: LifecycleTransition, *, previous: WorldState | None = None,
    ) -> LifecycleTransition:
        state = transition.state
        locals_ = {s.vessel_id: s for s in (state, *state.inactive_vessels.values())}
        records = {}
        for identity in set(transition.receipt.get("vessels", {})) & self.models.keys():
            local = locals_[identity]
            composition = cast("WorldCompositionSpec", self.composition)
            physics = cast("CompositionPhysics", composition.physics)
            if transition.receipt.get("scope") == "batch":
                memory = deepcopy(cast(
                    AdaptiveContinuation,
                    cast(ProcessLedger, physics.initial_states[identity].process)
                    .stochastic_continuation,
                ))
                policy = "declared_initial_adaptive_process"
            else:
                if previous is None:
                    raise ValueError("adaptive material initialization needs the previous state")
                old = {s.vessel_id: s for s in (previous, *previous.inactive_vessels.values())}[
                    identity
                ]
                memory = rebind_adaptive_continuation(
                    physics.network,
                    self.models[identity],
                    old.species_amounts,
                    local.species_amounts,
                    cast(Mapping[str, object],
                        cast(ProcessLedger, old.process).stochastic_continuation),
                    force_refresh=True,
                ).continuation
                policy = "retain_process_stream_and_rebind_original_channel_targets"
            locals_[identity] = local.replace(
                process=replace(cast(ProcessLedger, local.process),
                    stochastic_continuation=cast(dict[str, object], memory))
            )
            records[identity] = {"policy": policy, "time_s": memory["time_s"]}
        if records:
            active = locals_.pop(state.vessel_id)
            state = active.replace(inactive_vessels=locals_)
        for local in (state, *state.inactive_vessels.values()):
            if self.supports(local):
                self.validate(local)
        return replace(
            transition,
            state=state,
            receipt={
                **transition.receipt,
                **({"adaptive_initialization": records} if records else {}),
            },
        )

    def advance(
        self, state: WorldState, action: Mapping[str, Any], *, heat: bool = False,
        jacket_program: JacketTemperatureProgram | None = None,
    ) -> ModelProviderResult:
        if (
            heat
            or jacket_program is not None
            or action.get("operation", "wait") != "wait"
            or state.quenched
            or state.terminated
        ):
            return ModelProviderResult(
                {}, diagnostics={}, success=False, outcome="rejected",
                failure_reason="adaptive_fixed_boundary_only",
            )
        try:
            spec = self.validate(state, numerical=False)
            composition = cast("WorldCompositionSpec", self.composition)
            physics = cast("CompositionPhysics", composition.physics)
            result = advance_adaptive_hybrid(
                physics.network,
                spec,
                state.species_amounts,
                cast(Mapping[str, object],
                    cast(ProcessLedger, state.process).stochastic_continuation),
                action["duration_s"],
            )
            candidate = state.replace(
                species_amounts=result.amounts_mol,
                process=replace(
                    cast(ProcessLedger, state.process),
                    last_observation_record=None,
                    stochastic_continuation=cast(dict[str, object], result.continuation),
                ),
                ledger=advance_ledger(state.ledger, result.actual_duration_s),
            )
            self.validate(candidate, numerical=False)
        except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="adaptive_advance_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            diagnostics={
                "solver": ADAPTIVE_HYBRID_ALGORITHM,
                "stochastic_work": result.work,
                "partition_changes": [
                    {**change, "before": list(change["before"]), "after": list(change["after"])}
                    for change in result.partition_changes
                ],
                "energy_scope": "isothermal kinetics; heat balance not modelled",
            },
        )
