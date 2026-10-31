"""Continuous SDE delegate of the original stochastic advance coordinator."""

from copy import deepcopy
from dataclasses import replace
from math import ulp

from chemworld.physchem.stochastic_sde import (
    advance_sde,
    rebind_sde_continuation,
    sde_model,
    validate_sde_continuation,
)
from chemworld.physchem.stochastic_sde_specs import SDE_ALGORITHM
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult


class SDEStochasticRuntime:
    def __init__(self, composition):
        self.composition = composition

    @property
    def models(self):
        return (
            {}
            if self.composition is None or self.composition.physics is None
            else self.composition.physics.models_of("stochastic_sde")
        )

    def supports(self, state):
        continuation = state.process.stochastic_continuation
        return state.vessel_id in self.models or (
            continuation is not None and continuation.get("algorithm") == SDE_ALGORITHM
        )

    def validate(self, state, *, numerical=True):
        spec = self.models.get(state.vessel_id)
        if spec is None:
            raise ValueError("SDE process requires its actual physical model declaration")
        if state.temperature_K != spec.temperature_K:
            raise ValueError("SDE state changed its declared fixed temperature or volume")
        if (
            len(state.phases.phases) != 1
            or state.phase != "liquid"
            or next(iter(state.phases.phases.values())).species_amounts_mol != state.species_amounts
            or state.state_slots.slots
            or state.samples.samples
        ):
            raise ValueError("SDE requires one authoritative well-mixed phase")
        continuation = state.process.stochastic_continuation
        if continuation is None or continuation.get("time_s") != state.ledger.time_s:
            raise ValueError("SDE process and original world clocks disagree")
        if continuation.get("volume_L") != state.volume_L:
            raise ValueError("SDE numerical binding differs from the original world volume")
        validate_sde_continuation(
            self.composition.physics.network,
            spec,
            state.species_amounts,
            continuation,
            verify_endpoint=numerical,
        )
        return spec

    def validate_world(self) -> None:
        if not self.models:
            return
        physics = self.composition.physics
        if (
            physics.network.stochastic is not None
            or not set(self.models) <= physics.initial_states.keys()
        ):
            raise ValueError("SDE cannot compete with an integer SSA declaration")
        from chemworld.world.composition import physical_instruments

        for vessel in self.models:
            state = physics.initial_states[vessel]
            spec = self.validate(state)
            if any(
                vessel in models
                for kind, models in physics.models.items()
                if kind != "stochastic_sde"
            ):
                raise ValueError("SDE cannot silently ignore another physical model")
            if any(
                c.vessel == vessel
                and c.kind
                not in {
                    "stochastic",
                    "observation",
                    "material_input",
                    "lab_schedule",
                    "thermal",
                }
                for c in self.composition.components
            ):
                raise ValueError("SDE component enables an unsupported physical coupling")
            if any(
                c.vessel == vessel
                and c.kind == "thermal"
                and set(c.parameters) - {"duration_range_s"}
                for c in self.composition.components
            ):
                raise ValueError(
                    "SDE uses the common thermal component only to declare wait bounds"
                )
            if any(
                r.equipment_type not in {"batch_reactor", "instrument"}
                for r in state.equipment.equipment.values()
            ):
                raise ValueError("SDE has no declared continuous equipment coupling")
            for name in physical_instruments(self.composition, vessel):
                instrument = self.composition.modules.instrument_specs()[name]
                from chemworld.world.stochastic_sde_instruments import SDE_ALIQUOT_METER

                if name == SDE_ALIQUOT_METER:
                    if (
                        spec.minimum_volume_L is None
                        or state.volume_L - instrument.contract.sample_consumption_L
                        < spec.minimum_volume_L
                    ):
                        raise ValueError("SDE aliquot requires an explicit feasible volume floor")
                    if instrument.contract.sample_consumption_L < 8 * ulp(state.volume_L):
                        raise ValueError("SDE aliquot is below the original geometry resolution")
                    continue
                if (
                    name in {"molecule_counter", "molecule_capture"}
                    or instrument.discrete_sampling
                    or instrument.contract.destructive
                    or instrument.contract.sample_consumption_L
                ):
                    raise ValueError("SDE needs continuous readout and an explicitly bound sampler")

    def reconcile_material_change(self, before, after):
        spec = self.validate(before, numerical=False)
        if after.volume_L >= before.volume_L:
            raise ValueError("SDE sampling must withdraw a representable positive volume")
        memory = rebind_sde_continuation(
            self.composition.physics.network,
            spec,
            before.species_amounts,
            after.species_amounts,
            before.process.stochastic_continuation,
            volume_L=after.volume_L,
        )
        candidate = after.replace(process=replace(after.process, stochastic_continuation=memory))
        self.validate(candidate, numerical=False)
        return candidate

    def apply_parameters(self, state, action):
        if state.terminated or state.quenched:
            return ModelProviderResult(
                {}, diagnostics={}, success=False, outcome="rejected",
                failure_reason="sde_control_stopped",
            )
        try:
            spec = self.validate(state, numerical=False)
            if not spec.controls or not action.get("values"):
                raise ValueError("SDE control action requires declared nonempty inputs")
            memory = rebind_sde_continuation(
                self.composition.physics.network,
                spec,
                state.species_amounts,
                state.species_amounts,
                state.process.stochastic_continuation,
                control_values=action["values"],
            )
            candidate = state.replace(
                process=replace(
                    state.process, last_observation_record=None, stochastic_continuation=memory
                )
            )
            self.validate(candidate, numerical=False)
        except (ValueError, KeyError, TypeError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_sde_parameter_control",
                diagnostics={"control_error": str(exc)},
            )
        model = sde_model(self.composition.physics.network, spec, memory)
        return ModelProviderResult(
            {},
            state=candidate,
            diagnostics={
                "controlled_inputs": {
                    key: {"value": memory["control_values"][key], **model.control_metadata[key]}
                    for key in action["values"]
                },
                "numerical_lookahead_invalidated": (
                    memory["invalidations"]
                    != state.process.stochastic_continuation["invalidations"]
                ),
            },
        )

    def reconcile_initialization(self, transition, *, previous=None):
        affected = set(transition.receipt.get("vessels", {}))
        scope = transition.receipt.get("scope")
        state = transition.state
        locals_ = {s.vessel_id: s for s in (state, *state.inactive_vessels.values())}
        records = {}
        for identity in affected & self.models.keys():
            local = locals_[identity]
            if scope == "batch":
                memory = deepcopy(
                    self.composition.physics.initial_states[
                        identity
                    ].process.stochastic_continuation
                )
                policy = "declared_initial_process_seed_and_controls"
            else:
                if previous is None:
                    raise ValueError("SDE material initialization requires the previous real state")
                old = {s.vessel_id: s for s in (previous, *previous.inactive_vessels.values())}[
                    identity
                ]
                memory = rebind_sde_continuation(
                    self.composition.physics.network,
                    self.models[identity],
                    old.species_amounts,
                    local.species_amounts,
                    old.process.stochastic_continuation,
                    volume_L=local.volume_L,
                    force_invalidate=True,
                )
                policy = "retain_process_stream_and_controls_invalidate_numerical_lookahead"
            locals_[identity] = local.replace(
                process=replace(local.process, stochastic_continuation=memory)
            )
            records[identity] = {
                "policy": policy,
                "time_s": memory["time_s"],
                "invalidations": memory["invalidations"],
            }
        if records:
            active = locals_.pop(state.vessel_id)
            state = active.replace(inactive_vessels=locals_)
        for local in (state, *state.inactive_vessels.values()):
            if self.supports(local):
                self.validate(local)
        return replace(
            transition,
            state=state,
            receipt={**transition.receipt, **({"sde_initialization": records} if records else {})},
        )

    def advance(self, state, action, *, heat=False, jacket_program=None):
        if (
            heat
            or jacket_program is not None
            or action.get("operation", "wait") != "wait"
            or state.quenched
            or state.terminated
        ):
            return ModelProviderResult(
                {}, diagnostics={}, success=False, outcome="rejected",
                failure_reason="sde_fixed_boundary_only",
            )
        try:
            spec = self.validate(state, numerical=False)
        except (ValueError, TypeError, KeyError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="sde_contract_violation",
                diagnostics={"reason": str(exc)},
            )
        try:
            result = advance_sde(
                self.composition.physics.network,
                spec,
                state.species_amounts,
                state.process.stochastic_continuation,
                action["duration_s"],
            )
            candidate = state.replace(
                species_amounts=result.amounts_mol,
                process=replace(
                    state.process,
                    last_observation_record=None,
                    stochastic_continuation=result.continuation,
                ),
                ledger=advance_ledger(state.ledger, result.actual_duration_s),
            )
            self.validate(candidate, numerical=False)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="sde_advance_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            diagnostics={
                "solver": SDE_ALGORITHM,
                "stochastic_work": result.diagnostics,
                "energy_scope": "isothermal kinetics; heat balance not modelled",
            },
        )
