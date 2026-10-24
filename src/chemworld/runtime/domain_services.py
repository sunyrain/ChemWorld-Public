"""Operation composition services for ChemWorld runtime v2.

This module wires primitive helpers to focused transactional runtime services. It is
intentionally called by operation kernels rather than by ``ChemWorldEnv``
directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import fsum, isclose
from typing import Any

from chemworld.foundation import (
    OperationRecord,
    PhysicalConstitution,
    WorldState,
)
from chemworld.foundation.balances import TransitionBalance
from chemworld.runtime.constitution_factory import make_chemworld_constitution
from chemworld.runtime.domain_service_registry import (
    DomainServiceContract,
    DomainServiceRegistry,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.mechanisms import CompiledMechanism
from chemworld.runtime.record_services import ChemWorldOperationRecorder
from chemworld.runtime.solvent_transport import finish_solvent_transition, initialize_solvents
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.domain_modules import DomainModuleRegistry, ServiceContext
from chemworld.world.operations import operation_name
from chemworld.world.parameters import ChemWorldParameters


class ChemWorldDomainServices:
    """Runtime domain services called by operation kernels.

    Operation dispatch is table-driven so adding a new operation does not
    require editing a central ``if/elif`` block.
    """

    def __init__(
        self,
        world: ChemWorldParameters,
        constitution: PhysicalConstitution,
        compiled_mechanism: CompiledMechanism,
        service_registry: DomainServiceRegistry | None = None,
        partition_nominal_pair_contract: str | None = None,
        modules: DomainModuleRegistry | None = None,
        composition=None,
    ) -> None:
        self.world = world
        self.constitution = constitution
        self.modules = (
            modules
            or (service_registry.modules if service_registry else None)
            or DomainModuleRegistry.default()
        )
        self.service_registry = service_registry or DomainServiceRegistry.from_modules(self.modules)
        self.service_registry.validate_contract_integrity()
        self.species_view = MechanismSpeciesView(compiled_mechanism)
        self.operation_recorder = ChemWorldOperationRecorder(constitution, modules=self.modules)
        context = ServiceContext(
            world,
            constitution,
            self.species_view,
            partition_nominal_pair_contract,
            modules=self.modules,
            composition=composition,
        )
        for factory in self.modules.service_order():
            if hasattr(self, factory.attribute):
                raise ValueError(f"Service attribute already owned: {factory.attribute}")
            service = factory.factory(context)
            context.services[factory.service_id] = service
            setattr(self, factory.attribute, service)
        installed = self.service_registry.operation_map()
        if hasattr(self, "lab_schedule"):
            self.lab_schedule._domains = self
            # Late binding avoids a second scheduler or a circular service
            # construction dependency for timed in-situ formal observations.
            self.instrument_cost.shared_clock_advance = self.lab_schedule.advance
        self._operation_methods = {
            name: spec.bind(context.services)
            for name, spec in self.modules.operation_specs().items()
            if name in installed
        }

    def apply_operation(
        self,
        state: WorldState,
        action: dict[str, Any],
    ) -> tuple[WorldState, OperationRecord]:
        """Compatibility view; kernels consume the provider result directly."""
        result = self.evaluate_operation(state, action)
        if result.outcome in {"rejected", "numerical_failure"}:
            raise ValueError(f"operation has no committed candidate: {result.failure_reason}")
        next_state = state if result.state is None else result.state
        return next_state, self.operation_recorder.record(
            operation_name(action["operation"], modules=self.modules),
            state,
            next_state,
            dict(result.outputs["preconditions"]),
            action,
            constitution_checks=(),
        )

    def evaluate_operation(
        self,
        state: WorldState,
        action: dict[str, Any],
    ) -> ModelProviderResult:
        operation = operation_name(action["operation"], modules=self.modules)
        state = initialize_solvents(state)
        before = state
        preconditions = self.constitution.check_preconditions(operation, state, action)
        preconditions.update(
            self.modules.operation_specs()[operation].check_preconditions(
                state, self.constitution, action
            )
        )
        if not all(preconditions.values()):
            return ModelProviderResult(
                outputs={"preconditions": preconditions},
                diagnostics={},
                state=state,
                outcome="rejected",
                success=False,
                failure_reason="precondition_failed",
            )

        method = self.operation_methods().get(operation)
        if method is None:
            raise ValueError(f"Unsupported operation: {operation}")
        if state.lab_clock is not None:
            from chemworld.runtime.scheduling import shared_operation_allowed

            if not shared_operation_allowed(operation, state, self.modules, action):
                return ModelProviderResult(
                    {"preconditions": preconditions}, diagnostics={}, state=None, success=False,
                    outcome="rejected", failure_reason="unsupported_shared_clock_action",
                )
            evaluated = (
                self.lab_schedule.advance(state, action)
                if operation == "wait" else method(state, action)
            )
        else:
            evaluated = method(state, action)
        if isinstance(evaluated, ModelProviderResult):
            result = evaluated
            if result.outcome in {"rejected", "numerical_failure"}:
                return replace(result, outputs={**result.outputs, "preconditions": preconditions})
            if result.state is None:
                return ModelProviderResult(
                    outputs={"preconditions": preconditions},
                    diagnostics={"candidate_state_missing": True},
                    success=False,
                    outcome="numerical_failure",
                    failure_reason="provider_candidate_missing",
                )
            next_state = result.state
        else:
            next_state = evaluated
            result = ModelProviderResult(
                outputs={},
                diagnostics={},
                state=next_state,
                actual_duration_s=next_state.ledger.time_s - before.ledger.time_s,
            )

        shared_acquisition = (
            operation == "measure"
            and result.diagnostics.get("shared_acquisition_clock") == "lab_schedule"
        )
        if (state.lab_clock is not None and operation != "wait" and not shared_acquisition
                and next_state.ledger.time_s != state.ledger.time_s):
            return ModelProviderResult(
                {"preconditions": preconditions}, diagnostics={}, state=None,
                success=False, outcome="rejected",
                failure_reason="shared_event_must_not_advance_a_second_clock",
            )
        if not (state.lab_clock is not None and operation == "wait"):
            next_state = finish_solvent_transition(state, next_state, action)
        context_thermal = getattr(self, "reaction_thermal", None)
        if context_thermal is not None:
            next_state = context_thermal.with_risk_and_pressure(next_state)
        if state.vessel_elapsed_s and state.lab_clock is None:
            elapsed = dict(next_state.vessel_elapsed_s)
            elapsed[state.vessel_id] = (
                state.vessel_elapsed_s[state.vessel_id]
                + next_state.ledger.time_s - state.ledger.time_s
            )
            thermal_fields = ("energy_jacket_J", "heat_reaction_J", "heat_loss_J")
            thermal_deltas = result.vessel_thermal_deltas
            if operation != "select_vessel" and thermal_deltas is not None:
                previous = {state.vessel_id: state, **state.inactive_vessels}
                members = {next_state.vessel_id: next_state, **next_state.inactive_vessels}
                if (
                    set(thermal_deltas) - (previous.keys() & members.keys())
                    or any(not isclose(
                        fsum(values.get(key, 0.0) for values in thermal_deltas.values()),
                        getattr(next_state.ledger, key) - getattr(state.ledger, key),
                        abs_tol=1e-7, rel_tol=1e-8,
                    ) for key in thermal_fields)
                ):
                    return ModelProviderResult(
                        {"preconditions": preconditions}, diagnostics={}, success=False,
                        outcome="numerical_failure", failure_reason="local_system_heat_mismatch",
                    )
                for name, increments in thermal_deltas.items():
                    local = members[name]
                    record = local.thermal.vessels[name]
                    members[name] = local.replace(thermal=replace(
                        local.thermal, vessels={name: replace(record, **{
                            key: getattr(previous[name].thermal.vessels[name], key)
                            + increments.get(key, 0.0) for key in thermal_fields
                        })},
                    ))
                active = members.pop(next_state.vessel_id)
                next_state = active.replace(inactive_vessels=members)
            thermal = next_state.thermal.vessels[next_state.vessel_id]
            if operation != "select_vessel" and thermal_deltas is None:
                thermal = replace(
                    thermal,
                    **{
                        key: (
                            getattr(thermal, key)
                            + getattr(next_state.ledger, key)
                            - getattr(state.ledger, key)
                        )
                        for key in thermal_fields
                    },
                )
            next_state = next_state.replace(
                vessel_elapsed_s=elapsed,
                thermal=replace(next_state.thermal, vessels={next_state.vessel_id: thermal}),
            )
        if operation == "route_material" and context_thermal is not None:
            edge = state.metadata["component_network"]["connections"][action["connection"]]
            target = edge["target_vessel"]
            next_state = next_state.replace(
                inactive_vessels={
                    **next_state.inactive_vessels,
                    target: self.reaction_thermal.with_risk_and_pressure(
                        next_state.inactive_vessels[target]
                    ),
                }
            )
        return replace(
            result,
            state=next_state,
            outputs={**result.outputs, "preconditions": preconditions},
        )

    def operation_methods(
        self,
    ) -> dict[str, Callable[[WorldState, dict[str, Any]], WorldState | ModelProviderResult]]:
        return dict(self._operation_methods)

    def service_id_for_operation(self, operation: str) -> str:
        return self.service_registry.service_id_for_operation(operation)

    def to_dict(self) -> dict[str, Any]:
        return self.service_registry.to_dict()

    def validate_profile(self, profile: Any) -> None:
        self.service_registry.validate_profile(profile)

    def penalize_invalid(self, state: WorldState) -> WorldState:
        from chemworld.runtime.primitive_services import ChemWorldPrimitiveOperationServices

        return ChemWorldPrimitiveOperationServices.penalize_invalid(state)

    def record_operation(
        self,
        operation: str,
        before: WorldState,
        after: WorldState,
        preconditions: dict[str, bool],
        action: dict[str, Any],
        *,
        constitution_checks: tuple[dict[str, object], ...] | None = None,
        material_balance: TransitionBalance | None = None,
    ) -> OperationRecord:
        return self.operation_recorder.record(
            operation,
            before,
            after,
            preconditions,
            action,
            constitution_checks=constitution_checks,
            material_balance=material_balance,
        )


__all__ = [
    "ChemWorldDomainServices",
    "DomainServiceContract",
    "DomainServiceRegistry",
    "make_chemworld_constitution",
]
