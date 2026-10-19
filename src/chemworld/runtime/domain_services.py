"""Operation composition services for ChemWorld runtime v2.

This module wires primitive helpers to focused transactional runtime services. It is
intentionally called by operation kernels rather than by ``ChemWorldEnv``
directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from chemworld.foundation import (
    OperationRecord,
    PhysicalConstitution,
    WorldState,
)
from chemworld.runtime.component_network import route_material, select_vessel
from chemworld.runtime.constitution_factory import make_chemworld_constitution
from chemworld.runtime.continuous_streams import ContinuousStreamServices
from chemworld.runtime.control_program import ControlProgramServices
from chemworld.runtime.crystallization_services import ChemWorldCrystallizationServices
from chemworld.runtime.distillation_services import ChemWorldDistillationServices
from chemworld.runtime.domain_service_registry import (
    DomainServiceContract,
    DomainServiceRegistry,
)
from chemworld.runtime.electrochemical_services import ChemWorldElectrochemicalServices
from chemworld.runtime.flow_services import ChemWorldFlowServices
from chemworld.runtime.instrument_cost_services import ChemWorldInstrumentCostServices
from chemworld.runtime.material_routing import ChemWorldMaterialRoutingServices
from chemworld.runtime.mechanisms import CompiledMechanism
from chemworld.runtime.phase_separation_services import ChemWorldPhaseSeparationServices
from chemworld.runtime.primitive_services import ChemWorldPrimitiveOperationServices
from chemworld.runtime.reaction_thermal_services import ChemWorldReactionThermalServices
from chemworld.runtime.record_services import ChemWorldOperationRecorder
from chemworld.runtime.solvent_transport import finish_solvent_transition, initialize_solvents
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.runtime.spectral_settings import configure_instrument
from chemworld.world.control_contract import CONTROL_OPERATIONS
from chemworld.world.operations import operation_name
from chemworld.world.parameters import ChemWorldParameters
from chemworld.world.stream_contract import STREAM_OPERATIONS


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
    ) -> None:
        self.world = world
        self.constitution = constitution
        self.service_registry = service_registry or DomainServiceRegistry.default()
        self.service_registry.validate_contract_integrity()
        self.species_view = MechanismSpeciesView(compiled_mechanism)
        self.operation_recorder = ChemWorldOperationRecorder(constitution)
        self.primitive = ChemWorldPrimitiveOperationServices(world, self.species_view)
        self.material_routing = ChemWorldMaterialRoutingServices()
        self.reaction_thermal = ChemWorldReactionThermalServices(world, self.species_view)
        self.control_program = ControlProgramServices(self.reaction_thermal)
        self.continuous_streams = ContinuousStreamServices(world, self.species_view)
        self.phase_separation = ChemWorldPhaseSeparationServices(
            world,
            self.species_view,
            nominal_pair_contract=partition_nominal_pair_contract,
        )
        self.crystallization = ChemWorldCrystallizationServices(world, self.species_view)
        self.distillation = ChemWorldDistillationServices(world, self.species_view)
        self.flow = ChemWorldFlowServices(self.species_view, self.reaction_thermal)
        self.electrochemical = ChemWorldElectrochemicalServices(world, self.species_view)
        self.instrument_cost = ChemWorldInstrumentCostServices(constitution)

    def apply_operation(
        self,
        state: WorldState,
        action: dict[str, Any],
    ) -> tuple[WorldState, OperationRecord]:
        operation = operation_name(action["operation"])
        state = initialize_solvents(state)
        before = state
        preconditions = self.constitution.check_preconditions(operation, state, action)
        if not all(preconditions.values()):
            next_state = self.primitive.penalize_invalid(state)
            return next_state, self.operation_recorder.record(
                operation,
                before,
                next_state,
                preconditions,
                action,
            )

        try:
            next_state = self.operation_methods()[operation](state, action)
        except KeyError as exc:
            raise ValueError(f"Unsupported operation: {operation}") from exc

        next_state = finish_solvent_transition(state, next_state, action)
        next_state = self.reaction_thermal.with_risk_and_pressure(next_state)
        if state.vessel_elapsed_s:
            elapsed = dict(state.vessel_elapsed_s)
            elapsed[state.vessel_id] += next_state.ledger.time_s - state.ledger.time_s
            assert next_state.thermal is not None
            thermal = next_state.thermal.vessels[next_state.vessel_id]
            if operation != "select_vessel":
                thermal = replace(
                    thermal,
                    **{
                        key: (
                            getattr(thermal, key)
                            + getattr(next_state.ledger, key)
                            - getattr(state.ledger, key)
                        )
                        for key in ("energy_jacket_J", "heat_reaction_J", "heat_loss_J")
                    },
                )
            next_state = next_state.replace(
                vessel_elapsed_s=elapsed,
                thermal=replace(next_state.thermal, vessels={next_state.vessel_id: thermal}),
            )
        if operation == "route_material":
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
        return next_state, self.operation_recorder.record(
            operation,
            before,
            next_state,
            preconditions,
            action,
        )

    def operation_methods(
        self,
    ) -> dict[str, Callable[[WorldState, dict[str, Any]], WorldState]]:
        return {
            **dict.fromkeys(CONTROL_OPERATIONS, self.control_program.apply),
            **dict.fromkeys(STREAM_OPERATIONS, self.continuous_streams.apply),
            "select_vessel": select_vessel,
            "route_material": route_material,
            "create_container": self.material_routing.create_container,
            "transfer_material": self.material_routing.transfer_material,
            "add_reagent": self.primitive.add_reagent,
            "add_component": self.primitive.add_component,
            "add_solvent": self.primitive.add_solvent,
            "add_catalyst": self.primitive.add_catalyst,
            "heat": lambda state, action: self.reaction_thermal.integrate(state, action, heat=True),
            "wait": lambda state, action: self.reaction_thermal.integrate(
                state, action, heat=False
            ),
            "sample": self.primitive.sample,
            "quench": lambda state, _action: self.primitive.quench(state),
            "add_phase": self.phase_separation.add_phase,
            "add_extractant": self.phase_separation.add_extractant,
            "mix": self.phase_separation.mix_phases,
            "settle": self.phase_separation.settle_phases,
            "separate_phase": self.phase_separation.separate_phase,
            "wash": self.phase_separation.wash_phase,
            "dry": lambda state, _action: self.phase_separation.dry_phase(state),
            "concentrate": self.phase_separation.concentrate_phase,
            "transfer": self.phase_separation.transfer_phase,
            "seed_crystals": self.crystallization.seed_crystals,
            "cool_crystallize": self.crystallization.cool_crystallize,
            "filter_crystals": lambda state, _action: self.crystallization.filter_crystals(state),
            "resuspend_crystals": self.crystallization.resuspend_crystals,
            "evaporate": self.primitive.evaporate,
            "distill": self.distillation.distill,
            "collect_fraction": self.distillation.collect_fraction,
            "set_flow_rate": self.flow.set_flow_rate,
            "run_flow": self.flow.run_flow,
            "set_potential": self.electrochemical.set_potential,
            "electrolyze": self.electrochemical.electrolyze,
            "terminate": lambda state, _action: state.replace(terminated=True),
            "measure": self.instrument_cost.apply_measurement_cost,
            "configure_instrument": configure_instrument,
        }

    def service_id_for_operation(self, operation: str) -> str:
        return self.service_registry.service_id_for_operation(operation)

    def to_dict(self) -> dict[str, Any]:
        return self.service_registry.to_dict()

    def validate_profile(self, profile: Any) -> None:
        self.service_registry.validate_profile(profile)

    def penalize_invalid(self, state: WorldState) -> WorldState:
        return self.primitive.penalize_invalid(state)

    def record_operation(
        self,
        operation: str,
        before: WorldState,
        after: WorldState,
        preconditions: dict[str, bool],
        action: dict[str, Any],
    ) -> OperationRecord:
        return self.operation_recorder.record(operation, before, after, preconditions, action)


__all__ = [
    "ChemWorldDomainServices",
    "DomainServiceContract",
    "DomainServiceRegistry",
    "make_chemworld_constitution",
]
