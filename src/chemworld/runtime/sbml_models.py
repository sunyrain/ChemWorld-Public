"""Formal-model adapter of the existing native coordinator and transaction.

The installed EquipmentRecord owns all live formal coordinates and queue data.
The adapter has configuration references only; the world ledger owns time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ulp
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState, upsert_equipment_record
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.coordinate_events import CoordinateEventError, _check_memory, _finite_real
from chemworld.physchem.coordinate_hybrid import (
    CoordinateHybridMemory,
    CoordinateHybridResult,
    advance_coordinate_events,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.sbml_system import SBML_ENTITY_TYPE, SBMLSystemSpec
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

if TYPE_CHECKING:
    from chemworld.physchem.sbml_integer_events import IntegerEventAdvance
    from chemworld.physchem.stochastic_kinetics import StochasticAdvance, StochasticLeap
    from chemworld.world.composition import WorldCompositionSpec

STATE_FORMAT = "chemworld_formal_model_state_v1"


@dataclass(frozen=True)
class _FormalCoordinates:
    coordinates: dict[str, float]


@dataclass(frozen=True)
class _IntegerCandidate:
    final_state: _FormalCoordinates
    memory: dict[str, Any]
    actual_duration_s: float
    events: tuple[tuple[float, str], ...]
    segment_diagnostics: tuple[StochasticLeap, ...]
    work: dict[str, int]
    source_events_executed: int = 0


def model_record(state: WorldState) -> EquipmentRecord | None:
    records = [r for r in cast(EquipmentLedger, state.equipment).equipment.values()
               if r.equipment_type == SBML_ENTITY_TYPE and r.attached_vessel_id == state.vessel_id]
    if len(records) > 1:
        raise ValueError("one native entity must own each formal model state")
    return records[0] if records else None


def _view(
    state: WorldState, model: SBMLSystemSpec, coordinates: Mapping[str, float],
) -> CoupledState:
    _, volumes = model.frame.quantities(coordinates, state.ledger.time_s)
    # Required legacy coupled geometry is a transient reference only. Formal
    # equations use their explicitly owned V coordinates, not this field.
    return CoupledState({}, state.temperature_K, next(iter(volumes.values())),
                        time_s=state.ledger.time_s, coordinates=coordinates)


def _install(
    state: WorldState, identity: str, coordinates: Mapping[str, float],
    continuation: dict[str, Any] | CoordinateHybridMemory,
) -> WorldState:
    return state.replace(equipment=upsert_equipment_record(state.equipment,
        equipment_id=identity, equipment_type=SBML_ENTITY_TYPE, attached_vessel_id=state.vessel_id,
        settings={"format": STATE_FORMAT, "model_binding": state.vessel_id,
                  "coordinates": dict(coordinates), "continuation": (
                      continuation if isinstance(continuation, dict) else continuation.to_dict())},
        batch_reset="preserve", private_settings=("coordinates", "continuation")))


def _advance_candidate(
    state: WorldState, model: SBMLSystemSpec, coordinates: Mapping[str, float],
    memory: dict[str, Any] | CoordinateHybridMemory, duration: float,
) -> _IntegerCandidate | CoordinateHybridResult:
    result: IntegerEventAdvance | StochasticAdvance
    if model.integer_kinetics is not None:
        if model.integer_kinetics.requires_boundaries:
            from chemworld.physchem.sbml_integer_events import advance_integer_events

            result = advance_integer_events(model, _view(state, model, coordinates),
                                            cast(dict[str, Any], memory), duration)
            return _IntegerCandidate(_FormalCoordinates(result.coordinates), result.memory,
                result.actual_duration_s, result.events, result.batches, result.work,
                len(result.source_events))
        values, continuation, result = model.integer_kinetics.advance(
            coordinates, cast(dict[str, Any], memory),
            state.ledger.time_s, duration, model.solver_method,
            model.maximum_events, model.tau_policy)
        return _IntegerCandidate(_FormalCoordinates(values), continuation,
            result.actual_duration_s, result.events, result.batches, result.work)
    return advance_coordinate_events(model.continuous_system(), _view(state, model, coordinates),
        duration, cast(CoordinateHybridMemory, memory), model.events, roots=model.roots.roots,
        trigger_values=model.roots.trigger_values, max_events=model.maximum_events,
        max_segments=model.maximum_segments, max_pending=model.maximum_pending_events)


def install_initial_model(
    state: WorldState, model: SBMLSystemSpec, *, equipment_id: str = "formal_model", seed: int = 0,
) -> WorldState:
    if not isinstance(model, SBMLSystemSpec) or state.phase != "formal" or state.ledger.time_s != 0:
        raise ValueError("formal initialization needs a model and empty initial workspace")
    memory: dict[str, Any] | CoordinateHybridMemory
    if model.integer_kinetics is not None:
        from chemworld.physchem.sbml_stochastic import initial_memory

        if model.integer_kinetics.requires_boundaries:
            from chemworld.physchem.sbml_integer_events import initial_boundary_memory

            memory = initial_boundary_memory(model, seed)
        else:
            memory = initial_memory(seed, model.solver_method)
    else:
        memory = CoordinateHybridMemory.initialize(model.events)
    result = _advance_candidate(state, model, model.frame.initial_values, memory, 0.)
    return _install(state, equipment_id, result.final_state.coordinates, result.memory)


def validate_model_state(
    state: WorldState, model: SBMLSystemSpec,
) -> tuple[EquipmentRecord, dict[str, float], dict[str, Any] | CoordinateHybridMemory]:
    record = model_record(state)
    if record is None or not isinstance(model, SBMLSystemSpec) or state.phase != "formal":
        raise ValueError("formal model binding and its unique native state owner must agree")
    payload = record.settings
    if (set(record.private_settings) != {"coordinates", "continuation"}
            or set(payload) != {"format", "model_binding", "coordinates", "continuation"}
            or payload["format"] != STATE_FORMAT or payload["model_binding"] != state.vessel_id
            or not isinstance(payload["coordinates"], dict)
            or any(not _finite_real(v) for v in payload["coordinates"].values())):
        raise ValueError("unknown or invalid formal model state fields")
    coordinates = cast(dict[str, float], payload["coordinates"])
    memory: dict[str, Any] | CoordinateHybridMemory
    model.frame.validate_values(coordinates, state.ledger.time_s)
    if model.integer_kinetics is not None:
        from chemworld.physchem.stochastic_kinetics import _signature

        integer = model.integer_kinetics
        # A restored pending event must belong to exactly this source, geometry,
        # inputs and inventory. Only an explicit control may rebind a candidate.
        memory = payload["continuation"]
        expected = _signature(integer.counts(coordinates, state.ledger.time_s),
            integer.compile(coordinates, state.ledger.time_s),
            integer.policy(coordinates, model.solver_method,
                           model.maximum_events, model.tau_policy))
        if (not isinstance(memory, dict) or not isinstance(memory.get("process"), dict)
                or memory["process"].get("signature") != expected):
            raise ValueError("formal integer continuation signature disagrees with source/state")
        try:
            settled = _advance_candidate(state, model, coordinates, memory, 0.)
        except CoordinateEventError as exc:
            raise ValueError("invalid integer source-event snapshot continuation") from exc
        if model.integer_kinetics.requires_boundaries and (
                settled.final_state.coordinates != coordinates or settled.memory != memory):
            raise ValueError("integer source-event snapshot must have a settled original queue")
        return record, coordinates, memory
    memory = CoordinateHybridMemory.from_dict(payload["continuation"])
    _check_memory(memory.events, {event.event_id: event for event in model.events})
    if (set(memory.root_values)-{r.root_id for r in model.roots.roots}
            or len(memory.events.pending) > model.maximum_pending_events):
        raise ValueError("formal continuation differs from declared roots or pending budget")
    for pending in memory.events.pending:
        tolerance = 8*max(ulp(state.ledger.time_s), ulp(pending.due_time_s))
        if pending.due_time_s < state.ledger.time_s-tolerance:
            raise ValueError("formal snapshot contains an overdue unexecuted event")
    view = _view(state, model, coordinates)
    for root in model.roots.roots:
        if (root.root_id in memory.root_values
                and root.value(view) != memory.root_values[root.root_id]):
            raise ValueError("formal root continuation does not describe the saved candidate")
    return record, coordinates, memory


class SBMLAdvanceStrategy:
    supported_operations = ("wait",)

    def __init__(self, composition: WorldCompositionSpec | None = None) -> None:
        self.composition = composition

    def supports(self, state: WorldState) -> bool:
        return model_record(state) is not None

    def model(self, state: WorldState) -> SBMLSystemSpec:
        if self.composition is None or self.composition.physics is None:
            raise ValueError("formal model requires its native composition declaration")
        model = self.composition.physics.models_of("sbml").get(state.vessel_id)
        if not isinstance(model, SBMLSystemSpec):
            raise ValueError("formal native entity has no matching model declaration")
        return model

    def validate_state(self, state: WorldState) -> None:
        validate_model_state(state, self.model(state))

    def apply_parameters(self, state: WorldState, action: Mapping[str, Any]) -> ModelProviderResult:
        if state.terminated:
            return ModelProviderResult({}, {}, success=False, outcome="rejected",
                failure_reason="formal_model_terminated")
        try:
            model = self.model(state)
            record, coordinates, memory = validate_model_state(state, model)
            coordinates = model.frame.controlled_values(coordinates, state.ledger.time_s,
                                                         action["values"])
        except (ValueError, KeyError, ArithmeticError) as exc:
            return ModelProviderResult({}, {"control_error": str(exc)},
                success=False, outcome="rejected",
                failure_reason="invalid_formal_parameter_control")
        try:
            result = _advance_candidate(state, model, coordinates, memory, 0.)
            candidate = _install(state, record.equipment_id,
                result.final_state.coordinates, result.memory)
            validate_model_state(candidate, model)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult({}, {"solver_error": str(exc)}, success=False,
                outcome="numerical_failure",
                failure_reason="formal_control_event_settlement_failed")
        diagnostics: dict[str, Any] = {"events_executed": len(result.events),
                       "parameter_basis": "declared_source_units"}
        if model.integer_kinetics is not None and model.integer_kinetics.requires_boundaries:
            diagnostics["source_events_executed"] = cast(
                _IntegerCandidate, result).source_events_executed
        return ModelProviderResult({}, diagnostics, state=candidate, actual_duration_s=0.)

    def validate_world(self) -> None:
        if self.composition is None or self.composition.physics is None:
            return
        physics = self.composition.physics
        declared = {c.vessel for c in self.composition.components if c.kind == "sbml_model"}
        installed = {v for v, s in physics.initial_states.items() if self.supports(s)}
        if declared != installed or installed != set(physics.models_of("sbml")):
            raise ValueError("formal components, model declarations and native state owners differ")
        if installed and physics.lab_clock is None:
            raise ValueError("formal workspaces require the existing shared laboratory clock")
        for vessel in installed:
            state = physics.initial_states[vessel]
            validate_model_state(state, self.model(state))
            allowed = {"sbml_model", "observation", "lab_schedule"}
            if any(c.vessel == vessel and c.kind not in allowed
                   for c in self.composition.components):
                raise ValueError("formal workspace cannot enable physical material/thermal acts")
            if any(vessel in bindings for key, bindings in physics.models.items() if key != "sbml"):
                raise ValueError("formal entity cannot also bind a competing physical model")

    def advance(
        self, state: WorldState, action: Mapping[str, Any], *,
        heat: bool = False, jacket_program: object = None,
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or state.quenched or state.terminated:
            return ModelProviderResult({}, {}, success=False, outcome="rejected",
                failure_reason="formal_model_boundary_only")
        try:
            model = self.model(state)
            record, coordinates, memory = validate_model_state(state, model)
            result = _advance_candidate(state, model, coordinates, memory,
                                        float(action["duration_s"]))
            candidate = _install(state, record.equipment_id,
                                  result.final_state.coordinates, result.memory)
            candidate = candidate.replace(ledger=advance_ledger(state.ledger,
                                                                result.actual_duration_s))
            validate_model_state(candidate, model)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult({}, {"solver_error": str(exc),
                "attempted_events": (len(exc.executed_prefix)
                    if isinstance(exc, CoordinateEventError) else 0)},
                success=False, outcome="numerical_failure",
                failure_reason="formal_model_advance_failed")
        diagnostics: dict[str, Any] = {"solver": "shared_coupled_formal_kinetics",
            "events_executed": len(result.events), "segments": len(result.segment_diagnostics),
            "energy_scope": "not_declared_formal_kinetics",
            "chemical_inventory_scope": "formal_entity_coordinates_only"}
        if model.integer_kinetics is not None:
            diagnostics.update(solver=model.solver_method,
                stochastic_work=cast(_IntegerCandidate, result).work,
                stochastic_batches=len(result.segment_diagnostics),
                firings_committed=(cast(_IntegerCandidate, result).memory[
                                      "process"]["events_committed"]
                                  - cast(dict[str, Any], memory)["process"]["events_committed"]))
            if model.integer_kinetics.requires_boundaries:
                diagnostics["source_events_executed"] = cast(
                    _IntegerCandidate, result).source_events_executed
        return ModelProviderResult({}, diagnostics,
            state=candidate, actual_duration_s=result.actual_duration_s)
