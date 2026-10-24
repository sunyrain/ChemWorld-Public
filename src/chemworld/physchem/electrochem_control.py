"""Deterministic potentiostatic and galvanostatic setpoint recipes."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal

from chemworld._serialization import canonical_json_sha256

ControlMode = Literal["potentiostatic", "galvanostatic"]
ControlProfile = Literal["ramp", "hold"]


@dataclass(frozen=True)
class ElectrochemicalControlLimits:
    minimum_potential_V: float
    maximum_potential_V: float
    minimum_current_A: float
    maximum_current_A: float
    maximum_potential_slew_V_s: float
    maximum_current_slew_A_s: float
    provenance_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_potential_V", self.minimum_potential_V),
            ("maximum_potential_V", self.maximum_potential_V),
            ("minimum_current_A", self.minimum_current_A),
            ("maximum_current_A", self.maximum_current_A),
        ):
            _finite(value, name)
        if self.minimum_potential_V >= self.maximum_potential_V:
            raise ValueError("potential limits must be increasing")
        if self.minimum_current_A >= self.maximum_current_A:
            raise ValueError("current limits must be increasing")
        _positive(self.maximum_potential_slew_V_s, "maximum_potential_slew_V_s")
        _positive(self.maximum_current_slew_A_s, "maximum_current_slew_A_s")
        if not self.provenance_id:
            raise ValueError("provenance_id cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_potential_V": self.minimum_potential_V,
            "maximum_potential_V": self.maximum_potential_V,
            "minimum_current_A": self.minimum_current_A,
            "maximum_current_A": self.maximum_current_A,
            "maximum_potential_slew_V_s": self.maximum_potential_slew_V_s,
            "maximum_current_slew_A_s": self.maximum_current_slew_A_s,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class ElectrochemicalControlSegment:
    segment_id: str
    mode: ControlMode
    profile: ControlProfile
    target_value: float
    duration_s: float

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id cannot be empty")
        if self.mode not in {"potentiostatic", "galvanostatic"}:
            raise ValueError("unsupported control mode")
        if self.profile not in {"ramp", "hold"}:
            raise ValueError("unsupported control profile")
        _finite(self.target_value, "target_value")
        _positive(self.duration_s, "duration_s")

    @property
    def unit(self) -> str:
        return "V" if self.mode == "potentiostatic" else "A"

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "mode": self.mode,
            "profile": self.profile,
            "target_value": self.target_value,
            "duration_s": self.duration_s,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ElectrochemicalControlRecipe:
    recipe_id: str
    segments: tuple[ElectrochemicalControlSegment, ...]
    sample_interval_s: float
    provenance_id: str
    schema_version: str = "chemworld-electrochem-control-recipe-0.1"

    def __post_init__(self) -> None:
        if not self.recipe_id or not self.provenance_id:
            raise ValueError("recipe_id and provenance_id cannot be empty")
        if self.schema_version != "chemworld-electrochem-control-recipe-0.1":
            raise ValueError("unsupported electrochemical control recipe schema")
        if not self.segments:
            raise ValueError("control recipe requires at least one segment")
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("control segment ids must be unique")
        _positive(self.sample_interval_s, "sample_interval_s")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "segments": [segment.to_dict() for segment in self.segments],
            "sample_interval_s": self.sample_interval_s,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class ControlTracePoint:
    time_s: float
    segment_id: str
    mode: ControlMode
    profile: ControlProfile
    requested_value: float
    applied_value: float
    unit: str

    def to_dict(self) -> dict[str, object]:
        return {
            "time_s": self.time_s,
            "segment_id": self.segment_id,
            "mode": self.mode,
            "profile": self.profile,
            "requested_value": self.requested_value,
            "applied_value": self.applied_value,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ControlOperationLog:
    segment_id: str
    mode: ControlMode
    profile: ControlProfile
    start_time_s: float
    end_time_s: float
    start_value: float
    requested_target_value: float
    range_clipped_target_value: float
    applied_end_value: float
    unit: str
    range_clipped: bool
    slew_clipped: bool
    hold_step_exceeds_slew_limit: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "mode": self.mode,
            "profile": self.profile,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "start_value": self.start_value,
            "requested_target_value": self.requested_target_value,
            "range_clipped_target_value": self.range_clipped_target_value,
            "applied_end_value": self.applied_end_value,
            "unit": self.unit,
            "range_clipped": self.range_clipped,
            "slew_clipped": self.slew_clipped,
            "hold_step_exceeds_slew_limit": self.hold_step_exceeds_slew_limit,
        }


@dataclass(frozen=True)
class ElectrochemicalControlExecution:
    schema_version: str
    recipe_id: str
    recipe_hash: str
    execution_hash: str
    initial_potential_V: float
    initial_current_A: float
    final_potential_V: float
    final_current_A: float
    total_duration_s: float
    clipping_event_count: int
    operation_logs: tuple[ControlOperationLog, ...]
    trace: tuple[ControlTracePoint, ...]
    provenance: dict[str, str]

    def to_dict(self, *, include_execution_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "recipe_hash": self.recipe_hash,
            "initial_potential_V": self.initial_potential_V,
            "initial_current_A": self.initial_current_A,
            "final_potential_V": self.final_potential_V,
            "final_current_A": self.final_current_A,
            "total_duration_s": self.total_duration_s,
            "clipping_event_count": self.clipping_event_count,
            "operation_logs": [item.to_dict() for item in self.operation_logs],
            "trace": [item.to_dict() for item in self.trace],
            "provenance": dict(self.provenance),
        }
        if include_execution_hash:
            payload["execution_hash"] = self.execution_hash
        return payload


def execute_electrochemical_control_recipe(
    recipe: ElectrochemicalControlRecipe,
    limits: ElectrochemicalControlLimits,
    *,
    initial_potential_V: float,
    initial_current_A: float,
) -> ElectrochemicalControlExecution:
    _finite(initial_potential_V, "initial_potential_V")
    _finite(initial_current_A, "initial_current_A")
    potential = _clip(
        initial_potential_V,
        limits.minimum_potential_V,
        limits.maximum_potential_V,
    )
    current = _clip(
        initial_current_A,
        limits.minimum_current_A,
        limits.maximum_current_A,
    )
    time_s = 0.0
    logs: list[ControlOperationLog] = []
    trace: list[ControlTracePoint] = []
    clipping_events = int(potential != initial_potential_V) + int(current != initial_current_A)
    for segment in recipe.segments:
        if segment.mode == "potentiostatic":
            start = potential
            low = limits.minimum_potential_V
            high = limits.maximum_potential_V
            maximum_slew = limits.maximum_potential_slew_V_s
        else:
            start = current
            low = limits.minimum_current_A
            high = limits.maximum_current_A
            maximum_slew = limits.maximum_current_slew_A_s
        ranged_target = _clip(segment.target_value, low, high)
        range_clipped = ranged_target != segment.target_value
        slew_clipped = False
        hold_step_exceeds = False
        if segment.profile == "ramp":
            maximum_change = maximum_slew * segment.duration_s
            change = ranged_target - start
            applied_end = start + _clip(change, -maximum_change, maximum_change)
            slew_clipped = applied_end != ranged_target
        else:
            applied_end = ranged_target
            hold_step_exceeds = abs(applied_end - start) > maximum_slew * recipe.sample_interval_s
        clipping_events += int(range_clipped) + int(slew_clipped)
        point_count = max(1, ceil(segment.duration_s / recipe.sample_interval_s))
        for point_index in range(point_count + 1):
            local_fraction = point_index / point_count
            local_time = time_s + local_fraction * segment.duration_s
            if segment.profile == "ramp":
                requested = start + local_fraction * (segment.target_value - start)
                applied = start + local_fraction * (applied_end - start)
            else:
                requested = segment.target_value
                applied = applied_end
            trace.append(
                ControlTracePoint(
                    time_s=local_time,
                    segment_id=segment.segment_id,
                    mode=segment.mode,
                    profile=segment.profile,
                    requested_value=requested,
                    applied_value=applied,
                    unit=segment.unit,
                )
            )
        logs.append(
            ControlOperationLog(
                segment_id=segment.segment_id,
                mode=segment.mode,
                profile=segment.profile,
                start_time_s=time_s,
                end_time_s=time_s + segment.duration_s,
                start_value=start,
                requested_target_value=segment.target_value,
                range_clipped_target_value=ranged_target,
                applied_end_value=applied_end,
                unit=segment.unit,
                range_clipped=range_clipped,
                slew_clipped=slew_clipped,
                hold_step_exceeds_slew_limit=hold_step_exceeds,
            )
        )
        if segment.mode == "potentiostatic":
            potential = applied_end
        else:
            current = applied_end
        time_s += segment.duration_s
    recipe_hash = _sha256(
        {
            "recipe": recipe.to_dict(),
            "limits": limits.to_dict(),
            "initial_potential_V": initial_potential_V,
            "initial_current_A": initial_current_A,
        }
    )
    provisional = ElectrochemicalControlExecution(
        schema_version="chemworld-electrochem-control-execution-0.1",
        recipe_id=recipe.recipe_id,
        recipe_hash=recipe_hash,
        execution_hash="",
        initial_potential_V=initial_potential_V,
        initial_current_A=initial_current_A,
        final_potential_V=potential,
        final_current_A=current,
        total_duration_s=time_s,
        clipping_event_count=clipping_events,
        operation_logs=tuple(logs),
        trace=tuple(trace),
        provenance={"recipe": recipe.provenance_id, "limits": limits.provenance_id},
    )
    execution_hash = _sha256(provisional.to_dict(include_execution_hash=False))
    return ElectrochemicalControlExecution(
        **{
            **provisional.__dict__,
            "execution_hash": execution_hash,
        }
    )


def verify_electrochemical_control_replay(
    execution: ElectrochemicalControlExecution,
    recipe: ElectrochemicalControlRecipe,
    limits: ElectrochemicalControlLimits,
) -> bool:
    replay = execute_electrochemical_control_recipe(
        recipe,
        limits,
        initial_potential_V=execution.initial_potential_V,
        initial_current_A=execution.initial_current_A,
    )
    return replay.to_dict() == execution.to_dict()


def _sha256(payload: object) -> str:
    return canonical_json_sha256(payload, ensure_ascii=True)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _positive(value: float, field_name: str) -> None:
    if value <= 0.0 or not isfinite(value):
        raise ValueError(f"{field_name} must be positive and finite")


def _finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


__all__ = [
    "ControlOperationLog",
    "ControlTracePoint",
    "ElectrochemicalControlExecution",
    "ElectrochemicalControlLimits",
    "ElectrochemicalControlRecipe",
    "ElectrochemicalControlSegment",
    "execute_electrochemical_control_recipe",
    "verify_electrochemical_control_replay",
]
