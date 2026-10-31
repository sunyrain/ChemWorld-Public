"""Checkpointed control programs with finite, replayable feedback responses."""

from __future__ import annotations

from dataclasses import replace
from math import floor, isclose, isfinite
from typing import Any

from chemworld.foundation import WorldState, equipment_settings, upsert_equipment_record
from chemworld.foundation.gas import GasBoundary
from chemworld.physchem.reactor_shared import JacketTemperatureProgram, ScalarControlProgram
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.world.control_contract import CONTROL_ACTION_FIELDS, CONTROL_CHOICES, CONTROL_NUMERIC

CONTROLLER_ID = "process_controller"
SENSOR_COST = 0.00001


def electrical_device(state):
    records = [
        record
        for record in state.equipment.equipment.values()
        if record.attached_vessel_id == state.vessel_id and "electrical_control" in record.settings
    ]
    if len(records) != 1:
        raise ValueError("electrical control requires exactly one connected source")
    return records[0]


def configure_electrical_control(state, action):
    """Set one supported source programme; physical advancement remains shared."""
    record = electrical_device(state)
    settings = record.settings["electrical_control"]
    if settings.get("passive_load_modes"):
        from chemworld.runtime.reverse_electrodialysis import configured_red_load_program

        mode, program = configured_red_load_program(state, record, action)
        return _configured_electrical(state, record, {**settings, "load_mode": mode}, program)
    from chemworld.runtime.thevenin_control import NAMED_FIELDS

    if settings.get("named_control_modes"):
        from chemworld.runtime.thevenin_control import configured_program

        program = configured_program(state, record, action)
        return _configured_electrical(state, record, settings, program)
    if NAMED_FIELDS & action.keys():
        raise ValueError("this electrical source does not declare named circuit controls")
    fields = [
        (name, unit, limits)
        for name, unit, limits in (("potential_V", "V", "limits_V"), ("current_A", "A", "limits_A"))
        if name in action
    ]
    if len(fields) != 1:
        raise ValueError("configure exactly one electrical control quantity")
    name, unit, limit_key = fields[0]
    if unit not in settings.get("supported_units", ("A", "V")):
        raise ValueError("electrical source does not support this control unit")
    if limit_key not in settings:
        raise ValueError("electrical source does not support this control mode")
    target = float(action[name])
    duration = float(action.get("ramp_duration_s", 0.0))
    low, high = settings[limit_key]
    if not isfinite(target) or not low <= target <= high or not isfinite(duration) or duration < 0:
        raise ValueError("electrical target or ramp duration is outside device capability")
    if settings.get("constant_potential_only") and (unit != "V" or duration != 0):
        raise ValueError("electrical source requires a constant potential without a ramp")
    old = ScalarControlProgram(**settings["program"])
    if old.unit != unit and duration:
        raise ValueError("a ramp cannot interpolate between different control units")
    time = state.ledger.time_s
    points = (
        ((time, target),)
        if duration == 0
        else ((time, old.value_at(time)), (time + duration, target))
    )
    program = ScalarControlProgram(points, unit, mode="linear")
    return _configured_electrical(state, record, settings, program)


def _configured_electrical(state, record, settings, program):
    return state.replace(
        equipment=upsert_equipment_record(
            state.equipment,
            equipment_id=record.equipment_id,
            equipment_type=record.equipment_type,
            attached_vessel_id=record.attached_vessel_id,
            status="running",
            settings={
                **record.settings,
                "electrical_control": {
                    **settings,
                    "program": program.to_dict(),
                    "stages": [],
                    "stage_index": 0,
                    "stage_offset": 0,
                    "stage_elapsed_s": 0.0,
                    "elapsed_s": 0.0,
                    "status": "running",
                },
            },
        ),
        process=replace(state.process, last_observation_record=None),
    )


def _store_electrical(state, settings):
    record = electrical_device(state)
    return state.replace(
        equipment=upsert_equipment_record(
            state.equipment,
            equipment_id=record.equipment_id,
            equipment_type=record.equipment_type,
            attached_vessel_id=record.attached_vessel_id,
            status=settings.get("status", "running"),
            settings={**record.settings, "electrical_control": settings},
        )
    )


def queue_electrical_stage(state, action):
    """Append one finite scalar stage; feedback uses measured terminal V or A."""
    checked = configure_electrical_control(state, action)
    program = electrical_device(checked).settings["electrical_control"]["program"]
    settings = dict(electrical_device(state).settings["electrical_control"])
    if not settings.get("staged_control", False):
        raise ValueError("source has no installed staged controller")
    duration = float(action["stage_duration_s"])
    if not isfinite(duration) or not 0 < duration <= 14400:
        raise ValueError("stage requires a finite duration within 14400 s")
    stops = {
        key: float(action[key])
        for key in ("stop_voltage_below_V", "stop_voltage_above_V", "stop_abs_current_A")
        if key in action
    }
    if len(stops) > 1 or any(not isfinite(v) for v in stops.values()):
        raise ValueError("stage accepts at most one finite terminal threshold")
    for key, value in stops.items():
        limits = settings["limits_A" if key == "stop_abs_current_A" else "limits_V"]
        if not (
            0 <= value <= max(abs(x) for x in limits)
            if key == "stop_abs_current_A"
            else limits[0] <= value <= limits[1]
        ):
            raise ValueError("feedback threshold exceeds the source domain")
    index = settings.get("stage_index", 0)
    offset = settings.get("stage_offset", 0)
    # Completed stages live in operation history, not an ever-growing device
    # queue. Absolute indices remain stable across compaction and snapshots.
    stages = list(settings.get("stages", []))[index - offset :]
    settings["stage_offset"] = index
    if len(stages) >= 32:
        raise ValueError("a control programme contains at most 32 pending stages")
    target = program["target"] if "control_mode" in program else program
    stage = {
        "unit": target["unit"],
        "target": target["setpoints"][-1][1],
        "stage_duration_s": duration,
        **stops,
    }
    if "control_mode" in program:
        stage["program"] = program
    settings.update(stages=[*stages, stage], status="running")
    settings.setdefault("stage_index", 0)
    settings.setdefault("stage_elapsed_s", 0.0)
    settings.setdefault("elapsed_s", 0.0)
    if not stages:
        settings["stage_elapsed_s"] = 0.0
        settings["program"] = _electrical_stage_program(stage, state.ledger.time_s)
    candidate = _store_electrical(state, settings)
    if not stages and settings.get("named_control_modes"):
        from chemworld.runtime.thevenin_control import validate_active_program

        validate_active_program(candidate)
    return candidate


def _electrical_stage_program(stage, time_s):
    if "program" in stage:
        from chemworld.runtime.thevenin_control import stage_program

        return stage_program(stage["program"], time_s)
    return ScalarControlProgram(((time_s, stage["target"]),), stage["unit"]).to_dict()


def electrical_control_error(state, action):
    """Validate through the same pure candidate builder used by execution."""
    try:
        builder = (
            queue_electrical_stage
            if action["operation"] == "queue_electrical_stage"
            else configure_electrical_control
        )
        builder(state, action)
    except (KeyError, TypeError, ValueError, ArithmeticError, RuntimeError) as exc:
        return str(exc)
    return None


def advance_electrical_program(state, action, advance):
    """Drive the common interval loop with one selected physical strategy."""
    settings = dict(electrical_device(state).settings["electrical_control"])
    events = []

    def interval(candidate, settings, remaining):
        index = settings["stage_index"]
        offset = settings.get("stage_offset", 0)
        stage = settings["stages"][index - offset]
        dt = min(remaining, stage["stage_duration_s"] - settings["stage_elapsed_s"])
        candidate = _store_electrical(candidate, settings)
        result = advance(candidate, {**action, "duration_s": dt}, stage, index)
        if result.outcome in {"rejected", "numerical_failure"}:
            return candidate, result
        candidate = result.state
        settings["elapsed_s"] += result.actual_duration_s
        settings["stage_elapsed_s"] += result.actual_duration_s
        event = result.diagnostics.get("stopping_event")
        stage_finished = event == f"control:stage:{index}" or (
            result.outcome == "success"
            and settings["stage_elapsed_s"] >= stage["stage_duration_s"] - 1e-8
        )
        if stage_finished:
            events.append(
                {
                    "stage_index": index,
                    "time_s": candidate.ledger.time_s,
                    "reason": event or "duration",
                    "actual_duration_s": settings["stage_elapsed_s"],
                    "terminal": result.diagnostics.get("electrical_control", {}),
                }
            )
            settings["stage_index"] += 1
            settings["stage_elapsed_s"] = 0.0
            if settings["stage_index"] == offset + len(settings["stages"]):
                settings["status"] = "completed"
                next_stage = {"unit": "A", "target": 0.0}
            else:
                next_stage = settings["stages"][settings["stage_index"] - offset]
            settings["program"] = _electrical_stage_program(next_stage, candidate.ledger.time_s)
            if settings.get("named_control_modes"):
                from chemworld.runtime.thevenin_control import validate_active_program

                try:
                    validate_active_program(_store_electrical(candidate, settings))
                except (ValueError, ArithmeticError, RuntimeError) as exc:
                    return candidate, ModelProviderResult({}, {
                        "control_activation_error": str(exc),
                        "attempted_stage_index": settings["stage_index"],
                        "attempted_stage_time_s": candidate.ledger.time_s,
                    }, success=False, outcome="numerical_failure",
                        failure_reason="electrical_stage_activation_failed")
            result = replace(result, outcome="success")
        return candidate, result

    result = advance_control_intervals(
        state, settings, float(action["duration_s"]), interval, _store_electrical, as_provider=True
    )
    if isinstance(result, WorldState):
        result = ModelProviderResult(
            {}, diagnostics={}, state=result,
            actual_duration_s=result.ledger.time_s - state.ledger.time_s,
        )
    return replace(result, diagnostics={**result.diagnostics, "control_transitions": events})


def control_settings(state: WorldState) -> dict[str, Any]:
    return equipment_settings(state.equipment, CONTROLLER_ID)


def _store(state: WorldState, settings: dict[str, Any]) -> WorldState:
    return state.replace(
        equipment=upsert_equipment_record(
            state.equipment,
            equipment_id=CONTROLLER_ID,
            equipment_type="sampled_process_controller",
            attached_vessel_id=state.vessel_id,
            status=settings["status"],
            settings=settings,
        )
    )


def control_error(state: WorldState, operation: str, action: dict[str, Any]) -> str | None:
    try:
        from chemworld.runtime.coupled_thermal import thermal_control

        device = thermal_control(state)
        if device is not None:
            if state.terminated:
                return "Control requires an open episode"
            if operation not in {"pause_control", "resume_control"}:
                return "This device supports pause/resume; sampled programmes are not installed"
            return (
                "Only a paused thermal device can resume"
                if operation == "resume_control" and device.enabled
                else None
            )
        settings = control_settings(state)
        if state.terminated:
            raise ValueError("Control requires an open episode")
        if operation != "configure_control" and not settings:
            raise ValueError("Configure the controller before using it")
        for key in CONTROL_ACTION_FIELDS[operation]:
            value = action.get(key)
            if key in CONTROL_CHOICES:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in range(len(CONTROL_CHOICES[key]))
                ):
                    raise ValueError(f"Invalid control choice: {key}")
            else:
                low, high = (
                    CONTROL_NUMERIC[key][:2]
                    if key in CONTROL_NUMERIC
                    else (1.0, 14400.0)
                    if key == "duration_s"
                    else (250.0, 430.0)
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not isfinite(value)
                    or not low <= value <= high
                ):
                    raise ValueError(f"Invalid control field: {key}")
        if "target_temperature_K" in action:
            bounds = (
                state.metadata.get("component_network", {})
                .get("bounds", {})
                .get(state.vessel_id, {})
                .get("heat:target_temperature_K", (250.0, 430.0))
            )
            if not bounds[0] <= action["target_temperature_K"] <= bounds[1]:
                raise ValueError("Control target is outside the vessel's authored thermal domain")
        if (
            operation == "configure_control"
            and settings
            and not isclose(action["headspace_L"], settings["headspace_L"], rel_tol=1e-7)
        ):
            raise ValueError("An initialized gas plenum has fixed physical volume")
        if operation == "queue_control_stage" and len(settings["stages"]) >= 32:
            raise ValueError("A control program contains at most 32 stages")
        if operation == "advance_control" and (
            settings["status"] != "running" or state.volume_L <= 0
        ):
            raise ValueError("Control advancement requires liquid and a running controller")
        if operation == "resume_control" and settings["status"] != "paused":
            raise ValueError("Only a paused controller can resume")
        if operation == "set_control_feedback":
            sensor = action["feedback_sensor"]
            bounds = (250.0, 470.0) if sensor == 0 else (1.0, 550000.0)
            if sensor != 2 and not bounds[0] <= action["feedback_threshold"] <= bounds[1]:
                raise ValueError("Feedback threshold must use the sensor's native K or Pa domain")
    except (ValueError, KeyError, TypeError) as exc:
        return str(exc)
    return None


def advance_control_intervals(state, settings, requested, interval, store, *, as_provider=False):
    """Shared persistent stage clock for gas/thermal and electrical controls."""
    remaining = requested
    start_elapsed = float(settings["elapsed_s"])
    provider = None
    intervals = []
    while remaining > 1e-8 and settings["status"] == "running":
        before_time = state.ledger.time_s
        before_index = settings["stage_index"]
        state, provider = interval(state, settings, remaining)
        intervals.append(
            {
                "stage_index": before_index,
                "actual_duration_s": provider.actual_duration_s,
                "outcome": provider.outcome,
                "diagnostics": dict(provider.diagnostics),
            }
        )
        if as_provider:
            provider = replace(
                provider, diagnostics={**provider.diagnostics, "control_intervals": list(intervals)}
            )
        if provider.outcome in {"rejected", "numerical_failure"}:
            return provider
        dt = provider.actual_duration_s
        if abs(state.ledger.time_s - before_time - dt) > 1e-7:
            raise ValueError("Controller and plant clocks disagree")
        if (
            dt <= 1e-10
            and settings["stage_index"] == before_index
            and provider.outcome == "success"
            and settings["status"] == "running"
        ):
            raise ValueError("Controller interval made no physical or stage progress")
        remaining -= dt
        state = store(state, settings)
        if provider.outcome != "success":
            settings["status"] = "paused"
            break
    settings["last_requested_s"] = requested
    settings["last_executed_s"] = settings["elapsed_s"] - start_elapsed
    candidate = store(state, settings)
    if provider is not None and provider.outcome != "success":
        return replace(
            provider,
            state=candidate,
            actual_duration_s=settings["last_executed_s"],
        )
    if settings["status"] != "running" and remaining > 1e-8:
        return ModelProviderResult(
            outputs={},
            diagnostics={
                **(provider.diagnostics if provider else {}),
                "termination_reason": "control_" + settings["status"],
            },
            state=candidate,
            outcome="stopped",
            actual_duration_s=settings["last_executed_s"],
        )
    if as_provider and provider is not None:
        return replace(provider, state=candidate, actual_duration_s=settings["last_executed_s"])
    return candidate


class ControlProgramServices:
    def __init__(self, reaction_thermal: Any) -> None:
        self.reaction_thermal = reaction_thermal

    def apply(self, state: WorldState, action: dict[str, Any]) -> WorldState | ModelProviderResult:
        operation = str(action["operation"])
        error = control_error(state, operation, action)
        if error:
            raise ValueError(error)
        from chemworld.runtime.coupled_thermal import set_thermal_enabled, thermal_control

        if thermal_control(state) is not None:
            return set_thermal_enabled(state, operation == "resume_control")
        settings = control_settings(state)
        if operation == "configure_control":
            gas = (
                GasBoundary.from_dict(settings["gas"])
                if settings
                else GasBoundary.initialize(
                    float(action["headspace_L"]),
                    state.temperature_K,
                    state.pressure_Pa,
                )
            )
            settings = {
                **settings,
                **{key: action[key] for key in CONTROL_ACTION_FIELDS[operation]},
                "gas": gas.to_dict(),
                "status": "running",
                "stages": [],
                "stage_index": 0,
                "headspace_L": gas.volume_L,
                "stage_elapsed_s": 0.0,
                "elapsed_s": settings.get("elapsed_s", 0.0),
                "jacket_temperature_K": settings.get("jacket_temperature_K", state.temperature_K),
                "feedback_sensor": 2,
                "feedback_threshold": 350.0,
                "feedback_direction": 0,
                "feedback_response": 0,
                "sensor_count": settings.get("sensor_count", 0),
                "events": [],
                "last_samples": [],
                "program_generation": settings.get("program_generation", 0) + 1,
            }
        elif operation == "queue_control_stage":
            if settings["stage_index"] == len(settings["stages"]):
                settings["stage_elapsed_s"] = 0.0
            settings["stages"] = [
                *settings["stages"],
                {key: action[key] for key in CONTROL_ACTION_FIELDS[operation]},
            ]
        elif operation == "set_control_feedback":
            settings.update({key: action[key] for key in CONTROL_ACTION_FIELDS[operation]})
        elif operation in {"pause_control", "resume_control"}:
            settings["status"] = "paused" if operation == "pause_control" else "running"
        else:
            return self._advance(state, settings, float(action["duration_s"]))
        return _store(state, settings)

    def _advance(
        self,
        state: WorldState,
        settings: dict[str, Any],
        requested: float,
    ) -> WorldState | ModelProviderResult:
        settings["last_samples"] = []
        return advance_control_intervals(state, settings, requested, self._interval, _store)

    def _interval(self, state, settings, remaining):
        index = settings["stage_index"]
        stage = settings["stages"][index] if index < len(settings["stages"]) else settings
        interval = float(settings["control_interval_s"])
        elapsed = float(settings["elapsed_s"])
        boundary = (floor((elapsed + 1e-9) / interval) + 1) * interval
        dt = min(remaining, boundary - elapsed)
        if stage is not settings:
            dt = min(dt, float(stage["stage_duration_s"]) - settings["stage_elapsed_s"])
        if dt < 1e-6:
            raise ValueError("Control step is below the declared integration resolution")
        initial_jacket = float(settings["jacket_temperature_K"])
        target = float(stage["target_temperature_K"])
        reach_time = abs(target - initial_jacket) / float(stage["ramp_rate_K_s"])
        end_jacket = initial_jacket + max(
            -dt * stage["ramp_rate_K_s"],
            min(dt * stage["ramp_rate_K_s"], target - initial_jacket),
        )
        knots = [(0.0, initial_jacket)]
        if 0 < reach_time < dt:
            knots.append((reach_time, target))
        knots.append((dt, end_jacket))
        gas = GasBoundary.from_dict(settings["gas"])
        advancement = self.reaction_thermal.advancer.control_stage(
            state,
            {"duration_s": dt, "target_temperature_K": end_jacket},
            jacket_program=JacketTemperatureProgram(tuple(knots), mode="linear"),
            gas=gas,
            pressure_setpoint_Pa=float(stage["pressure_Pa"]),
            atmosphere=int(stage["atmosphere"]),
        )
        provider = advancement.provider
        if provider.outcome in {"rejected", "numerical_failure"}:
            return state, provider
        if provider.state is None:
            raise ValueError("plant result requires a candidate state")
        state = provider.state
        dt = provider.actual_duration_s
        end_jacket = initial_jacket + max(
            -dt * stage["ramp_rate_K_s"],
            min(dt * stage["ramp_rate_K_s"], target - initial_jacket),
        )
        updated = advancement.gas
        settings.update(
            gas=updated.to_dict(),
            last_gas_exchanges=advancement.exchanges,
            jacket_temperature_K=end_jacket,
            elapsed_s=elapsed + dt,
            stage_elapsed_s=settings["stage_elapsed_s"] + dt,
        )
        if abs(settings["elapsed_s"] - boundary) < 1e-7:
            reading = {
                "time_s": state.ledger.time_s,
                "control_time_s": settings["elapsed_s"],
                "temperature_K": round(state.temperature_K / 0.01) * 0.01,
                "pressure_Pa": round(updated.pressure_pa / 10) * 10,
            }
            settings["last_samples"].append(reading)
            settings["sensor_count"] += 1
            state = state.replace(
                ledger=state.ledger.with_updates(cost=state.ledger.cost + SENSOR_COST)
            )
            sensor = settings["feedback_sensor"]
            if sensor != 2:
                value = reading["temperature_K" if sensor == 0 else "pressure_Pa"]
                triggered = (
                    value >= settings["feedback_threshold"]
                    if settings["feedback_direction"] == 0
                    else value <= settings["feedback_threshold"]
                )
                if triggered:
                    settings["events"] = [
                        *settings["events"],
                        {
                            **reading,
                            "response": settings["feedback_response"],
                            "stage_index": index,
                        },
                    ]
                    if settings["feedback_response"] == 0 or index + 1 >= len(settings["stages"]):
                        settings["status"] = "paused"
                    else:
                        settings["stage_index"], settings["stage_elapsed_s"] = index + 1, 0.0
        if (
            stage is not settings
            and settings["stage_index"] == index
            and settings["stage_elapsed_s"] >= stage["stage_duration_s"] - 1e-8
        ):
            settings.update(
                {
                    key: stage[key]
                    for key in (
                        "target_temperature_K",
                        "pressure_Pa",
                        "atmosphere",
                        "ramp_rate_K_s",
                    )
                }
            )
            settings["stage_index"], settings["stage_elapsed_s"] = index + 1, 0.0
        state = _store(state, settings)
        state = self.reaction_thermal.with_risk_and_pressure(state)
        return state, provider


def public_control(state: WorldState) -> dict[str, Any] | None:
    settings = control_settings(state)
    if not settings:
        records = [
            r
            for r in state.equipment.equipment.values()
            if r.attached_vessel_id == state.vessel_id and "electrical_control" in r.settings
        ]
        if len(records) == 1:
            source = records[0].settings["electrical_control"]
            return {
                "kind": "electrical",
                **{
                    key: source[key]
                    for key in (
                        "program",
                        "limits_A",
                        "limits_V",
                        "stages",
                        "stage_index",
                        "stage_offset",
                        "stage_elapsed_s",
                        "elapsed_s",
                        "status",
                    )
                    if key in source
                },
            }
        return None
    gas = GasBoundary.from_dict(settings["gas"])
    return (
        {
            key: settings[key]
            for key in (
                "status",
                "stages",
                "stage_index",
                "stage_elapsed_s",
                "elapsed_s",
                "jacket_temperature_K",
                "target_temperature_K",
                "pressure_Pa",
                "atmosphere",
                "ramp_rate_K_s",
                "control_interval_s",
                "headspace_L",
                "feedback_sensor",
                "feedback_threshold",
                "feedback_direction",
                "feedback_response",
                "sensor_count",
            )
        }
        | {
            "latest_sensor": settings["last_samples"][-1] if settings["last_samples"] else None,
            "recent_events": settings["events"][-8:],
            "sensor_model": {
                "temperature_resolution_K": 0.01,
                "pressure_resolution_Pa": 10,
                "noise": "none",
                "cost_per_acquisition": SENSOR_COST,
            },
            "gas_model": (
                "fixed-volume separately thermostatted ideal-gas plenum; no gas-liquid chemistry"
            ),
        }
        | {
            "gas_resources": {
                "added_mol": list(gas.added_mol),
                "removed_mol": list(gas.removed_mol),
                "gas_ids": ["nitrogen", "oxygen", "argon"],
                "pump_work_J": gas.pump_work_J,
                "bath_heat_J": gas.bath_heat_J,
            }
        }
    )
