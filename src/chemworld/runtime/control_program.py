"""Checkpointed control programs with finite, replayable feedback responses."""

from __future__ import annotations

from math import floor, isclose, isfinite
from typing import Any

from chemworld.foundation import WorldState, equipment_settings, upsert_equipment_record
from chemworld.foundation.gas import GasBoundary
from chemworld.physchem.gas_control import advance_gas
from chemworld.physchem.reactor_shared import JacketTemperatureProgram
from chemworld.world.control_contract import CONTROL_ACTION_FIELDS, CONTROL_CHOICES, CONTROL_NUMERIC

CONTROLLER_ID = "process_controller"
SENSOR_COST = 0.00001


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


class ControlProgramServices:
    def __init__(self, reaction_thermal: Any) -> None:
        self.reaction_thermal = reaction_thermal

    def apply(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        operation = str(action["operation"])
        error = control_error(state, operation, action)
        if error:
            raise ValueError(error)
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

    def _advance(self, state: WorldState, settings: dict[str, Any], requested: float) -> WorldState:
        remaining = requested
        settings["last_samples"] = []
        start_elapsed = float(settings["elapsed_s"])
        while remaining > 1e-8 and settings["status"] == "running":
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
            before = state
            state = self.reaction_thermal.integrate(
                state,
                {"duration_s": dt, "target_temperature_K": end_jacket},
                heat=True,
                jacket_program=JacketTemperatureProgram(tuple(knots), mode="linear"),
            )
            gas = GasBoundary.from_dict(settings["gas"])
            updated = advance_gas(
                gas,
                duration_s=dt,
                final_temperature_K=state.temperature_K,
                pressure_setpoint_Pa=float(stage["pressure_Pa"]),
                atmosphere=int(stage["atmosphere"]),
            )
            gas_input = sum(updated.added_mol) - sum(gas.added_mol)
            pump = updated.pump_work_J - gas.pump_work_J
            bath = updated.bath_heat_J - gas.bath_heat_J
            settings.update(
                gas=updated.to_dict(),
                jacket_temperature_K=end_jacket,
                elapsed_s=elapsed + dt,
                stage_elapsed_s=settings["stage_elapsed_s"] + dt,
            )
            state = state.replace(
                pressure_Pa=updated.pressure_pa,
                ledger=state.ledger.with_updates(
                    cost=state.ledger.cost + gas_input * 0.5 + pump / 250000,
                    gas_pump_work_J=state.ledger.gas_pump_work_J + pump,
                    gas_bath_heat_J=state.ledger.gas_bath_heat_J + bath,
                ),
            )
            remaining -= dt
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
                        if settings["feedback_response"] == 0 or index + 1 >= len(
                            settings["stages"]
                        ):
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
            if abs(state.ledger.time_s - before.ledger.time_s - dt) > 1e-7:
                raise ValueError("Controller and plant clocks disagree")
        settings["last_requested_s"] = requested
        settings["last_executed_s"] = settings["elapsed_s"] - start_elapsed
        return _store(state, settings)


def public_control(state: WorldState) -> dict[str, Any] | None:
    settings = control_settings(state)
    if not settings:
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
