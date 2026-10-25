"""Continuous feedback equations; the original heater remains the only heat port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any

from chemworld.physchem.coupled_system import (
    CoupledCoordinate,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    HeatFlux,
    ProcessContribution,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec


@dataclass(frozen=True)
class ThermalFeedbackSpec:
    """Explicit direct-acting P/PI/PD/filtered-PID with actuator back-calculation.

    The manipulated variable is the original utility's jacket temperature, K.
    The device's setpoint is the desired sample temperature while installed.
    Disabled heater: integral freezes, measurement filter continues, heat is zero.
    """

    proportional_gain: float = 1.0
    integral_gain_s_inv: float = 0.1
    derivative_gain_s: float = 0.0
    derivative_filter_time_s: float = 0.5
    antiwindup_rate_s_inv: float = 0.5
    bias_temperature_K: float = 350.0
    minimum_command_K: float = 300.0
    maximum_command_K: float = 430.0
    pause_rule: str = "freeze-integral-track-measurement"

    def __post_init__(self) -> None:
        for key in (
            "proportional_gain",
            "integral_gain_s_inv",
            "derivative_gain_s",
            "antiwindup_rate_s_inv",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError("thermal feedback gains must be finite and nonnegative")
        if (
            isinstance(self.derivative_filter_time_s, bool)
            or not isfinite(self.derivative_filter_time_s)
            or self.derivative_filter_time_s <= 0
            or any(
                not isfinite(v) or v <= 0
                for v in (
                    self.bias_temperature_K,
                    self.minimum_command_K,
                    self.maximum_command_K,
                )
            )
            or not self.minimum_command_K < self.maximum_command_K
            or self.pause_rule != "freeze-integral-track-measurement"
            or (self.integral_gain_s_inv == 0 and self.antiwindup_rate_s_inv != 0)
        ):
            raise ValueError(
                "feedback requires finite filter/domain and explicit pause/P semantics"
            )

    @property
    def controller_type(self) -> str:
        return (
            "P"
            + ("I" if self.integral_gain_s_inv else "")
            + ("D" if self.derivative_gain_s else "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {"model": "finite-thermal-feedback-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ThermalFeedbackSpec:
        value = dict(payload)
        if value.pop("model", None) != "finite-thermal-feedback-v1":
            raise ValueError("unsupported thermal feedback model")
        return cls(**value)


def feedback_key(equipment_id: str, quantity: str) -> str:
    import json

    return json.dumps(("thermal_feedback", equipment_id, quantity), separators=(",", ":"))


def thermal_feedback_point(
    specification: ThermalFeedbackSpec,
    device: ThermalDeviceSpec,
    *,
    sample_temperature_K: float,
    integral_K: float,
    filtered_temperature_K: float,
) -> dict[str, Any]:
    """Return actual heater power and both memory derivatives, never advance time."""
    spec = specification
    if (
        device.mode != "finite_power"
        or device.conductance_W_K <= 0
        or not device.minimum_setpoint_K
        <= spec.minimum_command_K
        < spec.maximum_command_K
        <= device.maximum_setpoint_K
        or any(
            not isfinite(v)
            for v in (
                sample_temperature_K,
                integral_K,
                filtered_temperature_K,
            )
        )
        or sample_temperature_K <= 0
        or filtered_temperature_K <= 0
    ):
        raise ValueError("feedback requires positive finite-conductance heater and valid memory")
    error = device.setpoint_K - sample_temperature_K
    derivative = (sample_temperature_K - filtered_temperature_K) / spec.derivative_filter_time_s
    raw = (
        spec.bias_temperature_K
        + spec.proportional_gain * error
        + integral_K
        - spec.derivative_gain_s * derivative
    )
    command = min(spec.maximum_command_K, max(spec.minimum_command_K, raw))
    power = replace(device, setpoint_K=command).delivered_power_w(sample_temperature_K)
    realized = sample_temperature_K + power / (device.conductance_W_K * device.calibration_factor)
    integral_rate = (
        spec.integral_gain_s_inv * error + spec.antiwindup_rate_s_inv * (realized - raw)
        if device.enabled and spec.integral_gain_s_inv > 0
        else 0.0
    )
    return {
        "controller_type": spec.controller_type,
        "enabled": device.enabled,
        "pause_rule": spec.pause_rule,
        "target_temperature_K": device.setpoint_K,
        "sample_temperature_K": sample_temperature_K,
        "error_K": error,
        "integral_K": integral_K,
        "filtered_temperature_K": filtered_temperature_K,
        "raw_command_K": raw,
        "command_K": command,
        "realized_command_K": realized,
        "actual_heater_power_W": power,
        "integral_rate_K_s": integral_rate,
        "filter_rate_K_s": derivative,
        "command_limited": raw != command,
        "power_limited": device.enabled and abs(realized - command) > 1e-10,
    }


def augment_thermal_feedback(
    system: CoupledSystem,
    initial: CoupledState,
    specification: ThermalFeedbackSpec,
    device: ThermalDeviceSpec,
    equipment_id: str,
    *,
    integral_K: float,
    filtered_temperature_K: float,
) -> tuple[CoupledSystem, CoupledState]:
    """Add two non-material coordinates to the original local coupled solve."""
    if system.isothermal or system.residual_rhs_jacobian is not None:
        raise ValueError("feedback requires the original dynamic thermal ODE assembly")
    keys = (
        feedback_key(equipment_id, "integral_K"),
        feedback_key(equipment_id, "filtered_temperature_K"),
    )
    if not equipment_id or any(k in initial.coordinates for k in keys):
        raise ValueError("feedback equipment and coordinates must have unique ownership")

    def point(state: CoupledStateView) -> dict[str, Any]:
        return thermal_feedback_point(
            specification,
            device,
            sample_temperature_K=state.temperature_K,
            integral_K=state.coordinates[keys[0]],
            filtered_temperature_K=state.coordinates[keys[1]],
        )

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        value = point(state)
        return ProcessContribution(
            coordinate_rates={
                keys[0]: value["integral_rate_K_s"],
                keys[1]: value["filter_rate_K_s"],
            }
        )

    original_heat_flux = system.heat_flux

    def heat_flux(state: CoupledStateView) -> HeatFlux:
        # Replace only the original heater term. Environmental loss/work and
        # all species energies remain the original system's responsibility.
        return replace(
            original_heat_flux(state),
            heater_W=point(state)["actual_heater_power_W"],
        )

    packed = replace(
        initial,
        coordinates={
            **initial.coordinates,
            keys[0]: integral_K,
            keys[1]: filtered_temperature_K,
        },
    )
    point(packed)
    compiled = replace(
        system,
        processes=(
            *system.processes,
            CoupledProcess(
                f"thermal-feedback:{equipment_id}",
                evaluate,
            ),
        ),
        heat_flux=heat_flux,
        coordinates=(
            *system.coordinates,
            CoupledCoordinate(keys[0], "K", 100.0),
            CoupledCoordinate(keys[1], "K", 350.0, minimum=1e-9),
        ),
    )
    return compiled, packed
