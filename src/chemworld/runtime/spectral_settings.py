"""Instrument-local configurable acquisition settings, independent of material."""

from __future__ import annotations

from math import isfinite
from typing import Any

from chemworld.foundation import WorldState, equipment_settings, upsert_equipment_record
from chemworld.world.spectral_contract import (
    SPECTRAL_BOUNDS,
    SPECTRAL_INSTRUMENTS,
    default_spectral_settings,
)


def spectral_settings(state: WorldState, instrument: str) -> dict[str, float]:
    return default_spectral_settings() | equipment_settings(
        state.equipment, "spectral_config:" + instrument
    )


def public_spectral_settings(state: WorldState) -> dict[str, Any]:
    return {
        name: spectral_settings(state, name)
        for name in SPECTRAL_INSTRUMENTS
        if equipment_settings(state.equipment, "spectral_config:" + name)
    }


def configuration_error(state: WorldState, action: dict[str, Any]) -> str | None:
    if state.terminated:
        return "Configure instruments during an open episode"
    if action.get("instrument") not in SPECTRAL_INSTRUMENTS:
        return "Only nmr, ir and ms have configurable spectral acquisition"
    for key, (low, high, _, _) in SPECTRAL_BOUNDS.items():
        value = action.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not low <= value <= high
        ):
            return "Invalid spectral field: " + key
        if key == "scan_count" and value != int(value):
            return "scan_count must be an integer"
    return None


def configure_instrument(state: WorldState, action: dict[str, Any]) -> WorldState:
    error = configuration_error(state, action)
    if error:
        raise ValueError(error)
    return state.replace(
        equipment=upsert_equipment_record(
            state.equipment,
            equipment_id="spectral_config:" + action["instrument"],
            equipment_type="spectral_configuration",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={key: float(action[key]) for key in SPECTRAL_BOUNDS},
        )
    )
