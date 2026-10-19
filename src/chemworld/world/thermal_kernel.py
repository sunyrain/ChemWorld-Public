"""Thermal and safety-law module for ChemWorld."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chemworld.foundation import WorldState, selected_phase_id
from chemworld.world.mixtures import volumetric_heat_capacity, working_solvents
from chemworld.world.parameters import ChemWorldParameters


@dataclass(frozen=True)
class ThermalTransitionAccounting:
    """Closed control-volume energy ledger for a prescribed temperature change."""

    initial_temperature_K: float
    final_temperature_K: float
    duration_s: float
    sensible_energy_change_J: float
    jacket_energy_J: float
    heat_loss_J: float
    phase_change_heat_J: float
    energy_balance_residual_J: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "model_id": "lumped_thermal_transition_accounting_v1",
            "initial_temperature_K": self.initial_temperature_K,
            "final_temperature_K": self.final_temperature_K,
            "duration_s": self.duration_s,
            "sensible_energy_change_J": self.sensible_energy_change_J,
            "jacket_energy_J": self.jacket_energy_J,
            "heat_loss_J": self.heat_loss_J,
            "phase_change_heat_J": self.phase_change_heat_J,
            "energy_balance_residual_J": self.energy_balance_residual_J,
        }


def account_temperature_transition(
    *,
    state: WorldState,
    world: ChemWorldParameters,
    final_temperature_K: float,
    duration_s: float,
    phase_change_heat_J: float = 0.0,
) -> ThermalTransitionAccounting:
    """Close a signed lumped energy balance for any thermal unit operation.

    The sign convention matches the reaction runtime:
    ``dU = Q_jacket - Q_loss - Q_phase``.  Positive ``Q_loss`` is heat leaving
    a vessel warmer than its environment; exothermic phase/reaction heat is
    negative.
    """

    values = (final_temperature_K, duration_s, phase_change_heat_J)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("thermal transition inputs must be finite")
    if final_temperature_K <= 0.0 or duration_s <= 0.0:
        raise ValueError("thermal transition temperature and duration must be positive")
    heat_capacity_J_K = max(
        volumetric_heat_capacity(state, world.rho_cp_J_per_L_K) * state.volume_L, 1.0e-12
    )
    sensible = heat_capacity_J_K * (final_temperature_K - state.temperature_K)
    mean_temperature = 0.5 * (state.temperature_K + final_temperature_K)
    heat_loss = (
        float(world.ua_W_per_K)
        * (mean_temperature - float(world.environment_temperature_K))
        * duration_s
    )
    jacket = sensible + heat_loss + phase_change_heat_J
    residual = sensible - (jacket - heat_loss - phase_change_heat_J)
    return ThermalTransitionAccounting(
        initial_temperature_K=float(state.temperature_K),
        final_temperature_K=float(final_temperature_K),
        duration_s=float(duration_s),
        sensible_energy_change_J=float(sensible),
        jacket_energy_J=float(jacket),
        heat_loss_J=float(heat_loss),
        phase_change_heat_J=float(phase_change_heat_J),
        energy_balance_residual_J=float(residual),
    )


def pressure_and_risk(
    *,
    state: WorldState,
    solvent_risks: np.ndarray,
    pressure_override_Pa: float | None = None,
) -> tuple[float, float]:
    active_amounts = state.species_amounts
    active_phase_id = selected_phase_id(state.phases)
    if state.phases is not None and active_phase_id in state.phases.phases:
        active_amounts = state.phases.phases[active_phase_id].species_amounts_mol
    total_amount = sum(value for key, value in active_amounts.items() if not key.startswith("Cat"))
    concentration = 0.0 if state.volume_L <= 0 else total_amount / state.volume_L
    pressure = (
        101_325.0 * (state.temperature_K / 298.15) * (1.0 + 0.025 * concentration)
        if pressure_override_Pa is None
        else float(pressure_override_Pa)
    )
    if pressure <= 0.0 or not np.isfinite(pressure):
        raise ValueError("pressure_override_Pa must be positive and finite")
    exotherm_risk = min(1.0, abs(state.ledger.heat_reaction_J) / 2500.0)
    temperature_risk = 1.0 / (1.0 + np.exp(-(state.temperature_K - 405.0) / 13.0))
    concentration_risk = 1.0 / (1.0 + np.exp(-(concentration - 0.8) / 0.22))
    risk = float(
        np.clip(
            0.30 * temperature_risk
            + 0.20 * concentration_risk
            + 0.20 * exotherm_risk
            + 0.18 * working_solvents(state).linear_property(solvent_risks)
            + 0.12 * (pressure / 550_000.0),
            0.0,
            1.0,
        )
    )
    return float(pressure), risk


@dataclass(frozen=True)
class ThermalModuleSpec:
    module_id: str = "thermal"
    version: str = "0.3"
    laws: tuple[str, ...] = (
        "jacket_heat_input",
        "heat_loss_to_environment",
        "reaction_enthalpy",
        "temperature_pressure_risk_proxy",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "version": self.version,
            "laws": list(self.laws),
            "state_ledgers": [
                "energy_jacket_J",
                "heat_reaction_J",
                "heat_loss_J",
                "temperature_K",
                "pressure_Pa",
                "risk",
            ],
        }


__all__ = [
    "ThermalModuleSpec",
    "ThermalTransitionAccounting",
    "account_temperature_transition",
    "pressure_and_risk",
]
