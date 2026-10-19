"""Finite ideal-gas control-volume inventory and its open-system accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

GAS_IDS = ("nitrogen", "oxygen", "argon")
ATMOSPHERES = ("nitrogen", "argon", "air")
GAS_FEEDS = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.79, 0.21, 0.0))
GAS_R = 8.31446261815324
GAS_CV = (2.5 * GAS_R, 2.5 * GAS_R, 1.5 * GAS_R)


@dataclass(frozen=True)
class GasBoundary:
    """Fixed-volume, separately thermostatted pressure plenum; gases do not react."""

    volume_L: float
    temperature_K: float
    amounts_mol: tuple[float, ...]
    initial_mol: tuple[float, ...]
    added_mol: tuple[float, ...] = (0.0, 0.0, 0.0)
    removed_mol: tuple[float, ...] = (0.0, 0.0, 0.0)
    initial_energy_J: float = 0.0
    enthalpy_in_J: float = 0.0
    enthalpy_out_J: float = 0.0
    bath_heat_J: float = 0.0
    pump_work_J: float = 0.0

    def __post_init__(self) -> None:
        for name in ("amounts_mol", "initial_mol", "added_mol", "removed_mol"):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(isfinite(v) and v >= 0 for v in values):
                raise ValueError("Gas inventory requires three finite nonnegative components")
            object.__setattr__(self, name, values)
        if not 0.001 <= self.volume_L <= 0.1 or not 250 <= self.temperature_K <= 470:
            raise ValueError("Gas plenum volume/temperature outside the declared model domain")
        if not all(
            isfinite(v)
            for v in (
                self.initial_energy_J,
                self.enthalpy_in_J,
                self.enthalpy_out_J,
                self.bath_heat_J,
                self.pump_work_J,
            )
        ):
            raise ValueError("Gas energy records must be finite")
        if self.material_residual_mol > 1e-10 or abs(self.energy_residual_j) > 1e-7:
            raise ValueError("Gas material or energy ledger does not close")

    @classmethod
    def initialize(cls, volume_L: float, temperature_K: float, pressure_Pa: float) -> GasBoundary:
        total = pressure_Pa * volume_L * 0.001 / (GAS_R * temperature_K)
        amounts = tuple(total * f for f in GAS_FEEDS[2])
        energy = sum(n * cv * temperature_K for n, cv in zip(amounts, GAS_CV, strict=True))
        return cls(volume_L, temperature_K, amounts, amounts, initial_energy_J=energy)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GasBoundary:
        return cls(**value)

    @property
    def pressure_pa(self) -> float:
        return sum(self.amounts_mol) * GAS_R * self.temperature_K / (self.volume_L * 0.001)

    @property
    def energy_j(self) -> float:
        return sum(
            n * cv * self.temperature_K for n, cv in zip(self.amounts_mol, GAS_CV, strict=True)
        )

    @property
    def material_residual_mol(self) -> float:
        return max(
            abs(a + b - c - d)
            for a, b, c, d in zip(
                self.initial_mol, self.added_mol, self.removed_mol, self.amounts_mol, strict=True
            )
        )

    @property
    def energy_residual_j(self) -> float:
        return (
            self.energy_j
            - self.initial_energy_J
            - self.enthalpy_in_J
            + self.enthalpy_out_J
            - self.bath_heat_J
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
