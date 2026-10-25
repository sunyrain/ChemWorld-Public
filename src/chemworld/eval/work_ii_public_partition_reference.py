"""Explicit, simulator-free reference for one ideal two-liquid contact.

This is a kernel calibration slice, not the B3 final-assay observation map.
Inputs describe the actual contact volumes (including any initial volume).
The caller must establish the two-liquid domain; no stability oracle is hidden here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicContact:
    reference_coefficient: float
    temperature_K: float
    duration_s: float
    stirring_speed_rpm: float
    organic_volume_L: float
    aqueous_volume_L: float

    def __post_init__(self) -> None:
        values = (
            self.reference_coefficient,
            self.temperature_K,
            self.duration_s,
            self.stirring_speed_rpm,
            self.organic_volume_L,
            self.aqueous_volume_L,
        )
        if any(not math.isfinite(v) for v in values):
            raise ValueError("contact inputs must be finite")
        if min(self.reference_coefficient, self.organic_volume_L, self.aqueous_volume_L) <= 0:
            raise ValueError("coefficient and contact volumes must be positive")
        if self.duration_s < 0 or not 0 <= self.stirring_speed_rpm <= 2000:
            raise ValueError("contact controls outside the declared domain")
        if self.temperature_factor <= 0:
            raise ValueError("temperature factor must be positive")

    @property
    def temperature_factor(self) -> float:
        return 1 + 0.0025 * (self.temperature_K - 298.15)

    @property
    def mix_factor(self) -> float:
        return 0.75 + 0.25 * (1 - math.exp(-self.duration_s / 240)) * (
            0.70 + 0.30 * self.stirring_speed_rpm / 1200
        )

    @property
    def efficiency(self) -> float:
        return min(1.0, max(0.0, self.mix_factor))

    @property
    def entrainment(self) -> float:
        return min(
            0.04,
            max(
                0.0,
                0.01
                + 0.015 * self.stirring_speed_rpm / 1200 * (1 - math.exp(-self.duration_s / 120)),
            ),
        )


def forward(contact: PublicContact, exponent: float) -> dict[str, float]:
    """Predict fractions of initial product, before sampling or phase removal.

    Assumptions: ideal activities; coefficient/phase-volume multipliers one;
    product starts in the aqueous feed; one contact; automatic continuous phase.
    """
    if not math.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent must be finite and positive")
    coefficient = max(
        0.05,
        contact.reference_coefficient**exponent * contact.temperature_factor * contact.mix_factor,
    )
    equilibrium = (
        coefficient
        * contact.organic_volume_L
        / (coefficient * contact.organic_volume_L + contact.aqueous_volume_L)
    )
    contacted = contact.efficiency * equilibrium
    entrainment = contact.entrainment
    organic = (
        contacted + entrainment * (1 - contacted)
        if contact.organic_volume_L >= contact.aqueous_volume_L
        else contacted * (1 - entrainment)
    )
    return {
        "coefficient": coefficient,
        "organic_fraction": organic,
        "aqueous_fraction": 1 - organic,
    }


def invert_power_exponent(contact: PublicContact, organic_fraction: float) -> float:
    """Invert an unclipped, noiseless contact readout, conditional on the power family.

    No target exponent or scoring labels are accepted. This is not family selection.
    """
    if not math.isfinite(organic_fraction) or not 0 <= organic_fraction <= 1:
        raise ValueError("organic fraction must be finite and in [0,1]")
    if abs(math.log(contact.reference_coefficient)) < 1e-12:
        raise ValueError("unit reference coefficient carries no exponent information")
    entrainment = contact.entrainment
    contacted = (
        (organic_fraction - entrainment) / (1 - entrainment)
        if contact.organic_volume_L >= contact.aqueous_volume_L
        else organic_fraction / (1 - entrainment)
    )
    equilibrium = contacted / contact.efficiency
    if not 0 < equilibrium < 1:
        raise ValueError("readout is outside the invertible contact domain")
    coefficient = (
        equilibrium * contact.aqueous_volume_L / (contact.organic_volume_L * (1 - equilibrium))
    )
    if coefficient <= 0.05 + 1e-12:
        raise ValueError("coefficient floor prevents a unique exponent inversion")
    return math.log(coefficient / (contact.temperature_factor * contact.mix_factor)) / math.log(
        contact.reference_coefficient
    )
