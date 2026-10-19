"""Finite solvent-carrier inventory and declared mixture-property rules.

These are bounded medium models, not a real-solvent miscibility prediction.
Volumes are additive. Heat capacity is volume additive; positive kinetic and
solubility modifiers use log-volume interpolation of the world-specific endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, log
from typing import Any

SOLVENT_IDS = ("water", "ethanol", "acetonitrile", "toluene")
# Dimensionless surrogate endpoint ratios relative to the world's base rho*Cp.
# Names may be presented as anonymous S0..S3 by a task's material contract.
HEAT_CAPACITY_RATIOS = (1.0, 0.46, 0.42, 0.36)
DENSITY_RATIOS = (1.0, 0.79, 0.79, 0.87)
VISCOSITY_RATIOS = (1.0, 1.2, 0.37, 0.59)
RELATIVE_VOLATILITIES = (1.0, 2.0, 1.8, 0.7)
ORGANIC_DISTRIBUTION_RATIOS = (0.02, 1.2, 1.8, 24.0)
MIXTURE_MODEL_ID = "finite-additive-volume-log-property-mixtures-v1"


@dataclass(frozen=True)
class SolventInventory:
    volumes_L: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.volumes_L)
        if len(values) != len(SOLVENT_IDS) or not all(isfinite(v) and v >= 0 for v in values):
            raise ValueError("Solvent inventory requires four finite nonnegative volumes")
        object.__setattr__(self, "volumes_L", values)

    @classmethod
    def pure(cls, index: int, volume_L: float) -> SolventInventory:
        if isinstance(index, bool) or index not in range(len(SOLVENT_IDS)):
            raise ValueError("Solvent index outside the finite catalog")
        return cls(tuple(volume_L if i == index else 0.0 for i in range(len(SOLVENT_IDS))))

    @property
    def volume_L(self) -> float:  # noqa: N802 - unit-bearing inventory property
        return fsum(self.volumes_L)

    @property
    def fractions(self) -> tuple[float, ...]:
        volume = self.volume_L
        return tuple(v / volume for v in self.volumes_L) if volume else (1.0, 0.0, 0.0, 0.0)

    @property
    def dominant_index(self) -> int:
        return max(range(len(SOLVENT_IDS)), key=lambda i: self.volumes_L[i])

    def scale(self, factor: float) -> SolventInventory:
        if not isfinite(factor) or factor < 0:
            raise ValueError("Solvent scale must be finite and nonnegative")
        return SolventInventory(tuple(v * factor for v in self.volumes_L))

    def at_volume(self, volume_L: float) -> SolventInventory:
        if not isfinite(volume_L) or volume_L < 0:
            raise ValueError("Solvent volume must be finite and nonnegative")
        return SolventInventory(tuple(v * volume_L for v in self.fractions))

    def __add__(self, other: SolventInventory) -> SolventInventory:
        return SolventInventory(
            tuple(a + b for a, b in zip(self.volumes_L, other.volumes_L, strict=True))
        )

    def linear_property(self, endpoints: Any) -> float:
        values = tuple(float(v) for v in endpoints)
        if len(values) != 4 or not all(isfinite(v) for v in values):
            raise ValueError("Mixture property requires four finite endpoints")
        return fsum(x * v for x, v in zip(self.fractions, values, strict=True))

    def log_property(self, endpoints: Any) -> float:
        values = tuple(float(v) for v in endpoints)
        if len(values) != 4 or not all(isfinite(v) and v > 0 for v in values):
            raise ValueError("Log mixture property requires four positive endpoints")
        # Preserve exact endpoints, including floating-point identity.
        if sum(v > 0 for v in self.volumes_L) <= 1:
            return values[self.dominant_index]
        return exp(fsum(x * log(v) for x, v in zip(self.fractions, values, strict=True)))

    def split(
        self, volume_L: float, preference: tuple[float, ...] | None = None
    ) -> tuple[SolventInventory, SolventInventory]:
        """Conservative finite withdrawal; preference weights a separation cut.

        The saturating allocation v_i*k_i*q/(1+k_i*q) guarantees nonnegative
        residues and the requested receiver volume. This bounded separation
        law is explicitly a surrogate, not an equilibrium flash calculation.
        """
        if not isfinite(volume_L) or not 0 <= volume_L <= self.volume_L + 1e-12:
            raise ValueError("Solvent withdrawal exceeds available inventory")
        if volume_L >= self.volume_L:
            return self, SolventInventory()
        if volume_L == 0:
            return SolventInventory(), self
        if preference is None:
            taken = self.at_volume(volume_L)
        else:
            if len(preference) != 4 or not all(isfinite(k) and k > 0 for k in preference):
                raise ValueError("Separation preferences must be four positive finite numbers")
            low, high = 0.0, 1.0

            def volumes(q: float) -> tuple[float, ...]:
                return tuple(
                    v * (k * q / (1 + k * q))
                    for v, k in zip(self.volumes_L, preference, strict=True)
                )

            while fsum(volumes(high)) < volume_L:
                high *= 2
            for _ in range(80):
                midpoint = (low + high) / 2
                if fsum(volumes(midpoint)) < volume_L:
                    low = midpoint
                else:
                    high = midpoint
            taken = SolventInventory(volumes((low + high) / 2))
        remaining = SolventInventory(
            tuple(max(a - b, 0) for a, b in zip(self.volumes_L, taken.volumes_L, strict=True))
        )
        return taken, remaining

    def to_dict(self) -> dict[str, float]:
        return dict(zip(SOLVENT_IDS, self.volumes_L, strict=True))


@dataclass(frozen=True)
class SolventAccounting:
    initial: SolventInventory = SolventInventory()
    added: SolventInventory = SolventInventory()
    removed: SolventInventory = SolventInventory()

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key).to_dict() for key in ("initial", "added", "removed")}
