"""One finite opaque wall, with its own temperature in an original equipment slot."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any

from chemworld.foundation.state_ownership import StateFieldSpec, StateSlot, StateSlotSpec
from chemworld.physchem.coupled_system import CoupledCoordinate

STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
GAS_WALL_TYPE = "finite_gas_thermal_wall_v1"


def _number(value: float, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"wall {name} must be finite and {'positive' if positive else 'nonnegative'}"
        )


def wall_key(owner: str, kind: str, vessel: str | None = None) -> str:
    parts = ["thermal_wall", owner, kind]
    if vessel is not None:
        parts.append(vessel)
    return json.dumps(parts, separators=(",", ":"))


@dataclass(frozen=True)
class GasWallContact:
    """Signed gas-to-wall conduction at a named original gas vessel."""

    vessel_id: str
    conductance_W_K: float

    def __post_init__(self) -> None:
        if not isinstance(self.vessel_id, str) or not self.vessel_id:
            raise ValueError("wall contact requires a vessel identity")
        _number(self.conductance_W_K, "contact conductance", positive=True)


@dataclass(frozen=True)
class GasThermalWallSpec:
    """Lumped constant-C wall radiating to a prescribed large isothermal enclosure.

    The gas is not a radiating medium. Area is the external radiating area, the
    enclosure view factor is one, and the emissivity is grey and temperature
    independent. Internal contacts use explicitly supplied lumped conductances.
    """

    owner_id: str
    contacts: tuple[GasWallContact, ...]
    initial_temperature_K: float
    heat_capacity_J_K: float
    reference_temperature_K: float = 298.15
    environment_temperature_K: float = 300.0
    loss_conductance_W_K: float = 0.0
    emissivity: float = 0.0
    radiating_area_m2: float = 0.0
    minimum_temperature_K: float = 200.0
    maximum_temperature_K: float = 1500.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contacts", tuple(sorted(self.contacts, key=lambda c: c.vessel_id))
        )
        if not isinstance(self.owner_id, str) or not self.owner_id:
            raise ValueError("wall requires a stable equipment owner")
        if not self.contacts or len({c.vessel_id for c in self.contacts}) != len(self.contacts):
            raise ValueError("wall contacts must name distinct vessels")
        for name in (
            "initial_temperature_K",
            "heat_capacity_J_K",
            "reference_temperature_K",
            "environment_temperature_K",
            "minimum_temperature_K",
            "maximum_temperature_K",
        ):
            _number(getattr(self, name), name, positive=True)
        for name in ("loss_conductance_W_K", "emissivity", "radiating_area_m2"):
            _number(getattr(self, name), name)
        if self.emissivity > 1 or (self.emissivity > 0 and self.radiating_area_m2 == 0):
            raise ValueError("wall emissivity is in [0, 1] and radiation needs positive area")
        if not self.minimum_temperature_K < self.initial_temperature_K < self.maximum_temperature_K:
            raise ValueError("wall initial temperature must be interior to its declared interval")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "contacts": [asdict(c) for c in self.contacts]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasThermalWallSpec:
        values = dict(value)
        values["contacts"] = tuple(GasWallContact(**c) for c in values["contacts"])
        return cls(**values)

    def make_slot(self, vessel_id: str, temperature_K: float | None = None) -> StateSlot:
        return StateSlot(
            StateSlotSpec(
                f"{self.owner_id}:temperature",
                "equipment",
                self.owner_id,
                GAS_WALL_TYPE,
                (StateFieldSpec("temperature_K", "K", mixing="equal", public=True),),
                batch_reset="preserve",
            ),
            vessel_id,
            {
                "temperature_K": self.initial_temperature_K
                if temperature_K is None
                else temperature_K
            },
            {"temperature_K": self.initial_temperature_K},
        )

    def validate_slot(self, slot: StateSlot, vessel_id: str) -> None:
        expected = self.make_slot(vessel_id)
        if (
            slot.spec != expected.spec
            or slot.location_id != vessel_id
            or slot.initial_values != expected.initial_values
        ):
            raise ValueError("wall slot must retain its declared owner, location and initial state")
        temperature = slot.values["temperature_K"]
        if (
            not self.minimum_temperature_K - 1e-7
            <= temperature
            <= self.maximum_temperature_K + 1e-7
        ):
            raise ValueError("wall temperature exceeds its declared domain")

    def project(self, coordinates: Mapping[str, float], slot: StateSlot) -> StateSlot:
        return replace(
            slot, values={"temperature_K": coordinates[wall_key(self.owner_id, "temperature_K")]}
        )

    def coordinates(self) -> tuple[CoupledCoordinate, ...]:
        return (
            CoupledCoordinate(wall_key(self.owner_id, "temperature_K"), "K", 500),
            CoupledCoordinate(wall_key(self.owner_id, "environment_conduction_J"), "J", 1),
            CoupledCoordinate(wall_key(self.owner_id, "environment_radiation_J"), "J", 1),
            *(
                CoupledCoordinate(wall_key(self.owner_id, "contact_J", c.vessel_id), "J", 1)
                for c in self.contacts
            ),
        )

    def environment_losses(self, temperature_K: float) -> tuple[float, float]:
        """Positive heat leaves the wall. Stable quartic difference permits negative loss."""
        environment = self.environment_temperature_K
        difference = temperature_K - environment
        return (
            self.loss_conductance_W_K * difference,
            self.emissivity
            * STEFAN_BOLTZMANN_W_M2_K4
            * self.radiating_area_m2
            * difference
            * (temperature_K + environment)
            * (temperature_K**2 + environment**2),
        )
