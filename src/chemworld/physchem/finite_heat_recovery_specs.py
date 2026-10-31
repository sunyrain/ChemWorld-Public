"""Declared four-tank, two-circuit counterflow apparatus."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from chemworld.physchem.elements import explicit_formula_charge
from chemworld.physchem.enthalpy import caloric_temperature
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties


def number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class FiniteHeatRecoverySpec:
    properties: LiquidTransportProperties
    conductance_W_K: float
    source_note: str
    left_feed_vessel_id: str = "left_feed"
    right_feed_vessel_id: str = "right_feed"
    right_receiver_vessel_id: str = "right_receiver"
    pumps: tuple[PumpSpec, PumpSpec] = field(default_factory=lambda: (PumpSpec(), PumpSpec()))
    pipes: tuple[PipeSpec, PipeSpec] = field(
        default_factory=lambda: (PipeSpec(0.1, 0.002), PipeSpec(0.1, 0.002))
    )
    capacities_J_K: tuple[float, float, float, float] = (5.0, 7.0, 6.0, 8.0)
    receiver_capacities_L: tuple[float, float] = (1.0, 1.0)
    pickup_volumes_L: tuple[float, float] = (1e-6, 1e-6)
    reference_temperature_K: float = 298.15
    maximum_root_iterations: int = 40

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "properties",
            LiquidTransportProperties.from_dict(deepcopy(self.properties.to_dict())),
        )
        for key in (
            "pumps",
            "pipes",
            "capacities_J_K",
            "receiver_capacities_L",
            "pickup_volumes_L",
        ):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        names = (self.left_feed_vessel_id, self.right_feed_vessel_id, self.right_receiver_vessel_id)
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 3:
            raise ValueError("each heat-recovery tank requires a unique identity")
        if not isinstance(self.source_note, str) or not self.source_note.strip():
            raise ValueError("heat-recovery conductance requires an authored source and scope")
        number(self.conductance_W_K, "conductance")
        if (
            len(self.pumps) != 2
            or len(self.pipes) != 2
            or any(not isinstance(p, PumpSpec) for p in self.pumps)
            or any(not isinstance(p, PipeSpec) or p.elevation_change_m != 0 for p in self.pipes)
        ):
            raise ValueError("declare two original pumps/pipes with equal-elevation tanks")
        for key, count in (
            ("capacities_J_K", 4),
            ("receiver_capacities_L", 2),
            ("pickup_volumes_L", 2),
        ):
            if len(getattr(self, key)) != count:
                raise ValueError(f"{key} has the wrong number of physical owners")
            for value in getattr(self, key):
                number(value, key, positive=True)
        lo, hi = self.properties.temperature_bounds_K
        if (
            not lo
            <= number(self.reference_temperature_K, "reference temperature", positive=True)
            <= hi
        ):
            raise ValueError("body reference temperature is outside the declared interval")
        if (
            type(self.maximum_root_iterations) is not int
            or not 1 <= self.maximum_root_iterations <= 100
        ):
            raise ValueError("declare a root budget of 1 to 100 iterations")
        p = self.properties
        if (
            p.activity_model is not None
            or p.incompressible_model is not None
            or (p.volume_mixing_method not in (None, "Amgat"))
        ):
            raise ValueError("this heat-recovery slice uses original additive H and Amgat volume")
        if any(
            x.component.charge
            or explicit_formula_charge(x.component.formula) not in (None, 0)
            or x.component.default_phase not in {"liquid", "aqueous", "organic"}
            for x in p.packages
        ):
            raise ValueError("heat-recovery slice requires declared neutral single-liquid species")

    def enthalpy(self, index: int, amounts: Mapping[str, float], temperature: float) -> float:
        return self.properties.enthalpy(amounts, temperature) + self.capacities_J_K[index] * (
            temperature - self.reference_temperature_K
        )

    def at_ph(self, index: int, amounts: Mapping[str, float], enthalpy: float) -> float:
        return caloric_temperature(
            lambda t: self.enthalpy(index, amounts, t),
            enthalpy,
            self.properties.temperature_bounds_K,
            energy_basis="enthalpy",
            algorithm="residual",
            heat_capacity_at_temperature=lambda t: (
                self.properties.heat_capacity(amounts, t) + self.capacities_J_K[index]
            ),
        )

    def to_dict(self) -> dict:
        return {
            "schema": "native-finite-heat-recovery-v1",
            **self.__dict__,
            "properties": self.properties.to_dict(),
            "pumps": [p.to_dict() for p in self.pumps],
            "pipes": [p.to_dict() for p in self.pipes],
            "capacities_J_K": list(self.capacities_J_K),
            "receiver_capacities_L": list(self.receiver_capacities_L),
            "pickup_volumes_L": list(self.pickup_volumes_L),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FiniteHeatRecoverySpec:
        if data.get("schema") != "native-finite-heat-recovery-v1":
            raise ValueError("unknown finite heat-recovery model")
        return cls(
            **{
                **{k: v for k, v in data.items() if k != "schema"},
                "properties": LiquidTransportProperties.from_dict(data["properties"]),
                "pumps": tuple(PumpSpec(**p) for p in data["pumps"]),
                "pipes": tuple(PipeSpec(**p) for p in data["pipes"]),
            }
        )
