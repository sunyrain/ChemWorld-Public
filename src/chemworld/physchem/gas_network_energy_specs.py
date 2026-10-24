"""Thermal declarations for the existing finite gas nodes, not new inventories."""

from dataclasses import dataclass
from math import isfinite


def positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class GasNodeEnergySpec:
    """Gas and its declared vessel support have one actual node temperature."""

    vessel_id: str
    initial_temperature_K: float
    support_heat_capacity_J_K: float
    reference_temperature_K: float = 298.15
    environment_temperature_K: float = 300.0
    loss_conductance_W_K: float = 0.0
    minimum_temperature_K: float = 200.0
    maximum_temperature_K: float = 1500.0

    def __post_init__(self) -> None:
        if not isinstance(self.vessel_id, str) or not self.vessel_id:
            raise ValueError("gas node energy needs a vessel identity")
        for name in (
            "initial_temperature_K",
            "support_heat_capacity_J_K",
            "reference_temperature_K",
            "environment_temperature_K",
            "minimum_temperature_K",
            "maximum_temperature_K",
        ):
            positive(getattr(self, name), name)
        if (
            not self.minimum_temperature_K < self.initial_temperature_K < self.maximum_temperature_K
            or isinstance(self.loss_conductance_W_K, bool)
            or not isfinite(self.loss_conductance_W_K)
            or self.loss_conductance_W_K < 0
        ):
            raise ValueError("node requires interior initial temperature and nonnegative loss law")


@dataclass(frozen=True)
class GasThermalLinkSpec:
    """Zero-holdup conductive interface; heat has equal opposite endpoints."""

    connection_id: str
    source_vessel_id: str
    target_vessel_id: str
    conductance_W_K: float

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(v, str) or not v
                for v in (self.connection_id, self.source_vessel_id, self.target_vessel_id)
            )
            or self.source_vessel_id == self.target_vessel_id
        ):
            raise ValueError("heat link needs named distinct node endpoints")
        if (
            isinstance(self.conductance_W_K, bool)
            or not isfinite(self.conductance_W_K)
            or self.conductance_W_K < 0
        ):
            raise ValueError("heat-link conductance must be finite and nonnegative")
