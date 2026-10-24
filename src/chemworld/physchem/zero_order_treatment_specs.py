"""Declared empirical liquid splits; no inferred pollutant or sludge mechanism."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.elements import explicit_formula_charge, parse_formula
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class ZeroOrderTreatmentStage:
    stage_id: str
    reject_vessel_id: str
    water_recovery: float
    removal_to_concentrate: Mapping[str, float]
    specific_electricity_kWh_m3: float
    source_note: str
    maximum_inlet_concentration_mol_m3: Mapping[str, float]
    reject_capacity_L: float = 1.0
    maximum_inlet_flow_m3_s: float = 20 / 6e7
    model: str = "empirical_single_inlet_dual_outlet"

    def __post_init__(self) -> None:
        if (
            not self.stage_id
            or not self.reject_vessel_id
            or not self.source_note.strip()
            or self.model != "empirical_single_inlet_dual_outlet"
        ):
            raise ValueError("treatment stage requires identities, empirical scope and source")
        if not 0 < finite_number(self.water_recovery, "water recovery") <= 1:
            raise ValueError("water recovery must be in (0, 1]")
        removal = {
            k: finite_number(v, "solute removal") for k, v in self.removal_to_concentrate.items()
        }
        if any(v > 1 for v in removal.values()):
            raise ValueError("removal fractions must be in [0, 1]")
        if self.water_recovery == 1 and any(removal.values()):
            raise ValueError("an aqueous concentrate needs water; no dry removed-solute sink")
        limits = {
            k: finite_number(v, "inlet concentration domain", positive=True)
            for k, v in self.maximum_inlet_concentration_mol_m3.items()
        }
        if set(removal) != set(limits):
            raise ValueError("every removed solute needs an explicit concentration domain")
        for key in ("reject_capacity_L", "maximum_inlet_flow_m3_s"):
            finite_number(getattr(self, key), key, positive=True)
        finite_number(self.specific_electricity_kWh_m3, "additional specific electricity")
        object.__setattr__(self, "removal_to_concentrate", MappingProxyType(removal))
        object.__setattr__(self, "maximum_inlet_concentration_mol_m3", MappingProxyType(limits))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "removal_to_concentrate": dict(self.removal_to_concentrate),
            "maximum_inlet_concentration_mol_m3": dict(self.maximum_inlet_concentration_mol_m3),
        }


@dataclass(frozen=True)
class ZeroOrderTreatmentSpec:
    properties: LiquidTransportProperties
    water_id: str
    stages: tuple[ZeroOrderTreatmentStage, ...]
    temperature_K: float
    product_vessel_id: str = "product"
    product_capacity_L: float = 1.0
    pickup_volume_L: float = 1e-6
    pump: PumpSpec = field(default_factory=PumpSpec)
    pipe: PipeSpec = field(default_factory=lambda: PipeSpec(0.1, 0.002))

    def __post_init__(self) -> None:
        properties = LiquidTransportProperties.from_dict(deepcopy(self.properties.to_dict()))
        object.__setattr__(self, "properties", properties)
        object.__setattr__(self, "stages", tuple(self.stages))
        if (
            properties.activity_model is not None
            or properties.incompressible_model is not None
            or properties.volume_mixing_method not in (None, "Amgat")
        ):
            raise ValueError(
                "zero-order slice requires original additive H and Amgat liquid volume"
            )
        if not 1 <= len(self.stages) <= 16:
            raise ValueError("declare one to sixteen empirical stages")
        names = [self.product_vessel_id, *(s.reject_vessel_id for s in self.stages)]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("product and each concentrate require distinct finite vessels")
        if len({s.stage_id for s in self.stages}) != len(self.stages):
            raise ValueError("stage identities must be unique")
        if self.water_id not in properties.component_ids:
            raise ValueError("water identity is missing from the declared properties")
        if any(
            explicit_formula_charge(p.component.formula) not in (None, 0) or p.component.charge
            for p in properties.packages
        ):
            raise ValueError("this empirical liquid slice does not separate ionic charge")
        if any(
            p.component.default_phase not in {"liquid", "aqueous", "organic"}
            for p in properties.packages
        ):
            raise ValueError("solid/sludge properties require a different declared model")
        water = next(
            p.component for p in properties.packages if p.component.identifier == self.water_id
        )
        if parse_formula(water.formula) != {"H": 2, "O": 1}:
            raise ValueError("the declared water component must be neutral H2O")
        solutes = set(properties.component_ids) - {self.water_id}
        if any(set(stage.removal_to_concentrate) != solutes for stage in self.stages):
            raise ValueError("each stage must explicitly declare every nonwater split")
        finite_number(self.temperature_K, "temperature", positive=True)
        if (
            not properties.temperature_bounds_K[0]
            <= self.temperature_K
            <= (properties.temperature_bounds_K[1])
        ):
            raise ValueError("treatment isotherm lies outside declared property bounds")
        for key in ("product_capacity_L", "pickup_volume_L"):
            finite_number(getattr(self, key), key, positive=True)
        if self.pipe.elevation_change_m != 0:
            raise ValueError("first treatment slice has equal-elevation finite tanks")
        self.molar_terms()

    def molar_terms(self) -> tuple[dict[str, float], dict[str, float]]:
        enthalpies, volumes = {}, {}
        for key in self.properties.component_ids:
            pure = {name: float(name == key) for name in self.properties.component_ids}
            enthalpies[key] = self.properties.enthalpy(pure, self.temperature_K)
            volumes[key] = self.properties.evaluate(pure, self.temperature_K)[2]
        return enthalpies, volumes

    def enthalpy(self, index: int, amounts: Mapping[str, float], temperature: float) -> float:
        return float(self.properties.enthalpy(dict(amounts), temperature))

    def at_ph(self, index: int, amounts: Mapping[str, float], enthalpy: float) -> float:
        return self.temperature_K

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "native-zero-order-treatment-v1",
            "properties": self.properties.to_dict(),
            "water_id": self.water_id,
            "stages": [s.to_dict() for s in self.stages],
            "temperature_K": self.temperature_K,
            "product_vessel_id": self.product_vessel_id,
            "product_capacity_L": self.product_capacity_L,
            "pickup_volume_L": self.pickup_volume_L,
            "pump": self.pump.to_dict(),
            "pipe": self.pipe.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ZeroOrderTreatmentSpec:
        if data.get("schema") != "native-zero-order-treatment-v1":
            raise ValueError("unknown zero-order treatment declaration")
        return cls(
            **{
                **{k: v for k, v in data.items() if k != "schema"},
                "properties": LiquidTransportProperties.from_dict(data["properties"]),
                "stages": tuple(ZeroOrderTreatmentStage(**s) for s in data["stages"]),
                "pump": PumpSpec(**data["pump"]),
                "pipe": PipeSpec(**data["pipe"]),
            }
        )
