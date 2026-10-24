"""Declared neutral suspension, measured jar endpoint and empirical clarifier."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import cached_property
from math import isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties


def finite(value: float, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(
            name
            + " must be an explicit finite "
            + ("positive" if positive else "nonnegative")
            + " number"
        )


@dataclass(frozen=True)
class SuspensionParticleSpec:
    species_id: str
    formula: str
    molar_volume_m3_mol: float
    heat_capacity_J_mol_K: float
    reference_enthalpy_J_mol: float
    reference_temperature_K: float
    source_note: str

    def __post_init__(self) -> None:
        if not self.species_id or not self.source_note.strip():
            raise ValueError("particles require a chemical identity and property source/scope")
        component = ComponentSpec(self.species_id, self.formula, default_phase="solid")
        if component.charge:
            raise ValueError("this suspension slice requires neutral inert particles")
        for key in ("molar_volume_m3_mol", "heat_capacity_J_mol_K", "reference_temperature_K"):
            finite(getattr(self, key), key, positive=True)
        if isinstance(self.reference_enthalpy_J_mol, bool) or not isfinite(
            self.reference_enthalpy_J_mol
        ):
            raise ValueError("particle reference enthalpy must be finite")

    @property
    def molecular_weight_kg_mol(self) -> float:
        value = ComponentSpec(self.species_id, self.formula).molecular_weight_g_mol
        if value is None:
            raise ValueError("particle requires molecular weight")
        return value / 1000

    def enthalpy_J_mol(self, temperature_K: float) -> float:  # noqa: N802
        return self.reference_enthalpy_J_mol + self.heat_capacity_J_mol_K * (
            temperature_K - self.reference_temperature_K
        )


@dataclass(frozen=True)
class JarTestEndpointSpec:
    initial_ntu: float
    final_ntu: float
    slope_mg_L_ntu: float
    intercept_mg_L: float
    source_note: str

    def __post_init__(self) -> None:
        for key in ("initial_ntu", "final_ntu", "intercept_mg_L"):
            finite(getattr(self, key), key)
        finite(self.slope_mg_L_ntu, "TSS/NTU slope", positive=True)
        if self.final_ntu > self.initial_ntu or not self.source_note.strip():
            raise ValueError("declare a non-increasing measured jar endpoint and its scope")

    @property
    def initial_tss_kg_m3(self) -> float:
        return (self.slope_mg_L_ntu * self.initial_ntu + self.intercept_mg_L) * 1e-3

    @property
    def final_tss_kg_m3(self) -> float:
        return (self.slope_mg_L_ntu * self.final_ntu + self.intercept_mg_L) * 1e-3


@dataclass(frozen=True)
class CoagFlocMixingSpec:
    rapid_retention_s: float
    rapid_basins: int
    rapid_velocity_gradient_s_inv: float
    floc_retention_s: float
    paddle_length_m: float
    paddle_width_m: float
    paddle_rotation_Hz: float
    paddle_drag_coefficient: float
    paddle_velocity_fraction: float
    paddle_wheels: int
    paddles_per_wheel: int

    def __post_init__(self) -> None:
        for key in ("rapid_basins", "paddle_wheels", "paddles_per_wheel"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError("mixing equipment counts must be positive integers")
        for key in (
            "rapid_retention_s",
            "floc_retention_s",
            "paddle_length_m",
            "paddle_width_m",
            "paddle_drag_coefficient",
            "paddle_velocity_fraction",
        ):
            finite(getattr(self, key), key, positive=True)
        for key in ("rapid_velocity_gradient_s_inv", "paddle_rotation_Hz"):
            finite(getattr(self, key), key)
        if self.paddle_velocity_fraction > 1:
            raise ValueError("paddle fluid-velocity fraction must not exceed one")


@dataclass(frozen=True)
class ClarifierSplitSpec:
    dispersed_capture: float
    floc_capture: float
    sludge_liquid_fraction: float
    source_note: str

    def __post_init__(self) -> None:
        for key in ("dispersed_capture", "floc_capture", "sludge_liquid_fraction"):
            finite(getattr(self, key), key)
            if getattr(self, key) > 1:
                raise ValueError("declared separator fractions cannot exceed one")
        if not 0 < self.sludge_liquid_fraction < 1 or not self.source_note.strip():
            raise ValueError("both finite products require actual liquid and a split scope")


@dataclass(frozen=True)
class SuspensionMaterialSpec:
    liquid: LiquidTransportProperties
    particle: SuspensionParticleSpec
    water_id: str
    viscosity_Pa_s: float
    maximum_solid_volume_fraction: float
    viscosity_source_note: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "liquid", LiquidTransportProperties.from_dict(self.liquid.to_dict())
        )
        prop = self.liquid
        if (
            prop.activity_model is not None
            or prop.incompressible_model is not None
            or prop.volume_mixing_method not in (None, "Amgat")
            or self.water_id not in prop.component_ids
            or self.particle.species_id in prop.component_ids
        ):
            raise ValueError("suspension requires distinct neutral liquid and particle identities")
        for package in prop.packages:
            component = package.component
            density = package.by_property("liquid_density")
            if (
                component.charge
                or component.molecular_weight_g_mol is None
                or component.default_phase not in {"liquid", "aqueous"}
                or len(density) != 1
                or density[0].equation_id != "linear_liquid_density"
                or density[0].coefficients.get("alpha") != 0
            ):
                raise ValueError(
                    "first suspension slice requires neutral constant-volume carrier properties"
                )
            if component.identifier == self.water_id and dict(component.composition) != {
                "H": 2,
                "O": 1,
            }:
                raise ValueError("declared water must have H2O composition")
        finite(self.viscosity_Pa_s, "declared suspension viscosity", positive=True)
        finite(self.maximum_solid_volume_fraction, "solid volume domain", positive=True)
        if self.maximum_solid_volume_fraction >= 1 or not self.viscosity_source_note.strip():
            raise ValueError(
                "suspension viscosity requires a finite wet-suspension domain and source"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "liquid": self.liquid.to_dict(),
            "particle": asdict(self.particle),
            "water_id": self.water_id,
            "viscosity_Pa_s": self.viscosity_Pa_s,
            "maximum_solid_volume_fraction": self.maximum_solid_volume_fraction,
            "viscosity_source_note": self.viscosity_source_note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SuspensionMaterialSpec":
        return cls(
            **{
                **raw,
                "liquid": LiquidTransportProperties.from_dict(raw["liquid"]),
                "particle": SuspensionParticleSpec(**raw["particle"]),
            }
        )


@dataclass(frozen=True)
class SuspensionTankInventory:
    liquid_mol: Mapping[str, float]
    dispersed_mol: float = 0.0
    floc_mol: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "liquid_mol", MappingProxyType(dict(self.liquid_mol)))
        for value in (*self.liquid_mol.values(), self.dispersed_mol, self.floc_mol):
            finite(value, "actual suspension inventory")


@dataclass(frozen=True)
class SuspensionTreatmentSpec:
    material: SuspensionMaterialSpec
    jar_test: JarTestEndpointSpec
    mixing: CoagFlocMixingSpec
    clarifier: ClarifierSplitSpec
    bath: ThermalDeviceSpec
    pump: PumpSpec = field(default_factory=PumpSpec)
    pipe: PipeSpec = field(default_factory=lambda: PipeSpec(0.1, 0.002))
    capacities_L: tuple[float, float, float] = (0.1, 0.1, 0.1)
    pickup_volume_L: float = 0.001
    product_vessel_id: str = "clarified"
    sludge_vessel_id: str = "sludge"

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacities_L", tuple(self.capacities_L))
        if len(self.capacities_L) != 3:
            raise ValueError("finite suspension requires three capacities")
        for value in (*self.capacities_L, self.pickup_volume_L):
            finite(value, "finite vessel capacity/pickup", positive=True)
        if (
            self.pickup_volume_L >= self.capacities_L[0]
            or not self.product_vessel_id
            or not self.sludge_vessel_id
            or len({"reactor", self.product_vessel_id, self.sludge_vessel_id}) != 3
            or self.pipe.elevation_change_m != 0
            or not self.bath.enabled
            or self.bath.mode != "isothermal"
            or not self.material.liquid.temperature_bounds_K[0]
            <= self.temperature_K
            <= self.material.liquid.temperature_bounds_K[1]
        ):
            raise ValueError(
                "declare distinct finite tanks and a common isothermal, equal-elevation boundary"
            )

    @property
    def temperature_K(self) -> float:  # noqa: N802
        return self.bath.setpoint_K

    @property
    def pressure_Pa(self) -> float:  # noqa: N802
        return self.material.liquid.pressure_Pa

    @property
    def phase_keys(self) -> tuple[str, ...]:
        return (*("liquid:" + k for k in self.material.liquid.component_ids), "dispersed", "floc")

    @cached_property
    def molar_volumes(self) -> dict[str, float]:
        liquid = self.material.liquid
        return {
            **{
                "liquid:" + k: liquid.evaluate(
                    {j: float(j == k) for j in liquid.component_ids}, self.temperature_K
                )[2]
                for k in liquid.component_ids
            },
            "dispersed": self.material.particle.molar_volume_m3_mol,
            "floc": self.material.particle.molar_volume_m3_mol,
        }

    @cached_property
    def molar_enthalpies(self) -> dict[str, float]:
        liquid = self.material.liquid
        h = self.material.particle.enthalpy_J_mol(self.temperature_K)
        return {
            **{
                "liquid:" + k: liquid.enthalpy(
                    {j: float(j == k) for j in liquid.component_ids}, self.temperature_K
                )
                for k in liquid.component_ids
            },
            "dispersed": h,
            "floc": h,
        }

    @cached_property
    def molecular_weights(self) -> dict[str, float]:
        mw = self.material.particle.molecular_weight_kg_mol
        return {
            **{
                "liquid:" + p.component.identifier: float(p.component.molecular_weight_g_mol or 0)
                / 1000
                for p in self.material.liquid.packages
            },
            "dispersed": mw,
            "floc": mw,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "finite-jar-conditioned-suspension-v1",
            "material": self.material.to_dict(),
            "jar_test": asdict(self.jar_test),
            "mixing": asdict(self.mixing),
            "clarifier": asdict(self.clarifier),
            "bath": self.bath.to_dict(),
            "pump": asdict(self.pump),
            "pipe": asdict(self.pipe),
            "capacities_L": list(self.capacities_L),
            "pickup_volume_L": self.pickup_volume_L,
            "product_vessel_id": self.product_vessel_id,
            "sludge_vessel_id": self.sludge_vessel_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SuspensionTreatmentSpec":
        data = dict(raw)
        if data.pop("schema", None) != "finite-jar-conditioned-suspension-v1":
            raise ValueError("unknown finite suspension declaration")
        return cls(
            **{
                **data,
                "material": SuspensionMaterialSpec.from_dict(data["material"]),
                "jar_test": JarTestEndpointSpec(**data["jar_test"]),
                "mixing": CoagFlocMixingSpec(**data["mixing"]),
                "clarifier": ClarifierSplitSpec(**data["clarifier"]),
                "bath": ThermalDeviceSpec.from_dict(data["bath"]),
                "pump": PumpSpec(**data["pump"]),
                "pipe": PipeSpec(**data["pipe"]),
            }
        )
