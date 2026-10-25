"""Finite complete ASM1 workflow declarations and its explicit quality basis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from math import isfinite
from typing import Any

from chemworld.physchem.activated_sludge_kinetics import (
    ASM1_COMPONENTS,
    COMPONENT_BASES,
    COMPONENT_UNITS,
    EXTENT_BASES,
    ASM1Parameters,
    asm1_stoichiometry,
)
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties

ASM1_SCHEMA = "finite-asm1-activated-sludge-v1"
ASM1_MECHANISM = "asm1-watertap-1.8.0"
ASM1_EQUIPMENT = "finite_asm1_activated_sludge"
ASM1_MATERIAL = "asm1_components_v1"
ASM1_EXTENTS = "asm1_reaction_progress_v1"
ASM1_UTILITIES = "asm1_utilities_v1"
UTILITY_FIELDS = ("oxygen_supply_kg", "oxygen_offgas_kg", "nitrogen_offgas_kg")


@dataclass(frozen=True)
class ActivatedSludgeSpec:
    """One or all five ASM1 CSTRs plus the complete two-recycle/clarifier route.

    Clarifier splits are empirical source specifications, not a settling model.
    All original quality pools persist even in a single-reactor configuration.
    The carrier is the source constant-density water approximation; empirical
    organics do not pretend to add molecular volume, charge or formation heat.
    """

    parameters: ASM1Parameters = field(default_factory=ASM1Parameters)
    reactor_volumes_m3: tuple[float, ...] = (1e-5, 1e-5, 1.333e-5, 1.333e-5, 1.333e-5)
    kla_per_hour: tuple[float, ...] = (0.0, 0.0, 10.0, 10.0, 3.5)
    feed_capacity_m3: float = 0.001
    product_capacity_m3: float = 0.001
    sludge_capacity_m3: float = 0.0002
    feed_pickup_m3: float = 1e-7
    internal_recycle_fraction: float = 0.6
    clarifier_soluble_effluent_fraction: float = 0.48956
    clarifier_particulate_effluent_fraction: float = 0.00187
    sludge_recycle_fraction: float = 0.97955
    oxygen_equilibrium_kg_m3: float = 0.008
    oxygen_supply_capacity_kg: float = 0.001
    oxygen_offgas_capacity_kg: float = 0.001
    nitrogen_offgas_capacity_kg: float = 0.001
    aeration_power_limit_W: float = 10.0
    maximum_aeration_energy_J: float = 1e8
    electricity_mode: str = "calculated"
    fixed_electricity_kWh_m3: float = 0.0108
    pump: PumpSpec = field(default_factory=lambda: PumpSpec(1e5, 10.0, 0.7))
    pipe: PipeSpec = field(default_factory=lambda: PipeSpec(0.1, 0.002))
    recycle_pump: PumpSpec = field(default_factory=lambda: PumpSpec(1e5, 10.0, 0.7))
    recycle_pipe: PipeSpec = field(default_factory=lambda: PipeSpec(0.1, 0.002))
    temperature_K: float = 298.15
    pressure_Pa: float = 101325.0
    carrier_viscosity_Pa_s: float = 0.001
    maximum_bath_cooling_W: float = 100.0
    rtol: float = 1e-9
    atol: float = 1e-11
    maximum_step_s: float = 300.0

    def __post_init__(self) -> None:
        for key in ("reactor_volumes_m3", "kla_per_hour"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if (
            not isinstance(self.parameters, ASM1Parameters)
            or len(self.reactor_volumes_m3) not in (1, 5)
            or len(self.kla_per_hour) != len(self.reactor_volumes_m3)
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0 for v in self.reactor_volumes_m3
            )
            or any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in self.kla_per_hour)
        ):
            raise ValueError("complete ASM1 requires one or five finite reactor/kLa declarations")
        positive = (
            self.feed_capacity_m3,
            self.product_capacity_m3,
            self.sludge_capacity_m3,
            self.feed_pickup_m3,
            self.oxygen_equilibrium_kg_m3,
            self.oxygen_supply_capacity_kg,
            self.oxygen_offgas_capacity_kg,
            self.nitrogen_offgas_capacity_kg,
            self.maximum_aeration_energy_J,
            self.temperature_K,
            self.pressure_Pa,
            self.carrier_viscosity_Pa_s,
            self.maximum_bath_cooling_W,
            self.rtol,
            self.atol,
            self.maximum_step_s,
        )
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in positive):
            raise ValueError(
                "ASM1 capacities, source geometry and numerical bounds must be positive"
            )
        if (
            self.feed_pickup_m3 >= self.feed_capacity_m3
            or not 273.15 <= self.temperature_K <= 323.15
            or self.electricity_mode not in {"none", "fixed", "calculated"}
            or self.pipe.elevation_change_m != 0
            or self.recycle_pipe.elevation_change_m != 0
        ):
            raise ValueError("ASM1 source geometry, fixed T/P or electricity policy invalid")
        for value in (self.internal_recycle_fraction, self.sludge_recycle_fraction):
            if isinstance(value, bool) or not isfinite(value) or not 0 <= value < 1:
                raise ValueError("ASM1 recycles require fractions in [0,1)")
        if (
            not 0 < self.clarifier_soluble_effluent_fraction < 1
            or not 0 <= self.clarifier_particulate_effluent_fraction <= 1
            or any(
                isinstance(v, bool) or not isfinite(v) or v < 0
                for v in (self.aeration_power_limit_W, self.fixed_electricity_kWh_m3)
            )
        ):
            raise ValueError("ASM1 separation fractions or aeration powers invalid")

    @property
    def reactor_vessel_ids(self) -> tuple[str, ...]:
        return tuple(f"asm1_{i + 1}" for i in range(len(self.reactor_volumes_m3)))

    @property
    def member_vessel_ids(self) -> tuple[str, ...]:
        return ("reactor", *self.reactor_vessel_ids, "effluent", "sludge")

    @property
    def capacities_m3(self) -> tuple[float, ...]:
        return (
            self.feed_capacity_m3,
            *self.reactor_volumes_m3,
            self.product_capacity_m3,
            self.sludge_capacity_m3,
        )

    @property
    def carrier_density_kg_m3(self) -> float:
        return 997.0

    @cached_property
    def water_component(self) -> ComponentSpec:
        return ComponentSpec("H2O", "H2O")

    @property
    def water_molar_volume_m3(self) -> float:
        weight = self.water_component.molecular_weight_g_mol
        if weight is None:
            raise ValueError("ASM1 requires real carrier water molecular weight")
        return weight / 1000 / self.carrier_density_kg_m3

    @cached_property
    def properties(self) -> LiquidTransportProperties:
        correlations = (
            (
                "liquid_heat_capacity",
                "cp_polynomial",
                {"a": 4182 * self.water_molar_volume_m3 * self.carrier_density_kg_m3},
                "J/(mol*K)",
            ),
            (
                "liquid_density",
                "linear_liquid_density",
                {"rho_ref": 997.0, "T_ref": 298.15, "alpha": 0.0},
                "kg/m^3",
            ),
            (
                "liquid_viscosity",
                "andrade_viscosity",
                {"A": self.carrier_viscosity_Pa_s, "B": 0.0},
                "Pa*s",
            ),
            (
                "thermal_conductivity",
                "linear_thermal_conductivity",
                {"k_ref": 0.6, "T_ref": 298.15, "slope": 0.0},
                "W/(m*K)",
            ),
        )
        package = ComponentPropertyPackage(
            self.water_component,
            tuple(
                PropertyCorrelation(
                    f"asm1-carrier-{name}",
                    equation,
                    coefficients,
                    {"temperature": "K"},
                    unit,
                    property_id=name,
                    validity_ranges={"temperature": (273.15, 323.15)},
                    metadata={"phase": "liquid"},
                    source_note="ASM1 constant water rho/Cp; declared viscosity/conductivity",
                )
                for name, equation, coefficients, unit in correlations
            ),
        )
        return LiquidTransportProperties(
            (package,),
            (273.15, 323.15),
            self.pressure_Pa,
            caloric_references={
                "H2O": CaloricReference(
                    phase="liquid",
                    pressure_Pa=self.pressure_Pa,
                    source_note="ASM1 water sensible H; biochemical thermochemistry unspecified",
                )
            },
        )

    @property
    def network(self) -> ReactionNetworkSpec:
        return ReactionNetworkSpec("asm1-real-carrier-water", (SpeciesSpec("H2O", "H2O"),), ())

    def balance_declaration(self) -> dict[str, Any]:
        matrix = asm1_stoichiometry(self.parameters)
        matrix["R2"]["N2_N"] = (1 - self.parameters.Y_H) / (2.86 * self.parameters.Y_H)
        coefficients = {
            **dict.fromkeys(("S_NO", "S_NH", "S_ND", "X_ND", "N2_N"), 1.0),
            **dict.fromkeys(("X_BH", "X_BA"), self.parameters.i_xb),
            **dict.fromkeys(("X_P", "X_I"), self.parameters.i_xp),
        }
        return {
            "schema": "declared-empirical-reaction-balance-v1",
            "mechanism_id": ASM1_MECHANISM,
            "member_vessel_ids": list(self.member_vessel_ids),
            "quantities": {
                **{
                    k: {"unit": COMPONENT_UNITS[k], "basis": COMPONENT_BASES[k]}
                    for k in ASM1_COMPONENTS
                },
                "N2_N": {"unit": "kg", "basis": "N"},
            },
            "reactions": {
                r: {"unit": "kg", "basis": EXTENT_BASES[r], "stoichiometry": row}
                for r, row in matrix.items()
            },
            "inventory_fields": {
                ASM1_MATERIAL: {k: k for k in ASM1_COMPONENTS},
                ASM1_UTILITIES: {
                    "oxygen_supply_kg": "S_O",
                    "oxygen_offgas_kg": "S_O",
                    "nitrogen_offgas_kg": "N2_N",
                },
            },
            "extent_state_type": ASM1_EXTENTS,
            "invariants": {"total_N": {"unit": "kg", "coefficients": coefficients}},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ASM1_SCHEMA,
            **asdict(self),
            "reactor_volumes_m3": list(self.reactor_volumes_m3),
            "kla_per_hour": list(self.kla_per_hour),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActivatedSludgeSpec:
        values = dict(payload)
        if values.pop("schema", None) != ASM1_SCHEMA or set(values) != {
            f.name for f in fields(cls)
        }:
            raise ValueError("ASM1 workflow requires its exact complete model schema")
        values["parameters"] = ASM1Parameters.from_dict(values["parameters"])
        for name in ("pump", "recycle_pump"):
            values[name] = PumpSpec(**values[name])
        for name in ("pipe", "recycle_pipe"):
            values[name] = PipeSpec(**values[name])
        return cls(**values)
