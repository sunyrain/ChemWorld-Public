"""Finite single-phase tank transport on the shared n/H integrator.

Pipe and exchanger have negligible holdup. Reservoirs share pressure/elevation;
hydraulic dissipation enters liquid H and shaft inefficiency leaves to ambient.
The utility is an explicit fixed-temperature thermal boundary.
"""

from dataclasses import asdict, dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

import numpy as np

from chemworld.physchem.differential_meters import (
    DifferentialMeterSpec,
    liquid_differential_meter_pressure,
)
from chemworld.physchem.enthalpy import caloric_temperature
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.flow_meters import LiquidOrificeSpec, liquid_orifice_pressure
from chemworld.physchem.fluids_devices import LiquidControlValveSpec, liquid_valve_pressure_drop
from chemworld.physchem.incompressible_properties import IncompressibleMaterialModel
from chemworld.physchem.packed_beds import PackedBedSpec, packed_bed_pressure
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.transport import (
    FluidState,
    PipeSpec,
    PumpSpec,
    counterflow_effectiveness,
    internal_heat_transfer_coefficient,
    nusselt_internal_flow,
    overall_heat_transfer_coefficient,
    pipe_pressure_drop,
    prandtl_number,
    pump_operating_point,
)
from chemworld.physchem.transport_properties import liquid_transport_state


@dataclass(frozen=True)
class LiquidTransportProperties:
    packages: tuple[ComponentPropertyPackage, ...]
    temperature_bounds_K: tuple[float, float] = (280.0, 340.0)
    pressure_Pa: float = 101325.0
    caloric_references: dict[str, CaloricReference] | None = None
    viscosity_pressure_parameters: dict[str, dict[str, float]] | None = None
    volume_mixing_method: str | None = None
    volume_mixing_parameters: dict[str, dict[str, float]] | None = None
    volume_pressure_parameters: dict[str, float | str] | None = None
    volume_mixing_note: str = ""
    activity_model: ActivityModelSpec | None = None
    incompressible_model: IncompressibleMaterialModel | None = None

    def __post_init__(self):
        if not self.packages or len(set(self.component_ids)) != len(self.packages):
            raise ValueError("liquid transport needs distinct component packages")
        if self.activity_model is not None and (
            not isinstance(self.activity_model, ActivityModelSpec)
            or set(self.activity_model.component_ids) != set(self.component_ids)
        ):
            raise ValueError("transport activity model must match declared component identities")
        lo, hi = self.temperature_bounds_K
        if (
            not all(isfinite(x) for x in (lo, hi, self.pressure_Pa))
            or not 0 < lo < hi
            or self.pressure_Pa <= 0
        ):
            raise ValueError("liquid transport requires finite positive T/P bounds")
        if self.incompressible_model is not None:
            model = self.incompressible_model
            if not isinstance(model, IncompressibleMaterialModel):
                raise TypeError("incompressible transport requires an explicit material model")
            model.validate_components(tuple(p.component for p in self.packages))
            if (self.caloric_references or self.activity_model is not None
                    or self.viscosity_pressure_parameters or self.volume_mixing_method is not None
                    or self.volume_mixing_parameters or self.volume_pressure_parameters
                    or self.volume_mixing_note):
                raise ValueError("incompressible transport uses its own exclusive property model")
            if not model.fluid.temperature_bounds_K[0] <= lo < hi <= (
                model.fluid.temperature_bounds_K[1]
            ):
                raise ValueError("transport temperature interval exceeds incompressible source")
            for name in ("caloric_references", "viscosity_pressure_parameters",
                         "volume_mixing_parameters", "volume_pressure_parameters"):
                object.__setattr__(self, name, MappingProxyType({}))
            return
        references = self.caloric_references
        if references is None:
            references = {k: CaloricReference(phase="liquid") for k in self.component_ids}
        if set(references) != set(self.component_ids) or any(
            ref.phase != "liquid" or ref.energy_basis != "enthalpy"
            or ref.pressure_Pa != self.pressure_Pa
            for ref in references.values()
        ):
            raise ValueError("liquid transport needs matching liquid H references at its pressure")
        object.__setattr__(self, "caloric_references", MappingProxyType(dict(references)))
        parameters = self.viscosity_pressure_parameters or {}
        fields = {"critical_temperature_K", "critical_pressure_Pa", "acentric_factor"}
        if set(parameters) - set(self.component_ids) or any(
            set(p) != fields or any(
                isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v)
                for v in p.values()
            )
            or p["critical_temperature_K"] <= 0 or p["critical_pressure_Pa"] <= 0
            for p in parameters.values()
        ):
            raise ValueError("Lucas parameters require declared components and finite Tc/Pc/omega")
        object.__setattr__(self, "viscosity_pressure_parameters", MappingProxyType({
            key: MappingProxyType(dict(p)) for key, p in parameters.items()
        }))
        from chemworld.physchem.volume_mixtures import MIXTURE_VOLUME_METHODS

        method = self.volume_mixing_method or "Amgat"
        if method not in MIXTURE_VOLUME_METHODS:
            raise ValueError("unknown transport volume mixing method")
        volume_parameters = self.volume_mixing_parameters or {}
        pressure_parameters = self.volume_pressure_parameters or {}
        if method == "Amgat":
            if volume_parameters or pressure_parameters:
                raise ValueError("Amgat transport uses current pure volumes, not frozen parameters")
        else:
            fields = set(MIXTURE_VOLUME_METHODS[method]) - {"MWs"}
            if not self.volume_mixing_note.strip():
                raise ValueError("explicit volume mixing needs a source/scope note")
            if set(volume_parameters) != set(self.component_ids) or any(
                set(p) != fields for p in volume_parameters.values()
            ):
                raise ValueError("volume mixing parameters must match components and method fields")
            if "P" in pressure_parameters:
                raise ValueError("transport pressure belongs to the declared state")
            # Validate the selected law at the existing application's two
            # endpoints. Runtime evaluates each actual composition independently.
            from chemworld.physchem.specs import MixtureSpec
            from chemworld.physchem.volume_mixtures import mixture_volume_evaluation

            for temperature in self.temperature_bounds_K:
                mixture = MixtureSpec.from_mole_fractions(
                    tuple(p.component for p in self.packages),
                    (1 / len(self.packages),) * len(self.packages),
                    phase_label="liquid", temperature_K=temperature, pressure_Pa=self.pressure_Pa,
                )
                mixture_volume_evaluation(mixture, method=method,
                    component_parameters=volume_parameters, **pressure_parameters)
        object.__setattr__(self, "volume_mixing_parameters", MappingProxyType({
            key: MappingProxyType(dict(p)) for key, p in volume_parameters.items()
        }))
        object.__setattr__(self, "volume_pressure_parameters", MappingProxyType(
            dict(pressure_parameters)))

    @property
    def component_ids(self):
        return tuple(p.component.identifier for p in self.packages)

    @property
    def references(self):
        return self.caloric_references

    def enthalpy(self, amounts, temperature):
        return self._caloric_quantity(amounts, temperature, "enthalpy")

    def heat_capacity(self, amounts, temperature):
        """dH/dT at fixed liquid amounts and the same declared pressure/reference."""
        return self._caloric_quantity(amounts, temperature, "cp")

    def _caloric_quantity(self, amounts, temperature, quantity):
        if (
            set(amounts) != set(self.component_ids)
            or any(not isfinite(n) or n < 0 for n in amounts.values())
            or not self.temperature_bounds_K[0] <= temperature <= self.temperature_bounds_K[1]
        ):
            raise ValueError("liquid caloric request leaves its inventory or temperature domain")
        if not sum(amounts.values()):
            return 0.0
        report = evaluate_property_state(
            self.packages,
            amounts,
            temperature_K=temperature,
            pressure_Pa=self.pressure_Pa,
            phase="liquid",
            quantities=(quantity,),
            references=self.references,
            activity_model=self.activity_model,
            incompressible_model=self.incompressible_model,
        )
        return sum(amounts.values()) * report.properties[quantity].value

    def evaluate(self, amounts, temperature):
        if not self.temperature_bounds_K[0] <= temperature <= self.temperature_bounds_K[1]:
            raise ValueError("liquid transport request leaves its temperature domain")
        return liquid_transport_state(
            self.packages, amounts, temperature, self.pressure_Pa, self.references,
            viscosity_pressure_parameters=self.viscosity_pressure_parameters,
            volume_mixing_method=self.volume_mixing_method,
            volume_mixing_parameters=self.volume_mixing_parameters,
            volume_pressure_parameters=self.volume_pressure_parameters,
            activity_model=self.activity_model,
            incompressible_model=self.incompressible_model,
        )

    def to_dict(self):
        return {
            "packages": [p.to_dict() for p in self.packages],
            "temperature_bounds_K": list(self.temperature_bounds_K),
            "pressure_Pa": self.pressure_Pa,
            "caloric_references": {key: ref.to_dict() for key, ref in self.references.items()},
            "viscosity_pressure_parameters": {
                key: dict(p) for key, p in self.viscosity_pressure_parameters.items()
            },
            "volume_mixing_method": self.volume_mixing_method,
            "volume_mixing_parameters": {
                key: dict(p) for key, p in self.volume_mixing_parameters.items()
            },
            "volume_pressure_parameters": dict(self.volume_pressure_parameters),
            "volume_mixing_note": self.volume_mixing_note,
            **({"activity_model": self.activity_model.to_dict()}
               if self.activity_model is not None else {}),
            **({"incompressible_model": self.incompressible_model.to_dict()}
               if self.incompressible_model is not None else {}),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            tuple(
                ComponentPropertyPackage.from_dict(p)
                for p in value["packages"]
            ),
            tuple(value["temperature_bounds_K"]),
            value["pressure_Pa"],
            ({key: CaloricReference(**ref) for key, ref in value["caloric_references"].items()}
             if "caloric_references" in value else None),
            value.get("viscosity_pressure_parameters"),
            value.get("volume_mixing_method"),
            value.get("volume_mixing_parameters"),
            value.get("volume_pressure_parameters"),
            value.get("volume_mixing_note", ""),
            (ActivityModelSpec(**value["activity_model"])
             if value.get("activity_model") is not None else None),
            (IncompressibleMaterialModel.from_dict(value["incompressible_model"])
             if value.get("incompressible_model") is not None else None),
        )


@dataclass(frozen=True)
class TransportPathSpec:
    pipe: PipeSpec = field(default_factory=lambda: PipeSpec(2.0, 0.001))
    bed_length_m: float = 0.0
    bed_area_m2: float = 1e-4
    particle_diameter_m: float = 0.001
    void_fraction: float = 0.4
    exchanger_area_m2: float = 0.001
    utility_temperature_K: float = 303.15
    outer_h_W_m2_K: float = 500.0
    wall_thickness_m: float = 0.001
    wall_conductivity_W_m_K: float = 15.0
    control_valve: LiquidControlValveSpec | None = None
    orifice_meter: LiquidOrificeSpec | None = None
    bed_method: str = "Ergun"
    bed_tube_diameter_m: float | None = None
    bed_sphericity: float = 1.0
    differential_meter: DifferentialMeterSpec | None = None

    def __post_init__(self):
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.pop("pipe")
        values.pop("control_valve")
        values.pop("orifice_meter")
        values.pop("bed_method")
        values.pop("bed_tube_diameter_m")
        values.pop("differential_meter")
        if any(not isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("path parameters must be finite and nonnegative")
        if self.pipe.elevation_change_m != 0:
            raise ValueError("finite tank path requires equal elevation")
        if (
            min(
                self.bed_area_m2,
                self.particle_diameter_m,
                self.outer_h_W_m2_K,
                self.wall_conductivity_W_m_K,
                self.utility_temperature_K,
            )
            <= 0
            or not 0.05 < self.void_fraction < 0.95
        ):
            raise ValueError("path geometry and utility domain are invalid")
        PackedBedSpec(self.bed_method, self.bed_length_m or 1., self.particle_diameter_m,
                      self.void_fraction, self.bed_sphericity, self.bed_tube_diameter_m)
        if self.control_valve is not None and (
            self.control_valve.outlet_diameter_m != self.pipe.diameter_m
        ):
            raise ValueError("valve outlet must match the connected pipe diameter")
        if self.orifice_meter is not None and (
            self.orifice_meter.pipe_diameter_m != self.pipe.diameter_m
        ):
            raise ValueError("orifice pipe bore must match the connected pipe")
        if self.differential_meter is not None:
            meter = self.differential_meter
            if meter.pipe_diameter_m != self.pipe.diameter_m:
                raise ValueError("differential meter bore must match the connected pipe")
            if meter.reynolds_bounds is None or meter.installation_note == (
                "Formula evaluation; installation validity not qualified"
            ):
                raise ValueError(
                    "native meter path requires explicit Reynolds and installation scope"
                )
            if meter.specified_expansibility not in (None, 1):
                raise ValueError("single-liquid path requires unit expansibility")
            if meter.permanent_loss(1, 0.7) is None:
                raise ValueError("native meter path requires an explicit permanent-loss relation")

    def hydraulic_diagnostics(
        self, fluid: FluidState, flow: float, *, downstream_pressure_Pa: float | None = None,
        saturation_pressure_Pa: float | None = None, critical_pressure_Pa: float | None = None,
        check_choking: bool = True,
    ) -> dict[str, Any]:
        pressure = pipe_pressure_drop(
            self.pipe, fluid, volumetric_flow_m3_s=flow
        ).pressure_drop_total_Pa
        diagnostics: dict[str, Any] = {"pipe_pressure_drop_Pa": pressure}
        if self.bed_length_m:
            bed = packed_bed_pressure(
                PackedBedSpec(self.bed_method, self.bed_length_m, self.particle_diameter_m,
                              self.void_fraction, self.bed_sphericity, self.bed_tube_diameter_m),
                fluid, superficial_velocity_m_s=flow / self.bed_area_m2,
            )
            diagnostics["packed_bed"] = bed.to_dict()
            pressure += bed.pressure_drop_Pa
        diagnostics["pipe_and_bed_pressure_drop_Pa"] = pressure
        if self.orifice_meter is not None:
            meter = liquid_orifice_pressure(self.orifice_meter,
                mass_flow_kg_s=flow * fluid.density_kg_m3,
                density_kg_m3=fluid.density_kg_m3, viscosity_Pa_s=fluid.viscosity_Pa_s)
            diagnostics["orifice"] = meter.to_dict()
            pressure += meter.permanent_pressure_loss_Pa
        if self.differential_meter is not None:
            differential = liquid_differential_meter_pressure(
                self.differential_meter, fluid, mass_flow_kg_s=flow * fluid.density_kg_m3,
            )
            diagnostics["differential_meter"] = differential.to_dict()
            pressure += differential.permanent_pressure_loss_Pa
        if self.control_valve is not None:
            if downstream_pressure_Pa is None or saturation_pressure_Pa is None or (
                critical_pressure_Pa is None
            ):
                raise ValueError("installed IEC valve needs declared current liquid pressures")
            valve = liquid_valve_pressure_drop(self.control_valve, fluid, flow_m3_s=flow,
                downstream_pressure_Pa=downstream_pressure_Pa + pressure,
                saturation_pressure_Pa=saturation_pressure_Pa,
                critical_pressure_Pa=critical_pressure_Pa, check_choking=check_choking)
            diagnostics["control_valve"] = valve.to_dict()
            pressure += valve.pressure_drop_Pa
        diagnostics["total_pressure_drop_Pa"] = pressure
        return diagnostics

    def pressure_drop(self, fluid, flow, **kwargs):
        return self.hydraulic_diagnostics(fluid, flow, **kwargs)["total_pressure_drop_Pa"]

    def heat_rate(self, fluid, flow, temperature):
        if flow == 0 or self.exchanger_area_m2 == 0:
            return 0.0
        re = (
            fluid.density_kg_m3
            * flow
            / self.pipe.area_m2
            * self.pipe.diameter_m
            / fluid.viscosity_Pa_s
        )
        pr = prandtl_number(
            viscosity_Pa_s=fluid.viscosity_Pa_s,
            heat_capacity_J_kg_K=fluid.heat_capacity_J_kg_K,
            thermal_conductivity_W_m_K=fluid.thermal_conductivity_W_m_K,
        )
        inner = internal_heat_transfer_coefficient(
            nusselt=nusselt_internal_flow(
                reynolds=re, prandtl=pr, heating=self.utility_temperature_K > temperature
            ),
            thermal_conductivity_W_m_K=fluid.thermal_conductivity_W_m_K,
            diameter_m=self.pipe.diameter_m,
        )
        u = overall_heat_transfer_coefficient(
            inner_h_W_m2_K=inner,
            outer_h_W_m2_K=self.outer_h_W_m2_K,
            wall_thickness_m=self.wall_thickness_m,
            wall_conductivity_W_m_K=self.wall_conductivity_W_m_K,
        )
        capacity = flow * fluid.density_kg_m3 * fluid.heat_capacity_J_kg_K
        return (
            counterflow_effectiveness(ntu=u * self.exchanger_area_m2 / capacity, capacity_ratio=0)
            * capacity
            * (self.utility_temperature_K - temperature)
        )

    def to_dict(self):
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload["pipe"] = asdict(self.pipe)
        for name in ("control_valve", "orifice_meter", "differential_meter"):
            if payload[name] is not None:
                payload[name] = payload[name].to_dict()
        return payload

    @classmethod
    def from_dict(cls, value):
        payload = {**value, "pipe": PipeSpec(**value["pipe"])}
        for key, kind in (("control_valve", LiquidControlValveSpec),
                          ("orifice_meter", LiquidOrificeSpec),
                          ("differential_meter", DifferentialMeterSpec)):
            if payload.get(key) is not None:
                payload[key] = kind.from_dict(payload[key])
        return cls(**payload)


@dataclass(frozen=True)
class FiniteTransportSpec:
    properties: LiquidTransportProperties
    pump: PumpSpec = field(default_factory=PumpSpec)
    path: TransportPathSpec = field(default_factory=TransportPathSpec)
    source_vessel_id: str = "feed"
    capacities_j_k: tuple[float, float] = (5.0, 5.0)
    receiver_capacity_L: float = 0.1
    pickup_volume_L: float = 1e-6
    support_reference_temperature_K: float = 298.15
    caloric_algorithm: str = "residual"

    def __post_init__(self):
        if (
            not self.source_vessel_id
            or len(self.capacities_j_k) != 2
            or any(not isfinite(c) or c <= 0 for c in self.capacities_j_k)
            or not isfinite(self.receiver_capacity_L)
            or self.receiver_capacity_L <= 0
            or not isfinite(self.pickup_volume_L)
            or self.pickup_volume_L <= 0
            or not isfinite(self.support_reference_temperature_K)
            or self.support_reference_temperature_K <= 0
            or self.caloric_algorithm not in {"residual", "brentq"}
        ):
            raise ValueError("finite tank geometry and heat capacities are required")
        lo, hi = self.properties.temperature_bounds_K
        if not lo <= self.path.utility_temperature_K <= hi:
            raise ValueError("utility temperature must be in liquid property domain")
        if self.path.control_valve is not None:
            if len(self.properties.packages) != 1:
                raise ValueError("IEC valve requires one pure liquid property package")
            package = self.properties.packages[0]
            critical = package.component.metadata.get("critical_pressure_Pa")
            if not isinstance(critical, int | float) or not isfinite(critical) or critical <= 0:
                raise ValueError("IEC valve requires explicit critical_pressure_Pa metadata")
            if not package.by_property("vapor_pressure"):
                raise ValueError("IEC valve requires a current vapor-pressure correlation")

    def enthalpy(self, index, amounts, temperature):
        return self.properties.enthalpy(amounts, temperature) + self.capacities_j_k[index] * (
            temperature - self.support_reference_temperature_K
        )

    def at_ph(self, index, amounts, enthalpy):
        options = {} if self.caloric_algorithm == "brentq" else {
            "heat_capacity_at_temperature": lambda t: (
                self.properties.heat_capacity(amounts, t) + self.capacities_j_k[index]
            ),
        }
        return caloric_temperature(
            lambda t: self.enthalpy(index, amounts, t), enthalpy,
            self.properties.temperature_bounds_K, energy_basis="enthalpy",
            algorithm=self.caloric_algorithm, **options,
        )

    def to_dict(self):
        return {
            "properties": self.properties.to_dict(),
            "pump": self.pump.to_dict(),
            "path": self.path.to_dict(),
            "source_vessel_id": self.source_vessel_id,
            "support_reference_temperature_K": self.support_reference_temperature_K,
            "capacities_j_k": list(self.capacities_j_k),
            "receiver_capacity_L": self.receiver_capacity_L,
            "pickup_volume_L": self.pickup_volume_L,
            "caloric_algorithm": self.caloric_algorithm,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            **{
                **value,
                "properties": LiquidTransportProperties.from_dict(value["properties"]),
                "pump": PumpSpec(**value["pump"]),
                "path": TransportPathSpec.from_dict(value["path"]),
                "capacities_j_k": tuple(value["capacities_j_k"]),
            }
        )


@dataclass(frozen=True)
class TransportInventory:
    amounts_mol: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    time_s: float = 0.0
    heater_energy_J: float = 0.0
    condenser_energy_J: float = 0.0


def transport_rates(spec, state, temperatures, requested_flow_m3_s):
    n, temperature = state.amounts_mol[0], temperatures[0]
    valve_closed = spec.path.control_valve is not None and not spec.path.control_valve.is_open
    if requested_flow_m3_s == 0 or not sum(n.values()) or valve_closed:
        zeros = np.zeros((2, len(spec.properties.component_ids)))
        return PhaseInventoryRates(zeros, np.zeros(2), 0.0, 0.0, ()), {
            "actual_flow_m3_s": 0.0,
            "pressure_rise_Pa": 0.0,
            "shaft_power_W": 0.0,
            "hydraulic_power_W": 0.0,
            "external_loss_W": 0.0,
            "utility_heat_W": 0.0,
            "flow_limited": requested_flow_m3_s > 0,
            "source_temperature_K": temperature,
            "control_valve_closed": valve_closed,
        }
    fluid, h, volume = spec.properties.evaluate(n, temperature)
    valve_properties = {}
    if spec.path.control_valve is not None:
        package = spec.properties.packages[0]
        valve_properties = {
            "downstream_pressure_Pa": spec.properties.pressure_Pa,
            "saturation_pressure_Pa": package.evaluate("vapor_pressure",
                temperature_K=temperature, validity_policy="raise").to("Pa").value,
            "critical_pressure_Pa": package.component.metadata["critical_pressure_Pa"],
        }
    point = pump_operating_point(
        spec.pump, requested_flow_m3_s, lambda flow: spec.path.pressure_drop(
            fluid, flow, **valve_properties, check_choking=False)
    )
    flow = point["actual_flow_m3_s"]
    # Numerical pump trials may exceed choking, but the accepted material
    # path must remain inside its declared single-liquid model.
    devices = spec.path.hydraulic_diagnostics(fluid, flow, **valve_properties)
    amount_rates = np.array([n[k] * flow / volume for k in spec.properties.component_ids])
    rate = float(sum(amount_rates))
    hydraulic = point["hydraulic_power_W"]
    # With an adiabatic, well-mixed draining source T/x remain constant. All
    # work is delivered downstream; no motor inefficiency heats the source.
    stream_temperature = temperature
    outlet_margin = 0.0
    if rate:
        lo, hi = spec.properties.temperature_bounds_K
        target_h = h + hydraulic / rate
        low_h = spec.properties.enthalpy(n, lo) / sum(n.values())
        high_h = spec.properties.enthalpy(n, hi) / sum(n.values())
        outlet_margin = high_h - target_h
        # Only trial property evaluation is bounded. An independent event
        # stops an unsupported outlet before any inventory/work is committed;
        # the derivative below always retains the full hydraulic energy.
        stream_temperature = caloric_temperature(
            lambda t: spec.properties.enthalpy(n, t) / sum(n.values()),
            min(max(target_h, low_h), high_h), (lo, hi), energy_basis="enthalpy",
            algorithm=spec.caloric_algorithm,
            **({} if spec.caloric_algorithm == "brentq" else {
                "heat_capacity_at_temperature": lambda t: (
                    spec.properties.heat_capacity(n, t) / sum(n.values())
                ),
            }),
        )
    stream_fluid, _, stream_volume = spec.properties.evaluate(n, stream_temperature)
    stream_flow = flow * stream_volume / volume
    heat = spec.path.heat_rate(stream_fluid, stream_flow, stream_temperature)
    # Cp is local. Limit only against the actual bath enthalpy to prevent a
    # variable-Cp linearization from crossing a passive boundary temperature.
    bath_heat = rate * (
        spec.properties.enthalpy(n, spec.path.utility_temperature_K) / sum(n.values())
        - spec.properties.enthalpy(n, stream_temperature) / sum(n.values())
    )
    heat = float(np.sign(heat) * min(abs(heat), abs(bath_heat)))
    contribution = PhaseInventoryRates(
        np.array([-amount_rates, amount_rates]),
        np.array([-rate * h, rate * h + hydraulic + heat]),
        hydraulic + max(heat, 0),
        max(-heat, 0),
        (),
    )
    return contribution, {
        **point,
        "utility_heat_W": heat,
        "source_temperature_K": temperature,
        "source_viscosity_Pa_s": fluid.viscosity_Pa_s,
        "hydraulic_devices": devices,
        "pump_outlet_enthalpy_margin_J_mol": outlet_margin if rate else None,
    }


def advance_transport(spec, state, requested_flow_m3_s, duration_s, *, max_step_s=None):
    if (
        not isfinite(requested_flow_m3_s)
        or requested_flow_m3_s < 0
        or len(state.amounts_mol) != 2
        or len(state.enthalpies_J) != 2
    ):
        raise ValueError("transport needs two tank inventories and a finite nonnegative flow")

    def bounds(index, amounts):
        return tuple(spec.enthalpy(index, amounts, t) for t in spec.properties.temperature_bounds_K)

    def volume(index, state, temperatures):
        if not sum(state.amounts_mol[index].values()):
            return 0.0
        return 1000 * spec.properties.evaluate(state.amounts_mol[index], temperatures[index])[2]

    temperatures = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True))
    )
    initial_rates, point = transport_rates(spec, state, temperatures, requested_flow_m3_s)
    counters = {"empty_source_trial_extensions": 0}

    def rates(candidate, temperatures):
        if not sum(candidate.amounts_mol[0].values()):
            # The solver may trial past the positive pickup event. Extend the
            # draining source's invariant stream only for this rejected region;
            # the accepted endpoint still retains its declared finite heel.
            counters["empty_source_trial_extensions"] += 1
            return initial_rates
        return transport_rates(spec, candidate, temperatures, requested_flow_m3_s)[0]

    def outlet_margin(candidate, temperatures):
        margin = transport_rates(spec, candidate, temperatures, requested_flow_m3_s)[1].get(
            "pump_outlet_enthalpy_margin_J_mol"
        )
        return 1.0 if margin is None else margin

    stops = (
        [
            ("feed_pickup_exhausted", lambda s, t: volume(0, s, t) - spec.pickup_volume_L),
            ("receiver_full", lambda s, t: spec.receiver_capacity_L - volume(1, s, t)),
            ("pump_outlet_property_limit", outlet_margin),
        ]
        if requested_flow_m3_s
        else []
    )
    for i in (0, 1):
        stops.extend(
            (
                (
                    f"tank:{i}:minimum_temperature",
                    lambda s, t, i=i: s.enthalpies_J[i] - bounds(i, s.amounts_mol[i])[0],
                ),
                (
                    f"tank:{i}:maximum_temperature",
                    lambda s, t, i=i: bounds(i, s.amounts_mol[i])[1] - s.enthalpies_J[i],
                ),
            )
        )
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=rates,
        closure_indices=(0, 1),
        enthalpy_bounds=bounds,
        stop_functions=tuple(stops),
        max_step_s=max_step_s,
        initial_boundary="stop",
        solver_id="finite_pump_tanks_bdf_v1",
        diagnostic_counters=counters,
    )
    # These are exact constant-boundary integrals for this draining-source
    # model, not a second time advance or energy inferred from a residual.
    energy = {
        key.replace("_power_W", "_work_J").replace("_W", "_J"): point[key]
        * result.actual_duration_s
        for key in ("shaft_power_W", "hydraulic_power_W", "external_loss_W", "utility_heat_W")
    }
    return result, {**point, **energy}
