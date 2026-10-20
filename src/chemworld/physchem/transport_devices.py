"""Finite single-phase tank transport on the shared n/H integrator.

Pipe and exchanger have negligible holdup. Reservoirs share pressure/elevation;
hydraulic dissipation enters liquid H and shaft inefficiency leaves to ambient.
The utility is an explicit fixed-temperature thermal boundary.
"""

from dataclasses import asdict, dataclass, field
from math import isfinite

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport import (
    PipeSpec,
    PumpSpec,
    counterflow_effectiveness,
    internal_heat_transfer_coefficient,
    nusselt_internal_flow,
    overall_heat_transfer_coefficient,
    packed_bed_pressure_drop_ergun,
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

    def __post_init__(self):
        if not self.packages or len(set(self.component_ids)) != len(self.packages):
            raise ValueError("liquid transport needs distinct component packages")
        lo, hi = self.temperature_bounds_K
        if (
            not all(isfinite(x) for x in (lo, hi, self.pressure_Pa))
            or not 0 < lo < hi
            or self.pressure_Pa <= 0
        ):
            raise ValueError("liquid transport requires finite positive T/P bounds")

    @property
    def component_ids(self):
        return tuple(p.component.identifier for p in self.packages)

    @property
    def references(self):
        return {k: CaloricReference(phase="liquid") for k in self.component_ids}

    def enthalpy(self, amounts, temperature):
        if not sum(amounts.values()):
            return 0.0
        report = evaluate_property_state(
            self.packages,
            amounts,
            temperature_K=temperature,
            pressure_Pa=self.pressure_Pa,
            phase="liquid",
            quantities=("enthalpy",),
            references=self.references,
        )
        return sum(amounts.values()) * report.properties["enthalpy"].value

    def evaluate(self, amounts, temperature):
        return liquid_transport_state(
            self.packages, amounts, temperature, self.pressure_Pa, self.references
        )

    def to_dict(self):
        return {
            "packages": [p.to_dict() for p in self.packages],
            "temperature_bounds_K": list(self.temperature_bounds_K),
            "pressure_Pa": self.pressure_Pa,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            tuple(
                ComponentPropertyPackage(
                    ComponentSpec.from_dict(p["component"]),
                    tuple(PropertyCorrelation.from_dict(c) for c in p["correlations"]),
                )
                for p in value["packages"]
            ),
            tuple(value["temperature_bounds_K"]),
            value["pressure_Pa"],
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

    def __post_init__(self):
        values = asdict(self)
        values.pop("pipe")
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

    def pressure_drop(self, fluid, flow):
        pressure = pipe_pressure_drop(
            self.pipe, fluid, volumetric_flow_m3_s=flow
        ).pressure_drop_total_Pa
        if self.bed_length_m:
            pressure += packed_bed_pressure_drop_ergun(
                fluid=fluid,
                superficial_velocity_m_s=flow / self.bed_area_m2,
                bed_length_m=self.bed_length_m,
                particle_diameter_m=self.particle_diameter_m,
                void_fraction=self.void_fraction,
            ).pressure_drop_Pa
        return pressure

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
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "pipe": PipeSpec(**value["pipe"])})


@dataclass(frozen=True)
class FiniteTransportSpec:
    properties: LiquidTransportProperties
    pump: PumpSpec = field(default_factory=PumpSpec)
    path: TransportPathSpec = field(default_factory=TransportPathSpec)
    source_vessel_id: str = "feed"
    capacities_j_k: tuple[float, float] = (5.0, 5.0)
    receiver_capacity_L: float = 0.1
    pickup_volume_L: float = 1e-6

    def __post_init__(self):
        if (
            not self.source_vessel_id
            or len(self.capacities_j_k) != 2
            or any(not isfinite(c) or c <= 0 for c in self.capacities_j_k)
            or not isfinite(self.receiver_capacity_L)
            or self.receiver_capacity_L <= 0
            or not isfinite(self.pickup_volume_L)
            or self.pickup_volume_L <= 0
        ):
            raise ValueError("finite tank geometry and heat capacities are required")
        lo, hi = self.properties.temperature_bounds_K
        if not lo <= self.path.utility_temperature_K <= hi:
            raise ValueError("utility temperature must be in liquid property domain")

    def enthalpy(self, index, amounts, temperature):
        return self.properties.enthalpy(amounts, temperature) + self.capacities_j_k[index] * (
            temperature - 298.15
        )

    def at_ph(self, index, amounts, enthalpy):
        lo, hi = self.properties.temperature_bounds_K
        return brentq(lambda t: self.enthalpy(index, amounts, t) - enthalpy, lo, hi, xtol=1e-10)

    def to_dict(self):
        return {
            "properties": self.properties.to_dict(),
            "pump": self.pump.to_dict(),
            "path": self.path.to_dict(),
            "source_vessel_id": self.source_vessel_id,
            "capacities_j_k": list(self.capacities_j_k),
            "receiver_capacity_L": self.receiver_capacity_L,
            "pickup_volume_L": self.pickup_volume_L,
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
    if requested_flow_m3_s == 0 or not sum(n.values()):
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
        }
    fluid, h, volume = spec.properties.evaluate(n, temperature)
    point = pump_operating_point(
        spec.pump, requested_flow_m3_s, lambda flow: spec.path.pressure_drop(fluid, flow)
    )
    flow = point["actual_flow_m3_s"]
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
        stream_temperature = brentq(
            lambda t: (
                spec.properties.enthalpy(n, t) / sum(n.values()) - min(max(target_h, low_h), high_h)
            ),
            lo,
            hi,
            xtol=1e-10,
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
