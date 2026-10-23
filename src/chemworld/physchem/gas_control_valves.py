"""IEC gas control-valve sizing and fixed-installed-Kv constitutive relations.

The correlation's Q is the ideal reference volume at 273.15 K and 101325 Pa.
Its engineering Reynolds convention uses that same reference Q, as explicitly
specified by the reference sizing algorithm; actual inlet volume is separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.fluids_devices import (
    LiquidControlValveSpec,
    _correct_installed_valve_flow,
    _trim_factor,
    valve_piping_factors,
    valve_reynolds,
)
from chemworld.physchem.gas_flow import IsothermalGasState
from chemworld.physchem.property_reports import R_J_PER_MOL_K
from chemworld.physchem.transport import FluidState


@dataclass(frozen=True)
class GasControlValveSpec:
    kv_m3_h: float
    diameter_m: float
    inlet_diameter_m: float
    outlet_diameter_m: float
    recovery_factor: float
    style_modifier: float
    pressure_ratio_factor: float
    source_note: str
    is_open: bool = True
    expansion_basis: str = "installed"

    def __post_init__(self) -> None:
        _ = self.hydraulic_geometry
        if not isfinite(self.pressure_ratio_factor) or not 0 < self.pressure_ratio_factor <= 1:
            raise ValueError("valve pressure ratio factor must be in (0, 1]")
        if self.expansion_basis not in {"installed", "bare_valve_reference"}:
            raise ValueError("expansion basis must be installed or bare_valve_reference")

    @property
    def hydraulic_geometry(self) -> LiquidControlValveSpec:
        return LiquidControlValveSpec(
            self.kv_m3_h,
            self.diameter_m,
            self.inlet_diameter_m,
            self.outlet_diameter_m,
            self.recovery_factor,
            self.style_modifier,
            self.is_open,
            self.source_note,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GasControlValveSpec:
        return cls(**payload)


def gas_valve_piping_factors(spec: GasControlValveSpec) -> tuple[float, float]:
    fp, _ = valve_piping_factors(spec.hydraulic_geometry)
    ratio = spec.diameter_m / spec.inlet_diameter_m
    inlet_loss = 0.5 * (1 - ratio**2) ** 2 + 1 - ratio**4
    coefficient = (spec.kv_m3_h / (1000 * spec.diameter_m) ** 2) ** 2
    xtp = (
        spec.pressure_ratio_factor
        / fp**2
        / (1 + spec.pressure_ratio_factor * inlet_loss * coefficient / 0.0018)
    )
    return fp, xtp


def _gas_inputs(
    gas: IsothermalGasState,
    viscosity: float,
    gamma: float,
    p1: float,
    p2: float,
) -> tuple[FluidState, float]:
    _positive(viscosity, "gas dynamic viscosity")
    _positive(p1, "upstream absolute pressure")
    if not isfinite(p2) or not 0 <= p2 <= p1:
        raise ValueError("downstream absolute pressure must lie between zero and upstream")
    if not isfinite(gamma) or gamma <= 1:
        raise ValueError("gas heat capacity ratio must exceed one")
    return FluidState(
        gas.density(p1), viscosity
    ), R_J_PER_MOL_K / gas.specific_gas_constant_J_kg_K * 1000


def _turbulent_standard_flow(
    spec: GasControlValveSpec,
    gas: IsothermalGasState,
    gamma: float,
    molar_mass: float,
    p1: float,
    p2: float,
) -> tuple[float, float, float, float, bool]:
    fp, xtp = gas_valve_piping_factors(spec)
    ratio = (p1 - p2) / p1
    critical = gamma / 1.4 * xtp
    basis = xtp if spec.expansion_basis == "installed" else spec.pressure_ratio_factor
    expansion = max(2 / 3, 1 - ratio / (3 * gamma / 1.4 * basis))
    flow = (
        24.6
        * (p1 / 1000)
        * fp
        * expansion
        * spec.kv_m3_h
        / 3600
        * sqrt(min(ratio, critical) / (molar_mass * gas.temperature_K * gas.compressibility_factor))
    )
    return flow, fp, xtp, expansion, ratio >= critical * (1 - 2e-14)


@dataclass(frozen=True)
class GasValveResult:
    reference_flow_m3_s: float
    mass_flow_kg_s: float
    inlet_flow_m3_s: float
    upstream_pressure_Pa: float
    downstream_pressure_Pa: float
    valve_reynolds: float
    reynolds_factor: float
    piping_factor: float
    installed_pressure_ratio_factor: float
    expansion_factor: float
    choked: bool
    regime: str
    reference_temperature_K: float = 273.15
    reference_pressure_Pa: float = 101325.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gas_valve_flow(
    spec: GasControlValveSpec,
    gas: IsothermalGasState,
    *,
    viscosity_Pa_s: float,
    heat_capacity_ratio: float,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
) -> GasValveResult:
    fluid, molar_mass = _gas_inputs(
        gas, viscosity_Pa_s, heat_capacity_ratio, upstream_pressure_Pa, downstream_pressure_Pa
    )
    flow, fp, xtp, expansion, choked = _turbulent_standard_flow(
        spec, gas, heat_capacity_ratio, molar_mass, upstream_pressure_Pa, downstream_pressure_Pa
    )
    if not spec.is_open:
        flow = 0.0
        choked = False
    flow, reynolds, fr = _correct_installed_valve_flow(spec.hydraulic_geometry, fluid, flow)
    mass = flow * 101325 / (gas.specific_gas_constant_J_kg_K * 273.15)
    return GasValveResult(
        flow,
        mass,
        mass / fluid.density_kg_m3,
        upstream_pressure_Pa,
        downstream_pressure_Pa,
        reynolds,
        fr,
        fp,
        xtp,
        expansion,
        choked,
        "stopped" if not flow else "turbulent" if reynolds > 10000 else "laminar/transitional",
    )


@dataclass(frozen=True)
class GasValveSizingResult:
    kv_m3_h: float
    valve_reynolds: float
    reynolds_factor: float
    piping_factor: float
    installed_pressure_ratio_factor: float
    expansion_factor: float
    choked: bool
    laminar: bool
    iterations: int
    method: str


def size_gas_control_valve(
    geometry: GasControlValveSpec,
    gas: IsothermalGasState,
    *,
    reference_flow_m3_s: float,
    viscosity_Pa_s: float,
    heat_capacity_ratio: float,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
) -> GasValveSizingResult:
    """Select nominal Kv; same-bore low-Re uses the conservative 1.3-step protocol.

    Reducers converge to 1e-12, rather than stopping at the source's 1% change.
    The bare_valve_reference basis exposes the source's fixed-xT Y convention.
    """
    _positive(reference_flow_m3_s, "reference flow")
    fluid, molar_mass = _gas_inputs(
        gas, viscosity_Pa_s, heat_capacity_ratio, upstream_pressure_Pa, downstream_pressure_Pa
    )
    if downstream_pressure_Pa >= upstream_pressure_Pa or not geometry.is_open:
        raise ValueError("gas sizing requires an open valve and positive pressure drop")
    bare = replace(
        geometry,
        kv_m3_h=1.0,
        inlet_diameter_m=geometry.diameter_m,
        outlet_diameter_m=geometry.diameter_m,
    )
    unit_flow, _, _, _, _ = _turbulent_standard_flow(
        bare, gas, heat_capacity_ratio, molar_mass, upstream_pressure_Pa, downstream_pressure_Pa
    )
    nominal = reference_flow_m3_s / unit_flow
    spec = replace(geometry, kv_m3_h=nominal)
    reynolds = valve_reynolds(spec.hydraulic_geometry, fluid, reference_flow_m3_s)
    if reynolds <= 10000:
        if spec.hydraulic_geometry.has_reducers:
            raise ValueError("low-Re gas sizing with attached reducers lacks a declared correction")
        for iteration in range(1, 129):
            candidate = replace(spec, kv_m3_h=1.3 * spec.kv_m3_h)
            reynolds = valve_reynolds(candidate.hydraulic_geometry, fluid, reference_flow_m3_s)
            fr = _trim_factor(candidate.hydraulic_geometry, reynolds)
            if spec.kv_m3_h / fr < candidate.kv_m3_h:
                _, fp, xtp, expansion, choked = _turbulent_standard_flow(
                    candidate,
                    gas,
                    heat_capacity_ratio,
                    molar_mass,
                    upstream_pressure_Pa,
                    downstream_pressure_Pa,
                )
                return GasValveSizingResult(
                    candidate.kv_m3_h,
                    reynolds,
                    fr,
                    fp,
                    xtp,
                    expansion,
                    choked,
                    True,
                    iteration,
                    "IEC conservative 1.3-step nominal selection",
                )
            spec = candidate
        raise ValueError("low-Re gas valve selection did not converge in 128 steps")
    for iteration in range(1, 129):
        capacity, fp, xtp, expansion, choked = _turbulent_standard_flow(
            spec, gas, heat_capacity_ratio, molar_mass, upstream_pressure_Pa, downstream_pressure_Pa
        )
        updated = spec.kv_m3_h * reference_flow_m3_s / capacity
        if abs(updated / spec.kv_m3_h - 1) <= 1e-12:
            reynolds = valve_reynolds(spec.hydraulic_geometry, fluid, reference_flow_m3_s)
            return GasValveSizingResult(
                updated,
                reynolds,
                1,
                fp,
                xtp,
                expansion,
                choked,
                False,
                iteration,
                "IEC converged turbulent selection",
            )
        spec = replace(spec, kv_m3_h=updated)
    raise ValueError("gas reducer sizing did not converge in 128 steps; check geometry capacity")


def gas_valve_outlet_pressure(
    spec: GasControlValveSpec,
    gas: IsothermalGasState,
    *,
    reference_flow_m3_s: float,
    viscosity_Pa_s: float,
    heat_capacity_ratio: float,
    upstream_pressure_Pa: float,
) -> GasValveResult:
    """Unique subcritical outlet; at capacity return the highest choked outlet."""
    if not isfinite(reference_flow_m3_s) or reference_flow_m3_s < 0:
        raise ValueError("reference flow must be finite and nonnegative")
    if spec.expansion_basis != "installed":
        raise ValueError("pressure inversion requires the installed expansion relation")
    if not spec.is_open and reference_flow_m3_s > 0:
        raise ValueError("closed gas valve cannot pass positive flow")
    _, xtp = gas_valve_piping_factors(spec)
    max_ratio = min(1.0, heat_capacity_ratio / 1.4 * xtp)

    def evaluate(fraction: float) -> GasValveResult:
        return gas_valve_flow(
            spec,
            gas,
            viscosity_Pa_s=viscosity_Pa_s,
            heat_capacity_ratio=heat_capacity_ratio,
            upstream_pressure_Pa=upstream_pressure_Pa,
            downstream_pressure_Pa=upstream_pressure_Pa * (1 - fraction * max_ratio),
        )

    if reference_flow_m3_s == 0:
        return evaluate(0)
    maximum = evaluate(1)
    if reference_flow_m3_s > maximum.reference_flow_m3_s * (1 + 1e-12):
        raise ValueError("requested gas flow exceeds installed valve capacity")
    if abs(reference_flow_m3_s / maximum.reference_flow_m3_s - 1) <= 1e-12:
        return maximum
    target = reference_flow_m3_s / maximum.reference_flow_m3_s
    fraction = _engineering_root(
        lambda value: (
            (evaluate(value).reference_flow_m3_s / maximum.reference_flow_m3_s) ** 2 - target**2
        ),
        target**2,
        0,
        1,
        name="gas_valve_pressure_fraction",
        unit="dimensionless",
        scale=1,
    )
    return evaluate(fraction)
