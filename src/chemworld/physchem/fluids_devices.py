"""IEC liquid-valve sizing and installed-valve constitutive relations.

Sizing selects a nominal Kv; it does not advance material or represent a CFD
valve. Units at the API are SI except the explicitly named metric Kv (m3/h).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite, log10, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.transport import FluidState

IEC_REFERENCE_DENSITY = 999.1032907570233


@dataclass(frozen=True)
class LiquidControlValveSpec:
    kv_m3_h: float
    diameter_m: float
    inlet_diameter_m: float
    outlet_diameter_m: float
    recovery_factor: float
    style_modifier: float
    is_open: bool = True
    source_note: str = ""

    def __post_init__(self) -> None:
        for name in ("kv_m3_h", "diameter_m", "inlet_diameter_m", "outlet_diameter_m"):
            _positive(getattr(self, name), name)
        for name in ("recovery_factor", "style_modifier"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if type(self.is_open) is not bool or not self.source_note.strip():
            raise ValueError("valve requires explicit open state and coefficient provenance")
        if self.diameter_m > min(self.inlet_diameter_m, self.outlet_diameter_m):
            raise ValueError("IEC valve supports same bore or attached reducers, not larger valve")

    @property
    def has_reducers(self) -> bool:
        return self.diameter_m != self.inlet_diameter_m or (
            self.diameter_m != self.outlet_diameter_m)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LiquidControlValveSpec:
        return cls(**value)


def valve_critical_pressure_factor(
    saturation_pressure_Pa: float, critical_pressure_Pa: float,
) -> float:
    _positive(critical_pressure_Pa, "critical pressure")
    if not isfinite(saturation_pressure_Pa) or not (
        0 <= saturation_pressure_Pa < critical_pressure_Pa
    ):
        raise ValueError("liquid saturation pressure must be below critical pressure")
    return 0.96 - 0.28 * sqrt(saturation_pressure_Pa / critical_pressure_Pa)


def valve_piping_factors(spec: LiquidControlValveSpec) -> tuple[float, float]:
    d = spec.diameter_m * 1000
    inlet, outlet = spec.diameter_m / spec.inlet_diameter_m, (
        spec.diameter_m / spec.outlet_diameter_m)
    inlet_loss = 0.5 * (1 - inlet**2)**2 + 1 - inlet**4
    total_loss = inlet_loss + (1 - outlet**2)**2 - (1 - outlet**4)
    multiplier = (spec.kv_m3_h / d**2)**2 / 0.0016
    geometry = 1 / sqrt(1 + total_loss * multiplier)
    recovery = spec.recovery_factor / sqrt(
        1 + spec.recovery_factor**2 * inlet_loss * multiplier)
    return geometry, recovery


def valve_reynolds(
    spec: LiquidControlValveSpec, fluid: FluidState, flow_m3_s: float,
) -> float:
    if not isfinite(flow_m3_s) or flow_m3_s < 0:
        raise ValueError("valve flow must be finite and nonnegative")
    diameter = spec.inlet_diameter_m * 1000
    return (0.0707 * spec.style_modifier * flow_m3_s * 3600
        / (fluid.viscosity_Pa_s / fluid.density_kg_m3 * sqrt(
            spec.kv_m3_h * spec.recovery_factor))
        * (1 + spec.recovery_factor**2 * spec.kv_m3_h**2
            / (0.0016 * diameter**4))**0.25)


def valve_reynolds_factor(
    *, recovery_factor: float, kv_m3_h: float, diameter_m: float,
    reynolds: float, full_trim: bool,
) -> float:
    """IEC full/reduced-trim FR; diameters are converted to mm internally."""
    for name, value in (("recovery", recovery_factor), ("Kv", kv_m3_h),
                        ("diameter", diameter_m), ("Re", reynolds)):
        _positive(value, name)
    if recovery_factor > 1 or type(full_trim) is not bool:
        raise ValueError("invalid valve trim declaration")
    if reynolds > 10000:
        return 1.0
    capacity_per_area = kv_m3_h / (diameter_m * 1000)**2
    if full_trim:
        coefficient = 0.0016 / min(capacity_per_area, 0.04)**2
    else:
        coefficient = 1 + 140 * capacity_per_area**(2 / 3)
    creeping = 0.026 / recovery_factor * sqrt(coefficient * reynolds)
    if not full_trim:
        creeping = min(creeping, 1.0)
    transition = 1 + 0.33 * sqrt(recovery_factor) / coefficient**0.25 * log10(
        reynolds / 10000)
    answer = creeping if reynolds < 10 else min(creeping, transition)
    if answer <= 0 or not isfinite(answer):
        raise ValueError("IEC Reynolds correction leaves its supported positive domain")
    return answer


def _trim_factor(spec: LiquidControlValveSpec, reynolds: float) -> float:
    return valve_reynolds_factor(
        recovery_factor=spec.recovery_factor, kv_m3_h=spec.kv_m3_h,
        diameter_m=spec.diameter_m, reynolds=reynolds,
        full_trim=spec.kv_m3_h / (spec.diameter_m * 1000)**2 <= 0.016 * 0.865,
    )


def _correct_installed_valve_flow(
    spec: LiquidControlValveSpec, fluid: FluidState, turbulent_flow: float,
) -> tuple[float, float, float]:
    """Common fixed-Kv implicit Q = Qturbulent FR(Re(Q)) relation."""
    re_turbulent = valve_reynolds(spec, fluid, turbulent_flow)
    if re_turbulent > 10000 or turbulent_flow == 0:
        return turbulent_flow, re_turbulent, 1.0
    if spec.has_reducers:
        raise ValueError("low-Re valve with attached reducers is not supported")
    root_coefficient = _trim_factor(spec, 1e-8) / 1e-4
    creeping = turbulent_flow * root_coefficient**2 * re_turbulent
    for candidate in (creeping, turbulent_flow):
        reynolds = valve_reynolds(spec, fluid, candidate)
        if reynolds < 10 and abs(
            candidate / turbulent_flow / _trim_factor(spec, reynolds) - 1
        ) < 1e-10:
            return candidate, reynolds, _trim_factor(spec, reynolds)
    lower_fraction = 10 / re_turbulent
    if lower_fraction > 1:
        raise ValueError("pressure falls in the IEC low-Re branch discontinuity")

    def residual(fraction: float) -> float:
        reynolds = valve_reynolds(spec, fluid, fraction * turbulent_flow)
        return fraction / _trim_factor(spec, reynolds) - 1

    flow = turbulent_flow * _engineering_root(residual, .8, lower_fraction, 1.0,
        name="valve_flow_fraction", unit="dimensionless", scale=1)
    reynolds = valve_reynolds(spec, fluid, flow)
    return flow, reynolds, _trim_factor(spec, reynolds)


@dataclass(frozen=True)
class LiquidValveResult:
    flow_m3_s: float
    pressure_drop_Pa: float
    upstream_pressure_Pa: float
    downstream_pressure_Pa: float
    valve_reynolds: float
    reynolds_factor: float
    piping_factor: float
    installed_recovery_factor: float
    critical_pressure_drop_Pa: float
    regime: str
    choked: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def liquid_valve_pressure_drop(
    spec: LiquidControlValveSpec, fluid: FluidState, *, flow_m3_s: float,
    downstream_pressure_Pa: float, saturation_pressure_Pa: float,
    critical_pressure_Pa: float, check_choking: bool = True,
) -> LiquidValveResult:
    """Installed liquid valve at fixed Q; reject a two-phase/choked process state."""
    _positive(downstream_pressure_Pa, "downstream absolute pressure")
    ff = valve_critical_pressure_factor(saturation_pressure_Pa, critical_pressure_Pa)
    if downstream_pressure_Pa <= saturation_pressure_Pa:
        raise ValueError("single-phase valve outlet must exceed liquid saturation pressure")
    if not isfinite(flow_m3_s) or flow_m3_s < 0:
        raise ValueError("valve supports finite nonnegative forward flow")
    if not spec.is_open and flow_m3_s:
        raise ValueError("closed valve cannot pass a positive flow")
    fp, flp = valve_piping_factors(spec)
    if flow_m3_s == 0:
        return LiquidValveResult(0.0, 0.0, downstream_pressure_Pa, downstream_pressure_Pa,
            0.0, 1.0, fp, flp, (flp / fp)**2 * (
                downstream_pressure_Pa - ff * saturation_pressure_Pa), "stopped", False)
    reynolds = valve_reynolds(spec, fluid, flow_m3_s)
    if reynolds <= 10000 and spec.has_reducers:
        raise ValueError("low-Re valve with attached reducers is not supported")
    fr = _trim_factor(spec, reynolds) if reynolds <= 10000 else 1.0
    difference = 1000 * fluid.density_kg_m3 / IEC_REFERENCE_DENSITY * (
        flow_m3_s * 3600 / (0.1 * spec.kv_m3_h * fp * fr))**2
    upstream = downstream_pressure_Pa + difference
    critical = (flp / fp)**2 * (upstream - ff * saturation_pressure_Pa)
    if difference >= critical and check_choking:
        raise ValueError("single-phase valve path reached the liquid choking boundary")
    return LiquidValveResult(flow_m3_s, difference, upstream, downstream_pressure_Pa,
        reynolds, fr, fp, flp, critical,
        "laminar/transitional" if reynolds <= 10000 else "turbulent", difference >= critical)


def liquid_valve_flow(
    spec: LiquidControlValveSpec, fluid: FluidState, *, upstream_pressure_Pa: float,
    downstream_pressure_Pa: float, saturation_pressure_Pa: float, critical_pressure_Pa: float,
) -> LiquidValveResult:
    """Invert the installed, single-phase valve relation using the common root."""
    _positive(upstream_pressure_Pa, "upstream absolute pressure")
    _positive(downstream_pressure_Pa, "downstream absolute pressure")
    ff = valve_critical_pressure_factor(saturation_pressure_Pa, critical_pressure_Pa)
    if downstream_pressure_Pa <= saturation_pressure_Pa:
        raise ValueError("single-phase outlet must exceed saturation pressure")
    difference = upstream_pressure_Pa - downstream_pressure_Pa
    if difference < 0:
        raise ValueError("reverse valve flow is outside the declared forward path")
    if difference == 0 or not spec.is_open:
        return LiquidValveResult(0, difference, upstream_pressure_Pa, downstream_pressure_Pa,
            0, 1, 1, spec.recovery_factor, spec.recovery_factor**2 * (
                upstream_pressure_Pa - ff * saturation_pressure_Pa), "stopped", False)
    fp, flp = valve_piping_factors(spec)
    critical = (flp / fp)**2 * (upstream_pressure_Pa - ff * saturation_pressure_Pa)
    if difference >= critical:
        raise ValueError("single-phase valve path reached the liquid choking boundary")
    turbulent = 0.1 * spec.kv_m3_h * fp / 3600 * sqrt(
        difference / 1000 * IEC_REFERENCE_DENSITY / fluid.density_kg_m3)
    flow, _, _ = _correct_installed_valve_flow(spec, fluid, turbulent)
    return liquid_valve_pressure_drop(spec, fluid, flow_m3_s=flow,
        downstream_pressure_Pa=downstream_pressure_Pa,
        saturation_pressure_Pa=saturation_pressure_Pa, critical_pressure_Pa=critical_pressure_Pa)


@dataclass(frozen=True)
class LiquidValveSizingResult:
    kv_m3_h: float
    valve_reynolds: float
    reynolds_factor: float
    piping_factor: float
    installed_recovery_factor: float
    choked: bool
    laminar: bool
    iterations: int
    method: str


def size_liquid_control_valve(
    geometry: LiquidControlValveSpec, fluid: FluidState, *, flow_m3_s: float,
    upstream_pressure_Pa: float, downstream_pressure_Pa: float,
    saturation_pressure_Pa: float, critical_pressure_Pa: float,
) -> LiquidValveSizingResult:
    """IEC nominal Kv selection, including conservative low-Re trim selection.

    Reducer iteration is converged tightly; fluids 1.3.1 stops its corresponding
    iteration at 1%, so this result can legitimately differ within that interval.
    """
    for name, value in (("flow", flow_m3_s), ("upstream pressure", upstream_pressure_Pa),
                        ("downstream pressure", downstream_pressure_Pa)):
        _positive(value, name)
    if not upstream_pressure_Pa > downstream_pressure_Pa or (
        upstream_pressure_Pa <= saturation_pressure_Pa
    ):
        raise ValueError("liquid sizing requires positive pressure drop and liquid inlet")
    ff = valve_critical_pressure_factor(saturation_pressure_Pa, critical_pressure_Pa)
    pressure = (upstream_pressure_Pa - downstream_pressure_Pa) / 1000
    driving = (upstream_pressure_Pa - ff * saturation_pressure_Pa) / 1000
    nominal = flow_m3_s * 3600 / 0.1 * sqrt(
        fluid.density_kg_m3 / IEC_REFERENCE_DENSITY / min(
            pressure, geometry.recovery_factor**2 * driving))
    spec = replace(geometry, kv_m3_h=nominal)
    reynolds = valve_reynolds(spec, fluid, flow_m3_s)
    choked = pressure >= geometry.recovery_factor**2 * driving
    if reynolds <= 10000:
        if spec.has_reducers:
            raise ValueError("low-Re sizing with attached reducers is unsupported")
        for iteration in range(1, 129):
            candidate = replace(spec, kv_m3_h=1.3 * spec.kv_m3_h)
            re = valve_reynolds(candidate, fluid, flow_m3_s)
            fr = _trim_factor(candidate, re)
            if spec.kv_m3_h / fr < candidate.kv_m3_h:
                return LiquidValveSizingResult(candidate.kv_m3_h, re, fr, 1,
                    candidate.recovery_factor, choked, True, iteration,
                    "IEC conservative 1.3-step nominal selection")
            spec = candidate
        raise ValueError("IEC conservative low-Re selection did not converge in 128 steps")
    for iteration in range(1, 129):
        fp, flp = valve_piping_factors(spec)
        choked = pressure >= (flp / fp)**2 * driving
        coefficient = flow_m3_s * 3600 / 0.1 * sqrt(
            fluid.density_kg_m3 / IEC_REFERENCE_DENSITY / min(
                fp**2 * pressure, flp**2 * driving))
        if abs(coefficient / spec.kv_m3_h - 1) <= 1e-12:
            spec = replace(spec, kv_m3_h=coefficient)
            final_fp, final_flp = valve_piping_factors(spec)
            return LiquidValveSizingResult(coefficient, valve_reynolds(spec, fluid, flow_m3_s),
                1, final_fp, final_flp, choked, False, iteration,
                "IEC turbulent converged nominal selection")
        spec = replace(spec, kv_m3_h=coefficient)
    raise ValueError("IEC reducer sizing did not converge in 128 steps")
