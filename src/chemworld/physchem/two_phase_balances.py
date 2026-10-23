"""Signed gravitational and acceleration losses with explicit phase bases."""

from __future__ import annotations

from math import isfinite, radians, sin

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.two_phase_holdup import (
    TwoPhaseProperties,
    _quality,
    two_phase_mixture_density,
)


def _momentum_volume(p: TwoPhaseProperties, quality: float, alpha: float) -> float:
    _quality(quality)
    _quality(alpha)
    if (alpha == 0 and quality != 0) or (alpha == 1 and quality != 1):
        raise ValueError("a phase with positive mass cannot occupy zero cross section")
    liquid = 0.0 if quality == 1 else (1 - quality) ** 2 / (p.liquid_density_kg_m3 * (1 - alpha))
    vapor = 0.0 if quality == 0 else quality**2 / (p.vapor_density_kg_m3 * alpha)
    return liquid + vapor


def two_phase_acceleration_pressure_drop(
    inlet: TwoPhaseProperties,
    outlet: TwoPhaseProperties,
    *,
    mass_flux_kg_m2_s: float,
    inlet_quality: float,
    outlet_quality: float,
    inlet_void_fraction: float,
    outlet_void_fraction: float,
) -> float:
    """Constant-area momentum-flux change; positive is a pressure loss.

    Quality is mass fraction; void fraction is cross-sectional volume fraction.
    Pure-phase consistent endpoints are evaluated by their finite limits.
    """
    _positive(mass_flux_kg_m2_s, "mass flux")
    return mass_flux_kg_m2_s**2 * (
        _momentum_volume(outlet, outlet_quality, outlet_void_fraction)
        - _momentum_volume(inlet, inlet_quality, inlet_void_fraction)
    )


def two_phase_acceleration_gradient(
    properties: TwoPhaseProperties,
    *,
    mass_flux_kg_m2_s: float,
    vapor_quality: float,
    pressure_drop_gradient_Pa_m: float,
    liquid_volume_pressure_derivative_m3_kg_Pa: float,
    vapor_volume_pressure_derivative_m3_kg_Pa: float,
    quality_pressure_derivative_per_Pa: float,
    area_m2: float,
    area_gradient_m: float = 0.0,
) -> float:
    """Homogeneous acceleration loss gradient, with dP/dz = -drop_gradient.

    Momentum gives loss = G²[d(v)/dz - v/A dA/dz]. Thus an expanding
    constant-density channel gives a pressure recovery (negative loss).
    The reference's area term has the opposite sign; this API follows momentum
    conservation and records that difference rather than concealing it.
    """
    _positive(mass_flux_kg_m2_s, "mass flux")
    _positive(area_m2, "cross-sectional area")
    _quality(vapor_quality)
    for value in (
        pressure_drop_gradient_Pa_m,
        liquid_volume_pressure_derivative_m3_kg_Pa,
        vapor_volume_pressure_derivative_m3_kg_Pa,
        quality_pressure_derivative_per_Pa,
        area_gradient_m,
    ):
        if not isfinite(value):
            raise ValueError("derivatives and pressure gradient must be finite")
    x = vapor_quality
    vl, vg = 1 / properties.liquid_density_kg_m3, 1 / properties.vapor_density_kg_m3
    volume = (1 - x) * vl + x * vg
    derivative = (
        (vg - vl) * quality_pressure_derivative_per_Pa
        + (1 - x) * liquid_volume_pressure_derivative_m3_kg_Pa
        + x * vapor_volume_pressure_derivative_m3_kg_Pa
    )
    return -(mass_flux_kg_m2_s**2) * (
        pressure_drop_gradient_Pa_m * derivative + volume * area_gradient_m / area_m2
    )


def two_phase_gravity_gradient(
    properties: TwoPhaseProperties,
    *,
    void_fraction: float,
    inclination_degrees: float,
    gravity_m_s2: float = 9.80665,
) -> float:
    """Local signed pressure-loss gradient; uphill positive, downhill negative."""
    _positive(gravity_m_s2, "gravity")
    if not isfinite(inclination_degrees) or not -90 <= inclination_degrees <= 90:
        raise ValueError("inclination must lie in [-90, 90] degrees")
    return (
        gravity_m_s2
        * sin(radians(inclination_degrees))
        * two_phase_mixture_density(properties, void_fraction)
    )


def two_phase_gravity_pressure_drop(
    inlet: TwoPhaseProperties,
    *,
    inlet_void_fraction: float,
    length_m: float,
    inclination_degrees: float,
    outlet: TwoPhaseProperties | None = None,
    outlet_void_fraction: float | None = None,
    gravity_m_s2: float = 9.80665,
) -> float:
    """Trapezoidal segment integral of declared endpoint mixture densities."""
    if not isfinite(length_m) or length_m < 0:
        raise ValueError("length must be finite and nonnegative")
    gradient_in = two_phase_gravity_gradient(
        inlet,
        void_fraction=inlet_void_fraction,
        inclination_degrees=inclination_degrees,
        gravity_m_s2=gravity_m_s2,
    )
    gradient_out = two_phase_gravity_gradient(
        outlet or inlet,
        void_fraction=inlet_void_fraction if outlet_void_fraction is None else outlet_void_fraction,
        inclination_degrees=inclination_degrees,
        gravity_m_s2=gravity_m_s2,
    )
    return length_m * (gradient_in + gradient_out) / 2
