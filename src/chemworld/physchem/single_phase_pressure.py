"""Single-phase engineering losses with explicit momentum/energy conventions."""

from math import isfinite, pi, radians, sin

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.friction_correlations import pipe_friction_factor
from chemworld.physchem.transport import FluidState


def single_phase_friction_pressure_drop(
    fluid: FluidState,
    *,
    mass_flow_kg_s: float,
    diameter_m: float,
    length_m: float = 1.0,
    roughness_m: float = 0.0,
    method: str = "Colebrook",
    check_recommended_ranges: bool = True,
) -> float:
    _positive(mass_flow_kg_s, "mass flow")
    _positive(diameter_m, "diameter")
    if not isfinite(length_m) or length_m < 0:
        raise ValueError("length must be finite and nonnegative")
    flux = mass_flow_kg_s / (pi * diameter_m * diameter_m / 4)
    factor = pipe_friction_factor(
        flux * diameter_m / fluid.viscosity_Pa_s,
        roughness_m / diameter_m,
        method=method,
        check_recommended_ranges=check_recommended_ranges,
    )
    return factor * length_m / diameter_m * flux**2 / (2 * fluid.density_kg_m3)


def single_phase_acceleration_pressure_drop(
    *,
    mass_flow_kg_s: float,
    inlet_density_kg_m3: float,
    outlet_density_kg_m3: float,
    inlet_diameter_m: float,
    outlet_diameter_m: float | None = None,
) -> float:
    """Mean-density kinetic-energy approximation; positive is an acceleration loss.

    This is exact Bernoulli acceleration for constant density. Arithmetic mean
    density is only an explicit engineering approximation when density changes.
    It is not an exact arbitrary-compressibility momentum integral.
    """
    outlet_diameter = inlet_diameter_m if outlet_diameter_m is None else outlet_diameter_m
    for name, value in (
        ("mass flow", mass_flow_kg_s),
        ("inlet density", inlet_density_kg_m3),
        ("outlet density", outlet_density_kg_m3),
        ("inlet diameter", inlet_diameter_m),
        ("outlet diameter", outlet_diameter),
    ):
        _positive(value, name)
    inlet_velocity = mass_flow_kg_s / (inlet_density_kg_m3 * pi * inlet_diameter_m**2 / 4)
    outlet_velocity = mass_flow_kg_s / (outlet_density_kg_m3 * pi * outlet_diameter**2 / 4)
    return (
        (inlet_density_kg_m3 + outlet_density_kg_m3) / 4 * (outlet_velocity**2 - inlet_velocity**2)
    )


def single_phase_acceleration_gradient(
    *,
    mass_flux_kg_m2_s: float,
    density_kg_m3: float,
    volume_pressure_derivative_m3_kg_Pa: float,
    pressure_drop_gradient_Pa_m: float,
    area_m2: float,
    area_gradient_m: float = 0.0,
) -> float:
    """Momentum loss gradient, using -dP/dz as the positive pressure-drop input.

    Positive area gradient gives recovery. The upstream differential helper has
    the opposite area sign; this reproduces physics rather than that defect.
    """
    for name, value in (
        ("mass flux", mass_flux_kg_m2_s),
        ("density", density_kg_m3),
        ("area", area_m2),
    ):
        _positive(value, name)
    if not all(
        isfinite(value)
        for value in (
            volume_pressure_derivative_m3_kg_Pa,
            pressure_drop_gradient_Pa_m,
            area_gradient_m,
        )
    ):
        raise ValueError("derivatives must be finite")
    return -(mass_flux_kg_m2_s**2) * (
        pressure_drop_gradient_Pa_m * volume_pressure_derivative_m3_kg_Pa
        + area_gradient_m / (density_kg_m3 * area_m2)
    )


def single_phase_gravity_pressure_drop(
    density_kg_m3: float,
    *,
    length_m: float,
    inclination_degrees: float,
    gravity_m_s2: float = 9.80665,
) -> float:
    _positive(density_kg_m3, "density")
    _positive(gravity_m_s2, "gravity")
    if not isfinite(length_m) or length_m < 0:
        raise ValueError("length must be finite and nonnegative")
    if not isfinite(inclination_degrees) or not -90 <= inclination_degrees <= 90:
        raise ValueError("inclination must lie in [-90,90] degrees")
    return length_m * gravity_m_s2 * sin(radians(inclination_degrees)) * density_kg_m3
