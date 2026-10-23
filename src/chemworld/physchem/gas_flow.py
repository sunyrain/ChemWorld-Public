"""Constant-temperature gas-pipe engineering with explicit choking branches.

Density follows P/(Z R_specific T). Z and Darcy friction remain declared
constants; these sizing relations do not simulate gas inventories or CFD.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, expm1, isclose, isfinite, log, log1p, pi, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _engineering_root, _positive


@dataclass(frozen=True)
class IsothermalGasState:
    temperature_K: float
    specific_gas_constant_J_kg_K: float
    compressibility_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in ("temperature_K", "specific_gas_constant_J_kg_K", "compressibility_factor"):
            _positive(getattr(self, name), name)

    @property
    def pressure_over_density_j_kg(self) -> float:
        return (self.compressibility_factor * self.specific_gas_constant_J_kg_K
                * self.temperature_K)

    def density(self, pressure_Pa: float) -> float:
        _positive(pressure_Pa, "absolute gas pressure")
        return pressure_Pa / self.pressure_over_density_j_kg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IsothermalGasState:
        return cls(**payload)


def _exp_minus_linear(value: float) -> float:
    if abs(value) < 1e-3:
        return value**2 * (0.5 + value * (1 / 6 + value * (
            1 / 24 + value * (1 / 120 + value / 720))))
    return expm1(value) - value


def isothermal_critical_pressure_ratio(
    *, darcy_friction_factor: float, length_m: float, diameter_m: float,
) -> float:
    for name, value in (("Darcy friction factor", darcy_friction_factor),
                        ("length", length_m), ("diameter", diameter_m)):
        _positive(value, name)
    resistance = darcy_friction_factor * length_m / diameter_m
    _positive(resistance, "dimensionless pipe resistance")
    initial = sqrt(2 * resistance) if resistance < 1 else log1p(resistance) + .3
    upper = 2 * sqrt(resistance) + resistance if resistance < 1 else log1p(resistance) + 2
    critical_log = _engineering_root(
        lambda value: _exp_minus_linear(value) / resistance - 1,
        initial, 0, upper, name="critical_log_pressure_ratio", unit="dimensionless",
        scale=initial,
    )
    return exp(-critical_log / 2)


@dataclass(frozen=True)
class IsothermalGasPipeResult:
    mass_flow_kg_s: float
    upstream_pressure_Pa: float
    downstream_pressure_Pa: float
    diameter_m: float
    length_m: float
    darcy_friction_factor: float
    inlet_density_kg_m3: float
    outlet_density_kg_m3: float
    critical_outlet_pressure_Pa: float
    maximum_mass_flow_kg_s: float
    choked: bool
    solved_quantity: str = "mass_flow_kg_s"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pipe_mass(
    gas: IsothermalGasState, p1: float, p2: float, diameter: float,
    length: float, friction: float,
) -> float:
    if p1 == p2:
        return 0.0
    logarithm = log(p1 / p2)
    resistance = friction * length / diameter
    return pi * diameter**2 / 4 * p1 * sqrt(
        -expm1(-2 * logarithm)
        / (gas.pressure_over_density_j_kg * (resistance + 2 * logarithm)))


def isothermal_gas_pipe(
    gas: IsothermalGasState, *, upstream_pressure_Pa: float, downstream_pressure_Pa: float,
    diameter_m: float, length_m: float, darcy_friction_factor: float,
) -> IsothermalGasPipeResult:
    for name, value in (("upstream pressure", upstream_pressure_Pa),
                        ("downstream pressure", downstream_pressure_Pa)):
        _positive(value, name)
    if downstream_pressure_Pa > upstream_pressure_Pa:
        raise ValueError("isothermal gas pipe declares forward pressure-driven flow")
    critical = upstream_pressure_Pa * isothermal_critical_pressure_ratio(
        darcy_friction_factor=darcy_friction_factor, length_m=length_m, diameter_m=diameter_m)
    if downstream_pressure_Pa < critical * (1 - 2e-12):
        raise ValueError("outlet pressure lies below the isothermal choking boundary")
    mass = _pipe_mass(gas, upstream_pressure_Pa, downstream_pressure_Pa,
        diameter_m, length_m, darcy_friction_factor)
    capacity = _pipe_mass(gas, upstream_pressure_Pa, critical,
        diameter_m, length_m, darcy_friction_factor)
    return IsothermalGasPipeResult(mass, upstream_pressure_Pa, downstream_pressure_Pa,
        diameter_m, length_m, darcy_friction_factor, gas.density(upstream_pressure_Pa),
        gas.density(downstream_pressure_Pa), critical, capacity,
        isclose(downstream_pressure_Pa, critical, rel_tol=2e-12))


def solve_isothermal_gas_pipe(
    gas: IsothermalGasState, *, darcy_friction_factor: float,
    mass_flow_kg_s: float | None = None, upstream_pressure_Pa: float | None = None,
    downstream_pressure_Pa: float | None = None, diameter_m: float | None = None,
    length_m: float | None = None,
) -> IsothermalGasPipeResult:
    """Solve exactly one missing Q/P1/P2/D/L on the admissible subcritical branch.

    P1 inversion updates inlet density. Fixed-input-density pressure inversion
    is a different engineering contract and is not silently substituted.
    """
    values = {"mass_flow_kg_s": mass_flow_kg_s, "upstream_pressure_Pa": upstream_pressure_Pa,
        "downstream_pressure_Pa": downstream_pressure_Pa, "diameter_m": diameter_m,
        "length_m": length_m}
    missing = [key for key, value in values.items() if value is None]
    if len(missing) != 1:
        raise ValueError("declare exactly one unknown gas-pipe quantity")
    unknown = missing[0]
    _positive(darcy_friction_factor, "Darcy friction factor")
    for key, value in values.items():
        if value is not None:
            if key == "mass_flow_kg_s":
                if not isfinite(value) or value < 0:
                    raise ValueError("mass flow must be finite and nonnegative")
            else:
                _positive(value, key)
    if mass_flow_kg_s is None:
        assert upstream_pressure_Pa is not None and downstream_pressure_Pa is not None
        assert diameter_m is not None and length_m is not None
        return isothermal_gas_pipe(gas, upstream_pressure_Pa=upstream_pressure_Pa,
            downstream_pressure_Pa=downstream_pressure_Pa, diameter_m=diameter_m,
            length_m=length_m, darcy_friction_factor=darcy_friction_factor)
    mass = mass_flow_kg_s
    if unknown in {"upstream_pressure_Pa", "downstream_pressure_Pa"}:
        assert diameter_m is not None and length_m is not None
        fixed_diameter, fixed_length = diameter_m, length_m
        ratio = isothermal_critical_pressure_ratio(darcy_friction_factor=darcy_friction_factor,
            length_m=length_m, diameter_m=diameter_m)
        if upstream_pressure_Pa is None:
            assert downstream_pressure_Pa is not None
            p2 = downstream_pressure_Pa
            low, high = p2, p2 / ratio

            def pressure_mass(fraction: float) -> float:
                return _pipe_mass(gas, low + fraction * (high - low), p2,
                    fixed_diameter, fixed_length, darcy_friction_factor)
        else:
            p1 = upstream_pressure_Pa
            low, high = p1 * ratio, p1

            def pressure_mass(fraction: float) -> float:
                return _pipe_mass(gas, p1, high - fraction * (high - low),
                    fixed_diameter, fixed_length, darcy_friction_factor)
        capacity = pressure_mass(1)
        if mass > capacity * (1 + 2e-12):
            raise ValueError("requested mass flow exceeds the admissible isothermal capacity")
        if mass == 0:
            fraction = 0.0
        elif isclose(mass, capacity, rel_tol=2e-12):
            fraction = 1.0
        else:
            fraction = _engineering_root(
                lambda x: (pressure_mass(x) / capacity)**2 - (mass / capacity)**2,
                min(.95, (mass / capacity)**2), 0, 1,
                name="pipe_pressure_fraction", unit="dimensionless", scale=1.0,
            )
        if upstream_pressure_Pa is None:
            upstream_pressure_Pa = low + fraction * (high - low)
        else:
            downstream_pressure_Pa = high - fraction * (high - low)
    else:
        assert upstream_pressure_Pa is not None and downstream_pressure_Pa is not None
        if mass <= 0 or upstream_pressure_Pa <= downstream_pressure_Pa:
            raise ValueError("pipe dimensions need positive mass flow and pressure difference")
        p1, p2 = upstream_pressure_Pa, downstream_pressure_Pa
        logarithm = log(p1 / p2)
        if length_m is None:
            assert diameter_m is not None
            area = pi * diameter_m**2 / 4
            resistance = (area * p1 / mass)**2 * -expm1(-2 * logarithm) / (
                gas.pressure_over_density_j_kg) - 2 * logarithm
            if resistance <= 0:
                raise ValueError("requested state has no positive isothermal pipe length")
            length_m = diameter_m * resistance / darcy_friction_factor
        else:
            minimum_resistance = _exp_minus_linear(2 * logarithm)
            maximum_diameter = darcy_friction_factor * length_m / minimum_resistance
            capacity = _pipe_mass(gas, p1, p2, maximum_diameter,
                length_m, darcy_friction_factor)
            if mass > capacity * (1 + 2e-12):
                raise ValueError("diameter inverse exceeds the subcritical branch capacity")
            if isclose(mass, capacity, rel_tol=2e-12):
                diameter_m = maximum_diameter
            else:
                # Log diameter keeps engineering scale out of the Newton step.
                upper = log(maximum_diameter)
                initial = upper + 0.4 * log(mass / capacity)
                diameter_log = _engineering_root(
                    lambda x: (_pipe_mass(gas, p1, p2, exp(x), length_m,
                        darcy_friction_factor) / mass)**2 - 1,
                    initial, initial - 20, upper,
                    name="log_pipe_diameter", unit="dimensionless", scale=1.0,
                )
                diameter_m = exp(diameter_log)
    assert upstream_pressure_Pa is not None and downstream_pressure_Pa is not None
    assert diameter_m is not None and length_m is not None
    result = isothermal_gas_pipe(gas, upstream_pressure_Pa=upstream_pressure_Pa,
        downstream_pressure_Pa=downstream_pressure_Pa, diameter_m=diameter_m,
        length_m=length_m, darcy_friction_factor=darcy_friction_factor)
    if not isclose(result.mass_flow_kg_s, mass, rel_tol=5e-8, abs_tol=1e-12):
        raise ValueError("gas-pipe inverse did not satisfy its physical mass-flow residual")
    return IsothermalGasPipeResult(**{**result.to_dict(), "solved_quantity": unknown})
