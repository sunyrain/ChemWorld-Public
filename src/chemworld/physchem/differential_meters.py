"""Liquid/gas differential meters and inverses through the common root solver."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from math import exp, isfinite, log, pi, sqrt
from typing import Any

import numpy as np

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.meter_correlations import (
    METER_ALIASES,
    METER_TYPES,
    meter_beta,
    meter_discharge_coefficient,
    meter_expansibility,
)
from chemworld.physchem.meter_tables import MeterTableSurface
from chemworld.physchem.transport import FluidState


@dataclass(frozen=True)
class DifferentialMeterSpec:
    pipe_diameter_m: float
    secondary_dimension_m: float
    meter_type: str = "ISO 5167 orifice"
    taps: str = "corner"
    tap_position: str = "180 degree"
    specified_coefficient: float | None = None
    specified_expansibility: float | None = None
    coefficient_table: MeterTableSurface | None = None
    recovery_table: MeterTableSurface | None = None
    reynolds_bounds: tuple[float, float] | None = None
    minimum_pressure_ratio: float = 0.75
    installation_note: str = "Formula evaluation; installation validity not qualified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "meter_type", METER_ALIASES.get(self.meter_type, self.meter_type))
        meter_beta(self.pipe_diameter_m, self.secondary_dimension_m, self.meter_type)
        if self.meter_type not in METER_TYPES or not self.installation_note.strip():
            raise ValueError("known meter and explicit installation/formula scope required")
        for value in (self.specified_coefficient, self.specified_expansibility):
            if value is not None:
                _positive(value, "specified meter factor")
        if self.specified_expansibility is not None and self.specified_expansibility > 1:
            raise ValueError("meter expansibility cannot exceed one")
        if not 0 < self.minimum_pressure_ratio < 1:
            raise ValueError("minimum gas pressure ratio must lie in(0,1)")
        if self.reynolds_bounds is not None:
            lower, upper = self.reynolds_bounds
            if not all(isfinite(v) for v in (lower, upper)) or not 0 < lower < upper:
                raise ValueError("finite positive increasing Reynolds bounds required")
        if (
            self.meter_type.startswith("Hollingshead")
            and self.coefficient_table is None
            and self.specified_coefficient is None
        ):
            raise ValueError("Hollingshead meters require caller source calibration tables")
        if self.meter_type == "unspecified meter" and self.specified_coefficient is None:
            raise ValueError("unspecified meter requires calibrated discharge coefficient")

    @property
    def beta(self) -> float:
        return meter_beta(self.pipe_diameter_m, self.secondary_dimension_m, self.meter_type)

    def coefficient(self, reynolds: float) -> float:
        _positive(reynolds, "Reynolds number")
        if (
            self.reynolds_bounds
            and not self.reynolds_bounds[0] <= reynolds <= self.reynolds_bounds[1]
        ):
            raise ValueError("meter flow outside declared Reynolds domain")
        if self.specified_coefficient is not None:
            return self.specified_coefficient
        if self.coefficient_table is not None:
            return self.coefficient_table.evaluate(self.beta, reynolds)
        value = meter_discharge_coefficient(
            self.beta,
            self.pipe_diameter_m,
            reynolds,
            meter_type=self.meter_type,
            taps=self.taps,
            tap_position=self.tap_position,
        )
        _positive(value, "meter discharge coefficient")
        return value

    def expansibility(self, pressure_ratio: float, heat_capacity_ratio: float | None) -> float:
        if not 0 < pressure_ratio <= 1:
            raise ValueError("forward positive meter pressure ratio required")
        if heat_capacity_ratio is not None and pressure_ratio < self.minimum_pressure_ratio:
            raise ValueError("gas meter outside declared pressure-ratio domain")
        if self.specified_expansibility is not None:
            return self.specified_expansibility
        if heat_capacity_ratio is None:
            return 1.0
        method = (
            "cone"
            if "cone" in self.meter_type
            else "conical_orifice"
            if "conical" in self.meter_type
            else "orifice"
            if "orifice" in self.meter_type or self.meter_type == "unspecified meter"
            else "nozzle"
        )
        return meter_expansibility(
            self.beta,
            pressure_ratio,
            heat_capacity_ratio,
            method=method,
            enforce_pressure_domain=False,
        )

    def permanent_loss(self, difference: float, coefficient: float) -> float | None:
        if self.recovery_table is not None:
            return difference * self.recovery_table.evaluate(self.beta, self.pipe_diameter_m)
        if "cone" in self.meter_type:
            return difference * (1.09 - 0.813 * self.beta)
        if "wedge" in self.meter_type:
            return difference * (1.09 - 0.79 * self.beta)
        if "venturi" in self.meter_type:
            return None  # Supplied tube recovery curve required; nozzle source has no correlation.
        recovered = sqrt(1 - self.beta**4 * (1 - coefficient**2))
        return (
            difference
            * (recovered - coefficient * self.beta**2)
            / (recovered + coefficient * self.beta**2)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (getattr(self, name).to_dict() if getattr(self, name) is not None else None)
            if name in {"coefficient_table", "recovery_table"}
            else getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DifferentialMeterSpec":
        data = dict(value)
        for name in ("coefficient_table", "recovery_table"):
            if data.get(name) is not None:
                data[name] = MeterTableSurface.from_dict(data[name])
        if data.get("reynolds_bounds") is not None:
            data["reynolds_bounds"] = tuple(data["reynolds_bounds"])
        return cls(**data)


@dataclass(frozen=True)
class DifferentialMeterResult:
    spec: DifferentialMeterSpec
    mass_flow_kg_s: float
    upstream_pressure_Pa: float
    downstream_pressure_Pa: float
    reynolds: float
    discharge_coefficient: float | None
    expansibility: float
    permanent_loss_Pa: float | None
    normalized_mass_residual: float


@dataclass(frozen=True)
class LiquidMeterPressureResult:
    mass_flow_kg_s: float
    reynolds: float
    discharge_coefficient: float | None
    differential_pressure_Pa: float
    permanent_pressure_loss_Pa: float
    meter_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def liquid_differential_meter_pressure(
    spec: DifferentialMeterSpec, fluid: FluidState, *, mass_flow_kg_s: float
) -> LiquidMeterPressureResult:
    """Physical liquid-path consumer: explicit Re scope and permanent loss required."""
    if (
        spec.reynolds_bounds is None
        or spec.installation_note == "Formula evaluation; installation validity not qualified"
    ):
        raise ValueError(
            "liquid path requires explicit meter Reynolds bounds and installation/calibration scope"
        )
    if spec.specified_expansibility not in (None, 1):
        raise ValueError("single-liquid path does not accept a gas expansibility correction")
    if not isfinite(mass_flow_kg_s) or mass_flow_kg_s < 0:
        raise ValueError("finite nonnegative liquid meter flow required")
    if mass_flow_kg_s == 0:
        return LiquidMeterPressureResult(0, 0, None, 0, 0, spec.meter_type)
    reynolds = 4 * mass_flow_kg_s / (pi * spec.pipe_diameter_m * fluid.viscosity_Pa_s)
    coefficient = spec.coefficient(reynolds)
    area = pi * (spec.pipe_diameter_m * spec.beta) ** 2 / 4
    difference = (
        (mass_flow_kg_s / (coefficient * area)) ** 2
        * (1 - spec.beta**4)
        / (2 * fluid.density_kg_m3)
    )
    permanent = spec.permanent_loss(difference, coefficient)
    if permanent is None:
        raise ValueError("meter permanent loss is unavailable; supply an explicit recovery table")
    if not 0 <= permanent <= difference:
        raise ValueError("meter recovery must yield a passive permanent pressure loss")
    return LiquidMeterPressureResult(
        mass_flow_kg_s, reynolds, coefficient, difference, permanent, spec.meter_type
    )


def meter_flow_coefficient(beta: float, coefficient: float = 1.0) -> float:
    if not 0 < beta < 1:
        raise ValueError("physical effective beta required")
    _positive(coefficient, "discharge coefficient")
    return coefficient / sqrt(1 - beta**4)


def meter_discharge(
    *,
    pipe_diameter_m: float,
    beta: float,
    pressure_difference_Pa: float,
    density_kg_m3: float,
    coefficient: float,
    expansibility: float = 1.0,
) -> float:
    _positive(pipe_diameter_m, "pipe diameter")
    _positive(density_kg_m3, "inlet density")
    _positive(expansibility, "expansibility")
    if not isfinite(pressure_difference_Pa) or pressure_difference_Pa < 0:
        raise ValueError("finite nonnegative differential pressure required")
    return (
        pi
        * (pipe_diameter_m * beta) ** 2
        / 4
        * meter_flow_coefficient(beta, coefficient)
        * expansibility
        * sqrt(2 * density_kg_m3 * pressure_difference_Pa)
    )


def _meter_evaluate(
    spec: DifferentialMeterSpec,
    fluid: FluidState,
    mass: float,
    upstream: float,
    downstream: float,
    k: float | None,
) -> tuple[float, float, float, float]:
    reynolds = 4 * mass / (pi * spec.pipe_diameter_m * fluid.viscosity_Pa_s)
    coefficient = spec.coefficient(reynolds)
    epsilon = spec.expansibility(downstream / upstream, k)
    prediction = meter_discharge(
        pipe_diameter_m=spec.pipe_diameter_m,
        beta=spec.beta,
        pressure_difference_Pa=upstream - downstream,
        density_kg_m3=fluid.density_kg_m3,
        coefficient=coefficient,
        expansibility=epsilon,
    )
    return prediction, reynolds, coefficient, epsilon


def _meter_root(function: Callable[[float], float], lower: float, upper: float) -> float:
    """Find the first sign-change branch, then use the existing engineering root."""
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("meter inverse requires a finite increasing coordinate domain")
    previous = None
    for value in np.linspace(lower, upper, 129):
        try:
            residual = function(float(value))
        except ValueError:
            previous = None
            continue
        if abs(residual) < 1e-12:
            return float(value)
        if previous is not None and previous[1] * residual < 0:
            a, b, fa = previous[0], float(value), previous[1]
            for _ in range(8):
                mid = (a + b) / 2
                fm = function(mid)
                if fa * fm <= 0:
                    b = mid
                else:
                    a, fa = mid, fm
            return _engineering_root(
                function,
                (a + b) / 2,
                a,
                b,
                name="differential_meter_coordinate",
                unit="dimensionless",
                scale=1,
            )
        previous = (float(value), residual)
    raise ValueError("no differential-meter inverse branch in the supplied domain")


def solve_differential_meter(
    spec: DifferentialMeterSpec,
    fluid: FluidState,
    *,
    upstream_pressure_Pa: float | None = None,
    downstream_pressure_Pa: float | None = None,
    mass_flow_kg_s: float | None = None,
    heat_capacity_ratio: float | None = None,
    solve_secondary_dimension: bool = False,
    secondary_bounds_m: tuple[float, float] | None = None,
) -> DifferentialMeterResult:
    """Solve exactly one flow/pressure/dimension unknown; density is inlet density.

    Gas defaults to a pressure ratio of at least .75. These are steady
    subcritical correlations; phase inventory/choked-flow dynamics are not implied.
    Multiple empirical branches use the first sign change; caller bounds can
    select a geometry branch. Permanent loss may be unavailable independently
    of the calibrated meter's measured differential pressure.
    """
    missing = sum(
        value is None for value in (upstream_pressure_Pa, downstream_pressure_Pa, mass_flow_kg_s)
    )
    if missing != (0 if solve_secondary_dimension else 1):
        raise ValueError(
            "supply exactly one unknown flow/pressure, or all values for geometry inverse"
        )
    for value in (upstream_pressure_Pa, downstream_pressure_Pa):
        if value is not None:
            _positive(value, "absolute pressure")
    if mass_flow_kg_s is not None:
        _positive(mass_flow_kg_s, "mass flow")
    if upstream_pressure_Pa is not None and downstream_pressure_Pa is not None:
        if upstream_pressure_Pa < downstream_pressure_Pa:
            raise ValueError("differential meter requires forward pressure drop")
        if upstream_pressure_Pa == downstream_pressure_Pa and mass_flow_kg_s is None:
            return DifferentialMeterResult(
                spec, 0, upstream_pressure_Pa, downstream_pressure_Pa, 0, None, 1, 0, 0
            )
    p1, p2, mass = upstream_pressure_Pa, downstream_pressure_Pa, mass_flow_kg_s
    ratio_min = spec.minimum_pressure_ratio if heat_capacity_ratio is not None else 1e-10
    if mass is None:
        assert p1 is not None and p2 is not None
        fixed_p1, fixed_p2 = p1, p2
        bounds = spec.reynolds_bounds or (1e-6, 1e10)
        if (
            spec.coefficient_table is not None
            and spec.coefficient_table.options["outside"] == "raise"
        ):
            bounds = (
                max(bounds[0], spec.coefficient_table.y_bounds[0]),
                min(bounds[1], spec.coefficient_table.y_bounds[1]),
            )
        factor = pi * spec.pipe_diameter_m * fluid.viscosity_Pa_s / 4

        def residual(log_re: float) -> float:
            trial = exp(log_re) * factor
            return (
                _meter_evaluate(spec, fluid, trial, fixed_p1, fixed_p2, heat_capacity_ratio)[0]
                / trial
                - 1
            )

        mass = exp(_meter_root(residual, log(bounds[0]), log(bounds[1]))) * factor
    elif solve_secondary_dimension:
        assert p1 is not None and p2 is not None
        fixed_p1, fixed_p2 = p1, p2
        low, high = secondary_bounds_m or (
            spec.pipe_diameter_m * 1e-4,
            spec.pipe_diameter_m * (1 - 1e-6),
        )
        if not 0 < low < high < spec.pipe_diameter_m:
            raise ValueError("secondary bounds must increase strictly inside the pipe diameter")

        def residual_geometry(coordinate: float) -> float:
            trial = replace(spec, secondary_dimension_m=coordinate * spec.pipe_diameter_m)
            return (
                _meter_evaluate(trial, fluid, mass, fixed_p1, fixed_p2, heat_capacity_ratio)[0]
                / mass
                - 1
            )

        value = _meter_root(
            residual_geometry, low / spec.pipe_diameter_m, high / spec.pipe_diameter_m
        )
        spec = replace(spec, secondary_dimension_m=value * spec.pipe_diameter_m)
    elif p2 is None:
        assert p1 is not None
        fixed_p1 = p1
        p2 = p1 * _meter_root(
            lambda ratio: (
                _meter_evaluate(spec, fluid, mass, fixed_p1, fixed_p1 * ratio, heat_capacity_ratio)[
                    0
                ]
                / mass
                - 1
            ),
            ratio_min,
            1,
        )
    else:
        assert p2 is not None
        p1 = p2 / _meter_root(
            lambda ratio: (
                _meter_evaluate(spec, fluid, mass, p2 / ratio, p2, heat_capacity_ratio)[0] / mass
                - 1
            ),
            ratio_min,
            1,
        )
    assert p1 is not None and p2 is not None
    prediction, reynolds, coefficient, epsilon = _meter_evaluate(
        spec, fluid, mass, p1, p2, heat_capacity_ratio
    )
    return DifferentialMeterResult(
        spec,
        mass,
        p1,
        p2,
        reynolds,
        coefficient,
        epsilon,
        spec.permanent_loss(p1 - p2, coefficient),
        prediction / mass - 1,
    )
