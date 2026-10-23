"""Horizontal engineering flow maps with explicit, sourced chart boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import cos, isfinite, log10, pi, radians, sqrt
from typing import Any, cast

from scipy.interpolate import BSpline, make_interp_spline

from chemworld.foundation.units import convert_value
from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.friction_correlations import pipe_friction_factor
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.two_phase_holdup import TwoPhaseProperties, _quality, _viscosities


@dataclass(frozen=True)
class TwoPhaseRegimeResult:
    method: str
    regime: str
    liquid_superficial_velocity_m_s: float
    vapor_superficial_velocity_m_s: float
    parameters: dict[str, float]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _velocities(
    p: TwoPhaseProperties, mass: float, diameter: float, quality: float
) -> tuple[float, float]:
    _positive(mass, "mass flow")
    _positive(diameter, "diameter")
    _quality(quality, interior=True)
    area = pi * diameter * diameter / 4
    return (
        mass * (1 - quality) / (p.liquid_density_kg_m3 * area),
        mass * quality / (p.vapor_density_kg_m3 * area),
    )


def mandhane_gregory_aziz_regime(
    properties: TwoPhaseProperties,
    *,
    mass_flow_kg_s: float,
    diameter_m: float,
    vapor_quality: float,
) -> TwoPhaseRegimeResult:
    """Six-regime horizontal map; returned velocities are SI, unlike source ft/s."""
    p = properties
    mu_l, mu_g = _viscosities(p)
    if p.surface_tension_N_m is None:
        raise ValueError("Mandhane map requires surface tension")
    speed_l, speed_g = _velocities(p, mass_flow_kg_s, diameter_m, vapor_quality)
    liquid, gas = speed_l / 0.3048, speed_g / 0.3048
    capillary = (p.liquid_density_kg_m3 * 0.0724 / (999.552 * p.surface_tension_N_m)) ** 0.25
    gas_scale = (p.vapor_density_kg_m3 / 1.294292) ** 0.333 * capillary * (mu_g * 180000) ** 0.2
    liquid_scale = capillary * (mu_l * 1000) ** 0.2
    if liquid >= 14 * liquid_scale:
        regime = (
            "dispersed bubble"
            if gas <= 230 * (liquid / 14) ** 0.206 * gas_scale
            else "annular mist"
        )
    else:
        for limit, multiplier, reference, exponent in (
            (0.1, 14, 0.1, -0.368),
            (0.2, 14, 0.1, -0.415),
            (1.15, 10.5, 0.2, -0.816),
            (4.8, 2.5, 1, 0),
            (float("inf"), 2.5, 4.8, 0.248),
        ):
            if liquid <= limit:
                low_boundary = multiplier * (liquid / reference) ** exponent * gas_scale
                break
        for limit, multiplier, reference, exponent in (
            (0.1, 70, 0.01, -0.0675),
            (0.3, 60, 0.1, -0.415),
            (0.56, 38, 0.3, 0.0813),
            (1, 40, 0.56, 0.385),
            (2.5, 50, 1, 0.756),
            (float("inf"), 100, 2.5, 0.463),
        ):
            if liquid <= limit:
                high_boundary = multiplier * (liquid / reference) ** exponent * gas_scale
                break
        if gas <= low_boundary:
            regime = "elongated bubble" if liquid >= 0.5 / liquid_scale else "stratified"
        elif gas <= high_boundary:
            regime = "slug" if liquid > 0.3 * liquid_scale else "wave"
        else:
            regime = "annular mist"
    return TwoPhaseRegimeResult(
        "Mandhane-Gregory-Aziz",
        regime,
        speed_l,
        speed_g,
        {"gas_scale": gas_scale, "liquid_scale": liquid_scale},
        ("horizontal empirical flow map; input completeness is not material validity",),
    )


@dataclass(frozen=True)
class LogRegimeBoundary:
    """Caller-owned log10(X)/log10(boundary) ScientificTable, cubic by declaration.

    No copyrighted chart samples or spline coefficients are distributed here.
    Interpolation is explicitly in log coordinates, with not-a-knot endpoints.
    """

    source_curve: TabulatedScalar
    _spline: BSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        curve = self.source_curve
        if (
            len(curve.axes) != 1
            or curve.axes[0][0] != "log10_X"
            or (curve.axes[0][2] != "dimensionless" or curve.output_unit != "dimensionless")
        ):
            raise ValueError("boundary requires a dimensionless log10_X axis and log10 output")
        selected = set(curve.row_ids)
        rows = [row for row in curve.table.rows if row.row_id in selected]
        pairs = sorted(
            (
                convert_value(
                    cast(float, row.values[curve.axes[0][1]].value),
                    row.values[curve.axes[0][1]].unit,
                    "dimensionless",
                ),
                convert_value(
                    cast(float, row.values[curve.output_column].value),
                    row.values[curve.output_column].unit,
                    "dimensionless",
                ),
            )
            for row in rows
        )
        if len(pairs) < 4:
            raise ValueError("cubic boundary needs at least four distinct log-coordinate rows")
        object.__setattr__(
            self,
            "_spline",
            make_interp_spline(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                k=3,
            ),
        )

    def value(self, martinelli: float, *, allow_extrapolation: bool = False) -> float:
        _positive(martinelli, "Martinelli map coordinate")
        coordinate = log10(martinelli)
        lo, hi = self.source_curve.domain["log10_X"]
        if not allow_extrapolation and not lo <= coordinate <= hi:
            raise ValueError("flow-map point outside declared chart boundary domain")
        result = 10 ** float(self._spline(coordinate))
        _positive(result, "chart boundary")
        return result


@dataclass(frozen=True)
class TaitelDuklerBoundaries:
    annular: LogRegimeBoundary
    wavy: LogRegimeBoundary
    bubbly: LogRegimeBoundary


def taitel_dukler_regime(
    properties: TwoPhaseProperties,
    boundaries: TaitelDuklerBoundaries,
    *,
    mass_flow_kg_s: float,
    diameter_m: float,
    vapor_quality: float,
    inclination_degrees: float = 0.0,
    roughness_m: float = 0.0,
    gravity_m_s2: float = 9.80665,
    allow_chart_extrapolation: bool = False,
) -> TwoPhaseRegimeResult:
    """Five-regime horizontal/near-horizontal chart with explicit source curves.

    This reference convention uses Colebrook/2040, whereas the original paper
    used Blasius. The chosen convention is reported and is not hidden parity.
    """
    p = properties
    mu_l, mu_g = _viscosities(p)
    speed_l, speed_g = _velocities(p, mass_flow_kg_s, diameter_m, vapor_quality)
    _positive(gravity_m_s2, "gravity")
    if not isfinite(inclination_degrees) or not -90 < inclination_degrees < 90:
        raise ValueError("Taitel-Dukler requires nonvertical inclination (-90, 90)")
    if not isfinite(roughness_m) or not 0 <= roughness_m < diameter_m:
        raise ValueError("roughness must lie in [0, diameter)")
    re_l = speed_l * diameter_m * p.liquid_density_kg_m3 / mu_l
    re_g = speed_g * diameter_m * p.vapor_density_kg_m3 / mu_g
    loss_l = (
        pipe_friction_factor(re_l, roughness_m / diameter_m)
        / diameter_m
        * p.liquid_density_kg_m3
        * speed_l**2
        / 2
    )
    loss_g = (
        pipe_friction_factor(re_g, roughness_m / diameter_m)
        / diameter_m
        * p.vapor_density_kg_m3
        * speed_g**2
        / 2
    )
    buoyancy = (
        (p.liquid_density_kg_m3 - p.vapor_density_kg_m3)
        * gravity_m_s2
        * cos(radians(inclination_degrees))
    )
    martinelli, turbulence = sqrt(loss_l / loss_g), sqrt(loss_l / buoyancy)
    froude = speed_g * sqrt(p.vapor_density_kg_m3 / (diameter_m * buoyancy))
    wave = froude * sqrt(re_l)
    annular = boundaries.annular.value(martinelli, allow_extrapolation=allow_chart_extrapolation)
    if froude >= annular:
        if martinelli <= 1.7917:
            regime = "annular"
        else:
            bubbly = boundaries.bubbly.value(
                martinelli, allow_extrapolation=allow_chart_extrapolation
            )
            regime = "bubbly" if turbulence >= bubbly else "intermittent"
    else:
        wavy = boundaries.wavy.value(martinelli, allow_extrapolation=allow_chart_extrapolation)
        regime = "stratified wavy" if wave >= wavy else "stratified smooth"
    warnings: tuple[str, ...] = (
        "near-horizontal map; reference Colebrook convention, original paper used Blasius",
    )
    if allow_chart_extrapolation:
        warnings += ("chart extrapolation explicitly enabled; not physical qualification",)
    return TwoPhaseRegimeResult(
        "Taitel-Dukler",
        regime,
        speed_l,
        speed_g,
        {"X": martinelli, "T": turbulence, "F": froude, "K": wave},
        warnings,
    )
