"""Liquid differential-pressure orifices, with recoverable pressure separated.

ISO 5167-2:2003 / Reader-Harris--Gallagher, or explicitly calibrated C.
These are engineering constitutive laws, not owners of process inventory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from math import exp, isfinite, pi, sqrt
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)


def _positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _engineering_root(
    function: Callable[[float], float],
    initial: float,
    lower: float,
    upper: float,
    *,
    name: str,
    unit: str,
    scale: float,
) -> float:
    """Adapt one engineering residual to the existing shared root strategy."""
    variables = (ResidualVariable(name, unit, scale, "algebraic", lower, upper),)

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            return np.array([function(float(values[0]))])
        except ValueError as exc:
            raise ResidualDomainError(str(exc)) from exc

    result = solve_root(
        residual,
        [min(max(initial, lower), upper)],
        jacobian=lambda x: finite_jacobian(
            residual,
            x,
            [scale],
            lower=np.array([lower]),
            upper=np.array([upper]),
        ),
        variables=variables,
        residual_scales=[1.0],
        policy=RootPolicy(tolerance=2e-11, maximum_iterations=80),
    )
    if not result.success:
        raise ValueError(f"{name} inverse failed: {result.failure_reason}")
    return float(cast(NDArray[np.float64], result.value)[0])


def orifice_discharge_coefficient(
    reynolds: float,
    *,
    pipe_diameter_m: float,
    beta: float,
    taps: str = "corner",
) -> float:
    """Pure RHG equation; installation/Reynolds acceptance belongs to the spec."""
    _positive(reynolds, "Reynolds number")
    _positive(pipe_diameter_m, "pipe diameter")
    if not 0 < beta < 1 or taps not in {"corner", "flange", "D-D/2"}:
        raise ValueError("RHG requires physical beta and declared standard tap geometry")
    b, diameter = beta, pipe_diameter_m
    l1, l2 = {
        "corner": (0.0, 0.0),
        "D-D/2": (1.0, 0.47),
        "flange": (0.0254 / diameter, 0.0254 / diameter),
    }[taps]
    a = (19000 * b / reynolds) ** 0.8
    m2 = 2 * l2 / (1 - b)
    coefficient = (
        0.5961
        + 0.0261 * b**2
        - 0.216 * b**8
        + 0.000521 * (1e6 * b / reynolds) ** 0.7
        + (0.0188 + 0.0063 * a) * b**3.5 * (1e6 / reynolds) ** 0.3
        + (0.043 + 0.080 * exp(-10 * l1) - 0.123 * exp(-7 * l1))
        * (1 - 0.11 * a)
        * b**4
        / (1 - b**4)
        - 0.031 * (m2 - 0.8 * m2**1.1) * b**1.3
    )
    if diameter < 0.07112:
        coefficient += 0.011 * (0.75 - b) * (2.8 - diameter / 0.0254)
    return coefficient


@dataclass(frozen=True)
class LiquidOrificeSpec:
    pipe_diameter_m: float
    bore_diameter_m: float
    method: str = "iso5167"
    taps: str = "corner"
    discharge_coefficient: float | None = None
    reynolds_bounds: tuple[float, float] | None = None
    installation_note: str = ""

    def __post_init__(self) -> None:
        _positive(self.pipe_diameter_m, "pipe diameter")
        _positive(self.bore_diameter_m, "bore diameter")
        if not 0 < self.beta < 1:
            raise ValueError("orifice bore must be smaller than pipe")
        if not self.installation_note.strip():
            raise ValueError("declare standard-compliant installation or calibration scope")
        if self.method == "iso5167":
            if self.taps not in {"corner", "flange", "D-D/2"}:
                raise ValueError("ISO orifice requires corner, flange or D-D/2 taps")
            if not (
                0.05 <= self.pipe_diameter_m <= 1
                and self.bore_diameter_m >= 0.0125 - 1e-16
                and 0.1 - 1e-14 <= self.beta <= 0.75 + 1e-14
            ):
                raise ValueError("orifice geometry outside ISO 5167 domain")
            if self.discharge_coefficient is not None or self.reynolds_bounds is not None:
                raise ValueError("ISO coefficient and Reynolds domain are determined by standard")
        elif self.method == "calibrated":
            if self.discharge_coefficient is None or not (
                isfinite(self.discharge_coefficient) and 0 < self.discharge_coefficient <= 1
            ):
                raise ValueError("calibrated orifice requires 0 < C <= 1")
            if self.reynolds_bounds is None or len(self.reynolds_bounds) != 2:
                raise ValueError("calibrated orifice requires its Reynolds validity interval")
            lo, hi = self.reynolds_bounds
            if not all(isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi:
                raise ValueError("invalid calibration Reynolds interval")
        else:
            raise ValueError("unknown orifice method")

    @property
    def beta(self) -> float:
        return self.bore_diameter_m / self.pipe_diameter_m

    @property
    def minimum_reynolds(self) -> float:
        if self.method == "calibrated":
            return cast(tuple[float, float], self.reynolds_bounds)[0]
        if self.taps == "flange":
            return max(5000.0, 170000 * self.beta**2 * self.pipe_diameter_m)
        return max(5000.0, 16000 * self.beta**2)

    def coefficient(self, reynolds: float) -> float:
        _positive(reynolds, "Reynolds number")
        if reynolds < self.minimum_reynolds * (1 - 1e-13):
            raise ValueError("orifice flow below its Reynolds validity domain")
        if self.method == "calibrated":
            if reynolds > cast(tuple[float, float], self.reynolds_bounds)[1]:
                raise ValueError("orifice flow above its calibration domain")
            return cast(float, self.discharge_coefficient)
        return orifice_discharge_coefficient(
            reynolds,
            pipe_diameter_m=self.pipe_diameter_m,
            beta=self.beta,
            taps=self.taps,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LiquidOrificeSpec:
        data = dict(value)
        if data.get("reynolds_bounds") is not None:
            data["reynolds_bounds"] = tuple(data["reynolds_bounds"])
        return cls(**data)


@dataclass(frozen=True)
class LiquidOrificeResult:
    mass_flow_kg_s: float
    reynolds: float
    discharge_coefficient: float | None
    differential_pressure_Pa: float
    permanent_pressure_loss_Pa: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def liquid_orifice_pressure(
    spec: LiquidOrificeSpec,
    *,
    mass_flow_kg_s: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
) -> LiquidOrificeResult:
    _positive(density_kg_m3, "density")
    _positive(viscosity_Pa_s, "viscosity")
    if not isfinite(mass_flow_kg_s) or mass_flow_kg_s < 0:
        raise ValueError("liquid orifice supports finite forward flow")
    if mass_flow_kg_s == 0:
        return LiquidOrificeResult(0.0, 0.0, None, 0.0, 0.0, spec.method)
    reynolds = 4 * mass_flow_kg_s / (pi * spec.pipe_diameter_m * viscosity_Pa_s)
    coefficient = spec.coefficient(reynolds)
    area = pi * spec.bore_diameter_m**2 / 4
    difference = (
        (mass_flow_kg_s / (coefficient * area)) ** 2 * (1 - spec.beta**4) / (2 * density_kg_m3)
    )
    if spec.method == "iso5167" and difference > 250000 * (1 + 1e-12):
        raise ValueError("ISO orifice differential pressure exceeds 250 kPa")
    recovery = sqrt(1 - spec.beta**4 * (1 - coefficient**2))
    permanent = (
        difference
        * (recovery - coefficient * spec.beta**2)
        / (recovery + coefficient * spec.beta**2)
    )
    return LiquidOrificeResult(
        mass_flow_kg_s,
        reynolds,
        coefficient,
        difference,
        permanent,
        spec.method,
    )


def liquid_orifice_mass_flow(
    spec: LiquidOrificeSpec,
    *,
    differential_pressure_Pa: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
) -> LiquidOrificeResult:
    _positive(density_kg_m3, "density")
    _positive(viscosity_Pa_s, "viscosity")
    if not isfinite(differential_pressure_Pa) or differential_pressure_Pa < 0:
        raise ValueError("orifice differential pressure must be finite and nonnegative")
    if differential_pressure_Pa == 0:
        return liquid_orifice_pressure(
            spec, mass_flow_kg_s=0, density_kg_m3=density_kg_m3, viscosity_Pa_s=viscosity_Pa_s
        )
    if spec.method == "iso5167" and differential_pressure_Pa > 250000:
        raise ValueError("ISO orifice differential pressure exceeds 250 kPa")
    ideal_mass = (
        pi
        * spec.bore_diameter_m**2
        / 4
        * sqrt(2 * density_kg_m3 * differential_pressure_Pa / (1 - spec.beta**4))
    )
    if spec.method == "calibrated":
        mass = ideal_mass * cast(float, spec.discharge_coefficient)
    else:
        minimum = spec.minimum_reynolds * pi * spec.pipe_diameter_m * viscosity_Pa_s / 4
        if minimum > ideal_mass:
            raise ValueError("requested differential pressure implies sub-domain Reynolds flow")

        def residual(mass: float) -> float:
            re = 4 * mass / (pi * spec.pipe_diameter_m * viscosity_Pa_s)
            return mass / ideal_mass - spec.coefficient(re)

        if residual(minimum) > 1e-12:
            raise ValueError("requested differential pressure implies sub-domain Reynolds flow")
        mass = _engineering_root(
            residual,
            ideal_mass * 0.62,
            minimum,
            ideal_mass,
            name="orifice_mass_flow",
            unit="kg/s",
            scale=ideal_mass,
        )
    return liquid_orifice_pressure(
        spec, mass_flow_kg_s=mass, density_kg_m3=density_kg_m3, viscosity_Pa_s=viscosity_Pa_s
    )


def liquid_orifice_bore(
    spec: LiquidOrificeSpec,
    *,
    mass_flow_kg_s: float,
    differential_pressure_Pa: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
) -> float:
    """Solve the bore within the same method and declared installation family."""
    for name, value in (
        ("flow", mass_flow_kg_s),
        ("difference", differential_pressure_Pa),
        ("density", density_kg_m3),
        ("viscosity", viscosity_Pa_s),
    ):
        _positive(value, name)
    diameter = spec.pipe_diameter_m
    if spec.method == "iso5167":
        if differential_pressure_Pa > 250000:
            raise ValueError("ISO differential pressure exceeds 250 kPa")
        re = 4 * mass_flow_kg_s / (pi * diameter * viscosity_Pa_s)
        minimum, maximum = max(0.1, 0.0125 / diameter), 0.75
        if re < 5000:
            raise ValueError("orifice bore inverse is below ISO Reynolds domain")
        maximum = min(maximum, sqrt(re / (170000 * diameter if spec.taps == "flange" else 16000)))
    else:
        minimum, maximum = 1e-5, 1 - 1e-8
    if minimum >= maximum:
        raise ValueError("no admissible orifice bore interval")

    def residual(beta: float) -> float:
        candidate = replace(spec, bore_diameter_m=beta * diameter)
        re = 4 * mass_flow_kg_s / (pi * diameter * viscosity_Pa_s)
        coefficient = candidate.coefficient(re)
        # Evaluate the equation at trial geometry; final acceptance checks the
        # measured-pressure domain, not an artificial limit on rejected trials.
        prediction = (
            coefficient
            * pi
            * (beta * diameter) ** 2
            / 4
            * sqrt(2 * density_kg_m3 * differential_pressure_Pa / (1 - beta**4))
        )
        return prediction / mass_flow_kg_s - 1

    if residual(minimum) > 0 or residual(maximum) < 0:
        raise ValueError("requested orifice bore is outside the admitted geometry")
    beta = _engineering_root(
        residual,
        spec.beta,
        minimum,
        maximum,
        name="orifice_diameter_ratio",
        unit="dimensionless",
        scale=0.5,
    )
    return beta * diameter


def liquid_orifice_pressure_endpoint(
    spec: LiquidOrificeSpec,
    *,
    known_pressure_Pa: float,
    known_side: str,
    mass_flow_kg_s: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
) -> float:
    _positive(known_pressure_Pa, "known absolute pressure")
    result = liquid_orifice_pressure(
        spec,
        mass_flow_kg_s=mass_flow_kg_s,
        density_kg_m3=density_kg_m3,
        viscosity_Pa_s=viscosity_Pa_s,
    )
    if known_side not in {"upstream", "downstream"}:
        raise ValueError("known_side must be upstream or downstream")
    answer = (
        known_pressure_Pa
        + (-1 if known_side == "upstream" else 1) * result.differential_pressure_Pa
    )
    _positive(answer, "solved absolute pressure")
    return answer
