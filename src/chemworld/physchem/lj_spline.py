"""Lennard-Jones spline BH/WCA/UV equations on shared SI Helmholtz responses.

Equations are independently expressed after reading ThermoPack's published
models/source. Parameter provenance and the Apache notice accompany the data.
No reference runtime is imported. The fixed LJ/s cutoff has no tail correction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import isfinite, pi, sqrt
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    compensated_sum,
    derivative_depth,
    exponential,
    exponential_minus_one,
    hyperbolic_tangent,
    logarithm,
    nonnegative_power,
    scalar_value,
)
from chemworld.physchem.lj_spline_coefficients import BH, WCA
from chemworld.physchem.molecular_potential_extensions import _constant_zero, _PurePotential
from chemworld.physchem.saftvrmie import (
    MieSegment,
    mie_correction_coefficients,
    mie_effective_diameter,
)
from chemworld.physchem.second_order_pair import SecondOrderPair

SPLINE_START = (26 / 7) ** (1 / 6)
SPLINE_CUTOFF = 67 / 48 * SPLINE_START
WCA_MINIMUM = 2 ** (1 / 6)
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(128)
_F_ALPHA = mie_correction_coefficients(0.5357287350120993)


def lj_spline_potential(reduced_radius: Any) -> Any:
    """u/epsilon, including its fixed smooth cutoff, for any individual radius."""
    radius = scalar_value(reduced_radius)
    if isinstance(radius, bool) or not isfinite(radius) or radius <= 0:
        raise ValueError("LJ spline radius must be finite and positive")
    if radius < SPLINE_START:
        inverse6 = reduced_radius**-6
        return 4 * (inverse6**2 - inverse6)
    if radius < SPLINE_CUTOFF:
        offset = reduced_radius - SPLINE_CUTOFF
        return (
            -24192 / (3211 * SPLINE_START**2) * offset**2
            - 387072 / (61009 * SPLINE_START**3) * offset**3
        )
    return 0.0 * reduced_radius


def wca_diameter_ratio(reduced_temperature: Any) -> Any:
    """Explicit WCA reference diameter divided by sigma."""
    if scalar_value(reduced_temperature) <= 0:
        raise ValueError("WCA diameter needs positive temperature")
    return WCA_MINIMUM * (1 + reduced_temperature**0.5) ** (-1 / 6)


def _hard_sphere(packing: Any) -> Any:
    if not 0 <= scalar_value(packing) < 1:
        raise ValueError("LJ spline hard-sphere packing must lie in [0,1)")
    return (4 * packing - 3 * packing**2) / (1 - packing) ** 2


def _cavity(eta: Any, radius: Any) -> Any:
    """Hard-sphere cavity function and its C2 high-density continuation."""
    e = eta if scalar_value(eta) <= 0.52 else 0.52
    free = 1 - e
    a = (3 - e) / free**3 - 3
    b = -3 * e * (2 - e) / free**3
    c = logarithm((2 - e) / (2 * free**3)) - e * (2 - 6 * e + 3 * e**2) / free**3
    value = exponential(a + b * radius + c * radius**3)
    if scalar_value(eta) > 0.52:
        a1, b1 = -2 * (e - 4) / free**4, 3 * (e**2 - 2 * e - 2) / free**4
        c1 = -((2 * (e - 4) * e + 7) * e**2 + e + 1) / ((e - 2) * free**4)
        a2, b2 = -6 * (e - 5) / free**5, 6 * (e**2 - 2 * e - 5) / free**5
        c2 = -(2 * e**5 - 10 * e**4 + 5 * e**3 + 31 * e**2 - 29 * e - 11) / ((e - 2) ** 2 * free**5)
        first = a1 + b1 * radius + c1 * radius**3
        second = a2 + b2 * radius + c2 * radius**3
        offset = eta - e
        value *= 1 + first * offset + 0.5 * (first**2 + second) * offset**2
    return value


@lru_cache(maxsize=512)
def wca_cavity_helmholtz(reduced_temperature: Any, reduced_density: Any) -> Any:
    """Independent fixed quadrature of the WCA hard-sphere correction.

    Arguments and results are immutable numbers/jets. The bounded cache only
    shares this pure calculation between contribution-mask declarations; it
    stores no model, physical state or execution ownership.
    """
    t, rho = reduced_temperature, reduced_density
    diameter = wca_diameter_ratio(t)
    eta = pi * rho * diameter**3 / 6
    _hard_sphere(eta)
    if scalar_value(eta) == 0.52 and max(derivative_depth(t), derivative_depth(rho)) >= 3:
        raise ValueError("third cavity derivative at the C2 continuation boundary is undefined")
    terms = []
    for lower, upper, subtract in ((0.0, diameter, False), (diameter, WCA_MINIMUM, True)):
        half = (upper - lower) / 2
        middle = (upper + lower) / 2
        for node, weight in zip(_NODES, _WEIGHTS, strict=True):
            radius = middle + float(node) * half
            inverse6 = radius**-6
            exponent = -(4 * (inverse6**2 - inverse6) + 1) / t
            boltzmann = exponential_minus_one(exponent) if subtract else exponential(exponent)
            terms.append(
                float(weight) * half * _cavity(eta, radius / diameter) * boltzmann * radius**2
            )
    return -2 * pi * rho * compensated_sum(terms)


def _polynomial(coefficients: Sequence[float], value: Any) -> Any:
    return compensated_sum(c * value**i for i, c in enumerate(coefficients))


def _density_order(value: Any) -> int:
    """Count active density directions at vacuum, ignoring pure T directions."""
    if isinstance(value, FirstDerivative):
        return int(not _constant_zero(value.slope)) + _density_order(value.value)
    if isinstance(value, SecondOrderPair):
        return 0 if _constant_zero(value) else 2
    return 0


def _bh_polynomial(term: int, tau: Any, rho: Any) -> Any:
    return compensated_sum(
        BH[term][5 * i + j] * tau**i * rho ** (5 - j) for i in range(3) for j in range(5)
    )


def _wca_dispersion(term: int, y: Any, rho: Any) -> Any:
    coefficients = WCA[term]
    if term == 0:
        return (
            2
            * pi
            * compensated_sum(
                _polynomial(coefficients[3 * i : 3 * i + 3], y) * rho ** (i + 1) for i in range(4)
            )
        )
    if term == 2:
        p1 = _polynomial(coefficients[:3], y)
        p2 = _polynomial(coefficients[5:8], y)
        p3 = _polynomial(coefficients[3:5], y)
        p4 = _polynomial(coefficients[8:10], y)
        p5 = _polynomial(coefficients[11:13], y)
        argument = p2 * nonnegative_power(rho, p4) + p3 * nonnegative_power(rho, p5)
        return rho * p1 * (1 + coefficients[10] * hyperbolic_tangent(argument))
    p1, p2, p3, p4 = (_polynomial(coefficients[3 * i : 3 * i + 3], y) for i in range(4))
    if term == 1:
        argument = p2 * (nonnegative_power(rho, coefficients[13]) + p3 * nonnegative_power(rho, p4))
        return p1 * rho - rho * (p1 + coefficients[12]) * hyperbolic_tangent(argument)
    argument = p4 * nonnegative_power(rho, coefficients[13])
    return (
        p1 * nonnegative_power(rho, coefficients[12] + 1)
        + p2 * rho
        + rho * p3 * hyperbolic_tangent(argument)
    )


class _LJSpline(_PurePotential):
    def potential_energy_over_k_K(self, separation_m: Any) -> Any:  # noqa: N802 - SI suffix
        return self.well_depth_K * lj_spline_potential(separation_m / self.diameter_m)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return compensated_sum(self.contributions(temperature, density, fractions).values())

    def contributions(
        self, temperature: Any, density: Any, fractions: Sequence[Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _validate_flags(self, names: Sequence[str]) -> None:
        if any(type(getattr(self, name)) is not bool for name in names):
            raise ValueError("LJ spline model controls must be boolean")


@dataclass(frozen=True)
class LJSplineBHModel(_LJSpline):
    use_lafitte_a3: bool = False
    enable_chi_correction: bool = True
    enable_hs: bool = True
    enable_a1: bool = True
    enable_a2: bool = True
    enable_a3: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_flags(
            (
                "use_lafitte_a3",
                "enable_chi_correction",
                "enable_hs",
                "enable_a1",
                "enable_a2",
                "enable_a3",
            )
        )

    @cached_property
    def _segment(self) -> MieSegment:
        return MieSegment(
            self.component_id, 1.0, self.diameter_m, self.well_depth_K, self.source, 12.0, 6.0
        )

    def diameter_ratio(self, temperature: Any) -> Any:
        return mie_effective_diameter(self._segment, temperature) / self.diameter_m

    def contributions(
        self, temperature: Any, density: Any, fractions: Sequence[Any]
    ) -> dict[str, Any]:
        t, rho = self._reduced(temperature, density, fractions)
        diameter = self.diameter_ratio(temperature)
        eta = pi * rho * diameter**3 / 6
        hard = _hard_sphere(eta)
        tau = 1 - diameter
        terms = {"hard_sphere": hard if self.enable_hs else 0.0, "a1": 0.0, "a2": 0.0, "a3": 0.0}
        if self.enable_a1:
            terms["a1"] = 2 * pi * _bh_polynomial(0, tau, rho) / t
        if self.enable_a2:
            chi = _bh_polynomial(3, tau, rho) if self.enable_chi_correction else 0.0
            khs = (1 - eta) ** 4 / (1 + 4 * eta + 4 * eta**2 - 4 * eta**3 + eta**4)
            terms["a2"] = -pi * _bh_polynomial(1, tau, rho) * khs * (1 + chi) / t**2
        if self.enable_a3:
            if self.use_lafitte_a3:
                packing = pi * rho / 6
                terms["a3"] = (
                    -_F_ALPHA[3]
                    * packing
                    * exponential(_F_ALPHA[4] * packing + _F_ALPHA[5] * packing**2)
                    / t**3
                )
            else:
                terms["a3"] = _bh_polynomial(2, tau, rho) / t**3
        return terms


@dataclass(frozen=True)
class LJSplineWCAModel(_LJSpline):
    enable_cavity: bool = True
    enable_hs: bool = True
    enable_a1: bool = True
    enable_a2: bool = True
    enable_a3: bool = True
    enable_a4: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_flags(
            ("enable_cavity", "enable_hs", "enable_a1", "enable_a2", "enable_a3", "enable_a4")
        )

    def diameter_ratio(self, temperature: Any) -> Any:
        return wca_diameter_ratio(temperature / self.well_depth_K)

    def contributions(
        self, temperature: Any, density: Any, fractions: Sequence[Any]
    ) -> dict[str, Any]:
        t, rho = self._reduced(temperature, density, fractions)
        diameter = wca_diameter_ratio(t)
        hard = _hard_sphere(pi * rho * diameter**3 / 6)
        terms = {
            "hard_sphere": hard if self.enable_hs else 0.0,
            "cavity": 0.0,
            "a1": 0.0,
            "a2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
        }
        if self.enable_hs and self.enable_cavity:
            terms["cavity"] = wca_cavity_helmholtz(t, rho)
        if _constant_zero(rho):
            return terms
        y = WCA_MINIMUM - diameter
        if scalar_value(rho) == 0 and any((self.enable_a2, self.enable_a3, self.enable_a4)):
            if _density_order(rho) >= 2:
                raise ValueError(
                    "noninteger WCA terms have no finite vacuum second density derivative"
                )
            # Each nonanalytic correction starts beyond rho^1. Its exact
            # first-density limit retains all simultaneous T derivatives.
            linear = (
                2 * pi * _polynomial(WCA[0][:3], y),
                _polynomial(WCA[1][:3], y),
                _polynomial(WCA[2][:3], y),
                _polynomial(WCA[3][3:6], y),
            )
            for i in range(4):
                if getattr(self, f"enable_a{i + 1}"):
                    terms[f"a{i + 1}"] = rho * linear[i] / t ** (i + 1)
            return terms
        for i in range(4):
            if getattr(self, f"enable_a{i + 1}"):
                terms[f"a{i + 1}"] = _wca_dispersion(i, WCA_MINIMUM - diameter, rho) / t ** (i + 1)
        return terms


@dataclass(frozen=True)
class LJSplineUVModel(_LJSpline):
    use_temperature_dependent_u_fraction: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_flags(("use_temperature_dependent_u_fraction",))

    def diameter_ratio(self, temperature: Any) -> Any:
        return wca_diameter_ratio(temperature / self.well_depth_K)

    def contributions(
        self, temperature: Any, density: Any, fractions: Sequence[Any]
    ) -> dict[str, Any]:
        t, rho = self._reduced(temperature, density, fractions)
        diameter = wca_diameter_ratio(t)
        hard = _hard_sphere(pi * rho * diameter**3 / 6)
        cavity = wca_cavity_helmholtz(t, rho)
        cs = (
            -0.8513603,
            0.2977649,
            -0.1287404,
            -0.3377645,
            0.7343770,
            -0.9839062,
            -0.4592915,
            0.7963438,
        )
        a1 = (
            2
            * pi
            * rho
            * compensated_sum((cs[2 * i] + cs[2 * i + 1] * diameter**3) * rho**i for i in range(4))
            / t
        )
        virial_a1 = 2 * pi * (cs[0] + cs[1] * diameter**3) / t
        virial_ljs = _polynomial((1.345, -1.336, -3.85, 1.295, -0.416), 1 / t)
        coefficients = (
            1.92840364363978 + 5 * 0.443165896265079,
            0.520120816141761
            + 5 * 0.182526759234412
            + 25 * 0.0110319989659929
            - 125 * 0.0000797813995328348,
            5 * 0.0129885156087242 + 25 * 0.00641039871789327 + 125 * 0.0000185866741090323,
        )
        q = WCA_MINIMUM * (
            1
            + sqrt(4 * pi) * t**0.5
            + coefficients[0] * t
            + coefficients[1] * t**1.5
            + coefficients[2] * t**2
        ) ** (-1 / 24)
        delta_b2 = (virial_ljs - 2 * pi * q**3 / 3 - virial_a1) * rho
        if scalar_value(rho) == 0:
            if _density_order(rho) >= 3:
                raise ValueError(
                    "UV fractional virial crossover has no finite vacuum third density derivative"
                )
            # The omitted terms multiply delta_b2~rho and start above rho^2.
            fraction: Any = 0.0 if self.use_temperature_dependent_u_fraction else 0.11072527 * rho
        elif self.use_temperature_dependent_u_fraction:
            sig = sqrt(0.436593) / t / (1 + 0.436593 / t**2) ** 0.5
            fraction = (1.01236 + sig * (0.99164 - 1.01236)) * hyperbolic_tangent(
                2.63226 * nonnegative_power(rho, 1.21219)
                + 2.25005 * nonnegative_power(rho, 3.91411)
            )
        else:
            fraction = hyperbolic_tangent(
                0.11072527 * rho
                + 2.5809656 * nonnegative_power(rho, 1.2333385)
                + 2.3534453 * nonnegative_power(rho, 4.124129)
            )
        return {
            "hard_sphere": hard,
            "cavity": cavity,
            "a1": a1,
            "virial": (1 - fraction) * delta_b2,
        }
