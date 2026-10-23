"""Inverse-temperature cubic profiles and empirical species transitions."""

from __future__ import annotations

from math import exp, log, log1p

from scipy.interpolate import CubicSpline


def chemistry_factor(height: float, ratio: float, center: float, *scales: float) -> float:
    arguments = [(height - center) / scale for scale in scales]
    if max(arguments) > 70:
        return 1.0
    if max(arguments) < -70:
        return exp(ratio)
    return exp(ratio / (1 + sum(exp(value) for value in arguments) / len(arguments)))


def mixed_density(
    diffusive: float, mixed: float, width: float, mean_mass: float, mass: float
) -> float:
    power = width / (mean_mass - mass)
    contrast = power * log(mixed / diffusive)
    if contrast < -10:
        return diffusive
    if contrast > 10:
        return mixed
    return diffusive * exp(log1p(exp(contrast)) / power)


class InverseTemperatureProfile:
    def __init__(
        self,
        heights: tuple[float, ...],
        temperatures: tuple[float, ...],
        gradients: tuple[float, float],
        radius: float,
        gravity: float,
    ) -> None:
        self.top, self.radius, self.gravity = heights[0], radius, gravity
        self.top_temperature = temperatures[0]
        self.span = self.potential(heights[-1])
        coords = [self.potential(height) / self.span for height in heights]
        boundary = (
            (1, -gradients[0] * self.span / temperatures[0] ** 2),
            (
                1,
                -gradients[1]
                * self.span
                / temperatures[-1] ** 2
                * ((radius + heights[-1]) / (radius + heights[0])) ** 2,
            ),
        )
        self.spline = CubicSpline(coords, [1 / t for t in temperatures], bc_type=boundary)

    def potential(self, height: float) -> float:
        return (height - self.top) * (self.radius + self.top) / (self.radius + height)

    def temperature(self, height: float) -> float:
        return 1 / float(self.spline(self.potential(height) / self.span))

    def density(self, height: float, initial: float, mass: float, alpha: float = 0) -> float:
        coordinate = self.potential(height) / self.span
        integral = float(self.spline.integrate(0, coordinate))
        exponent = (
            mass * self.gravity / (1 + self.top / self.radius) ** 2 * self.span / 831.4 * integral
        )
        return (
            initial
            * (self.top_temperature / self.temperature(height)) ** (1 + alpha)
            * exp(-min(50, exponent))
        )


class BatesProfile:
    def __init__(
        self,
        exosphere: float,
        base_temperature: float,
        base_height: float,
        slope: float,
        heights: tuple[float, ...],
        temperatures: tuple[float, ...],
        bottom_gradient: float,
        radius: float,
        gravity: float,
    ) -> None:
        self.exosphere, self.base_temperature = exosphere, base_temperature
        self.base_height, self.slope, self.top = base_height, slope, heights[0]
        self.radius, self.gravity = radius, gravity
        top_temperature = self.temperature_above(self.top)
        top_gradient = (
            (exosphere - top_temperature)
            * slope
            * ((radius + base_height) / (radius + self.top)) ** 2
        )
        self.lower = InverseTemperatureProfile(
            heights,
            (top_temperature, *temperatures[1:]),
            (top_gradient, bottom_gradient),
            radius,
            gravity,
        )

    def temperature_above(self, height: float) -> float:
        potential = (
            (height - self.base_height) * (self.radius + self.base_height) / (self.radius + height)
        )
        return self.exosphere - (self.exosphere - self.base_temperature) * exp(
            -self.slope * potential
        )

    def temperature(self, height: float) -> float:
        return (
            self.temperature_above(height) if height >= self.top else self.lower.temperature(height)
        )

    def density(self, height: float, initial: float, mass: float, alpha: float = 0) -> float:
        join = max(self.top, height)
        temperature = self.temperature_above(join)
        potential = (
            (join - self.base_height) * (self.radius + self.base_height) / (self.radius + join)
        )
        gamma = (
            mass
            * self.gravity
            / (1 + self.base_height / self.radius) ** 2
            / (self.slope * 831.4 * self.exosphere)
        )
        density = initial * (self.base_temperature / temperature) ** (1 + alpha + gamma)
        density *= min(50, exp(-self.slope * gamma * potential))
        if height < self.top:
            density = self.lower.density(max(height, 72.5), density, mass, alpha)
        return density
