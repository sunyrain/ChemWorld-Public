"""SI vessel heads from meridian and oblique-cone surface integrals.

Public engineering geometry equations were consulted. This implementation uses
one parameterized surface for volume, wetted area and level derivatives, without
importing a reference package. Head depths are positive; ``concave`` records the
direction separately. Horizontal head heights use the cylinder's bottom datum.
Deep spherical caps can extend below that datum and above the cylinder diameter.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import pairwise
from math import acos, asin, cos, hypot, isfinite, pi, sin, sqrt
from typing import Any

from scipy.integrate import quad

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.geometric_primitives import circular_segment_area

HEAD_KINDS = ("flat", "conical", "ellipsoidal", "spherical", "torispherical", "guppy")


def _integral(function: Callable[[float], float], low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(quad(function, low, high, epsabs=2e-12, epsrel=2e-11, limit=150)[0])


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class TankHead:
    """Dimensionless head prescription, reusable while sizing the vessel.

    ``depth_ratio`` is axial depth divided by cylinder diameter. Torispherical
    heads use crown/knuckle radii divided by diameter instead. A guppy head is
    the oblique cone whose apex lies above the top edge of its circular base.
    Its area is integrated on that actual surface, not an elliptical-cone
    shortcut. Flat heads have zero displaced volume and a circular end plate.
    """

    kind: str = "flat"
    depth_ratio: float = 0.25
    crown_ratio: float = 1.0
    knuckle_ratio: float = 0.06
    concave: bool = False

    def __post_init__(self) -> None:
        if self.kind not in HEAD_KINDS:
            raise ValueError(f"unknown tank head {self.kind!r}")
        if self.kind == "flat":
            if self.concave:
                raise ValueError("a flat head has no concavity")
        elif self.kind == "torispherical":
            if not (
                isfinite(self.crown_ratio)
                and isfinite(self.knuckle_ratio)
                and 0 < self.knuckle_ratio < 0.5
                and self.crown_ratio >= 0.5
                and self.crown_ratio > self.knuckle_ratio
            ):
                raise ValueError("physical crown >= D/2 and 0 < knuckle < D/2 required")
        else:
            _positive(self.depth_ratio, "head depth ratio")
        if self.concave and self.kind == "spherical" and self.depth_ratio > 0.5:
            raise ValueError("a deep spherical intrusion would intersect the cylindrical wall")

    @property
    def depth(self) -> float:
        """Depth for unit cylinder diameter; multiply by actual diameter."""
        if self.kind == "flat":
            return 0.0
        if self.kind == "torispherical":
            angle = asin((0.5 - self.knuckle_ratio) / (self.crown_ratio - self.knuckle_ratio))
            return self.crown_ratio * (1 - cos(angle)) + self.knuckle_ratio * cos(angle)
        return self.depth_ratio

    @property
    def radial_extent(self) -> float:
        if self.kind == "spherical" and self.depth > 0.5:
            return (self.depth**2 + 0.25) / (2 * self.depth)
        return 0.5

    def _segments(self) -> tuple[tuple[str, float, float], ...]:
        if self.kind == "conical":
            return (("conical", 0.0, 1.0),)
        if self.kind == "ellipsoidal":
            return (("ellipsoidal", 0.0, pi / 2),)
        if self.kind == "spherical":
            radius = (self.depth**2 + 0.25) / (2 * self.depth)
            end = acos(_clip(1 - self.depth / radius, -1, 1))
            return (("spherical", 0.0, end),)
        if self.kind == "torispherical":
            angle = asin((0.5 - self.knuckle_ratio) / (self.crown_ratio - self.knuckle_ratio))
            return (("crown", 0.0, angle), ("knuckle", angle, pi / 2))
        return ()

    def _meridian(self, segment: str, coordinate: float) -> tuple[float, float, float, float]:
        """Axial distance from tip, radius, axial derivative, arc derivative."""
        a = self.depth
        if segment == "conical":
            return a * coordinate, 0.5 * coordinate, a, hypot(a, 0.5)
        if segment == "ellipsoidal":
            return (
                a * (1 - cos(coordinate)),
                0.5 * sin(coordinate),
                a * sin(coordinate),
                hypot(a * sin(coordinate), 0.5 * cos(coordinate)),
            )
        if segment in ("spherical", "crown"):
            radius = (a * a + 0.25) / (2 * a) if segment == "spherical" else self.crown_ratio
            return (
                radius * (1 - cos(coordinate)),
                radius * sin(coordinate),
                radius * sin(coordinate),
                radius,
            )
        radius = self.knuckle_ratio
        return (
            a - radius * cos(coordinate),
            0.5 - radius + radius * sin(coordinate),
            radius * sin(coordinate),
            radius,
        )

    def _parameter_at_height(self, segment: str, height: float) -> float:
        if segment == "conical":
            return height / self.depth
        if segment == "ellipsoidal":
            return acos(_clip(1 - height / self.depth, -1, 1))
        if segment in ("spherical", "crown"):
            radius = (
                (self.depth**2 + 0.25) / (2 * self.depth)
                if segment == "spherical"
                else self.crown_ratio
            )
            return acos(_clip(1 - height / radius, -1, 1))
        return acos(_clip((self.depth - height) / self.knuckle_ratio, -1, 1))

    def _radius_cuts(self, segment: str, radius: float) -> tuple[float, ...]:
        if segment == "conical":
            return (2 * radius,)
        if segment == "ellipsoidal":
            return (asin(_clip(2 * radius, -1, 1)),)
        if segment == "knuckle":
            return (asin(_clip((radius - 0.5 + self.knuckle_ratio) / self.knuckle_ratio, -1, 1)),)
        scale = (
            (self.depth**2 + 0.25) / (2 * self.depth)
            if segment == "spherical"
            else self.crown_ratio
        )
        angle = asin(_clip(radius / scale, -1, 1))
        return angle, pi - angle

    @lru_cache(maxsize=128)  # noqa: B019 - bounded cache of immutable value objects
    def _full(self, measure: str) -> float:
        if self.kind == "flat":
            return pi / 4 if measure == "area" else 0.0
        if self.kind == "guppy":
            if measure == "volume":
                return pi * self.depth / 12
            return _integral(lambda t: 0.5 * hypot(self.depth, 0.5 * (1 - cos(t))), 0, pi)
        return self._axis_integral(measure, "vertical", self.depth)

    def total_volume(self, diameter_m: float) -> float:
        _positive(diameter_m, "diameter")
        return diameter_m**3 * self._full("volume")

    def total_area(self, diameter_m: float) -> float:
        _positive(diameter_m, "diameter")
        return diameter_m**2 * self._full("area")

    def _axis_integral(self, measure: str, orientation: str, height: float) -> float:
        result = 0.0
        cuts: tuple[float, ...]
        for segment, low, high in self._segments():
            if orientation == "vertical":
                high = min(high, self._parameter_at_height(segment, height))
                cuts = (low, high)
            else:
                cuts = tuple(
                    sorted(
                        {
                            low,
                            high,
                            *(
                                t
                                for t in self._radius_cuts(segment, abs(height - 0.5))
                                if low < t < high
                            ),
                        }
                    )
                )

            def value(t: float, segment: str = segment) -> float:
                _, radius, axial, arc = self._meridian(segment, t)
                if orientation == "vertical":
                    return pi * radius**2 * axial if measure == "volume" else 2 * pi * radius * arc
                y = height - 0.5
                if radius == 0:
                    return 0.0
                if measure == "volume":
                    return circular_segment_area(2 * radius, y + radius) * axial
                if measure == "cross":
                    return 2 * sqrt(max(0, radius**2 - y**2)) * axial
                angle = acos(_clip(-y / radius, -1, 1))
                return 2 * angle * radius * arc

            result += sum(_integral(value, left, right) for left, right in pairwise(cuts))
        return result

    def _guppy(self, measure: str, height: float) -> float:
        h = _clip(height, 0, 1)
        if measure == "volume":
            end = acos(1 - 2 * h)
            return _integral(lambda t: self.depth / 6 * (1 - cos(t)) * sin(t) ** 2, 0, end)
        if measure == "cross":
            return 4 * self.depth / 3 * h * sqrt(h * (1 - h))
        if h <= 0:
            return 0.0
        if h >= 1:
            return self._full("area")
        level = 2 * h - 1

        def value(t: float) -> float:
            c = cos(t)
            stop = (level - c) / (1 - c)
            return hypot(self.depth, 0.5 * (1 - c)) * (stop - stop**2 / 2)

        return _integral(value, acos(level), pi)

    def horizontal_measure(
        self, diameter_m: float, height_m: float, measure: str = "volume"
    ) -> float:
        """Head-only volume, wetted area or dV/dh; cylinder bottom is h=0.

        Concavity changes tank composition, not this positive head measure.
        Deep spherical caps are integrated over their actual radial envelope.
        """
        _positive(diameter_m, "diameter")
        if not isfinite(height_m) or measure not in ("volume", "area", "cross"):
            raise ValueError("finite height and volume/area/cross measure required")
        h = height_m / diameter_m
        power = 3 if measure == "volume" else 2
        if self.kind == "flat":
            result = circular_segment_area(1, h) if measure == "area" else 0.0
        elif self.kind == "guppy":
            result = self._guppy(measure, h)
        elif h <= 0.5 - self.radial_extent:
            result = 0.0
        elif h >= 0.5 + self.radial_extent:
            result = 0.0 if measure == "cross" else self._full(measure)
        elif h > 0.5 and measure != "cross":
            result = self._full(measure) - self._axis_integral(measure, "horizontal", 1 - h)
        else:
            result = self._axis_integral(measure, "horizontal", h)
        return result * diameter_m**power

    def vertical_measure(
        self, diameter_m: float, height_m: float, measure: str = "volume"
    ) -> float:
        """Convex cap measured upward from its tip, clipped at the base plane."""
        _positive(diameter_m, "diameter")
        if not isfinite(height_m) or measure not in ("volume", "area", "cross"):
            raise ValueError("finite height and volume/area/cross measure required")
        h = _clip(height_m / diameter_m, 0, self.depth)
        if self.kind == "flat":
            return pi * diameter_m**2 / 4 if measure == "area" and height_m >= 0 else 0.0
        if measure == "cross":
            if not 0 <= height_m / diameter_m <= self.depth:
                return 0.0
            if self.kind == "guppy":
                return pi * diameter_m**2 / 4 * (h / self.depth) ** 2
            for segment, low, high in self._segments():
                if self._meridian(segment, low)[0] <= h <= self._meridian(segment, high)[0] + 1e-15:
                    radius = self._meridian(segment, self._parameter_at_height(segment, h))[1]
                    return pi * (diameter_m * radius) ** 2
            return 0.0
        power = 3 if measure == "volume" else 2
        if h <= 0:
            return 0.0
        if self.kind == "guppy":
            return self._full(measure) * (h / self.depth) ** power * diameter_m**power
        return self._axis_integral(measure, "vertical", h) * diameter_m**power

    def vertical_capacity(self, diameter_m: float, height_m: float) -> float:
        """Bottom head plus unbounded straight shell; supports concave heads."""
        _positive(diameter_m, "diameter")
        if not isfinite(height_m):
            raise ValueError("finite height required")
        h = max(0, height_m)
        area = pi * diameter_m**2 / 4
        depth = self.depth * diameter_m
        if self.concave:
            return max(
                0,
                area * h
                - self.total_volume(diameter_m)
                + self.vertical_measure(diameter_m, depth - h),
            )
        return self.vertical_measure(diameter_m, h) + area * max(0, h - depth)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TankHead":
        return cls(**values)
