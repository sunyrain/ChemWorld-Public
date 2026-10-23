"""Stable bounded geometric measures; dimensions are SI, ratios dimensionless."""

from dataclasses import dataclass, field
from math import atan2, isfinite, pi, sin, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _engineering_root, _positive


def cylinder_geometry(
    diameter_m: float, length_m: float, *, holes: tuple[tuple[float, int], ...] = ()
) -> tuple[float, float]:
    """Volume and total area including end faces and through-hole walls.

    Hole counts/areas are checked; positions are unspecified, so spatial hole
    overlap and wall thickness are not certified by this aggregate measure.
    """
    _positive(diameter_m, "diameter")
    _positive(length_m, "length")
    cap = pi * diameter_m**2 / 4
    perimeter = pi * diameter_m
    for bore, count in holes:
        _positive(bore, "hole diameter")
        if bore >= diameter_m or isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError("physical bore and positive whole hole count required")
        cap -= count * pi * bore**2 / 4
        perimeter += count * pi * bore
    if cap <= 0:
        raise ValueError("holes consume all cross-sectional area")
    return cap * length_m, 2 * cap + perimeter * length_m


def circular_segment_area(diameter_m: float, height_m: float) -> float:
    _positive(diameter_m, "diameter")
    if not isfinite(height_m):
        raise ValueError("finite height required")
    if height_m <= 0:
        return 0.0
    radius = diameter_m / 2
    if height_m >= diameter_m:
        return pi * radius**2
    if height_m > radius:
        return pi * radius**2 - circular_segment_area(diameter_m, diameter_m - height_m)
    theta = atan2(sqrt(height_m * (diameter_m - height_m)), radius - height_m)
    measure = (
        2 * theta**3 / 3 - 2 * theta**5 / 15 + 4 * theta**7 / 315
        if theta < 1e-3
        else theta - 0.5 * sin(2 * theta)
    )
    return radius**2 * measure


def circular_segment_height(diameter_m: float, area_m2: float) -> float:
    _positive(diameter_m, "diameter")
    total = pi * diameter_m**2 / 4
    if not isfinite(area_m2) or not 0 <= area_m2 <= total:
        raise ValueError("segment area outside physical circle")
    if area_m2 == 0:
        return 0.0
    if area_m2 == total:
        return diameter_m
    original_fraction = area_m2 / total
    fraction = min(original_fraction, 1 - original_fraction)
    estimate = (3 * pi * fraction / 16) ** (2 / 3)

    def residual(coordinate: float) -> float:
        return circular_segment_area(1, estimate * coordinate) / (pi / 4 * fraction) - 1

    height = estimate * _engineering_root(
        residual,
        1,
        0,
        min(2, 1 / estimate),
        name="circle_segment_height_scaled",
        unit="dimensionless",
        scale=1,
    )
    return diameter_m * (1 - height if original_fraction > 0.5 else height)


def partial_sphere_geometry(diameter_m: float, height_m: float) -> tuple[float, float]:
    """Filled volume and wetted spherical surface, excluding free-surface disk."""
    _positive(diameter_m, "diameter")
    if not isfinite(height_m):
        raise ValueError("finite sphere height required")
    height = min(diameter_m, max(0, height_m))
    return pi * height**2 * (diameter_m / 2 - height / 3), pi * diameter_m * height


def shape_factor(
    kind: str,
    *,
    area_m2: float | None = None,
    volume_m3: float | None = None,
    perimeter_m: float | None = None,
    minimum_diameter_m: float | None = None,
    maximum_diameter_m: float | None = None,
) -> float:
    if kind == "sphericity":
        if area_m2 is None or volume_m3 is None:
            raise ValueError("surface and volume required")
        _positive(area_m2, "surface area")
        _positive(volume_m3, "volume")
        return pi ** (1 / 3) * (6 * volume_m3) ** (2 / 3) / area_m2
    if kind == "circularity":
        if area_m2 is None or perimeter_m is None:
            raise ValueError("area and perimeter required")
        _positive(area_m2, "area")
        _positive(perimeter_m, "perimeter")
        return 4 * pi * area_m2 / perimeter_m**2
    if kind == "aspect_ratio":
        if minimum_diameter_m is None or maximum_diameter_m is None:
            raise ValueError("both diameters required")
        _positive(minimum_diameter_m, "minimum diameter")
        _positive(maximum_diameter_m, "maximum diameter")
        if minimum_diameter_m > maximum_diameter_m:
            raise ValueError("diameters ordered incorrectly")
        return minimum_diameter_m / maximum_diameter_m
    raise ValueError("unknown shape factor")


@dataclass(frozen=True)
class CoolingTowerGeometry:
    inlet_height_m: float
    outlet_height_m: float
    outlet_diameter_m: float
    inlet_diameter_m: float | None = None
    base_diameter_m: float | None = None
    throat_diameter_m: float | None = None
    throat_height_m: float | None = None
    support_height_m: float | None = None
    support_diameter_m: float | None = None
    support_count: int | None = None
    inlet_rounding_m: float | None = None
    _lower: float = field(init=False, repr=False)
    _upper: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for value in (self.inlet_height_m, self.outlet_height_m, self.outlet_diameter_m):
            _positive(value, "tower dimension")
        throat_h = (
            self.throat_height_m
            if self.throat_height_m is not None
            else 2 * self.outlet_height_m / 3
        )
        throat_d = self.throat_diameter_m
        if throat_d is None:
            if self.inlet_diameter_m is None:
                raise ValueError("throat or inlet diameter required")
            throat_d = 0.63 * self.inlet_diameter_m
        if not self.inlet_height_m < throat_h < self.outlet_height_m:
            raise ValueError("throat height must lie between inlet and outlet")
        if not 0 < throat_d < self.outlet_diameter_m:
            raise ValueError("throat must be narrower than outlet")
        object.__setattr__(self, "throat_height_m", throat_h)
        object.__setattr__(self, "throat_diameter_m", throat_d)
        if self.base_diameter_m is not None:
            if self.base_diameter_m <= throat_d:
                raise ValueError("base must exceed throat")
            lower = throat_d * throat_h / sqrt(self.base_diameter_m**2 - throat_d**2)
            inlet = throat_d * sqrt(1 + ((throat_h - self.inlet_height_m) / lower) ** 2)
            if self.inlet_diameter_m is not None and abs(inlet / self.inlet_diameter_m - 1) > 1e-9:
                raise ValueError("conflicting inlet and base diameters")
            object.__setattr__(self, "inlet_diameter_m", inlet)
        else:
            if self.inlet_diameter_m is None or self.inlet_diameter_m <= throat_d:
                raise ValueError("inlet diameter exceeding throat required")
            lower = (
                throat_d
                * (throat_h - self.inlet_height_m)
                / sqrt(self.inlet_diameter_m**2 - throat_d**2)
            )
            object.__setattr__(
                self, "base_diameter_m", throat_d * sqrt(1 + (throat_h / lower) ** 2)
            )
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(
            self,
            "_upper",
            throat_d
            * (self.outlet_height_m - throat_h)
            / sqrt(self.outlet_diameter_m**2 - throat_d**2),
        )

    def diameter(self, height_m: float) -> float:
        if not isfinite(height_m) or not 0 <= height_m <= self.outlet_height_m:
            raise ValueError("height outside tower")
        assert self.throat_height_m is not None and self.throat_diameter_m is not None
        offset = height_m - self.throat_height_m
        scale = self._lower if offset <= 0 else self._upper
        return self.throat_diameter_m * sqrt(1 + (offset / scale) ** 2)

    def sample_profile(self, points: int = 101) -> tuple[tuple[float, float], ...]:
        if points < 2:
            raise ValueError("at least two profile points required")
        return tuple(
            (h, self.diameter(h))
            for h in (self.outlet_height_m * i / (points - 1) for i in range(points))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CoolingTowerGeometry":
        return cls(**value)
