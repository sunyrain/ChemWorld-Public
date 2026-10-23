"""Serializable vessel capacity, wetting, level calibration and sizing.

Geometric state is derived from liquid volume; this module owns no inventory.
All inverses use the common engineering residual solver. Head surface integrals
share one geometry, including concave intrusions and deep spherical envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from math import acos, ceil, exp, isfinite, log, pi, sqrt
from typing import Any, cast

import numpy as np
from numpy.polynomial.chebyshev import chebinterpolate, chebval
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.geometric_primitives import circular_segment_area
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)
from chemworld.physchem.tank_heads import TankHead, _clip

_FLAT_HEAD = TankHead()


def cylindrical_wetted_area(
    diameter_m: float, length_m: float, height_m: float, *, horizontal: bool = True
) -> float:
    _positive(diameter_m, "diameter")
    if not isfinite(length_m) or length_m < 0 or not isfinite(height_m):
        raise ValueError("nonnegative length and finite height required")
    if horizontal:
        return length_m * diameter_m * acos(_clip(1 - 2 * height_m / diameter_m, -1, 1))
    return pi * diameter_m * _clip(height_m, 0, length_m)


@dataclass(frozen=True)
class TankGeometry:
    diameter_m: float
    length_m: float
    horizontal: bool = True
    head_a: TankHead = _FLAT_HEAD
    head_b: TankHead = _FLAT_HEAD

    def __post_init__(self) -> None:
        _positive(self.diameter_m, "diameter")
        if not isfinite(self.length_m) or self.length_m < 0:
            raise ValueError("nonnegative shell length required")
        if not isinstance(self.horizontal, bool):
            raise ValueError("orientation must be boolean")
        intrusions = sum(
            head.depth * self.diameter_m for head in (self.head_a, self.head_b) if head.concave
        )
        if intrusions > self.length_m:
            raise ValueError("intruding heads must fit without overlap inside the straight shell")
        if self.total_volume_m3 <= 0:
            raise ValueError("tank must enclose positive volume")

    @property
    def volume_parts_m3(self) -> tuple[float, float, float, float]:
        parts = tuple(
            head.total_volume(self.diameter_m) * (-1 if head.concave else 1)
            for head in (self.head_a, self.head_b)
        )
        body = pi * self.diameter_m**2 * self.length_m / 4
        return parts[0] + parts[1] + body, parts[0], parts[1], body

    @property
    def total_volume_m3(self) -> float:
        return self.volume_parts_m3[0]

    @property
    def area_parts_m2(self) -> tuple[float, float, float, float]:
        a, b = (head.total_area(self.diameter_m) for head in (self.head_a, self.head_b))
        body = pi * self.diameter_m * self.length_m
        return a + b + body, a, b, body

    @property
    def total_area_m2(self) -> float:
        return self.area_parts_m2[0]

    @property
    def maximum_height_m(self) -> float:
        if self.horizontal:
            return 2 * self.diameter_m * max(self.head_a.radial_extent, self.head_b.radial_extent)
        return self.length_m + self.diameter_m * sum(
            head.depth for head in (self.head_a, self.head_b) if not head.concave
        )

    def _shell_level(self, height_m: float) -> float:
        if not isfinite(height_m):
            raise ValueError("finite fill height required")
        if self.horizontal:
            return height_m - (self.maximum_height_m - self.diameter_m) / 2
        return height_m - (0 if self.head_a.concave else self.head_a.depth * self.diameter_m)

    def _vertical_head_measure(
        self, head: TankHead, shell_level: float, bottom: bool, measure: str
    ) -> float:
        d = self.diameter_m
        depth = d * head.depth
        if head.kind == "flat":
            return (
                pi * d**2 / 4
                if measure == "area" and shell_level >= (0 if bottom else self.length_m)
                else 0.0
            )
        total = head.total_volume(d) if measure == "volume" else head.total_area(d)
        if bottom:
            if head.concave:
                return total - head.vertical_measure(d, depth - shell_level, measure)
            return head.vertical_measure(d, shell_level + depth, measure)
        if head.concave:
            return head.vertical_measure(d, shell_level - self.length_m + depth, measure)
        return total - head.vertical_measure(d, depth + self.length_m - shell_level, measure)

    def volume_at_height(self, height_m: float) -> float:
        level = self._shell_level(height_m)
        if height_m <= 0:
            return 0.0
        if height_m >= self.maximum_height_m:
            return self.total_volume_m3
        if self.horizontal:
            body = circular_segment_area(self.diameter_m, level) * self.length_m
            heads = (
                head.horizontal_measure(self.diameter_m, level)
                for head in (self.head_a, self.head_b)
            )
        else:
            body = pi * self.diameter_m**2 / 4 * _clip(level, 0, self.length_m)
            heads = (
                self._vertical_head_measure(head, level, bottom, "volume")
                for head, bottom in ((self.head_a, True), (self.head_b, False))
            )
        return body + sum(
            value * (-1 if head.concave else 1)
            for head, value in zip((self.head_a, self.head_b), heads, strict=True)
        )

    def wetted_area_at_height(self, height_m: float) -> float:
        """Solid-liquid contact; at h=0 a vertical flat bottom includes its disk.

        This is the closed-set / right-limit convention of vessel calibration;
        the liquid free surface is reported separately by ``cross_section``.
        """
        level = self._shell_level(height_m)
        if height_m < 0:
            return 0.0
        if height_m >= self.maximum_height_m:
            return self.total_area_m2
        body = cylindrical_wetted_area(
            self.diameter_m, self.length_m, level, horizontal=self.horizontal
        )
        if self.horizontal:
            return body + sum(
                head.horizontal_measure(self.diameter_m, level, "area")
                for head in (self.head_a, self.head_b)
            )
        return body + sum(
            self._vertical_head_measure(head, level, bottom, "area")
            for head, bottom in ((self.head_a, True), (self.head_b, False))
        )

    def cross_section_at_height(self, height_m: float) -> float:
        """dV/dh from geometric slices, without finite differencing volumes.

        At the exact empty/full endpoints this follows the tank convention of
        zero exposed free surface. One-sided limits can differ at flat ends.
        """
        level = self._shell_level(height_m)
        if height_m <= 0 or height_m >= self.maximum_height_m:
            return 0.0
        if self.horizontal:
            body = 2 * self.length_m * sqrt(max(0, level * (self.diameter_m - level)))
            heads = (
                head.horizontal_measure(self.diameter_m, level, "cross")
                for head in (self.head_a, self.head_b)
            )
        else:
            body = pi * self.diameter_m**2 / 4 if 0 <= level <= self.length_m else 0.0
            coordinates = (
                self.head_a.depth * self.diameter_m - level
                if self.head_a.concave
                else level + self.head_a.depth * self.diameter_m,
                level - self.length_m + self.head_b.depth * self.diameter_m
                if self.head_b.concave
                else self.head_b.depth * self.diameter_m + self.length_m - level,
            )
            heads = (
                head.vertical_measure(self.diameter_m, coordinate, "cross")
                if head.kind != "flat" and 0 < coordinate < head.depth * self.diameter_m
                else 0.0
                for head, coordinate in zip((self.head_a, self.head_b), coordinates, strict=True)
            )
        return max(
            0,
            body
            + sum(
                value * (-1 if head.concave else 1)
                for head, value in zip((self.head_a, self.head_b), heads, strict=True)
            ),
        )

    def height_for_volume(self, volume_m3: float) -> float:
        total = self.total_volume_m3
        if not isfinite(volume_m3) or not 0 <= volume_m3 <= total:
            raise ValueError("liquid volume outside vessel capacity")
        if volume_m3 == 0:
            return 0.0
        if volume_m3 == total:
            return self.maximum_height_m
        fraction = volume_m3 / total
        # Normalized residual preserves low-fill relative accuracy. The common
        # bounded solver chooses Newton or bisection steps as appropriate.
        scale = min(fraction, 1 - fraction)

        def residual(x: float) -> float:
            return (self.volume_at_height(x * self.maximum_height_m) / total - fraction) / scale

        return self.maximum_height_m * _engineering_root(
            residual, fraction, 0, 1, name="tank_fill_fraction", unit="dimensionless", scale=1
        )

    def liquid_state(
        self, volume_m3: float, *, density_kg_m3: float, gravity_m_s2: float = 9.80665
    ) -> dict[str, float]:
        """Consume the current liquid volume/property without creating inventory."""
        _positive(density_kg_m3, "density")
        _positive(gravity_m_s2, "gravity")
        height = self.height_for_volume(volume_m3)
        return {
            "liquid_volume_m3": volume_m3,
            "headspace_volume_m3": self.total_volume_m3 - volume_m3,
            "height_m": height,
            "wetted_area_m2": self.wetted_area_at_height(height),
            "free_surface_area_m2": self.cross_section_at_height(height),
            "bottom_gauge_pressure_Pa": density_kg_m3 * gravity_m_s2 * height,
        }

    @classmethod
    def from_dimensions(
        cls,
        *,
        diameter_m: float | None = None,
        length_m: float | None = None,
        length_over_diameter: float | None = None,
        volume_m3: float | None = None,
        horizontal: bool = True,
        head_a: TankHead = _FLAT_HEAD,
        head_b: TankHead = _FLAT_HEAD,
    ) -> TankGeometry:
        """Resolve D/L/L:D, or total volume plus exactly one of those dimensions."""
        dimensions = (diameter_m, length_m, length_over_diameter)
        for value in (diameter_m, length_over_diameter):
            if value is not None:
                _positive(value, "sizing dimension")
        if length_m is not None and (not isfinite(length_m) or length_m < 0):
            raise ValueError("nonnegative shell length required")
        count = sum(value is not None for value in dimensions)
        if volume_m3 is None:
            if count != 2:
                raise ValueError("exactly two of diameter, length and length/diameter required")
            if diameter_m is None:
                assert length_m is not None and length_over_diameter is not None
                diameter_m = length_m / length_over_diameter
            if length_m is None:
                assert length_over_diameter is not None
                length_m = diameter_m * length_over_diameter
        else:
            _positive(volume_m3, "target volume")
            if count != 1:
                raise ValueError("volume sizing needs exactly one of D, L or L/D")
            head_factor = sum(
                head.total_volume(1) * (-1 if head.concave else 1) for head in (head_a, head_b)
            )
            if diameter_m is not None:
                length_m = (volume_m3 - head_factor * diameter_m**3) / (pi * diameter_m**2 / 4)
            elif length_over_diameter is not None:
                coefficient = pi * length_over_diameter / 4 + head_factor
                if coefficient <= 0:
                    raise ValueError("specified aspect ratio leaves no capacity")
                diameter_m = (volume_m3 / coefficient) ** (1 / 3)
                length_m = diameter_m * length_over_diameter
            else:
                assert length_m is not None
                if length_m == 0:
                    if head_factor <= 0:
                        raise ValueError("zero shell length needs positive head capacity")
                    return cls((volume_m3 / head_factor) ** (1 / 3), 0, horizontal, head_a, head_b)
                estimate = sqrt(4 * volume_m3 / (pi * length_m))

                def residual(x: float) -> float:
                    diameter = exp(x)
                    return (
                        pi * diameter**2 * cast(float, length_m) / 4 + head_factor * diameter**3
                    ) / volume_m3 - 1

                diameter_m = exp(
                    _engineering_root(
                        residual,
                        log(estimate),
                        log(estimate) - 20,
                        log(estimate) + 20,
                        name="tank_log_diameter",
                        unit="dimensionless",
                        scale=1,
                    )
                )
        return cls(diameter_m, cast(float, length_m), horizontal, head_a, head_b)

    def specification(self, name: str, height_m: float | None = None) -> float:
        if name == "volume":
            return self.total_volume_m3
        if name == "area":
            return self.total_area_m2
        if height_m is None:
            raise ValueError("partial specifications require fill height")
        functions = {
            "partial_volume": self.volume_at_height,
            "wetted_area": self.wetted_area_at_height,
            "cross_section": self.cross_section_at_height,
        }
        if name not in functions:
            raise ValueError("unknown vessel specification")
        return functions[name](height_m)

    @classmethod
    def from_two_specs(
        cls,
        specs: dict[str, float],
        *,
        height_m: float | None = None,
        horizontal: bool = True,
        head_a: TankHead = _FLAT_HEAD,
        head_b: TankHead = _FLAT_HEAD,
        initial_diameter_m: float = 1,
        initial_length_m: float = 3,
    ) -> TankGeometry:
        """Two independent specifications; the initial geometry selects a root.

        Some area/volume combinations admit multiple valid aspect ratios. No
        global uniqueness is claimed; both requested residuals must converge.
        """
        if len(specs) != 2:
            raise ValueError("two distinct specifications required")
        for value in (*specs.values(), initial_diameter_m, initial_length_m):
            _positive(value, "specification or initial dimension")
        variables = tuple(
            ResidualVariable(name, "dimensionless", 1, "algebraic", -14, 14)
            for name in ("log_diameter", "log_length")
        )

        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            try:
                tank = cls(exp(float(x[0])), exp(float(x[1])), horizontal, head_a, head_b)
                return np.array(
                    [
                        tank.specification(name, height_m) / value - 1
                        for name, value in specs.items()
                    ]
                )
            except ValueError as error:
                raise ResidualDomainError(str(error)) from error

        solution = solve_root(
            residual,
            [log(initial_diameter_m), log(initial_length_m)],
            jacobian=lambda x: finite_jacobian(
                residual, x, [1, 1], lower=np.array([-14, -14]), upper=np.array([14, 14])
            ),
            variables=variables,
            residual_scales=[1, 1],
            policy=RootPolicy(tolerance=2e-10, maximum_iterations=120),
        )
        if not solution.success:
            raise ValueError(f"tank specification solve failed: {solution.failure_reason}")
        values = cast(NDArray[np.float64], solution.value)
        return cls(exp(float(values[0])), exp(float(values[1])), horizontal, head_a, head_b)

    def with_wall_thickness(
        self,
        thickness_m: float,
        *,
        head_a_thickness_m: float | None = None,
        head_b_thickness_m: float | None = None,
    ) -> TankGeometry:
        """Engineering parameter expansion, not a certified normal surface offset.

        Matches the common tank-calculator convention: D += 2t, L += tA+tB,
        head signed depths += tA/tB; torispherical f/k stay fixed. The shell
        volume difference is an approximation for shaped heads.
        """
        a = thickness_m if head_a_thickness_m is None else head_a_thickness_m
        b = thickness_m if head_b_thickness_m is None else head_b_thickness_m
        if not all(isfinite(v) for v in (thickness_m, a, b)):
            raise ValueError("finite thickness required")
        diameter = self.diameter_m + 2 * thickness_m
        _positive(diameter, "expanded diameter")
        heads = []
        for head, thickness in ((self.head_a, a), (self.head_b, b)):
            if head.kind in ("flat", "torispherical"):
                heads.append(head)
            else:
                depth = (-1 if head.concave else 1) * head.depth * self.diameter_m + thickness
                heads.append(
                    TankHead()
                    if depth == 0
                    else replace(head, depth_ratio=abs(depth) / diameter, concave=depth < 0)
                )
        return TankGeometry(diameter, self.length_m + a + b, self.horizontal, *heads)

    def table(self, points: int = 101, *, step_m: float | None = None) -> TankCalibrationTable:
        if step_m is not None:
            _positive(step_m, "level spacing")
            points = ceil(self.maximum_height_m / step_m) + 1
        if isinstance(points, bool) or int(points) != points or points < 2:
            raise ValueError("at least two integer calibration points required")
        heights = tuple(float(v) for v in np.linspace(0, self.maximum_height_m, points))
        return TankCalibrationTable(heights, tuple(self.volume_at_height(h) for h in heights))

    def chebyshev(
        self,
        forward_degree: int | None = 50,
        inverse_degree: int | None = 200,
        *,
        tolerance: float = 1e-6,
        maximum_degree: int = 2048,
    ) -> TankChebyshev:
        """Fit fixed degrees or adapt None degrees on 65 offset validation nodes.

        Tolerance is normalized by total height/volume and describes that grid,
        not a rigorous uniform error bound. Exhausting the degree budget raises.
        """
        _positive(tolerance, "approximation tolerance")
        for degree in (forward_degree, inverse_degree):
            if degree is not None and (
                isinstance(degree, bool) or int(degree) != degree or degree < 2
            ):
                raise ValueError("polynomial degrees must be integers >= 2")
        if (
            isinstance(maximum_degree, bool)
            or int(maximum_degree) != maximum_degree
            or maximum_degree < 16
        ):
            raise ValueError("adaptive maximum degree must be an integer >= 16")

        def fit(forward: bool, degree: int | None) -> tuple[float, ...]:
            function = self.volume_at_height if forward else self.height_for_volume
            domain = self.maximum_height_m if forward else self.total_volume_m3
            scale = self.total_volume_m3 if forward else self.maximum_height_m

            def values(x: NDArray[np.float64]) -> NDArray[np.float64]:
                return np.array([function(float(t + 1) * domain / 2) for t in x])

            current = 16 if degree is None else degree
            validation_x = np.linspace(-1, 1, 131, dtype=np.float64)[1::2]
            expected = values(validation_x) if degree is None else None
            while True:
                coefficients = chebinterpolate(values, current)  # type: ignore[type-var]
                if (
                    expected is None
                    or float(np.max(np.abs(chebval(validation_x, coefficients) - expected)))
                    <= tolerance * scale
                ):
                    return tuple(float(v) for v in coefficients)
                if current >= maximum_degree:
                    raise ValueError("adaptive Chebyshev calibration exceeded its degree budget")
                current = min(maximum_degree, 2 * current)

        forward = fit(True, forward_degree)
        backward = fit(False, inverse_degree)
        return TankChebyshev(
            self.maximum_height_m, self.total_volume_m3, tuple(forward), tuple(backward)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diameter_m": self.diameter_m,
            "length_m": self.length_m,
            "horizontal": self.horizontal,
            "head_a": self.head_a.to_dict(),
            "head_b": self.head_b.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TankGeometry:
        return cls(
            **{
                **values,
                "head_a": TankHead.from_dict(values["head_a"]),
                "head_b": TankHead.from_dict(values["head_b"]),
            }
        )


@dataclass(frozen=True)
class TankCalibrationTable:
    heights_m: tuple[float, ...]
    volumes_m3: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.heights_m) != len(self.volumes_m3) or len(self.heights_m) < 2:
            raise ValueError("paired calibration columns required")
        for column in (self.heights_m, self.volumes_m3):
            if (
                not all(isfinite(v) for v in column)
                or column[0] != 0
                or any(a >= b for a, b in pairwise(column))
            ):
                raise ValueError(
                    "finite, strictly increasing calibration columns starting at zero required"
                )

    def height_for_volume(self, volume_m3: float) -> float:
        if not isfinite(volume_m3) or not 0 <= volume_m3 <= self.volumes_m3[-1]:
            raise ValueError("volume outside calibration")
        return float(
            PchipInterpolator(self.volumes_m3, self.heights_m, extrapolate=False)(volume_m3)
        )

    def volume_at_height(self, height_m: float) -> float:
        if not isfinite(height_m) or not 0 <= height_m <= self.heights_m[-1]:
            raise ValueError("height outside calibration")
        return float(
            PchipInterpolator(self.heights_m, self.volumes_m3, extrapolate=False)(height_m)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"heights_m": self.heights_m, "volumes_m3": self.volumes_m3}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TankCalibrationTable:
        return cls(tuple(values["heights_m"]), tuple(values["volumes_m3"]))


@dataclass(frozen=True)
class TankChebyshev:
    maximum_height_m: float
    total_volume_m3: float
    forward_coefficients: tuple[float, ...]
    inverse_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        _positive(self.maximum_height_m, "calibration height")
        _positive(self.total_volume_m3, "calibration volume")
        for column in (self.forward_coefficients, self.inverse_coefficients):
            if len(column) < 3 or not all(isfinite(v) for v in column):
                raise ValueError("finite Chebyshev coefficients required")

    def volume_at_height(self, height_m: float) -> float:
        if not isfinite(height_m) or not 0 <= height_m <= self.maximum_height_m:
            raise ValueError("height outside approximation")
        if height_m in (0, self.maximum_height_m):
            return 0.0 if height_m == 0 else self.total_volume_m3
        return _clip(
            float(chebval(2 * height_m / self.maximum_height_m - 1, self.forward_coefficients)),
            0,
            self.total_volume_m3,
        )

    def height_for_volume(self, volume_m3: float) -> float:
        if not isfinite(volume_m3) or not 0 <= volume_m3 <= self.total_volume_m3:
            raise ValueError("volume outside approximation")
        if volume_m3 in (0, self.total_volume_m3):
            return 0.0 if volume_m3 == 0 else self.maximum_height_m
        return _clip(
            float(chebval(2 * volume_m3 / self.total_volume_m3 - 1, self.inverse_coefficients)),
            0,
            self.maximum_height_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_height_m": self.maximum_height_m,
            "total_volume_m3": self.total_volume_m3,
            "forward_coefficients": self.forward_coefficients,
            "inverse_coefficients": self.inverse_coefficients,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TankChebyshev:
        return cls(
            float(values["maximum_height_m"]),
            float(values["total_volume_m3"]),
            tuple(values["forward_coefficients"]),
            tuple(values["inverse_coefficients"]),
        )
