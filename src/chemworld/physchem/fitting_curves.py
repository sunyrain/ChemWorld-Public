"""Caller-supplied fitting curves on the common scientific parameter tables.

Linear curves use x/value columns. Polynomial and spline rows use index/value
columns grouped as coefficients, knots_x and knots_y. Curve degree and policy
are explicit; no chart observations, spline arrays or fit coefficients bundled.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import BSpline, bisplev

if TYPE_CHECKING:
    from chemworld.data.scientific_inputs import ScientificTable


class FittingCurve:
    def __init__(
        self,
        table: ScientificTable,
        *,
        kind: str,
        degrees: tuple[int, ...] = (),
        extrapolate: bool = False,
    ) -> None:
        if table.kind != "parameters" or kind not in {
            "linear",
            "polynomial",
            "spline1d",
            "spline2d",
        }:
            raise ValueError("explicit fitting curve parameter representation required")
        self.table, self.kind, self.degrees, self.extrapolate = (
            table,
            kind,
            tuple(degrees),
            extrapolate,
        )
        groups: dict[str, list[tuple[float, float]]] = {}
        for row in table.rows:
            coordinate = row.values["x" if kind == "linear" else "index"].to("dimensionless")
            value = row.values["value"].to("dimensionless")
            if coordinate.quality != "valid" or value.quality != "valid":
                raise ValueError("valid fitting curve cells required")
            groups.setdefault(row.group_id, []).append(
                (float(coordinate.value), float(value.value))
            )
        self._arrays = {}
        for name, pairs in groups.items():
            pairs.sort()
            if len({p[0] for p in pairs}) != len(pairs):
                raise ValueError("duplicate fitting curve coordinate/index")
            self._arrays[name] = np.array([p[1] for p in pairs])
            if kind == "linear":
                self.x = tuple(p[0] for p in pairs)
        if kind == "linear":
            if len(groups) != 1 or len(self.x) < 2:
                raise ValueError("linear fitting curve requires one group and two points")
            self.y = next(iter(self._arrays.values()))
        elif "coefficients" not in groups or not len(groups["coefficients"]):
            raise ValueError("fitting curve coefficients required")
        elif kind.startswith("spline"):
            dimensions = 1 if kind == "spline1d" else 2
            if len(degrees) != dimensions or any(k not in (1, 2, 3, 4, 5) for k in degrees):
                raise ValueError("one declared spline degree per axis required")
            for axis, k in zip(("x", "y"), degrees, strict=False):
                knots = self._arrays.get(f"knots_{axis}")
                if knots is None or len(knots) < 2 * k + 2 or np.any(np.diff(knots) < 0):
                    raise ValueError("valid nondecreasing fitting spline knots required")

    def evaluate(self, x: float, y: float | None = None) -> float:
        if not isfinite(x) or (y is not None and not isfinite(y)):
            raise ValueError("finite fitting curve coordinate required")
        if self.kind == "linear":
            if self.extrapolate and not self.x[0] <= x <= self.x[-1]:
                index = 0 if x < self.x[0] else -2
                return float(
                    self.y[index]
                    + (x - self.x[index])
                    * (self.y[index + 1] - self.y[index])
                    / (self.x[index + 1] - self.x[index])
                )
            return float(np.interp(x, self.x, self.y))
        coefficients = self._arrays["coefficients"]
        if self.kind == "polynomial":
            # Coefficients are stored in descending powers, matching published fits.
            return float(np.polyval(coefficients, x))
        knots = self._arrays["knots_x"]
        if self.kind == "spline1d":
            k = self.degrees[0]
            if not self.extrapolate:
                x = float(np.clip(x, knots[k], knots[-k - 1]))
            return float(BSpline(knots, coefficients, k, extrapolate=self.extrapolate)(x))
        if y is None:
            raise ValueError("two coordinates required for a fitting surface")
        return float(bisplev(x, y, (knots, self._arrays["knots_y"], coefficients, *self.degrees)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table.to_dict(),
            "kind": self.kind,
            "degrees": self.degrees,
            "extrapolate": self.extrapolate,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FittingCurve:
        from chemworld.data.scientific_inputs import ScientificTable

        return cls(
            ScientificTable.from_dict(value["table"]),
            kind=value["kind"],
            degrees=tuple(value["degrees"]),
            extrapolate=value["extrapolate"],
        )


class FittingCurves:
    def __init__(self, curves: dict[str, FittingCurve]) -> None:
        self.curves = dict(curves)

    def evaluate(self, name: str, x: float, y: float | None = None) -> float:
        if name not in self.curves:
            raise ValueError(f"caller fitting curve required: {name}")
        return self.curves[name].evaluate(x, y)

    def to_dict(self) -> dict[str, Any]:
        return {name: curve.to_dict() for name, curve in self.curves.items()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FittingCurves:
        return cls({name: FittingCurve.from_dict(curve) for name, curve in value.items()})
