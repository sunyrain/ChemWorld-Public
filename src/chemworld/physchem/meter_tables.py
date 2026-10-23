"""Source-aware meter calibration/recovery surfaces from common input tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

if TYPE_CHECKING:
    from chemworld.data.scientific_inputs import ScientificTable


class MeterTableSurface:
    """Rectangular source grid, polynomial in x and optionally log(y).

    Single-x grids are one-dimensional in y. Default strict bounds can be
    explicitly replaced by source-style clamping or linear1D extrapolation.
    No Hollingshead or venturi recovery observations are bundled.
    """

    def __init__(
        self,
        table: ScientificTable,
        *,
        x_column: str = "beta",
        y_column: str = "reynolds",
        value_column: str = "coefficient",
        y_unit: str = "dimensionless",
        log_y: bool = True,
        degree_x: int = 3,
        degree_y: int = 3,
        outside: str = "raise",
    ) -> None:
        if (
            outside not in {"raise", "clamp", "linear"}
            or degree_x not in {1, 2, 3}
            or degree_y not in {1, 2, 3}
        ):
            raise ValueError("declared spline degree and outside policy required")
        self.table = table
        self.options = {
            "x_column": x_column,
            "y_column": y_column,
            "value_column": value_column,
            "y_unit": y_unit,
            "log_y": log_y,
            "degree_x": degree_x,
            "degree_y": degree_y,
            "outside": outside,
        }
        triples = []
        for row in table.rows:
            cells = (
                row.values[x_column].to("dimensionless"),
                row.values[y_column].to(y_unit),
                row.values[value_column].to("dimensionless"),
            )
            if any(cell.quality != "valid" for cell in cells):
                raise ValueError("meter surface requires valid source cells")
            x, y, z = (cell.value for cell in cells)
            if not 0 < x < 1 or y <= 0 or z <= 0:
                raise ValueError("physical beta and positive meter coordinates/values required")
            triples.append((x, y, z))
        self.x = np.array(sorted({row[0] for row in triples}))
        self.y = np.array(sorted({row[1] for row in triples}))
        if len(self.y) < 2 or not len(self.x) or len(triples) != len(self.x) * len(self.y):
            raise ValueError("complete rectangular meter source grid required")
        mapping = {(x, y): z for x, y, z in triples}
        if len(mapping) != len(triples):
            raise ValueError("duplicate calibration coordinates")
        values = np.array([[mapping[x, y] for y in self.y] for x in self.x])
        ys = np.log(self.y) if log_y else self.y
        self.one_dimensional = len(self.x) == 1
        if self.one_dimensional:
            self.interpolator = interp1d(
                ys, values[0], kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            self.spline = None
        else:
            if outside == "linear":
                raise ValueError(
                    "linear extrapolation is defined only for one-dimensional source curves"
                )
            self.spline = RectBivariateSpline(
                self.x,
                ys,
                values,
                kx=min(degree_x, len(self.x) - 1),
                ky=min(degree_y, len(self.y) - 1),
                s=0,
            )
            self.interpolator = None

    @property
    def y_bounds(self) -> tuple[float, float]:
        return float(self.y[0]), float(self.y[-1])

    def evaluate(self, beta: float, coordinate: float) -> float:
        if (
            not np.isfinite(beta)
            or not 0 < beta < 1
            or not np.isfinite(coordinate)
            or coordinate <= 0
        ):
            raise ValueError("physical meter table coordinates required")
        policy = self.options["outside"]
        if policy == "raise" and (
            not self.y[0] <= coordinate <= self.y[-1]
            or (not self.one_dimensional and not self.x[0] <= beta <= self.x[-1])
        ):
            raise ValueError("meter source table coordinate outside supplied domain")
        if policy == "clamp":
            beta = float(np.clip(beta, self.x[0], self.x[-1]))
            coordinate = float(np.clip(coordinate, self.y[0], self.y[-1]))
        y = np.log(coordinate) if self.options["log_y"] else coordinate
        if self.one_dimensional:
            assert self.interpolator is not None
            value = float(self.interpolator(y))
        else:
            assert self.spline is not None
            value = float(self.spline(beta, y)[0, 0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("interpolated meter coefficient is not positive")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict(), **self.options}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MeterTableSurface:
        from chemworld.data.scientific_inputs import ScientificTable

        return cls(
            ScientificTable.from_dict(value["table"]),
            **{k: v for k, v in value.items() if k != "table"},
        )
