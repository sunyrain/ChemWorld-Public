"""Sourced scalar parameter tables, not fits or an alternative input format.

Rectilinear multilinear interpolation has explicit finite domains. Original
ScientificTable units, uncertainty declarations and source identities survive
serialization. Stored input uncertainty is not an output prediction interval.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from math import isfinite, prod
from numbers import Real
from types import MappingProxyType

import numpy as np

from chemworld.data.scientific_inputs import (
    ScientificTable,
    _keys,
    _sequence,
    _text,
    strict_json_loads,
)
from chemworld.foundation.units import convert_value, unit_spec
from chemworld.physchem.caloric_math import FirstDerivative

TABLE_FUNCTION_VERSION = "rectilinear-scalar-parameter-function-1"
MAX_GRID_NODES = 262144


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("table function inputs must be finite real numbers")
    return float(value)


@dataclass(frozen=True)
class TabulatedScalar:
    """A constant or up to three varying axes on an explicitly selected table.

    Each axis binds (input_name, column_id, evaluation_unit). Numeric value and
    gradient arguments use those evaluation units, not guessed source units.
    Omitted axes mean declared independence; a one-point varying axis is rejected.
    """

    table: ScientificTable
    axes: tuple[tuple[str, str, str], ...]
    output_column: str
    output_unit: str
    row_ids: tuple[str, ...] = ()
    _points: tuple[tuple[float, ...], ...] = field(init=False, repr=False, compare=False)
    _values: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.table, ScientificTable) or self.table.kind != "parameters":
            raise ValueError(
                "a parameter function needs an explicit parameter table, not observations")
        axes = tuple(tuple(_sequence(axis, "axis binding"))
                     for axis in _sequence(self.axes, "axes"))
        if len(axes) > 3 or any(len(axis) != 3 for axis in axes):
            raise ValueError("a scalar table supports zero to three named varying axes")
        for axis in axes:
            for token in axis:
                _text(token, "axis binding")
            unit_spec(axis[2])
        if (len({a[0] for a in axes}) != len(axes)
                or len({a[1] for a in axes}) != len(axes)):
            raise ValueError("axis input names and column identities must be unique")
        _text(self.output_column, "output_column")
        unit_spec(self.output_unit)
        columns = {column.quantity_id: column for column in self.table.columns}
        if self.output_column not in columns or self.output_column in {a[1] for a in axes}:
            raise ValueError("the output must be a separate declared column")
        if any(column not in columns for _, column, _ in axes):
            raise ValueError("an axis references an undeclared table column")
        rows = {row.row_id: row for row in self.table.rows}
        selected = tuple(_sequence(self.row_ids, "row_ids")) or tuple(rows)
        for identity in selected:
            _text(identity, "row identity")
        if len(set(selected)) != len(selected) or any(key not in rows for key in selected):
            raise ValueError("selected parameter rows must exist exactly once")
        coordinate_rows, values = [], []
        for identity in selected:
            row = rows[identity]
            coordinate_rows.append(tuple(
                convert_value(row.values[column].value, row.values[column].unit, unit)
                for _, column, unit in axes))
            cell = row.values[self.output_column]
            values.append(convert_value(cell.value, cell.unit, self.output_unit))
        points = tuple(tuple(sorted({r[i] for r in coordinate_rows})) for i in range(len(axes)))
        if any(not isfinite(value) for axis in points for value in axis):
            raise ValueError("converted parameter axes must remain finite")
        if any(len(axis) < 2 for axis in points):
            raise ValueError("varying axes need at least two points; independence must be explicit")
        size = prod(map(len, points))
        if size > MAX_GRID_NODES:
            raise ValueError("parameter grid exceeds the declared node limit")
        if len(set(coordinate_rows)) != len(coordinate_rows):
            raise ValueError(
                "multiple parameter rows occupy one grid point; select sources explicitly")
        if size != len(selected):
            raise ValueError("parameter rows must cover the complete Cartesian grid")
        tensor = np.empty(tuple(map(len, points)))
        indices = [{value: i for i, value in enumerate(axis)} for axis in points]
        for row, value in zip(coordinate_rows, values, strict=True):
            tensor[tuple(index[x] for index, x in zip(indices, row, strict=True))] = value
        if not np.all(np.isfinite(tensor)):
            raise ValueError("converted parameter values must remain finite")
        tensor.setflags(write=False)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "row_ids", selected)
        object.__setattr__(self, "_points", points)
        object.__setattr__(self, "_values", tensor)

    @property
    def domain(self):
        return MappingProxyType({axis[0]: (points[0], points[-1])
                                 for axis, points in zip(self.axes, self._points, strict=True)})

    @property
    def value_range(self):
        """Vertex range, not an uncertainty interval or an outward-rounded error bound."""
        return float(self._values.min()), float(self._values.max())

    def _location(self, inputs, *, trial):
        if not isinstance(trial, bool):
            raise ValueError("numerical trial extension must be explicitly Boolean")
        if not isinstance(inputs, Mapping) or inputs.keys() != {axis[0] for axis in self.axes}:
            raise ValueError("supply exactly the declared function inputs in evaluation units")
        point = tuple(_number(inputs[axis[0]]) for axis in self.axes)
        cells = []
        for value, axis in zip(point, self._points, strict=True):
            if not trial and not axis[0] <= value <= axis[-1]:
                raise ValueError("parameter function point is outside its declared data domain")
            # This selects the boundary segment for an explicit numerical trial
            # extension. It never clips a state/input value to the data domain.
            index = int(np.searchsorted(axis, value, side="right"))-1
            cells.append(max(0, min(len(axis)-2, index)))
        return point, tuple(cells)

    def _interpolate(self, point, cells):
        if not self.axes:
            return float(self._values)
        fractions = tuple((x-axis[i])/(axis[i+1]-axis[i])
                          for x, axis, i in zip(point, self._points, cells, strict=True))
        result = 0.0
        for corners in product((0, 1), repeat=len(self.axes)):
            value = float(self._values[tuple(i+b for i, b in zip(cells, corners, strict=True))])
            for b, fraction in zip(corners, fractions, strict=True):
                value *= fraction if b else 1-fraction
            result += value
        return result

    def value(self, inputs, *, trial=False):
        point, cells = self._location(inputs, trial=trial)
        result = self._interpolate(point, cells)
        if not isfinite(result):
            raise ValueError("nonfinite parameter interpolation")
        return result

    def _partial(self, point, cells, index):
        dual = list(point)
        dual[index] = FirstDerivative(point[index], 1.0)
        result = self._interpolate(dual, cells)
        return result.slope

    def _exact_partial(self, point, cells, index):
        """Knot certificate for the actual finite binary data, not a smoothed fit."""
        point = tuple(Fraction(x) for x in point)
        grid = tuple(tuple(Fraction(x) for x in axis) for axis in self._points)
        total = Fraction(0)
        for corner in product((0, 1), repeat=len(self.axes)):
            value = Fraction(float(self._values[tuple(
                i+b for i, b in zip(cells, corner, strict=True))]))
            for j, (b, i, axis) in enumerate(zip(corner, cells, grid, strict=True)):
                width = axis[i+1]-axis[i]
                if j == index:
                    value *= Fraction(1 if b else -1)/width
                else:
                    fraction = (point[j]-axis[i])/width
                    value *= fraction if b else 1-fraction
            total += value
        return total

    def gradient(self, inputs, *, trial=False):
        """Analytic partials; no invented two-sided derivative at a grid kink.

        Numerical trial extension is explicit. Public boundary derivatives are
        rejected; interior knots require exactly matching one-sided derivatives
        of the finite binary data, certified by rational arithmetic.
        """
        point, cells = self._location(inputs, trial=trial)
        result = {}
        for j, (axis, grid, x) in enumerate(zip(self.axes, self._points, point, strict=True)):
            if not trial and x in (grid[0], grid[-1]):
                raise ValueError("a data-domain boundary has no declared two-sided derivative")
            derivative = self._partial(point, cells, j)
            if x in grid[1:-1]:
                k = grid.index(x)
                other_cells = list(cells)
                other_cells[j] = k-1
                left = self._exact_partial(point, other_cells, j)
                if left != self._exact_partial(point, cells, j):
                    raise ValueError("table knot has no unique derivative; do not linearize it")
            if not isfinite(derivative):
                raise ValueError("nonfinite parameter derivative")
            result[axis[0]] = derivative
        return MappingProxyType(result)

    def positive_coordinate_product_derivative(self, input_name):
        """Certify d(x*f)/dx > 0 on every cell side, not from sampled points.

        For a rectilinear multilinear f this derivative is affine in x and
        multilinear in other axes. Its extrema occur on cell vertices. Exact
        binary-data arithmetic avoids guessing a positive margin at zero.
        """
        names = [axis[0] for axis in self.axes]
        if input_name not in names:
            return bool(np.all(self._values > 0))  # Explicit independence of x.
        j = names.index(input_name)
        grid = tuple(Fraction(x) for x in self._points[j])
        others = [range(len(points)) for i, points in enumerate(self._points) if i != j]
        for location in product(*others):
            for k in range(len(grid)-1):
                indices = list(location)
                indices.insert(j, k)
                low = Fraction(float(self._values[tuple(indices)]))
                indices[j] += 1
                high = Fraction(float(self._values[tuple(indices)]))
                slope = (high-low)/(grid[k+1]-grid[k])
                if min(low+grid[k]*slope, high+grid[k+1]*slope) <= 0:
                    return False
        return True

    def to_dict(self):
        return {"schema_version": TABLE_FUNCTION_VERSION, "table": self.table.to_dict(),
                "axes": [list(axis) for axis in self.axes], "output_column": self.output_column,
                "output_unit": self.output_unit, "row_ids": list(self.row_ids)}

    @classmethod
    def from_dict(cls, payload):
        _keys(payload, (
            "schema_version", "table", "axes", "output_column", "output_unit", "row_ids"))
        if payload["schema_version"] != TABLE_FUNCTION_VERSION:
            raise ValueError("unsupported scalar parameter table version")
        return cls(ScientificTable.from_dict(payload["table"]), payload["axes"],
                   payload["output_column"], payload["output_unit"], payload["row_ids"])

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(strict_json_loads(text))
