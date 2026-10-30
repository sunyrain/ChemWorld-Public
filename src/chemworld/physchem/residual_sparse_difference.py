"""Sparse finite differences, with no physical model or state ownership.

The caller declares a conservative *structural* column pattern, including any
domain coupling relevant to the affected rows. A pattern inferred only from
zeros at one point is insufficient. Coloring cannot discover missing coupling
or prove the separability of an arbitrary hidden global domain constraint.

Rows in one color are disjoint. Sample offsets and central/one-sided formulas
match chemworld.physchem.residual_system.finite_jacobian. If a combined sample
raises ResidualDomainError, that entire color uses the existing scalar-column
implementation. No inventory is clipped, seeded, or otherwise repaired.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from chemworld.physchem.residual_system import (
    ResidualDomainError,
    RootFunction,
    VectorInput,
    finite_jacobian,
)

DERIVATIVE_MODE = "finite_difference"


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def validate_column_sparsity(
    sparsity: Sequence[tuple[int | np.integer, ...]], columns: int, rows: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Validate explicit row-index tuples; return canonical ascending row tuples."""
    if not _integer(columns) or columns < 0:
        raise ValueError("column count must be a nonnegative integer")
    if rows is not None and (not _integer(rows) or rows < 0):
        raise ValueError("row count must be a nonnegative integer")
    if not isinstance(sparsity, (tuple, list)) or len(sparsity) != columns:
        raise ValueError("sparsity needs exactly one explicit row tuple per input column")
    result = []
    for entries in sparsity:
        if not isinstance(entries, tuple):
            raise ValueError("each sparsity column must be a row-index tuple")
        if any(not _integer(row) for row in entries):
            raise ValueError("sparsity rows must be integers, never booleans")
        converted = tuple(int(row) for row in entries)
        if len(set(converted)) != len(converted):
            raise ValueError("sparsity rows must be unique within each column")
        if any(row < 0 or (rows is not None and row >= rows) for row in converted):
            raise ValueError("sparsity row outside declared residual dimension")
        result.append(tuple(sorted(converted)))
    return tuple(result)


def _column_indices(
    indices: Sequence[int | np.integer] | np.ndarray | None, size: int,
) -> tuple[int, ...]:
    if indices is None:
        return tuple(range(size))
    if isinstance(indices, np.ndarray):
        if indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional integer sequence")
    elif not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("indices must be a one-dimensional integer sequence")
    if any(not _integer(index) for index in indices):
        raise ValueError("indices must contain integers, never booleans")
    converted = tuple(int(index) for index in indices)
    if len(set(converted)) != len(converted) or any(not 0 <= i < size for i in converted):
        raise ValueError("indices must be unique valid input columns")
    return converted


def _numeric_vector(
    value: ArrayLike, size: int | None, name: str, *, infinite: bool = False,
) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "fiu" or (size is not None and raw.size != size):
        raise ValueError(f"{name} must be a one-dimensional numeric vector of the correct size")
    result = raw.astype(float, copy=True)
    if np.isnan(result).any() or (not infinite and not np.isfinite(result).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _evaluate(function: RootFunction, point: VectorInput) -> NDArray[np.float64]:
    argument = np.asarray(point, dtype=float).copy()
    argument.setflags(write=False)
    result = np.asarray(function(argument), dtype=float)
    if result.ndim != 1:
        raise ValueError("function must return one residual vector")
    if not np.isfinite(result).all():
        raise ResidualDomainError("finite-difference residual is nonfinite")
    return result.copy()


def _greedy_colors(
    pattern: Sequence[Sequence[int]], selected: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    colors: list[list[int]] = []
    occupied: list[set[int]] = []
    for column in selected:
        rows = set(pattern[column])
        for group, used in zip(colors, occupied, strict=True):
            if rows.isdisjoint(used):
                group.append(column)
                used.update(rows)
                break
        else:
            colors.append([column])
            occupied.append(rows)
    return tuple(tuple(group) for group in colors)


def _sample_plan(
    point: NDArray[np.float64], scales: NDArray[np.float64], lower: NDArray[np.float64],
    upper: NDArray[np.float64], relative_step: float, column: int,
) -> list[tuple[int, float, float]]:
    """Use exactly the existing dense routine's step, room and sign choices."""
    value = point[column]
    step = relative_step * max(abs(value), scales[column])
    room = {-1: max(0.0, value - lower[column]), 1: max(0.0, upper[column] - value)}
    central = min(room.values()) >= step
    signs = (-1, 1) if central else tuple(sorted(room, key=room.__getitem__, reverse=True))
    samples = []
    for sign in signs:
        offset = min(step, room[sign])
        if offset <= 0:
            continue
        shifted = value + sign * offset
        delta = shifted - value
        if delta == 0 or shifted < lower[column] or shifted > upper[column]:
            continue
        samples.append((sign, shifted, delta))
        if not central:
            # On a domain rejection, the dense fallback will try the alternate
            # direction exactly as finite_jacobian does. Do not invent a step.
            break
    return samples


def sparse_finite_jacobian(
    function: RootFunction,
    point: VectorInput,
    scales: VectorInput,
    *,
    sparsity: Sequence[tuple[int | np.integer, ...]],
    lower: VectorInput | None = None,
    upper: VectorInput | None = None,
    relative_step: float = 6e-6,
    indices: Sequence[int | np.integer] | np.ndarray | None = None,
) -> csc_matrix:
    """Return m-by-len(indices) CSC numerical Jacobian in requested column order.

    ``sparsity[j]`` is a tuple of possible residual rows for original column j,
    even when indices requests a subset. Empty tuples represent declared zero
    columns. Domain rejection does not permit inserting artificial zero columns.
    A malformed function or a non-domain exception propagates to the caller.
    """
    if not callable(function):
        raise ValueError("function must be callable")
    point = _numeric_vector(point, None, "point")
    scales = _numeric_vector(scales, point.size, "scales")
    if np.any(scales <= 0):
        raise ValueError("scales must be strictly positive")
    if isinstance(relative_step, (bool, np.bool_)) or not isinstance(relative_step, (int, float)):
        raise ValueError("relative step must be a positive finite number")
    if not isfinite(relative_step) or relative_step <= 0:
        raise ValueError("relative step must be a positive finite number")
    selected = _column_indices(indices, point.size)
    pattern = validate_column_sparsity(sparsity, point.size)
    low = (
        np.full(point.size, -np.inf)
        if lower is None
        else _numeric_vector(lower, point.size, "lower bounds", infinite=True)
    )
    high = (
        np.full(point.size, np.inf)
        if upper is None
        else _numeric_vector(upper, point.size, "upper bounds", infinite=True)
    )
    if np.any(low > high):
        raise ValueError("lower bounds must not exceed upper bounds")
    if np.any(point < low) or np.any(point > high):
        raise ResidualDomainError("finite-difference point is outside declared bounds")
    baseline = _evaluate(function, point)
    pattern = validate_column_sparsity(pattern, point.size, baseline.size)
    answers = {}

    def sample(trial: NDArray[np.float64]) -> NDArray[np.float64]:
        value = _evaluate(function, trial)
        if value.shape != baseline.shape:
            raise ValueError("function changed its residual dimension")
        return value

    for group in _greedy_colors(pattern, selected):
        plans = {j: _sample_plan(point, scales, low, high, relative_step, j) for j in group}
        try:
            if any(not plan for plan in plans.values()):
                raise ResidualDomainError("no admissible finite-difference stencil")
            first_trial = point.copy()
            for column in group:
                first_trial[column] = plans[column][0][1]
            first = sample(first_trial)
            central_columns = [j for j in group if len(plans[j]) == 2]
            second = None
            if central_columns:
                second_trial = point.copy()
                for column in central_columns:
                    second_trial[column] = plans[column][1][1]
                second = sample(second_trial)
            for column in group:
                rows = np.asarray(pattern[column], dtype=int)
                negative_or_single = first[rows]
                if len(plans[column]) == 2:
                    _, _, negative_hb = plans[column][0]
                    _, _, hf = plans[column][1]
                    hb = -negative_hb
                    # A two-sample plan puts this column in central_columns above.
                    answers[column] = (hb / (hf + hb)) * ((
                        cast(NDArray[np.float64], second)[rows] - baseline[rows]) / hf) + (
                        hf / (hf + hb)
                    ) * ((baseline[rows] - negative_or_single) / hb)
                else:
                    answers[column] = (negative_or_single - baseline[rows]) / plans[column][0][2]
        except ResidualDomainError:
            # Keep the actual scalar-column rejection semantics. Only domain
            # errors trigger this path; arbitrary exceptions are not swallowed.
            dense = finite_jacobian(
                sample,
                point,
                scales,
                lower=low,
                upper=high,
                relative_step=relative_step,
                indices=group,
            )
            for position, column in enumerate(group):
                answers[column] = dense[np.asarray(pattern[column], dtype=int), position]

    data: list[float] = []
    row_indices: list[int] = []
    pointers = [0]
    for column in selected:
        data.extend(answers[column])
        row_indices.extend(pattern[column])
        pointers.append(len(data))
    result = csc_matrix(
        (
            np.asarray(data, dtype=float),
            np.asarray(row_indices, dtype=int),
            np.asarray(pointers, dtype=int),
        ),
        shape=(baseline.size, len(selected)),
    )
    # Exact arithmetic zeros stay exact zeros; no thresholding is applied.
    result.eliminate_zeros()
    return result


__all__ = ["DERIVATIVE_MODE", "sparse_finite_jacobian", "validate_column_sparsity"]
