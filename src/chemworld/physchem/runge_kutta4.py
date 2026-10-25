"""Classical fixed RK4 mathematical step; no clock, state or material ownership."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class RK4Step:
    values: tuple[float, ...]
    evaluations: int


def rk4_step(rhs, time_s, values, duration_s):
    for value in (time_s, duration_s):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError("RK4 time and duration must be finite nonnegative reals")
    y = np.asarray(values)
    if y.ndim != 1 or y.size == 0 or y.dtype.kind not in "iuf" or not np.isfinite(y).all():
        raise ValueError("RK4 requires a finite real vector")
    if not callable(rhs):
        raise ValueError("RK4 requires an explicit vector field")
    y = y.astype(float)
    if duration_s == 0:
        return RK4Step(tuple(map(float, y)), 0)
    if not isfinite(time_s + duration_s) or time_s + duration_s <= time_s:
        raise ValueError("RK4 endpoint cannot advance its finite clock")

    def f(t, at):
        result = np.asarray(rhs(t, at.copy()))
        if (
            result.shape != y.shape
            or result.dtype.kind not in "iuf"
            or not np.isfinite(result).all()
        ):
            raise ArithmeticError("RK4 vector field has invalid shape or nonfinite values")
        return result.astype(float)

    h = float(duration_s)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        k1 = f(time_s, y)
        k2 = f(time_s + h / 2, y + h * k1 / 2)
        k3 = f(time_s + h / 2, y + h * k2 / 2)
        k4 = f(time_s + h, y + h * k3)
        result = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.isfinite(result).all():
        raise ArithmeticError("RK4 produced a nonfinite candidate")
    return RK4Step(tuple(map(float, result)), 4)
