"""Rössler's weak second-order RI5 step for an Itô drift/diffusion callback.

Coefficients: SIAM J. Numer. Anal. 47 (2009), Table 5.2, RI5,
https://doi.org/10.1137/060673308. The weak random-variable convention follows
the RI5 choice used by COPASI. These variables are not sampled Wiener paths.
This numerical primitive owns no inventory, clock, RNG stream or boundary rule.
Trial values remain signed; physical acceptance belongs to the caller's transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import Self, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

RI5Callback: TypeAlias = Callable[[float, NDArray[np.float64]], ArrayLike]


class RI5DrawData(TypedDict):
    schema: str
    hat: list[float]
    tilde: list[int]

RI5_DRAW_SCHEMA = "chemworld-ri5-weak-draw-v1"
MAX_RI5_NOISE_CHANNELS = 256
_SQRT3 = sqrt(3.0)


@dataclass(frozen=True)
class RI5Draw:
    """Dimensionless three-point and two-point variables, explicit and JSON-safe."""

    hat: tuple[float, ...]
    tilde: tuple[int, ...]

    def __post_init__(self) -> None:
        try:
            hat, tilde = tuple(self.hat), tuple(self.tilde)
        except TypeError as exc:
            raise ValueError("RI5 random variables require finite sequences") from exc
        if len(hat) != len(tilde) or len(hat) > MAX_RI5_NOISE_CHANNELS:
            raise ValueError("RI5 random variables require matching bounded dimensions")
        if any(
            isinstance(v, bool) or not isinstance(cast(object, v), Real)
            or v not in {-_SQRT3, 0.0, _SQRT3}
            for v in hat
        ):
            raise ValueError("RI5 hat variables must lie on the declared three-point lattice")
        if any(type(v) is not int or v not in {-1, 1} for v in tilde):
            raise ValueError("RI5 tilde variables must be exact signed integers")
        object.__setattr__(self, "hat", tuple(float(v) for v in hat))
        object.__setattr__(self, "tilde", tilde)

    @property
    def dimension(self) -> int:
        return len(self.hat)

    def to_dict(self) -> RI5DrawData:
        return {"schema": RI5_DRAW_SCHEMA, "hat": list(self.hat), "tilde": list(self.tilde)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "hat", "tilde"}
            or value["schema"] != RI5_DRAW_SCHEMA
            or not isinstance(value["hat"], list)
            or not isinstance(value["tilde"], list)
        ):
            raise ValueError("unknown RI5 random-variable encoding")
        # The constructor validates each element against the two exact lattices.
        return cls(tuple(cast(list[float], value["hat"])), tuple(cast(list[int], value["tilde"])))


def draw_ri5(rng: np.random.Generator, dimension: int) -> RI5Draw:
    """Draw only from the caller-owned generator; an empty diffusion consumes none."""
    if (
        not isinstance(rng, np.random.Generator)
        or type(dimension) is not int
        or not 0 <= dimension <= MAX_RI5_NOISE_CHANNELS
    ):
        raise ValueError("RI5 requires an explicit generator and bounded noise dimension")
    if dimension == 0:
        return RI5Draw((), ())
    indices = rng.integers(0, 6, size=dimension)
    hat = tuple(-_SQRT3 if i == 0 else _SQRT3 if i == 1 else 0.0 for i in indices)
    tilde = tuple(1 if i else -1 for i in rng.integers(0, 2, size=dimension))
    return RI5Draw(hat, tilde)


@dataclass(frozen=True)
class RI5Step:
    values: tuple[float, ...]
    drift_evaluations: int
    diffusion_evaluations: int


def _finite_array(
    value: ArrayLike, shape: tuple[int, ...], label: str,
) -> NDArray[np.float64]:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("non-real numerical dtype")
        result = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RI5 {label} must be a real numeric array") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"RI5 {label} has an invalid shape or nonfinite value")
    return result.copy()


def ri5_step(
    drift: RI5Callback,
    diffusion: RI5Callback | None,
    time_s: float,
    values: ArrayLike,
    duration_s: float,
    draw: RI5Draw,
) -> RI5Step:
    """Evaluate one fixed step using already drawn weak random variables.

    Callbacks receive time and a private state vector. Drift returns shape (n,),
    diffusion shape (n,m), where m is the draw dimension. The physical compiler
    owns dimensions and boundary policy; this step never clips a trial or result.
    The RI5 hat sign convention is weak in law, not a strong Brownian increment.
    """
    if (
        not isinstance(draw, RI5Draw)
        or not callable(drift)
        or (draw.dimension and not callable(diffusion))
    ):
        raise ValueError("RI5 requires declared drift, diffusion and random variables")
    for value in (time_s, duration_s):
        if (
            isinstance(value, bool)
            or not isinstance(cast(object, value), Real)
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError("RI5 time and duration must be finite and nonnegative")
    time_s, h = float(time_s), float(duration_s)
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("RI5 state must be a nonempty finite real vector")
    y = _finite_array(raw, raw.shape, "state")
    n, m = len(y), draw.dimension
    if h == 0:
        return RI5Step(tuple(map(float, y)), 0, 0)
    if not isfinite(time_s + h) or time_s + h == time_s:
        raise ValueError("RI5 step cannot advance its finite physical clock")
    evaluations = [0, 0]

    def a(at: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        evaluations[0] += 1
        return _finite_array(drift(at, state.copy()), (n,), "drift")

    def b(at: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        if not m:
            return np.zeros((n, 0))
        evaluations[1] += 1
        return _finite_array(cast(RI5Callback, diffusion)(at, state.copy()), (n, m), "diffusion")

    xi = np.array(draw.hat)
    eta = np.array(draw.tilde)
    integrals = np.empty((m, m))
    for k in range(m):
        for ell in range(m):
            correction = -1.0 if k == ell else -eta[k] if k < ell else eta[ell]
            integrals[k, ell] = 0.5 * (xi[k] * xi[ell] + correction)
    diagonal = np.diag(integrals).copy()
    np.fill_diagonal(integrals, 0.0)
    root_h = sqrt(h)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a1 = a(time_s, y)
        b1 = b(time_s, y)
        cross = b1 @ integrals.T
        a2 = a(time_s + h, y + h * a1 + root_h / 3.0 * (b1 @ xi))
        b2, bb2, b3, bb3 = (np.empty((n, m)) for _ in range(4))
        for k in range(m):
            b2[:, k] = b(time_s + h / 4.0, y + h / 4.0 * a1 + root_h / 2.0 * b1[:, k])[:, k]
            bb2[:, k] = b(time_s, y + root_h * cross[:, k])[:, k]
            b3[:, k] = b(time_s + h / 4.0, y + h / 4.0 * a1 - root_h / 2.0 * b1[:, k])[:, k]
            bb3[:, k] = b(time_s, y - root_h * cross[:, k])[:, k]
        a3 = a(
            time_s + 5.0 * h / 12.0,
            y + h * (25.0 / 144.0 * a1 + 35.0 / 144.0 * a2) - 5.0 * root_h / 6.0 * (b1 @ xi),
        )
        result = y + h * (a1 / 10.0 + 3.0 / 14.0 * a2 + 24.0 / 35.0 * a3)
        for k in range(m):
            result += root_h * (
                xi[k] * (b1[:, k] - b2[:, k] - b3[:, k])
                + diagonal[k] * (b2[:, k] - b3[:, k])
                + xi[k] * (b1[:, k] / 2.0 - bb2[:, k] / 4.0 - bb3[:, k] / 4.0)
                + (bb2[:, k] - bb3[:, k]) / 2.0
            )
    if not np.isfinite(result).all():
        raise ArithmeticError("RI5 produced a nonfinite candidate")
    return RI5Step(tuple(map(float, result)), *evaluations)
