"""Density virial mixing on the common analytic derivative arithmetic.

SI units; composition derivatives explicitly select unconstrained fractions or
n-1 independent fractions (the last component closes the simplex). Parameters
are held fixed during differentiation. No upstream implementation is imported.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from math import factorial, isclose, isfinite, log, sqrt
from typing import Any, TypeAlias, cast

from chemworld.physchem.caloric_math import FirstDerivative, scalar_value
from chemworld.physchem.virial_kernels import _real, virial_derivatives

Scalar: TypeAlias = float | FirstDerivative[Any, Any]
Matrix: TypeAlias = tuple[tuple[float, ...], ...]
Tensor3: TypeAlias = tuple[Matrix, ...]


def _vector(values: Iterable[float], name: str, *, positive: bool = False) -> tuple[float, ...]:
    result = tuple(_real(v, name, positive=positive) for v in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _square(values: Iterable[Iterable[float]], n: int, name: str) -> Matrix:
    rows = tuple(tuple(_real(v, name) for v in row) for row in values)
    if len(rows) != n or any(len(row) != n for row in rows):
        raise ValueError(f"{name} must be square and aligned")
    if any(
        not isclose(rows[i][j], rows[j][i], rel_tol=1e-12, abs_tol=0)
        for i, j in product(range(n), repeat=2)
    ):
        raise ValueError(f"{name} must be symmetric")
    return rows


def lee_kesler_cross_volumes(critical_volumes: Iterable[float]) -> Matrix:
    v = _vector(critical_volumes, "critical volume", positive=True)
    return tuple(tuple((a ** (1 / 3) + b ** (1 / 3)) ** 3 / 8 for b in v) for a in v)


def tarakad_danner_interactions(critical_volumes: Iterable[float]) -> Matrix:
    v = _vector(critical_volumes, "critical volume", positive=True)
    cross = lee_kesler_cross_volumes(v)
    return tuple(
        tuple(0.0 if i == j else 1 - sqrt(a * b) / cross[i][j] for j, b in enumerate(v))
        for i, a in enumerate(v)
    )


def meng_duan_interactions(
    species: Iterable[tuple[str, int]], *, unknown: str = "raise",
) -> Matrix:
    """Explicit (class, carbon count) inputs, not structural inference from CAS.

    Classes: alkane, carbon_dioxide, nitrogen, other. The alkane caller must
    establish the applicable structure. Unknown pairs need an explicit zero
    policy to reproduce the public API's fallback. Constants and alkane form
    are public-call recovered and holdout checked, not paper-table verified.
    """
    if unknown not in {"raise", "zero"}:
        raise ValueError("unknown-pair policy must be raise or zero")
    species = tuple(cast(tuple[str, int], tuple(s)) for s in species)
    if not species:
        raise ValueError("species must not be empty")
    for kind, count in species:
        if kind not in {"alkane", "carbon_dioxide", "nitrogen", "other"}:
            raise ValueError("unknown explicit Meng-Duan class")
        if type(count) is not int or (count < 1 if kind == "alkane" else count != 0):
            raise ValueError(
                "alkane requires positive integer carbon count; other classes use zero"
            )

    def pair(a: tuple[str, int], b: tuple[str, int]) -> float:
        ka, na = a
        kb, nb = b
        if ka == kb == "alkane":
            return 0.00678 / (1 + 0.336 * min(na, nb)) * log(1 + abs(na - nb)) ** 3.5
        if ka == "alkane":
            ka, kb, na, nb = kb, ka, nb, na
        if kb == "alkane" and ka in {"carbon_dioxide", "nitrogen"}:
            slope = 0.07475 if ka == "carbon_dioxide" else 0.04311
            return slope * log(1 + nb) ** 1.5
        if ka == kb and ka != "other":
            return 0.0
        if unknown == "zero":
            return 0.0
        raise ValueError(
            "Meng-Duan pair has no declared correlation; supply an explicit interaction"
        )

    return tuple(
        tuple(0.0 if i == j else pair(a, b) for j, b in enumerate(species))
        for i, a in enumerate(species)
    )


def virial_cross_parameters(
    critical_temperatures: Iterable[float],
    critical_pressures: Iterable[float],
    critical_volumes: Iterable[float],
    acentric_factors: Iterable[float],
    *,
    interactions: Iterable[Iterable[float]] | None = None,
) -> tuple[tuple[dict[str, float], ...], ...]:
    """Tarakad-Danner Tc/Pc/omega and Lee-Kesler Vc, with explicit optional kij."""
    t = _vector(critical_temperatures, "critical temperature", positive=True)
    p = _vector(critical_pressures, "critical pressure", positive=True)
    v = _vector(critical_volumes, "critical volume", positive=True)
    w = _vector(acentric_factors, "acentric factor")
    n = len(t)
    if any(len(x) != n for x in (p, v, w)):
        raise ValueError("cross-parameter vectors must be aligned")
    k = tarakad_danner_interactions(v) if interactions is None else _square(interactions, n, "kij")
    if any(k[i][i] != 0 for i in range(n)) or any(a >= 1 for row in k for a in row):
        raise ValueError("kij diagonal must be zero and cross temperatures positive")
    vc = lee_kesler_cross_volumes(v)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            tc = sqrt(t[i] * t[j]) * (1 - k[i][j])
            row.append(
                {
                    "Tc": tc,
                    "Pc": tc * (p[i] * v[i] / t[i] + p[j] * v[j] / t[j]) / (2 * vc[i][j]),
                    "Vc": vc[i][j],
                    "omega": (w[i] + w[j]) / 2,
                }
            )
        result.append(tuple(row))
    return tuple(result)


def virial_coefficient_matrices(
    method: str, temperature_K: float, parameter_rows: Iterable[Iterable[Mapping[str, float]]],
) -> tuple[Matrix, ...]:
    """Return four symmetric coefficient matrices, ordered by T derivative."""
    rows = tuple(
        tuple(virial_derivatives(method, temperature_K, p) for p in row) for row in parameter_rows
    )
    n = len(rows)
    if not n or any(len(row) != n for row in rows):
        raise ValueError("parameter rows must be nonempty and square")
    return tuple(
        _square(
            [[rows[i][j][d] for j in range(n)] for i in range(n)],
            n,
            "virial temperature derivative",
        )
        for d in range(4)
    )


def _partial(
    function: Callable[[Sequence[Scalar]], Scalar], values: Sequence[Scalar], axes: Sequence[int],
) -> Scalar:
    """Seed the existing first-order algebra; no second derivative engine."""
    if not axes:
        return function(values)
    axis, *rest = axes
    seeded: tuple[FirstDerivative[Any, float], ...] = tuple(
        FirstDerivative(value, float(i == axis)) for i, value in enumerate(values)
    )
    result = _partial(function, seeded, rest)
    return result.slope if isinstance(result, FirstDerivative) else 0.0


def _zero(value: Scalar) -> bool:
    if isinstance(value, FirstDerivative):
        return _zero(value.value) and _zero(value.slope)
    return value == 0


def _cube_root(value: Scalar) -> Scalar:
    primal = scalar_value(value)
    if primal == 0:
        if _zero(value):
            return 0.0
        raise ValueError("zero C product has no declared differentiable temperature continuation")
    return value ** (1 / 3) if primal > 0 else -((-value) ** (1 / 3))


@dataclass(frozen=True)
class VirialCoefficientResponse:
    temperature_derivatives: tuple[float, ...]
    gradient: tuple[float, ...]
    hessian: Matrix
    third: Tensor3
    temperature_gradient: tuple[float, ...]
    temperature_hessian: Matrix
    second_temperature_gradient: tuple[float, ...]
    coordinates: str

    @property
    def value(self) -> float:
        return self.temperature_derivatives[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def virial_mixture_response(
    fractions: Iterable[float], coefficient_derivatives: Iterable[Iterable[Iterable[float]]], *,
    kind: str = "B", coordinates: str = "unconstrained",
) -> VirialCoefficientResponse:
    """B quadratic or signed-real Orentlicher-Prausnitz C cubic mixing.

    coefficient_derivatives contains value, dT, dT2, dT3 matrices at one T.
    No normalization is applied: derivative coordinates remain explicit.
    Zero T-varying cross products are rejected when their cube-root derivative
    is not defined by these inputs; pure diagonal Ciii = Cii stays smooth.
    """
    z = _vector(fractions, "mole fractions")
    if any(x < 0 for x in z) or not isclose(sum(z), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("nonnegative mole fractions must sum to one")
    if kind not in {"B", "C"} or coordinates not in {"unconstrained", "independent"}:
        raise ValueError("invalid virial kind or composition coordinates")
    n = len(z)
    matrices = tuple(_square(a, n, "coefficient derivative") for a in coefficient_derivatives)
    if len(matrices) != 4:
        raise ValueError("supply all four temperature coefficient matrices explicitly")
    count = n if coordinates == "unconstrained" else n - 1
    point = (0.0, *z[:count])

    def formula(q: Sequence[Scalar]) -> Scalar:
        t, *ys = q
        if coordinates == "independent":
            ys.append(1 - sum(ys))
        matrix = [
            [sum(matrices[d][i][j] * t**d / factorial(d) for d in range(4)) for j in range(n)]
            for i in range(n)
        ]
        if kind == "B":
            return sum(ys[i] * ys[j] * matrix[i][j] for i, j in product(range(n), repeat=2))
        return sum(
            ys[i]
            * ys[j]
            * ys[k]
            * (
                matrix[i][i]
                if i == j == k
                else _cube_root(matrix[i][j] * matrix[j][k] * matrix[i][k])
            )
            for i, j, k in product(range(n), repeat=3)
        )

    def d(*axes: int) -> float:
        value = float(cast(float, _partial(formula, point, axes)))
        if not isfinite(value):
            raise ValueError("nonfinite mixture derivative")
        return value

    axes = range(1, count + 1)
    return VirialCoefficientResponse(
        tuple(d(*(0,) * order) for order in range(4)),
        tuple(d(i) for i in axes),
        tuple(tuple(d(i, j) for j in axes) for i in axes),
        tuple(tuple(tuple(d(i, j, k) for k in axes) for j in axes) for i in axes),
        tuple(d(0, i) for i in axes),
        tuple(tuple(d(0, i, j) for j in axes) for i in axes),
        tuple(d(0, 0, i) for i in axes),
        coordinates,
    )
