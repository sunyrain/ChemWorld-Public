"""Correlation-integral multipoles and stationary Gray鈥揋ubbins polarization.

SI parameter declarations, Angstrom-conditioned arithmetic, and the common
residual solver/analytic jets. The GT and Gray permanent energy equations are
algebraically identical after moment conversion, except for signed quadrupoles
and the explicit unlike-energy convention. No upstream evaluator is embedded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite, pi, prod, sqrt
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    compensated_sum,
    derivative_depth,
    exponential,
    logarithm,
    scalar_value,
)
from chemworld.physchem.residual_system import (
    ResidualVariable,
    RootPolicy,
    linear_solve,
    solve_root,
)
from chemworld.physchem.saft_polar_extensions_data import (
    GOTTSCHALK_J,
    GOTTSCHALK_K,
    GUBBINSTWU_J,
    GUBBINSTWU_K,
    LUCKAS_J,
    LUCKAS_K,
)

_MOMENT_SCALE = sqrt(1 / (4 * pi * 8.8541878128e-12 * 1.380649e-23))
_TABLES: dict[str, tuple[dict, dict]] = {
    "Luckas": (LUCKAS_J, LUCKAS_K),
    "GubbinsTwu": (GUBBINSTWU_J, GUBBINSTWU_K),
    "Gottschalk": (GOTTSCHALK_J, GOTTSCHALK_K),
}


def _zero(value: Any) -> bool:
    if isinstance(value, FirstDerivative):
        return _zero(value.value) and _zero(value.slope)
    if hasattr(value, "dxy"):
        return all(_zero(getattr(value, f)) for f in ("value", "dx", "dy", "dxx", "dxy", "dyy"))
    return bool(value == 0)


def correlation_integral(
    family: str, order: int | tuple[Any, Any], temperature_reduced: Any, density_reduced: Any
) -> Any:
    """J_n or K_(n1,n2), including every published upstream table entry.

    Gottschalk K also accepts two triplets (orders with entries above nine).
    Luckas J has a finite vacuum value but no finite density Taylor expansion;
    a density jet at that boundary is explicitly rejected. At sqrt(2) it is C2,
    so third-order jets are rejected instead of choosing one side of the cusp.
    """
    t, r = temperature_reduced, density_reduced
    if (
        family not in _TABLES
        or not isfinite(scalar_value(t))
        or scalar_value(t) <= 0
        or not isfinite(scalar_value(r))
        or scalar_value(r) < 0
    ):
        raise ValueError("correlation integral requires known family, T*>0 and rho*>=0")
    pair = isinstance(order, tuple)
    key: Any = order
    if family == "Gottschalk" and isinstance(order, tuple):
        key = tuple(tuple(map(int, str(v))) if isinstance(v, int) else tuple(v) for v in order)
    try:
        a = _TABLES[family][int(pair)][key]
    except (KeyError, TypeError) as exc:
        raise ValueError("undeclared correlation-integral order") from exc
    if family == "GubbinsTwu":
        lt = logarithm(t)
        value = exponential(
            a[0] * r * r * lt + a[1] * r * r + a[2] * r * lt + a[3] * r + a[4] * lt + a[5]
        )
        return -value if order == (334, 445) else value
    if family == "Luckas":
        width = 4 if pair else 3
        b = [compensated_sum(a[width * i + j] * r**i for i in range(4)) for j in range(width)]
        if pair:
            w = (1 - r / sqrt(2)) ** 4
            return b[0] + b[1] * t + b[2] * exponential(2 * w) + b[3] * exponential(3 * w)
        if scalar_value(r) == 0:
            if not _zero(r):
                raise ValueError("Luckas J has no finite density Taylor expansion at vacuum")
            exponent = 1 / t
        else:
            logarithm_r = logarithm(r / sqrt(2))
            if scalar_value(logarithm_r) == 0 and derivative_depth(r) >= 3:
                raise ValueError("Luckas J is only C2 at rho*=sqrt(2)")
            magnitude = logarithm_r if scalar_value(logarithm_r) >= 0 else -logarithm_r
            cube = magnitude**3
            exponent = cube / (t * cube + 4)
        if not isinstance(order, int):
            raise AssertionError("invariant violated: isinstance(order, int)")
        return (b[0] + b[1] * t ** (0.3 + 0.05 * order) + b[2] * t ** (1 / order)) * exponential(
            exponent
        )
    if not pair:
        if not isinstance(order, int):
            raise AssertionError("invariant violated: isinstance(order, int)")
        value = compensated_sum(a[4 * i + j] * r**i * t**j for i in range(5) for j in range(4))
        value += exponential(1 / t) * compensated_sum(
            a[20 + 3 * i + j] * r**i * t**j for i in range(5) for j in range(3)
        )
        return value ** (order - 2)
    return compensated_sum(
        [
            *(
                a[2 * i + j - 1] * r**i * exponential(j * (1 - r / 3) / t)
                for i in range(4)
                for j in (1, 2)
            ),
            *(
                a[8 + 2 * i + j - 1] * r**i * exponential(j * (1 - r / 3) ** 2 / t)
                for i in range(4)
                for j in (1, 2)
            ),
            *(a[16 + 4 * i + j] * r**i * t**j for i in range(6) for j in range(4)),
        ]
    )


@dataclass(frozen=True)
class CorrelationMultipole:
    component_id: str
    dipole_C_m: float
    quadrupole_C_m2: float
    source: str
    alpha_parallel_m3: float = 0
    alpha_perpendicular_m3: float = 0

    def __post_init__(self) -> None:
        if (
            not self.component_id
            or not self.source
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (
                    self.dipole_C_m,
                    self.quadrupole_C_m2,
                    self.alpha_parallel_m3,
                    self.alpha_perpendicular_m3,
                )
            )
            or min(self.alpha_parallel_m3, self.alpha_perpendicular_m3) < 0
        ):
            raise ValueError(
                "multipoles require identity, source and finite SI moments/polarizability"
            )


@dataclass(frozen=True)
class CorrelationPolar:
    moments: tuple[CorrelationMultipole, ...]
    source: str
    formulation: str = "GubbinsTwu"
    integrals: str = "GubbinsTwu"
    density_reduction: str = "packing_fraction"
    third_order_scale: float = 1
    cross_third_order_scale: float = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "moments", tuple(self.moments))
        if (
            not self.source
            or not self.moments
            or len({m.component_id for m in self.moments}) != len(self.moments)
            or self.formulation not in ("GubbinsTwu", "GrayGubbins")
            or self.integrals not in _TABLES
            or self.density_reduction not in ("packing_fraction", "Gubbins")
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (self.third_order_scale, self.cross_third_order_scale)
            )
        ):
            raise ValueError(
                "correlation polar declaration has invalid identity, family or controls"
            )
        if self.formulation == "GubbinsTwu" and (
            any(m.alpha_parallel_m3 or m.alpha_perpendicular_m3 for m in self.moments)
            or self.third_order_scale != 1
            or self.cross_third_order_scale != 1
        ):
            raise ValueError("polarizability and C3 controls belong to Gray-Gubbins formulation")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "correlation-polar", **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrelationPolar:
        values = dict(data)
        if values.pop("kind", None) != "correlation-polar":
            raise ValueError("unknown correlation polar encoding")
        values["moments"] = tuple(CorrelationMultipole(**p) for p in values["moments"])
        return cls(**values)


def _integrals(
    spec: CorrelationPolar, t: Any, r: Any, s: Sequence[float], e: Sequence[Sequence[float]]
) -> tuple[dict, dict]:
    pair, triple = {}, {}
    n = len(s)
    sij = {(i, j): (s[i] + s[j]) / 2 for i, j in product(range(n), repeat=2)}
    ks = {}
    for i, j in product(range(n), repeat=2):
        for order in (6, 8, 10, 11, 13, 15):
            pair[i, j, order] = (
                4
                * pi
                / sij[i, j] ** (order - 3)
                * correlation_integral(spec.integrals, order, t / e[i][j], r)
            )
        for korder in ((222, 333), (233, 344), (334, 445), (444, 555)):
            ks[i, j, korder] = correlation_integral(spec.integrals, korder, t / e[i][j], r)
    for i, j, k in product(range(n), repeat=3):
        a, b, c = sij[i, j], sij[i, k], sij[j, k]
        factors = (
            64 * pi**3 / 5 * sqrt(14 * pi / 5) / (a * b * c),
            2048 * pi**3 / 7 * sqrt(3 * pi) / (a * (b * c) ** 2),
            -4096 * pi**3 / 9 * sqrt(22 * pi / 7) / ((a * b) ** 2 * c**3),
            8192 * pi**3 / 81 * sqrt(2002 * pi) / (a * b * c) ** 3,
        )
        for h, korder in enumerate(((222, 333), (233, 344), (334, 445), (444, 555))):
            value = ks[i, j, korder] * ks[i, k, korder] * ks[j, k, korder]
            if scalar_value(value) == 0 and not _zero(value):
                raise ValueError(
                    "triplet integral has no finite cube-root Taylor expansion at zero"
                )
            root = (
                -((-value) ** (1 / 3))
                if scalar_value(value) < 0
                else value ** (1 / 3)
                if scalar_value(value)
                else 0
            )
            triple[i, j, k, h] = factors[h] * root
    return pair, triple


def _energy_terms(
    spec: CorrelationPolar, t: Any, rho: Any, u: Sequence[Any], pair: dict, triple: dict
) -> list[tuple[tuple[int, ...], Any, Any]]:
    """(composition monomial, alpha2/rho, alpha3/rho), expanded for face limits."""
    q = [m.quadrupole_C_m2 * _MOMENT_SCALE * 1e25 for m in spec.moments]
    if spec.formulation == "GubbinsTwu":
        q = list(map(abs, q))
    a = [(m.alpha_parallel_m3 + 2 * m.alpha_perpendicular_m3) / 3 * 1e30 for m in spec.moments]
    g = [(m.alpha_parallel_m3 - m.alpha_perpendicular_m3) * 1e30 for m in spec.moments]
    d = [v * v / (3 * t) for v in u]
    terms: list[tuple[tuple[int, ...], Any, Any]] = []
    n = len(u)
    for i, j in product(range(n), repeat=2):
        second = -(
            1.5 * (d[i] * d[j] + d[i] * a[j] + a[i] * d[j]) * pair[i, j, 6]
            + 1.5 * (d[i] + a[i]) * q[j] ** 2 / t * pair[i, j, 8]
            + 0.7 * q[i] ** 2 * q[j] ** 2 / t**2 * pair[i, j, 10]
        )
        third = (
            (3.6 * d[i] * d[j] + 2.4 * g[i] * d[j] + 0.16 * g[i] * g[j])
            * q[i]
            * q[j]
            / t
            * pair[i, j, 11]
            + 12 / 35 * (3 * d[i] + g[i]) * q[i] * q[j] ** 3 / t**2 * pair[i, j, 13]
            + 36 / 245 * q[i] ** 3 * q[j] ** 3 / t**3 * pair[i, j, 15]
        )
        if not _zero(second) or not _zero(third):
            terms.append(((i, j), second, spec.third_order_scale * third))
        for k in range(n):
            zproduct = (
                d[i] * d[j] * d[k]
                + d[i] * d[j] * a[k]
                + d[i] * a[j] * d[k]
                + a[i] * d[j] * d[k]
                + d[i] * a[j] * a[k]
                + a[i] * d[j] * a[k]
                + a[i] * a[j] * d[k]
            )
            third = (
                0.5 * zproduct * triple[i, j, k, 0]
                + spec.cross_third_order_scale
                * (
                    3 / 160 * (d[i] + a[i]) * (d[j] + a[j]) * q[k] ** 2 / t * triple[i, j, k, 1]
                    + 3 / 640 * (d[i] + a[i]) * q[j] ** 2 * q[k] ** 2 / t**2 * triple[i, j, k, 2]
                )
                + 1 / 6400 * q[i] ** 2 * q[j] ** 2 * q[k] ** 2 / t**3 * triple[i, j, k, 3]
            )
            if not _zero(third):
                terms.append(((i, j, k), 0, spec.third_order_scale * rho * third))
    return terms


def _pade(terms: list, rho: Any, x: Sequence[Any], gradient_index: int | None = None) -> Any:
    if not terms:
        return rho * 0
    common = Counter(terms[0][0])
    for indices, _, _ in terms[1:]:
        common &= Counter(indices)
    factors = list(common.elements())
    weights, da, db = [], [], []
    for indices, second, third in terms:
        rest = list(indices)
        for i in factors:
            rest.remove(i)
        weight = prod(x[i] for i in rest)
        if gradient_index is None:
            weights.append((weight * second, weight * third))
        else:
            weights.append(
                (
                    weight * second.value
                    if isinstance(second, FirstDerivative)
                    else weight * second,
                    weight * third.value if isinstance(third, FirstDerivative) else weight * third,
                )
            )
            if gradient_index not in factors:
                if gradient_index not in rest:
                    continue
                rest.remove(gradient_index)
                weight = prod(x[i] for i in rest)
            da.append(weight * second.slope if isinstance(second, FirstDerivative) else 0)
            db.append(weight * third.slope if isinstance(third, FirstDerivative) else 0)
    a = compensated_sum(v[0] for v in weights)
    b = compensated_sum(v[1] for v in weights)
    denominator = a - b
    if scalar_value(denominator) == 0:
        if _zero(a) and _zero(b):
            return rho * 0
        raise ValueError("polar Pade denominator or composition Taylor limit is singular")
    if gradient_index is None:
        return prod(x[i] for i in factors) * rho * a * a / denominator
    if gradient_index in factors:
        factors.remove(gradient_index)
    return (
        prod(x[i] for i in factors)
        * rho
        * (a * (a - 2 * b) * compensated_sum(da) + a * a * compensated_sum(db))
        / denominator**2
    )


def _lift(value: Any) -> FirstDerivative:
    return FirstDerivative(value, 0)


def _gradient(
    spec: CorrelationPolar,
    t: Any,
    rho: Any,
    x: Sequence[Any],
    u: Sequence[Any],
    pair: dict,
    triple: dict,
    i: int,
) -> Any:
    """Moment derivative per molecule of species i; no division by x_i."""
    terms = _energy_terms(
        spec,
        _lift(t),
        _lift(rho),
        [FirstDerivative(v, int(j == i)) for j, v in enumerate(u)],
        {k: _lift(v) for k, v in pair.items()},
        {k: _lift(v) for k, v in triple.items()},
    )
    return _pade(terms, rho, x, i)


def _polarized_moments(
    spec: CorrelationPolar,
    t: Any,
    rho: Any,
    x: Sequence[Any],
    permanent: Sequence[float],
    pair: dict,
    triple: dict,
) -> tuple[Any, ...]:
    active = [i for i, m in enumerate(spec.moments) if m.alpha_parallel_m3 and permanent[i]]
    if not active:
        return tuple(permanent)
    alpha = [m.alpha_parallel_m3 * 1e30 for m in spec.moments]

    def residual(
        values: Sequence[Any],
        temperature: Any,
        density: Any,
        fractions: Sequence[Any],
        pairs: dict,
        triples: dict,
    ) -> list[Any]:
        u = list(permanent)
        for i, value in zip(active, values, strict=True):
            u[i] = value
        return [
            u[i]
            - permanent[i]
            + temperature
            * alpha[i]
            * _gradient(spec, temperature, density, fractions, u, pairs, triples, i)
            for i in active
        ]

    tp, rp = scalar_value(t), scalar_value(rho)
    xp = [scalar_value(v) for v in x]
    pp = {k: scalar_value(v) for k, v in pair.items()}
    kp = {k: scalar_value(v) for k, v in triple.items()}

    def jacobian(values: Sequence[float]) -> np.ndarray:
        columns = []
        for j in range(len(active)):
            lifted = [FirstDerivative(v, int(j == k)) for k, v in enumerate(values)]
            out = residual(
                lifted,
                _lift(tp),
                _lift(rp),
                list(map(_lift, xp)),
                {k: _lift(v) for k, v in pp.items()},
                {k: _lift(v) for k, v in kp.items()},
            )
            columns.append([v.slope if isinstance(v, FirstDerivative) else 0 for v in out])
        return np.asarray(columns, dtype=float).T

    scale = np.asarray([max(abs(permanent[i]), 1) for i in active])
    root = solve_root(
        lambda v: np.asarray(residual(v, tp, rp, xp, pp, kp), dtype=float),
        np.asarray([permanent[i] for i in active]),
        jacobian=jacobian,
        variables=tuple(
            ResidualVariable(f"induced_{i}", "sqrt(K A3)", role="algebraic") for i in active
        ),
        residual_scales=scale,
        policy=RootPolicy(tolerance=2e-12, maximum_iterations=60),
    )
    if not root.success:
        raise ValueError("self-consistent polarization failed: " + root.failure_reason)
    jac = jacobian(root.value)
    # The per-species normalization is a positive diagonal congruence of the
    # free-energy Hessian. Reject a stationary polarization catastrophe.
    present = [j for j, i in enumerate(active) if xp[i] > 0]
    if present:
        metric = np.asarray([xp[active[j]] / (tp * alpha[active[j]]) for j in present])
        stability = metric[:, None] * jac[np.ix_(present, present)]
        if np.min(np.linalg.eigvalsh((stability + stability.T) / 2)) <= 0:
            raise ValueError("self-consistent polarization is not a local energy minimum")
    values: list[Any] = list(root.value)
    depth = max(derivative_depth(v) for v in (t, rho, *x, *pair.values(), *triple.values()))
    if depth:
        inverse = linear_solve(jac, np.eye(len(active)))
        for _ in range(depth):
            errors = residual(values, t, rho, x, pair, triple)
            values = [
                values[i]
                - compensated_sum(float(inverse[i, j]) * errors[j] for j in range(len(active)))
                for i in range(len(active))
            ]
    result: list[Any] = list(permanent)
    for i, value in zip(active, values, strict=True):
        result[i] = value
    return tuple(result)


def correlation_polar_helmholtz_contributions(
    spec: CorrelationPolar,
    temperature_K: Any,
    number_density_A3: Any,
    eta: Any,
    component_ids: Sequence[str],
    mole_fractions: Sequence[Any],
    diameters_A: Sequence[float],
    well_depths_K: Sequence[float],
    interaction: Sequence[Sequence[float]],
    energy_mixing: str = "Lafitte",
) -> dict[str, Any]:
    """Mie consumer entry point; thermodynamic derivatives remain shared jets."""
    if tuple(m.component_id for m in spec.moments) != tuple(component_ids):
        raise ValueError("polar moment ordering must match EOS component ordering")
    if energy_mixing not in ("Lafitte", "Lorentz-Berthelot"):
        raise ValueError("unknown unlike-energy mixing convention")
    t, rho, x, s, e = temperature_K, number_density_A3, mole_fractions, diameters_A, well_depths_K
    n = len(spec.moments)
    if (
        len(x) != n
        or len(s) != n
        or len(e) != n
        or len(interaction) != n
        or any(len(row) != n for row in interaction)
        or any(not isfinite(scalar_value(v)) or scalar_value(v) < 0 for v in (rho, eta, *x))
        or not isfinite(scalar_value(t))
        or scalar_value(t) <= 0
        or any(not isfinite(v) or v <= 0 for v in (*s, *e))
    ):
        raise ValueError("correlation polar requires consistent finite physical state/parameters")
    if not any(m.dipole_C_m or m.quadrupole_C_m2 for m in spec.moments):
        return {"correlation_polar": rho * 0}
    r = (
        eta / (pi / 6)
        if spec.density_reduction == "packing_fraction"
        else rho
        * compensated_sum(
            x[i] * x[j] * ((s[i] + s[j]) / 2) ** 3 for i, j in product(range(n), repeat=2)
        )
    )
    eps = [[sqrt(e[i] * e[j]) for j in range(n)] for i in range(n)]
    if spec.formulation == "GrayGubbins":
        eps = [
            [
                eps[i][j]
                * (1 - interaction[i][j])
                * (
                    (s[i] * s[j]) ** 1.5 / ((s[i] + s[j]) / 2) ** 3
                    if energy_mixing == "Lafitte"
                    else 1
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
    if any(v <= 0 or not isfinite(v) for row in eps for v in row):
        raise ValueError("correlation integral requires positive unlike well depth")
    pair, triple = _integrals(spec, t, r, s, eps)
    permanent = tuple(m.dipole_C_m * _MOMENT_SCALE * 1e15 for m in spec.moments)
    effective = _polarized_moments(spec, t, rho, x, permanent, pair, triple)
    energy = _pade(_energy_terms(spec, t, rho, effective, pair, triple), rho, x)
    cost = compensated_sum(
        x[i] * (effective[i] - permanent[i]) ** 2 / (2 * t * m.alpha_parallel_m3 * 1e30)
        for i, m in enumerate(spec.moments)
        if m.alpha_parallel_m3
    )
    return {"correlation_polar": energy + cost}
