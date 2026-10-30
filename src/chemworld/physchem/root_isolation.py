"""Bounded interval root isolation using the existing shared point-root solver.

Callbacks supply outward floating-point enclosures of the function and Jacobian.
Krawczyk exclusion/contraction and scalar monotonicity cover the declared box;
an exhausted or singular region is retained, never reclassified as no solution.
These numerical enclosures are not a formal exact-real proof system.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from math import inf, isfinite, nextafter
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chemworld.physchem.caloric_math import Interval
from chemworld.physchem.residual_system import (
    ResidualNumericalError,
    ResidualVariable,
    RootPolicy,
    linear_solve,
    solve_root,
)


@dataclass(frozen=True)
class IsolationPolicy:
    coordinate_tolerance: float = 1e-8
    maximum_boxes: int = 8192
    maximum_depth: int = 64
    root: RootPolicy = field(
        default_factory=lambda: RootPolicy(tolerance=1e-12, maximum_iterations=80)
    )

    def __post_init__(self) -> None:
        if not isfinite(self.coordinate_tolerance) or self.coordinate_tolerance <= 0:
            raise ValueError("isolation needs a positive finite coordinate tolerance")
        if any(type(n) is not int or n < 1 for n in (self.maximum_boxes, self.maximum_depth)):
            raise ValueError("isolation needs positive integer work limits")
        if not isinstance(self.root, RootPolicy):
            raise ValueError("isolation requires the common RootPolicy")


DEFAULT_ISOLATION_POLICY = IsolationPolicy()


class IsolatedRoot(TypedDict):
    value: list[float]
    lower: list[float]
    upper: list[float]
    certificate: str
    coordinate_error_bound: list[float]
    root_diagnostics: dict[str, Any]


class UnresolvedBox(TypedDict):
    lower: list[float]
    upper: list[float]
    reason: str


@dataclass(frozen=True)
class IsolatedRoots:
    roots: tuple[IsolatedRoot, ...]
    unresolved: tuple[UnresolvedBox, ...]
    diagnostics: dict[str, Any]

    @property
    def complete(self) -> bool:
        return not self.unresolved

    @property
    def status(self) -> Literal["unresolved", "no_solution", "unique", "multiple"]:
        return (
            "unresolved"
            if not self.complete
            else "no_solution"
            if not self.roots
            else "unique"
            if len(self.roots) == 1
            else "multiple"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "complete": self.complete,
            "roots": list(self.roots),
            "unresolved": list(self.unresolved),
            "diagnostics": dict(self.diagnostics),
        }


def isolate_roots(
    function: Callable[[NDArray[np.float64]], ArrayLike],
    jacobian: Callable[[NDArray[np.float64]], ArrayLike],
    enclosure: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        tuple[tuple[Interval, ...], tuple[tuple[Interval, ...], ...]],
    ],
    *,
    variables: Iterable[ResidualVariable],
    residual_scales: ArrayLike,
    policy: IsolationPolicy = DEFAULT_ISOLATION_POLICY,
) -> IsolatedRoots:
    """Enclose every possible root or explicitly retain an unresolved sub-box.

    enclosure(lower, upper) returns (tuple[Interval], tuple[tuple[Interval]]).
    Coordinates are normalized by declared variable scales; f uses row scales.
    All numerical point roots are delegated to residual_system.solve_root.
    """
    variables = tuple(variables)
    n = len(variables)
    scales = np.array([v.scale for v in variables])
    rows = np.asarray(residual_scales, dtype=float)
    if (
        not 1 <= n <= 8
        or rows.shape != (n,)
        or not np.all(np.isfinite(rows))
        or np.any(rows <= 0)
        or any(v.lower is None or v.upper is None for v in variables)
    ):
        raise ValueError("root isolation requires finite bounded square residuals")
    lo = np.array([v.lower for v in variables]) / scales
    hi = np.array([v.upper for v in variables]) / scales
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("root isolation requires finite coordinate bounds")
    if np.any(lo >= hi):
        raise ValueError("root isolation coordinate scaling collapsed the declared domain")
    pending = [(lo, hi, 0)]
    roots: list[IsolatedRoot] = []
    unresolved: list[UnresolvedBox] = []
    count, excluded, contractions = 0, 0, 0

    def remaining(a: NDArray[np.float64], b: NDArray[np.float64], reason: str) -> None:
        unresolved.append(
            {"lower": (a * scales).tolist(), "upper": (b * scales).tolist(), "reason": reason}
        )

    def value(u: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.asarray(function(u * scales), dtype=float) / rows
        if result.shape != (n,) or not np.all(np.isfinite(result)):
            raise ValueError("root callback returned a nonfinite/wrong-shaped vector")
        return result

    def accept(
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        certificate: str,
        error_bound: Callable[[NDArray[np.float64]], float],
    ) -> None:
        bounded = tuple(
            replace(v, lower=float(low), upper=float(high))
            for v, low, high in zip(variables, a * scales, b * scales, strict=True)
        )
        answer = solve_root(
            function,
            (a + b) * scales / 2,
            jacobian=jacobian,
            variables=bounded,
            residual_scales=rows,
            policy=policy.root,
        )
        if not answer.success:
            remaining(a, b, "point_root:" + answer.failure_reason)
            return
        root = np.asarray(answer.value) / scales
        bound = error_bound(root)
        if not np.isfinite(bound) or bound > policy.coordinate_tolerance:
            remaining(a, b, "point_root_coordinate_precision")
            return
        record: IsolatedRoot = {
            "value": answer.value.tolist(),
            "lower": (a * scales).tolist(),
            "upper": (b * scales).tolist(),
            "certificate": certificate,
            "coordinate_error_bound": (bound * scales).tolist(),
            "root_diagnostics": dict(answer.diagnostics),
        }
        # Overlapping boxes can share a boundary root. Do not merge close roots
        # from disjoint boxes, which would conceal a distinct physical branch.
        for old in roots:
            overlap = np.all(
                np.maximum(record["lower"], old["lower"])
                <= np.minimum(record["upper"], old["upper"])
            )
            if overlap and np.max(np.abs((answer.value - old["value"]) / scales)) <= (
                policy.coordinate_tolerance
            ):
                return
        roots.append(record)

    while pending and count < policy.maximum_boxes:
        a, b, depth = pending.pop()
        count += 1
        fbox, jbox = enclosure(a * scales, b * scales)
        if len(fbox) != n or len(jbox) != n or any(len(r) != n for r in jbox):
            raise ValueError("root enclosure must match the declared square system")
        fbox = tuple(f / row for f, row in zip(fbox, rows, strict=True))
        jbox = tuple(tuple(jbox[i][k] * (scales[k] / rows[i]) for k in range(n)) for i in range(n))
        if any(f.lower > 0 or f.upper < 0 for f in fbox):
            excluded += 1
            continue
        if np.any(a >= b):
            remaining(a, b, "floating_point_interval_collapsed")
            continue
        if n == 1 and (jbox[0][0].lower > 0 or jbox[0][0].upper < 0):
            fa, fb = value(a)[0], value(b)[0]
            if fa * fb > 0 and min(abs(fa), abs(fb)) > policy.root.tolerance:
                excluded += 1
                continue
            if fa * fb <= 0 or min(abs(fa), abs(fb)) <= policy.root.tolerance:
                slope = min(abs(jbox[0][0].lower), abs(jbox[0][0].upper))

                def scalar_error(u: NDArray[np.float64], slope: float = slope) -> float:
                    return float(abs(value(u)[0]) / slope)

                accept(
                    a, b, "monotone_scalar_branch", scalar_error
                )
                continue
        middle = (a + b) / 2
        jm = np.asarray(jacobian(middle * scales), dtype=float) * scales[None, :] / rows[:, None]
        if jm.shape != (n, n) or not np.all(np.isfinite(jm)):
            raise ValueError("root Jacobian must be a finite square matrix")
        try:
            singular_values = np.linalg.svd(jm, compute_uv=False)
            if singular_values[-1] <= 1e-12 * max(1.0, singular_values[0]):
                raise ResidualNumericalError("numerically_rank_deficient_jacobian")
            inverse = linear_solve(jm, np.eye(n))
        except ResidualNumericalError:
            inverse = None
        if inverse is not None:
            center = middle - inverse @ value(middle)
            matrix = tuple(
                tuple(
                    Interval(float(i == j), float(i == j))
                    - sum((inverse[i, k] * jbox[k][j] for k in range(n)), Interval(0, 0))
                    for j in range(n)
                )
                for i in range(n)
            )
            contraction = max(sum(max(abs(t.lower), abs(t.upper)) for t in r) for r in matrix)
            radii = tuple(
                Interval(float(low - m), float(high - m))
                for low, high, m in zip(a, b, middle, strict=True)
            )
            kbox = tuple(
                Interval(nextafter(float(c), -inf), nextafter(float(c), inf))
                + sum((m * r for m, r in zip(line, radii, strict=True)), Interval(0, 0))
                for c, line in zip(center, matrix, strict=True)
            )
            ka = np.maximum(a, [k.lower for k in kbox])
            kb = np.minimum(b, [k.upper for k in kbox])
            if np.any(ka > kb):
                excluded += 1
                continue
            inside = all(a[i] < k.lower and k.upper < b[i] for i, k in enumerate(kbox))
            if inside and contraction < 1:

                def contraction_error(
                    u: NDArray[np.float64],
                    c: NDArray[np.float64] = inverse,
                    q: float = contraction,
                ) -> float:
                    return float(np.linalg.norm(c @ value(u), np.inf) / (1 - q))

                accept(
                    a,
                    b,
                    "krawczyk_interior_contraction",
                    contraction_error,
                )
                continue
            if (
                depth < policy.maximum_depth
                and np.max((kb - ka) / np.maximum(b - a, np.finfo(float).tiny)) < 0.75
            ):
                pending.append((ka, kb, depth + 1))
                contractions += 1
                continue
        else:
            # A numerically constant singular Jacobian cannot certify a root or
            # uniqueness. Retain the entire region instead of exponential work.
            variation = max(t.upper - t.lower for row in jbox for t in row)
            magnitude = max(1.0, max(abs(t.lower) + abs(t.upper) for row in jbox for t in row))
            if variation <= 1e-14 * magnitude:
                remaining(a, b, "rank_deficient_constant_jacobian")
                continue
        if depth >= policy.maximum_depth or np.max(b - a) <= policy.coordinate_tolerance:
            remaining(a, b, "isolation_resolution_or_depth_limit")
            continue
        dimension = int(np.argmax(b - a))
        cut = float(middle[dimension])
        if cut in (a[dimension], b[dimension]):
            remaining(a, b, "floating_point_subdivision_limit")
            continue
        left, right = b.copy(), a.copy()
        left[dimension] = right[dimension] = cut
        pending.extend(((right, b, depth + 1), (a, left, depth + 1)))
    for a, b, _ in pending:
        remaining(a, b, "isolation_box_budget_exhausted")
    roots.sort(key=lambda r: r["value"])
    return IsolatedRoots(
        tuple(roots),
        tuple(unresolved),
        {
            "visited_boxes": count,
            "excluded_boxes": excluded,
            "contractions": contractions,
            "maximum_boxes": policy.maximum_boxes,
            "maximum_depth": policy.maximum_depth,
            "coordinate_tolerance_in_declared_scales": policy.coordinate_tolerance,
            "point_root_policy": {
                "tolerance": policy.root.tolerance,
                "maximum_iterations": policy.root.maximum_iterations,
            },
            "scope": "outward floating-point enclosures on declared bounds; not exact-real proof",
        },
    )
