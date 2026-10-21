"""Bounded root coverage, including branches that must remain unresolved."""

import numpy as np
import pytest

from chemworld.physchem.caloric_math import Interval
from chemworld.physchem.residual_system import ResidualVariable
from chemworld.physchem.root_isolation import (
    DEFAULT_ISOLATION_POLICY,
    IsolationPolicy,
    isolate_roots,
)


def scalar(function, derivative, enclosure, *, policy=DEFAULT_ISOLATION_POLICY):
    return isolate_roots(
        lambda x: [function(x[0])],
        lambda x: [[derivative(x[0])]],
        lambda a, b: enclosure(Interval(a[0], b[0])),
        variables=(ResidualVariable("x", "dimensionless", lower=-2, upper=2),),
        residual_scales=(1.0,),
        policy=policy,
    )


def test_multiple_and_empty_branches_are_covered_not_guessed():
    answer = scalar(lambda x: x * x - 1, lambda x: 2 * x, lambda x: ((x * x - 1,), ((2 * x,),)))
    assert answer.status == "multiple"
    assert [r["value"][0] for r in answer.roots] == pytest.approx([-1, 1])
    empty = scalar(lambda x: x * x + 1, lambda x: 2 * x, lambda x: ((x**2 + 1,), ((2 * x,),)))
    assert empty.status == "no_solution"


def test_tangent_and_work_limit_remain_unresolved():
    tangent = scalar(lambda x: x * x, lambda x: 2 * x, lambda x: ((x**2,), ((2 * x,),)))
    assert tangent.status == "unresolved"
    limited = scalar(
        lambda x: x * x - 1,
        lambda x: 2 * x,
        lambda x: ((x * x - 1,), ((2 * x,),)),
        policy=IsolationPolicy(maximum_boxes=1),
    )
    assert limited.status == "unresolved"
    assert limited.diagnostics["visited_boxes"] == 1


def test_scaled_coupled_system_and_rank_deficiency():
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    target = np.array([4.0, 7.0])

    def enclose(a, b):
        x, y = (Interval(low, high) for low, high in zip(a, b, strict=True))
        return (2 * x + y - 4, x + 3 * y - 7), (
            (Interval(2, 2), Interval(1, 1)),
            (Interval(1, 1), Interval(3, 3)),
        )

    variables = (
        ResidualVariable("x", "V", 10, lower=-10, upper=10),
        ResidualVariable("y", "V", 0.1, lower=-10, upper=10),
    )
    result = isolate_roots(
        lambda x: matrix @ x - target,
        lambda x: matrix,
        enclose,
        variables=variables,
        residual_scales=(5, 2),
    )
    assert result.status == "unique"
    assert result.roots[0]["value"] == pytest.approx([1.0, 2.0])
    singular = isolate_roots(
        lambda x: [sum(x) - 1, 2 * sum(x) - 2],
        lambda x: [[1.0, 1.0], [2.0, 2.0]],
        lambda a, b: (
            (Interval(sum(a) - 1, sum(b) - 1), Interval(2 * sum(a) - 2, 2 * sum(b) - 2)),
            ((Interval(1, 1), Interval(1, 1)), (Interval(2, 2), Interval(2, 2))),
        ),
        variables=variables,
        residual_scales=(1, 1),
    )
    assert singular.status == "unresolved"
    assert not singular.roots
    assert singular.unresolved[0]["reason"] == "rank_deficient_constant_jacobian"


@pytest.mark.parametrize(
    "changes",
    [{"maximum_boxes": 0}, {"maximum_depth": True}, {"coordinate_tolerance": float("nan")}],
)
def test_invalid_policy_rejects(changes):
    with pytest.raises(ValueError):
        IsolationPolicy(**changes)
