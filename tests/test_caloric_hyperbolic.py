from math import cosh, tanh

import pytest

from chemworld.physchem.caloric_math import (
    Interval,
    derivative_values,
    hyperbolic_tangent,
)
from chemworld.physchem.second_order_pair import SecondOrderPair


@pytest.mark.parametrize("x", [0., .7, -.7, 20., -20., 400., -400.])
def test_tanh_full_chain_rule_and_saturated_derivatives(x):
    first = 0. if abs(x) > 350 else 1/cosh(x)**2
    expected = (tanh(x), first, -2*tanh(x)*first, (6*tanh(x)**2-2)*first)
    assert derivative_values(hyperbolic_tangent, x, order=3) == pytest.approx(
        expected, rel=1e-14, abs=1e-30)
    pair = hyperbolic_tangent(SecondOrderPair(x, dx=2, dy=3))
    assert (pair.dx, pair.dy, pair.dxx, pair.dxy, pair.dyy) == pytest.approx(
        (2*first, 3*first, 4*expected[2], 6*expected[2], 9*expected[2]), rel=1e-14, abs=1e-30)


def test_tanh_interval_contains_monotone_endpoints():
    result = hyperbolic_tangent(Interval(-.3, .8))
    assert result.lower <= tanh(-.3) <= tanh(.8) <= result.upper
