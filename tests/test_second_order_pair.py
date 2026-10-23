"""Independent analytic checks for the fused two-coordinate chain rules."""

from math import exp, log

import pytest

from chemworld.physchem.caloric_math import (
    derivative_depth,
    exponential,
    exponential_minus_one,
    logarithm,
    nonnegative_power,
    scalar_value,
)
from chemworld.physchem.second_order_pair import SecondOrderPair


def components(value):
    return value.value, value.dx, value.dy, value.dxx, value.dxy, value.dyy


@pytest.mark.parametrize("x,y", [(2.0, 3.0), (0.01, 0.2), (1.5, -0.5)])
def test_mixed_chain_rule_against_closed_derivatives(x, y):
    a, b = SecondOrderPair(x, dx=1), SecondOrderPair(y, dy=1)
    result = exponential(a * b) + logarithm(a + b) + a / b
    e, s = exp(x * y), x + y
    assert components(result) == pytest.approx(
        (
            e + log(s) + x / y,
            e * y + 1 / s + 1 / y,
            e * x + 1 / s - x / y**2,
            e * y * y - 1 / s**2,
            e * (1 + x * y) - 1 / s**2 - 1 / y**2,
            e * x * x - 1 / s**2 + 2 * x / y**3,
        ),
        rel=2e-13,
        abs=2e-13,
    )


def test_polynomial_at_zero_and_constant_operations():
    x, y = SecondOrderPair(0, dx=1), SecondOrderPair(3, dy=1)
    result = 2 * x * x * y + 4 - y + 2 / y
    assert components(result) == pytest.approx((4 - 3 + 2 / 3, 0, -1 - 2 / 9, 12, 0, 4 / 27))
    assert components(x**0) == (1, 0, 0, 0, 0, 0)
    assert components(nonnegative_power(x, 2)) == (0, 0, 0, 2, 0, 0)
    assert scalar_value(y) == 3
    assert derivative_depth(y) == 2
    with pytest.raises(ValueError, match="does not exist"):
        nonnegative_power(x, 1.5)
    with pytest.raises(ValueError, match="undefined"):
        nonnegative_power(x, y)


def test_small_expm1_and_variable_power():
    x, y = SecondOrderPair(1e-20, dx=1), SecondOrderPair(2, dy=1)
    assert exponential_minus_one(x).value == pytest.approx(1e-20, rel=1e-15, abs=0)
    result = nonnegative_power(SecondOrderPair(3, dx=1), y)
    assert components(result) == pytest.approx(
        (9, 6, 9 * log(3), 2, 3 * (1 + 2 * log(3)), 9 * log(3) ** 2)
    )
