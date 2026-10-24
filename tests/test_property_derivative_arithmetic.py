"""Higher pressure derivatives reuse native chain rules, with analytic identities."""

from math import exp, log

import pytest

from chemworld.physchem.caloric_math import (
    derivative_values,
    exponential,
    logarithm,
    positive_part,
)


@pytest.mark.parametrize("order", range(4))
def test_nested_chain_rules_recover_polynomial_log_and_exponential_derivatives(order):
    T = 2.5
    cases = (
        (lambda x: x**4, (T**4, 4 * T**3, 12 * T * T, 24 * T)),
        (lambda x: logarithm(x), (log(T), 1 / T, -1 / T**2, 2 / T**3)),
        (
            lambda x: exponential(2 * x),
            (exp(2 * T), 2 * exp(2 * T), 4 * exp(2 * T), 8 * exp(2 * T)),
        ),
        (lambda x: 1 / x, (1 / T, -1 / T**2, 2 / T**3, -6 / T**4)),
        (lambda x: x**0.5, (T**0.5, 0.5 * T**-0.5, -0.25 * T**-1.5, 0.375 * T**-2.5)),
        (lambda x: 3, (3, 0, 0, 0)),
    )
    for function, expected in cases:
        assert derivative_values(function, T, order=order) == pytest.approx(expected[: order + 1])


def test_nested_composition_and_explicit_piecewise_branch():
    # exp(log(x)) and the inactive TRC-style positive part exercise composition.
    assert derivative_values(lambda x: exponential(logarithm(x)), 3, order=3) == pytest.approx(
        (3, 1, 0, 0), abs=1e-14
    )
    assert derivative_values(lambda x: positive_part(x - 4) ** 4, 3, order=3) == (0, 0, 0, 0)
    assert derivative_values(lambda x: positive_part(x - 4) ** 4, 5, order=3) == (1, 4, 12, 24)
    for order in (-1, 5, True, 1.0):
        with pytest.raises(ValueError):
            derivative_values(lambda x: x, 2, order=order)
