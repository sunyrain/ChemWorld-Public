"""Precision is opt-in scalar arithmetic, retaining the original shared jets."""

import pytest

from chemworld.physchem.caloric_math import (
    compensated_sum,
    derivative_values,
    exponential,
    exponential_minus_one,
    logarithm,
)
from chemworld.physchem.precise_scalar import PreciseScalar
from chemworld.physchem.second_order_pair import SecondOrderPair


def test_shared_pair_retains_cancelled_primal_and_all_derivatives():
    t = SecondOrderPair(PreciseScalar(1.), dx=1)
    r = SecondOrderPair(PreciseScalar(2.), dy=1)
    result = 1e30*t*r+t*t*r+(-1e30)*t*r
    assert [float(getattr(result, key)) for key in ("value", "dx", "dy", "dxx", "dxy", "dyy")] == [
        2., 4., 1., 4., 2., 0.]
    values = [PreciseScalar(1e30), PreciseScalar(1), PreciseScalar(-1e30)]
    assert float(compensated_sum(values)) == 1


def test_precise_transcendentals_keep_nested_chain_rules_and_small_expm1():
    result = derivative_values(lambda t: logarithm(exponential(t)), PreciseScalar(2), order=3)
    assert [float(v) for v in result] == pytest.approx([2., 1., 0., 0.], abs=1e-45)
    tiny = PreciseScalar(1e-30)
    assert float(exponential_minus_one(tiny)/tiny) == pytest.approx(1, abs=1e-15)
    r = SecondOrderPair(PreciseScalar(2), dx=1)
    expression = logarithm(exponential_minus_one(r)+1)
    assert [float(expression.value), float(expression.dx), float(expression.dxx)] == pytest.approx(
        [2, 1, 0], abs=1e-45)


def test_binary_source_values_and_hash_contract_are_preserved():
    x = PreciseScalar(.1)
    assert x == .1
    assert hash(x) == hash(.1)
    assert float(1/x) == 10
    assert float(2-x) == 1.9
    assert x < .2 and x > .01
    assert float(abs(-x)) == .1
    with pytest.raises(TypeError):
        PreciseScalar(object())
