"""Shared Gibbs packed seeds against explicit multivariate analytic derivatives."""

import numpy as np
import pytest

from chemworld.physchem.caloric_math import exponential, logarithm
from chemworld.physchem.calphad_response import analytic_response


def test_packed_nested_seeds_preserve_cross_derivatives_and_disconnected_outputs():
    calls = []

    def function(z):
        calls.append(1)
        x, y, w = z
        return (x*x*y+exponential(w)*logarithm(y)+x/y, 42., w)

    x, y, w = 1.2, 2.3, -.4
    values, jacobian, hessian = analytic_response(function, (x, y, w))
    e = np.exp(w)
    assert len(calls) == 1
    assert values == pytest.approx((x*x*y+e*np.log(y)+x/y, 42, w))
    assert jacobian[0] == pytest.approx((2*x*y+1/y, x*x+e/y-x/y**2, e*np.log(y)))
    assert hessian[0] == pytest.approx(np.array([
        [2*y, 2*x-1/y**2, 0],
        [2*x-1/y**2, -e/y**2+2*x/y**3, e/y],
        [0, e/y, e*np.log(y)],
    ]))
    assert np.array_equal(jacobian[1], np.zeros(3))
    assert np.array_equal(jacobian[2], [0, 0, 1])
    assert np.array_equal(hessian[1:], np.zeros((2, 3, 3)))
