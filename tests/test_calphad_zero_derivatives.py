"""Absent CEF constituents are constant zero; active zero-boundary jets diverge."""

import numpy as np
import pytest

from chemworld.physchem.caloric_math import FirstDerivative
from chemworld.physchem.calphad import _xlogx
from chemworld.physchem.second_order_pair import SecondOrderPair


@pytest.mark.parametrize("value", [
    0.0,
    FirstDerivative(0.0),
    FirstDerivative(0.0, np.zeros(3)),
    FirstDerivative(FirstDerivative(0.0, np.zeros(3)), FirstDerivative(0.0, np.zeros(3))),
    SecondOrderPair(0.0),
])
def test_constant_absent_constituent_has_zero_mixing_energy(value):
    assert _xlogx(value) == 0.0


@pytest.mark.parametrize("value", [
    FirstDerivative(0.0, 1.0),
    FirstDerivative(0.0, np.array([0.0, 1.0, 0.0])),
    FirstDerivative(FirstDerivative(0.0, 1.0), 0.0),
    FirstDerivative(FirstDerivative(0.0, 0.0), FirstDerivative(0.0, 1.0)),
    FirstDerivative(0.0, np.array([FirstDerivative(0.0, 1.0)], dtype=object)),
    SecondOrderPair(0.0, dxy=1.0),
])
def test_nonconstant_zero_site_fraction_keeps_divergent_derivative_refusal(value):
    with pytest.raises(ValueError, match="diverges at zero"):
        _xlogx(value)


def test_positive_site_vector_gradient_retains_analytic_expression():
    directions = np.array([1.0, -2.0, 0.0])
    result = _xlogx(FirstDerivative(0.2, directions))
    assert result.value == pytest.approx(0.2 * np.log(0.2))
    np.testing.assert_allclose(result.slope, (1 + np.log(0.2)) * directions)
