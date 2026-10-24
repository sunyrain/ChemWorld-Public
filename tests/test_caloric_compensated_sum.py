from chemworld.physchem.caloric_math import compensated_sum, partial_derivative
from chemworld.physchem.second_order_pair import SecondOrderPair


def test_large_cancelling_values_retain_small_finite_energy():
    assert compensated_sum((1e16, 1., -1e16)) == 1.


def test_cancellation_preserves_each_gradient_and_hessian_entry():
    small = SecondOrderPair(1., 2., 3., 4., 5., 6.)
    large = SecondOrderPair(*(1e16 for _ in range(6)))
    assert compensated_sum((large, small, -large)) == small


def test_nested_third_derivative_survives_cancelling_energy_terms():
    def energy(values):
        x = values["x"]
        return compensated_sum((1e16*x**3, x**3, -1e16*x**3))

    assert partial_derivative(energy, {"x": 1.}, ("x", "x", "x")) == 6.
