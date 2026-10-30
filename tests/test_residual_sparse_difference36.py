"""Pure mathematical tests: no materials, transport or physical solve."""

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csc

from chemworld.physchem.residual_sparse_difference import (
    DERIVATIVE_MODE,
    sparse_finite_jacobian,
    validate_column_sparsity,
)
from chemworld.physchem.residual_system import ResidualDomainError, finite_jacobian


def test_shared_evaluator_uses_pattern_only_for_numerical_state_and_validates_dimensions():
    from chemworld.physchem.residual_system import (
        ResidualDerivatives,
        ResidualEquation,
        ResidualEvaluator,
        ResidualSystem,
        ResidualVariable,
    )

    variables = tuple(ResidualVariable(f"y{i}", "1") for i in range(2))
    equations = tuple(ResidualEquation(f"f{i}", "1") for i in range(2))

    def residual(t, y, rate, p):
        return [y[0] ** 2 + 2 * rate[0] + p[0], y[1] ** 2 + 3 * rate[1] - t]

    def system(pattern):
        return ResidualSystem(
            "local-math",
            variables,
            equations,
            residual,
            ResidualDerivatives("finite_difference", state_sparsity=pattern),
            parameters=(ResidualVariable("p", "1", role="parameter"),),
            parameter_values=(0.5,),
        )

    for pattern in (((0,),), ((2,), (1,))):
        with pytest.raises(ValueError, match="sparsity"):
            system(pattern)
    evaluator = ResidualEvaluator(system(((0,), (1,))))
    y, rate = np.array([0.3, 0.7]), np.zeros(2)
    state = evaluator.derivative("state", 0.2, y, rate)
    assert isspmatrix_csc(state)
    np.testing.assert_allclose(state.toarray(), np.diag(2 * y), atol=1e-10)
    np.testing.assert_allclose(
        evaluator.derivative("state", 0.2, y, rate, columns=[1, 0]).toarray(),
        np.diag(2 * y)[:, [1, 0]],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        evaluator.derivative("rate", 0.2, y, rate), np.diag([2.0, 3.0]), atol=1e-10
    )
    np.testing.assert_allclose(
        evaluator.derivative("parameter", 0.2, y, rate), [[1.0], [0.0]], atol=1e-10
    )
    np.testing.assert_allclose(evaluator.derivative("time", 0.2, y, rate), [0.0, -1.0], atol=1e-10)
    assert evaluator.diagnostics()["derivative_modes"]["state"] == "finite_difference"


def test_shared_root_reports_nonfinite_intermediate_products_without_accepting_a_root():
    from chemworld.physchem.residual_system import ResidualVariable, solve_root

    with np.errstate(over="ignore", invalid="ignore"):
        result = solve_root(
            lambda y: np.array([1e200 * (y[0] - 1)]),
            [0.0],
            jacobian=lambda y: np.array([[1e200]]),
            variables=(ResidualVariable("huge", "1"),),
            residual_scales=[1.0],
        )
    assert not result.success and result.value is None
    assert result.failure_reason == "nonfinite_scaled_gradient_or_merit"


class Counted:
    def __init__(self, function):
        self.function = function
        self.calls = 0

    def __call__(self, point):
        self.calls += 1
        return self.function(point)


def local_function(point):
    result = np.sin(point) + 0.2 * point**3
    result[1:] += 0.3 * point[:-1] ** 2
    result[:-1] += np.exp(0.1 * point[1:])
    return result


def local_analytic(point):
    result = np.diag(np.cos(point) + 0.6 * point**2)
    for row in range(len(point)):
        if row:
            result[row, row - 1] = 0.6 * point[row - 1]
        if row + 1 < len(point):
            result[row, row + 1] = 0.1 * np.exp(0.1 * point[row + 1])
    return result


def local_pattern(size):
    return tuple(tuple(range(max(0, j - 1), min(size, j + 2))) for j in range(size))


def test_nonlinear_local_tridiagonal_matches_independent_analytic_and_dense():
    point = np.linspace(-0.8, 0.9, 60)
    initial = point.copy()
    scales = np.ones_like(point)
    colored, dense = Counted(local_function), Counted(local_function)
    actual = sparse_finite_jacobian(colored, point, scales, sparsity=local_pattern(len(point)))
    expected = finite_jacobian(dense, point, scales)
    assert isspmatrix_csc(actual) and actual.has_sorted_indices
    np.testing.assert_array_equal(actual.toarray(), expected)
    np.testing.assert_allclose(actual.toarray(), local_analytic(point), atol=5e-10, rtol=2e-9)
    np.testing.assert_array_equal(point, initial)
    assert colored.calls == 7 and dense.calls == 121
    assert colored.calls * 10 < dense.calls
    assert DERIVATIVE_MODE == "finite_difference"


@pytest.mark.parametrize("selected", [[17, 2, 19, 1], np.array([7, 0, 9]), (3,), ()])
def test_subset_preserves_requested_original_column_order(selected):
    point = np.linspace(0.1, 0.8, 20)
    actual = sparse_finite_jacobian(
        local_function, point, np.ones(20), sparsity=local_pattern(20), indices=selected
    )
    dense = finite_jacobian(local_function, point, np.ones(20), indices=selected)
    assert actual.shape == (20, len(selected))
    np.testing.assert_array_equal(actual.toarray(), dense)


def test_mixed_lower_upper_and_central_offsets_match_dense_bitwise():
    point = np.array([0.0, 1.0, 0.3, -2.0])
    lower = np.array([0.0, -np.inf, -1.0, -2.0])
    upper = np.array([np.inf, 1.0, 1.0, np.inf])

    def function(y):
        return np.sin(y) + y**3

    colored = Counted(function)
    actual = sparse_finite_jacobian(
        colored,
        point,
        np.ones(4),
        sparsity=((0,), (1,), (2,), (3,)),
        lower=lower,
        upper=upper,
    )
    expected = finite_jacobian(function, point, np.ones(4), lower=lower, upper=upper)
    np.testing.assert_array_equal(actual.toarray(), expected)
    assert colored.calls == 3


def test_actual_representable_offsets_and_asymmetric_near_bound_match_dense():
    point = np.array([1e-8, 1.0 - 1e-8, 0.3])
    lower, upper = np.zeros(3), np.ones(3)

    def function(y):
        return np.exp(y)

    actual = sparse_finite_jacobian(
        function,
        point,
        np.ones(3),
        sparsity=((0,), (1,), (2,)),
        lower=lower,
        upper=upper,
    )
    expected = finite_jacobian(function, point, np.ones(3), lower=lower, upper=upper)
    np.testing.assert_array_equal(actual.toarray(), expected)


def test_combined_domain_rejection_falls_back_entire_color_to_original_columns():
    point = np.array([0.2, 0.4, 0.6, 0.8])
    rejected = []

    def function(y):
        if np.count_nonzero(y != point) > 1:
            rejected.append(y.copy())
            raise ResidualDomainError("combined trial deliberately unavailable")
        return y**2 + 2 * y

    colored = Counted(function)
    actual = sparse_finite_jacobian(colored, point, np.ones(4), sparsity=((0,), (1,), (2,), (3,)))
    expected = finite_jacobian(function, point, np.ones(4))
    np.testing.assert_array_equal(actual.toarray(), expected)
    assert len(rejected) == 1
    assert colored.calls == 11  # original baseline + rejected group + dense fallback


def test_second_combined_sample_rejection_discards_the_first_sample_too():
    point = np.array([0.1, 0.3, 0.5])

    def function(y):
        if np.count_nonzero(y > point) > 1:
            raise ResidualDomainError("positive combined sample unavailable")
        return y**3

    colored = Counted(function)
    actual = sparse_finite_jacobian(colored, point, np.ones(3), sparsity=((0,), (1,), (2,)))
    np.testing.assert_array_equal(actual.toarray(), finite_jacobian(function, point, np.ones(3)))
    assert colored.calls == 10


def test_domain_rejection_preserves_dense_alternate_boundary_direction():
    point = np.array([1e-8, 0.3])

    def function(y):
        if y[0] > point[0]:
            raise ResidualDomainError("roomier positive direction unavailable")
        return y**2

    kwargs = {"lower": np.zeros(2), "upper": np.ones(2)}
    actual = sparse_finite_jacobian(function, point, np.ones(2), sparsity=((0,), (1,)), **kwargs)
    expected = finite_jacobian(function, point, np.ones(2), **kwargs)
    np.testing.assert_array_equal(actual.toarray(), expected)


def test_both_scalar_directions_rejected_remain_failure_not_zero_column():
    point = np.zeros(3)

    def function(y):
        if y[0] != 0:
            raise ResidualDomainError("first coordinate cannot be perturbed")
        return y**2

    for run in (
        lambda: sparse_finite_jacobian(function, point, np.ones(3), sparsity=((0,), (1,), (2,))),
        lambda: finite_jacobian(function, point, np.ones(3)),
    ):
        with pytest.raises(ResidualDomainError, match="no admissible finite-difference stencil"):
            run()


def test_fixed_bound_with_no_stencil_remains_rejected_even_for_declared_zero_column():
    with pytest.raises(ResidualDomainError, match="no admissible finite-difference stencil"):
        sparse_finite_jacobian(
            lambda y: np.zeros(1), [0.0], [1.0], sparsity=((),), lower=[0.0], upper=[0.0]
        )


def test_non_domain_exception_from_combined_sample_is_not_swallowed():
    original = RuntimeError("original callback exception")

    def function(y):
        if np.count_nonzero(y) > 1:
            raise original
        return y**2

    counted = Counted(function)
    with pytest.raises(RuntimeError) as caught:
        sparse_finite_jacobian(counted, [0.0, 0.0], [1.0, 1.0], sparsity=((0,), (1,)))
    assert caught.value is original
    assert counted.calls == 2


def test_rectangular_residual_and_exact_structural_zeros_have_canonical_csc():
    def function(y):
        return np.array([y[0] ** 2, y[1] ** 3, y[0] + y[2], 7.0])

    point = np.array([0.0, 0.2, 0.4, 0.6])
    pattern = ((2, 0), (1,), (2,), ())
    actual = sparse_finite_jacobian(function, point, np.ones(4), sparsity=pattern)
    np.testing.assert_array_equal(actual.toarray(), finite_jacobian(function, point, np.ones(4)))
    assert actual.shape == (4, 4) and actual.has_canonical_format
    assert actual[:, 3].nnz == 0 and actual[3, :].nnz == 0
    assert actual[0, 0] == 0.0


@pytest.mark.parametrize(
    "pattern",
    [
        None,
        "bad",
        ((0,),),
        ((0,), (1,), (2,)),
        ([0], (1,)),
        ((True,), (1,)),
        ((np.bool_(False),), (1,)),
        ((0.0,), (1,)),
        ((-1,), (1,)),
        ((0, 0), (1,)),
        ((2,), (1,)),
        ((0,), (2,)),
        (("0",), (1,)),
    ],
)
def test_malformed_patterns_are_rejected(pattern):
    with pytest.raises(ValueError):
        sparse_finite_jacobian(lambda y: y**2, [0.2, 0.4], [1.0, 1.0], sparsity=pattern)


def test_pattern_numpy_integer_rows_are_valid_but_booleans_are_not():
    assert validate_column_sparsity([(np.int64(1), np.int32(0)), ()], 2, 2) == ((0, 1), ())


@pytest.mark.parametrize("selected", [[True], [0.0], [-1], [2], [0, 0], [[0]], "0", {0}])
def test_invalid_requested_columns_reject(selected):
    with pytest.raises(ValueError):
        sparse_finite_jacobian(
            lambda y: y**2, [0.2, 0.4], [1.0, 1.0], sparsity=((0,), (1,)), indices=selected
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"point": [[0.2, 0.4]]},
        {"point": [np.nan, 0.4]},
        {"scales": [0.0, 1.0]},
        {"scales": [True, False]},
        {"scales": [1.0]},
        {"relative_step": 0.0},
        {"relative_step": np.inf},
        {"relative_step": True},
        {"lower": [0.0]},
        {"lower": [0.0, np.nan]},
        {"lower": [1.0, 0.0], "upper": [0.0, 1.0]},
        {"upper": [0.1, 1.0]},
    ],
)
def test_invalid_numeric_contract_rejects(changes):
    arguments = {"point": [0.2, 0.4], "scales": [1.0, 1.0], "sparsity": ((0,), (1,))}
    arguments.update(changes)
    with pytest.raises(ValueError):
        sparse_finite_jacobian(lambda y: y**2, **arguments)


def test_nonfinite_function_and_dimension_change_are_not_accepted():
    with pytest.raises(ResidualDomainError, match="nonfinite"):
        sparse_finite_jacobian(lambda y: [np.inf], [0.2], [1.0], sparsity=((0,),))

    def changing(y):
        return np.ones(1 if y[0] == 0.2 else 2)

    with pytest.raises(ValueError, match="dimension"):
        sparse_finite_jacobian(changing, [0.2], [1.0], sparsity=((0,),))
