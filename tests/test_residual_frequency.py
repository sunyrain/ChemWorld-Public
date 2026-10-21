"""Functional frequency-response contracts; not a model qualification block."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from chemworld.physchem.residual_frequency import residual_frequency_response
from chemworld.physchem.residual_solver import mass_matrix_system
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEquation,
    ResidualVariable,
    linear_solve,
)


def rc(mode="analytic", scale=1):
    # R=3 ohm, C=2 F and R0=0.4 ohm; the algebraic row is the output.
    return mass_matrix_system(
        "rc",
        (
            ResidualVariable("capacitor", "V", scale, lower=-5, upper=5),
            ResidualVariable("output", "V", 2 / scale, "algebraic"),
        ),
        (ResidualEquation("rc", "A", 1e-3 * scale), ResidualEquation("terminal", "V", 1e3 / scale)),
        sparse.diags([2.0, 0.0]),
        lambda t, y, p: np.array([p[0] - y[0] / 3, y[1] - y[0] - 0.4 * p[0]]),
        derivatives=mode,
        rhs_jacobian=None
        if mode == "finite_difference"
        else lambda t, y, p: sparse.csc_matrix([[-1 / 3, 0], [-1, 1]]),
        rhs_parameter_jacobian=None
        if mode == "finite_difference"
        else lambda t, y, p: np.array([[1.0], [-0.4]]),
        parameters=(ResidualVariable("input", "A", 0.2 * scale, "parameter"),),
        parameter_values=(0.0,),
    )


@pytest.mark.parametrize("mode", ["analytic", "hybrid", "finite_difference"])
@pytest.mark.parametrize("scale", [1e-4, 1, 1e4])
def test_rc_with_direct_feedthrough_and_scaling(mode, scale):
    f = np.array([10, 0.001, 0.1, 0.001])
    state = np.zeros(2)
    result = residual_frequency_response(rc(mode, scale), state, f)
    expected = 3 / (1 + 2j * np.pi * f * 6)
    assert result.success
    assert [p.frequency_Hz for p in result.points] == f.tolist()
    np.testing.assert_allclose([p.response[0, 0] for p in result.points], expected, rtol=1e-10)
    np.testing.assert_allclose(
        [p.response[1, 0] for p in result.points], expected + 0.4, rtol=1e-10
    )
    assert all(p.relative_linear_residual < 1e-10 for p in result.points)
    assert not result.points[0].response.flags.writeable
    np.testing.assert_array_equal(state, 0)
    assert result.diagnostics["physical_time_advanced_s"] == 0
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("sparse_rhs", [False, True])
@pytest.mark.parametrize("complex_matrix", [False, True])
def test_shared_sparse_solve_never_discards_imaginary_parts(sparse_rhs, complex_matrix):
    matrix = sparse.csc_matrix([[2, 1], [1, 3]], dtype=complex if complex_matrix else float)
    if complex_matrix:
        matrix = matrix + 1j * sparse.eye(2)
    forcing = np.array([[1 + 2j, 3j], [2 - 1j, 1]])
    original = forcing.copy()
    answer = linear_solve(matrix, sparse.csc_matrix(forcing) if sparse_rhs else forcing)
    if sparse_rhs:
        answer = answer.toarray()
    np.testing.assert_allclose(matrix @ answer, forcing, rtol=1e-12)
    np.testing.assert_array_equal(forcing, original)


def test_nondiagonal_mass_multiple_inputs_and_singular_frequency_retained():
    # An undamped oscillator has a pole at omega=1. A left row transform
    # exercises a nondiagonal mass matrix without changing the response.
    transform = np.array([[2.0, 1.0], [1.0, 3.0]])
    a = np.array([[0.0, -1.0], [1.0, 0.0]])
    system = mass_matrix_system(
        "oscillator",
        tuple(ResidualVariable(k, "1") for k in ("x", "y")),
        tuple(ResidualEquation(k, "1/s") for k in ("x", "y")),
        transform,
        lambda t, y, p: transform @ (a @ y + p),
        rhs_jacobian=lambda t, y, p: transform @ a,
        rhs_parameter_jacobian=lambda t, y, p: transform,
        parameters=tuple(ResidualVariable(k, "1/s", role="parameter") for k in ("u", "v")),
        parameter_values=(0, 0),
    )
    result = residual_frequency_response(system, [0, 0], np.array([0.5, 1, 2]) / (2 * np.pi))
    assert [p.success for p in result.points] == [True, False, True]
    assert not result.success and result.points[1].failure_reason == "singular_jacobian"
    for index, omega in ((0, 0.5), (2, 2)):
        expected = np.linalg.inv(1j * omega * np.eye(2) - a)
        np.testing.assert_allclose(result.points[index].response, expected, atol=1e-12)
    assert result.to_dict()["points"][1]["response_real"] is None
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("frequencies", [[], [0], [-1], [np.inf], [np.nan], [[1]], [1e308]])
def test_invalid_frequency_rejects(frequencies):
    with pytest.raises(ValueError, match="finite positive Hz"):
        residual_frequency_response(rc(), [0, 0], frequencies)


def test_nonstationary_out_of_bounds_missing_derivative_and_input_reject():
    system = rc()
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        residual_frequency_response(system, [1, 1], [1])
    with pytest.raises(ResidualDomainError, match="outside declared bounds"):
        residual_frequency_response(system, [6, 6], [1])
    with pytest.raises(ValueError, match="analytic parameter derivative is unavailable"):
        residual_frequency_response(
            replace(system, derivatives=replace(system.derivatives, parameter=None)), [0, 0], [1]
        )
    with pytest.raises(ValueError, match="input parameters"):
        residual_frequency_response(
            replace(system, parameters=(), parameter_values=()), [0, 0], [1]
        )


def test_programming_errors_are_not_reported_as_scientific_frequency_failures():
    def broken(*args):
        raise KeyError("bad callback")

    system = replace(rc(), derivatives=replace(rc().derivatives, state=broken))
    with pytest.raises(KeyError, match="bad callback"):
        residual_frequency_response(system, [0, 0], [1])
