"""Functional mathematical contracts, separate from the fixed 21C science block."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionBoundary, DiffusionMesh
from chemworld.physchem.residual_solver import (
    ResidualEvent,
    ResidualSolvePolicy,
    consistent_initial_conditions,
    mass_matrix_system,
)
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualEvaluator,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_root,
    solve_steady_residual,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_residual


def quadratic_dae(mode="analytic", sparse_jac=False):
    def convert(values):
        return sparse.csc_matrix(values) if sparse_jac else np.asarray(values)

    providers = (
        ResidualDerivatives(
            "analytic",
            state=lambda t, y, v, p: convert([[p[0], 0], [-2 * y[0], 1]]),
            rate=lambda t, y, v, p: convert([[1, 0], [0, 0]]),
            parameter=lambda t, y, v, p: convert([[y[0]], [0]]),
            time=lambda t, y, v, p: np.zeros(2),
        )
        if mode == "analytic"
        else ResidualDerivatives(mode)
    )
    return ResidualSystem(
        "quadratic-constraint",
        (
            ResidualVariable("x", "mol", lower=0),
            ResidualVariable("z", "mol^2", role="algebraic", lower=0),
        ),
        (ResidualEquation("decay", "mol/s"), ResidualEquation("constraint", "mol^2")),
        lambda t, y, v, p: np.array([v[0] + p[0] * y[0], y[1] - y[0] ** 2]),
        providers,
        (ResidualVariable("k", "1/s", role="parameter", lower=0),),
        (0.7,),
    )


@pytest.mark.parametrize("mode", ["analytic", "finite_difference"])
def test_consistent_initial_conditions_hold_differential_coordinates_and_constraint_tangent(mode):
    system = quadratic_dae(mode)
    original = np.array([2.0, 0.2])
    result = consistent_initial_conditions(system, original, time_s=0.3)
    assert result.success, result
    assert result.point.state == pytest.approx([2, 4], abs=1e-10)
    assert result.point.rate == pytest.approx([-1.4, -5.6], abs=1e-8)
    assert original.tolist() == [2, 0.2]
    assert not result.point.state.flags.writeable and not result.point.rate.flags.writeable
    assert result.diagnostics["derivative_mode"] == mode
    assert result.diagnostics["maximum_scaled_residual"] <= 1e-12


def test_nonautonomous_constraint_rate_consumes_explicit_time_derivative():
    system = quadratic_dae()
    system = replace(
        system,
        residual=lambda t, y, v, p: np.array([v[0] + p[0] * y[0], y[1] - (y[0] + t) ** 2]),
        derivatives=replace(
            system.derivatives,
            state=lambda t, y, v, p: np.array([[p[0], 0], [-2 * (y[0] + t), 1]]),
            time=lambda t, y, v, p: np.array([0, -2 * (y[0] + t)]),
        ),
    )
    result = consistent_initial_conditions(system, [2, 1], time_s=0.3)
    assert result.success
    assert result.point.state == pytest.approx([2, 2.3**2])
    assert result.point.rate == pytest.approx([-1.4, 2 * 2.3 * (-1.4 + 1)])


@pytest.mark.parametrize("method", ["BDF", "Radau"])
@pytest.mark.parametrize(
    "mode,sparse_jac", [("analytic", False), ("analytic", True), ("finite_difference", False)]
)
def test_dae_event_uses_consistent_algebraic_state_and_retains_root_outside_output_grid(
    method,
    mode,
    sparse_jac,
):
    system = quadratic_dae(mode, sparse_jac)
    report = solve_residual(
        system,
        [1, 0.4],
        time_span_s=(0, 3),
        evaluation_times_s=[0.2, 1.5, 3],
        policy=ResidualSolvePolicy(
            ode=ODESolverPolicy("event-test", method=method, rtol=1e-9, atol=1e-12)
        ),
        events=[ResidualEvent("half-x", lambda t, y, p: y[1] - 0.25, direction=-1)],
    )
    assert report.success, report.diagnostics
    assert report.stopping_event == "half-x"
    assert report.final_point.time_s == pytest.approx(np.log(2) / 0.7, rel=2e-8)
    assert report.final_point.state == pytest.approx([0.5, 0.25], abs=1e-9)
    assert [p.time_s for p in report.points[:-1]] == [0, 0.2]
    assert report.events[0][1].state == pytest.approx(report.final_point.state)
    for point in report.points:
        assert point.state == pytest.approx(
            [np.exp(-0.7 * point.time_s), np.exp(-1.4 * point.time_s)], rel=3e-8
        )
        assert abs(point.rate[1] - 2 * point.state[0] * point.rate[0]) < 1e-8


@pytest.mark.parametrize("sparse_mass", [False, True])
@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_nondiagonal_mass_matrix_solves_implicit_ode_without_mutating_inputs(sparse_mass, method):
    mass = np.array([[2.0, 1.0], [1.0, 3.0]])
    j = -mass @ np.diag([1.0, 2.0])
    system = mass_matrix_system(
        "mass-ode",
        [ResidualVariable("a", "mol"), ResidualVariable("b", "mol")],
        [ResidualEquation("a", "mol/s"), ResidualEquation("b", "mol/s")],
        sparse.csc_matrix(mass) if sparse_mass else mass,
        lambda t, y, p: j @ y,
        rhs_jacobian=lambda t, y, p: sparse.csc_matrix(j),
    )
    mass[:] = 0
    initial = np.array([2.0, 3.0])
    report = solve_residual(
        system,
        initial,
        time_span_s=(0, 1),
        evaluation_times_s=[0, 1],
        policy=ResidualSolvePolicy(ode=ODESolverPolicy("mass", method=method)),
    )
    assert report.success, report.diagnostics
    assert initial.tolist() == [2, 3]
    assert report.final_point.state == pytest.approx([2 / np.e, 3 / np.e**2], rel=2e-7)


def test_nonlinear_rate_residual_uses_its_rate_jacobian():
    system = quadratic_dae()
    system = replace(
        system,
        residual=lambda t, y, v, p: np.array([v[0] ** 3 + v[0] + y[0], y[1] - y[0] ** 2]),
        derivatives=replace(
            system.derivatives,
            state=lambda t, y, v, p: np.array([[1, 0], [-2 * y[0], 1]]),
            rate=lambda t, y, v, p: np.diag([3 * v[0] ** 2 + 1, 0]),
        ),
    )
    result = consistent_initial_conditions(system, [2, 0.5])
    assert result.success
    assert result.point.state == pytest.approx([2, 4])
    assert result.point.rate == pytest.approx([-1, -4], abs=1e-10)


@pytest.mark.parametrize("mode", ["analytic", "finite_difference"])
def test_sparse_steady_root_and_parameter_response(mode):
    derivatives = (
        ResidualDerivatives(
            "analytic",
            state=lambda t, y, v, p: sparse.csc_matrix([[2 * y[0], 0], [-np.cos(y[0]), 1]]),
            rate=lambda t, y, v, p: sparse.csc_matrix((2, 2)),
            parameter=lambda t, y, v, p: sparse.csc_matrix([[-1], [0]]),
        )
        if mode == "analytic"
        else ResidualDerivatives(mode)
    )
    system = ResidualSystem(
        "bounded-root",
        (
            ResidualVariable("x", "1", role="algebraic", lower=0, upper=3),
            ResidualVariable("z", "1", role="algebraic", lower=-1, upper=1),
        ),
        (ResidualEquation("energy", "1"), ResidualEquation("closure", "1")),
        lambda t, y, v, p: [y[0] ** 2 - p[0], y[1] - np.sin(y[0])],
        derivatives,
        (ResidualVariable("p", "1", role="parameter"),),
        (4.0,),
    )
    result = solve_steady_residual(system, [0.2, 0], sensitivity=True)
    assert result.root.success, result.root
    assert result.root.value == pytest.approx([2, np.sin(2)], abs=1e-9)
    assert result.parameter_response[:, 0] == pytest.approx([0.25, np.cos(2) / 4], abs=1e-9)
    with pytest.raises(ValueError, match="pure algebraic"):
        solve_residual(system, result.root.value, time_span_s=(0, 1))


def test_missing_analytic_derivatives_are_explicit_and_never_silently_differenced():
    system = quadratic_dae()
    no_time = replace(system, derivatives=replace(system.derivatives, time=None))
    with pytest.raises(ValueError, match="time derivative is unavailable"):
        consistent_initial_conditions(no_time, [1, 1])
    no_parameter = replace(system, derivatives=replace(system.derivatives, parameter=None))
    with pytest.raises(ValueError, match="parameter derivative is unavailable"):
        ResidualEvaluator(no_parameter).derivative("parameter", 0, [1, 1], [-0.7, -1.4])


@pytest.mark.parametrize("kind", ["high-index", "algebraic-rate", "singular-mass"])
def test_unsupported_or_singular_dae_returns_failure_without_candidate(kind):
    system = quadratic_dae()
    if kind == "high-index":
        system = replace(
            system,
            residual=lambda t, y, v, p: [v[0], y[0] - 1],
            derivatives=replace(
                system.derivatives, state=lambda t, y, v, p: np.array([[0, 0], [1, 0]])
            ),
        )
    elif kind == "algebraic-rate":
        system = replace(
            system,
            residual=lambda t, y, v, p: [v[0], v[1] + y[1] - 1],
            derivatives=replace(
                system.derivatives,
                state=lambda t, y, v, p: np.diag([0, 1]),
                rate=lambda t, y, v, p: np.eye(2),
            ),
        )
    else:
        system = replace(
            system,
            residual=lambda t, y, v, p: [y[0] - 1, y[1] - 1],
            derivatives=replace(
                system.derivatives,
                state=lambda t, y, v, p: np.eye(2),
                rate=lambda t, y, v, p: np.zeros((2, 2)),
            ),
        )
    result = solve_residual(system, [1, 1], time_span_s=(0, 1))
    assert not result.success and result.final_point is None and result.points == ()
    assert (
        "singular" in result.failure_reason or "unsupported_algebraic_rate" in result.failure_reason
    )


def test_root_budget_bounds_and_nonfinite_trials_have_readable_failures():
    variable = [ResidualVariable("x", "1", lower=0, upper=2)]
    for initial, expected in (([-1], "initial_outside_bounds"), ([0], "iteration_limit")):
        result = solve_root(
            lambda y: np.exp(y) - 2,
            initial,
            jacobian=lambda y: np.diag(np.exp(y)),
            variables=variable,
            residual_scales=[1],
            policy=RootPolicy(maximum_iterations=1),
        )
        assert not result.success and result.value is None
        assert result.failure_reason == expected
    result = solve_root(
        lambda y: [np.nan], [1], jacobian=lambda y: [[1]], variables=variable, residual_scales=[1]
    )
    assert not result.success and "nonfinite" in result.failure_reason


def test_callback_coding_errors_propagate_and_domain_failures_are_diagnostic():
    def bad(t, y, v, p):
        raise KeyError("broken model")

    system = replace(quadratic_dae(), residual=bad)
    with pytest.raises(KeyError, match="broken model"):
        solve_residual(system, [1, 1], time_span_s=(0, 1))

    def outside(t, y, v, p):
        raise ResidualDomainError("declared model domain")

    result = solve_residual(replace(system, residual=outside), [1, 1], time_span_s=(0, 1))
    assert not result.success and "declared model domain" in result.failure_reason


def test_zero_duration_and_initial_terminal_event_do_not_integrate():
    for end, events in ((0, ()), (1, [ResidualEvent("initial", lambda t, y, p: y[1] - 1)])):
        report = solve_residual(quadratic_dae(), [1, 0.5], time_span_s=(0, end), events=events)
        assert report.success and len(report.points) == 1
        assert report.final_point.time_s == 0 and report.diagnostics["ode"] == {}
        assert report.final_point.state == pytest.approx([1, 1])


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
@pytest.mark.parametrize("basis", ["amount", "concentration"])
@pytest.mark.parametrize("flux", [0.0, 0.05])
def test_spatial_residual_uses_real_cell_volumes_and_preserves_reaction_and_boundary_balance(
    geometry,
    basis,
    flux,
):
    mesh = getattr(DiffusionMesh, geometry)(1.0, 6)
    diffusion = ConservativeDiffusion(mesh, 0.05, right=DiffusionBoundary("flux", flux))
    system = diffusion.residual_system(
        [f"cell:{i}" for i in range(6)],
        system_id="spatial-balance",
        basis=basis,
        reaction_rates=lambda t, n, p: -0.2 * n,
        reaction_jacobian=lambda t, n, p: -0.2 * sparse.eye(6),
    )
    concentration = 2 + mesh.centers_m
    initial = concentration * mesh.cell_volumes_m3 if basis == "amount" else concentration
    report = solve_residual(system, initial, time_span_s=(0, 0.4), evaluation_times_s=[0, 0.4])
    assert report.success, report.diagnostics
    amounts = report.final_point.state * (1 if basis == "amount" else mesh.cell_volumes_m3)
    n0 = sum(concentration * mesh.cell_volumes_m3)
    outlet = flux * mesh.face_areas_m2[-1]
    expected = (n0 + outlet / 0.2) * np.exp(-0.2 * 0.4) - outlet / 0.2
    assert sum(amounts) == pytest.approx(expected, rel=1e-8, abs=1e-10)
    assert not np.allclose(amounts, concentration * mesh.cell_volumes_m3 * np.exp(-0.2 * 0.4))


@pytest.mark.parametrize("basis", ["amount", "concentration"])
def test_concentration_boundaries_preserve_the_uniform_stationary_solution(basis):
    mesh = DiffusionMesh.planar(2, 5, area_m2=3)
    diffusion = ConservativeDiffusion(
        mesh, 0.1, DiffusionBoundary("concentration", 4), DiffusionBoundary("concentration", 4)
    )
    system = diffusion.residual_system(
        [f"c{i}" for i in range(5)], system_id="steady-boundary", basis=basis
    )
    initial = 4 * mesh.cell_volumes_m3 if basis == "amount" else np.full(5, 4.0)
    report = solve_residual(system, initial, time_span_s=(0, 1))
    assert report.success
    assert report.final_point.state == pytest.approx(initial, abs=1e-10)


def test_callback_arrays_cannot_mutate_input_or_internal_parameter_state():
    def mutate(t, y, v, p):
        p[0] = 99
        return y

    system = replace(quadratic_dae(), residual=mutate)
    with pytest.raises(ValueError, match="read-only"):
        consistent_initial_conditions(system, [1, 1])
    assert system.parameter_values == (0.7,)


def test_singular_steady_sensitivity_is_not_reported_as_unique_response():
    system = quadratic_dae()
    system = replace(
        system,
        residual=lambda t, y, v, p: [0, y[1] - 1],
        derivatives=replace(system.derivatives, state=lambda t, y, v, p: np.diag([0, 1])),
    )
    with pytest.raises(ResidualNumericalError, match="singular"):
        solve_steady_residual(system, [1, 1], sensitivity=True)


def test_root_at_bound_succeeds_but_constrained_nonroot_never_does():
    variables = [ResidualVariable("x", "1", lower=0, upper=2)]
    root = solve_root(
        lambda y: y**2 - 4,
        [0.5],
        jacobian=lambda y: np.diag(2 * y),
        variables=variables,
        residual_scales=[1],
    )
    assert root.success and root.value == pytest.approx([2])
    assert root.diagnostics["projected_newton_steps"] > 0
    failure = solve_root(
        lambda y: y - 3,
        [0.5],
        jacobian=lambda y: [[1]],
        variables=variables,
        residual_scales=[1],
    )
    assert not failure.success and failure.value is None
    assert failure.diagnostics["scaled_residual_norm"] == pytest.approx(1)


def test_projected_gradient_recovers_when_projected_newton_points_uphill():
    result = solve_root(
        lambda y: [y[0] + 10 * y[1] + 1, y[1] - y[0] ** 2 + 1],
        [0, 0],
        jacobian=lambda y: sparse.csc_matrix([[1, 10], [-2 * y[0], 1]]),
        variables=[
            ResidualVariable("x", "1", lower=-2, upper=2),
            ResidualVariable("z", "1", lower=0, upper=2),
        ],
        residual_scales=[1, 1],
    )
    assert result.success, result
    assert result.value == pytest.approx([-1, 0], abs=1e-10)
    assert result.diagnostics["gradient_steps"] > 0


@pytest.mark.parametrize("position", [0, 0.5, 1])
def test_numerical_parameter_derivative_respects_a_narrow_declared_domain(position):
    system = ResidualSystem(
        "narrow-parameter",
        (ResidualVariable("y", "mol"),),
        (ResidualEquation("balance", "mol/s"),),
        lambda t, y, v, p: [v[0] + y[0] + 2 * p[0]],
        ResidualDerivatives("finite_difference"),
        (ResidualVariable("k", "mol/s", role="parameter", lower=0, upper=1e-12),),
        (position * 1e-12,),
    )
    derivative = ResidualEvaluator(system).derivative("parameter", 0, [0], [0])
    assert derivative[:, 0] == pytest.approx([2], rel=1e-10)


@pytest.mark.parametrize("mode", ["analytic", "finite_difference"])
def test_large_and_small_native_units_share_the_scaled_solver(mode):
    scales = np.array([1e-20, 1e20])
    system = mass_matrix_system(
        "scaled-decay",
        [ResidualVariable("tiny", "mol", scales[0]), ResidualVariable("large", "mol", scales[1])],
        [
            ResidualEquation("tiny", "mol/s", scales[0]),
            ResidualEquation("large", "mol/s", scales[1]),
        ],
        sparse.eye(2),
        lambda t, y, p: -y,
        derivatives=mode,
        rhs_jacobian=(lambda t, y, p: -sparse.eye(2)) if mode == "analytic" else None,
    )
    report = solve_residual(system, scales, time_span_s=(0, 1), evaluation_times_s=[1])
    assert report.success, report
    assert report.final_point.state / scales == pytest.approx([1 / np.e] * 2, rel=2e-7)


def test_nonterminal_events_are_chronological_and_do_not_truncate_the_requested_end():
    report = solve_residual(
        quadratic_dae(),
        [1, 0.5],
        time_span_s=(0, 2),
        evaluation_times_s=[0.2],
        events=[
            ResidualEvent("late", lambda t, y, p: y[0] - 0.4, -1, False),
            ResidualEvent("early", lambda t, y, p: y[0] - 0.8, -1, False),
        ],
    )
    assert report.success and report.stopping_event is None
    assert [p.time_s for p in report.points] == [0, 0.2, 2]
    assert [name for name, _ in report.events] == ["early", "late"]
    assert [p.time_s for _, p in report.events] == pytest.approx(
        [-np.log(0.8) / 0.7, -np.log(0.4) / 0.7],
        rel=3e-7,
    )


def test_algebraic_coordinates_can_precede_differential_coordinates():
    system = quadratic_dae()
    order = [1, 0]
    swapped = replace(
        system,
        variables=tuple(system.variables[i] for i in order),
        equations=tuple(system.equations[i] for i in order),
        residual=lambda t, y, v, p: system.residual(t, y[order], v[order], p)[order],
        derivatives=replace(
            system.derivatives,
            state=lambda t, y, v, p: system.derivatives.state(t, y[order], v[order], p)[order][
                :, order
            ],
            rate=lambda t, y, v, p: system.derivatives.rate(t, y[order], v[order], p)[order][
                :, order
            ],
        ),
    )
    report = solve_residual(swapped, [0.5, 1], time_span_s=(0, 1))
    assert report.success, report
    assert report.final_point.state == pytest.approx([np.exp(-1.4), np.exp(-0.7)], rel=3e-7)


@pytest.mark.parametrize("grid", [[0.2, 0.2], [-0.1], [1.1], [np.nan], [[0.5]]])
def test_invalid_output_grids_are_rejected_before_model_evaluation(grid):
    def unused(t, y, v, p):
        raise AssertionError("invalid input must not evaluate the model")

    with pytest.raises(ValueError, match="evaluation grid"):
        solve_residual(
            replace(quadratic_dae(), residual=unused),
            [1, 1],
            time_span_s=(0, 1),
            evaluation_times_s=grid,
        )


def test_late_domain_failure_discards_every_candidate():
    system = quadratic_dae()

    def residual(t, y, v, p):
        if t > 0.3:
            raise ResidualDomainError("model valid only through 0.3 s")
        return system.residual(t, y, v, p)

    report = solve_residual(replace(system, residual=residual), [1, 1], time_span_s=(0, 1))
    assert not report.success and not report.points and report.final_point is None
    assert "model valid only" in report.failure_reason
    assert report.diagnostics["root_calls"] > 2
