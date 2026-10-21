"""Independent diffusion modes and real shared-advance mathematical consumers."""

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.finite_volume import (
    ConservativeDiffusion,
    DiffusionBoundary,
    DiffusionMesh,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


def mesh(geometry, cells):
    return (DiffusionMesh.planar(1, cells, area_m2=.4) if geometry == "planar"
            else DiffusionMesh.spherical(1, cells, particle_count=2))


def mode_averages(grid, time):
    left, right = np.asarray(grid.faces_m[:-1]), np.asarray(grid.faces_m[1:])
    decay = .2 * np.exp(-.1 * pi ** 2 * time)
    if grid.geometry == "planar":
        return 2 + decay * (np.sin(pi * right) - np.sin(pi * left)) / (pi * (right - left))
    # Exact integral of r² sinc(r), independent of the finite-volume stencil.
    def integral(r):
        return (np.sin(pi * r) - pi * r * np.cos(pi * r)) / pi ** 3
    return 1 + decay * 3 * (integral(right) - integral(left)) / (right ** 3 - left ** 3)


def coupled(operator, concentration, *, method="BDF"):
    keys = tuple(f"material:cell:{i}" for i in range(operator.mesh.cells))
    open_sides = {side: 0.0 for side, boundary in
                  (("left", operator.left), ("right", operator.right))
                  if boundary.kind != "symmetry" and not (
                      boundary.kind == "flux" and boundary.value == 0)}
    binding = operator.couple(keys, process_id="diffusion", owner_id="fixture-material",
                              boundary_enthalpies_J_mol=open_sides)
    # Identical, zero-reference internal energies make this an explicitly
    # isothermal mathematical transport fixture; no electrochemical heat claim.
    system = CoupledSystem(
        ConstantCvThermodynamics({key: SpeciesInternalEnergy(0, 0) for key in keys},
                                 support_heat_capacity_J_K=1),
        processes=(binding.process,), ports=binding.ports, isothermal=True,
        conservation_laws={"material_mol": dict.fromkeys(keys, 1)},
        inventory_scales_mol=dict.fromkeys(keys, sum(operator.mesh.cell_volumes_m3)),
        policy=ODESolverPolicy("finite-volume-fixture-v1", method=method,
                               rtol=1e-10, atol=1e-13),
    )
    state = CoupledState(dict(zip(keys, concentration * operator.mesh.cell_volumes_m3,
                                  strict=True)), 298.15,
                         sum(operator.mesh.cell_volumes_m3) * 1000)
    return system, state


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
def test_analytic_modes_converge_at_second_order_and_close_boundary_receipts(geometry):
    errors = []
    for cells in (20, 40, 80):
        grid = mesh(geometry, cells)
        operator = ConservativeDiffusion(grid, .1, right=(
            DiffusionBoundary("concentration", 1) if geometry == "spherical"
            else DiffusionBoundary()))
        system, state = coupled(operator, mode_averages(grid, 0))
        result = system.advance(state, .2, evaluation_times_s=(0, .1, .2))
        assert result.outcome == "success" and result.actual_duration_s == .2
        exact = mode_averages(grid, .2)
        actual = np.array(list(result.final_state.amounts_mol.values())) / grid.cell_volumes_m3
        errors.append(float(np.sqrt(np.average((actual - exact) ** 2,
                                               weights=grid.cell_volumes_m3))))
        assert np.max(np.abs(actual - exact)) < 1e-3
        assert all(min(point.amounts_mol.values()) >= 0 for point in result.trajectory)
        delta = sum(result.final_state.amounts_mol.values()) - sum(state.amounts_mol.values())
        boundary = sum(port.sign * result.final_state.exchanged_mol[port.port_id]
                       for port in system.ports)
        assert delta == pytest.approx(boundary, abs=1e-10)
        assert abs(result.conservation_residuals["material_mol"]) < 1e-10
        assert result.solver_diagnostic["solver_calls"] == 1
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    print(f"{geometry}: cells=20,40,80 errors={errors}, orders={orders.tolist()}")
    assert min(orders) > 1.8


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
def test_no_flux_constant_field_and_each_internal_face_cancel(geometry):
    grid = mesh(geometry, 20)
    operator = ConservativeDiffusion(grid, .1)
    amounts = 3 * grid.cell_volumes_m3
    assert np.max(np.abs(operator.evaluate(amounts).amount_rates_mol_s)) < 1e-13
    amounts = np.random.default_rng(12).uniform(.1, 2, grid.cells) * grid.cell_volumes_m3
    result = operator.evaluate(amounts)
    individual_faces = np.zeros((grid.cells - 1, grid.cells))
    for face, rate in enumerate(result.face_rates_mol_s[1:-1], 1):
        individual_faces[face - 1, face - 1] -= rate
        individual_faces[face - 1, face] += rate
    np.testing.assert_array_equal(individual_faces.sum(axis=1), 0)
    np.testing.assert_allclose(individual_faces.sum(axis=0), result.amount_rates_mol_s,
                               rtol=1e-14, atol=1e-14)
    assert sum(result.amount_rates_mol_s) == pytest.approx(0, abs=1e-12)
    assert operator.inventory_jacobian_s_inv.nnz <= 3 * grid.cells


@pytest.mark.parametrize("exponent", [1, 2, 3])
def test_spherical_shell_average_quadratic_has_exact_flux_on_graded_mesh(exponent):
    faces = 1 - (1 - np.linspace(0, 1, 13))**exponent
    grid = DiffusionMesh("spherical", tuple(faces), particle_count=2)
    a, b = faces[:-1], faces[1:]
    # Independent exact volume integral, not midpoint sampling.
    means = 2 + .3 * 3 / 5 * (b**5 - a**5) / (b**3 - a**3)
    operator = ConservativeDiffusion(grid, .1, right=DiffusionBoundary("concentration", 2.3))
    result = operator.evaluate(means * grid.cell_volumes_m3)
    expected = -.1 * .6 * faces * (8 * pi * faces**2)
    np.testing.assert_allclose(result.face_rates_mol_s, expected, atol=2e-10, rtol=1e-9)
    np.testing.assert_allclose(result.amount_rates_mol_s / grid.cell_volumes_m3,
                               6 * .1 * .3, atol=2e-7)
    boundary = means[-1] + .6 * grid.spherical_surface_distance_m
    assert boundary == pytest.approx(2.3, abs=1e-12)


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
def test_sparse_inventory_jacobian_matches_independent_perturbations(geometry):
    grid = mesh(geometry, 20)
    operator = ConservativeDiffusion(grid, .3, right=DiffusionBoundary("concentration", 2))
    amounts = np.random.default_rng(5).uniform(.5, 1.5, grid.cells) * grid.cell_volumes_m3
    finite_difference = np.zeros((grid.cells, grid.cells))
    for column in range(grid.cells):
        perturbation = np.zeros(grid.cells)
        epsilon = amounts[column] * 1e-4
        perturbation[column] = epsilon
        finite_difference[:, column] = (
            operator.evaluate(amounts + perturbation).amount_rates_mol_s
            - operator.evaluate(amounts - perturbation).amount_rates_mol_s
        ) / (2 * epsilon)
    np.testing.assert_allclose(operator.inventory_jacobian_s_inv.toarray(), finite_difference,
                               rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
@pytest.mark.parametrize("outward_flux", [-.02, .02])
def test_prescribed_boundary_flux_uses_physical_area_and_one_shared_clock(geometry, outward_flux):
    grid = mesh(geometry, 20)
    operator = ConservativeDiffusion(grid, .1, right=DiffusionBoundary("flux", outward_flux))
    system, state = coupled(operator, np.ones(grid.cells))
    result = system.advance(state, .05)
    delta = sum(result.final_state.amounts_mol.values()) - sum(state.amounts_mol.values())
    assert delta == pytest.approx(-outward_flux * grid.face_areas_m2[-1] * .05, abs=1e-10)
    assert result.actual_duration_s == .05
    receipts = result.actual_exchanges
    assert len(receipts) == 1 and receipts[0]["unit"] == "mol"
    assert receipts[0]["direction"] == ("out" if outward_flux > 0 else "in")


@pytest.mark.parametrize("geometry", ["planar", "spherical"])
def test_outward_flux_stops_at_real_depletion_and_accounts_only_actual_time(geometry):
    grid = mesh(geometry, 1)
    operator = ConservativeDiffusion(grid, 0, right=DiffusionBoundary("flux", .2))
    system, state = coupled(operator, np.ones(grid.cells))
    expected_time = sum(state.amounts_mol.values()) / (.2 * grid.face_areas_m2[-1])
    result = system.advance(state, expected_time * 2,
                            evaluation_times_s=(0, expected_time * 2))
    assert result.outcome == "stopped" and result.stopping_event == "diffusion:depleted:right"
    assert result.actual_duration_s == pytest.approx(expected_time, rel=1e-8)
    assert min(result.final_state.amounts_mol.values()) >= 0
    assert sum(result.final_state.amounts_mol.values()) < 1e-9
    assert result.actual_exchanges[0]["amount"] == pytest.approx(sum(state.amounts_mol.values()))
    assert result.trajectory[-1].time_s < expected_time * 2


def test_nonzero_diffusion_stops_at_boundary_cell_depletion_with_material_remaining():
    grid = DiffusionMesh.planar(1, 20)
    operator = ConservativeDiffusion(grid, .1, right=DiffusionBoundary("flux", 1))
    system, state = coupled(operator, np.ones(grid.cells))
    result = system.advance(state, 2)
    assert result.outcome == "stopped" and 0 < result.actual_duration_s < 1
    assert min(result.final_state.amounts_mol.values()) >= 0
    assert sum(result.final_state.amounts_mol.values()) > 0
    assert result.final_state.amounts_mol["material:cell:19"] < 1e-10
    assert result.actual_exchanges[0]["amount"] == pytest.approx(result.actual_duration_s)


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_closed_diffusion_preserves_finite_inventory_and_concentration_bounds(method):
    grid = DiffusionMesh.spherical(1e-5, 20, particle_count=1e8)
    concentration = np.linspace(3000, 27000, grid.cells)
    system, state = coupled(ConservativeDiffusion(grid, 1e-14), concentration, method=method)
    result = system.advance(state, 1000, evaluation_times_s=(0, 100, 1000))
    for point in result.trajectory:
        actual = np.asarray(list(point.amounts_mol.values())) / grid.cell_volumes_m3
        assert np.min(actual) >= 3000 - 1e-6 and np.max(actual) <= 27000 + 1e-6
        assert sum(point.amounts_mol.values()) == pytest.approx(sum(state.amounts_mol.values()),
                                                               abs=1e-12)
    assert result.actual_exchanges == ()


def test_planar_left_flux_and_empty_dirichlet_inflow_use_the_same_sign_convention():
    grid = DiffusionMesh.planar(1, 1, area_m2=2)
    operator = ConservativeDiffusion(grid, .1, left=DiffusionBoundary("flux", -.2),
                                     right=DiffusionBoundary("flux", .1))
    assert operator.evaluate([2]).amount_rates_mol_s[0] == pytest.approx(.2)
    system, state = coupled(operator, np.ones(1))
    result = system.advance(state, .5)
    assert sum(result.final_state.amounts_mol.values()) == pytest.approx(2.1)
    inflow = ConservativeDiffusion(grid, .1, right=DiffusionBoundary("concentration", 1))
    system, state = coupled(inflow, np.zeros(1))
    result = system.advance(state, 1)
    # For one cell, dc/dt=(D/(L/2)/L)*(1-c), independent of implementation.
    assert sum(result.final_state.amounts_mol.values()) / 2 == pytest.approx(
        1 - np.exp(-.2), rel=1e-8)


def test_stencil_reaches_sparse_backend_and_unknown_processes_stay_dense(monkeypatch):
    import chemworld.physchem.coupled_system as module
    operator = ConservativeDiffusion(DiffusionMesh.planar(1, 20), .1)
    system, state = coupled(operator, np.linspace(1, 2, 20), method="Radau")
    captured = []
    real = module.solve_ode

    def solve(*args, **kwargs):
        captured.append(kwargs["jac_sparsity"])
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "solve_ode", solve)
    result = system.advance(state, .01)
    assert result.outcome == "success" and len(captured) == 1
    pattern = captured[0].toarray()
    assert pattern.shape == (25, 25) and pattern[:20, :20].sum() == 58
    assert not np.any(pattern[:, 21:])
    generic = CoupledProcess("unknown", lambda _: ProcessContribution())
    conservative = replace(system, processes=(*system.processes, generic))
    assert np.all(conservative.jacobian_sparsity().toarray()[:20, :20])
    with pytest.raises(ValueError, match="unknown inventory"):
        replace(system, processes=(replace(system.processes[0],
                                          inventory_dependencies={"absent": ()}),))


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf")])
def test_invalid_inventory_and_boundary_values_are_rejected(invalid):
    operator = ConservativeDiffusion(DiffusionMesh.planar(1, 1), .1)
    with pytest.raises(ValueError):
        operator.evaluate([invalid])
    with pytest.raises(ValueError):
        DiffusionBoundary("concentration", invalid)


def test_geometry_symmetry_units_and_boundary_enthalpy_are_explicit():
    with pytest.raises(ValueError):
        DiffusionMesh.planar(1, 2.5)
    with pytest.raises(ValueError):
        DiffusionMesh("spherical", (0, 1, .5))
    for boundary in (DiffusionBoundary("flux", 1), DiffusionBoundary("concentration", 1)):
        with pytest.raises(ValueError, match="center"):
            ConservativeDiffusion(DiffusionMesh.spherical(1, 2), .1, left=boundary)
    operator = ConservativeDiffusion(DiffusionMesh.planar(1, 2), .1,
                                     right=DiffusionBoundary("concentration", 1))
    with pytest.raises(ValueError, match="enthalpy"):
        operator.couple(("a", "b"), process_id="fixture", owner_id="material")
    for method, pattern in (("DOP853", np.eye(1)), ("BDF", np.ones((2, 2))),
                            ("BDF", [[float("nan")]])):
        with pytest.raises(ValueError, match="sparsity"):
            solve_ode(lambda _, y: -y, [1], time_span_s=(0, 1), jac_sparsity=pattern,
                      policy=ODESolverPolicy("invalid-sparse", method=method))
