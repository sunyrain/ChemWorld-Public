"""Exact quartic interface and unrestricted binary profile functionality."""

import json
from dataclasses import replace
from math import log, sqrt

import numpy as np
import pytest

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_interface import (
    SquareGradientInfluence,
    molecular_interface_bulk,
    molecular_planar_interface,
    pure_square_gradient_interface,
)
from reference.run_molecular_interface import analytic_pair


def test_exact_pure_quartic_tension_and_profile():
    model, states, influence, gamma = analytic_pair()
    grid = (1040., 1400., 2000., 3000., 4000., 4600., 4960.)
    result = pure_square_gradient_interface(model, *states, influence, interior_density_grid=grid)
    assert result.success, result.to_dict()
    assert result.surface_tension_N_m == pytest.approx(gamma, rel=2e-8)
    expected = [log((v-1000)/(5000-v))/(4000*sqrt(2*1e-8/1e-18)) for v in grid]
    assert result.positions_m == pytest.approx(expected, abs=1e-13)
    restored = SquareGradientInfluence.from_dict(json.loads(json.dumps(influence.to_dict())))
    assert restored == influence
    json.dumps(result.to_dict(), allow_nan=False)
    # The interface interior is a spinodal; the free-energy API must still work.
    bulk = molecular_interface_bulk(model, 300, (3000.,))
    assert bulk.helmholtz_density_J_m3 == pytest.approx(6e4)


def test_binary_free_profile_has_curved_density_path():
    model, states, influence, gamma = analytic_pair(True)
    result = molecular_planar_interface(model, *states, influence,
        mesh=DiffusionMesh.planar(40e-9, 64), component_density_bounds=((100., 8000.),)*2)
    assert result.success, result.to_dict()
    assert result.surface_tension_N_m == pytest.approx(gamma, rel=.02)
    profile = np.array(result.component_densities_mol_m3)
    fractions = (profile-np.array(model.left))/(np.array(model.right)-model.left)
    assert np.max(abs(fractions[:, 0]-fractions[:, 1])) > .04
    assert result.diagnostics["stationarity_residual_RT"] < 1e-5


def test_invalid_coexistence_influence_and_numerical_failure():
    model, states, influence, _ = analytic_pair(True)
    for matrix in (((1e-18, 1e-18), (1e-18, 1e-18)),
                   ((1e-18, 2e-18), (2e-18, 1e-18))):
        with pytest.raises(ValueError, match="positive definite"):
            replace(influence, coefficients_J_m5_mol2=matrix)
    with pytest.raises(ValueError, match="coexist"):
        molecular_planar_interface(model, states[0], replace(states[1], pressure_Pa=2e5), influence,
            mesh=DiffusionMesh.planar(40e-9, 16), component_density_bounds=((100., 8000.),)*2)
    failed = molecular_planar_interface(model, *states, influence,
        mesh=DiffusionMesh.planar(40e-9, 16), component_density_bounds=((100., 8000.),)*2,
        max_iterations=1)
    assert not failed.success


def test_bulk_positive_trace_mu_and_exact_absent_support():
    first = CPAComponent("a", .2, 2e-5, 500, .6, "explicit dilute trace fixture")
    model = CPAModel((first, replace(first, component_id="b")), ((0, 0), (0, 0)),
                     (200, 500), "identical cubic component trace fixture")
    absent = molecular_interface_bulk(model, 400, (10, 0))
    trace = molecular_interface_bulk(model, 400, (10, 1e-14))
    assert absent.chemical_potentials_J_mol[1] is None
    assert all(np.isfinite(v) for v in trace.chemical_potentials_J_mol)
    assert trace.chemical_potentials_J_mol[1]-trace.chemical_potentials_J_mol[0] == pytest.approx(
        model.gas_constant_J_mol_K*400*log(1e-15), rel=1e-12)
