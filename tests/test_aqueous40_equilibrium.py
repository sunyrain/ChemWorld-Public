"""Finite-water equilibria from declared reactions and the common root contract."""

import json
from dataclasses import replace
from math import log, sqrt

import numpy as np
import pytest
from scipy.optimize import brentq

from chemworld.physchem.aqueous_reaction_equilibrium import (
    AqueousEquilibriumSpec,
    aqueous_chemical_potential_jacobian,
    build_aqueous_equilibrium_system,
    solve_aqueous_reaction_equilibrium,
)
from chemworld.physchem.residual_system import ResidualEvaluator, RootPolicy, finite_jacobian
from reference.aqueous40_fixtures import (
    KA,
    KW,
    WATER_MASS,
    add_inert_species,
    complexation_model,
    feed,
    isomer_model,
    model,
)


def solve(spec, amounts=None, **kwargs):
    return solve_aqueous_reaction_equilibrium(
        spec,
        feed(spec) if amounts is None else amounts,
        temperature_K=kwargs.pop("temperature_K", 298.15),
        pressure_Pa=kwargs.pop("pressure_Pa", 1e5),
        **kwargs,
    )


def assert_closed(result):
    assert result.success, (result.failure_reason, result.diagnostics)
    assert max(abs(x) for x in result.element_residuals_mol.values()) < 2e-9
    assert abs(result.charge_residual_mol) < 2e-10
    assert (
        max((abs(x) for x in result.reaction_residuals.values() if x is not None), default=0) < 2e-8
    )
    assert result.diagnostics["maximum_scaled_conservation_residual"] < 2e-9
    assert result.property_report.valid
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["success"]


def test_ideal_acid_matches_quadratic_and_preserves_feed():
    spec = model()
    before = feed(spec)
    initial = before.copy()
    result = solve(spec, before)
    assert_closed(result)
    expected = 2 * KA * 0.1 / (KA + sqrt(KA * KA + 4 * KA * 0.1))
    assert result.amounts_mol["H+"] == pytest.approx(expected, rel=2e-8)
    assert result.amounts_mol["A-"] == pytest.approx(expected, rel=2e-8)
    assert result.amounts_mol["HA"] == pytest.approx(0.1 - expected, rel=2e-8)
    assert result.amounts_mol["H2O"] == pytest.approx(initial["H2O"], rel=2e-12)
    assert before == initial
    assert result.diagnostics["root"]["algorithm"] == "scaled_projected_newton_with_gradient"


def test_finite_water_matches_independent_scalar_equation():
    spec = model(network="water")
    initial = feed(spec)
    nw0 = initial["H2O"]

    def equation(x):
        m = x / ((nw0 - x) * WATER_MASS)
        return 2 * log(m) + 2 * WATER_MASS * m - log(KW)

    extent = brentq(equation, 1e-12, 1.0, xtol=1e-14)
    result = solve(spec, initial)
    assert_closed(result)
    assert result.amounts_mol["H+"] == pytest.approx(extent, rel=2e-9)
    assert result.amounts_mol["H2O"] == pytest.approx(nw0 - extent, rel=2e-12)
    assert result.chemical_potential_report.solvent_mass_kg < 1.0


@pytest.mark.parametrize(
    "family", ["ideal", "DebyeHuckelLimiting", "DebyeHuckel", "Davies", "SIT", "Pitzer"]
)
def test_all_activity_families_finite_water_and_scaling(family):
    spec = model(family, network="water", redundant=True)
    result = solve(spec)
    assert_closed(result)
    small = solve(spec, feed(spec, water_kg=0.01))
    assert_closed(small)
    for key, value in result.amounts_mol.items():
        assert small.amounts_mol[key] == pytest.approx(value * 0.01, rel=2e-8, abs=1e-12)
    assert result.diagnostics["reaction_subspace_dimension"] == 1
    assert result.reaction_residuals["reverse"] == pytest.approx(
        -2 * result.reaction_residuals["water"], abs=1e-12
    )


def test_reaction_subset_keeps_extra_invariants_and_zero_element_species():
    spec = add_inert_species(model(network="mixed"))
    spec = replace(spec, reaction_ids=("water",))
    initial = feed(spec)
    result = solve(spec, initial)
    assert_closed(result)
    assert result.amounts_mol["HA"] == pytest.approx(initial["HA"], rel=1e-12)
    assert result.amounts_mol["A-"] == 0
    assert result.amounts_mol["inert-C"] == 0
    assert set(result.diagnostics["structurally_absent_species"]) == {"A-", "inert-C"}
    assert result.diagnostics["reaction_subspace_dimension"] == 1


def test_mixed_water_acid_and_codec():
    spec = model("Pitzer", network="mixed", redundant=True)
    restored = AqueousEquilibriumSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert_closed(solve(restored))


def test_analytic_hessian_and_shared_residual_jacobian():
    spec = model("Pitzer", network="mixed")
    amounts = {"H2O": 40.0, "H+": 0.22, "OH-": 0.2, "HA": 0.08, "A-": 0.02}
    ids = tuple(amounts)
    analytical = aqueous_chemical_potential_jacobian(
        spec.database, amounts, temperature_K=310, pressure_Pa=5e5, species_ids=ids
    )

    def chemical(values):
        report = spec.database.chemical_potentials(
            dict(zip(ids, values, strict=True)), temperature_K=310, pressure_Pa=5e5
        )
        from chemworld.physchem.property_reports import R_J_PER_MOL_K

        return np.array([report.chemical_potentials_J_mol[k] for k in ids]) / (R_J_PER_MOL_K * 310)

    values = np.array(list(amounts.values()))
    numerical = finite_jacobian(chemical, values, values, relative_step=2e-5)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(analytical, analytical.T, rtol=0, atol=1e-12)
    problem = build_aqueous_equilibrium_system(spec, amounts, temperature_K=310, pressure_Pa=5e5)
    evaluator = ResidualEvaluator(problem.system)
    zero = np.zeros(len(problem.initial))
    jac = evaluator.derivative("state", 0, problem.initial, zero)
    fd = finite_jacobian(
        lambda v: evaluator.value(0, v, zero),
        problem.initial,
        np.ones(len(zero)),
        relative_step=2e-5,
    )
    np.testing.assert_allclose(jac, fd, rtol=2e-5, atol=2e-7)


def test_numerical_derivative_mode_and_explicit_guess():
    spec = model("Davies", network="water")
    actual = solve(spec)
    assert_closed(actual)
    numerical = solve(
        replace(spec, derivative_mode="finite_difference"), initial_guess_mol=actual.amounts_mol
    )
    assert_closed(numerical)
    assert numerical.amounts_mol == pytest.approx(actual.amounts_mol, rel=1e-8)


def test_invalid_domains_and_failed_root_never_return_committable_inventory():
    spec = model(network="water")
    with pytest.raises(ValueError, match="electroneutral"):
        solve(spec, {**feed(spec), "H+": 0.1})
    with pytest.raises(ValueError, match="positive actual water"):
        solve(spec, {"H2O": 0})
    with pytest.raises(ValueError, match="domain"):
        solve(spec, temperature_K=400)
    with pytest.raises(ValueError, match="declared"):
        solve(spec, {**feed(spec), "unknown": 1})
    result = solve(replace(spec, root_policy=RootPolicy(maximum_iterations=1)))
    assert not result.success
    assert result.amounts_mol is None and result.property_report is None
    assert result.failure_reason == "iteration_limit"


def test_no_reactions_preserves_all_positive_stock():
    spec = replace(model(network="water"), reaction_ids=())
    result = solve(spec)
    assert_closed(result)
    assert result.amounts_mol == feed(spec)
    assert result.diagnostics["local_curvature"] == "no_reaction_freedom"


def test_complexation_matches_independent_quadratic():
    spec = complexation_model()
    initial = {"H2O": 1 / WATER_MASS, "M+": 0.2, "L-": 0.2, "ML": 0.0}
    result = solve(spec, initial)
    assert_closed(result)
    free = 0.4 / (1 + sqrt(1 + 4 * 10 * 0.2))
    assert result.amounts_mol["M+"] == pytest.approx(free, rel=2e-8)
    assert result.amounts_mol["L-"] == pytest.approx(free, rel=2e-8)
    assert result.amounts_mol["ML"] == pytest.approx(0.2 - free, rel=2e-8)


def test_unresolved_boundary_and_unstable_stationary_point_are_explicit_failures():
    spec = isomer_model(blocked=True)
    initial = {"H2O": 1 / WATER_MASS, "A": 1.0, "B": 0.0, "C": 0.0}
    result = solve(spec, initial)
    assert not result.success
    assert result.failure_reason == "no_resolved_positive_feasible_state"
    assert result.amounts_mol is None
    assert initial["B"] == initial["C"] == 0
    spec = isomer_model(unstable=True)
    result = solve(spec, {"H2O": 1 / WATER_MASS, "A": 0.5, "B": 0.5})
    assert not result.success
    assert result.failure_reason == "stationary_root_has_negative_curvature"
    assert result.amounts_mol is None
    assert result.diagnostics["constrained_curvature_per_mol"][0] < 0
