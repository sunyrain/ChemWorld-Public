"""Unrestricted ternary profiles, explicit DAE limits and critical scaling."""

import json
from dataclasses import dataclass, replace
from math import sqrt

import numpy as np
import pytest

from chemworld.physchem.caloric_math import logarithm, scalar_value
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_interface import (
    PolynomialSquareGradientInfluence,
    SingularSquareGradientInfluence,
    SquareGradientInfluence,
    molecular_influence_from_dict,
    molecular_planar_interface,
    pure_square_gradient_interface,
)
from reference.run_molecular_interface import QuarticModel


def states_for(model, left, right, temperature=300.0):
    return tuple(
        molecular_eos_state(model, temperature, sum(c), tuple(v / sum(c) for v in c))
        for c in (left, right)
    )


def test_three_independent_profile_coordinates_and_exact_scalar_sum():
    model = QuarticModel((1000.0, 2000.0, 2500.0), (5000.0, 6000.0, 6500.0), (1e-8, 2e-8, 3e-8))
    influence = SquareGradientInfluence(
        model.component_ids,
        tuple(tuple(1e-18 if i == j else 0.0 for j in range(3)) for i in range(3)),
        (200.0, 400.0),
        "declared three independent quartic interfaces",
    )
    result = molecular_planar_interface(
        model,
        *states_for(model, model.left, model.right),
        influence,
        mesh=DiffusionMesh.planar(40e-9, 96),
        component_density_bounds=((100.0, 8000.0),) * 3,
    )
    assert result.success, result.to_dict()
    expected = sum(
        sqrt(2 * a * 1e-18) * (hi - lo) ** 3 / 6
        for a, lo, hi in zip(model.amplitudes, model.left, model.right, strict=True)
    )
    assert result.surface_tension_N_m == pytest.approx(expected, rel=0.02)
    assert result.diagnostics["active_components"] == 3
    assert result.diagnostics["influence_rank"] == 3
    assert result.diagnostics["root_iterations_total"] <= 500
    profiles = np.asarray(result.component_densities_mol_m3)
    fractions = (profiles - np.asarray(model.left)) / (np.asarray(model.right) - model.left)
    assert np.max(np.ptp(fractions, axis=1)) > 0.06


@dataclass(frozen=True)
class CurvedAlgebraicValley:
    stiffness: float = 10.0
    component_ids: tuple[str, ...] = ("order", "adsorbate", "solvent")
    gas_constant_J_mol_K: float = 8.31446261815324

    @staticmethod
    def path(q):
        bend = (q - 1000) * (5000 - q) / 4000
        return (q, 2000 + 0.5 * (q - 1000) + 0.2 * bend, 1500 + 0.2 * (q - 1000) - 0.1 * bend)

    def alphar(self, temperature, density, fractions):
        c = tuple(density * x for x in fractions)
        path = self.path(c[0])
        free = 1e-8 * (c[0] - 1000) ** 2 * (5000 - c[0]) ** 2
        free += self.stiffness / 2 * ((c[1] - path[1]) ** 2 + (c[2] - path[2]) ** 2)
        ideal = sum(v * (logarithm(v) - 1) for v in c if scalar_value(v) > 0)
        return ((free - 1e5) / (self.gas_constant_J_mol_K * temperature) - ideal) / density


def test_rank_one_dae_has_curved_free_composition_and_exact_tension():
    model = CurvedAlgebraicValley()
    states = states_for(model, model.path(1000.0), model.path(5000.0))
    influence = SingularSquareGradientInfluence(
        model.component_ids,
        ((1e-9,), (0.0,), (0.0,)),
        (200.0, 400.0),
        "one gradient direction, two algebraic densities",
    )
    result = molecular_planar_interface(
        model,
        *states,
        influence,
        mesh=DiffusionMesh.planar(40e-9, 64),
        component_density_bounds=((100.0, 8000.0),) * 3,
    )
    assert result.success, result.to_dict()
    expected = sqrt(2e-8 * 1e-18) * 4000**3 / 6
    assert result.surface_tension_N_m == pytest.approx(expected, rel=0.02)
    profile = np.asarray(result.component_densities_mol_m3)
    for row in profile:
        assert row == pytest.approx(model.path(row[0]), abs=5e-5)
    assert result.diagnostics["nullspace_dimension"] == 2
    assert result.diagnostics["minimum_nullspace_bulk_curvature_J_m3_mol2"] == pytest.approx(
        10.0, rel=1e-10
    )
    assert result.diagnostics["singular_regularization"].startswith("none")
    assert result.diagnostics["algebraic_nullspace_residual_RT"] <= sqrt(3) * 1e-7
    assert molecular_influence_from_dict(json.loads(json.dumps(influence.to_dict()))) == influence
    json.dumps(result.to_dict(), allow_nan=False)


def test_singular_model_does_not_accept_an_unstable_algebraic_phase():
    model = CurvedAlgebraicValley(stiffness=-0.01)
    states = states_for(model, model.path(1000.0), model.path(5000.0))
    influence = SingularSquareGradientInfluence(
        model.component_ids,
        ((1e-9,), (0.0,), (0.0,)),
        (200.0, 400.0),
        "explicit unstable nullspace control",
    )
    with pytest.raises(ValueError, match="algebraic nullspace"):
        molecular_planar_interface(
            model,
            *states,
            influence,
            mesh=DiffusionMesh.planar(40e-9, 16),
            component_density_bounds=((100.0, 8000.0),) * 3,
        )


@pytest.mark.parametrize("gap", [4000.0, 400.0, 40.0])
def test_near_coexistence_cubic_tension_and_inverse_gap_width(gap):
    lo, hi = 3000 - gap / 2, 3000 + gap / 2
    model = QuarticModel((lo,), (hi,), (1e-8,))
    influence = SquareGradientInfluence(
        model.component_ids,
        ((1e-18,),),
        (200.0, 400.0),
        "fixed gradient coefficient approaching coalescence",
    )
    result = pure_square_gradient_interface(
        model,
        *states_for(model, model.left, model.right),
        influence,
        interior_density_grid=(lo + 0.1 * gap, lo + 0.5 * gap, lo + 0.9 * gap),
        quadrature_tolerance=1e-7,
    )
    assert result.success, result.to_dict()
    assert result.surface_tension_N_m == pytest.approx(sqrt(2e-26) * gap**3 / 6, rel=2e-6)
    assert (result.positions_m[-1] - result.positions_m[0]) * gap == pytest.approx(
        2 * np.log(9) / sqrt(2e10), rel=2e-6
    )
    if gap == 40:
        assert result.diagnostics["grand_potential_evaluation"] == "Hessian_path_integral"


def test_temperature_polynomial_mixing_codec_and_full_domain_validation():
    influence = PolynomialSquareGradientInfluence(
        ("a", "b"),
        ((1e-18, 0.1e-18), (4e-18,)),
        ((0.0, 0.3), (0.3, 0.0)),
        (250.0, 400.0),
        "declared polynomial",
        reference_temperature_K=300.0,
        temperature_scale_K=100.0,
    )
    matrix = np.asarray(influence.matrix_at(350.0))
    assert matrix[0, 0] == pytest.approx(1.05e-18, abs=0, rel=1e-14)
    assert matrix[0, 1] == pytest.approx(0.7 * sqrt(1.05e-18 * 4e-18), abs=0, rel=1e-14)
    assert molecular_influence_from_dict(json.loads(json.dumps(influence.to_dict()))) == influence
    rank_one = replace(influence, interaction=((0.0, 0.0), (0.0, 0.0)), declared_rank=1)
    assert np.linalg.matrix_rank(np.asarray(rank_one.matrix_at(350.0))) == 1
    with pytest.raises(ValueError, match="positive definite"):
        replace(rank_one, declared_rank=None)
    with pytest.raises(ValueError, match="temperature domain"):
        PolynomialSquareGradientInfluence(
            ("a",),
            ((0.24e-18, -1e-18, 1e-18),),
            ((0.0,),),
            (300.0, 400.0),
            "positive endpoints but negative interior",
            reference_temperature_K=300.0,
            temperature_scale_K=100.0,
        )


def test_exact_coalescence_and_unresolved_declared_rank_are_not_regularized():
    model = QuarticModel()
    states = states_for(model, model.left, model.right)
    influence = SquareGradientInfluence(
        model.component_ids, ((1e-18,),), (200.0, 400.0), "constant"
    )
    with pytest.raises(ValueError, match="distinct phases"):
        pure_square_gradient_interface(
            model, states[0], states[0], influence, interior_density_grid=(2000.0,)
        )
    with pytest.raises(ValueError, match="independent factors"):
        SingularSquareGradientInfluence(
            ("a", "b", "c"),
            ((1e-9, 0.0), (0.0, 1e-30), (0.0, 0.0)),
            (200.0, 400.0),
            "unresolved rank-two factors",
        )


def covolume_trial_case():
    """Actual failed Rolo-F binary forward; measured tension is not an oracle here."""
    ids = ("n-heptane", "n-decane")
    r, oma, omb = 8.314, 0.4572355289213825, 0.07779607390388854
    constants = ((540.2, 2735730.0, 0.88001221408), (617.7, 2103000.0, 1.0634945355648))
    model = CPAModel(
        tuple(
            CPAComponent(
                name,
                oma * r * r * tc * tc / pc,
                omb * r * tc / pc,
                tc,
                kappa,
                "HEOS constants; explicit PhasePy PR formula",
            )
            for name, (tc, pc, kappa) in zip(ids, constants, strict=True)
        ),
        ((0.0, 0.0), (0.0, 0.0)),
        (290.0, 340.0),
        "Rolo table 2 x2=.251,T293.15 functional domain-trial regression; air omitted",
        cubic_model="peng_robinson",
        gas_constant_J_mol_K=r,
    )
    states = (
        molecular_eos_state(
            model, 293.15, 1.5252897135966168, (0.9899287923592117, 0.010071207640788282)
        ),
        molecular_eos_state(model, 293.15, 6084.986547272513, (0.749, 0.251)),
    )
    density = np.array([np.array(s.mole_fractions) * s.molar_density_mol_m3 for s in states])
    bounds = tuple(
        (float(0.01 * min(density[:, i])), float(1.5 * max(density[:, i]))) for i in range(2)
    )
    influence = PolynomialSquareGradientInfluence(
        ids,
        (
            (5.996182276665448e-19, 1.441548875545607e-20),
            (1.4230174717842928e-18, -6.475902319830202e-20),
        ),
        ((0.0, 0.1), (0.1, 0.0)),
        (290.0, 340.0),
        "original failed F parameters",
        298.15,
        100.0,
    )
    return model, states, bounds, influence


def test_cubic_covolume_trial_is_rejected_without_changing_density_box():
    model, states, bounds, influence = covolume_trial_case()
    result = molecular_planar_interface(
        model,
        *states,
        influence,
        mesh=DiffusionMesh.planar(12e-9, 256),
        component_density_bounds=bounds,
    )
    assert result.success, result.to_dict()
    diagnostics = result.diagnostics
    rejected = diagnostics["minimization_domain_rejections"]
    assert rejected["count"] > 0
    assert any("covolume pole" in reason for reason in rejected["reasons"])
    assert diagnostics["root_iterations_total"] <= 500
    assert diagnostics["stationarity_residual_RT"] <= 1e-7
    assert diagnostics["first_integral_variation_relative"] <= 0.05
    assert diagnostics["scaled_minimum_curvature"] >= -1e-7
    assert result.surface_tension_N_m > 0
    profile = np.asarray(result.component_densities_mol_m3)
    assert np.all(profile >= np.array(bounds)[:, 0])
    assert np.all(profile <= np.array(bounds)[:, 1])
    json.dumps(result.to_dict(), allow_nan=False)
    continued = molecular_planar_interface(
        model,
        *states,
        replace(influence, interaction=((0.0, 0.1005), (0.1005, 0.0))),
        mesh=DiffusionMesh.planar(12e-9, 256),
        component_density_bounds=bounds,
        initial_profile=result.component_densities_mol_m3,
    )
    assert continued.success, continued.to_dict()
    assert continued.diagnostics["minimization"]["iterations"] == 0
    assert continued.diagnostics["warm_start"]["physical_success"]
    assert continued.diagnostics["root_iterations_total"] <= 500
    assert continued.diagnostics["stationarity_residual_RT"] <= 1e-7
    assert continued.diagnostics["scaled_minimum_curvature"] >= -1e-7
    assert continued.diagnostics["first_integral_variation_relative"] <= 0.05


def test_cubic_invalid_initial_profile_is_not_a_penalized_physical_solution():
    model, states, bounds, influence = covolume_trial_case()
    mesh = DiffusionMesh.planar(12e-9, 8)
    initial = np.tile(np.array(bounds)[:, 1], (mesh.cells + 1, 1))
    initial[0], initial[-1] = (np.array(s.mole_fractions) * s.molar_density_mol_m3 for s in states)
    result = molecular_planar_interface(
        model,
        *states,
        influence,
        mesh=mesh,
        component_density_bounds=bounds,
        initial_profile=initial,
    )
    assert not result.success
    assert "covolume pole" in result.failure_reason
    assert result.diagnostics["minimization_domain_rejections"]["count"] == 0
    assert "minimization" not in result.diagnostics


def test_small_residual_mesh_saddle_still_searches_for_a_translation_minimum():
    model, _, _, influence = covolume_trial_case()
    states = (
        molecular_eos_state(
            model, 333.15, 2.9672647953666864, (0.8489509429409229, 0.1510490570590772)
        ),
        molecular_eos_state(model, 333.15, 4947.462664759153, (0.248, 0.752)),
    )
    density = np.array([np.array(s.mole_fractions) * s.molar_density_mol_m3 for s in states])
    bounds = tuple(
        (float(0.01 * min(density[:, i])), float(1.5 * max(density[:, i]))) for i in range(2)
    )
    result = molecular_planar_interface(
        model,
        *states,
        influence,
        mesh=DiffusionMesh.planar(12e-9, 256),
        component_density_bounds=bounds,
    )
    assert result.success, result.to_dict()
    assert result.diagnostics["translation_initialization"]["stationary_candidates"] > 0
    assert result.diagnostics["scaled_minimum_curvature"] > 0
    assert result.diagnostics["stationarity_residual_RT"] <= 1e-7
    assert result.diagnostics["first_integral_variation_relative"] <= 0.05
    assert result.diagnostics["root_iterations_total"] <= 500


def test_unqualified_supplied_profile_falls_back_within_original_root_budget():
    model = QuarticModel()
    states = states_for(model, model.left, model.right)
    influence = SquareGradientInfluence(
        model.component_ids, ((1e-18,),), (200.0, 400.0), "small-budget fallback control"
    )
    result = molecular_planar_interface(
        model,
        *states,
        influence,
        mesh=DiffusionMesh.planar(40e-9, 64),
        component_density_bounds=((100.0, 8000.0),),
        initial_profile=np.linspace(model.left, model.right, 65),
        max_iterations=2,
    )
    assert not result.success
    assert not result.diagnostics["warm_start"]["root_success"]
    assert result.diagnostics["warm_start"]["root_iterations_total"] > 0
    assert result.diagnostics["minimization"]["iterations"] <= 2
    assert result.diagnostics["root_iterations_total"] <= 2
