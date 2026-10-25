"""Analytic coexistence, real cubic equations, fold continuation and refusal."""

import json
from dataclasses import dataclass, replace
from math import log

import numpy as np
import pytest
from tests.test_molecular_flash import cubic_model

from chemworld.physchem.caloric_math import logarithm
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_phase_boundary import (
    MolecularBoundaryResult,
    MolecularBoundarySeed,
    MolecularEnvelopeResult,
    MolecularPhaseBoundarySpec,
    solve_molecular_azeotrope,
    solve_molecular_phase_boundary,
    trace_molecular_phase_envelope,
)
from chemworld.physchem.residual_system import RootPolicy


@dataclass(frozen=True)
class AnalyticCoexistence:
    """A(v)=10(u-1)^2(u-5)^2-Psat(T)v+20u(x-.35)^2, u=v/.001.

    The residual removes the ideal density term. At pure conditions or the
    binary azeotrope x=.35, equal-G phases have rho=1000/200 and P=Psat.
    Psat has a pressure minimum at 300 K. Positive quartic curvature makes
    both endpoints mechanically stable; ideal mixing makes the binary stable.
    """

    component_ids: tuple[str, ...] = ("a",)
    gas_constant_J_mol_K: float = 8.31446261815324
    composition_coupling: float = 20.0

    def alphar(self, temperature, density, fractions):
        u = 1000 / density
        psat = 1e5 + 10 * (temperature - 300) ** 2
        energy = 10 * (u - 1) ** 2 * (u - 5) ** 2 - psat / density
        if len(fractions) > 1:
            energy += self.composition_coupling * u * (fractions[0] - 0.35) ** 2
        return energy / (self.gas_constant_J_mol_K * temperature) - logarithm(density)


def analytic_spec(binary=False, mode="local_coexistence"):
    return MolecularPhaseBoundarySpec(
        AnalyticCoexistence(("a", "b") if binary else ("a",)),
        (0.35, 0.65) if binary else (1.0,),
        "bubble",
        MolecularDensityBasin("liquid", (800.0, 1250.0), 1000.0),
        MolecularDensityBasin("vapor", (185.0, 220.0), 200.0),
        (250.0, 350.0),
        (8e4, 2e5),
        stability_mode=mode,
    )


@pytest.mark.parametrize("kind", ("bubble", "dew"))
@pytest.mark.parametrize("coordinate", ("temperature", "pressure"))
def test_pure_coexistence_matches_independent_polynomial_endpoints(kind, coordinate):
    spec = replace(analytic_spec(), kind=kind)
    seed = MolecularBoundarySeed(279.0, 1010.0, 198.0, (1.0,))
    target = {"temperature_K": 280.0} if coordinate == "temperature" else {"pressure_Pa": 104000.0}
    result = solve_molecular_phase_boundary(spec, seed, **target)
    assert result.success, result.to_dict()
    assert result.liquid.temperature_K == pytest.approx(280.0, abs=2e-6)
    assert result.liquid.molar_density_mol_m3 == pytest.approx(1000.0, rel=2e-7)
    assert result.vapor.molar_density_mol_m3 == pytest.approx(200.0, rel=2e-7)
    assert result.liquid.pressure_Pa == pytest.approx(104000.0, rel=2e-8)
    assert result.chemical_potential_error_RT < 1e-8


def test_pseudoarclength_crosses_pressure_turn_and_reverse_traverses_same_branch():
    spec = analytic_spec()
    initial = solve_molecular_phase_boundary(
        spec, MolecularBoundarySeed(280.0, 1000.0, 200.0, (1.0,)), temperature_K=280.0
    )
    curve = trace_molecular_phase_envelope(spec, initial, steps=12, arclength_step=0.012)
    assert curve.success and curve.completed_steps == 12, curve.to_dict()
    temps = [p.liquid.temperature_K for p in curve.points]
    pressures = [p.liquid.pressure_Pa for p in curve.points]
    assert min(temps) < 300 < max(temps)
    assert min(pressures) < min(pressures[0], pressures[-1])
    assert any(t["coordinate"] == "pressure_Pa" for t in curve.turning_intervals)
    for point in curve.points:
        assert point.liquid.pressure_Pa == pytest.approx(
            1e5 + 10 * (point.liquid.temperature_K - 300) ** 2, rel=2e-9
        )
    for point in curve.points[1:]:
        assert abs(point.diagnostics["arclength_residual"]) < 1e-9
    backward = trace_molecular_phase_envelope(
        spec, curve.points[-1], steps=12, arclength_step=0.012, direction=-1
    )
    assert backward.success
    assert backward.points[-1].liquid.temperature_K == pytest.approx(280.0, abs=2e-5)
    restored = MolecularEnvelopeResult.from_dict(
        json.loads(json.dumps(curve.to_dict(), allow_nan=False))
    )
    assert restored.to_dict() == curve.to_dict()


def test_isolated_binary_azeotrope_closes_composition_and_pressure():
    spec = analytic_spec(binary=True)
    result = solve_molecular_azeotrope(
        spec, MolecularBoundarySeed(280.0, 1005.0, 199.0, (0.4, 0.6)), temperature_K=280.0
    )
    assert result.success, result.to_dict()
    assert result.liquid.mole_fractions == pytest.approx((0.35, 0.65), abs=1e-7)
    assert result.vapor.mole_fractions == result.liquid.mole_fractions
    assert result.liquid.molar_density_mol_m3 == pytest.approx(1000.0, rel=2e-7)
    assert result.vapor.molar_density_mol_m3 == pytest.approx(200.0, rel=2e-7)
    assert result.phase_separation == pytest.approx(log(5), abs=2e-7)
    assert result.kind == "azeotrope"


def test_azeotropic_continuum_is_not_an_isolated_solution():
    spec = analytic_spec(binary=True)
    spec = replace(spec, model=replace(spec.model, composition_coupling=0.0))
    result = solve_molecular_azeotrope(
        spec, MolecularBoundarySeed(280.0, 1000.0, 200.0, (0.35, 0.65)), temperature_K=280.0
    )
    assert not result.success and result.failure_reason == "nonisolated_or_degenerate_azeotrope"


def test_finite_tpd_verification_and_failed_search_remain_distinct():
    spec = analytic_spec(mode="finite_tpd")
    seed = MolecularBoundarySeed(280.0, 1000.0, 200.0, (1.0,))
    result = solve_molecular_phase_boundary(spec, seed, temperature_K=280.0)
    assert result.success, result.to_dict()
    assert len(result.stability_reports) == 2
    assert all(r["complete"] and not r["unstable"] for r in result.stability_reports)
    binary = analytic_spec(binary=True, mode="finite_tpd")
    exhausted = solve_molecular_phase_boundary(
        replace(binary, stability_max_iterations=1),
        MolecularBoundarySeed(280.0, 1000.0, 200.0, (0.35, 0.65)),
        temperature_K=280.0,
    )
    assert not exhausted.success
    assert exhausted.failure_reason == "remaining_or_unresolved_boundary_stability"
    assert exhausted.liquid is not None  # Diagnostic candidate is not an accepted equilibrium.


def test_real_peng_robinson_bubble_dew_and_pressure_roundtrip():
    spec = MolecularPhaseBoundarySpec(
        cubic_model(),
        (0.5, 0.5),
        "bubble",
        MolecularDensityBasin("liquid", (7500.0, 13500.0), 10000.0),
        MolecularDensityBasin("vapor", (1.0, 2000.0), 250.0),
        (250.0, 340.0),
        (1e4, 2e6),
        stability_mode="local_coexistence",
    )
    bubble = solve_molecular_phase_boundary(
        spec, MolecularBoundarySeed(300.0, 10800.0, 250.0, (0.8, 0.2)), temperature_K=300.0
    )
    assert bubble.success, bubble.to_dict()
    assert bubble.vapor.mole_fractions[0] > 0.5
    dew_spec = replace(spec, kind="dew", fixed_composition=bubble.vapor.mole_fractions)
    dew = solve_molecular_phase_boundary(
        dew_spec,
        MolecularBoundarySeed(
            299.0, bubble.liquid.molar_density_mol_m3, bubble.vapor.molar_density_mol_m3, (0.5, 0.5)
        ),
        pressure_Pa=bubble.liquid.pressure_Pa,
    )
    assert dew.success, dew.to_dict()
    assert dew.vapor.temperature_K == pytest.approx(300.0, abs=1e-5)
    assert dew.liquid.mole_fractions == pytest.approx((0.5, 0.5), abs=1e-7)
    restored_spec = MolecularPhaseBoundarySpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored_spec.to_dict() == spec.to_dict()
    restored = MolecularBoundaryResult.from_dict(
        json.loads(json.dumps(bubble.to_dict(), allow_nan=False))
    )
    assert restored.to_dict() == bubble.to_dict()


def test_failed_continuation_preserves_points_and_attempts_without_bridge():
    spec = analytic_spec()
    initial = solve_molecular_phase_boundary(
        spec, MolecularBoundarySeed(349.0, 1000.0, 200.0, (1.0,)), temperature_K=349.0
    )
    result = trace_molecular_phase_envelope(
        spec, initial, steps=4, arclength_step=0.1, maximum_step_reductions=1
    )
    assert not result.success and result.completed_steps == 0
    assert len(result.points) == 1 and len(result.attempts) == 2
    assert all(not a["success"] for a in result.attempts)
    assert result.termination_reason.startswith("continuation_unresolved")


def test_explicit_domain_failures_and_strict_json_fields():
    spec = analytic_spec()
    seed = MolecularBoundarySeed(280.0, 1000.0, 200.0, (1.0,))
    for kwargs in ({}, {"temperature_K": 280.0, "pressure_Pa": 1e5}, {"temperature_K": 200.0}):
        with pytest.raises(ValueError):
            solve_molecular_phase_boundary(spec, seed, **kwargs)
    bad = solve_molecular_phase_boundary(
        spec, replace(seed, liquid_density_mol_m3=2000.0), temperature_K=280.0
    )
    assert not bad.success and bad.failure_reason == "initial_outside_bounds"
    no_budget = solve_molecular_phase_boundary(
        spec,
        replace(seed, liquid_density_mol_m3=900.0),
        temperature_K=280.0,
        policy=RootPolicy(maximum_iterations=1),
    )
    assert not no_budget.success
    with pytest.raises(ValueError, match="exactly"):
        MolecularBoundarySeed.from_dict({**seed.to_dict(), "unknown": 1})
    with pytest.raises(ValueError, match="finite"):
        MolecularBoundaryResult(False, "bubble", coordinates=(float("nan"),))
    with pytest.raises(ValueError, match="two active"):
        solve_molecular_azeotrope(spec, seed, temperature_K=280.0)


def test_exact_absent_support_is_preserved_and_positive_trace_is_not_dropped():
    model = AnalyticCoexistence(("a", "b", "c"), composition_coupling=0.0)
    for trace in (0.0, 1e-15):
        xs = (0.35, 0.65 - trace, trace)
        spec = replace(analytic_spec(binary=True), model=model, fixed_composition=xs)
        point = solve_molecular_phase_boundary(
            spec, MolecularBoundarySeed(280.0, 1000.0, 200.0, xs), temperature_K=280.0
        )
        assert point.success, point.to_dict()
        assert point.liquid.mole_fractions[2] == trace
        assert (point.vapor.mole_fractions[2] > 0) == (trace > 0)
        assert np.allclose(point.vapor.mole_fractions, xs, atol=0.0, rtol=1e-13)


@dataclass(frozen=True)
class AnalyticVDW:
    component_ids: tuple[str, ...] = ("a",)
    gas_constant_J_mol_K: float = 8.31446261815324

    def alphar(self, temperature, density, fractions):
        attraction = 27 * self.gas_constant_J_mol_K * 1e-4 * 300 / 8
        return -logarithm(1 - 1e-4 * density) - attraction * density / (
            self.gas_constant_J_mol_K * temperature
        )


def test_near_critical_endpoint_uses_hk_verification_and_trivial_root_is_rejected():
    spec = MolecularPhaseBoundarySpec(
        AnalyticVDW(),
        (1.0,),
        "bubble",
        MolecularDensityBasin("liquid", (1000.0, 7000.0), 4000.0),
        MolecularDensityBasin("vapor", (1000.0, 7000.0), 3000.0),
        (250.0, 350.0),
        (1e5, 1e7),
        stability_mode="local_coexistence",
    )
    point = solve_molecular_phase_boundary(
        spec, MolecularBoundarySeed(299.99, 3372.0, 3295.0, (1.0,)), temperature_K=299.99
    )
    assert point.success, point.to_dict()
    curve = trace_molecular_phase_envelope(
        spec, point, steps=5, arclength_step=0.01, critical_separation=0.05
    )
    assert curve.success and curve.completed_steps == 0, curve.to_dict()
    assert curve.termination_reason == "near_critical_verified"
    response = curve.critical_result["response"]
    assert response["temperature_K"] == pytest.approx(300.0, abs=2e-5)
    assert response["molar_density_mol_m3"] == pytest.approx(1 / (3e-4), rel=2e-6)
    trivial = solve_molecular_phase_boundary(
        spec, MolecularBoundarySeed(320.0, 3000.0, 3000.0, (1.0,)), temperature_K=320.0
    )
    assert not trivial.success and trivial.failure_reason == "trivial_or_merged_phases"
