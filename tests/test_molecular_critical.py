"""Analytic critical limits, full composition derivatives and failure semantics."""

from dataclasses import dataclass
from math import log

import numpy as np
import pytest

from chemworld.physchem.caloric_math import logarithm
from chemworld.physchem.molecular_critical import (
    molecular_composition_response,
    solve_molecular_critical,
    trace_molecular_critical,
)
from chemworld.physchem.residual_system import RootPolicy


@dataclass(frozen=True)
class VDW:
    component_ids: tuple[str, ...] = ("a",)
    gas_constant_J_mol_K: float = 8.31446261815324
    a: float = 0.36
    b: float = 4.2e-5

    def alphar(self, temperature, density, fractions):
        return -logarithm(1 - self.b * density) - self.a * density / (
            self.gas_constant_J_mol_K * temperature
        )

    @property
    def tc(self):
        return 8 * self.a / (27 * self.gas_constant_J_mol_K * self.b)

    @property
    def rhoc(self):
        return 1 / (3 * self.b)


def solve(model, x, **kwargs):
    return solve_molecular_critical(
        model,
        x,
        initial_temperature_K=model.tc * 1.04,
        initial_density_mol_m3=model.rhoc * 0.95,
        temperature_bounds_K=(model.tc * 0.7, model.tc * 1.5),
        density_bounds_mol_m3=(model.rhoc * 0.3, model.rhoc * 2),
        **kwargs,
    )


@pytest.mark.parametrize(
    "x", [(1.0,), (0.3, 0.7), (0.2, 0.3, 0.5), (0.1, 0.2, 0.3, 0.4), (0.0, 0.4, 0.6, 0.0)]
)
def test_analytic_critical_point_and_exact_active_face(x):
    model = VDW(tuple("abcd"[: len(x)]))
    result = solve(model, x)
    assert result.success, result.to_dict()
    r = result.response
    assert r["temperature_K"] == pytest.approx(model.tc, rel=1e-8)
    assert r["molar_density_mol_m3"] == pytest.approx(model.rhoc, rel=1e-7)
    assert r["pressure_Pa"] == pytest.approx(model.a / (27 * model.b**2), rel=1e-8)
    assert abs(r["scaled_eigenvalues"][0]) < 2e-9
    assert abs(r["third_mode_derivative_scaled"]) < 2e-9
    assert r["relaxed_fourth_derivative_scaled"] > 0
    assert r["density_susceptibility_mol2_J_m3"] is None


def test_fixed_pressure_response_and_gibbs_duhem():
    model, x = VDW(("a", "b", "c")), np.array([0.2, 0.3, 0.5])
    t, rho = 400.0, 500.0
    r = molecular_composition_response(model, t, rho * x)
    # Identical intermolecular interactions have exact ideal composition response
    # at fixed P, despite their nonideal pressure and density susceptibility.
    ideal = np.diag(1 / x[:-1]) + np.full((2, 2), 1 / x[-1])
    assert np.allclose(
        r["simplex_fixed_TP_mu_difference_J_mol"],
        model.gas_constant_J_mol_K * t * ideal,
        rtol=1e-11,
    )
    assert np.max(np.abs(r["fixed_TP_density_response_mol_m3"])) < 1e-10
    h = np.array(r["density_hessian_J_m3_mol2"])
    assert np.allclose(h @ (rho * x), r["dP_dcomponent_density_Pa_m3_mol"], rtol=1e-12)
    assert np.allclose(h @ np.array(r["density_susceptibility_mol2_J_m3"]), np.eye(3), atol=1e-12)


def test_trace_retains_declared_path_and_critical_inputs():
    model = VDW(("a", "b"))
    path = [(z, 1 - z) for z in (0.2, 0.3, 0.4, 0.5, 0.6)]
    result = trace_molecular_critical(
        model,
        path,
        initial_temperature_K=model.tc * 1.04,
        initial_density_mol_m3=model.rhoc * 0.95,
        temperature_bounds_K=(model.tc * 0.7, model.tc * 1.5),
        density_bounds_mol_m3=(model.rhoc * 0.3, model.rhoc * 2),
    )
    assert result["success"] and result["completed_stations"] == 5
    assert [r["mole_fractions"] for r in result["stations"]] == [list(x) for x in path]
    assert (
        max(
            abs(r["result"]["response"]["temperature_K"] / model.tc - 1) for r in result["stations"]
        )
        < 1e-8
    )


def test_budget_and_invalid_composition_are_not_success():
    model = VDW()
    result = solve(model, (1.0,), policy=RootPolicy(1e-14, 1))
    assert not result.success
    assert result.diagnostics["actual_condition_evaluations"] > 1
    with pytest.raises(ValueError, match="sum to one"):
        solve(model, (0.5,))
    for x in ((-1.0,), (0.0,), (float("nan"),)):
        with pytest.raises(ValueError):
            solve(model, x)


@dataclass(frozen=True)
class Quartic:
    coupling: float
    transverse: float = 1.0
    component_ids: tuple[str, ...] = ("a", "b")
    gas_constant_J_mol_K: float = 1.0

    def alphar(self, temperature, density, fractions):
        c = [density * x for x in fractions]
        s, y = c[0] - 1, c[1] - 1
        # Exact total f/(RT): local soft mode with cubic transverse coupling.
        f = s**4 / 24 + self.transverse * y * y / 2 + self.coupling * s * s * y / 2
        ideal = sum(v * (logarithm(v) - 1) for v in c)
        # Positive constant shifts pressure negative; use negative reference.
        return (f - ideal - 1) / density


@pytest.mark.parametrize("coupling, expected", [(0.0, 8.0), (1.0, -16.0)])
def test_relaxed_fourth_not_just_directional_fourth(coupling, expected):
    response = molecular_composition_response(Quartic(coupling), 1.0, (1.0, 1.0))
    assert response["fourth_mode_derivative_scaled"] == pytest.approx(8.0)
    assert response["relaxed_fourth_derivative_scaled"] == pytest.approx(expected)
    result = solve_molecular_critical(
        Quartic(coupling),
        (0.5, 0.5),
        initial_temperature_K=1.0,
        initial_density_mol_m3=2.0,
        temperature_bounds_K=(0.5, 2.0),
        density_bounds_mol_m3=(1.0, 3.0),
    )
    assert result.success == (coupling == 0)
    if coupling:
        assert result.failure_reason == "nonpositive_relaxed_fourth_order"


def test_degenerate_multiple_zero_modes_remain_failure():
    result = solve_molecular_critical(
        Quartic(0.0, transverse=0.0),
        (0.5, 0.5),
        initial_temperature_K=1.0,
        initial_density_mol_m3=2.0,
        temperature_bounds_K=(0.5, 2.0),
        density_bounds_mol_m3=(1.0, 3.0),
    )
    assert not result.success
    assert result.failure_reason == "degenerate_or_unstable_transverse_mode"


def test_component_permutation_is_a_change_of_coordinates():
    model = VDW(("a", "b", "c"))
    x, order = np.array([0.1, 0.3, 0.6]), [2, 0, 1]
    a = molecular_composition_response(model, 400.0, 500 * x)
    b = molecular_composition_response(
        VDW(tuple(model.component_ids[i] for i in order)), 400.0, 500 * x[order]
    )
    assert np.allclose(
        np.array(a["density_hessian_J_m3_mol2"])[np.ix_(order, order)],
        b["density_hessian_J_m3_mol2"],
        rtol=1e-12,
    )
    assert a["pressure_Pa"] == pytest.approx(b["pressure_Pa"], rel=1e-12)
    assert log(a["pressure_Pa"] / b["pressure_Pa"]) == pytest.approx(0.0, abs=1e-12)


def test_optional_pressure_validator_applies_without_a_fluids_attribute():
    class Limited(VDW):
        def validate_pressure(self, pressure):
            if pressure > 1.0:
                raise ValueError("declared wrapper pressure ceiling")

    with pytest.raises(ValueError, match="wrapper pressure ceiling"):
        molecular_composition_response(Limited(), 400.0, (500.0,))
    result = solve(Limited(), (1.0,))
    assert not result.success and "wrapper pressure ceiling" in result.failure_reason
