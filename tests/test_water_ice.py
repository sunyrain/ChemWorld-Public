import json
import math

import pytest

from chemworld.physchem import (
    ice_ih_gibbs_derivatives,
    ice_ih_state,
    ice_sublimation_pressure,
    ice_sublimation_temperature,
    water_melting_pressure,
    water_melting_temperature,
)
from chemworld.physchem.second_order_pair import SecondOrderPair


@pytest.mark.parametrize("T,p", [(50, 1000), (100, 100e6), (230, 5e6), (273.15, 101325)])
def test_same_gibbs_thermodynamics_and_derivatives(T, p):
    g = ice_ih_gibbs_derivatives(T, p)
    state = ice_ih_state(T, p)
    assert state.enthalpy_J_kg == pytest.approx(
        state.internal_energy_J_kg+p/state.mass_density_kg_m3)
    assert state.gibbs_energy_J_kg == pytest.approx(state.enthalpy_J_kg-T*state.entropy_J_kg_K)
    assert state.helmholtz_energy_J_kg == pytest.approx(
        state.internal_energy_J_kg-T*state.entropy_J_kg_K)
    assert state.cp_J_kg_K-state.cv_J_kg_K == pytest.approx(
        T*state.thermal_expansion_K_inv**2/state.mass_density_kg_m3
        / state.isothermal_compressibility_Pa_inv)
    assert state.cp_J_kg_K/state.cv_J_kg_K == pytest.approx(
        state.isothermal_compressibility_Pa_inv/state.isentropic_compressibility_Pa_inv)
    lo_t, hi_t = ice_ih_gibbs_derivatives(T-.001, p), ice_ih_gibbs_derivatives(T+.001, p)
    lo_p, hi_p = ice_ih_gibbs_derivatives(T, p-100), ice_ih_gibbs_derivatives(T, p+100)
    for analytic, numerical in (
        (g.dx, (hi_p.value-lo_p.value)/200), (g.dy, (hi_t.value-lo_t.value)/.002),
        (g.dxx, (hi_p.dx-lo_p.dx)/200), (g.dyy, (hi_t.dy-lo_t.dy)/.002),
        (g.dxy, (hi_t.dx-lo_t.dx)/.002), (g.dxy, (hi_p.dy-lo_p.dy)/200),
    ):
        assert analytic == pytest.approx(numerical, rel=3e-6, abs=0)
    report = json.loads(json.dumps(state.to_dict(), allow_nan=False))
    assert report["stability_scope"] == "homogeneous_ice_Ih_only"


@pytest.mark.parametrize("T", [0, .001, 1, 100, 273.16])
def test_entropy_reference_only_changes_entropy_and_free_energies(T):
    conventional = ice_ih_state(T, 101325)
    absolute = ice_ih_state(T, 101325, entropy_reference="absolute")
    change = .18913e3 + .332733756492168e4
    assert absolute.entropy_J_kg_K-conventional.entropy_J_kg_K == pytest.approx(change)
    assert absolute.gibbs_energy_J_kg-conventional.gibbs_energy_J_kg == pytest.approx(-T*change)
    for name in ("enthalpy_J_kg", "internal_energy_J_kg", "cp_J_kg_K", "cv_J_kg_K",
                 "mass_density_kg_m3", "isothermal_compressibility_Pa_inv"):
        assert getattr(absolute, name) == pytest.approx(getattr(conventional, name), rel=2e-14)


def test_zero_temperature_limit_and_debye_law():
    zero = ice_ih_state(0, 101325, entropy_reference="absolute")
    assert zero.entropy_J_kg_K == .18913e3
    assert zero.gibbs_energy_J_kg == -.632020233335886e6
    assert zero.cp_J_kg_K == zero.cv_J_kg_K == zero.thermal_expansion_K_inv == 0
    assert zero.isentropic_compressibility_Pa_inv == zero.isothermal_compressibility_Pa_inv
    low = ice_ih_state(.001, 101325)
    high = ice_ih_state(.002, 101325)
    assert high.cp_J_kg_K/low.cp_J_kg_K == pytest.approx(8, rel=1e-6)


@pytest.mark.parametrize("T", [50, 75, 130, 230, 273.15, 273.16])
def test_sublimation_boundary_roundtrip(T):
    p = ice_sublimation_pressure(T)
    assert p > 0
    assert ice_sublimation_temperature(p) == pytest.approx(T, abs=2e-8, rel=0)


@pytest.mark.parametrize("phase,T", [("Ih", 260), ("III", 254), ("V", 265),
                                   ("VI", 320), ("VII", 550)])
def test_melting_boundary_derivatives_and_inverse(phase, T):
    p = water_melting_pressure(SecondOrderPair(T, dx=1), ice_phase=phase)
    left = water_melting_pressure(T-.001, ice_phase=phase)
    right = water_melting_pressure(T+.001, ice_phase=phase)
    assert p.dx == pytest.approx((right-left)/.002, rel=1e-7)
    assert p.dx < 0 if phase == "Ih" else p.dx > 0
    assert water_melting_temperature(p.value, ice_phase=phase) == pytest.approx(T, abs=2e-8)


def test_sublimation_derivatives_and_triple_point():
    T = 230
    point = ice_sublimation_pressure(SecondOrderPair(T, dx=1))
    lo = ice_sublimation_pressure(SecondOrderPair(T-.001, dx=1))
    hi = ice_sublimation_pressure(SecondOrderPair(T+.001, dx=1))
    assert point.dx == pytest.approx((hi.value-lo.value)/.002, rel=1e-7)
    assert point.dxx == pytest.approx((hi.dx-lo.dx)/.002, rel=1e-7)
    assert ice_sublimation_pressure(273.16) == pytest.approx(611.657, rel=1e-14)
    assert water_melting_pressure(273.16) == 611.657


@pytest.mark.parametrize("T,p", [(True, 1000), (math.nan, 1000), (-1, 1000),
                               (273.16001, 1000), (100, True), (100, math.inf),
                               (100, -1), (100, 210e6+1)])
def test_ice_domain_refusal(T, p):
    with pytest.raises(ValueError):
        ice_ih_state(T, p)


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, 0, 49.99, 273.16001])
def test_sublimation_domain_refusal(bad):
    with pytest.raises(ValueError):
        ice_sublimation_pressure(bad)


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, 0, -1, 612])
def test_sublimation_pressure_refusal(bad):
    with pytest.raises(ValueError):
        ice_sublimation_temperature(bad)


def test_unknown_phase_reference_and_melting_range_refusal():
    with pytest.raises(ValueError, match="entropy_reference"):
        ice_ih_state(100, 1000, entropy_reference="invented")
    for function in (water_melting_pressure, water_melting_temperature):
        with pytest.raises(ValueError, match="ice_phase"):
            function(260, ice_phase="II")
    with pytest.raises(ValueError, match="requires"):
        water_melting_pressure(300)
    with pytest.raises(ValueError, match="outside"):
        water_melting_temperature(1)
    with pytest.raises(ValueError, match="outside"):
        ice_sublimation_temperature(ice_sublimation_pressure(50)/2)
