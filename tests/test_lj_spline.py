"""LJ/s controls, SI semantics, cutoff and one-sided derivative limits."""

import json
from dataclasses import replace
from math import isfinite, pi

import pytest

from chemworld.physchem.caloric_math import derivative_values, partial_derivative
from chemworld.physchem.lj_spline import (
    SPLINE_CUTOFF,
    SPLINE_START,
    LJSplineBHModel,
    LJSplineUVModel,
    LJSplineWCAModel,
    _cavity,
    lj_spline_potential,
    wca_diameter_ratio,
)
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL

CLASSES = (LJSplineBHModel, LJSplineWCAModel, LJSplineUVModel)


def make(cls, **kwargs):
    return cls("synthetic", 3.42e-10, 124.0, (50.0, 2000.0), "Explicit test parameters", **kwargs)


@pytest.mark.parametrize("cls", CLASSES)
def test_codec_scaling_and_actual_common_inverse(cls):
    base = make(cls)
    assert cls.from_dict(json.loads(json.dumps(base.to_dict()))) == base
    scaled = replace(base, diameter_m=2 * base.diameter_m, well_depth_K=2 * base.well_depth_K)
    assert base.alphar(300.0, 1000.0, (1.0,)) == pytest.approx(
        scaled.alphar(600.0, 125.0, (1.0,)), rel=1e-13
    )
    state = molecular_eos_state(base, 496.0, 1000.0, (1.0,))
    root = molecular_density_from_pressure(
        base,
        496.0,
        state.pressure_Pa,
        (1.0,),
        initial_density_mol_m3=1100.0,
        density_bounds_mol_m3=(500.0, 1500.0),
    )
    assert root.molar_density_mol_m3 == pytest.approx(1000.0, rel=1e-10)


@pytest.mark.parametrize("cls", CLASSES)
def test_domains_and_vacuum_temperature_derivatives(cls):
    base = make(cls)
    assert base.alphar(300.0, 0.0, (1.0,)) == pytest.approx(0.0, abs=1e-15)
    for i in range(1, 4):
        assert (
            molecular_scaled_derivative(base, 300.0, 0.0, (1.0,), inverse_temperature_order=i)
            == 0.0
        )
    for t, rho, x in (
        (0.0, 1.0, (1.0,)),
        (300.0, -1.0, (1.0,)),
        (300.0, 1.0, (0.5, 0.5)),
        (300.0, 1e10, (1.0,)),
    ):
        with pytest.raises(ValueError):
            base.alphar(t, rho, x)


@pytest.mark.parametrize("cls", (LJSplineBHModel, LJSplineWCAModel))
def test_additive_masks_and_all_off_ideal(cls):
    base = make(cls)
    flags = ["enable_hs", "enable_a1", "enable_a2", "enable_a3"]
    if cls is LJSplineWCAModel:
        flags += ["enable_a4"]
    off = replace(base, **dict.fromkeys(flags, False))
    state = molecular_eos_state(off, 300.0, 1000.0, (1.0,))
    assert state.compressibility_factor == 1.0
    assert state.residual_helmholtz_J_mol == 0.0
    total = sum(replace(off, **{name: True}).alphar(300.0, 1000.0, (1.0,)) for name in flags)
    assert total == pytest.approx(base.alphar(300.0, 1000.0, (1.0,)), rel=2e-14)
    with pytest.raises(ValueError, match="boolean"):
        replace(base, enable_hs=1)


def test_cavity_disabled_with_hard_sphere_and_independent_option_effects():
    base = make(LJSplineWCAModel)
    off = replace(base, enable_hs=False)
    assert off.alphar(300.0, 1000.0, (1.0,)) == replace(off, enable_cavity=False).alphar(
        300.0, 1000.0, (1.0,)
    )
    assert base.alphar(300.0, 1000.0, (1.0,)) != replace(base, enable_cavity=False).alphar(
        300.0, 1000.0, (1.0,)
    )
    bh = make(LJSplineBHModel)
    assert bh.alphar(300.0, 1000.0, (1.0,)) != replace(bh, use_lafitte_a3=True).alphar(
        300.0, 1000.0, (1.0,)
    )
    assert bh.alphar(300.0, 1000.0, (1.0,)) != replace(bh, enable_chi_correction=False).alphar(
        300.0, 1000.0, (1.0,)
    )
    uv = make(LJSplineUVModel)
    assert uv.alphar(300.0, 1000.0, (1.0,)) != replace(
        uv, use_temperature_dependent_u_fraction=True
    ).alphar(300.0, 1000.0, (1.0,))


def test_potential_cutoff_and_c1_matching_for_unsorted_individual_radii():
    assert lj_spline_potential(1.0) == 0.0
    assert lj_spline_potential(2 ** (1 / 6)) == pytest.approx(-1.0)
    assert [lj_spline_potential(x) for x in (2.0, 1.0, SPLINE_CUTOFF)] == [0.0, 0.0, 0.0]
    for join in (SPLINE_START, SPLINE_CUTOFF):
        left = derivative_values(lj_spline_potential, join - 1e-8, order=1)
        right = derivative_values(lj_spline_potential, join + 1e-8, order=1)
        assert left == pytest.approx(right, abs=1e-6)
    for radius in (0.0, -1.0, float("inf"), True):
        with pytest.raises(ValueError):
            lj_spline_potential(radius)


def test_cavity_continuation_c2_and_shared_jet_mixed_derivatives():
    left = derivative_values(lambda eta: _cavity(eta, 0.8), 0.52 - 1e-10, order=2)
    right = derivative_values(lambda eta: _cavity(eta, 0.8), 0.52 + 1e-10, order=2)
    assert left == pytest.approx(right, rel=1e-7)
    third = derivative_values(lambda eta: _cavity(eta, 0.8), 0.6, order=3)[3]
    assert third == pytest.approx(0.0, abs=1e-14)
    base = make(LJSplineWCAModel)
    values = {"T": 300.0, "rho": 1000.0}
    a = partial_derivative(
        lambda v: base.alphar(v["T"], v["rho"], (1.0,)), values, ("T", "rho", "T")
    )
    b = partial_derivative(
        lambda v: base.alphar(v["T"], v["rho"], (1.0,)), values, ("rho", "T", "T")
    )
    assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("cls", CLASSES)
def test_finite_first_vacuum_density_limit_including_mixed_temperature(cls):
    base = make(cls)
    derivative = derivative_values(lambda rho: base.alphar(300.0, rho, (1.0,)), 0.0, order=1)[1]
    assert isfinite(derivative)
    # The right-side linear limit is mathematically defined, even though
    # individual fractional-power factors have singular derivatives.
    assert base.alphar(300.0, 1e-5, (1.0,)) / 1e-5 == pytest.approx(derivative, rel=3e-5)
    mixed = partial_derivative(
        lambda v: base.alphar(v["T"], v["rho"], (1.0,)), {"T": 300.0, "rho": 0.0}, ("T", "T", "rho")
    )
    assert isfinite(mixed)


def test_actual_nonanalytic_vacuum_rejections():
    wca = make(LJSplineWCAModel)
    with pytest.raises(ValueError, match="vacuum second"):
        derivative_values(lambda rho: wca.alphar(300.0, rho, (1.0,)), 0.0, order=2)
    polynomial = replace(wca, enable_a2=False, enable_a3=False, enable_a4=False)
    assert all(
        isfinite(x)
        for x in derivative_values(lambda r: polynomial.alphar(300.0, r, (1.0,)), 0.0, order=3)
    )
    for flag in (False, True):
        uv = make(LJSplineUVModel, use_temperature_dependent_u_fraction=flag)
        assert all(
            isfinite(x)
            for x in derivative_values(lambda r, uv=uv: uv.alphar(300.0, r, (1.0,)), 0.0, order=2)
        )
        with pytest.raises(ValueError, match="vacuum third"):
            derivative_values(lambda r, uv=uv: uv.alphar(300.0, r, (1.0,)), 0.0, order=3)


def test_cavity_second_derivative_consumer_at_piecewise_boundary():
    base = make(LJSplineWCAModel)
    t = 248.0
    eta_per_rho = pi * wca_diameter_ratio(t / base.well_depth_K) ** 3 / 6
    rho = 0.52 / (eta_per_rho * AVOGADRO_PER_MOL * base.diameter_m**3)
    # State consumers request second derivatives, which remain continuous.
    assert isfinite(molecular_eos_state(base, t, rho, (1.0,), require_stable=False).pressure_Pa)
