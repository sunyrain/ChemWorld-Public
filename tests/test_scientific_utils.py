"""Physical contracts of the shared composition and fluid-response primitives."""

from math import isfinite

import pytest

from chemworld.physchem import composition_math as c
from chemworld.physchem import fluid_response as f


def test_basis_conversions_conserve_amount_and_use_declared_reference_volume():
    n, mw, vm = (0.0, 3.0, 5.0), (18.0, 44.0, 70.0), (2e-5, 5e-5, 9e-5)
    mass = c.convert_component_flows(n, source="molar", target="mass", molecular_weights=mw)
    volume = c.convert_component_flows(
        mass,
        source="mass",
        target="reference_volume",
        molecular_weights=mw,
        reference_molar_volumes=vm,
    )
    assert mass == pytest.approx((0.0, 0.132, 0.350))
    assert volume == pytest.approx((0.0, 0.00015, 0.00045))
    assert c.convert_component_flows(
        volume, source="reference_volume", target="molar", reference_molar_volumes=vm
    ) == pytest.approx(n)
    assert c.fraction_basis(c.fraction_basis(n, mw), mw, inverse=True) == pytest.approx(
        c.normalized_fractions(n)
    )
    assert c.fraction_basis(c.fraction_basis(n, vm), vm, inverse=True) == pytest.approx(
        c.normalized_fractions(n)
    )


def test_normalization_handles_large_amounts_without_silent_regularization():
    assert c.normalized_fractions((.21, .79)) == (.21, .79)
    assert c.normalized_fractions((1e308, 1e308, 0.0)) == (0.5, 0.5, 0.0)
    original = [0.0, 2.0, 3.0]
    output = c.regularized_fractions(original, 1e-10)
    assert original == [0.0, 2.0, 3.0]
    assert sum(output) == pytest.approx(1.0) and output[0] > 0
    assert c.regularized_fractions((2.0, 3.0)) == (2.0, 3.0)


@pytest.mark.parametrize("values", [(), (0.0, 0.0), (-1.0, 2.0), (float("nan"), 1.0), (True, 2.0)])
def test_undefined_compositions_fail(values):
    with pytest.raises(ValueError):
        c.normalized_fractions(values)


def test_mixing_bounds_continuity_and_inactive_extreme_component():
    xs, values = (0.2, 0.8), (1e-20, 1e20)
    for exponent in (-100.0, -1.0, -1e-12, 0.0, 1e-12, 1.0, 100.0):
        result = c.weighted_property(xs, values, rule="power", exponent=exponent)
        assert isfinite(result) and values[0] * (1 - 1e-12) <= result <= values[1] * (1 + 1e-12)
    geometric = c.weighted_property(xs, values, rule="logarithmic")
    assert c.weighted_property(xs, values, rule="power", exponent=1e-12) == pytest.approx(
        geometric, rel=1e-8
    )
    assert c.weighted_property(
        (0.0, 1.0), (1e300, 2.0), rule="power", exponent=100.0
    ) == pytest.approx(2.0)
    assert c.weighted_property(xs, (-5.0, 10.0)) == pytest.approx(7.0)


def test_three_halves_power_mean_obeys_published_definition():
    # 0.25*1**1.5 + 0.75*9**1.5 = 20.5; the root must be 2/3.
    assert c.weighted_property((0.25, 0.75), (1.0, 9.0), rule="power", exponent=1.5) == (
        pytest.approx(20.5 ** (2 / 3))
    )


def test_component_flow_union_preserves_each_species_and_source_vectors():
    ids = (("B", "A"), ("A", "C"), ("C", "B"))
    streams = ((2.0, 3.0), (4.0, 5.0), (6.0, 7.0))
    combined, flows = c.combine_component_flows(ids, partial_flows=streams)
    assert dict(zip(combined, flows, strict=True)) == {"A": 7.0, "B": 9.0, "C": 11.0}
    assert streams == ((2.0, 3.0), (4.0, 5.0), (6.0, 7.0))
    assert c.combine_component_flows(
        (("B", "A"), ("A", "B")), partial_flows=(None, (3.0, 7.0))
    ) == (("B", "A"), (7.0, 3.0))


@pytest.mark.parametrize(
    "inputs",
    [
        ((None, None), (0.3, 0.7), (None, None), (18.0, 44.0)),
        ((1.0, None), (None, 1.0), (None, None), (18.0, 44.0)),
        ((1.0, None), (0.1, None), (None, 0.5), (18.0, 44.0)),
        ((1.0, None), (None, None), (None, None), (18.0, 44.0)),
        ((1.0, None), (None, 1.1), (None, None), (18.0, 44.0)),
    ],
)
def test_stream_specs_require_unique_complete_positive_scale(inputs):
    with pytest.raises(ValueError):
        c.solve_stream_composition(*inputs)


def test_stream_solver_scales_with_amount_but_retains_composition():
    a = c.solve_stream_composition(
        (2.0, None, None), (None, 0.3, None), (None, None, 0.6756756756756757), (18.0, 44.0, 70.0)
    )
    b = c.solve_stream_composition(
        (20.0, None, None), (None, 0.3, None), (None, None, 0.6756756756756757), (18.0, 44.0, 70.0)
    )
    assert a[0] == pytest.approx((2.0, 3.0, 5.0))
    assert b[0] == pytest.approx(tuple(10 * v for v in a[0]))
    assert b[1] == pytest.approx(a[1]) and b[2] == pytest.approx(a[2])


def test_partial_molar_euler_and_amount_scaling():
    xs, gradient = (0.7, 0.2, 0.1), (-5.0, 3.0, 9.0)
    partial = c.partial_molar_properties(13.0, gradient, xs)
    assert sum(x * p for x, p in zip(xs, partial, strict=True)) == pytest.approx(13.0)
    for n in (0.1, 1.0, 10.0):
        dn = c.composition_gradient_to_moles(gradient, xs, total_moles=n)
        assert sum(x * d for x, d in zip(xs, dn, strict=True)) == pytest.approx(0.0, abs=1e-14)
        assert c.partial_molar_from_mole_gradient(13.0, dn, total_moles=n) == pytest.approx(partial)
    assert c.constrained_composition_gradient((3.0,)) == ()
    assert c.constrained_composition_hessian(((7.0,),)) == ()


def test_ideal_gas_response_identities_and_velocity_units():
    t, p, cp, mw = 400.0, 2e5, 30.0, 28.97
    v, cv = f.R * t / p, cp - f.R
    pt, pv = f.R / v, -f.R * t / v**2
    assert f.compressibility_factor(t, p, v) == pytest.approx(1.0)
    assert f.heat_capacity_difference(t, pt, pv) == pytest.approx(f.R)
    assert f.joule_thomson(t, v, cp, dV_dT=v / t) == pytest.approx(0.0, abs=1e-19)
    assert f.isentropic_exponent_pv(cp, cv, v, p, pv) == pytest.approx(cp / cv)
    assert f.isentropic_exponent_pt(cp, p, v / t) == pytest.approx(cp / cv)
    assert f.isentropic_exponent_tv(cv, v, pt) == pytest.approx(cp / cv)
    sound = f.acoustic_velocity(v, pv, cp, cv, molecular_weight=mw)
    assert sound**2 == pytest.approx(cp / cv * f.R * t * 1000 / mw)
    assert f.velocity_basis(sound, mw) == pytest.approx(f.acoustic_velocity(v, pv, cp, cv))


@pytest.mark.parametrize(
    "scale", ["density", "specific_gravity", "API", "Baume_light", "Baume_heavy"]
)
def test_gravity_scale_roundtrips_with_explicit_reference(scale):
    value = f.gravity_scale(850.0, target=scale, reference_density=998.0)
    assert f.gravity_scale(value, source=scale, target="density", reference_density=998.0) == (
        pytest.approx(850.0)
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: c.fraction_basis((0.5, 0.5), (18.0,)),
        lambda: c.convert_component_flows((1.0,), source="mass", target="molar"),
        lambda: c.weighted_property((0.2, 0.2), (1.0, 2.0)),
        lambda: c.weighted_property((0.5, 0.5), (0.0, 2.0), rule="logarithmic"),
        lambda: c.combine_component_flows((("A", "A"),), partial_flows=((1.0, 2.0),)),
        lambda: c.constrained_composition_hessian(((1.0, 2.0),)),
        lambda: c.composition_gradient_to_moles((1.0, 2.0), (0.2, 0.8), total_moles=0),
        lambda: f.joule_thomson(300.0, 0.01, 30.0),
        lambda: f.joule_thomson(300.0, 0.01, 30.0, dV_dT=1e-5, expansion=1e-3),
        lambda: f.acoustic_velocity(0.01, 1e5, 30.0, 20.0),
        lambda: f.phase_from_response(-1.0),
        lambda: f.gravity_scale(-131.5, source="API"),
        lambda: f.vapor_mass_fraction(1.1, 18.0, 44.0),
        lambda: f.parachor(100.0, 800.0, 800.0, 0.02),
        lambda: f.heat_capacity_difference(300.0, 100.0, 0.0),
    ],
)
def test_invalid_scientific_domains_are_explicit(call):
    with pytest.raises(ValueError):
        call()
