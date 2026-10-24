"""Quantum limiting cases, thermodynamic identities and declared boundaries."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.caloric_math import derivative_values, partial_derivative
from chemworld.physchem.cubic_mixing import GibbsExcessMixingSpec, cubic_residual_from_parameters
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.quantum_molecular import (
    QuantumMiePair,
    QuantumMieSegment,
    SAFTVRQMieModel,
    santos_hard_sphere,
)
from chemworld.physchem.quantum_molecular_cubic import QuantumCubicComponent, QuantumCubicModel
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel


def mie_model(order=1, hs="nonadditive-Santos"):
    components = (
        QuantumMieSegment("He", 1., 2.7443e-10, 5.4195, "Aasen2019-FH1", 9., 6., 6.6464764e-27),
        QuantumMieSegment("Ne", 1., 2.8e-10, 32., "synthetic", 12., 6., 3.35e-26))
    return SAFTVRQMieModel(components, ((0., .02), (.02, 0.)), (5., 200.), "synthetic mixture",
                          feynman_hibbs_order=order, hard_sphere_reference=hs)


def cubic_model():
    return QuantumCubicModel((QuantumCubicComponent("He", .004, 1.5e-5, 5.1953,
        (.48558, 1.7173, .30271), 1.4912, 3.2634, -3.1791e-6, "synthetic a0/b"),
        QuantumCubicComponent("Ne", .04, 1.9e-5, 44.4, (.3, 1.2, .8), .4673,
            2.4634, 2e-6, "synthetic a0/b")), ((0., .01), (.01, 0.)), (5., 200.), "synthetic")


@pytest.mark.parametrize("order", (0, 1, 2))
def test_quantum_pair_roots_and_derivative_lifting(order):
    pair = mie_model(order).pair(0, 0)
    sigma, epsilon, diameter = pair.geometry(30.)
    assert sigma >= 1 and epsilon > 0 and 0 < diameter < sigma
    assert pair.potential_K(30., sigma) == pytest.approx(0., abs=2e-12)
    zero_identity = derivative_values(lambda t: pair.potential_K(t, pair.geometry(t)[0]),
                                     30., order=3)
    assert zero_identity == pytest.approx((0.,)*4, abs=2e-12)
    assert all(abs(v) < 100 for v in derivative_values(
        lambda t: pair.geometry(t)[2], 30., order=3))


@pytest.mark.parametrize("order", (0, 1, 2))
@pytest.mark.parametrize("hs", ("additive-exact", "nonadditive-Santos"))
def test_quantum_vacuum_endpoints_permutation_and_roundtrip(order, hs):
    model = mie_model(order, hs)
    assert SAFTVRQMieModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    assert model.alphar(80, 0, (.3, .7)) == pytest.approx(0, abs=2e-13)
    for i, x in enumerate(((1, 0), (0, 1))):
        pure = replace(model, components=(model.components[i],), interaction=((0.,),),
                       diameter_interaction=None, repulsion_interaction=None)
        assert model.alphar(80, 1000, x) == pytest.approx(pure.alphar(80, 1000, (1,)))
    reverse = replace(model, components=model.components[::-1])
    assert model.alphar(80, 1000, (.3, .7)) == pytest.approx(
        reverse.alphar(80, 1000, (.7, .3)), abs=2e-13)
    slope = partial_derivative(lambda v: model.alphar(80, v['r'], (.3, .7)), {'r': 0}, ('r',))
    assert model.alphar(80, 1e-4, (.3, .7))/1e-4 == pytest.approx(slope, rel=2e-6)


def test_pure_classical_limit_uses_same_lafitte_model():
    quantum = mie_model(2)
    component = quantum.components[0]
    classical = SAFTVRMieModel((MieSegment(component.component_id, 1, component.diameter_m,
        component.well_depth_K, component.source, component.repulsive_exponent, 6),),
        ((0.,),), (5., 200.), 'same parameters')
    quantum = replace(quantum, components=(component,), interaction=((0.,),),
        diameter_interaction=None, repulsion_interaction=None, planck_J_s=0.)
    for i in range(4):
        for j in range(4-i):
            args = {'inverse_temperature_order': i, 'density_order': j}
            assert molecular_scaled_derivative(quantum, 30., 5000., (1,), **args) == pytest.approx(
                molecular_scaled_derivative(classical, 30., 5000., (1,), **args),
                rel=2e-10, abs=2e-12)


def test_santos_pure_and_additive_equal_sphere_limits():
    eta = .25
    rho = eta*6/3.141592653589793
    expected = (4*eta-3*eta**2)/(1-eta)**2
    assert santos_hard_sphere(rho, (1.,), ((1.,),)) == pytest.approx(expected)
    assert santos_hard_sphere(rho, (.2, .8), ((1., 1.), (1., 1.))) == pytest.approx(expected)


@pytest.mark.parametrize('hs', ('additive-exact', 'nonadditive-Santos'))
def test_zero_planck_mixture_limit_keeps_the_same_hard_sphere_reference(hs):
    model = mie_model(2, hs)
    quantum_zero = replace(model, planck_J_s=0.)
    classical = replace(model, feynman_hibbs_order=0)
    for i in range(4):
        for j in range(4-i):
            orders = {'inverse_temperature_order': i, 'density_order': j}
            assert molecular_scaled_derivative(quantum_zero, 30., 5000., (.3, .7), **orders
                ) == pytest.approx(molecular_scaled_derivative(
                    classical, 30., 5000., (.3, .7), **orders), rel=2e-10, abs=2e-12)


@pytest.mark.parametrize("model", (mie_model(1), mie_model(2, "additive-exact"), cubic_model()))
def test_common_responses_inverse_and_extensive_chemical_identity(model):
    t, rho, x = 80., 1000., (.3, .7)
    state = molecular_eos_state(model, t, rho, x)
    mus = molecular_residual_chemical_potentials(model, t, rho, x)
    assert sum(xi*mu for xi, mu in zip(x, mus, strict=True)) == pytest.approx(
        state.residual_helmholtz_J_mol+state.pressure_Pa/rho-model.gas_constant_J_mol_K*t)
    assert state.residual_helmholtz_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol-t*state.residual_entropy_J_mol_K)
    recovered = molecular_density_from_pressure(model, t, state.pressure_Pa, x,
        initial_density_mol_m3=1100., density_bounds_mol_m3=(500., 1500.))
    assert recovered.molar_density_mol_m3 == pytest.approx(rho, rel=2e-10)


def test_cubic_beta_translation_and_classical_limit():
    model = cubic_model()
    assert QuantumCubicModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    c = model.components[0]
    assert c.covolume(c.critical_temperature_K) == pytest.approx(c.covolume_m3_mol)
    assert derivative_values(c.covolume, 30., order=3)[1] < 0
    c = replace(c, beta_xi_K=0, volume_translation_m3_mol=0, twu_coefficients=(0., 1., 1.))
    model = replace(model, components=(c,), interaction=((0.,),), covolume_interaction=None)
    expected = cubic_residual_from_parameters('peng_robinson', c.attraction_Pa_m6_mol2,
        c.covolume_m3_mol, 30., 1000.)/(model.gas_constant_J_mol_K*30)
    assert model.alphar(30., 1000., (1.,)) == pytest.approx(expected)
    assert model.alphar(30., 0., (1.,)) == pytest.approx(0.)


def test_explicit_validation_rejects_undeclared_settings():
    with pytest.raises(ValueError, match='mass'):
        replace(mie_model().components[0], molecular_mass_kg=0)
    with pytest.raises(ValueError, match='positive'):
        replace(mie_model().components[0], segment_count=0)
    with pytest.raises(ValueError, match='order'):
        replace(mie_model(), feynman_hibbs_order=3)
    with pytest.raises(ValueError, match='domain'):
        mie_model().alphar(1, 1000, (.3, .7))
    with pytest.raises(ValueError, match='packing'):
        mie_model().alphar(80, 1e6, (.3, .7))
    with pytest.raises(ValueError, match='positive'):
        QuantumMiePair(3e-10, 10., 12., 6., 1e-27, planck_J_s=-1)
    with pytest.raises(ValueError, match='positive'):
        cubic_model().alphar(80, 1e7, (1., 0.))


@pytest.mark.parametrize('rule', ['huron_vidal', 'mhv', 'mhv1', 'mhv2', 'wong_sandler'])
def test_quantum_cubic_ge_shared_consumption_and_pure_limits(rule):
    model = cubic_model()
    # Keep the exact MHV branch criterion: a weak synthetic attraction has no
    # zero-pressure liquid root; strengthen the positive fixture explicitly.
    original_components = model.components
    model = replace(model, components=(replace(model.components[0],
        attraction_Pa_m6_mol2=.01), model.components[1]))
    activity = ActivityModelSpec('synthetic', model.component_ids, 'nrtl', {
        'tau_b:He|Ne': 2., 'tau_b:Ne|He': 1., 'alpha:He|Ne': .3, 'alpha:Ne|He': .3})
    coefficients = (-.53,) if rule == 'mhv1' else (-.4347, -.003654) if rule == 'mhv2' else ()
    ge = GibbsExcessMixingSpec(rule, activity, coefficients,
                              {'He|Ne': .01} if rule == 'wong_sandler' else {})
    model = replace(model, ge_mixing=ge, interaction=((0., 0.), (0., 0.)))
    assert model == QuantumCubicModel.from_dict(json.loads(json.dumps(model.to_dict())))
    for x in ((1., 0.), (0., 1.)):
        assert model.alphar(5., 100., x) == pytest.approx(
            replace(model, ge_mixing=None).alphar(5., 100., x), rel=2e-12)
    state = molecular_eos_state(model, 5., 100., (.3, .7), require_stable=False)
    mus = molecular_residual_chemical_potentials(model, 5., 100., (.3, .7))
    assert .3*mus[0]+.7*mus[1] == pytest.approx(
        state.residual_helmholtz_J_mol+state.pressure_Pa/100-model.gas_constant_J_mol_K*5)
    with pytest.raises(ValueError, match='kij/lij'):
        replace(model, covolume_interaction=((0., .1), (.1, 0.)))
    if rule == 'mhv':
        with pytest.raises(ValueError, match='zero-pressure'):
            replace(model, components=original_components).alphar(5., 100., (1., 0.))
