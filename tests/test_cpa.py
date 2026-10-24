"""CPA reuses cubic equations without manufacturing critical parameters."""

import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.cubic_mixing import GibbsExcessMixingSpec
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation


def model(shape='srk', contact='Kontogeorgis'):
    c = CPAComponent('test', .2, 2e-5, 500, .6, 'synthetic')
    v = c.covolume_m3_mol*.03/AVOGADRO_PER_MOL
    sites = SAFTAssociation((AssociationSite('test', 'H'), AssociationSite('test', 'O')),
                            ((0, 1800), (1800, 0)), ((0, v), (v, 0)), 'synthetic')
    return CPAModel((c,), ((0,),), (100, 1000), 'synthetic', sites,
                    cubic_model=shape, radial_distribution=contact)


@pytest.mark.parametrize('shape', ['srk', 'peng_robinson'])
def test_nonassociating_limit_pressure_and_vacuum(shape):
    eos = replace(model(shape), association=None)
    t, rho = 400, 4000
    a = .2*(1+.6*(1-(t/500)**.5))**2
    v, b, R = 1/rho, 2e-5, eos.gas_constant_J_mol_K
    denom = v*(v+b) if shape == 'srk' else v*(v+b)+b*(v-b)
    assert molecular_eos_state(eos, t, rho, (1,)).pressure_Pa == pytest.approx(
        R*t/(v-b)-a/denom, rel=1e-12)
    assert eos.alphar(t, 0, (1,)) == 0
    assert partial_derivative(lambda q: eos.alphar(t, q['r'], (1,)), {'r': 0}, ('r',)) == (
        pytest.approx(b-a/(R*t)))


@pytest.mark.parametrize('contact', ['Kontogeorgis', 'Carnahan-Starling'])
def test_association_exact_two_site_response_and_serialization(contact):
    eos = model(contact=contact)
    t, rho = 400, 4000
    eta = rho*2e-5/4
    g = 1/(1-1.9*eta) if contact == 'Kontogeorgis' else (1-eta/2)/(1-eta)**3
    q = rho*2e-5*.03*(exp(1800/t)-1)*g
    x = 2/(1+(1+4*q)**.5)
    assert eos.contributions(t, rho, (1,))['association'] == pytest.approx(2*log(x)-x+1)
    assert eos == CPAModel.from_dict(json.loads(json.dumps(eos.to_dict())))


def test_cpa_invalid_shape_parameters_and_density_pole():
    with pytest.raises(ValueError):
        replace(model(), cubic_model='unrecognized')
    with pytest.raises(ValueError):
        replace(model().components[0], covolume_m3_mol=-1)
    with pytest.raises(ValueError, match='pole'):
        model().alphar(400, 5e4, (1,))


@pytest.mark.parametrize('rule', ['huron_vidal', 'mhv', 'mhv1', 'mhv2', 'wong_sandler'])
@pytest.mark.parametrize('shape', ['srk', 'peng_robinson'])
def test_ge_mixing_pure_limits_serialization_and_chemical_potential_identity(rule, shape):
    c = model(shape).components[0]
    activity = ActivityModelSpec('synthetic', ('test', 'other'), 'nrtl', {
        'tau_b:test|other': 100., 'tau_b:other|test': 50.,
        'alpha:test|other': .3, 'alpha:other|test': .3})
    coefficients = (-.53,) if rule == 'mhv1' else (-.4347, -.003654) if rule == 'mhv2' else ()
    ge = GibbsExcessMixingSpec(rule, activity, coefficients,
                              {'other|test': .03} if rule == 'wong_sandler' else {})
    eos = replace(model(shape), components=(c, replace(c, component_id='other',
        attraction_Pa_m6_mol2=.3, covolume_m3_mol=3e-5)), interaction=((0, 0), (0, 0)),
        ge_mixing=ge)
    assert eos == CPAModel.from_dict(json.loads(json.dumps(eos.to_dict())))
    for x in ((1, 0), (0, 1)):
        assert eos.alphar(200, 3000, x) == pytest.approx(
            replace(eos, ge_mixing=None).alphar(200, 3000, x), rel=1e-12)
    state = molecular_eos_state(eos, 200, 100, (.3, .7), require_stable=False)
    assert sum(x*phi for x, phi in zip((.3, .7), state.log_fugacity_coefficients, strict=True)) == (
        pytest.approx(eos.alphar(200, 100, (.3, .7))+state.compressibility_factor-1
                      -log(state.compressibility_factor)))
    if shape == 'peng_robinson' and rule == 'mhv':
        with pytest.raises(ValueError, match='zero-pressure'):
            eos.alphar(250, 3000, (1, 0))
    with pytest.raises(ValueError, match='kij'):
        replace(eos, interaction=((0, .1), (.1, 0)))
