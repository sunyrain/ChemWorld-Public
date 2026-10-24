"""Dufal topology, analytic mass-action limits and Mie integration."""

import json
from dataclasses import replace
from math import expm1, log, sqrt

import pytest

from chemworld.physchem.caloric_math import derivative_values, partial_derivative
from chemworld.physchem.dufal_association import DufalAssociation, dufal_integral
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel


def association(counts=(2, 2), m=1):
    sites = SAFTAssociation((AssociationSite('A', 'e', counts[0]),
        AssociationSite('A', 'H', counts[1])), ((0, 1600), (1600, 0)),
        ((0, 496.66e-30), (496.66e-30, 0)), 'Dufal water HB, explicit topology')
    return DufalAssociation(('A',), (m,), (3.0555e-10,), ((418.,),), ((35.823,),),
                            sites, 'Dufal water kernel, synthetic topology/m if changed')


@pytest.mark.parametrize('counts', ((1, 1), (1, 2), (2, 2), (1, 3)))
@pytest.mark.parametrize('density', (0, 1, 1000, 15000))
def test_pure_site_counts_match_independent_quadratic(counts, density):
    model = association(counts)
    strength = expm1(1600/450)*496.66e-30*model.contacts(450, density, (1,))[0][0]
    c, (a, b) = density*AVOGADRO_PER_MOL*strength, counts
    bonds = 2*c*a*b/(1+c*(a+b)+sqrt(1+2*c*(a+b)+c*c*(a-b)**2))
    xs = (1-bonds/a, 1-bonds/b)
    expected = sum(n*(log(x)-x/2+.5) for n, x in zip(counts, xs, strict=True))
    assert model.alphar(450, density, (1,)) == pytest.approx(expected, abs=2e-12)


def test_vacuum_derivatives_and_density_convention():
    model = association(m=1.7989)
    molecular = replace(model, density_basis='molecular')
    assert model.contacts(450, 1000, (1,)) == molecular.contacts(450, 1798.9, (1,))
    assert model.alphar(450, 1000, (1,)) != molecular.alphar(450, 1000, (1,))
    slope = partial_derivative(lambda v: model.alphar(450, v['r'], (1,)), {'r': 0}, ('r',))
    strength = 496.66e-30*expm1(1600/450)*dufal_integral(450/418, 0, 35.823)
    assert slope == pytest.approx(-4*AVOGADRO_PER_MOL*strength, rel=2e-12)
    # The derivative of order two differentiated once agrees with order three.
    derivatives = derivative_values(lambda t: model.alphar(t, 1000, (1,)), 450, order=3)
    h = .02
    left = derivative_values(lambda t: model.alphar(t, 1000, (1,)), 450-h, order=2)[2]
    right = derivative_values(lambda t: model.alphar(t, 1000, (1,)), 450+h, order=2)[2]
    assert derivatives[3] == pytest.approx((right-left)/(2*h), rel=2e-7)


def test_full_mie_roundtrip_masks_and_residual_identity():
    assoc = association()
    base = SAFTVRMieModel((MieSegment('A', 1, 3.0555e-10, 418, 'Dufal', 35.823),),
                         ((0.,),), (250, 800), 'Dufal water')
    full = replace(base, association=assoc)
    assert full == SAFTVRMieModel.from_dict(json.loads(json.dumps(full.to_dict())))
    assert full.alphar(600, 1000, (1,))-base.alphar(600, 1000, (1,)) == pytest.approx(
        assoc.alphar(600, 1000, (1,)))
    masked = replace(assoc, sites=replace(assoc.sites, energy_K=((0, 0), (0, 0))))
    assert masked.alphar(450, 1000, (1,)) == 0
    state = molecular_eos_state(full, 600, 1000, (1,))
    assert state.residual_helmholtz_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol-600*state.residual_entropy_J_mol_K)
    with pytest.raises(ValueError, match='ordering'):
        replace(base, association=replace(assoc, component_ids=('B',), sites=replace(
            assoc.sites, sites=(AssociationSite('B', 'e', 2), AssociationSite('B', 'H', 2)))))


@pytest.mark.parametrize('kwargs', ({'density_basis': 'automatic'},
    {'pair_repulsive_exponents': ((3,),)}, {'diameters_m': (-1,)}))
def test_invalid_parameter_domains(kwargs):
    with pytest.raises(ValueError):
        replace(association(), **kwargs)
