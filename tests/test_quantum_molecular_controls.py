"""Chain/control limits, mass units and independent cutoff identities."""

import json
from dataclasses import replace
from math import pi, sqrt

import pytest
from scipy.integrate import quad

from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.dufal_association import DufalAssociation
from chemworld.physchem.molecular_eos import molecular_scaled_derivative
from chemworld.physchem.quantum_molecular import QuantumMieSegment, SAFTVRQMieModel
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel


def chain_model(order=1):
    return SAFTVRQMieModel((
        QuantumMieSegment('He', 1.5, 2.7443e-10, 5.4195, 'explicit', 9., 6., 6.6464764e-27),
        QuantumMieSegment('Ne', 2.3, 2.8e-10, 32., 'explicit', 12., 6., 3.35e-26)),
        ((0., .02), (.02, 0.)), (5., 200.), 'synthetic chains', order,
        hard_sphere_reference='Lafitte')


def pure_model(order=1):
    model = chain_model(order)
    return replace(model, components=(model.components[0],), interaction=((0.,),),
                   diameter_interaction=None, repulsion_interaction=None)


def test_fh0_chain_and_composition_derivatives_reduce_to_classical_model():
    quantum = chain_model(0)
    classical = SAFTVRMieModel(tuple(MieSegment(c.component_id, c.segment_count,
        c.diameter_m, c.well_depth_K, c.source, c.repulsive_exponent, c.attractive_exponent)
        for c in quantum.components), quantum.interaction, quantum.temperature_range_K, 'same')
    for x in ((1., 0.), (.4, .6), (0., 1.)):
        for i in range(4):
            for j in range(4-i):
                orders = {'inverse_temperature_order': i, 'density_order': j}
                assert molecular_scaled_derivative(quantum, 80., 5000., x, **orders) ==\
                    pytest.approx(molecular_scaled_derivative(classical, 80., 5000., x,
                        **orders), rel=2e-10, abs=2e-12)


@pytest.mark.parametrize('order', [0, 1, 2])
def test_disabled_terms_ideal_limit_and_chain_response_to_dispersion_switches(order):
    model = chain_model(order)
    off = replace(model, enable_hard_sphere=False, enable_hs_extra=False,
        enable_a1=False, enable_a2=False, enable_a3=False, enable_chain=False)
    assert derivative_values(lambda t: off.alphar(t, 5000., (.4, .6)), 80., order=3) == (0.,)*4
    original = model.contributions(80., 5000., (.4, .6))
    without_a1 = replace(model, enable_a1=False).contributions(80., 5000., (.4, .6))
    assert without_a1['dispersion1'] == 0
    assert without_a1['chain'] != pytest.approx(original['chain'])
    assert without_a1['hard_sphere'] == original['hard_sphere']
    spheres = replace(model, components=tuple(replace(c, segment_count=1.)
                                             for c in model.components))
    assert spheres.contributions(80., 5000., (.4, .6))['chain'] == 0


def test_mass_helpers_units_and_temperature_cache_equivalence():
    model = chain_model()
    c = model.components[0]
    altered = model.with_de_boer_parameter(0, 2*model.de_boer_parameter(0))
    assert altered.components[0].molecular_mass_kg == pytest.approx(c.molecular_mass_kg/4)
    assert altered.de_boer_parameter(0) == pytest.approx(2*model.de_boer_parameter(0))
    assert model.de_broglie_wavelength(0, 80.) == pytest.approx(model.planck_J_s/sqrt(
        2*pi*c.molecular_mass_kg*model.boltzmann_J_K*80.))
    assert derivative_values(lambda t: model.de_broglie_wavelength(0, t), 80., order=3)[1] ==\
        pytest.approx(-model.de_broglie_wavelength(0, 80.)/160.)
    assert replace(model, enable_temperature_cache=False).alphar(80., 1000., (.4, .6)) ==\
        pytest.approx(model.alphar(80., 1000., (.4, .6)), abs=1e-13)


@pytest.mark.parametrize('order', [0, 1, 2])
@pytest.mark.parametrize('shift', [False, True])
def test_cutoff_free_energy_matches_independent_radial_tail_integral(order, shift):
    model = replace(pure_model(order), enable_truncation_correction=True,
                    enable_shift_correction=shift, reduced_radius_cut=2.5)
    pair, t, rho, cutoff = model.pair(0, 0), 80., 5000., model.reduced_radius_cut
    integral = quad(lambda r: pair.potential_K(t, r)*r*r, cutoff, float('inf'),
                    epsabs=1e-11, epsrel=1e-11)[0]
    if shift:
        integral += pair.potential_K(t, cutoff)*cutoff**3/3
    expected = -2*pi*rho*6.02214076e23*pair.diameter_m**3*integral/t
    assert model.contributions(t, rho, (1.,))['truncation'] == pytest.approx(expected, rel=1e-11)
    assert model.potential_K(t, 0, 0, cutoff+.1) == 0.
    assert model.potential_K(t, 0, 0, cutoff+.1, 1) == 0.
    assert model.potential_K(t, 0, 0, 1.5, 1) == pair.potential_K(t, 1.5, 1)
    if shift:
        assert abs(model.potential_K(t, 0, 0, cutoff-1e-9)) < 1e-8
    if order:
        with pytest.raises(ValueError, match='truncated-alpha'):
            model.alpha(t)
    else:
        assert model.alpha(t)[0][0] > 0.


def test_control_roundtrip_and_unsupported_domains_are_explicit():
    model = replace(pure_model(), exact_binary_dhs=False, enable_hs_extra=True,
                    enable_a1=False, enable_truncation_correction=True,
                    enable_shift_correction=True, reduced_radius_cut=4.)
    assert SAFTVRQMieModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    mixture = replace(chain_model(), enable_truncation_correction=True)
    assert mixture.potential_K(80., 0, 1, 4.) == 0.
    with pytest.raises(ValueError, match='pure fluids'):
        mixture.alphar(80., 1000., (.4, .6))
    with pytest.raises(ValueError, match='booleans'):
        replace(model, enable_a1=1)
    with pytest.raises(ValueError, match='booleans'):
        replace(model, exact_binary_dhs=0)
    with pytest.raises(ValueError, match='positive cutoff'):
        replace(model, reduced_radius_cut=0.)
    with pytest.raises(ValueError, match='positive'):
        model.with_de_boer_parameter(0, 0.)


def test_quantum_association_composition_uses_shared_sites_and_serialization():
    component = QuantumMieSegment('H2O', 1., 3.0555e-10, 418., 'Dufal2015',
                                 35.823, 6., 2.9915076e-26)
    sites = SAFTAssociation((AssociationSite('H2O', 'e', 2), AssociationSite('H2O', 'H', 2)),
        ((0., 1600.), (1600., 0.)), ((0., 496.66e-30), (496.66e-30, 0.)), 'Dufal2015')
    association = DufalAssociation(('H2O',), (1.,), (component.diameter_m,), ((418.,),),
        ((35.823,),), sites, 'Dufal2015')
    model = SAFTVRQMieModel((component,), ((0.,),), (200., 800.), 'quantum association',
                            association=association)
    decoded = SAFTVRQMieModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert decoded == model
    bare = replace(model, association=None)
    assert derivative_values(lambda t: decoded.alphar(t, 1000., (1.,))
        -bare.alphar(t, 1000., (1.,)), 300., order=3) == pytest.approx(
            derivative_values(lambda t: association.alphar(t, 1000., (1.,)), 300., order=3),
            rel=1e-10, abs=1e-12)
