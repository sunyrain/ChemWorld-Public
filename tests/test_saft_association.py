"""Independent site-balance solutions and implicit derivative contracts."""

import json
from dataclasses import replace
from itertools import product
from math import isclose, log

import pytest

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.saft_association import (
    AssociationSite,
    GeometricAssociationPair,
    SAFTAssociation,
    association_helmholtz,
    association_site_fractions,
)
from chemworld.physchem.second_order_pair import SecondOrderPair


@pytest.mark.parametrize("strength", [0, 1e-12, .1, 10, 1e6, 1e12])
def test_single_site_root_and_first_three_derivatives(strength):
    # q X^2+X=1 has a stable analytic solution, including the vacuum limit.
    expected = 2/(1+(1+4*strength)**.5)
    assert association_site_fractions((1,), ((strength,),))[0] == pytest.approx(expected)
    for order in (1, 2, 3):
        actual = partial_derivative(
            lambda v: association_site_fractions((1,), ((v["q"],),))[0],
            {"q": strength}, ("q",)*order)
        reference = partial_derivative(lambda v: 2/(1+(1+4*v["q"])**.5),
                                       {"q": strength}, ("q",)*order)
        assert actual == pytest.approx(reference, rel=2e-11, abs=1e-60)


def test_two_site_unequal_populations_mixed_derivatives_and_pair_jet():
    # With only AB bonds: pA(1-XA)=pB(1-XB). Eliminating XB gives this quadratic.
    def exact(v):
        a, b, q = v["a"], v["b"], v["q"]
        c = 1+(b-a)*q
        return 2/(c+(c*c+4*a*q)**.5)

    def implicit(v):
        return association_site_fractions((v["a"], v["b"]), ((0, v["q"]), (v["q"], 0)))[0]

    values = {"a": .3, "b": 1.4, "q": 23.0}
    assert implicit(values) == pytest.approx(exact(values), rel=1e-13)
    for order in (1, 2, 3):
        for variables in product(values, repeat=order):
            assert partial_derivative(implicit, values, variables) == pytest.approx(
                partial_derivative(exact, values, variables), rel=2e-10, abs=1e-13)
    jets = {"a": SecondOrderPair(.3, dx=1), "b": 1.4, "q": SecondOrderPair(23, dy=1)}
    actual, expected = implicit(jets), exact(jets)
    for field in ("value", "dx", "dy", "dxx", "dxy", "dyy"):
        assert getattr(actual, field) == pytest.approx(getattr(expected, field), rel=2e-11)


def associated_model():
    sites = tuple(AssociationSite(c, s) for c in ("alcohol", "water") for s in ("H", "O"))
    energies = tuple(tuple(1500 if a.site_id != b.site_id else 0 for b in sites) for a in sites)
    volumes = tuple(tuple(1e-30 if a.site_id != b.site_id else 0 for b in sites) for a in sites)
    association = SAFTAssociation(sites, energies, volumes, "synthetic test")
    return PCSAFTModel((SAFTSegment("alcohol", 2, 3.5e-10, 220, "synthetic"),
                       SAFTSegment("water", 1.5, 3e-10, 200, "synthetic")),
                      ((0, .03), (.03, 0)), (100, 1000), "synthetic", association=association)


def test_multiple_sites_balance_zero_populations_and_permutation():
    populations = (0, .2, 1.8, 1)
    coupling = ((0, 4, 9, 1), (4, 0, 5, 2), (9, 5, 0, 3), (1, 2, 3, 0))
    result = association_site_fractions(populations, coupling)
    assert all(0 < x <= 1 for x in result)
    for i, x in enumerate(result):
        assert x*(1+sum(populations[j]*coupling[i][j]*result[j] for j in range(4))) == (
            pytest.approx(1, abs=5e-13))
    reversed_result = association_site_fractions(populations[::-1],
                                                tuple(r[::-1] for r in coupling[::-1]))
    assert result == pytest.approx(reversed_result[::-1], abs=1e-14)


def test_pcsaft_association_serialization_caloric_and_composition_limits():
    model = associated_model()
    restored = PCSAFTModel.from_dict(json.loads(json.dumps(model.to_dict())))
    for xs in ((1, 0), (0, 1), (.3, .7)):
        assert model.alphar(400, 0, xs) == 0
        state = molecular_eos_state(model, 600, 2000, xs)
        assert state == molecular_eos_state(restored, 600, 2000, xs)
        assert state.residual_internal_energy_J_mol == pytest.approx(
            state.residual_helmholtz_J_mol+600*state.residual_entropy_J_mol_K)
        # Independent extensive-A Euler identity for all partial chemical potentials.
        assert sum(x*phi for x, phi in zip(xs, state.log_fugacity_coefficients, strict=True)) == (
            pytest.approx(model.alphar(600, 2000, xs)+state.compressibility_factor-1
                          -log(state.compressibility_factor)))
    disabled = replace(model, association=replace(model.association,
        energy_K=tuple((0,)*4 for _ in range(4))))
    assert isclose(disabled.alphar(400, 5000, (.3, .7)),
                   replace(model, association=None).alphar(400, 5000, (.3, .7)), abs_tol=1e-14)


def test_invalid_site_parameters_and_populations():
    with pytest.raises(ValueError):
        AssociationSite("water", "H", True)
    for populations, matrix in (((-1,), ((0,),)), ((1,), ((-1,),)), ((1,), ((float("nan"),),))):
        with pytest.raises(ValueError, match="nonnegative"):
            association_site_fractions(populations, matrix)
    with pytest.raises(ValueError, match="symmetric"):
        replace(associated_model().association, energy_K=((0, 1, 2, 3),)*4)


def test_equivalent_site_multiplicities_and_simplified_pure_limit():
    # Two equivalent donor and two equivalent acceptor sites can be grouped.
    explicit = association_site_fractions((1,)*4,
        ((0, 0, 7, 7), (0, 0, 7, 7), (7, 7, 0, 0), (7, 7, 0, 0)))
    grouped = association_site_fractions((2, 2), ((0, 7), (7, 0)))
    assert explicit == pytest.approx((grouped[0],)*2+(grouped[1],)*2)
    model = associated_model()
    for xs in ((1, 0), (0, 1)):
        original = molecular_eos_state(model, 600, 1000, xs)
        simplified = molecular_eos_state(replace(model, simplified=True), 600, 1000, xs)
        assert original.pressure_Pa == pytest.approx(simplified.pressure_Pa, rel=1e-13)
        assert original.residual_cv_J_mol_K == pytest.approx(simplified.residual_cv_J_mol_K)
    assert model.alphar(400, 3000, (.3, .7)) != pytest.approx(
        replace(model, simplified=True).alphar(400, 3000, (.3, .7)), rel=1e-7)


def test_geometric_strength_vacuum_and_explicit_site_references():
    # Identical single-site self bonds make Delta uniform: an exact quadratic.
    from chemworld.physchem.caloric_math import exponential_minus_one, logarithm

    spec = SAFTAssociation((AssociationSite('A', 's'), AssociationSite('B', 's')),
        ((800, 0), (0, 800)), ((1e-28, 0), (0, 1e-28)), 'synthetic',
        (GeometricAssociationPair((0, 1), ((0, 0), (1, 1))),))
    def native(v):
        return association_helmholtz(spec, v['T'], v['r'], ('A', 'B'), (.3, .7),
                                     ((1, 1), (1, 1)))
    def exact(v):
        q = v['r']*1e-28*exponential_minus_one(800/v['T'])
        x = 2/(1+(1+4*q)**.5)
        return logarithm(x)-x/2+.5
    for rho in (0, 1e26):
        for order in range(1, 4):
            for names in product(('T', 'r'), repeat=order):
                values = {'T': 400., 'r': rho}
                assert partial_derivative(native, values, names) == pytest.approx(
                    partial_derivative(exact, values, names), rel=2e-9, abs=1e-95)
    assert spec == SAFTAssociation.from_dict(json.loads(json.dumps(spec.to_dict())))
    with pytest.raises(ValueError, match='also declare'):
        replace(spec, energy_K=((800, 100), (100, 800)))
    with pytest.raises(ValueError, match='self bond'):
        replace(spec, geometric_pairs=(GeometricAssociationPair((0, 1), ((0, 0), (0, 0))),))
    with pytest.raises(ValueError, match='duplicate'):
        replace(spec, geometric_pairs=spec.geometric_pairs*2)
