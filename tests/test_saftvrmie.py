"""Independent Mie integrals and common residual-model contracts."""

import json
from dataclasses import replace
from math import exp, log

import pytest
from scipy.integrate import quad

from chemworld.physchem.caloric_math import derivative_values, partial_derivative
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.saftvrmie import (
    MieSegment,
    SAFTVRMieModel,
    _perturbation_term,
    mie_effective_diameter,
)

PARAMS = ((1, 3.7412e-10, 153.36, 12.650), (1.4373, 3.7257e-10, 206.12, 12.400),
          (1.6845, 3.9056e-10, 239.89, 13.006), (3.1, 4e-10, 320, 20))


@pytest.mark.parametrize('parameters', PARAMS)
@pytest.mark.parametrize('temperature', (180, 300, 600))
def test_barker_henderson_integral_and_three_temperature_derivatives(parameters, temperature):
    m, sigma, epsilon, lr = parameters
    c = MieSegment('A', m, sigma, epsilon, 'synthetic', lr, 6)
    actual = derivative_values(lambda t: mie_effective_diameter(c, t)/sigma,
                               temperature, order=3)
    for order in range(4):
        def integrand(r, order=order):
            if r == 0:
                return 1. if order == 0 else 0.
            v = lr/(lr-6)*(lr/6)**(6/(lr-6))*epsilon/temperature*(r**-lr-r**-6)
            if v > 700:
                return 1. if order == 0 else 0.
            return (1-exp(-v) if order == 0 else
                    exp(-v)*(-v if order == 1 else 2*v-v*v if order == 2 else -6*v+6*v*v-v**3))
        expected, error = quad(integrand, 0, 1, epsabs=2e-14, epsrel=2e-13)
        assert error < 1e-12
        assert actual[order]*temperature**order == pytest.approx(expected, rel=2e-9, abs=1e-13)


def model():
    return SAFTVRMieModel((MieSegment('A', 1, 3.7412e-10, 153.36, 'synthetic', 12.65),
        MieSegment('B', 3.1, 4e-10, 320, 'synthetic', 20)), ((0, .02), (.02, 0)),
        (100, 1000), 'synthetic')


def test_vacuum_pure_limits_permutation_and_roundtrip():
    eos = model()
    assert eos == SAFTVRMieModel.from_dict(json.loads(json.dumps(eos.to_dict())))
    for i, x in enumerate(((1, 0), (0, 1))):
        pure = replace(eos, components=(eos.components[i],), interaction=((0,),))
        assert eos.alphar(300, 6000, x) == pytest.approx(pure.alphar(300, 6000, (1,)))
        assert eos.alphar(300, 0, x) == pytest.approx(0, abs=2e-14)
    reverse = replace(eos, components=eos.components[::-1])
    assert eos.alphar(300, 6000, (.3, .7)) == pytest.approx(reverse.alphar(300, 6000, (.7, .3)))
    slope = partial_derivative(lambda v: eos.alphar(300, v['r'], (.3, .7)), {'r': 0}, ('r',))
    assert eos.alphar(300, 1e-4, (.3, .7))/1e-4 == pytest.approx(slope, rel=2e-6)
    state = molecular_eos_state(eos, 600, 1000, (.3, .7))
    assert state.residual_helmholtz_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol-600*state.residual_entropy_J_mol_K)
    assert sum(x*p for x, p in zip((.3, .7), state.log_fugacity_coefficients, strict=True)) == (
        pytest.approx(eos.alphar(600, 1000, (.3, .7))+state.compressibility_factor-1
                      -log(state.compressibility_factor)))


def test_exponent_four_continuous_integral_and_declared_domain():
    for packing in (0, .2):
        actual = _perturbation_term(4., 1.05, packing)
        left = _perturbation_term(4-1e-5, 1.05, packing)
        right = _perturbation_term(4+1e-5, 1.05, packing)
        assert actual == pytest.approx(tuple((a+b)/2 for a, b in zip(left, right, strict=True)),
                                      rel=2e-9)
    with pytest.raises(ValueError, match='exponents'):
        replace(model().components[0], attractive_exponent=3)
    with pytest.raises(ValueError, match='domain'):
        model().alphar(50, 1000, (.3, .7))
    with pytest.raises(ValueError, match='packing'):
        model().alphar(300, 1e6, (.3, .7))
