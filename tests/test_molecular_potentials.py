"""Potential distinctions, scale conversion and the shared EOS consumers."""

import json
from dataclasses import replace
from math import exp

import pytest
from scipy.integrate import quad

from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.molecular_eos import molecular_eos_state, molecular_scaled_derivative
from chemworld.physchem.molecular_potentials import (
    POTENTIAL_FAMILIES,
    PurePotentialModel,
    SoftSAFTModel,
    _johnson_integrals,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, SAFTSegment


def pure(family):
    return PurePotentialModel(SAFTSegment('A', 1, 3.4e-10, 120, 'synthetic scales'),
        family, (60, 1200), 'published model potential',
        14 if family.startswith('exp6') else 1.5 if family.startswith('square') else None)


@pytest.mark.parametrize('family', POTENTIAL_FAMILIES)
def test_vacuum_roundtrip_si_scaling_and_common_thermodynamics(family):
    model = pure(family)
    assert PurePotentialModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    assert model.alphar(360, 0, (1,)) == pytest.approx(0, abs=2e-14)
    rho = .3/(AVOGADRO_PER_MOL*model.component.diameter_m**3)
    scaled = replace(model, component=replace(model.component, diameter_m=6.8e-10,
                                               well_depth_K=240))
    assert model.alphar(360, rho, (1,)) == pytest.approx(scaled.alphar(720, rho/8, (1,)))
    state = molecular_eos_state(model, 360, rho, (1,), require_stable=False)
    assert state.residual_helmholtz_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol-360*state.residual_entropy_J_mol_K)
    assert state.residual_enthalpy_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol+state.pressure_Pa/rho-model.gas_constant_J_mol_K*360)
    for i in range(4):
        assert molecular_scaled_derivative(model, 360, 0, (1,),
                                            inverse_temperature_order=i) == pytest.approx(0)


def test_johnson_integrals_near_zero_and_series_boundary():
    for i in range(1, 7):
        r = 1e-5
        assert _johnson_integrals(r)[i-1]/r**(2*i) == pytest.approx(1/(2*i), rel=1e-9)
        for point in (1-1e-7, 1+1e-7):
            actual = derivative_values(lambda r, i=i: _johnson_integrals(r)[i-1], point, order=3)
            p, r, decay = 2*i-1, point, exp(-3*point*point)
            value = quad(lambda s, p=p: exp(-3*s*s)*s**p, 0, point, epsabs=1e-14)[0]
            expected = (value, decay*r**p, decay*(p*r**(p-1)-6*r**(p+1)),
                decay*(p*(p-1)*r**(p-2)-(12*p+6)*r**p+36*r**(p+2)))
            assert actual == pytest.approx(expected, rel=2e-8, abs=2e-12)


def test_softsaft_pure_monomer_permutation_and_endpoints():
    a = pure('LJ-Johnson-1993').component
    b = SAFTSegment('B', 1.8, 3.8e-10, 180, 'synthetic')
    c = SAFTSegment('C', 3, 4e-10, 220, 'synthetic')
    model = SoftSAFTModel((a, b, c), (60, 1200), 'synthetic')
    assert model == SoftSAFTModel.from_dict(json.loads(json.dumps(model.to_dict())))
    for i in range(3):
        x = tuple(float(j == i) for j in range(3))
        selected = replace(model, components=(model.components[i],))
        assert model.alphar(600, 1000, x) == pytest.approx(selected.alphar(600, 1000, (1,)))
    assert model.alphar(600, 1000, (1, 0, 0)) == pytest.approx(
        pure('LJ-Johnson-1993').alphar(600, 1000, (1,)))
    assert model.alphar(600, 1000, (.2, .3, .5)) == pytest.approx(
        replace(model, components=(c, b, a)).alphar(600, 1000, (.5, .3, .2)))


def test_distinct_potentials_and_explicit_domain_rejections():
    assert pure('PeTS-2018').alphar(300, 1000, (1,)) != pure('LJ-Johnson-1993').alphar(
        300, 1000, (1,))
    for kwargs in ({'family': 'automatic'}, {'shape_parameter': 14}):
        with pytest.raises(ValueError):
            replace(pure('LJ-Johnson-1993'), **kwargs)
    with pytest.raises(ValueError, match='shape'):
        replace(pure('exp6-Kataoka-1992'), shape_parameter=8)
    with pytest.raises(ValueError, match='shape'):
        replace(pure('square-well-2009'), shape_parameter=1)
    with pytest.raises(ValueError, match='normalized'):
        pure('PeTS-2018').alphar(300, 1000, (.2, .8))
    with pytest.raises(ValueError, match='packing'):
        pure('PeTS-2018').alphar(300, 1e6, (1,))


def test_square_well_range_two_is_continuous():
    model = replace(pure('square-well-2009'), shape_parameter=2)
    values = [replace(model, shape_parameter=w).alphar(360, 1000, (1,))
              for w in (2-1e-7, 2, 2+1e-7)]
    assert values == pytest.approx([values[1]]*3, rel=2e-6)
