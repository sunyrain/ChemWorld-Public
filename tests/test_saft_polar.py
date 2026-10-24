"""Polar perturbations preserve limiting physics and derivative semantics."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.molecular_eos import (
    molecular_eos_state,
    molecular_residual_chemical_potentials,
)
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.saft_polar import SAFTMultipole, SAFTPolar


def model(convention="Gross-Vrabec"):
    components = tuple(SAFTSegment(c, m, s*1e-10, e, "synthetic") for c, m, s, e in (
        ("D", 2.7, 3.2, 220), ("Q", 1.6, 3.1, 165), ("N", 1, 3.7, 150)))
    polar = SAFTPolar(tuple(SAFTMultipole(c, d, q, "synthetic") for c, d, q in (
        ("D", 1.5, 0), ("Q", 0, 1.2), ("N", 0, 0))), "synthetic", convention)
    return PCSAFTModel(components, ((0, .02, 0), (.02, 0, 0), (0, 0, 0)),
                      (100, 1000), "synthetic", polar=polar)


@pytest.mark.parametrize("convention", ["Gross-Vrabec", "teqp-0.23.2"])
def test_vacuum_absent_components_and_serialization(convention):
    eos = model(convention)
    restored = PCSAFTModel.from_dict(json.loads(json.dumps(eos.to_dict())))
    for xs in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (.2, .3, .5)):
        assert eos.alphar(400, 0, xs) == 0
        for order in (1, 2, 3):
            slope = partial_derivative(lambda v, xs=xs: eos.alphar(400, v["r"], xs),
                                       {"r": 0}, ("r",)*order)
            assert abs(slope) < 1
        assert molecular_eos_state(eos, 600, 1000, xs) == (
            molecular_eos_state(restored, 600, 1000, xs))


def test_zero_cross_population_chemical_potential_and_nonpolar_limit():
    eos = model()
    for xs, tracer in (((0, .8, .2), 0), ((.8, 0, .2), 1)):
        exact = molecular_residual_chemical_potentials(eos, 400, 5000, xs)[tracer]
        h = 1e-5
        values = []
        for i in range(5):
            amounts = list(xs)
            amounts[tracer] += i*h
            total = sum(amounts)
            values.append(total*eos.alphar(400, 5000*total, tuple(v/total for v in amounts)))
        numerical = sum(c*f for c, f in zip((-25, 48, -36, 16, -3), values, strict=True))/(12*h)
        assert exact == pytest.approx(eos.gas_constant_J_mol_K*400*numerical, rel=2e-8)
    zeros = replace(eos.polar, moments=tuple(replace(p, reduced_dipole_squared=0,
                    reduced_quadrupole_squared=0) for p in eos.polar.moments))
    assert replace(eos, polar=zeros).alphar(400, 5000, (.2, .3, .5)) == (
        replace(eos, polar=None).alphar(400, 5000, (.2, .3, .5)))


def test_contributions_permutation_and_model_conventions():
    eos = model()
    xs = (.2, .3, .5)
    contributions = eos.contributions(400, 5000, xs)
    assert all(contributions[k] < 0 for k in ('DD', 'QQ', 'DQ'))
    reverse = replace(eos, components=eos.components[::-1],
        interaction=tuple(row[::-1] for row in eos.interaction[::-1]),
        polar=replace(eos.polar, moments=eos.polar.moments[::-1]))
    assert reverse.alphar(400, 5000, xs[::-1]) == pytest.approx(eos.alphar(400, 5000, xs))
    assert eos.alphar(400, 5000, xs) != pytest.approx(model('teqp-0.23.2').alphar(400, 5000, xs))
    with pytest.raises(ValueError, match="convention"):
        replace(eos.polar, convention="silently-mix-sources")
    with pytest.raises(ValueError, match="nonnegative"):
        replace(eos.polar.moments[0], reduced_dipole_squared=-1)
