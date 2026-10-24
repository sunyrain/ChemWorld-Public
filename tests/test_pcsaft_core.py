import json
from dataclasses import replace
from math import pi

import pytest

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment, hard_sphere_chain


def model():
    return PCSAFTModel(
        (SAFTSegment("a", 1, 3.7e-10, 150, "synthetic"),
         SAFTSegment("b", 2, 3.6e-10, 210, "synthetic")),
        ((0, 0.02), (0.02, 0)), (100, 1000), "synthetic physics test")


def test_sphere_limit_and_vacuum_derivative():
    for eta in (0, 1e-6, 0.1, 0.4):
        density = 6*eta/pi
        hard, packing = hard_sphere_chain(density, (1,), (1,), (1,))
        assert packing == pytest.approx(eta)
        assert hard == pytest.approx(eta*(4-3*eta)/(1-eta)**2, abs=1e-14)
    slope = partial_derivative(
        lambda p: hard_sphere_chain(p["density"], (1,), (1,), (1,))[0],
        {"density": 0}, ("density",))
    assert slope == pytest.approx(2*pi/3)


@pytest.mark.parametrize("composition", [(1, 0), (0, 1), (0.3, 0.7)])
def test_residual_energy_identities_and_local_density_inverse(composition):
    eos = model()
    state = molecular_eos_state(eos, 500, 1000, composition)
    R, t = eos.gas_constant_J_mol_K, 500
    assert state.residual_enthalpy_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol+R*t*(state.compressibility_factor-1))
    assert state.residual_helmholtz_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol-t*state.residual_entropy_J_mol_K)
    solved = molecular_density_from_pressure(
        eos, t, state.pressure_Pa, composition, initial_density_mol_m3=1100,
        density_bounds_mol_m3=(1, 5000))
    assert solved.molar_density_mol_m3 == pytest.approx(1000, rel=1e-10)
    assert state.mechanically_stable
    restored = PCSAFTModel.from_dict(json.loads(json.dumps(eos.to_dict())))
    assert molecular_eos_state(restored, 500, 1000, composition) == state


def test_pure_component_and_permutation_limits():
    eos = model()
    for i, fractions in enumerate(((1, 0), (0, 1))):
        pure = PCSAFTModel((eos.components[i],), ((0,),), (100, 1000), "synthetic")
        assert eos.alphar(300, 10000, fractions) == pytest.approx(
            pure.alphar(300, 10000, (1,)), abs=1e-14)
    reverse = replace(eos, components=eos.components[::-1])
    a, b = (molecular_eos_state(m, 500, 2000, x) for m, x in (
        (eos, (0.3, 0.7)), (reverse, (0.7, 0.3))))
    assert a.pressure_Pa == pytest.approx(b.pressure_Pa, rel=1e-13)
    assert a.log_fugacity_coefficients == pytest.approx(b.log_fugacity_coefficients[::-1])


def test_fugacity_gibbs_duhem_at_fixed_pressure():
    eos = model()
    x0, dx, t = 0.4, 1e-5, 500
    pressure = molecular_eos_state(eos, t, 2000, (x0, 1-x0)).pressure_Pa
    reports = [molecular_density_from_pressure(
        eos, t, pressure, (x, 1-x), initial_density_mol_m3=2000,
        density_bounds_mol_m3=(100, 5000)) for x in (x0-dx, x0+dx)]
    slopes = [(hi-lo)/(2*dx) for lo, hi in zip(
        reports[0].log_fugacity_coefficients, reports[1].log_fugacity_coefficients, strict=True)]
    assert x0*slopes[0]+(1-x0)*slopes[1] == pytest.approx(0, abs=1e-9)


def test_derivative_coordinates_and_missing_reference_are_explicit():
    eos = model()
    ar10 = molecular_scaled_derivative(eos, 400, 1000, (.2, .8), inverse_temperature_order=1)
    actual = molecular_eos_state(eos, 400, 1000, (.2, .8))
    assert actual.residual_internal_energy_J_mol == pytest.approx(
        eos.gas_constant_J_mol_K*400*ar10)
    assert actual.to_dict()["energy_reference"] == "ideal_gas_at_same_T_rho_composition"
    with pytest.raises(ValueError, match="totaling"):
        molecular_scaled_derivative(eos, 400, 1000, (.2, .8), density_order=4)
    for bad in ((1, 1), (-.1, 1.1), (1,)):
        with pytest.raises(ValueError):
            eos.alphar(400, 1000, bad)
    for t, rho in ((50, 100), (400, -1), (400, 1e6)):
        with pytest.raises(ValueError):
            eos.alphar(t, rho, (.2, .8))
    with pytest.raises(ValueError, match="symmetric"):
        replace(eos, interaction=((0, .1), (.2, 0)))
    with pytest.raises(ValueError, match="coefficient"):
        replace(eos, coefficient_set="unimplemented-association")
