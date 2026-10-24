"""Pure potential declaration, analytic network and common-consumer behavior."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_scaled_derivative,
)
from chemworld.physchem.molecular_potential_extensions import (
    ChaparroMieModel,
    PohlMieModel,
    TwoCenterLJModel,
    _network_derivatives,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL

CLASSES = (TwoCenterLJModel, PohlMieModel, ChaparroMieModel)


def model(cls, **kwargs):
    return cls("synthetic", 3.4e-10, 120.0, (50.0, 2000.0), "Explicit test input", **kwargs)


@pytest.mark.parametrize("cls", CLASSES)
def test_roundtrip_and_si_scaling(cls):
    base = model(cls)
    restored = cls.from_dict(json.loads(json.dumps(base.to_dict())))
    assert restored == base
    scaled = replace(base, diameter_m=base.diameter_m * 2, well_depth_K=base.well_depth_K * 2)
    assert base.alphar(300.0, 1500.0, (1.0,)) == pytest.approx(
        scaled.alphar(600.0, 1500.0 / 8, (1.0,))
    )
    assert base.alphar(300.0, 0.0, (1.0,)) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("cls", CLASSES)
def test_common_pressure_density_and_residual_identities(cls):
    base = model(cls)
    state = molecular_eos_state(base, 480.0, 1000.0, (1.0,))
    rt = base.gas_constant_J_mol_K * 480
    assert state.residual_enthalpy_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol + rt * (state.compressibility_factor - 1)
    )
    assert state.residual_helmholtz_J_mol == pytest.approx(
        state.residual_internal_energy_J_mol - 480 * state.residual_entropy_J_mol_K
    )
    root = molecular_density_from_pressure(
        base,
        480.0,
        state.pressure_Pa,
        (1.0,),
        initial_density_mol_m3=1100.0,
        density_bounds_mol_m3=(500.0, 1500.0),
    )
    assert root.molar_density_mol_m3 == pytest.approx(1000.0, rel=1e-10)


@pytest.mark.parametrize("cls", CLASSES)
def test_input_rejections(cls):
    with pytest.raises(ValueError):
        model(cls, gas_constant_J_mol_K=True)
    base = model(cls)
    for t, rho, fractions in (
        (0.0, 1.0, (1.0,)),
        (300.0, -1.0, (1.0,)),
        (300.0, 1.0, (0.5, 0.5)),
        (float("nan"), 1.0, (1.0,)),
        (300.0, 1.0, (True,)),
    ):
        with pytest.raises(ValueError):
            base.alphar(t, rho, fractions)
    with pytest.raises(ValueError):
        cls.from_dict({**base.to_dict(), "kind": "wrong"})


@pytest.mark.parametrize("author", ("Mecke", "Lisal"))
@pytest.mark.parametrize("pole", ("dipole", "quadrupole"))
def test_zero_moment_reduction_and_fractional_vacuum_derivative(author, pole):
    zero = model(TwoCenterLJModel, author=author, polar=pole)
    none = replace(zero, polar="none")
    assert zero.alphar(300.0, 1000.0, (1.0,)) == none.alphar(300.0, 1000.0, (1.0,))
    active = replace(zero, reduced_moment_squared=1.0)
    for order in range(1, 4):
        assert (
            molecular_scaled_derivative(active, 300.0, 0.0, (1.0,), inverse_temperature_order=order)
            == 0.0
        )
    assert derivative_values(lambda rho: active.alphar(300.0, rho, (1.0,)), 0.0, order=1)[0] == 0.0
    with pytest.raises(ValueError, match="derivative"):
        molecular_scaled_derivative(
            active, 300.0, 0.0, (1.0,), density_order=2 if pole == "dipole" else 3
        )


def test_two_center_zero_length_convention_and_packing():
    zero = model(TwoCenterLJModel)
    near = replace(zero, elongation=1e-12)
    assert near._reduction[3] == pytest.approx(4 * zero._reduction[3], rel=1e-10)
    with pytest.raises(ValueError, match="packing"):
        zero.alphar(300.0, 1e7, (1.0,))
    for kwargs in (
        {"author": "unknown"},
        {"elongation": -1},
        {"elongation": 1.1},
        {"reduced_moment_squared": 1},
        {"polar": "both"},
    ):
        with pytest.raises(ValueError):
            model(TwoCenterLJModel, **kwargs)


def test_mie_parameter_boundaries():
    for lr in (10.9, 13.1, float("inf"), True):
        with pytest.raises(ValueError):
            model(PohlMieModel, repulsive_exponent=lr)
    for lr, la in ((6.0, 6.0), (2.0, 1.0), (12.0, 3.0), (12.0, float("nan"))):
        with pytest.raises(ValueError):
            model(ChaparroMieModel, repulsive_exponent=lr, attractive_exponent=la)


def test_published_chaparro_value_and_tensor_symmetries():
    base = model(ChaparroMieModel)
    rho = 0.135 / (AVOGADRO_PER_MOL * base.diameter_m**3)
    assert base.alphar(1.4 * 120, rho, (1.0,)) == pytest.approx(-0.509239537652789 / 1.4, rel=1e-12)
    tensors = _network_derivatives(base.attraction_integral, 0.135, 1 / 1.4, 3)
    assert np.allclose(tensors[2], tensors[2].T, rtol=1e-13, atol=1e-13)
    assert np.allclose(tensors[3], tensors[3].transpose(1, 0, 2), rtol=1e-13, atol=1e-13)
    assert np.allclose(tensors[3], tensors[3].transpose(2, 1, 0), rtol=1e-13, atol=1e-13)


def test_neural_derivatives_against_independent_complex_step():
    base = model(ChaparroMieModel)
    alpha, rho, beta = base.attraction_integral, 0.25, 0.6
    tensors = _network_derivatives(alpha, rho, beta, 3)
    # Differentiate analytic gradient/Hessian with a tiny complex step. No
    # production finite differences and no upstream derivative implementation.
    from chemworld.physchem.molecular_potential_extensions import _BIASES, _OUTPUT, _WEIGHTS

    def complex_forward(r, b):
        x = np.array((alpha, r, b), dtype=complex)
        g = np.array(((0, 0), (1, 0), (0, 1)), dtype=complex)
        h = np.zeros((3, 2, 2), dtype=complex)
        for w, c in zip(_WEIGHTS, _BIASES, strict=True):
            x = np.tanh(x @ w + c)
            g0, h0 = w.T @ g, np.einsum("nm,nij->mij", w, h)
            g = (1 - x * x)[:, None] * g0
            h = (1 - x * x)[:, None, None] * h0 - (2 * x * (1 - x * x))[:, None, None] * np.einsum(
                "ni,nj->nij", g0, g0
            )
        return x @ _OUTPUT, np.einsum("n,ni->i", _OUTPUT, g), np.einsum("n,nij->ij", _OUTPUT, h)

    for axis in (0, 1):
        args = [rho, beta]
        args[axis] += 1e-25j
        value, g, h = complex_forward(*args)
        assert value.imag / 1e-25 == pytest.approx(tensors[1][axis], rel=1e-12)
        assert np.allclose(g.imag / 1e-25, tensors[2][:, axis], rtol=1e-12, atol=1e-12)
        assert np.allclose(h.imag / 1e-25, tensors[3][:, :, axis], rtol=1e-12, atol=1e-12)
