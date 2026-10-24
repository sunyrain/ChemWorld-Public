"""Planar weighted densities, stable vacuum, bulk EOS and analytic responses."""

import json
from math import pi

import mpmath as mp
import numpy as np
import pytest

from chemworld.physchem.caloric_math import derivative_values, partial_derivative
from chemworld.physchem.fundamental_measure import (
    FundamentalMeasureBulkModel,
    FundamentalMeasureFunctional,
    fmt_contact_value,
)
from chemworld.physchem.molecular_consumers import (
    molecular_model_from_dict,
    molecular_model_to_dict,
)
from chemworld.physchem.molecular_eos import molecular_density_from_pressure, molecular_eos_state
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, hard_sphere_chain


@pytest.mark.parametrize("kind", ("RF", "WB", "WBII"))
def test_common_bulk_state_codec_inverse_and_local_density(kind):
    model = FundamentalMeasureBulkModel(("A", "B"), (3e-10, 4e-10), "explicit", kind)
    assert FundamentalMeasureBulkModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    assert molecular_model_from_dict(molecular_model_to_dict(model)) == model
    local = FundamentalMeasureFunctional(kind)
    assert FundamentalMeasureFunctional.from_dict(local.to_dict()) == local
    rho, x = 1000.0, (0.3, 0.7)
    values = model.weighted_densities(rho, x)
    assert local.energy_density(values) / (rho * AVOGADRO_PER_MOL) == pytest.approx(
        model.alphar(300, rho, x)
    )
    state = molecular_eos_state(model, 300, rho, x)
    assert state.residual_internal_energy_J_mol == 0
    solved = molecular_density_from_pressure(
        model,
        300,
        state.pressure_Pa,
        x,
        initial_density_mol_m3=900,
        density_bounds_mol_m3=(500, 1500),
    )
    assert solved.molar_density_mol_m3 == pytest.approx(rho, rel=1e-10)
    assert model.alphar(300, 0, x) == 0
    assert all(np.isfinite(derivative_values(lambda r: model.alphar(300, r, x), 0.0, order=3)))
    if kind == "WB":
        expected, _ = hard_sphere_chain(rho * AVOGADRO_PER_MOL, x, (1.0, 1.0), (3e-10, 4e-10))
        assert model.alphar(300, rho, x) == pytest.approx(expected)


@pytest.mark.parametrize("kind", ("RF", "WB", "WBII"))
def test_vacuum_gradient_hessian_and_pure_bulk_equation(kind):
    local = FundamentalMeasureFunctional(kind)
    values, grad, hess = local.evaluate_many([[0.0] * 6], gradient=True, hessian=True)
    assert values[0] == 0
    assert np.all(grad == 0)
    assert hess[0, 0, 3] == hess[0, 3, 0] == 1
    assert hess[0, 1, 2] == hess[0, 2, 1] == 1
    assert hess[0, 4, 5] == hess[0, 5, 4] == -1
    model = FundamentalMeasureBulkModel(("A",), (1.0,), "explicit", kind)
    for eta in (1e-10, 0.01, 0.1, 0.5):
        rho = eta / (pi / 6 * AVOGADRO_PER_MOL)
        state = molecular_eos_state(model, 300, rho, (1.0,))
        expected = (
            (1 + eta + eta**2) / (1 - eta) ** 3
            if kind == "RF"
            else (1 + eta + eta**2 - eta**3) / (1 - eta) ** 3
        )
        assert state.compressibility_factor == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("kind", ("WB", "WBII"))
def test_dilute_coefficients_against_independent_70_digit_formula(kind):
    with mp.workdps(70):
        n0, n1, n2, v1, v2 = map(mp.mpf, (0.7, 0.4, 0.9, 0.03, -0.07))

        def independent(eta):
            q = 1 - eta
            if kind == "WB":
                pair = 1 / q
                cubic = (eta + q * q * mp.log(q)) / (36 * mp.pi * eta * eta * q * q)
            else:
                p2 = (2 - eta + 2 * q * mp.log(q) / eta) / 3
                p3 = (2 * eta - 3 * eta**2 + 2 * eta**3 + 2 * q * q * mp.log(q)) / (3 * eta**2)
                pair = (1 + p2) / q
                cubic = (1 - p3) / (24 * mp.pi * q * q)
            return -n0 * mp.log(q) + (n1 * n2 - v1 * v2) * pair + (n2**3 - 3 * n2 * v2 * v2) * cubic

        for eta in (1e-12, 0.03, 0.049, 0.051):
            native = derivative_values(
                lambda e: FundamentalMeasureFunctional(kind).energy_density(
                    (0.7, 0.4, 0.9, e, 0.03, -0.07)
                ),
                eta,
                order=3,
            )
            expected = [float(mp.diff(independent, mp.mpf(eta), n)) for n in range(4)]
            assert native == pytest.approx(expected, rel=2e-9, abs=2e-10)


def test_contact_gradient_and_explicit_limits():
    row = (0.7, 0.4, 0.9, 0.2, 0.03, -0.07)
    value = fmt_contact_value(row, 0.2)
    x = {f"n{i}": v for i, v in enumerate(row)}
    for i in (2, 3, 5):
        actual = partial_derivative(
            lambda v: fmt_contact_value(tuple(v.values()), 0.2), x, (f"n{i}",)
        )
        lo, hi = list(row), list(row)
        lo[i] -= 1e-6
        hi[i] += 1e-6
        assert actual == pytest.approx(
            (fmt_contact_value(hi, 0.2) - fmt_contact_value(lo, 0.2)) / 2e-6, rel=1e-7
        )
    assert value > 1
    assert fmt_contact_value((0.0,) * 6, 0.2) == 1
    with pytest.raises(ValueError, match="derivative"):
        derivative_values(lambda n: fmt_contact_value((0.0, 0.0, n, 0.0, 0.0, 0.0), 0.2), 0.0)
    for row in ((1, 1, 1, 1, 0, 0), (1, 1, 1, 0.1, 2, 0), (1, 1, -1, 0.1, 0, 0)):
        with pytest.raises(ValueError):
            FundamentalMeasureFunctional().energy_density(row)
