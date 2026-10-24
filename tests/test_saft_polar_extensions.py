"""Integral catalog, independent polarization calculus and actual EOS consumers."""

from dataclasses import replace
from itertools import product
from math import pi, sqrt

import mpmath as mp
import numpy as np
import pytest

from chemworld.physchem.caloric_math import FirstDerivative, partial_derivative
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.saft_polar_extensions import (
    _MOMENT_SCALE,
    _TABLES,
    CorrelationMultipole,
    CorrelationPolar,
    _energy_terms,
    _gradient,
    _integrals,
    _pade,
    _polarized_moments,
    correlation_integral,
    correlation_polar_helmholtz_contributions,
)
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel


def mp_integral(family, order, t, r):
    """Independent high-precision literal correlation equations and data only."""
    pair = isinstance(order, tuple)
    key = order
    if family == "Gottschalk" and pair and isinstance(order[0], int):
        key = tuple(tuple(map(int, str(v))) for v in order)
    a = tuple(mp.mpf(str(v)) for v in _TABLES[family][int(pair)][key])
    if family == "GubbinsTwu":
        basis = [r * r * mp.log(t), r * r, r * mp.log(t), r, mp.log(t), 1]
        return (-1 if order == (334, 445) else 1) * mp.exp(mp.fdot(a, basis))
    if family == "Luckas":
        width = 4 if pair else 3
        b = [mp.fsum(a[width * i + j] * r**i for i in range(4)) for j in range(width)]
        if pair:
            z = mp.exp((1 - r / mp.sqrt(2)) ** 4)
            return b[0] + b[1] * t + b[2] * z**2 + b[3] * z**3
        return (
            b[0]
            + b[1] * t ** (mp.mpf(".3") + mp.mpf(".05") * order)
            + b[2] * t ** (mp.mpf(1) / order)
        ) * mp.exp(1 / (t + 4 / abs(mp.log(r / mp.sqrt(2))) ** 3))
    if not pair:
        return (
            mp.fsum(a[4 * i + j] * r**i * t**j for i in range(5) for j in range(4))
            + mp.exp(1 / t)
            * mp.fsum(a[20 + 3 * i + j] * r**i * t**j for i in range(5) for j in range(3))
        ) ** (order - 2)
    return mp.fsum(
        a[2 * i + j - 1] * r**i * mp.exp(j * (1 - r / 3) / t)
        + a[8 + 2 * i + j - 1] * r**i * mp.exp(j * (1 - r / 3) ** 2 / t)
        for i in range(4)
        for j in (1, 2)
    ) + mp.fsum(a[16 + 4 * i + j] * r**i * t**j for i in range(6) for j in range(4))


@pytest.mark.parametrize(
    "family,order", [(f, o) for f, tables in _TABLES.items() for table in tables for o in table]
)
def test_complete_integral_catalog(family, order):
    with mp.workdps(50):
        expected = float(mp_integral(family, order, mp.mpf("1.7"), mp.mpf(".4")))
    assert correlation_integral(family, order, 1.7, 0.4) == pytest.approx(
        expected, rel=1e-9, abs=1e-10
    )


def spec(family="GubbinsTwu", polarizable=False):
    return CorrelationPolar(
        (
            CorrelationMultipole(
                "a",
                3.33564e-30,
                0,
                "synthetic",
                1e-30 if polarizable else 0,
                0.6e-30 if polarizable else 0,
            ),
        ),
        "synthetic",
        "GrayGubbins",
        family,
        "Gubbins",
    )


def direct(p, t, rho, x=(1,), s=(3.4,), e=(180,)):
    return correlation_polar_helmholtz_contributions(
        p,
        t,
        rho,
        pi / 6 * rho * 40,
        tuple(m.component_id for m in p.moments),
        x,
        s,
        e,
        tuple(tuple(0 for _ in x) for _ in x),
    )["correlation_polar"]


def mp_pure_dipole(t, rho, family):
    """One-component stationary energy, with an independent scalar root/gradient."""
    s, e = mp.mpf("3.4"), mp.mpf(180)
    u0 = mp.mpf(str(3.33564e-30 * _MOMENT_SCALE * 1e15))
    parallel, perpendicular = mp.mpf(1), mp.mpf(".6")
    iso = (parallel + 2 * perpendicular) / 3
    j6 = 4 * mp.pi / s**3 * mp_integral(family, 6, t / e, rho * s**3)
    k = (
        64
        * mp.pi**3
        / 5
        * mp.sqrt(14 * mp.pi / 5)
        / s**3
        * mp_integral(family, (222, 333), t / e, rho * s**3)
    )

    def perturb(u):
        z = u * u / (3 * t) + iso
        a = -mp.mpf("1.5") * (z * z - iso * iso) * j6
        b = rho * (z**3 - iso**3) * k / 2
        return rho * a * a / (a - b)

    effective = mp.findroot(
        lambda u: u - u0 + t * parallel * mp.diff(perturb, u), (u0, u0 * mp.mpf("1.01"))
    )
    return perturb(effective) + (effective - u0) ** 2 / (2 * t * parallel)


@pytest.mark.parametrize("family", ("GubbinsTwu", "Luckas"))
def test_polarizable_pure_all_third_derivatives_against_independent_mp_root(family):
    p = spec(family, True)
    with mp.workdps(45):
        for i, j in [(a, b) for a in range(4) for b in range(4 - a)]:
            coordinates = ("T",) * i + ("rho",) * j
            value = (
                partial_derivative(
                    lambda v: direct(p, v["T"], v["rho"]), {"T": 350.0, "rho": 0.001}, coordinates
                )
                if coordinates
                else direct(p, 350, 0.001)
            )
            expected = mp.diff(
                lambda t, r: mp_pure_dipole(t, r, family), (mp.mpf(350), mp.mpf(".001")), (i, j)
            )
            assert value == pytest.approx(float(expected), rel=2e-8, abs=1e-9)


def mixture():
    return CorrelationPolar(
        (
            CorrelationMultipole("a", 3.33564e-30, 0.4e-40, "synthetic", 1e-30, 0.6e-30),
            CorrelationMultipole("b", 2e-30, 2e-40, "synthetic", 1.3e-30, 0.9e-30),
        ),
        "synthetic",
        "GrayGubbins",
        "GubbinsTwu",
        "Gubbins",
        0.8,
        1.2,
    )


def test_mixture_moment_gradient_stationarity_and_species_permutation():
    p = mixture()
    t, rho, x, s, e = 350.0, 0.001, (0.3, 0.7), (3.4, 3.8), ((180, 200), (200, 230))
    r = rho * sum(x[i] * x[j] * ((s[i] + s[j]) / 2) ** 3 for i, j in product(range(2), repeat=2))
    pair, triple = _integrals(p, t, r, s, e)
    permanent = [m.dipole_C_m * _MOMENT_SCALE * 1e15 for m in p.moments]
    u = _polarized_moments(p, t, rho, x, permanent, pair, triple)
    for i in range(2):
        step = 0.001
        plus, minus = list(u), list(u)
        plus[i] += step
        minus[i] -= step
        fd = (
            _pade(_energy_terms(p, t, rho, plus, pair, triple), rho, x)
            - _pade(_energy_terms(p, t, rho, minus, pair, triple), rho, x)
        ) / (2 * step * x[i])
        analytic = _gradient(p, t, rho, x, u, pair, triple, i)
        assert analytic == pytest.approx(fd, rel=2e-8)
        assert u[i] - permanent[i] + t * p.moments[
            i
        ].alpha_parallel_m3 * 1e30 * fd == pytest.approx(0, abs=1e-9)
    reverse = replace(p, moments=p.moments[::-1])
    assert direct(p, t, rho, x, s, (180, 230)) == pytest.approx(
        direct(reverse, t, rho, x[::-1], s[::-1], (230, 180)), abs=2e-15
    )


@pytest.mark.parametrize("family", ("Luckas", "GubbinsTwu", "Gottschalk"))
def test_zero_population_limits_and_composition_jets(family):
    p = replace(spec(family), moments=(spec().moments[0], CorrelationMultipole("b", 0, 0, "inert")))
    value = partial_derivative(
        lambda v: direct(p, 350.0, 0.001, (v["x"], 1 - v["x"]), (3.4, 3.8), (180, 230)),
        {"x": 0.0},
        ("x", "x"),
    )
    h = 1e-5
    expected = 2 * direct(p, 350, 0.001, (h, 1 - h), (3.4, 3.8), (180, 230)) / h**2
    assert value == pytest.approx(expected, rel=2e-5)
    assert direct(p, 350, 0.001, (0, 1), (3.4, 3.8), (180, 230)) == 0


def test_mixture_polarizable_composition_face_and_derivative():
    p = mixture()

    def value(x):
        return direct(p, 350, 0.001, (x, 1 - x), (3.4, 3.8), (180, 230))

    for x in (0.0, 0.3, 1.0):
        actual = partial_derivative(lambda v: value(v["x"]), {"x": x}, ("x",))
        h = 1e-5
        if x == 0:
            expected = (-3 * value(0) + 4 * value(h) - value(2 * h)) / (2 * h)
        elif x == 1:
            expected = (3 * value(1) - 4 * value(1 - h) + value(1 - 2 * h)) / (2 * h)
        else:
            expected = (value(x + h) - value(x - h)) / (2 * h)
        assert actual == pytest.approx(expected, rel=2e-7, abs=1e-10)


def test_declarations_vacuum_and_luckas_nonsmooth_boundaries():
    p = spec()
    assert CorrelationPolar.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="Gray-Gubbins"):
        replace(spec(polarizable=True), formulation="GubbinsTwu")
    with pytest.raises(ValueError, match="finite SI"):
        CorrelationMultipole("a", 1, 1, "s", -1)
    with pytest.raises(ValueError, match="undeclared"):
        correlation_integral("GubbinsTwu", 25, 1.5, 0.2)
    for family in ("GubbinsTwu", "Gottschalk"):
        assert direct(spec(family), 350, 0) == 0
        assert np.isfinite(
            partial_derivative(
                lambda v, family=family: direct(spec(family), 350, v["rho"]),
                {"rho": 0.0},
                ("rho",) * 3,
            )
        )
    assert direct(spec("Luckas"), 350, 0) == 0
    with pytest.raises(ValueError, match="vacuum"):
        correlation_integral("Luckas", 6, 1.5, FirstDerivative(0, 1))
    assert np.isfinite(correlation_integral("Luckas", 6, 1.5, sqrt(2)))
    with pytest.raises(ValueError, match="only C2"):
        partial_derivative(
            lambda v: correlation_integral("Luckas", 6, 1.5, v["rho"]),
            {"rho": sqrt(2)},
            ("rho",) * 3,
        )


def test_actual_mie_state_roundtrip_density_inverse_and_chemical_potentials():
    p = mixture()
    cs = (
        MieSegment("a", 1, 3.4e-10, 180, "synthetic"),
        MieSegment("b", 1.7, 3.8e-10, 230, "synthetic", 15),
    )
    model = SAFTVRMieModel(cs, ((0, 0.02), (0.02, 0)), (100, 1000), "synthetic", polar=p)
    assert SAFTVRMieModel.from_dict(model.to_dict()) == model
    t, rho, x = 350, 1000, (0.3, 0.7)
    state = molecular_eos_state(model, t, rho, x)
    inverse = molecular_density_from_pressure(
        model,
        t,
        state.pressure_Pa,
        x,
        initial_density_mol_m3=900,
        density_bounds_mol_m3=(500, 1500),
    )
    assert inverse.molar_density_mol_m3 == pytest.approx(rho, rel=1e-10)
    mus = molecular_residual_chemical_potentials(model, t, rho, x)
    assert np.dot(mus, x) == pytest.approx(
        state.residual_helmholtz_J_mol
        + (state.compressibility_factor - 1) * model.gas_constant_J_mol_K * t,
        rel=1e-11,
    )
    for i, j in ((3, 0), (2, 1), (1, 2), (0, 3)):
        assert np.isfinite(
            molecular_scaled_derivative(
                model, t, rho, x, inverse_temperature_order=i, density_order=j
            )
        )
