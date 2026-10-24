"""Classical controls share the molecular EOS contract without quantum inputs."""

import json
from dataclasses import replace
from itertools import product
from math import pi, sqrt

import pytest

from chemworld.physchem.mie_reference_options import MieReferenceOptions
from chemworld.physchem.molecular_eos import (
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel, mie_prefactor


def model(pure=False):
    components = (
        MieSegment("A", 1.4, 3.7e-10, 180, "Synthetic", 12.4),
        MieSegment("B", 2.1, 4.2e-10, 250, "Synthetic", 16.0),
    )
    if pure:
        components = components[:1]
    interaction = ((0.0,),) if pure else ((0.0, 0.03), (0.03, 0.0))
    return SAFTVRMieModel(components, interaction, (100, 1000), "Synthetic options check")


@pytest.mark.parametrize("fractions", ((1.0, 0.0), (0.0, 1.0), (0.3, 0.7)))
@pytest.mark.parametrize("mixing", ("Lafitte", "Lorentz-Berthelot"))
def test_default_options_preserve_classical_derivatives(fractions, mixing):
    native = replace(model(), energy_mixing=mixing)
    kernel = MieReferenceOptions().bind(native)
    for i in range(4):
        for j in range(4 - i):
            opts = {"inverse_temperature_order": i, "density_order": j}
            assert molecular_scaled_derivative(
                kernel, 300, 1000, fractions, **opts
            ) == pytest.approx(
                molecular_scaled_derivative(native, 300, 1000, fractions, **opts),
                rel=2e-12,
                abs=2e-12,
            )
    assert molecular_residual_chemical_potentials(kernel, 300, 1000, fractions) == pytest.approx(
        molecular_residual_chemical_potentials(native, 300, 1000, fractions), rel=2e-12
    )
    assert "mass" not in str(kernel.pair(0, 1))


def test_full_control_roundtrip_and_nonzero_pair_interactions():
    options = MieReferenceOptions(
        hard_sphere_reference="nonadditive-Santos",
        diameter_interaction=((0.0, 0.04), (0.04, 0.0)),
        repulsion_interaction=((0.0, -0.03), (-0.03, 0.0)),
        exact_binary_dhs=False,
        enable_hs_extra=True,
        enable_a2=False,
        enable_temperature_cache=False,
    )
    assert MieReferenceOptions.from_dict(json.loads(json.dumps(options.to_dict()))) == options
    kernel = options.bind(model())
    pair = kernel.pair(0, 1)
    sigma = (3.7e-10 + 4.2e-10) / 2 * 0.96
    assert pair.diameter_m == pytest.approx(sigma)
    assert pair.well_depth_K == pytest.approx(
        0.97 * sqrt(180 * 250) * (3.7e-10 * 4.2e-10) ** 1.5 / sigma**3
    )
    assert pair.repulsive_exponent == pytest.approx(3 + 1.03 * sqrt((12.4 - 3) * (16 - 3)))
    assert kernel.contributions(300, 1000, (0.4, 0.6))["dispersion2"] == 0
    integrated = replace(model(), reference_options=options)
    restored = SAFTVRMieModel.from_dict(json.loads(json.dumps(integrated.to_dict())))
    assert restored == integrated
    assert restored.contributions(300, 1000, (0.4, 0.6)) == pytest.approx(
        kernel.contributions(300, 1000, (0.4, 0.6))
    )


def test_reference_options_preserve_the_shared_polar_extension():
    from chemworld.physchem.saft_polar_extensions import CorrelationMultipole, CorrelationPolar

    polar = CorrelationPolar(
        (
            CorrelationMultipole("A", 1e-30, 0.0, "synthetic"),
            CorrelationMultipole("B", 2e-30, 0.0, "synthetic"),
        ),
        "synthetic",
    )
    baseline = replace(model(), polar=polar)
    options = MieReferenceOptions(enable_a2=False)
    integrated = replace(baseline, reference_options=options)
    restored = SAFTVRMieModel.from_dict(json.loads(json.dumps(integrated.to_dict())))
    parts = restored.contributions(350, 500, (0.3, 0.7))
    assert parts["dispersion2"] == 0
    assert parts["correlation_polar"] == pytest.approx(
        baseline.contributions(350, 500, (0.3, 0.7))["correlation_polar"]
    )


def test_pure_hs_presets_have_same_hard_sphere_limit():
    native = model(pure=True)
    # Santos follows source molecular-density convention, which differs for
    # m != 1; use a monomer to test the physically shared pure HS limit.
    native = replace(native, components=(replace(native.components[0], segment_count=1.0),))
    baseline = native.contributions(300, 1000, (1.0,))["hard_sphere"]
    for hs, exact, extra in product(
        ("Lafitte", "additive", "nonadditive-Santos"), (False, True), (False, True)
    ):
        options = MieReferenceOptions(
            hard_sphere_reference=hs, exact_binary_dhs=exact, enable_hs_extra=extra
        )
        terms = options.bind(native).contributions(300, 1000, (1.0,))
        assert terms["hard_sphere"] == pytest.approx(baseline)
        assert terms["hard_sphere_extra"] == pytest.approx(0.0, abs=1e-14)


def test_all_terms_off_and_hs_extra_remains_independent():
    options = MieReferenceOptions(
        enable_hard_sphere=False,
        enable_a1=False,
        enable_a2=False,
        enable_a3=False,
        enable_chain=False,
        enable_hs_extra=False,
    )
    assert options.bind(model()).alphar(300, 1000, (0.3, 0.7)) == 0.0
    assert replace(options, enable_hs_extra=True).bind(model()).alphar(300, 1000, (0.3, 0.7)) != 0.0


def test_pure_tail_integral_shift_and_molecular_density_convention():
    native = model(pure=True)
    options = MieReferenceOptions(
        enable_hard_sphere=False,
        enable_a1=False,
        enable_a2=False,
        enable_a3=False,
        enable_chain=False,
        enable_truncation_correction=True,
        enable_shift_correction=True,
        reduced_radius_cut=3.0,
    )
    kernel = options.bind(native)
    pair = kernel.pair(0, 0)
    lr, la, rc = pair.repulsive_exponent, pair.attractive_exponent, options.reduced_radius_cut
    expected = (
        -2
        * pi
        * 1000
        * AVOGADRO_PER_MOL
        * pair.diameter_m**3
        * pair.well_depth_K
        / 300
        * mie_prefactor(lr, la)
        * (
            rc ** (3 - lr) / (lr - 3)
            - rc ** (3 - la) / (la - 3)
            + (rc ** (3 - lr) - rc ** (3 - la)) / 3
        )
    )
    assert kernel.alphar(300, 1000, (1.0,)) == pytest.approx(expected)
    assert kernel.potential_K(0, 0, 300, rc) == 0.0
    assert kernel.potential_K(0, 0, 300, rc - 1e-8) == pytest.approx(0.0, abs=2e-7)
    assert kernel.potential_K(0, 0, 300, 1.2, 1) == pair.potential_K(300, 1.2, 1)
    mixture = options.bind(model())
    assert mixture.potential_K(0, 1, 300, 4.0) == 0.0
    with pytest.raises(ValueError, match="pure fluids"):
        mixture.alphar(300, 1000, (0.4, 0.6))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"hard_sphere_reference": "invalid"},
        {"enable_chain": 1},
        {"exact_binary_dhs": 1},
        {"reduced_radius_cut": 0.0},
        {"diameter_interaction": ((0.0, 0.1), (0.2, 0.0))},
        {"repulsion_interaction": ((0.1,),)},
    ),
)
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        MieReferenceOptions(**kwargs)


def test_matrix_dimensions_and_invalid_mixed_exponent_rejected():
    with pytest.raises(ValueError, match="ordering"):
        MieReferenceOptions(diameter_interaction=((0.0,),)).bind(model())
    with pytest.raises(ValueError, match="repulsive"):
        MieReferenceOptions(repulsion_interaction=((0.0, 0.99), (0.99, 0.0))).bind(model())
