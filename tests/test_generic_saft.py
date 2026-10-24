"""GenericSAFT composition reaches the existing thermodynamic consumers."""

import json
from dataclasses import replace
from itertools import product
from math import log, sqrt

import pytest

from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.generic_saft import GenericSAFTModel, MaskedSAFTAssociation
from chemworld.physchem.generic_saft_import import generic_saft_from_teqp
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL


def specification(kind="PCSAFT", association=None, count=2):
    rows = [
        {
            "name": f"C{i}",
            "m": 1.0 + 0.4 * i,
            "sigma_Angstrom": 3.7 + 0.1 * i,
            "epsilon_over_k": 180 + 20 * i,
            "lambda_r": 12 + i,
            "lambda_a": 6,
            "BibTeXKey": "Explicit test",
        }
        for i in range(count)
    ]
    data = {
        "coeffs": rows,
        "kmat": [[0.0 if i == j else 0.02 for j in range(count)] for i in range(count)],
    }
    if kind in ("softSAFT", "Johnson+Johnson"):
        data = {
            "m": [r["m"] for r in rows],
            "sigma / m": [r["sigma_Angstrom"] * 1e-10 for r in rows],
            "epsilon/kB / K": [r["epsilon_over_k"] for r in rows],
        }
    elif kind in ("2CLJ", "2CLJF"):
        data = {"author": "2CLJF_Lisal", "L^*": 0.3}
    result = {"kind": "genericSAFT", "model": {"nonpolar": {"kind": kind, "model": data}}}
    if association:
        assoc = {"molecule_sites": [["e", "H"] for _ in rows]}
        if association == "canonical":
            assoc.update(
                {
                    "Delta_rule": "CR1",
                    "b / m^3/mol": [3e-5] * count,
                    "beta": [0.01] * count,
                    "epsilon / J/mol": [1000 * R_J_PER_MOL_K] * count,
                }
            )
        else:
            assoc.update(
                {
                    "Delta_rule": "Dufal",
                    "sigma / m": [3.0555e-10] * count,
                    "epsilon/kB / K": [418] * count,
                    "lambda_r": [35.823] * count,
                    "kmat": [[0.0] * count for _ in rows],
                    "epsilon_HB / J/mol": [1600 * R_J_PER_MOL_K] * count,
                    "K_HB / m^3": [496.66e-30] * count,
                }
            )
        result["model"]["association"] = {"kind": association, "model": assoc}
    return result


def imported(kind="PCSAFT", association=None, count=2, **kwargs):
    return generic_saft_from_teqp(
        specification(kind, association, count),
        component_ids=tuple(f"C{i}" for i in range(count)),
        temperature_range_K=(100, 1000),
        source="Fixed GenericSAFT test",
        canonical_radial_distribution="Carnahan-Starling",
        two_center_scales=(3.7e-10, 180.0),
        **kwargs,
    ).model


@pytest.mark.parametrize(
    "kind,association",
    tuple(product(("PCSAFT", "SAFTVRMie", "softSAFT", "2CLJ"), (None, "canonical", "Dufal"))),
)
def test_all_public_contribution_combinations_roundtrip_and_consume(kind, association):
    model = imported(kind, association, 1)
    rebuilt = GenericSAFTModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert rebuilt.to_dict() == model.to_dict()
    state = molecular_eos_state(rebuilt, 600, 1000, (1.0,))
    recovered = molecular_density_from_pressure(
        rebuilt,
        600,
        state.pressure_Pa,
        (1.0,),
        initial_density_mol_m3=950,
        density_bounds_mol_m3=(1, 2000),
    )
    assert recovered.molar_density_mol_m3 == pytest.approx(1000, rel=1e-10)
    mu = molecular_residual_chemical_potentials(rebuilt, 600, 1000, (1.0,))[0]
    assert mu == pytest.approx(
        state.residual_helmholtz_J_mol
        + (state.compressibility_factor - 1) * 600 * model.gas_constant_J_mol_K
    )


@pytest.mark.parametrize("x", ((1.0, 0.0), (0.3, 0.7), (0.0, 1.0)))
def test_mixture_endpoint_derivatives_sum_existing_contributions(x):
    model = imported("PCSAFT", "Dufal")
    for i in range(4):
        for j in range(4 - i):
            options = {"inverse_temperature_order": i, "density_order": j}
            actual = molecular_scaled_derivative(model, 500, 1000, x, **options)
            expected = molecular_scaled_derivative(model.nonpolar, 500, 1000, x, **options)
            expected += molecular_scaled_derivative(model.association, 500, 1000, x, **options)
            assert actual == pytest.approx(expected, abs=1e-12, rel=1e-11)
    assert len(molecular_residual_chemical_potentials(model, 500, 1000, x)) == 2


def test_directed_site_mask_uses_shared_solver_with_exact_one_way_solution():
    model = imported("PCSAFT", "canonical", 1)
    assert isinstance(model.association, MaskedSAFTAssociation)
    directed = replace(model.association, interaction_mask=((False, True), (False, False)))
    kernel = directed.kernel
    contact = kernel.contacts(500, 1000, (1.0,))[0][0]
    spec = kernel.sites
    from math import expm1

    delta = spec.bonding_volume_m3[0][1] * expm1(spec.energy_K[0][1] / 500) * contact
    x = 1 / (1 + 1000 * AVOGADRO_PER_MOL * delta)
    assert directed.alphar(500, 1000, (1.0,)) == pytest.approx(log(x) - x / 2 + 0.5)
    assert replace(model, association=directed).alphar(500, 1000, (1.0,)) == pytest.approx(
        model.nonpolar.alphar(500, 1000, (1.0,)) + log(x) - x / 2 + 0.5
    )


def test_canonical_two_site_solution_and_radial_choice():
    model = imported("PCSAFT", "canonical", 1)
    card = model.association.kernel
    from math import expm1

    for radial in ("Carnahan-Starling", "Kontogeorgis"):
        kernel = replace(card, radial_distribution=radial)
        contact = kernel.contacts(500, 1000, (1.0,))[0][0]
        delta = kernel.sites.bonding_volume_m3[0][1] * expm1(1000 / 500) * contact
        x = 2 / (1 + sqrt(1 + 4 * 1000 * AVOGADRO_PER_MOL * delta))
        masked = replace(model.association, kernel=kernel)
        assert masked.alphar(500, 1000, (1.0,)) == pytest.approx(2 * (log(x) - x / 2 + 0.5))


def test_source_ignored_polar_and_mie_flags_are_reported():
    spec = specification("SAFTVRMie", count=1)
    spec["model"]["polar"] = {"arbitrary": "ignored upstream"}
    spec["model"]["nonpolar"]["model"]["SAFTVRMie_flags"] = {"epsilon_ij": "Lorentz-Berthelot"}
    result = generic_saft_from_teqp(
        spec,
        component_ids=("A",),
        temperature_range_K=(100, 1000),
        source="Explicit source behavior",
    )
    assert len(result.notes) == 2
    assert result.model.nonpolar.energy_mixing == "Lafitte"
    assert result.model.nonpolar.polar is None


def test_pcsaft_internal_polar_and_named_library_preserved():
    spec = specification("PCSAFT", count=1)
    row = spec["model"]["nonpolar"]["model"].pop("coeffs")[0]
    row.update({"(mu^*)^2": 1.0, "nmu": 1.0, "(Q^*)^2": 0.5, "nQ": 1.0})
    spec["model"]["nonpolar"]["model"]["names"] = ["material"]
    kwargs = {"component_ids": ("A",), "temperature_range_K": (100, 1000), "source": "Explicit"}
    with pytest.raises(ValueError, match="parameter_library"):
        generic_saft_from_teqp(spec, **kwargs)
    result = generic_saft_from_teqp(spec, parameter_library={"material": row}, **kwargs)
    assert result.model.nonpolar.polar.contributions == ("DD", "QQ")
    assert result.model.nonpolar.polar.moments[0].component_id == "A"


def test_source_connectivity_self_mask_and_empty_sites():
    spec = specification("PCSAFT", "canonical", 1)
    opts = {"self_association_mask": [False], "interaction_partners": {"e": ["H"], "H": ["e"]}}
    spec["model"]["association"]["model"]["options"] = opts
    kwargs = {
        "component_ids": ("A",),
        "temperature_range_K": (100, 1000),
        "source": "Explicit",
        "canonical_radial_distribution": "Carnahan-Starling",
    }
    result = generic_saft_from_teqp(spec, **kwargs)
    assert result.model.association.alphar(500, 1000, (1.0,)) == 0.0
    spec["model"]["association"]["model"]["molecule_sites"] = [[]]
    assert generic_saft_from_teqp(spec, **kwargs).model.association is None


def test_missing_scales_radial_or_component_alignment_rejected():
    kwargs = {"component_ids": ("A",), "temperature_range_K": (100, 1000), "source": "Explicit"}
    with pytest.raises(ValueError, match="radial"):
        generic_saft_from_teqp(specification("PCSAFT", "canonical", 1), **kwargs)
    with pytest.raises(ValueError, match="scales"):
        generic_saft_from_teqp(specification("2CLJ", count=1), **kwargs)
    one, two = imported(count=1), imported(association="Dufal")
    with pytest.raises(ValueError, match="ordering"):
        replace(one, association=two.association)
