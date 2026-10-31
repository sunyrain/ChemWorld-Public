"""Equilibrium contracts independent of historical world cohorts and runners."""

import pytest

from chemworld.physchem.equilibrium_entity_panel import (
    EquilibriumEntityProfile,
    solve_equilibrium_entity,
)
from chemworld.physchem.equilibrium_mechanism import solve_coupled_weak_acid_precipitation


@pytest.mark.parametrize(
    "family,beta",
    [
        ("direct_free_ion_precipitation", None),
        ("aqueous_ion_pair_intermediate", 6000.0),
    ],
)
@pytest.mark.parametrize("concentration", [0.0001, 0.02, 0.5])
def test_coupled_network_conserves_acid_and_cation(family, beta, concentration):
    volume = 0.024
    result = solve_coupled_weak_acid_precipitation(
        acid_total_mol=concentration * volume,
        volume_L=volume,
        pka=4.8,
        log10_ksp=-5.2,
        mechanism_family=family,
        association_beta_L_per_mol=beta,
    )
    solid = result.precipitated_mol / volume
    acid = (
        result.undissociated_acid_mol_L
        + result.free_conjugate_base_mol_L
        + result.aqueous_pair_mol_L
        + solid
    )
    cation = result.free_cation_mol_L + result.aqueous_pair_mol_L + solid
    assert acid == pytest.approx(concentration, rel=1e-7, abs=1e-14)
    assert cation == pytest.approx(0.15 * concentration, rel=1e-7, abs=1e-14)
    assert result.equilibrium_residual < 1e-8


def test_entity_properties_change_response_but_preserve_topology_and_scale():
    responses = []
    for profile in (
        EquilibriumEntityProfile("one", -0.5, -4.0, 0.10, 1.0),
        EquilibriumEntityProfile("two", 0.5, -6.0, 0.20, 1.0),
    ):
        small = solve_equilibrium_entity(
            profile=profile,
            acid_total_mol=0.0048,
            volume_L=0.024,
            base_pka=4.8,
        )
        large = solve_equilibrium_entity(
            profile=profile,
            acid_total_mol=0.0096,
            volume_L=0.048,
            base_pka=4.8,
        )
        assert small.mechanism_family == large.mechanism_family == "direct_free_ion_precipitation"
        assert small.aqueous_pair_mol_L == large.aqueous_pair_mol_L == 0.0
        assert small.equilibrium_residual < 1e-8
        assert large.equilibrium_residual < 1e-8
        for field in ("pH", "acid_dissociation_fraction", "precipitation_signal"):
            assert getattr(small, field) == pytest.approx(getattr(large, field), abs=1e-12)
        responses.append(small.precipitation_signal)
    assert responses[0] != pytest.approx(responses[1])


def test_solver_registers_distinct_mass_conserving_networks() -> None:
    direct = solve_coupled_weak_acid_precipitation(
        acid_total_mol=0.012,
        volume_L=0.024,
        pka=4.839404789086824,
        log10_ksp=-5.2,
        mechanism_family="direct_free_ion_precipitation",
    )
    paired = solve_coupled_weak_acid_precipitation(
        acid_total_mol=0.012,
        volume_L=0.024,
        pka=4.839404789086824,
        log10_ksp=-5.2,
        mechanism_family="aqueous_ion_pair_intermediate",
        association_beta_L_per_mol=6000.0,
    )

    assert direct.aqueous_pair_mol_L == 0.0
    assert paired.aqueous_pair_mol_L > 0.0
    assert direct.equilibrium_residual < 1.0e-8
    assert paired.equilibrium_residual < 1.0e-8
    assert paired.precipitation_signal != pytest.approx(direct.precipitation_signal)


def test_solver_is_scale_invariant_at_fixed_concentration() -> None:
    small = solve_coupled_weak_acid_precipitation(
        acid_total_mol=0.0003,
        volume_L=0.012,
        pka=5.019404789086824,
        log10_ksp=-5.2,
        mechanism_family="direct_free_ion_precipitation",
    )
    large = solve_coupled_weak_acid_precipitation(
        acid_total_mol=0.0018,
        volume_L=0.072,
        pka=5.019404789086824,
        log10_ksp=-5.2,
        mechanism_family="direct_free_ion_precipitation",
    )

    assert small.pH == pytest.approx(large.pH, abs=1.0e-10)
    assert small.acid_dissociation_fraction == pytest.approx(
        large.acid_dissociation_fraction, abs=1.0e-10
    )
    assert small.precipitation_signal == pytest.approx(large.precipitation_signal, abs=1.0e-10)
