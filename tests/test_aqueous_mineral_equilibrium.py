"""Actual solvent mass, simultaneous reactions and finite phase boundaries."""

from dataclasses import replace
from math import log10, sqrt

import pytest

from chemworld.physchem.aqueous_mineral import (
    CHARGES,
    WATER_MOLAR_MASS_KG_MOL,
    CarbonateEquilibriumSpec,
    carbonate_feed,
    element_amounts,
    solve_carbonate_equilibrium,
)
from chemworld.physchem.equilibrium import davies_aqueous_activity_coefficient
from chemworld.physchem.equilibrium_chemistry import net_charge_equivalents


def check_balances(initial, final):
    assert element_amounts(final.amounts_mol) == pytest.approx(
        element_amounts(initial), rel=1e-10, abs=1e-14
    )
    assert abs(net_charge_equivalents(final.amounts_mol, CHARGES)) <= 1e-11
    assert final.solvent_mass_kg == pytest.approx(
        final.amounts_mol["H2O"] * WATER_MOLAR_MASS_KG_MOL, rel=1e-14
    )
    assert all(v >= 0 for v in final.amounts_mol.values())


def test_water_activity_ph_and_explicit_water_ionization():
    initial = carbonate_feed(water_kg=1)
    result = solve_carbonate_equilibrium(CarbonateEquilibriumSpec(), initial)
    check_balances(initial, result)
    assert result.pH == pytest.approx(7, abs=1e-10)
    assert result.amounts_mol["H2O"] < initial["H2O"]
    assert initial["H2O"] - result.amounts_mol["H2O"] == pytest.approx(
        result.amounts_mol["OH-"], abs=1e-14
    )
    assert result.saturation_ratio == 0
    assert result.log10_saturation_ratio is None


@pytest.mark.parametrize("solid", [1e-6, 0.001])
def test_calcite_dissolves_only_its_actual_stock_or_retains_saturated_phase(solid):
    initial = carbonate_feed(water_kg=1, calcite_mol=solid)
    result = solve_carbonate_equilibrium(CarbonateEquilibriumSpec(), initial)
    check_balances(initial, result)
    assert result.amounts_mol["Ca++"] + result.calcite_mol == pytest.approx(solid, abs=1e-14)
    if solid == 1e-6:
        assert result.calcite_mol == 0
        assert result.saturation_ratio < 1
    else:
        assert 0 < result.calcite_mol < solid
        assert result.saturation_ratio == pytest.approx(1, abs=1e-8)


def test_precipitation_is_finite_and_excluding_the_phase_retains_supersaturation():
    initial = carbonate_feed(
        water_kg=1, sodium_bicarbonate_mol=0.01, calcium_chloride_mol=0.01, sodium_chloride_mol=0.01
    )
    model = CarbonateEquilibriumSpec()
    present = solve_carbonate_equilibrium(model, initial)
    excluded = solve_carbonate_equilibrium(replace(model, allow_calcite=False), initial)
    check_balances(initial, present)
    check_balances(initial, excluded)
    assert 0 < present.calcite_mol < 0.01
    assert present.saturation_ratio == pytest.approx(1, abs=1e-8)
    assert excluded.calcite_mol == 0
    assert excluded.saturation_ratio > 1
    assert present.amounts_mol["Na+"] == excluded.amounts_mol["Na+"] == initial["Na+"]
    assert present.amounts_mol["Cl-"] == excluded.amounts_mol["Cl-"] == initial["Cl-"]


@pytest.mark.parametrize("acid,base,expected", [(0.001, 0, 3), (0, 0.001, 11)])
def test_ideal_strong_input_matches_independent_dilute_quadratic(acid, base, expected):
    model = CarbonateEquilibriumSpec(davies_A=0)
    initial = carbonate_feed(water_kg=1, hydrochloric_acid_mol=acid, sodium_hydroxide_mol=base)
    result = solve_carbonate_equilibrium(model, initial)
    check_balances(initial, result)
    # Independent fixed-solvent dilute limit; finite-water correction is far
    # smaller than the declared pH allowance at these actual input amounts.
    delta = acid - base
    hydrogen = (
        (delta + sqrt(delta * delta + 4e-14)) / 2
        if delta >= 0
        else 2e-14 / (sqrt(delta * delta + 4e-14) - delta)
    )
    assert result.pH == pytest.approx(-log10(hydrogen), abs=1e-8)
    assert result.pH == pytest.approx(expected, abs=1e-7)


def test_fixed_finite_mineral_boundary_preserves_elements_without_forcing_saturation():
    initial = carbonate_feed(water_kg=1, calcite_mol=0.001)
    result = solve_carbonate_equilibrium(
        CarbonateEquilibriumSpec(), initial, fixed_calcite_mol=0.0005
    )
    check_balances(initial, result)
    assert result.calcite_mol == 0.0005
    assert result.saturation_ratio > 1
    assert result.mineral_status == "held"


def test_ionic_strength_and_temperature_domains_fail_without_clipping_or_ion_repair():
    model = CarbonateEquilibriumSpec()
    with pytest.raises(ValueError, match="ionic-strength"):
        solve_carbonate_equilibrium(model, carbonate_feed(water_kg=1, sodium_chloride_mol=0.2))
    with pytest.raises(ValueError, match="temperature"):
        solve_carbonate_equilibrium(model, carbonate_feed(water_kg=1), temperature_K=300)
    initial = carbonate_feed(water_kg=1)
    initial["H+"] = 0.001
    with pytest.raises(ValueError, match="counterion repair"):
        solve_carbonate_equilibrium(model, initial)


def test_authored_davies_parameter_reuses_existing_activity_kernel():
    result = davies_aqueous_activity_coefficient(
        ionic_strength_mol_kg=0.05, charge=2, debye_huckel_A=0.509, davies_b=0.3
    )
    assert result.activity_coefficient == pytest.approx(
        10 ** (-0.509 * 4 * (sqrt(0.05) / (1 + sqrt(0.05)) - 0.3 * 0.05)), rel=1e-14
    )
