"""Finite crystal mass, size, provenance and the common thermal advance."""

from dataclasses import replace

import pytest

from chemworld.physchem.coupled_system import HeatFlux
from chemworld.physchem.crystal_thermal_kinetics import (
    CrystalPopulationState,
    CrystalThermalSpec,
    ThermalCrystalCohort,
    advance_crystal_population,
    crystal_population_energy,
)
from chemworld.physchem.crystallization_units import (
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
)
from chemworld.physchem.property_packages import PropertyPackageThermodynamics, SpeciesCaloricModel
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation


def crystal_fixture(*, saturation=0.0, diameter=20e-6, nucleation=0.0):
    model = CrystalThermalSpec(
        "P",
        "I",
        SolubilityCurveSpec("synthetic-Cs", 1.0, 298.15, 20000, 250, 430, "synthetic"),
        CrystallizationKineticsSpec(
            "synthetic-growth",
            nucleation,
            2,
            2e-8,
            1,
            1200,
            0.1,
            8e-6,
            impurity_occlusion_mol_per_mol=0.02,
            supersaturation_occlusion_factor=0.5,
            provenance_id="synthetic",
        ),
        {"P": "solid:P", "I": "solid:I"},
    )
    properties = {}
    for name, cv, energy, phase in (
        ("P", 100, 0, "liquid"),
        ("I", 120, 0, "liquid"),
        ("solid:P", 100, -20000, "solid"),
        ("solid:I", 120, 0, "solid"),
    ):
        properties[name] = SpeciesCaloricModel(
            PropertyCorrelation(
                name + ":Cv",
                "cp_polynomial",
                {"a": cv},
                {"temperature": "K"},
                "J/(mol*K)",
                property_id="isochoric_heat_capacity",
                validity_ranges={"temperature": (250, 430)},
                source_note="Synthetic declared Cv",
            ),
            CaloricReference(phase=phase, energy_basis="internal_energy", energy_J_mol=energy),
        )
    thermal = PropertyPackageThermodynamics(properties, 60, 298.15)
    state = CrystalPopulationState(
        {"P": saturation * 0.02, "I": 0.002},
        (
            ThermalCrystalCohort(
                0.001 / model.particle_moles(diameter), diameter, 0.0001, 0.0003, 0.001
            ),
        ),
        298.15,
        0.02,
        0.0002 if saturation else 0.0,
    )
    return model, thermal, state


def quantities(model, state):
    total = dict(state.liquid_amounts_mol)
    seed = state.dissolved_seed_mol
    for cohort in state.cohorts:
        total["P"] += cohort.target_mol
        total["I"] += cohort.impurity_mol
        seed += cohort.seed_mol
    return total, seed


def test_crystal_caloric_reuse_keeps_exact_temperature_model_and_validity_policy(monkeypatch):
    from chemworld.physchem.crystal_caloric_layout import CrystalCaloricLayout

    _, thermal, _ = crystal_fixture()
    original = SpeciesCaloricModel.evaluate
    calls = []

    def evaluate(self, temperature, quantities, *, strict=True):
        calls.append((id(self), temperature, tuple(quantities), strict))
        return original(self, temperature, quantities, strict=strict)

    monkeypatch.setattr(SpeciesCaloricModel, "evaluate", evaluate)
    cached = CrystalCaloricLayout(
        {"a": thermal.species["P"], "b": thermal.species["P"], "solid": thermal.species["solid:P"]},
        60,
        298.15,
    )
    first = cached._at("a", 298.15, ("cv",), strict=False)
    assert cached._at("b", 298.15, ("cv",), strict=False) is first
    assert len(calls) == 1
    cached._at("a", 298.15, ("cv",), strict=True)
    cached._at("solid", 298.15, ("cv",), strict=True)
    cached._at("a", 298.15 + 1e-10, ("cv",), strict=True)
    assert len(calls) == 4
    cached._at("a", 450, ("cv",), strict=False)
    with pytest.raises(RuntimeError, match="outside"):
        cached._at("a", 450, ("cv",), strict=True)


def test_finite_small_crystal_disappears_and_releases_actual_occlusion_and_seed():
    model, thermal, state = crystal_fixture()
    result = advance_crystal_population(model, state, thermal, 9000)
    final = result.final_state
    assert not final.cohorts
    assert final.liquid_amounts_mol == pytest.approx({"P": 0.001, "I": 0.0021}, abs=1e-12)
    assert final.dissolved_seed_mol == pytest.approx(0.0003, abs=1e-12)
    assert final.temperature_K < state.temperature_K
    assert abs(result.energy_balance_residual_J) < 1e-6


def test_actual_temperature_and_finite_power_cannot_preallocate_hot_dissolution():
    model, thermal, state = crystal_fixture(saturation=1.0, diameter=100e-6)
    result = advance_crystal_population(model, state, thermal, 1, heat_flux=lambda _: HeatFlux(5))
    final = result.final_state
    assert 298.15 < final.temperature_K < 298.24
    assert final.cohorts[0].target_mol > 0.000999
    assert result.boundary_heat_J == pytest.approx(5, abs=1e-8)
    assert quantities(model, final)[0] == pytest.approx(quantities(model, state)[0], abs=1e-12)
    assert quantities(model, final)[1] == pytest.approx(quantities(model, state)[1], abs=1e-12)
    assert crystal_population_energy(model, final, thermal) == pytest.approx(
        crystal_population_energy(model, state, thermal) + 5, abs=1e-6
    )


def test_isothermal_saturated_crystal_retains_mass_and_size():
    model, thermal, state = crystal_fixture(saturation=1.0)
    result = advance_crystal_population(model, state, thermal, 1000, isothermal=True)
    assert result.final_state.cohorts[0].diameter_m == pytest.approx(20e-6, abs=1e-15)
    assert result.final_state.liquid_amounts_mol == state.liquid_amounts_mol
    assert result.boundary_heat_J == 0


def test_crystal_caloric_energy_must_match_the_solubility_reference():
    model, thermal, state = crystal_fixture()
    wrong = replace(
        thermal.species["solid:P"],
        reference=replace(thermal.species["solid:P"].reference, energy_J_mol=-19000),
    )
    thermal = replace(thermal, species={**thermal.species, "solid:P": wrong})
    with pytest.raises(ValueError, match="caloric offsets"):
        advance_crystal_population(model, state, thermal, 1)


@pytest.mark.parametrize("saturation,newborn", [(0., False), (1., False), (2., False), (2., True)])
@pytest.mark.parametrize("legacy", [False, True])
def test_compact_crystal_transfers_keep_full_rates_and_custom_caloric_records(
    saturation, newborn, legacy,
):
    from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
    from chemworld.physchem.crystal_thermal_kinetics import _layout, _process
    from chemworld.runtime.legacy_crystal_thermal import LegacyCrystalEnthalpy

    model, thermal, state = crystal_fixture(saturation=saturation, nucleation=2e7 if newborn else 0)
    if legacy:
        thermal = LegacyCrystalEnthalpy({'P': 0., 'I': 0., 'solid:P': -20000., 'solid:I': 0.}, 60.)
    thermal, amounts, liquid, solid, cohorts = _layout(model, state, thermal, newborn=newborn)

    class CustomCalorics(type(thermal)):
        pass

    custom = CustomCalorics(thermal.species, thermal.support_heat_capacity_J_K,
                           thermal.reference_temperature_K)
    candidate = CoupledStateView(amounts, state.temperature_K, state.volume_L)
    reference_process = _process(model, custom, liquid, solid, cohorts)
    assert len(reference_process.evaluate(candidate).internal_transfers) == 2 * len(cohorts)
    current_process = _process(model, thermal, liquid, solid, cohorts)
    current = assemble_coupled_rates(
        candidate, processes=(current_process,), thermodynamics=thermal,
        heat_flux=lambda _: HeatFlux(2., 1.))
    reference = assemble_coupled_rates(
        candidate, processes=(reference_process,), thermodynamics=custom,
        heat_flux=lambda _: HeatFlux(2., 1.))
    assert current == reference


def test_nucleation_transfers_real_mass_without_fabricated_particle_inventory():
    model, thermal, state = crystal_fixture(saturation=2, nucleation=2e7)
    state = replace(state, cohorts=())
    result = advance_crystal_population(model, state, thermal, 60, isothermal=True)
    final = result.final_state
    assert len(final.cohorts) == 2
    assert final.cohorts[0].diameter_m > final.cohorts[1].diameter_m
    assert quantities(model, final)[0] == pytest.approx(quantities(model, state)[0], abs=1e-10)
    assert quantities(model, final)[1] == pytest.approx(quantities(model, state)[1], abs=1e-10)
    assert abs(result.energy_balance_residual_J) < 1e-6


def test_tiny_cohort_uses_its_own_mass_scale_for_finite_radial_recession():
    model, thermal, state = crystal_fixture()
    n = 1e-30
    state = replace(
        state,
        cohorts=(
            ThermalCrystalCohort(n / model.particle_moles(20e-6), 20e-6, n * 0.1, n * 0.3, n),
        ),
    )
    result = advance_crystal_population(model, state, thermal, 300, isothermal=True)
    # Its finite-bath feedback is negligible: d=d0-2*k*t, independently.
    assert result.final_state.cohorts[0].diameter_m == pytest.approx(8e-6, abs=1e-12)
    assert quantities(model, result.final_state)[0]["P"] == pytest.approx(n, rel=1e-7, abs=0)


def test_host_disappearance_releases_only_resolved_tiny_residue_and_accounts_heat():
    from chemworld.physchem.coupled_system import CoupledState
    from chemworld.physchem.crystal_thermal_kinetics import _layout, _release_vanished_hosts

    model, thermal, state = crystal_fixture()
    properties = dict(thermal.species)
    properties["solid:I"] = replace(
        properties["solid:I"],
        reference=replace(properties["solid:I"].reference, energy_J_mol=-1000),
    )
    thermal = replace(thermal, species=properties)
    packed_thermal, amounts, liquid, solid, _ = _layout(model, state, thermal)
    amounts[solid[0]["seed"]] = amounts[solid[0]["grown"]] = 0.0
    amounts[solid[0]["impurity"]] = 1e-12
    packed = CoupledState(amounts, 298.15, 0.02)
    final, transfers = _release_vanished_hosts(
        model, packed, packed_thermal, liquid, solid, isothermal=True, impurity_stock_mol=0.0021
    )
    assert transfers[0]["occluded_mol"] == 1e-12
    assert final.amounts_mol[solid[0]["impurity"]] == 0
    assert final.amounts_mol[liquid["I"][0]] == pytest.approx(0.002 + 1e-12, abs=1e-18)
    assert final.boundary_heat_J == pytest.approx(1e-9, abs=1e-18)
    amounts[solid[0]["impurity"]] = 0.0001
    with pytest.raises(RuntimeError, match="unresolved impurity"):
        _release_vanished_hosts(
            model,
            replace(packed, amounts_mol=amounts),
            packed_thermal,
            liquid,
            solid,
            isothermal=True,
            impurity_stock_mol=0.0021,
        )


def test_finite_impurity_exhaustion_keeps_clean_growth_advancing_without_negative_stock():
    model, thermal, state = crystal_fixture(saturation=2, diameter=100e-6)
    state = replace(state, liquid_amounts_mol={"P": 0.04, "I": 1e-8})
    result = advance_crystal_population(model, state, thermal, 9000, isothermal=True)
    assert result.actual_duration_s == 9000
    assert result.final_state.cohorts[0].target_mol > 0.015
    assert 0 <= result.final_state.liquid_amounts_mol["I"] < 1e-12
    assert quantities(model, result.final_state)[0]["I"] == pytest.approx(0.00010001, abs=1e-12)


@pytest.mark.parametrize("saturation", [0.0, 2.0])
@pytest.mark.parametrize("isothermal", [False, True])
@pytest.mark.parametrize("cohort_count", [1, 5])
def test_cohort_jacobian_matches_independent_full_difference(
    monkeypatch, saturation, isothermal, cohort_count
):
    import numpy as np

    from chemworld.physchem import solver_backend

    class InspectedError(Exception):
        pass

    def inspect(rhs, initial, **kwargs):
        assert kwargs["jac_sparsity"] is None
        actual = kwargs["jac"](0, initial).toarray()
        numerical = np.zeros_like(actual)
        for column, value in enumerate(initial):
            step = 1e-6 * (abs(value) if value != 0 else 1.0)
            plus, minus = initial.copy(), initial.copy()
            plus[column] += step
            minus[column] -= step
            numerical[:, column] = (
                (rhs(0, plus) - rhs(0, initial)) / step
                if value == 0
                else (rhs(0, plus) - rhs(0, minus)) / (2 * step)
            )
        np.testing.assert_allclose(actual, numerical, rtol=1e-3, atol=1e-7)
        raise InspectedError

    monkeypatch.setattr(solver_backend, "solve_ode", inspect)
    model, thermal, state = crystal_fixture(saturation=saturation)
    cohort = state.cohorts[0]
    state = replace(
        state,
        cohorts=tuple(
            replace(
                cohort,
                particle_count=cohort.particle_count / cohort_count / (1 + i / 10) ** 3,
                diameter_m=cohort.diameter_m * (1 + i / 10),
                target_mol=cohort.target_mol / cohort_count,
                seed_mol=cohort.seed_mol / cohort_count,
                impurity_mol=cohort.impurity_mol / cohort_count,
            )
            for i in range(cohort_count)
        ),
    )
    with pytest.raises(InspectedError):
        advance_crystal_population(model, state, thermal, 100, isothermal=isothermal)
