from __future__ import annotations

import pytest

from chemworld.physchem import (
    CrystallizationKineticsSpec,
    MaturityLevel,
    SolubilityCurveSpec,
    cooling_crystallization,
    separation_model_cards,
    validate_model_card,
)


def _solubility_curve() -> SolubilityCurveSpec:
    return SolubilityCurveSpec(
        model_id="vanthoff_target_solubility",
        reference_solubility_mol_L=1.0,
        reference_temperature_K=320.0,
        dissolution_enthalpy_J_mol=20_000.0,
        minimum_temperature_K=275.0,
        maximum_temperature_K=325.0,
        provenance_id="synthetic-vanthoff-solubility-case",
    )


def _kinetics(
    *,
    nucleation_coefficient: float = 2.0e7,
    occlusion: float = 0.02,
) -> CrystallizationKineticsSpec:
    return CrystallizationKineticsSpec(
        model_id="compact_target_population_balance",
        primary_nucleation_coefficient_per_L_s=nucleation_coefficient,
        primary_nucleation_exponent=2.0,
        growth_coefficient_m_s=2.0e-8,
        growth_exponent=1.0,
        crystal_density_kg_m3=1200.0,
        target_molecular_weight_kg_mol=0.100,
        nucleus_diameter_m=8.0e-6,
        impurity_occlusion_mol_per_mol=occlusion,
        supersaturation_occlusion_factor=0.5,
        fines_threshold_m=20.0e-6,
        provenance_id="synthetic-nucleation-growth-case",
    )


def _run(
    *,
    final_temperature_K: float = 280.0,
    kinetics: CrystallizationKineticsSpec | None = None,
    seed_mass_g: float = 0.05,
):
    return cooling_crystallization(
        {"target": 1.0, "impurity": 0.10},
        target_component="target",
        impurity_component="impurity",
        solvent_volume_L=1.0,
        initial_temperature_K=320.0,
        final_temperature_K=final_temperature_K,
        duration_s=1800.0,
        solubility_curve=_solubility_curve(),
        kinetics=kinetics or _kinetics(),
        seed_mass_g=seed_mass_g,
        seed_diameter_m=100.0e-6,
        time_steps=120,
    )


def test_vanthoff_solubility_curve_decreases_on_cooling() -> None:
    curve = _solubility_curve()

    assert curve.solubility_mol_per_l(320.0) == pytest.approx(1.0)
    assert curve.solubility_mol_per_l(280.0) < curve.solubility_mol_per_l(300.0) < 1.0
    with pytest.raises(ValueError, match="outside"):
        curve.solubility_mol_per_l(260.0)


def test_cooling_crystallization_reports_supersaturation_balance_and_csd() -> None:
    result = _run()

    assert result.crystallized_from_solution_mol > 0.0
    assert result.maximum_supersaturation_ratio > 1.0
    assert result.target_recovery > 0.0
    assert result.crystal_purity < 1.0
    assert result.impurity_occluded_mol > 0.0
    assert result.material_balance_error_mol < 1.0e-10
    csd = result.crystal_size_distribution
    assert csd.total_particle_count > 0.0
    assert 0.0 < csd.d10_m <= csd.d50_m <= csd.d90_m
    assert csd.cohort_count > 1
    assert len(result.step_reports) == 120
    assert result.provenance == {
        "solubility_curve": "synthetic-vanthoff-solubility-case",
        "kinetics": "synthetic-nucleation-growth-case",
    }


def test_deeper_cooling_increases_crystal_recovery() -> None:
    mild = _run(final_temperature_K=300.0)
    deep = _run(final_temperature_K=280.0)

    assert deep.target_recovery > mild.target_recovery
    assert deep.maximum_supersaturation_ratio >= mild.maximum_supersaturation_ratio


def test_seeding_enables_growth_when_primary_nucleation_is_disabled() -> None:
    kinetics = _kinetics(nucleation_coefficient=0.0)
    unseeded = _run(kinetics=kinetics, seed_mass_g=0.0)
    seeded = _run(kinetics=kinetics, seed_mass_g=0.05)

    assert unseeded.crystallized_from_solution_mol == pytest.approx(0.0)
    assert "no_crystal_population_formed" in unseeded.warnings
    assert seeded.crystallized_from_solution_mol > 0.0
    assert seeded.crystal_size_distribution.total_particle_count > 0.0


def test_impurity_occlusion_coefficient_controls_crystal_purity() -> None:
    clean = _run(kinetics=_kinetics(occlusion=0.0))
    occluding = _run(kinetics=_kinetics(occlusion=0.10))

    assert clean.crystal_purity == pytest.approx(1.0)
    assert occluding.impurity_occluded_mol > clean.impurity_occluded_mol
    assert occluding.crystal_purity < clean.crystal_purity
    assert occluding.material_balance_error_mol < 1.0e-10


def test_crystallization_rejects_heating_and_invalid_component_contracts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        cooling_crystallization(
            {"target": 1.0},
            target_component="target",
            impurity_component=None,
            solvent_volume_L=1.0,
            initial_temperature_K=300.0,
            final_temperature_K=310.0,
            duration_s=100.0,
            solubility_curve=_solubility_curve(),
            kinetics=_kinetics(),
        )
    with pytest.raises(ValueError, match="impurity_component"):
        cooling_crystallization(
            {"target": 1.0},
            target_component="target",
            impurity_component="missing",
            solvent_volume_L=1.0,
            initial_temperature_K=320.0,
            final_temperature_K=300.0,
            duration_s=100.0,
            solubility_curve=_solubility_curve(),
            kinetics=_kinetics(),
        )


def test_crystallization_model_card_is_professional_candidate_and_auditable() -> None:
    card = {
        item.model_id: item for item in separation_model_cards()
    }["cooling_crystallization_population_balance_v1"]

    assert card.maturity is MaturityLevel.PROFESSIONAL_CANDIDATE
    assert validate_model_card(card) == []
    assert "CSD" in card.validation_evidence[1].description
