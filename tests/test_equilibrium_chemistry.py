from __future__ import annotations

from math import log

import pytest

from chemworld.physchem import (
    EquilibriumReactionSpec,
    EquilibriumSystemSpec,
    GibbsMinimizationSpec,
    GibbsSpeciesSpec,
    SolubilityProductSpec,
    apply_precipitation_hooks,
    aqueous_ph_observation,
    balance_charge_by_adjusting_ion,
    diagnose_gibbs_minimization,
    equilibrium_chemistry_model_cards,
    equilibrium_constant_vant_hoff,
    ionic_strength,
    ionic_strength_from_amounts,
    net_charge_equivalents,
    precipitate_if_supersaturated,
    reaction_extent_bounds,
    reaction_quotient,
    solid_solubility_mole_fraction,
    solve_gibbs_minimization,
    solve_mass_action_equilibrium,
    solve_monoprotic_acid_base,
    solve_reaction_extent,
    validate_model_card,
    water_ion_product,
)


def test_single_reaction_extent_respects_nonnegativity_and_equilibrium_ratio() -> None:
    reaction = EquilibriumReactionSpec.from_equation(
        reaction_id="isomerization",
        equation="A <=> B",
        log10_k_ref=0.6020599913279624,
    )
    result = solve_reaction_extent(
        reaction,
        {"A": 1.0, "B": 0.0},
        volume_L=1.0,
        temperature_K=298.15,
    )

    assert result.converged
    assert result.final_amounts_mol["A"] >= 0.0
    assert result.final_amounts_mol["B"] >= 0.0
    assert result.final_amounts_mol["B"] / result.final_amounts_mol["A"] == pytest.approx(
        4.0,
        rel=1e-7,
    )
    assert result.extents_mol["isomerization"] == pytest.approx(0.8)
    assert result.reaction_quotients["isomerization"] == pytest.approx(
        result.equilibrium_constants["isomerization"],
    )


def test_reaction_extent_bounds_allow_reverse_extent_without_negative_products() -> None:
    reaction = EquilibriumReactionSpec.from_equation(
        reaction_id="reversible",
        equation="A <=> B",
        log10_k_ref=0.0,
    )
    lower, upper = reaction_extent_bounds(reaction, {"A": 0.2, "B": 0.8})

    assert lower == pytest.approx(-0.8)
    assert upper == pytest.approx(0.2)


def test_mass_action_multi_reaction_solver_returns_finite_residuals() -> None:
    first = EquilibriumReactionSpec.from_equation(
        reaction_id="ab",
        equation="A <=> B",
        log10_k_ref=0.0,
    )
    second = EquilibriumReactionSpec.from_equation(
        reaction_id="bc",
        equation="B <=> C",
        log10_k_ref=0.0,
    )
    system = EquilibriumSystemSpec(
        system_id="three_species_chain",
        reactions=(first, second),
        temperature_K=298.15,
        volume_L=1.0,
    )
    result = solve_mass_action_equilibrium(system, {"A": 1.0})

    assert result.converged
    assert result.final_amounts_mol["A"] == pytest.approx(1 / 3, rel=1e-5)
    assert result.final_amounts_mol["B"] == pytest.approx(1 / 3, rel=1e-5)
    assert result.final_amounts_mol["C"] == pytest.approx(1 / 3, rel=1e-5)
    assert max(abs(value) for value in result.residuals_log.values()) < 1e-6


def test_gibbs_minimization_matches_analytical_ideal_isomerization() -> None:
    temperature = 298.15
    gas_constant = 8.31446261815324
    spec = GibbsMinimizationSpec(
        system_id="ideal_isomerization",
        species=(
            GibbsSpeciesSpec(
                "A",
                "liquid",
                {"X": 1.0},
                standard_gibbs_J_mol=0.0,
            ),
            GibbsSpeciesSpec(
                "B",
                "liquid",
                {"X": 1.0},
                standard_gibbs_J_mol=-gas_constant * temperature * log(4.0),
            ),
        ),
        temperature_K=temperature,
    )

    result = solve_gibbs_minimization(spec, {"A": 1.0, "B": 0.0})

    assert result.converged
    assert result.final_amounts_mol["B"] / result.final_amounts_mol["A"] == pytest.approx(
        4.0,
        rel=1e-7,
    )
    assert result.final_amounts_mol["A"] + result.final_amounts_mol["B"] == pytest.approx(1.0)
    assert abs(result.element_balance_residuals_mol["X"]) < 1e-8
    assert abs(result.charge_balance_residual_eq) < 1e-8
    assert result.phase_amounts_mol["liquid"] == pytest.approx(1.0)
    assert result.to_dict()["metadata"]["solver"] == "scipy_slsqp_gibbs_minimization"
    assert result.diagnostic is not None
    assert result.diagnostic.status == "ok"
    assert result.diagnostic.max_element_residual_mol < 1e-8
    assert result.diagnostic.stationarity_residual_J_mol < 1e-4
    assert result.to_dict()["metadata"]["diagnostic"]["status"] == "ok"

    recomputed = diagnose_gibbs_minimization(spec, result)
    assert recomputed.to_dict() == result.diagnostic.to_dict()


def test_gibbs_minimization_handles_stoichiometric_compound_formation() -> None:
    spec = GibbsMinimizationSpec(
        system_id="water_formation",
        species=(
            GibbsSpeciesSpec("H2", "gas", {"H": 2.0}, standard_gibbs_J_mol=0.0),
            GibbsSpeciesSpec("O2", "gas", {"O": 2.0}, standard_gibbs_J_mol=0.0),
            GibbsSpeciesSpec(
                "H2O",
                "gas",
                {"H": 2.0, "O": 1.0},
                standard_gibbs_J_mol=-120_000.0,
            ),
        ),
        temperature_K=298.15,
    )

    result = solve_gibbs_minimization(spec, {"H2": 1.0, "O2": 0.5, "H2O": 0.0})

    assert result.converged
    assert result.final_amounts_mol["H2O"] > 0.999
    assert result.final_amounts_mol["H2"] < 1e-3
    assert result.final_amounts_mol["O2"] < 1e-3
    assert result.diagnostic is not None
    assert result.diagnostic.status == "ok"
    assert result.diagnostic.constraint_matrix_rank == 2
    assert result.diagnostic.degrees_of_freedom == 1
    assert result.diagnostic.max_element_residual_mol < 1e-8


def test_gibbs_minimization_enforces_phase_restrictions_and_charge() -> None:
    species = (
        GibbsSpeciesSpec("Na+", "aqueous", {"Na": 1.0}, charge=1.0),
        GibbsSpeciesSpec("Cl-", "aqueous", {"Cl": 1.0}, charge=-1.0),
        GibbsSpeciesSpec(
            "NaCl(s)",
            "solid",
            {"Na": 1.0, "Cl": 1.0},
            standard_gibbs_J_mol=-20_000.0,
        ),
    )
    aqueous_only = GibbsMinimizationSpec(
        system_id="restricted_salt",
        species=species,
        allowed_phases=("aqueous",),
    )
    with_solid = GibbsMinimizationSpec(
        system_id="solid_allowed_salt",
        species=species,
        allowed_phases=("aqueous", "solid"),
    )

    restricted = solve_gibbs_minimization(
        aqueous_only,
        {"Na+": 1.0, "Cl-": 1.0, "NaCl(s)": 0.0},
    )
    precipitated = solve_gibbs_minimization(
        with_solid,
        {"Na+": 1.0, "Cl-": 1.0, "NaCl(s)": 0.0},
    )

    assert restricted.converged
    assert restricted.final_amounts_mol["NaCl(s)"] == pytest.approx(0.0)
    assert restricted.charge_balance_residual_eq == pytest.approx(0.0)
    assert restricted.active_phases == ("aqueous",)
    assert precipitated.converged
    assert precipitated.final_amounts_mol["NaCl(s)"] > 0.99
    assert abs(precipitated.element_balance_residuals_mol["Na"]) < 1e-8
    assert abs(precipitated.element_balance_residuals_mol["Cl"]) < 1e-8
    assert abs(precipitated.charge_balance_residual_eq) < 1e-8
    assert precipitated.diagnostic is not None
    assert precipitated.diagnostic.status in {"ok", "warning"}
    assert precipitated.diagnostic.convexity_class == (
        "convex_with_linear_pure_condensed_phase_terms"
    )
    assert "NaCl(s)" in precipitated.diagnostic.active_species_ids


def test_vant_hoff_temperature_dependence_has_correct_direction() -> None:
    endothermic_low = equilibrium_constant_vant_hoff(
        log10_k_ref=0.0,
        temperature_K=280.0,
        delta_h_J_mol=30_000.0,
    )
    endothermic_high = equilibrium_constant_vant_hoff(
        log10_k_ref=0.0,
        temperature_K=340.0,
        delta_h_J_mol=30_000.0,
    )
    exothermic_low = equilibrium_constant_vant_hoff(
        log10_k_ref=0.0,
        temperature_K=280.0,
        delta_h_J_mol=-30_000.0,
    )
    exothermic_high = equilibrium_constant_vant_hoff(
        log10_k_ref=0.0,
        temperature_K=340.0,
        delta_h_J_mol=-30_000.0,
    )

    assert endothermic_high > endothermic_low
    assert exothermic_high < exothermic_low


def test_reaction_quotient_uses_concentration_activities() -> None:
    reaction = EquilibriumReactionSpec.from_equation(
        reaction_id="dimer",
        equation="2 A <=> B",
        log10_k_ref=0.0,
    )
    quotient = reaction_quotient(
        {"A": 0.5, "B": 0.25},
        reaction.stoichiometry,
        volume_L=0.5,
    )

    assert quotient == pytest.approx(0.5)


def test_monoprotic_acid_base_solves_ph_and_charge_balance() -> None:
    result = solve_monoprotic_acid_base(
        acid_total_mol=0.1,
        volume_L=1.0,
        pka=4.76,
    )

    assert result.pH == pytest.approx(2.88, abs=0.05)
    assert result.acid_dissociation_fraction < 0.02
    assert abs(result.charge_balance_error_eq) < 1e-10
    assert result.ionic_strength_mol_kg > 0.0
    assert result.to_dict()["pH"] == pytest.approx(result.pH)

    observation = aqueous_ph_observation(result, noise_std_pH=0.0, seed=7)
    payload = observation.to_dict()
    assert payload["raw_signal"]["signal_type"] == "potentiometric_ph"
    assert payload["processed_estimate"]["pH"] == pytest.approx(result.pH, abs=0.01)
    assert payload["observed_mask"]["species_amounts_mol"] is False
    assert "species_amounts_mol" not in payload["processed_estimate"]


def test_monoprotic_activity_coefficient_ratio_changes_executed_equilibrium() -> None:
    ideal = solve_monoprotic_acid_base(
        acid_total_mol=0.1,
        volume_L=1.0,
        pka=4.76,
    )
    nonideal = solve_monoprotic_acid_base(
        acid_total_mol=0.1,
        volume_L=1.0,
        pka=4.76,
        activity_coefficient_ratio=4.0,
    )

    assert nonideal.pH > ideal.pH
    assert nonideal.acid_dissociation_fraction < ideal.acid_dissociation_fraction
    assert nonideal.metadata["activity_coefficient_ratio"] == 4.0
    assert abs(nonideal.charge_balance_error_eq) < 1e-10


def test_water_ion_product_changes_with_temperature() -> None:
    cold = water_ion_product(273.15)
    room = water_ion_product(298.15)
    hot = water_ion_product(373.15)

    assert cold < room < hot


def test_precipitation_removes_ions_only_after_saturation() -> None:
    spec = SolubilityProductSpec(
        precipitate_id="AgCl(s)",
        cation_id="Ag+",
        anion_id="Cl-",
        ksp=1e-10,
    )
    undersaturated = precipitate_if_supersaturated(
        {"Ag+": 1e-6, "Cl-": 1e-6},
        spec,
        volume_L=1.0,
    )
    supersaturated = precipitate_if_supersaturated(
        {"Ag+": 1e-3, "Cl-": 1e-3},
        spec,
        volume_L=1.0,
    )

    assert undersaturated.precipitated_mol == 0.0
    assert supersaturated.precipitated_mol > 9e-4
    assert supersaturated.final_amounts_mol["Ag+"] == pytest.approx(1e-5, rel=1e-4)
    assert supersaturated.final_amounts_mol["Cl-"] == pytest.approx(1e-5, rel=1e-4)
    assert supersaturated.ion_product == pytest.approx(spec.ksp, rel=1e-6)
    assert supersaturated.material_balance_error_mol < 1e-12


def test_precipitation_bracket_reaches_exact_stoichiometric_limit() -> None:
    """An inward-shifted upper bracket stays supersaturated in this regime."""

    spec = SolubilityProductSpec(
        precipitate_id="XY(s)",
        cation_id="X+",
        anion_id="Y-",
        ksp=1e-30,
    )
    result = precipitate_if_supersaturated(
        {"X+": 1.0, "Y-": 1e-6},
        spec,
        volume_L=1.0,
    )

    assert result.precipitated_mol == pytest.approx(1e-6, abs=1e-14)
    assert result.final_amounts_mol["Y-"] == pytest.approx(0.0, abs=1e-14)
    assert result.ion_product <= spec.ksp
    assert result.material_balance_error_mol < 1e-12
    assert result.metadata["solver_status"] in {"brentq", "stoichiometric_limit"}


def test_precipitation_hooks_apply_multiple_solubility_specs() -> None:
    hooks = apply_precipitation_hooks(
        {"Ag+": 1e-3, "Cl-": 1e-3, "Ba2+": 2e-3, "SO4--": 2e-3},
        (
            SolubilityProductSpec("AgCl(s)", "Ag+", "Cl-", ksp=1e-10),
            SolubilityProductSpec("BaSO4(s)", "Ba2+", "SO4--", ksp=1e-10),
        ),
        volume_L=1.0,
    )

    assert hooks.metadata["status"] == "converged"
    assert hooks.total_precipitated_mol > 2e-3
    assert hooks.material_balance_error_mol < 1e-12
    assert {event["precipitate_id"] for event in hooks.precipitation_events} == {
        "AgCl(s)",
        "BaSO4(s)",
    }
    assert hooks.final_amounts_mol["AgCl(s)"] > 9e-4
    assert hooks.final_amounts_mol["BaSO4(s)"] > 1.9e-3
    assert hooks.to_dict()["precipitation_events"]
    assert hooks.converged
    assert hooks.warnings == ()


def test_precipitation_hooks_report_max_pass_failure_explicitly() -> None:
    hooks = apply_precipitation_hooks(
        {"Ag+": 1e-3, "Cl-": 1e-3},
        (
            SolubilityProductSpec("AgCl(s)", "Ag+", "Cl-", ksp=1e-6),
            SolubilityProductSpec("AgCl(s)", "Ag+", "Cl-", ksp=1e-10),
        ),
        volume_L=1.0,
        max_passes=1,
    )

    assert hooks.metadata["status"] == "max_passes_reached"
    assert hooks.converged is False
    assert hooks.warnings == ("max_passes_reached",)
    assert hooks.to_dict()["converged"] is False


def test_charge_balance_adjusts_selected_ion_to_electroneutrality() -> None:
    amounts = {"Na+": 0.08, "Cl-": 0.10}
    charges = {"Na+": 1, "Cl-": -1}
    result = balance_charge_by_adjusting_ion(amounts, charges, adjustable_species_id="Na+")

    assert net_charge_equivalents(amounts, charges) == pytest.approx(-0.02)
    assert result.adjustment_mol == pytest.approx(0.02)
    assert result.adjusted_amounts_mol["Na+"] == pytest.approx(0.10)
    assert abs(result.final_charge_eq) < 1e-12


def test_ionic_strength_from_molality_and_amounts() -> None:
    molal = ionic_strength({"Na+": 0.1, "Cl-": 0.1}, {"Na+": 1, "Cl-": -1})
    from_amounts = ionic_strength_from_amounts(
        {"Na+": 0.1, "Cl-": 0.1},
        {"Na+": 1, "Cl-": -1},
        solvent_mass_kg=1.0,
    )

    assert molal == pytest.approx(0.1)
    assert from_amounts == pytest.approx(molal)


def test_solid_solubility_mole_fraction_increases_with_temperature() -> None:
    cold = solid_solubility_mole_fraction(
        temperature_K=280.0,
        melting_temperature_K=350.0,
        enthalpy_fusion_J_mol=18_000.0,
    )
    warm = solid_solubility_mole_fraction(
        temperature_K=320.0,
        melting_temperature_K=350.0,
        enthalpy_fusion_J_mol=18_000.0,
    )

    assert 0.0 < cold < warm < 1.0


def test_equilibrium_chemistry_model_cards_include_gibbs_minimization() -> None:
    assert all(validate_model_card(item) == [] for item in equilibrium_chemistry_model_cards())
    card = next(
        item
        for item in equilibrium_chemistry_model_cards()
        if item.model_id == "fixed_tp_ideal_gibbs_minimization"
    )

    assert card.maturity.value == "reference_validated"
    assert any("element constraints" in equation for equation in card.equations)
    assert any(
        item.model_id == "aqueous_acid_base_ph_observation"
        for item in equilibrium_chemistry_model_cards()
    )


def test_equilibrium_chemistry_validation_fails_fast() -> None:
    with pytest.raises(ValueError, match="reactant"):
        EquilibriumReactionSpec("bad", {"B": 1.0}, log10_k_ref=0.0)
    with pytest.raises(ValueError, match="volume_L"):
        solve_reaction_extent(
            EquilibriumReactionSpec.from_equation(
                reaction_id="ab",
                equation="A <=> B",
                log10_k_ref=0.0,
            ),
            {"A": 1.0},
            volume_L=0.0,
            temperature_K=298.15,
        )
    with pytest.raises(ValueError, match="negative"):
        ionic_strength({"Na+": -0.1}, {"Na+": 1})
    with pytest.raises(ValueError, match="charged"):
        balance_charge_by_adjusting_ion(
            {"Na+": 0.1},
            {"Na+": 0},
            adjustable_species_id="Na+",
        )
    with pytest.raises(ValueError, match="phase restrictions exclude"):
        solve_gibbs_minimization(
            GibbsMinimizationSpec(
                system_id="bad_phase",
                species=(GibbsSpeciesSpec("S", "solid", {"X": 1.0}),),
                allowed_phases=("liquid",),
            ),
            {"S": 1.0},
        )
    with pytest.raises(ValueError, match="Allowed phases cannot carry"):
        solve_gibbs_minimization(
            GibbsMinimizationSpec(
                system_id="missing_element_carrier",
                species=(GibbsSpeciesSpec("S", "solid", {"X": 1.0}),),
                allowed_phases=("liquid",),
            ),
            {"S": 0.0},
            target_element_amounts_mol={"X": 1.0},
        )
